"""CPU-only L6-B scheduler and native binding contracts."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from semcache.edgelora import r5_d5_l6b as binding, r5_d5_l6b_runner as runner
from semcache.edgelora.r5_d5_lightperf import approval_modules
from semcache.edgelora.r5_d5_lightperf_io import SummaryWriter
from semcache.edgelora.r5_d5_lightperf_workload import CONDITIONS
from test_r5_d5_sweep import workload,policy
from semcache.edgelora.r5_d5_lightperf_replay import replay_stream
from semcache.edgelora.r5_d5_lightperf_workload import knobs,preflight
from semcache.edgelora.r5_d5_impact import MASK_INDEX_REUSE,resource_bounds
from semcache.edgelora.communication import digest


class Binding(unittest.TestCase):
    def test_sealed_parent_pair_changes_native_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            w,tasks=workload('2wikimqa',50)
            pool=[t for t in tasks if t['adapter_id']=='Adapter_A'][:4]+[t for t in tasks if t['adapter_id']=='Adapter_C'][:6]
            for i,t in enumerate(pool):
                t.update(request_sequence_number=i,policy=policy(w['special_tokens'],'quality',8))
                t['prepared']['max_new_tokens']=8
            block=dict(w['blocks'][0],tasks=pool)
            block['block_id']=digest({k:v for k,v in block.items() if k!='block_id'})
            w['blocks']=[block]
            previous,selected=[replay_stream(w,'2wikimqa',n,enable_continuous_replay=True) for n in (20,50)]
            src=dict(commit='FIXTURE',approval_scope='CLEAN_REVIEWED_SOURCE')
            perf=knobs(50,.25)
            # Synthetic planning input only; exercise the production 50-request forecast and binding.
            base=preflight(selected['tasks'],perf,ud_operation_seconds=1.)
            ud=(7824.54483968-(2600+base['link_pacing_upper_seconds']))/38400
            r=resource_bounds(preflight(selected['tasks'],perf,ud_operation_seconds=ud),MASK_INDEX_REUSE)
            frozen=dict(source=dict(commit=binding.l6.BASE),stream=previous,count=20,plan_id='fixture',
                        impact_implementation=MASK_INDEX_REUSE,ud_operation_seconds=ud,
                        performance_knobs=perf,transport=dict(rows=list(range(20))))
            from semcache.edgelora import r5_d5_lightperf as light
            from semcache.edgelora import r5_d5_lightperf_runtime as runtime
            from semcache.edgelora.r5_d5_lightperf_io import SCHEMA, MODE, ENGINE
            src['content_fingerprint']='FIXTURE'
            frozen.update(schema=SCHEMA,mode=MODE,engine=ENGINE, original_experiment='fixture',multi='fixture',
                          dataset='2wikimqa',conditions=list(CONDITIONS),validation_mode='REQUEST_SCOPED_VERIFIED',
                          thresholds=light.THRESHOLDS,warmup_requests=0,approx_requires_explicit_run_opt_in=True,
                          correctness_qualified_speedup='NOT_ESTABLISHED',continuous_contract=None,strict_pilot=None,
                          workload_mode='CONTINUOUS_REPLAY',enable_continuous_replay=True)
            ref={'scope':'FIXTURE'}
            a=dict(schema='D5_L6A_CPU_V2',capacity_bytes_per_adapter=512*2**20,phase='LONG_REPLAY_CPU',source=src,count=50,stream=selected,reference=ref,
                   transport=dict(rows=list(range(50))),entry_limit=64,cycles=binding.l6.CYCLES,
                   impact_implementation=frozen['impact_implementation'],resources=binding.l6.long_deadline(r),
                   allocations=[dict(condition=c,worker_deadline_seconds=9000,requests=50,
                                reset='EMPTY_ONCE_PER_CONDITION') for c in CONDITIONS])
            a['plan_id']=digest(a)
            plan=root/'plan';plan.mkdir();w=SummaryWriter(plan);w.write('plan.json',a);pi=w.seal()
            audit=root/'audit';audit.mkdir();w=SummaryWriter(audit)
            cpu=dict(plan_inventory=pi,plan_id=a['plan_id'],source=src,tests=1,failed=0,errors=0,skipped=0,
                     modules=binding.l6.modules(a),versions=binding.native_versions(),status='CPU_APPROVED')
            w.write('cpu_validation.json',cpu);w.seal()
            with patch.object(binding.l6,'NS',root),patch.object(binding,'historical_reference',return_value=(frozen,None,ref)):
                p=binding.native_plan(plan,audit)
                binding.validate_binding(p)
                self.assertIn('test_r5_d5_l6b',approval_modules(p))
                native=root/'native';native.mkdir();writer=SummaryWriter(native)
                writer.write('plan.json',p);ni=writer.seal()
                fields=dict(stream=selected,transport=a['transport'],resources=preflight(selected['tasks'],perf,ud_operation_seconds=ud))
                with patch.object(light,'source_identity',return_value=src),patch.object(light,'material',return_value=fields):
                    self.assertEqual(light.load_plan(native),(p,ni))
                with patch.object(light,'source_identity',return_value=dict(src,commit='DIFFERENT')),self.assertRaises(ValueError):
                    light.load_plan(native)
                with patch.object(runtime,'source_identity',return_value=src):
                    with self.assertRaises((KeyError,ValueError)):
                        runtime.bound_receipts(native,audit)
                    proof=root/'native_cpu';proof.mkdir();writer=SummaryWriter(proof)
                    writer.write('cpu_validation.json',dict(cpu,engine=ENGINE,status='PASS',scope='CLEAN_NATIVE_CPU_APPROVAL',
                                 plan_inventory=ni,modules=approval_modules(p),native_parity='TINY_OPT_STRICT_LIGHT_CPU_PASS'))
                    ai=writer.seal()
                    self.assertEqual(runtime.bound_receipts(native,proof),(p,ni,ai))
                for key,value in [('entry_limit',128),('l6_plan_id','other'),('resources',r),('source',{}),('stream',previous)]:
                    bad=dict(p,**{key:value})
                    with self.subTest(key=key),self.assertRaises(ValueError): binding.validate_binding(bad)
                wrong=root/'wrong';wrong.mkdir();w=SummaryWriter(wrong)
                w.write('cpu_validation.json',dict(cpu,plan_id='other'));w.seal()
                with self.assertRaisesRegex(ValueError,'approval'):binding.native_plan(plan,wrong)


class Scheduler(unittest.TestCase):
    def test_held_registered_chain_and_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner,'ROOT',Path('/data/khuss/repos/GP_semcache_lightperf')), \
                patch.object(runner,'ready',return_value=dict(plan_id='fixture')),patch.object(runner,'assets',return_value={}):
            root=Path(tmp);calls=[];runner.write_once(root/'ready.json',{})
            def mock(argv):
                calls.append(argv)
                return str(len(calls))
            with patch.object(runner,'command',side_effect=mock):
                ids=runner.submit(root,'moana-y3')
                with self.assertRaises(FileExistsError):runner.submit(root,'moana-y3')
            self.assertEqual(ids,['2','4','6'])
            for i,c in enumerate(CONDITIONS):
                test,actual=calls[2*i:2*i+2]
                self.assertEqual(test,[actual[0],'--test-only',*actual[1:]])
                self.assertTrue({'--hold','--no-requeue','--nodelist=moana-y3'}<=set(actual))
                if i:self.assertIn('--dependency=afterok:'+ids[i-1],actual)
            self.assertEqual(calls[6:],[['scontrol','release',j] for j in ids])

    def test_time_and_identity_guards(self):
        p=dict(resources=dict(slurm_seconds=16200,worker_deadline_seconds=9000))
        for remaining in (9120,9121,16201):
            line='JobId=123 JobState=RUNNING NodeList=moana-y3 TimeLimit=04:30:00 Requeue=0 Partition=batch_eebme_ugrad EndTime='+datetime.fromtimestamp(remaining+100).isoformat()
            with patch.dict(runner.os.environ,SLURM_JOB_ID='123',SLURMD_NODENAME='moana-y3'),patch.object(runner,'command',return_value=line),patch.object(runner.time,'time',return_value=100):
                if remaining==9121:self.assertEqual(runner.budget(p),9121)
                else:
                    with self.assertRaises(ValueError):runner.budget(p)
        a=dict(node='moana-y3',gpu=dict(uuid='00000000-0000-0000-0000-000000000001'),affinity=[0,1,2,3])
        for b in (dict(a,affinity=[4,5,6,7]),dict(a,gpu=dict(uuid='00000000-0000-0000-0000-000000000002'))):
            with self.assertRaisesRegex(ValueError,'NOT_COMPARABLE'):runner.matched(a,b)

    def test_array_and_restart_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            t=dict(node='moana-y3',python=runner.sys.executable)
            runner.write_once(root/'trial.json',t)
            c=CONDITIONS[0];runner.write_once(root/(c+'.job.json'),dict(condition=c,trial=digest(t),job='123'))
            e=dict(SLURM_JOB_ID='123',SLURMD_NODENAME=t['node'],SLURM_JOB_NUM_NODES='1',SLURM_CPUS_PER_TASK='4',SLURM_MEM_PER_NODE='65536')
            for key in ('SLURM_ARRAY_TASK_ID','SLURM_RESTART_COUNT'):
                with patch.dict(runner.os.environ,dict(e,**{key:'1'}),clear=True),self.assertRaises(ValueError):
                    runner.allocation(root,c)

    def test_registered_worker_three_fresh_outputs_with_shared_identity(self):
        from test_r5_d5_l6a import LongReplay
        from contextlib import ExitStack
        from types import SimpleNamespace
        import json
        runs,_=LongReplay().runs()
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp)
            p=dict(source={'commit':'FIXTURE'},plan_id='FIXTURE')
            offline=dict(HF_HOME=tmp,TMPDIR=tmp)
            (root/'hub').mkdir()
            runner.write_once(root/'ready.json',{})
            t=dict(node='moana-y3',assets=offline,plan_id='FIXTURE',ready=digest({}),python=runner.sys.executable)
            runner.write_once(root/'trial.json',t)
            env=dict(SLURM_JOB_NUM_NODES='1',SLURM_CPUS_PER_TASK='4',SLURM_MEM_PER_NODE='65536',
                     SLURMD_NODENAME='moana-y3',CUDA_VISIBLE_DEVICES='0')
            stack.enter_context(patch.dict(runner.os.environ,env))
            stack.enter_context(patch.object(runner,'ready',return_value=p))
            stack.enter_context(patch.object(runner,'verified'))
            stack.enter_context(patch.object(runner.os,'sched_getaffinity',return_value={0,1,2,3}))
            stack.enter_context(patch.object(runner.os,'sched_setaffinity'))
            stack.enter_context(patch.object(runner,'budget',return_value=10000))
            gpu=runs[0]['gpu']
            probe=stack.enter_context(patch.object(runner.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(gpu))))
            commands=[]
            def official(argv,**opts):
                commands.append(argv[0])
                if argv[0]=='run':
                    self.assertEqual(opts['environment']['D5L_ALLOCATION_REMAINING_SECONDS'],'10000')
                    self.assertEqual(opts['environment']['CUDA_VISIBLE_DEVICES'],'0')
                    self.assertEqual('--enable-approximate-experiment' in argv,c==CONDITIONS[2])
                    self.assertIn('--enable-continuous-replay',argv)
                    self.assertIn('--enable-extended-experiment',argv)
                    self.assertEqual(Path(argv[argv.index('--output-dir')+1]),root/c)
                    runner.write_once(root/c/'run.json',dict(runs[i],cpu_concurrency=dict(affinity=[0,1,2,3])))
                return dict(status='LIGHT_RECEIPT_INTEGRITY_PASS',binding='FIXTURE')
            stack.enter_context(patch.object(runner,'official',side_effect=official))
            for i,c in enumerate(CONDITIONS):
                runner.os.environ.update(SLURM_JOB_ID=str(100+i),CUDA_VISIBLE_DEVICES='0')
                runner.write_once(root/(c+'.job.json'),dict(job=str(100+i),trial=digest(t),condition=c))
                self.assertEqual(runner.worker(root,c),'CONDITION_COMPLETE')
            self.assertEqual(commands,['run','verify']*3)
            self.assertEqual(probe.call_count,3)

    def test_submission_failure_stays_held_and_cannot_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner,'ROOT',Path('/data/khuss/repos/GP_semcache_lightperf')), \
                patch.object(runner,'ready',return_value=dict(plan_id='fixture')),patch.object(runner,'assets',return_value={}):
            root=Path(tmp)
            runner.write_once(root/'ready.json',{})
            with patch.object(runner,'command',side_effect=['','101',ValueError('partition rejected')]) as command:
                with self.assertRaisesRegex(ValueError,'partition rejected'):runner.submit(root,'moana-y3')
                with self.assertRaises(FileExistsError):runner.submit(root,'moana-y3')
            self.assertEqual(command.call_count,3)
            self.assertFalse(any(call.args[0][0]=='scontrol' for call in command.call_args_list))

    def test_timeout_reconciliation_is_write_once(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner,'command',return_value='42|TIMEOUT|0:9'):
            root=Path(tmp);c=CONDITIONS[0]
            runner.write_once(root/(c+'.job.json'),dict(job='42'))
            dest=root/(c+'.scheduler_failure.json')
            self.assertEqual(runner.scheduler_status(root)[c]['state'],'TIMEOUT')
            self.assertFalse(dest.exists())
            runner.scheduler_status(root,record_failures=True)
            self.assertEqual(runner.read_json(dest)['status'],'PARTIAL_BLOCKED')
            before=dest.stat()
            runner.scheduler_status(root,record_failures=True)
            self.assertEqual(before.st_mtime_ns,dest.stat().st_mtime_ns)

    def test_prepare_requires_two_fresh_approvals(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as s:
            root=Path(tmp)/'native'
            s.enter_context(patch.dict(runner.os.environ,{}))
            s.enter_context(patch.object(runner,'assets',return_value=dict(HF_HOME=tmp,TMPDIR=tmp)))
            s.enter_context(patch.object(runner,'NS',Path(tmp)))
            s.enter_context(patch.object(binding.l6,'NS',Path(tmp)/'cpu_plans'))
            s.enter_context(patch.object(runner.l6,'source'))
            s.enter_context(patch.object(runner.subprocess,'run'))
            steps=[]
            def execute(args):
                steps.append(args.command)
                return dict(status='STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING' if args.command=='prepare-cpu' else 'CPU_APPROVED')
            s.enter_context(patch.object(runner.l6,'execute',side_effect=execute))
            s.enter_context(patch.object(runner.l6,'checked'))
            s.enter_context(patch.object(runner,'native_plan',return_value=dict(plan_id='FIXTURE')))
            official=s.enter_context(patch.object(runner,'official'))
            check=s.enter_context(patch.object(runner.light,'approval',side_effect=ValueError('missing native audit')))
            with self.assertRaisesRegex(ValueError,'missing native audit'):
                runner.prepare(root,'fixture','fixture')
            self.assertEqual(steps,['prepare-cpu','cpu-validate'])
            official.assert_called_once_with(['cpu-validate','--experiment',root/'plan','--output-dir',root/'cpu'])
            check.assert_called_once_with(root/'plan',root/'cpu')
            self.assertFalse((root/'ready.json').exists())
