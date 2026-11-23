"""CPU recovery contracts. No scheduler or GPU calls."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from semcache.edgelora import r5_d5_l6_recovery as rec
from semcache.edgelora.communication import digest
from semcache.edgelora.r5_d5_lightperf_workload import CONDITIONS
import test_r5_d5_l6a as fixtures


class Recovery(unittest.TestCase):
    def test_affinity_numbers_can_differ_but_gpu_and_node_cannot(self):
        old=dict(node='moana-y3',affinity=[40,41,88,89],gpu=dict(uuid='0dff7717-a97a-b957-b478-f26213d1b38e'))
        new=dict(old,affinity=[1,2,3,4])
        rec.gpu_node_match(old,new)
        for bad in (dict(new,node='moana-y2'),dict(new,affinity=[1]),dict(new,gpu=dict(uuid='00000000-0000-0000-0000-000000000001'))):
            with self.assertRaises(ValueError):rec.gpu_node_match(old,bad)

    def test_dependency_addition_deletion_and_algorithm_changes_rejected(self):
        key='src/semcache/cache/global_cache.py'
        for old,new in (({}, {key:'a'}),({key:'a'},{}),({key:'a'},{key:'b'})):
            with self.assertRaisesRegex(ValueError,'dependency'):rec.dependencies(old,new)
        result=rec.dependencies({key:'a'},{key:'a','src/semcache/edgelora/r5_d5_l6_recovery.py':'new'})
        self.assertEqual(result['status'],'ALGORITHM_DEPENDENCIES_IDENTICAL')

    def test_recovery_plan_binds_original_inventory_and_payload(self):
        base=dict(source=dict(files={}),entry_limit=64,count=50,stream={'fixture':1},plan_id='base')
        old=dict(base,source=dict(files={}))
        evidence=dict(plan=old,bindings={'run':'fixture'},allocations=[])
        contract=dict(original=str(rec.ORIGINAL),evidence={k:v for k,v in evidence.items() if k!='plan'},
                      dependencies=rec.dependencies({},{}))
        with patch.object(rec,'original',return_value=evidence):
            result=rec.bind_plan(base,contract)
            self.assertNotEqual(result['plan_id'],'base')
            for change in (dict(base,entry_limit=128),dict(base,count=20),dict(base,stream={})):
                with self.assertRaises(ValueError):rec.bind_plan(change,contract)
            bad=deepcopy(contract);bad['evidence']['bindings']={}
            with self.assertRaises(ValueError):rec.bind_plan(base,bad)

    def test_only_one_approx_job_dry_run_and_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec.runner.write_once(root/'ready.json',{})
            p=dict(recovery={},plan_id='FIXTURE')
            evidence=dict(allocations=[dict(node='moana-y3')])
            with patch.object(rec.runner,'ready',return_value=p),patch.object(rec,'original',return_value=evidence), \
                 patch.object(rec.runner,'assets',return_value={}),patch.object(rec,'ROOT',Path('/data/khuss/repos/GP_semcache_lightperf')), \
                 patch.object(rec.runner,'command',side_effect=['','4242','']) as command:
                dry=rec.submit(root,dry_run=True)
                command.assert_not_called()
                self.assertFalse((root/'trial.json').exists())
                args=dry['argv']
                self.assertIn('--hold',args);self.assertIn('--no-requeue',args)
                self.assertEqual(args[-1],CONDITIONS[2])
                self.assertTrue(args[-4].endswith('jobs/d5_l6_recovery.sh'))
                rec.submit(root)
                self.assertEqual(command.call_count,3)
                self.assertEqual(command.call_args_list[-1].args[0],['scontrol','release','4242'])
                with self.assertRaises(FileExistsError):rec.submit(root)
                self.assertFalse((root/(CONDITIONS[0]+'.job.json')).exists())

    def test_missing_original_blocks_without_launch(self):
        with patch.object(rec,'ORIGINAL',Path('/tmp/ABSENT_RECOVERY_REFERENCE')),patch.object(rec.runner,'command') as cmd:
            with self.assertRaisesRegex(ValueError,'BLOCKED_MISSING_REFERENCE'):rec.original(rec.ORIGINAL)
            cmd.assert_not_called()

    def test_cross_trial_analysis_has_no_matched_complete_or_speedup(self):
        runs,stream=fixtures.LongReplay().runs()
        evidence=dict(allocations=[])
        p=dict(recovery={'evidence':evidence},stream=stream)
        alloc=dict(node='moana-y3',affinity=[1,2,3,4],gpu=runs[0]['gpu'])
        p['recovery']['evidence']=dict(allocations=[alloc],bindings={})
        runs[2]['source']={'changed':'CONTROL_ONLY'}
        runs[2]['cpu_concurrency']=dict(affinity=[1,2,3,4])
        with tempfile.TemporaryDirectory() as tmp,patch.object(rec.runner,'ready',return_value=p), \
             patch.object(rec,'original',return_value=dict(allocations=[alloc],bindings={})), \
             patch.object(rec.runner,'verified',return_value=alloc), \
             patch.object(rec,'verify_run',side_effect=[(r,{}) for r in runs]),patch.object(rec,'verify',return_value={}), \
             patch('semcache.edgelora.r5_d5_l6a_analysis.write_report') as write:
            rec.analyze(Path(tmp))
            report=write.call_args.args[0]
            self.assertEqual(report['status'],'LIMITED_COMPARABILITY')
            self.assertEqual(report['strict_comparison']['status'],'NOT_COMPARABLE')
            self.assertEqual(len(report['cycle_metrics']),15)
            self.assertEqual([x['requests'] for x in report['cumulative_metrics']],[10,20,30,40,50]*3)
            self.assertTrue(all(x['descriptive_full_over_condition_e2e'] is None for x in report['cycle_metrics']))
            self.assertFalse((Path(tmp)/'matched.json').exists())

    def test_approx_worker_uses_new_four_core_allocation(self):
        from contextlib import ExitStack
        import json
        gpu=dict(uuid='0dff7717-a97a-b957-b478-f26213d1b38e')
        old=dict(node='moana-y3',affinity=[40,41,88,89],gpu=gpu)
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp)
            rec.runner.write_once(root/'ready.json',{})
            contract=dict(original=str(rec.ORIGINAL),evidence=dict(allocations=[old]))
            p=dict(plan_id='fixture',source=dict(commit='fixture'),recovery=contract)
            assets=dict(HF_HOME=tmp,TMPDIR=tmp)
            trial=dict(node='moana-y3',assets=assets,plan_id='fixture',ready=digest({}))
            stack.enter_context(patch.dict(rec.os.environ,CUDA_VISIBLE_DEVICES='0',SLURM_JOB_ID='newjob'))
            for name,value in (('allocation',trial),('ready',p),('assets',assets),('budget',10000)):
                stack.enter_context(patch.object(rec.runner,name,return_value=value))
            stack.enter_context(patch.object(rec.os,'sched_getaffinity',return_value={1,2,3,4}))
            pin=stack.enter_context(patch.object(rec.os,'sched_setaffinity'))
            probe=stack.enter_context(patch.object(rec.runner.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(gpu))))
            stack.enter_context(patch.object(rec,'original',return_value=dict(allocations=[old])))
            stack.enter_context(patch.object(rec,'topology',return_value=dict(scope='FIXTURE')))
            calls=[]
            def official(argv,**opts):
                calls.append(argv[0])
                if argv[0]=='run':
                    self.assertEqual(argv[argv.index('--condition')+1],CONDITIONS[2])
                    self.assertIn('--enable-approximate-experiment',argv)
                    rec.runner.write_once(root/CONDITIONS[2]/'run.json',dict(gpu=gpu,cpu_concurrency=dict(affinity=[1,2,3,4])))
                return dict(status='LIGHT_RECEIPT_INTEGRITY_PASS',binding='FIXTURE')
            stack.enter_context(patch.object(rec.runner,'official',side_effect=official))
            self.assertEqual(rec.runner.worker(root,CONDITIONS[2],recovery_mode=True),'CONDITION_COMPLETE')
            self.assertEqual(calls,['run','verify'])
            pin.assert_called_once_with(0,[1,2,3,4])
            probe.assert_called_once()
            rec.os.environ['CUDA_VISIBLE_DEVICES']='0'
            with self.assertRaisesRegex(ValueError,'APPROX only'):
                rec.runner.worker(root,CONDITIONS[0],recovery_mode=True)
