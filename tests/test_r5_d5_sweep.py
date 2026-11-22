"""Offline UNIT_ONLY sweep fixtures. No Slurm/GPU or real approvals are created."""
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from semcache.edgelora import r5_d5_sweep as sweep
from semcache.edgelora import r5_d5_sweep_analysis as analysis
from semcache.edgelora.communication import digest
from semcache.edgelora.r5_d5_lightperf_io import ENGINE,MODE,SCHEMA,SummaryWriter,verify
from semcache.edgelora.r5_d5_lightperf_workload import knobs,preflight,CONDITIONS
from semcache.edgelora.r5_d5_lightperf_replay import replay_stream
from semcache.edgelora.r5_d5_lightperf_replay_analysis import run_segments,compare_segments
from semcache.edgelora.r5_d5_lightperf_analysis import compare
from semcache.edgelora.r4c_workload import policy
from test_r5_d5_lightperf import workload,Clock


def replay():
    w,tasks=workload('2wikimqa',n=10)
    for t in tasks:
        t['prepared']['max_new_tokens']=8
        t['policy']=policy(w['special_tokens'],'quality',8)
    for b in w['blocks']:
        b['block_id']=digest({k:v for k,v in b.items() if k!='block_id'})
    return replay_stream(w,'2wikimqa',20,enable_continuous_replay=True)


def frozen_fixture(root):
    assets=root/'UNIT_ONLY_ASSETS';assets.mkdir()
    p=dict(engine=ENGINE,schema=SCHEMA,mode=MODE,dataset='2wikimqa',count=20,stream=replay(),seed=42,
        source=dict(approval_scope='CLEAN_REVIEWED_SOURCE',commit='UNIT_ONLY_NOT_REAL_APPROVAL',content_fingerprint='UNIT_ONLY'),
        original_experiment=str(assets),multi=str(assets),strict_pilot=None,baseline_plan=None,continuous_contract=None,
        ud_operation_seconds=.001,workload_mode='CONTINUOUS_REPLAY',enable_continuous_replay=True,
        impact_implementation=sweep.MASK_INDEX_REUSE,performance_knobs=knobs(50,.25),
        validation_mode='REQUEST_SCOPED_VERIFIED',conditions=list(CONDITIONS),warmup_requests=0,approx_requires_explicit_run_opt_in=True)
    for key in ('original_inventory','multi_inventory','base_prepared','tokenizer_identity','semantic_identity',
                'dataset_inventories','dataset_paths','dataset_revision','dataset_split','strict_pilot_inventory','thresholds','transport'):
        p[key]={'UNIT_ONLY':key}
    p['plan_id']=digest(p)
    plan=root/'frozen';plan.mkdir();writer=SummaryWriter(plan);writer.write('plan.json',p);pi=writer.seal()
    cpu=root/'cpu';cpu.mkdir();writer=SummaryWriter(cpu)
    writer.write('cpu_validation.json',dict(engine=ENGINE,status='PASS',failed=0,errors=0,skipped=0,
        scope='CLEAN_NATIVE_CPU_APPROVAL',source=p['source'],plan_inventory=pi))
    writer.seal()
    return plan/'plan.json',cpu,p


def run_fixtures():
    selected=replay();runs=[]
    for ci,condition in enumerate(CONDITIONS):
        rows=[]
        for i,t in enumerate(selected['tasks']):
            rows.append(dict(request_id=t['request_id'],sample_id=t['sample_id'],user_id=t['user_id'],adapter_id=t['adapter_id'],
                input_tokens=4,output_tokens=1,generated_token_ids=[9] if ci==2 and i>=6 else [3],
                arrival_ns=i*200_000_000,completion_ns=i*200_000_000+int((30+i-ci)*1e6),
                application_e2e_ms=30.+i-ci,application_ttft_ms=5.+i/10,prefill_ms=3.,decode_ms=2.,
                fresh_qkv_rows=384 if ci==0 else 288,skipped_qkv_rows=0 if ci==0 else 96,physical_hit=ci!=0,
                quality=dict(metric='UNIT_ONLY_F1',score=.01),replay_provenance=selected['occurrences'][i],
                policy_counts=dict(INSERT=1 if i<10 else 0,EVICT=1 if i==19 else 0,CHU=0,PBR=0),
                wire={op:dict(wire_bytes=100+i,socket_call_wall_ms=1.+i/10) for op in ('send','recv')},
                cache_completion_costs=dict(owner_completion_ms=2.+i/10)))
        runs.append(dict(dataset='2wikimqa',scale=20,condition=condition,tasks_sha256=selected['tasks_sha256'],
            source=dict(commit='UNIT_ONLY',content_fingerprint='UNIT_ONLY'),performance_knobs=knobs(50,.25),
            backend={'UNIT_ONLY':True},precision_policy='UNIT_ONLY',cpu_concurrency=dict(affinity=[1,2,3,4],es_threads=4),
            warmup_requests=0,measurement_scope='NATIVE_LIGHT',impact_implementation=sweep.MASK_INDEX_REUSE,
            workload_mode='CONTINUOUS_REPLAY',replay_stream_sha256=digest(selected),requests=rows,sequence_span_ms=3849.-ci,
            gpu=dict(uuid='GPU-00000000-0000-0000-0000-000000000001',name='UNIT_ONLY',device='cuda:0',total_bytes=24*2**30)))
    return runs


def official_fixture_docs(runs):
    bindings=[dict(run={'UNIT_ONLY':i},plan={'UNIT_ONLY':'plan'},cpu={'UNIT_ONLY':'cpu'}) for i in range(3)]
    a=dict(bindings=bindings,comparisons=[compare(runs[0],r) for r in runs[1:]])
    positions=dict(runs=[dict(run_segments(r),impact_implementation=sweep.MASK_INDEX_REUSE) for r in runs],
        comparisons=[dict(dataset=r['dataset'],scale=20,condition=r['condition'],impact_implementation=sweep.MASK_INDEX_REUSE,
            **compare_segments(runs[0],r)) for r in runs[1:]])
    return a,positions,bindings


class FactorSupport(unittest.TestCase):
    def test_allowed_and_rejected_values(self):
        for factor in (1.,.5,.25,.10):
            self.assertEqual(knobs(50,factor)['ud_throughput_factor'],factor)
        for factor in (.09,.2,-.1,0,float('nan'),float('inf'),-float('inf'),True,'0.10'):
            with self.subTest(factor=factor),self.assertRaises(ValueError):knobs(50,factor)

    def test_cli_accepts_point_one_and_rejects_invalid(self):
        from semcache.edgelora import r5_d5_lightperf as api
        import contextlib,io
        args=['prepare','--original-experiment','UNIT_ONLY','--multi','UNIT_ONLY','--dataset','2wikimqa',
              '--count','20','--output-dir','UNIT_ONLY','--ud-throughput-factor']
        with patch.object(api,'prepare',return_value={'UNIT_ONLY':True}) as prepare,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(api.main(args+['0.10']),0)
        self.assertEqual(prepare.call_args.kwargs['ud_throughput_factor'],.10)
        for value in ('0.09','0.2','-1','nan','inf'):
            with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):api.main(args+[value])

    def test_actual_compute_runs_once_and_point_one_paces_nine_times_raw(self):
        from semcache.edgelora.r5_d5_lightperf_emulation import UDServiceEmulator
        clock=Clock();calls=[];service=UDServiceEmulator(.10,clock=clock,sleep=clock.sleep)
        def compute():calls.append(1);clock.now+=.004;return ['UNIT_ONLY',42]
        self.assertEqual(service.run_lora(compute,deadline=20.),['UNIT_ONLY',42])
        report=service.report()
        self.assertEqual(len(calls),1);self.assertAlmostEqual(report['measured_compute_ms'],4.)
        self.assertAlmostEqual(report['pacing_ms'],36.);self.assertAlmostEqual(report['measured_service_ms'],40.)
        self.assertEqual(report['scope'],'EMULATED_UD_SERVICE_TIME')
        with self.assertRaises(TimeoutError):service.run_lora(compute,deadline=clock()+.03)

    def test_preflight_preserves_calibration_and_deadline_caps(self):
        tasks=replay()['tasks']
        blocked=preflight(tasks,knobs(50,.10))
        self.assertIn('UD_OPERATION_CALIBRATION_REQUIRED',blocked['reasons'])
        bounded=preflight(tasks,knobs(50,.10),ud_operation_seconds=.001)
        self.assertAlmostEqual(bounded['ud_extra_service_seconds'],160*32*.001*9)
        self.assertLessEqual(bounded['worker_deadline_seconds'],7200)
        self.assertEqual(bounded['slurm_seconds'],16200)
        blocked=preflight(tasks,knobs(50,.10),ud_operation_seconds=.5)
        self.assertEqual(blocked['status'],'BLOCKED_INFEASIBLE')
        self.assertIn('DEADLINE_OR_SLURM_BOUND',blocked['reasons'])
        with self.assertRaises(ValueError):preflight(tasks,knobs(50,.10),slurm_seconds=16201)


class SweepContracts(unittest.TestCase):
    def test_matrix_exact_twelve_and_balanced_condition_orders(self):
        cells=sweep.cells();self.assertEqual(len(cells),12)
        self.assertEqual(len({c['id'] for c in cells}),12)
        self.assertEqual(cells[3]['id'],'bw050_ud010')
        for c in cells:self.assertEqual(set(c['order']),set(CONDITIONS))
        self.assertEqual({c['performance_knobs']['ud_throughput_factor'] for c in cells},{1.,.5,.25,.1})
        with self.assertRaises(ValueError):sweep.select('all',submit=True)
        with self.assertRaises(ValueError):sweep.select(','.join(c['id'] for c in cells),submit=True)
        for invalid in ('','bw050_ud020','bw050_ud010,bw050_ud010'):
            with self.assertRaises(ValueError):sweep.select(invalid)

    def test_missing_reference_returns_blocked_without_cpu_or_scheduler_calls(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(sweep.subprocess,'run') as run:
            result=sweep.check(Path(tmp)/'absent.json',Path(tmp)/'absent_cpu','moana-y2')
        self.assertEqual(result['status'],'BLOCKED_MISSING_REFERENCE')
        self.assertEqual(len(result['matrix']),12);run.assert_not_called()

    def test_reference_reads_true_nulls_and_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,cpu,p=frozen_fixture(Path(tmp));before=(plan.read_bytes(),verify(cpu))
            read,contract,summary=sweep.historical_reference(plan,cpu)
            self.assertEqual(read,p);self.assertIsNone(contract)
            for key in ('continuous_contract','strict_pilot','baseline_plan'):self.assertIsNone(summary['fields'][key])
            self.assertEqual(before,(plan.read_bytes(),verify(cpu)))
            self.assertEqual(summary['tasks_sha256'],p['stream']['tasks_sha256'])

    def test_reference_tamper_and_changed_workload_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,cpu,p=frozen_fixture(Path(tmp))
            changed=deepcopy(p);changed['seed']=99
            with self.assertRaises(ValueError):sweep.same_workload(changed,p)
            changed=deepcopy(p);changed['stream']['tasks'].reverse()
            with self.assertRaises(ValueError):sweep.same_workload(changed,p)
            plan.write_text('{}')
            with self.assertRaises(ValueError):sweep.historical_reference(plan,cpu)

    def test_output_must_not_modify_reference_or_original_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,cpu,p=frozen_fixture(root)
            reference=sweep.historical_reference(plan,cpu)[2]
            for bad in (plan.parent,plan.parent/'new_state',cpu/'new_state',Path(p['multi'])/'new_state',root):
                with self.subTest(path=bad),self.assertRaisesRegex(ValueError,'overlaps'):
                    sweep.protect_reference_output(bad,p,reference)
            sweep.protect_reference_output(root/'fresh_sibling',p,reference)

    def test_review_guard_precedes_creation_or_official_prepare(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(sweep,'source_identity',side_effect=ValueError('new clean review required')), \
             patch.object(sweep,'official') as official:
            args=SimpleNamespace(state=Path(tmp)/'untouched',cells='all',reviewed_commit='OLD',reviewed_fingerprint='OLD',
                                 frozen_plan=Path(tmp)/'absent',frozen_cpu=Path(tmp)/'absent')
            with self.assertRaisesRegex(ValueError,'clean review'):sweep.prepare_cpu(args)
            self.assertFalse(args.state.exists());official.assert_not_called()

    def test_prepare_uses_exact_reference_fields_and_unique_plan_audit_per_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,cpu,p=frozen_fixture(root);state=root/'state'
            source=dict(commit='UNIT_ONLY_NEW',content_fingerprint='UNIT_ONLY_NEW',approval_scope='CLEAN_REVIEWED_SOURCE')
            args=SimpleNamespace(state=state,cells='bw050_ud100,bw050_ud010',reviewed_commit='UNIT_ONLY',reviewed_fingerprint='UNIT_ONLY',frozen_plan=plan,frozen_cpu=cpu)
            calls=[]
            def official(argv,**kwargs):
                argv=list(map(str,argv));calls.append(argv);command=argv[0]
                base=state/'prepared'/('bw050_ud100' if len([c for c in calls if c[0]=='prepare'])==1 else 'bw050_ud010')
                if command=='prepare':
                    output=Path(argv[argv.index('--output-dir')+1]);output.mkdir()
                    new=deepcopy(p);new['source']=source
                    new['performance_knobs']=knobs(float(argv[argv.index('--bandwidth-mbps')+1]),float(argv[argv.index('--ud-throughput-factor')+1]))
                    new['resources']={'status':'STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING','reasons':[]}
                    new.pop('plan_id');new['plan_id']=digest(new)
                    writer=SummaryWriter(output);writer.write('plan.json',new);writer.seal()
                if command=='cpu-validate':
                    output=Path(argv[argv.index('--output-dir')+1]);output.mkdir();writer=SummaryWriter(output)
                    writer.write('cpu_validation.json',{'UNIT_ONLY':True});writer.seal()
                return {'UNIT_ONLY':command}
            with patch.object(sweep,'source_identity',return_value=source),patch.object(sweep,'definition',return_value={'source':source,'reference':sweep.historical_reference(plan,cpu)[2]}),patch.object(sweep,'official',side_effect=official):
                sweep.prepare_cpu(args)
                with self.assertRaisesRegex(ValueError,'fresh cell'):sweep.prepare_cpu(args)
            prepared=[c for c in calls if c[0]=='prepare'];self.assertEqual(len(prepared),2)
            self.assertEqual([c[0] for c in calls],['prepare','preflight','cpu-validate','inspect']*2)
            for c in prepared:
                self.assertNotIn('--development',c);self.assertNotIn('--continuous-contract',c)
                self.assertNotIn('--strict-pilot',c);self.assertNotIn('--baseline-plan',c)
                self.assertEqual(c[c.index('--original-experiment')+1],p['original_experiment'])
                self.assertEqual(c[c.index('--ud-operation-seconds')+1],str(p['ud_operation_seconds']))
            self.assertNotEqual(prepared[0][prepared[0].index('--output-dir')+1],prepared[1][prepared[1].index('--output-dir')+1])

    def test_deadline_endtime_is_strict_and_never_extended(self):
        now=datetime(2026,11,22,0,0).timestamp()
        self.assertEqual(sweep.remaining_seconds('EndTime=2026-11-22T04:30:00',7200,now),16200)
        for text in ('EndTime=2026-11-22T02:02:00','EndTime=2026-11-22T05:00:00','EndTime=Unknown','NoEndTime'):
            with self.assertRaises(ValueError):sweep.remaining_seconds(text,7200,now)

    def test_auto_requires_local_documented_subset_semantics(self):
        with self.assertRaises(ValueError):sweep.auto_node_arguments('Request all nodes in the list')
        args=sweep.auto_node_arguments(sweep.MAN_NODELIST_TEXT)
        self.assertEqual(args,['--nodelist='+','.join(sweep.D5_NODES)])
        self.assertNotIn('moana-y6',args[0])

    def test_payload_hash_and_slurm_guard_fail_before_any_gpu_probe(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(sweep.subprocess,'run') as run:
            p=Path(tmp)/'submission.json';sweep.write_once(p,{'UNIT_ONLY':True})
            with self.assertRaises(ValueError):sweep.job(SimpleNamespace(payload=p,payload_sha256='TAMPERED'))
            run.assert_not_called()

    def test_submit_only_selected_canaries_one_job_each_afterok_and_no_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=SimpleNamespace(state=root,cells=','.join(sweep.CANARIES),node='moana-y2',trial=0,tmpdir=root,hf_home=root)
            commands=[];next_id=[100]
            def run(argv,**kwargs):
                commands.append(argv)
                if argv[0]=='sbatch' and '--test-only' not in argv:
                    value=str(next_id[0]);next_id[0]+=1
                    self.assertNotIn('CUDA_VISIBLE_DEVICES',kwargs['env'])
                    return SimpleNamespace(returncode=0,stdout=value+'\n',stderr='')
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            with patch.object(sweep,'definition',return_value={'source':{'UNIT_ONLY':True},'python':'UNIT_ONLY'}), \
                 patch.object(sweep,'registered') as registered,patch.object(sweep.subprocess,'run',side_effect=run):
                sweep.submit(args)
            registered.assert_any_call(root,sweep.CANARIES[0],{'source':{'UNIT_ONLY':True},'python':'UNIT_ONLY'})
            jobs=[c for c in commands if c[0]=='sbatch' and '--test-only' not in c]
            self.assertEqual(len(jobs),2)
            self.assertIn('--dependency=afterok:100',jobs[1]);self.assertIn('--hold',jobs[0])
            self.assertTrue(all('--nodes=1-1' in c and '--ntasks=1' in c and '--nodelist=moana-y2' in c for c in jobs))
            self.assertFalse(any(any('array' in x for x in c) for c in jobs))
            self.assertEqual([c for c in commands if c[:2]==['scontrol','release']],[['scontrol','release','100'],['scontrol','release','101']])

    def test_missing_other_cell_approval_stops_before_first_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=SimpleNamespace(state=root,cells=','.join(sweep.CANARIES),node='moana-y1',trial=0,tmpdir=root,hf_home=root)
            with patch.object(sweep,'definition',return_value={}),patch.object(sweep,'registered',side_effect=[None,ValueError('stale audit')]),patch.object(sweep.subprocess,'run') as run:
                with self.assertRaisesRegex(ValueError,'stale audit'):sweep.submit(args)
                run.assert_not_called()

    def test_registered_cell_rejects_other_knob_and_audit_binding(self):
        from semcache.edgelora import r5_d5_lightperf as api
        for wrong in ('knob','audit'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);plan,cpu,p=frozen_fixture(root);cid='bw050_ud010'
                p['performance_knobs']=knobs(50,.25 if wrong=='knob' else .10)
                ready=dict(schema=sweep.SCHEMA,cell=cid,source=p['source'],plan_inventory={'UNIT_ONLY':'p'},
                           cpu_inventory={'UNIT_ONLY':'wrong' if wrong=='audit' else 'a'},performance_knobs=p['performance_knobs'])
                sweep.write_once(root/'prepared'/cid/'ready.json',ready)
                with patch.object(api,'approval',return_value=(p,{'UNIT_ONLY':'p'},{'UNIT_ONLY':'a'})):
                    with self.assertRaises(ValueError):sweep.registered(root,cid,{'source':p['source']})

    def test_noncanary_is_blocked_until_official_canary_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=SimpleNamespace(state=root,cells='bw100_ud100',node='moana-y3',trial=0,tmpdir=root,hf_home=root)
            with patch.object(sweep,'definition',return_value={}),patch.object(sweep.subprocess,'run') as run:
                with self.assertRaisesRegex(ValueError,'canaries'):sweep.submit(args)
                run.assert_not_called()

    def test_worker_uses_one_dynamic_uuid_same_affinity_and_three_fresh_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);attempt=root/'attempts/unit';attempt.mkdir(parents=True)
            source=dict(commit='UNIT_ONLY',content_fingerprint='UNIT_ONLY')
            payload=attempt/'submission.json'
            data=dict(schema=sweep.SCHEMA,state=str(root),cell='bw050_ud025',trial=0,source=source,
                      requested_node='auto',tmpdir=str(root),hf_home=str(root),order=[CONDITIONS[2],CONDITIONS[0],CONDITIONS[1]])
            sweep.write_once(payload,data);sha=digest(data)
            bound=[dict(kind='SUBMITTED',attempt=str(attempt),job_id='123',payload_sha256=sha)]
            native=[];controls=[]
            gpu=dict(uuid='GPU-00000000-0000-0000-0000-000000000123',name='UNIT_ONLY',total_bytes=24*2**30)
            def process(argv,**kwargs):
                controls.append(argv)
                if argv[0]=='scontrol':return SimpleNamespace(returncode=0,stdout='JobId=123 Partition=batch_eebme_ugrad EndTime=UNIT_ONLY',stderr='')
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'2')
                return SimpleNamespace(returncode=0,stdout=json.dumps(gpu),stderr='')
            def official(argv,**kwargs):
                native.append(list(map(str,argv)))
                if argv[0]=='run':
                    output=Path(argv[argv.index('--output-dir')+1]);output.mkdir()
                    sweep.write_once(output/'run.json',dict(gpu=gpu,cpu_concurrency=dict(affinity=[1,2,3,4])))
                return {'UNIT_ONLY':True}
            env=dict(SLURM_JOB_ID='123',SLURM_JOB_NUM_NODES='1',SLURM_CPUS_PER_TASK='4',SLURM_MEM_PER_NODE='65536',
                     SLURMD_NODENAME='moana-y2',CUDA_VISIBLE_DEVICES='2')
            with patch.dict(os.environ,env,clear=True),patch.object(sweep,'definition',return_value={'source':source}), \
                 patch.object(sweep,'events',return_value=bound),patch.object(sweep,'registered',return_value=({'resources':{'worker_deadline_seconds':7200}},root/'prepared')), \
                 patch.object(Path,'is_relative_to',return_value=True),patch.object(sweep.os,'sched_getaffinity',return_value={1,2,3,4}), \
                 patch.object(sweep.os,'sched_setaffinity') as pin,patch.object(sweep,'remaining_seconds',return_value=15000) as budget, \
                 patch.object(sweep.subprocess,'run',side_effect=process),patch.object(sweep,'official',side_effect=official), \
                 patch.object(analysis,'validated_cell',return_value={'gate':'MATCHED_COMPLETE'}):
                result=sweep.job(SimpleNamespace(payload=payload,payload_sha256=sha))
            self.assertEqual(result['status'],'MATCHED_COMPLETE');self.assertEqual(budget.call_count,4)
            pin.assert_called_once_with(0,[1,2,3,4])
            runs=[c for c in native if c[0]=='run'];self.assertEqual(len(runs),3)
            self.assertEqual([c[c.index('--condition')+1] for c in runs],data['order'])
            self.assertEqual(len({c[c.index('--output-dir')+1] for c in runs}),3)
            self.assertTrue(all(c[c.index('--expected-gpu-uuid')+1]==gpu['uuid'] for c in runs))
            self.assertIn('--enable-approximate-experiment',runs[0]);self.assertNotIn('--enable-approximate-experiment',runs[1])
            self.assertEqual([c[0] for c in native].count('verify'),3)
            self.assertEqual([c[0] for c in native].count('analyze'),1)


class Aggregation(unittest.TestCase):
    def test_exact_twenty_required_approx_six_is_allowed(self):
        runs=run_fixtures();a,p,b=official_fixture_docs(runs)
        by,comp,gate=analysis.review_runs(runs,a,p,b)
        self.assertEqual(gate,'MATCHED_COMPLETE');self.assertEqual(comp[CONDITIONS[2]]['generated_token_agreements'],6)
        runs[1]['requests'][0]['generated_token_ids']=[99];a,p,b=official_fixture_docs(runs)
        self.assertEqual(analysis.review_runs(runs,a,p,b)[2],'PARTIAL_BLOCKED')

    def test_not_comparable_is_preserved_and_not_silently_accepted(self):
        runs=run_fixtures();runs[2]['cpu_concurrency']['affinity']=[5,6,7,8]
        a,p,b=official_fixture_docs(runs);_,comp,gate=analysis.review_runs(runs,a,p,b)
        self.assertEqual(gate,'PARTIAL_BLOCKED');self.assertIsNone(comp[CONDITIONS[2]]['speedup'])
        self.assertIn('cpu_concurrency',comp[CONDITIONS[2]]['mismatches'])

    def test_analysis_position_or_binding_tamper_is_rejected(self):
        for which in ('analysis','positions','binding'):
            runs=run_fixtures();a,p,b=official_fixture_docs(runs)
            if which=='analysis':a['comparisons'][0]['speedup']=1000
            if which=='positions':p['runs'][0]['windows'][0]['requests']=99
            if which=='binding':a['bindings']=[]
            with self.subTest(which=which),self.assertRaises(ValueError):analysis.review_runs(runs,a,p,b)

    def test_validated_bins_raw_values_and_csv_are_write_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);attempt=root/'state/attempts/unit';attempt.mkdir(parents=True)
            runs=run_fixtures();a,p,b=official_fixture_docs(runs)
            sweep.write_once(attempt/'registry.json',[dict(run=str(i),plan='UNIT_ONLY',cpu_audit='UNIT_ONLY') for i in range(3)])
            sweep.write_once(attempt/'submission.json',dict(cell='bw050_ud025',trial=0))
            sweep.write_once(attempt/'allocation.json',dict(order=list(CONDITIONS),node='moana-y1',gpu=runs[0]['gpu'],affinity=[1,2,3,4]))
            (attempt/'analysis').mkdir();sweep.write_once(attempt/'analysis/analysis.json',a);sweep.write_once(attempt/'analysis/position_analysis.json',p)
            sweep.write_once(root/'state/definition.json',dict(schema=sweep.SCHEMA,source=runs[0]['source'],reference=dict(tasks_sha256=runs[0]['tasks_sha256'])))
            def load(run,plan,cpu):i=int(run);return deepcopy(runs[i]),deepcopy(b[i])
            with patch.object(analysis,'verify_run',side_effect=load) as verified,patch.object(analysis,'verify',return_value={'UNIT_ONLY':True}):
                review=analysis.validated_cell(attempt)
                first=next(r for r in review['rows'] if r['condition']==CONDITIONS[0] and r['segment']=='first_10')
                repeat=next(r for r in review['rows'] if r['condition']==CONDITIONS[0] and r['segment']=='repeat_10')
                self.assertEqual(first['e2e_mean_ms'],34.5);self.assertEqual(repeat['e2e_mean_ms'],44.5)
                self.assertEqual(first['cache_insert'],10);self.assertEqual(repeat['cache_insert'],0)
                self.assertEqual(repeat['cache_evict'],1);self.assertEqual(first['tcp_send_inclusive_ms'],1.45)
                self.assertAlmostEqual(first['owner_completion_ms'],2.45)
                output=root/'out';analysis.aggregate(root/'state',output)
                self.assertTrue((output/'grid.csv').is_file());self.assertTrue((output/'first10_repeat10.csv').is_file())
                body=(output/'sweep.json').read_text();self.assertIn('low F1',body);self.assertIn('NOT_ESTABLISHED',body)
                with self.assertRaises(ValueError):analysis.aggregate(root/'state',output)
                self.assertGreaterEqual(verified.call_count,6)

    def test_not_comparable_exports_null_speedup_even_when_other_pair_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp);runs=run_fixtures();runs[2]['backend']={'UNIT_ONLY':'DIFFERENT'}
            a,p,b=official_fixture_docs(runs)
            sweep.write_once(attempt/'registry.json',[dict(run=str(i),plan='UNIT_ONLY',cpu_audit='UNIT_ONLY') for i in range(3)])
            sweep.write_once(attempt/'submission.json',dict(cell='bw050_ud025',trial=0))
            sweep.write_once(attempt/'allocation.json',dict(order=list(CONDITIONS),node='moana-y1',gpu=runs[0]['gpu'],affinity=[1,2,3,4]))
            sweep.write_once(attempt/'analysis/analysis.json',a);sweep.write_once(attempt/'analysis/position_analysis.json',p)
            with patch.object(analysis,'verify_run',side_effect=lambda r,p,c:(deepcopy(runs[int(r)]),deepcopy(b[int(r)]))), \
                 patch.object(analysis,'verify',return_value={'UNIT_ONLY':True}):
                result=analysis.validated_cell(attempt)
            self.assertEqual(result['gate'],'PARTIAL_BLOCKED')
            self.assertTrue(all(r['measured_descriptive_speedup'] is None for r in result['rows'] if r['segment']=='all_20'))

    def test_missing_timers_stay_unmeasured_not_zero(self):
        runs=run_fixtures();runs[0]['requests'][0].pop('cache_completion_costs')
        value=analysis.segment(runs[0]['requests'][:10])['timings']['owner_completion_ms']
        self.assertEqual(value['status'],'PARTIALLY_MEASURED');self.assertEqual(value['measured_requests'],9)


if __name__=='__main__':unittest.main()
