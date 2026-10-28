"""CPU-only synthetic session schema, launcher CLI and aggregation regressions."""
import copy
import importlib.util
import io
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch,Mock
from semcache.diagnostics.r2c3_analysis import aggregate,summarize,csv_summaries,approve_pilot,differences,coverage_changes
from semcache.diagnostics.r2c3_impact_replay import ARMS,BASES,SETTINGS
from semcache.diagnostics.r2c3_benchmark import paired,Runtime
from semcache.diagnostics.r1b3_break_even import MissingEvidence

ROOT=Path(__file__).resolve().parents[1]


def runner():
    spec=importlib.util.spec_from_file_location('c3_runner',ROOT/'scripts/92_run_r2c3_impact_ablation.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def session(run='a',saving=25.,count=20):
    length=512; layers=2; reused=510; case='L512_covmax'
    policies={b:dict(before_digest=b+'before',after_digest=b+'after',inputs_digest=b+'input',events_digest=b+'events') for b in BASES}
    refs={b:b+'ref' for b in BASES}; values={b:b+'values' for b in BASES}
    c=dict(status='VALID',length=length,requested_coverage='max',condition_identity='source_tokens_weights',
        structural_valid=True,cache_integrity_valid=True,memory_valid=True,source_token_integrity_valid=True,
        ablation_equivalence_pass=True,impact_replay_reduction_calls=0,measured_pairs=count,
        actual_reused_tokens=reused,actual_coverage=reused/length,reference_digests=refs,
        impact_values_digests=values,policy_proofs=policies,full_relative_strict={a:False if a in ARMS[3:] else True for a in ARMS},
        memory_after={'gpu':{'peak_allocated_bytes':8*1024**3}})
    config=dict(phase='latency',independent_run_id=run,model='facebook/opt-2.7b',dtype='float16',
        warmup_runs=3,measured_runs=count,capacity_mib=512,cache_device='cuda',device='cuda:0',seed=42,lengths=[512],coverage_values=['max'])
    config['order_seed']=4100+sum(map(ord,run))
    manifest=dict(experiment='R2-C3',execution_status='COMPLETE',GPU_VALIDATION='EXECUTED',
        environment_verified=True,
        classification='ORACLE_DIAGNOSTIC',safe_reuse_claimed=False,correctness_qualified_speedup=False,
        final_weights_verified=True,benchmark_config=config,process_instance='process_'+run,
        environment_identity='same',hardware_identity={'uuid':'GPU-same','device':'cuda:0'},
        code={'sha256':'code'},implementation_sha256={'runner.py':'sha'},model_weight_identity='weights',
        settings=dict(SETTINGS),conditions={case:c})
    rows=[]
    for phase,total in [('warmup',3),('measured',count)]:
        for i in range(total):
            arms={}
            for arm in ARMS:
                n=reused if arm in ARMS[3:] else 0
                base=BASES[1] if arm in ARMS[3:] else BASES[0]
                arms[arm]=dict(reused_tokens=n,actual_coverage=n/length,native_projection_rows=(length-n)*layers*3,
                    skipped_projection_rows=n*layers*3,selection_hash=base,selected_provenance_digest=base,
                    policy=policies[base] if arm!='FULL_NATIVE' else None,
                    reference_digest=refs[base] if arm!='FULL_NATIVE' else None,
                    impact_values_digest=values[base] if arm!='FULL_NATIVE' else None,
                    real_reduction_calls=int(arm.endswith('_ON')),dispatch_calls=int(arm!='FULL_NATIVE'),fallback_calls=0)
            wall=dict(zip(ARMS,(20.,40.,30.,60.,60.-saving)))
            rows.append(dict(independent_run_id=run,case_id=case,length=length,requested_coverage='max',projection_layers=layers,
                condition_identity=c['condition_identity'],phase=phase,pair_index=i,benchmark_phase='latency',
                instrumentation='minimal',execution_order=list(ARMS),wall_ms=wall,differences=differences(wall),arms=copy.deepcopy(arms),
                full_relative_quality={a:{'allclose':a not in ARMS[3:],'tolerance_violations':int(a in ARMS[3:]),'correctness_status':'STRICT_FAIL' if a in ARMS[3:] else 'STRICT_PASS'} for a in ARMS},
                on_replay_parity={b:dict(exact_logits=True,policy_equal=True,coverage_and_rows_equal=True,
                    impact_values_equal=True,strict_metrics={'allclose':True},impact_replay_reduction_calls=0) for b in BASES},
                ablation_equivalence_pass=True,structural_valid=True,cache_integrity_valid=True,impact_replay_reduction_calls=0,
                correctness_qualified_speedup=False,endpoint_policy='all_token_logits_use_cache_false_attention_true',
                model=config['model'],dtype=config['dtype']))
    return manifest,rows


class AnalysisTests(unittest.TestCase):
    def test_negative_signs_strict_fail_and_no_service_qualification(self):
        manifests=[session(str(i),saving=-2.) for i in range(3)]
        result=aggregate(manifests,draws=100)[0]
        self.assertEqual(result['comparisons']['raw_on_minus_replay_ms']['equal_session_mean_ms'],-2.)
        self.assertEqual(result['comparisons']['no_reuse_on_minus_full_ms']['equal_session_mean_ms'],20.)
        self.assertFalse(result['full_relative_strict'][ARMS[3]])
        self.assertTrue(result['ablation_equivalence_pass'])
        self.assertFalse(result['correctness_qualified_speedup']); self.assertFalse(result['production_speedup_claimed'])

    def test_session_weights_bootstrap_reproducibility_outliers_not_deleted(self):
        sessions=[session('a',10),session('b',20),session('c',-5,200)]
        a=aggregate(sessions,seed=3,draws=100); b=aggregate(sessions,seed=3,draws=100)
        self.assertEqual(a,b)
        self.assertAlmostEqual(a[0]['comparisons']['raw_on_minus_replay_ms']['equal_session_mean_ms'],25/3)
        self.assertIn('process session means',a[0]['resampling_unit'])
        m,rows=session(); rows[-1]['wall_ms'][ARMS[4]]=500.; rows[-1]['differences']=differences(rows[-1]['wall_ms'])
        s=summarize(rows)[0]; self.assertEqual(s['count'],20);self.assertFalse(s['outliers_removed'])
        self.assertEqual(csv_summaries(rows)[0]['count'],20)

    def test_reject_incomplete_duplicate_incompatible_sessions_and_missing_evidence(self):
        for mutation in ('count','duplicate','process','order_seed','gpu','code','implementation','adapter','weights','dtype','phase',
                         'run_failed','gpu_unverified','pair_missing','pair_duplicate','arm_missing','window_rows',
                         'reduction','policy','reference','condition','qualified','strict_missing','gate','source_integrity'):
            with self.subTest(mutation=mutation):
                data=[session(str(i)) for i in range(3)]; m,rows=data[1]
                if mutation=='count': data.pop()
                if mutation=='duplicate': data[1]=data[0]
                if mutation=='process': m['process_instance']=data[0][0]['process_instance']
                if mutation=='order_seed': m['benchmark_config']['order_seed']=data[0][0]['benchmark_config']['order_seed']
                if mutation=='gpu': m['hardware_identity']['uuid']='GPU-other'
                if mutation=='code': m['code']['sha256']='old'
                if mutation=='implementation': m['implementation_sha256']={'runner':'other'}
                if mutation=='adapter': m['settings']['adapter']='user_b'
                if mutation=='weights': m['model_weight_identity']='other'
                if mutation=='dtype': m['benchmark_config']['dtype']='float32'
                if mutation=='phase': m['benchmark_config']['phase']='pilot'
                if mutation=='run_failed': m['execution_status']='COMPLETE_WITH_FAILURES'
                if mutation=='gpu_unverified': m['GPU_VALIDATION']='NOT_RUN'
                if mutation=='pair_missing': rows.pop()
                if mutation=='pair_duplicate': rows.append(copy.deepcopy(rows[-1]))
                if mutation=='arm_missing': rows[-1]['wall_ms'].pop(ARMS[4])
                if mutation=='window_rows': rows[-1]['arms'][ARMS[4]]['skipped_projection_rows']=0
                if mutation=='reduction': rows[-1]['arms'][ARMS[4]]['real_reduction_calls']=1
                if mutation=='policy': rows[-1]['arms'][ARMS[4]]['policy']={}
                if mutation=='reference': rows[-1]['arms'][ARMS[4]]['reference_digest']=None
                if mutation=='condition': rows[-1]['condition_identity']='changed'
                if mutation=='qualified': rows[-1]['correctness_qualified_speedup']=True
                if mutation=='strict_missing': rows[-1]['full_relative_quality']={}
                if mutation=='gate': rows[-1]['on_replay_parity'][BASES[1]]['exact_logits']=False
                if mutation=='source_integrity': m['conditions']['L512_covmax']['source_token_integrity_valid']=False
                with self.assertRaises((ValueError,KeyError)): aggregate(data,draws=100)

    def test_pilot_compatibility_and_uuid(self):
        m,_=session('pilot'); m['benchmark_config']['phase']='pilot'
        c=m['conditions']['L512_covmax']; c['status']='CAPACITY_VALIDATED'
        pilot=dict(experiment='R2-C3',execution_status='COMPLETE',environment_identity=m['environment_identity'],
            model_weights_verified=True,conditions=copy.deepcopy(m['conditions']))
        current=copy.deepcopy(m);current['benchmark_config']['phase']='latency'
        self.assertTrue(approve_pilot(m,pilot,current,'L512_covmax')['structural_valid'])
        for field in ('gpu','env','code','weights','gate','memory','phase'):
            bad=copy.deepcopy(m);pc=copy.deepcopy(pilot)
            if field=='gpu': bad['hardware_identity']['uuid']='GPU-other'
            if field=='env': bad['environment_identity']='other'
            if field=='code': bad['code']['sha256']='old'
            if field=='weights': bad['final_weights_verified']=False
            if field=='gate': bad['conditions']['L512_covmax']['ablation_equivalence_pass']=False; pc['conditions']=copy.deepcopy(bad['conditions'])
            if field=='memory': bad['conditions']['L512_covmax']['memory_after']={}; pc['conditions']=copy.deepcopy(bad['conditions'])
            if field=='phase': bad['benchmark_config']['phase']='latency'
            with self.assertRaises((ValueError,KeyError)): approve_pilot(bad,pc,current,'L512_covmax')

    def test_coverage_sensitivity_reports_measured_positions_only(self):
        r=aggregate([session(str(i)) for i in range(3)],draws=100)[0]
        low=copy.deepcopy(r);low.update(case_id='L512_cov0.0',actual_reused_tokens=0,actual_coverage=0.)
        high=copy.deepcopy(r)
        high['arm_equal_session_mean_ms'][ARMS[3]]+=8.
        change=coverage_changes([low,high])[0]
        self.assertEqual(change['high_minus_low_arm_wall_ms'][ARMS[3]],8.)
        self.assertEqual(change['high_actual_coverage'],510/512)


class RunnerTests(unittest.TestCase):
    def test_launcher_cli_contract_default_six_conditions(self):
        m=runner(); common=['--model','facebook/opt-2.7b','--dtype','float16','--prompt-lengths','512,1024',
            '--coverages','0,.5,max','--device','cuda:0','--cache-device','cuda','--capacity-mib','512','--seed','42',
            '--order-seed','4102','--independent-run-id','run2','--warmup-runs','3','--measured-runs','20','--output-dir','unused']
        args=m.parse_args(['--phase','latency','--pilot-inputs','fresh']+common)
        self.assertEqual([f'L{n}_cov{c}' for n in args.lengths for c in args.coverage_values],
            ['L512_cov0.0','L512_cov0.5','L512_covmax','L1024_cov0.0','L1024_cov0.5','L1024_covmax'])
        for argv in (['--phase','latency','--output-dir','x'],['--phase','analyze','--inputs','a','b','--output-dir','x']):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): m.parse_args(argv)

    def test_cpu_only_analyze_uses_input_provenance_never_loads_opt(self):
        m=runner()
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'Runtime',side_effect=AssertionError('GPU loader forbidden')):
            paths=[]
            for i in range(3):
                directory=Path(tmp)/str(i); directory.mkdir();manifest,rows=session(str(i))
                ref=directory/'impact_references.json';ref.write_text(json.dumps({'synthetic_test_fixture':True}))
                manifest['impact_reference_artifact_sha256']=hashlib.sha256(ref.read_bytes()).hexdigest()
                (directory/'manifest.json').write_text(json.dumps(manifest));(directory/'ablation_raw_pairs.json').write_text(json.dumps(rows))
                paths.append(str(directory))
            out=Path(tmp)/'aggregate'
            self.assertEqual(m.main(['--phase','analyze','--inputs',*paths,'--output-dir',str(out)]),0)
            saved=json.loads((out/'manifest.json').read_text())
            self.assertEqual(saved['GPU_VALIDATION'],'AGGREGATED_GPU_RUNS')
            self.assertFalse(saved['analysis_process']['gpu_inference_performed'])
            self.assertEqual(saved['aggregated_experimental_config']['benchmark_config']['dtype'],'float16')
            self.assertFalse(saved['correctness_qualified_speedup'])
            with self.assertRaises(FileExistsError): m.main(['--phase','analyze','--inputs',*paths,'--output-dir',str(out)])
            (Path(paths[0])/'impact_references.json').write_text('{"altered_fixture": true}')
            with redirect_stderr(io.StringIO()):
                self.assertEqual(m.main(['--phase','analyze','--inputs',*paths,'--output-dir',str(Path(tmp)/'bad_aggregate')]),1)
            failed=json.loads((Path(tmp)/'bad_aggregate/manifest.json').read_text())
            self.assertEqual(failed['execution_status'],'FAILED');self.assertIn('reference artifact',failed['failure']['failure'])

    def test_five_arm_order_complete_samples_and_raw_strict_failure_preserved(self):
        calls=[]; _,samples=session(); base=samples[-1]
        def invoke(arm):
            calls.append(arm); compact=base['arms'][arm]
            record=dict(compact,request_ms=base['wall_ms'][arm],wrapper_invoke_wall_ms=999.,
                coverage_audit={k:compact[k] for k in ('reused_tokens','actual_coverage','skipped_projection_rows','native_projection_rows')},
                projection_audit={'native':compact['native_projection_rows'],'skipped':compact['skipped_projection_rows']},
                selected_provenance=[],output_shape=[1,512,64],impact_real_reduction_calls=compact['real_reduction_calls'],
                impact_dispatch_calls=compact['dispatch_calls'],impact_fallback_calls=0)
            return (2 if arm in ARMS[3:] else 1),record
        runtime=NS(model=NS(modules=lambda:[]),length=512,guard=lambda:None,invoke_arm=invoke)
        a=NS(warmup_runs=3,measured_runs=20,order_seed=42,phase='latency')
        identity={k:base[k] for k in ('independent_run_id','case_id','length','requested_coverage','projection_layers','condition_identity','model','dtype','endpoint_policy')}
        with patch.dict('sys.modules',{'torch':NS(equal=lambda a,b:a==b)}),\
             patch('semcache.diagnostics.r2c3_benchmark.strict_metrics',side_effect=lambda a,b:{'allclose':a==b,'tolerance_violations':int(a!=b)}):
            one=paired(runtime,a,identity,lambda **kwargs:None)
            two=paired(runtime,a,identity,lambda **kwargs:None)
        self.assertEqual([r['execution_order'] for r in one],[r['execution_order'] for r in two])
        self.assertEqual(len(calls),2*23*5)
        measured=[r for r in one if r['phase']=='measured']
        self.assertTrue(all(sum(r['execution_order'][0]==arm for r in measured)==4 for arm in ARMS))
        self.assertEqual(summarize(one)[0]['count'],20)
        self.assertTrue(all(not r['full_relative_quality'][ARMS[3]]['allclose'] and r['ablation_equivalence_pass'] for r in one))

    def test_missing_reference_does_not_generate_one_inside_measured_request(self):
        runtime=Runtime.__new__(Runtime);runtime.references={}; runtime.binding=lambda base:{'base_mode':base}
        with patch('semcache.diagnostics.r2c1_benchmark.Runtime.invoke',side_effect=AssertionError('should not forward')):
            with self.assertRaises(MissingEvidence): runtime.invoke_arm(ARMS[4])

    def test_runner_pilot_latency_and_failure_artifacts(self):
        for failure in (None,'policy','OOM'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                m=runner(); pilot=Path(tmp)/'pilot'
                _,template=session(); sample=template[-1]
                class FakeRuntime:
                    def __init__(self,a):
                        self.a=a; self.provenance={'gpu':{'gpu_uuid':'GPU-same'}}; self.meta={'revision':'control_fixture'}
                        self.storage='cuda:0'; self.weights='weights'; self.namespace='namespace'
                        p=NS(element_size=lambda:2,numel=lambda:1)
                        self.model=NS(parameters=lambda:iter([p]),config=NS(max_position_embeddings=2048,
                            num_hidden_layers=2,num_attention_heads=2,hidden_size=16,vocab_size=64))
                        self.adapter=NS(layers=[0,1])
                        self.encoder=NS(metadata={'fixture':True},max_length=512,tokenizer=lambda text,**kw:{'input_ids':list(range(512))})
                    def setup_identity(self): return 'weights'
                    def prepare(self,n):
                        if failure=='OOM': raise RuntimeError('CUDA out of memory: fixture')
                        self.ids=list(range(n)); self.length=n; self.snapshot='source'; self.source_hashes={'q':'captured'}
                        self.catalog=NS(resident_bytes=1000); self.costs={}; self.preflight={}; self.text='fixture'
                    def guard(self,*args): pass
                    def model_guard(self,*args): pass
                def reference(r):
                    r.reference_digests={b:b+'ref' for b in BASES}
                    records={b:dict(impact_values_digest=b+'values',policy=sample['arms'][b+'_IMPACT_ON']['policy']) for b in BASES}
                    refs={b:dict(provenance={'mode':b,'snapshot':'source'},digest=r.reference_digests[b]) for b in BASES}
                    return refs,{b:None for b in BASES},records
                def fake_pairs(r,a,identity,save):
                    _,rows=session(a.independent_run_id,count=a.measured_runs)
                    rows=[dict(row,**identity,benchmark_phase=a.phase) for row in rows if row['phase']=='measured' or row['pair_index']<a.warmup_runs]
                    save(rows=rows)
                def replay(*args):
                    if failure=='policy':
                        error=MissingEvidence('policy provenance fixture failure');error.evidence={'expected':'a','actual':'b'};raise error
                    return {'passed':True}
                memory={'gpu':dict(free_bytes=18*1024**3,total_bytes=24*1024**3,allocated_bytes=5*1024**3,
                    reserved_bytes=6*1024**3,peak_allocated_bytes=9*1024**3)}
                coverage=dict(reused_tokens=510,actual_coverage=510/512,skipped_projection_rows=3060,native_projection_rows=12)
                recorder=Mock();recorder.finish.return_value={'status':'MEASURED'}
                patches=[patch.object(m,'Runtime',FakeRuntime),patch.object(m,'backend_audit',side_effect=lambda r: {'arm':r.arm,'effective':'eager'}),
                    patch.object(m,'memory',return_value=memory),patch.object(m,'monitor',return_value={'other_processes':[]}),
                    patch.object(m,'validate_physical',return_value={'audits':{'RAW_SEMCACHE_QKV':{'coverage':coverage}}}),
                    patch.object(m,'capture_references',side_effect=reference),patch.object(m,'validate_replay',side_effect=replay),
                    patch.object(m,'PilotPeakRecorder',return_value=recorder),patch.object(m,'paired',side_effect=fake_pairs)]
                from contextlib import ExitStack
                with ExitStack() as stack:
                    for item in patches: stack.enter_context(item)
                    stack.enter_context(redirect_stderr(io.StringIO()))
                    code=m.main(['--phase','pilot','--prompt-lengths','512','--coverages','max','--warmup-runs','1',
                        '--measured-runs','2','--output-dir',str(pilot)])
                    manifest=json.loads((pilot/'manifest.json').read_text())
                    pc=json.loads((pilot/'capacity_pilot.json').read_text())
                    case=manifest['conditions']['L512_covmax']
                    if failure:
                        self.assertEqual(code,1);self.assertEqual(manifest['execution_status'],'FAILED')
                        self.assertEqual(pc['execution_status'],'FAILED');self.assertIn('traceback',case)
                        self.assertEqual(case['length'],512);self.assertFalse(case['ablation_equivalence_pass'])
                        if failure=='policy': self.assertEqual(case['failed_gate_evidence']['actual'],'b')
                        else: self.assertEqual(case['status'],'OOM')
                    else:
                        self.assertEqual(code,0);self.assertEqual(pc['execution_status'],'COMPLETE')
                        self.assertEqual(case['status'],'CAPACITY_VALIDATED');self.assertTrue(case['ablation_equivalence_pass'])
                        self.assertEqual(case['impact_replay_reduction_calls'],0)
                        out=Path(tmp)/'latency'
                        code=m.main(['--phase','latency','--prompt-lengths','512','--coverages','max','--pilot-inputs',str(pilot),
                            '--warmup-runs','3','--measured-runs','20','--output-dir',str(out)])
                        self.assertEqual(code,0)
                        lm=json.loads((out/'manifest.json').read_text());lc=lm['conditions']['L512_covmax']
                        self.assertEqual(lm['execution_status'],'COMPLETE');self.assertEqual(lm['GPU_VALIDATION'],'EXECUTED')
                        self.assertEqual(lc['measured_pairs'],20);self.assertFalse(lc['full_relative_strict'][ARMS[3]])
                        self.assertTrue((out/'ablation_session_summary.csv').exists())
                        self.assertFalse(lm['correctness_qualified_speedup'])

    def test_model_loading_failure_keeps_failed_manifest_and_pilot(self):
        m=runner()
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'Runtime',side_effect=ImportError('Torch unavailable')),redirect_stderr(io.StringIO()):
            path=Path(tmp)/'run'
            self.assertEqual(m.main(['--phase','pilot','--output-dir',str(path)]),1)
            manifest=json.loads((path/'manifest.json').read_text())
            self.assertEqual(len(manifest['conditions']),6);self.assertEqual(manifest['execution_status'],'FAILED')
            self.assertEqual(manifest['GPU_VALIDATION'],'NOT_RUN')
            self.assertIn('Torch unavailable',manifest['failure']['traceback'])
            self.assertEqual(json.loads((path/'capacity_pilot.json').read_text())['execution_status'],'FAILED')


if __name__=='__main__': unittest.main()
