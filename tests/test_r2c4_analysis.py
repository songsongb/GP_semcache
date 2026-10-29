"""Synthetic independent-session, negative-control and CPU-only CLI contracts."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from contextlib import redirect_stderr
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch,Mock
from test_r2c3_analysis import session as c3session
from semcache.diagnostics.r2c4_cache_path import ARMS,RAW,SNAPSHOT,MAPPING
from semcache.diagnostics.r2c4_analysis import aggregate,summarize,approve_pilot,differences
from semcache.diagnostics.r2c4_benchmark import paired

ROOT=Path(__file__).resolve().parents[1]


def runner():
    spec=importlib.util.spec_from_file_location('c4_runner_test',ROOT/'scripts/93_run_r2c4_cache_path.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def session(run='a',coverage='max',saving=18.,count=20):
    m,old=c3session(run,count=count);m['experiment']='R2-C4';m['production_speedup_claimed']=False
    n=1024;reused=0 if coverage==0 else 510 if coverage==.5 else 1023
    key=f'L{n}_cov{coverage}'
    c=m['conditions'].pop('L512_covmax');c.update(length=n,requested_coverage=coverage,actual_reused_tokens=reused,actual_coverage=reused/n)
    c['full_relative_strict']={a:not(reused and a in RAW) for a in ARMS}
    m['conditions']={key:c};m['benchmark_config'].update(lengths=[n],coverage_values=[coverage])
    rows=[]
    for row in old:
        arms={}
        for a in ARMS:
            original=copy.deepcopy(row['arms']['FULL_NATIVE' if a==ARMS[0] else 'SEMCACHE_NO_REUSE_IMPACT_REPLAY' if a==ARMS[1] else 'RAW_SEMCACHE_QKV_IMPACT_REPLAY'])
            r=reused if a in RAW else 0;ss=bool(r and a in SNAPSHOT);ms=bool(r and a in MAPPING)
            original.update(reused_tokens=r,actual_coverage=r/n,native_projection_rows=(n-r)*6,skipped_projection_rows=r*6,
                qkv_digest='same-executed-QKV' if a in RAW else None)
            calls={k:v for k,v in [('request_mapping_select',int(a!='FULL_NATIVE')),('executor_setup',int(a!='FULL_NATIVE')),
                ('executor_cleanup',int(a!='FULL_NATIVE')),('source_snapshot_validate',int(bool(r))-int(ss)),
                ('source_snapshot_oracle_dispatch',int(ss)),('prepared_mapping_get',int(bool(r))-int(ms)),('prepared_mapping_oracle_dispatch',int(ms))] if v}
            original['cache_path']=dict(timing_boundary_verified=True,snapshot_bypassed=ss,mapping_bypassed=ms,
                snapshot_intervention='APPLICABLE' if r else 'NOT_APPLICABLE',mapping_intervention='APPLICABLE' if r else 'NOT_APPLICABLE',
                calls={'request_internal':calls,'request_external':{}},mapping_lookup_kind='PREBOUND_NOT_A_CACHE_LOOKUP' if ms else 'ORIGINAL_CACHE_LOOKUP' if r else 'NOT_APPLICABLE',
                mapping_cache_hit=None if ms or not r else True,prebind_cache_hit=True if ms else None,excluded_costs_ms={})
            arms[a]=original
        wall=dict(zip(ARMS,(20.,30.,60.,50.,55.,60.-saving)))
        rows.append(dict(row,case_id=key,length=n,requested_coverage=coverage,arms=arms,wall_ms=wall,differences=differences(wall),execution_order=list(ARMS),
            raw_equivalence={a:dict(exact_logits=True,exact_qkv=True,policy_equal=True,impact_values_equal=True,source_and_rows_equal=True,
                qkv_digest='same-executed-QKV',strict_metrics={'allclose':True},impact_replay_reduction_calls=0) for a in RAW},
            full_relative_quality={a:dict(allclose=not(reused and a in RAW),tolerance_violations=int(bool(reused and a in RAW))) for a in ARMS}))
    return m,rows


def diagnostic(source):
    m=copy.deepcopy(source);m['benchmark_config'].update(phase='breakdown',measured_runs=1,warmup_runs=1)
    result=[]
    for key,c in m['conditions'].items():
        c.update(status='DIAGNOSED',timing_boundary_verified=True)
        for arm in ARMS:
            result.append(dict(case_id=key,arm=arm,repeat=0,instrumentation='breakdown',timing_boundary_verified=True,
                components=[],instrumented_request_wall_ms=999.))
    return m,result


class AnalysisTests(unittest.TestCase):
    def test_interaction_signed_values_and_strict_fail_not_qualified(self):
        sessions=[session(str(i)) for i in range(3)]
        result=aggregate(sessions,diagnostic(sessions[0][0]),draws=100)[0]
        self.assertEqual(result['comparisons']['interaction_ms']['observed_equal_session_mean_ms'],3.)
        self.assertEqual(result['comparisons']['baseline_minus_full_ms']['observed_equal_session_mean_ms'],40.)
        self.assertFalse(result['full_relative_strict'][ARMS[2]]);self.assertFalse(result['correctness_qualified_speedup'])
        negative=[session(str(i),saving=-2) for i in range(3)]
        self.assertEqual(aggregate(negative,diagnostic(negative[0][0]),draws=100)[0]['comparisons']['baseline_minus_both_ms']['attributable_difference_ms'],-2.)

    def test_zero_reuse_noise_is_not_attributed_to_removed_work(self):
        data=[session(str(i),coverage=0.) for i in range(3)]
        r=aggregate(data,diagnostic(data[0][0]),draws=100)[0]
        for k in ('baseline_minus_snapshot_ms','baseline_minus_mapping_ms','baseline_minus_both_ms','interaction_ms'):
            self.assertEqual(r['comparisons'][k]['status'],'NOT_APPLICABLE')
            self.assertIsNone(r['comparisons'][k]['attributable_difference_ms'])
        self.assertEqual(r['comparisons']['baseline_minus_both_ms']['observed_equal_session_mean_ms'],18.)

    def test_equal_session_weights_outliers_and_seeded_bootstrap(self):
        data=[session('a',saving=10),session('b',saving=20),session('c',saving=-5,count=200)]
        a=aggregate(data,diagnostic(data[0][0]),seed=7,draws=100)
        self.assertEqual(a,aggregate(data,diagnostic(data[0][0]),seed=7,draws=100))
        self.assertAlmostEqual(a[0]['comparisons']['baseline_minus_both_ms']['observed_equal_session_mean_ms'],25/3)
        self.assertFalse(a[0]['outliers_removed'])
        m,rows=session();rows[-1]['wall_ms'][ARMS[5]]=500.;rows[-1]['differences']=differences(rows[-1]['wall_ms'])
        self.assertEqual(summarize(rows)[0]['count'],20)

    def test_reject_incomplete_duplicate_gpu_code_or_diagnostic_mismatch(self):
        for mutation in ('missing','duplicate','pid','seed','gpu','code','weights','phase','bad_gate','counter',
                         'qkv','policy','missing_row','duplicate_row','profile','bad_diag','missing_diag','diag_gpu','diag_duplicates'):
            with self.subTest(mutation=mutation):
                data=[session(str(i)) for i in range(3)];bd=diagnostic(data[0][0]);m,rows=data[1]
                if mutation=='missing':data.pop()
                if mutation=='duplicate':data[1]=data[0]
                if mutation=='pid':m['process_instance']=data[0][0]['process_instance']
                if mutation=='seed':m['benchmark_config']['order_seed']=data[0][0]['benchmark_config']['order_seed']
                if mutation=='gpu':m['hardware_identity']['uuid']='GPU-other'
                if mutation=='code':m['code']['sha256']='other'
                if mutation=='weights':m['model_weight_identity']='other'
                if mutation=='phase':m['benchmark_config']['phase']='pilot'
                if mutation=='bad_gate':m['conditions']['L1024_covmax']['ablation_equivalence_pass']=False
                if mutation=='counter':rows[-1]['arms'][ARMS[3]]['cache_path']['calls']['request_internal']['source_snapshot_validate']=1
                if mutation=='qkv':rows[-1]['arms'][ARMS[4]]['qkv_digest']='corrupt'
                if mutation=='policy':rows[-1]['arms'][ARMS[5]]['policy']={}
                if mutation=='missing_row':rows.pop()
                if mutation=='duplicate_row':rows.append(rows[-1])
                if mutation=='profile':rows[-1]['instrumentation']='breakdown'
                if mutation=='bad_diag':bd[0]['conditions']['L1024_covmax']['timing_boundary_verified']=False
                if mutation=='missing_diag':bd[1].pop()
                if mutation=='diag_gpu':bd[0]['hardware_identity']['uuid']='GPU-other'
                if mutation=='diag_duplicates':bd[1][-1]=copy.deepcopy(bd[1][0])
                with self.assertRaises((ValueError,KeyError)):aggregate(data,bd,draws=100)

    def test_pilot_fail_closed_and_compatibility(self):
        pm,_=session('p');pm['benchmark_config']['phase']='pilot';pm['conditions']['L1024_covmax']['status']='CAPACITY_VALIDATED'
        pc=dict(experiment='R2-C4',execution_status='COMPLETE',environment_identity=pm['environment_identity'],model_weights_verified=True,conditions=copy.deepcopy(pm['conditions']))
        current=copy.deepcopy(pm);current['benchmark_config']['phase']='latency';current['execution_status']='RUNNING'
        self.assertTrue(approve_pilot(pm,pc,current,'L1024_covmax')['memory_valid'])
        for field in ('gpu','code','weights','incomplete','gate','memory'):
            bad=copy.deepcopy(pm);cap=copy.deepcopy(pc)
            if field=='gpu':bad['hardware_identity']['uuid']='GPU-other'
            if field=='code':bad['code']['sha256']='old'
            if field=='weights':bad['final_weights_verified']=False
            if field=='incomplete':bad['execution_status']='COMPLETE_WITH_FAILURES'
            if field=='gate':bad['conditions']['L1024_covmax']['ablation_equivalence_pass']=False;cap['conditions']=bad['conditions']
            if field=='memory':bad['conditions']['L1024_covmax']['memory_after']={};cap['conditions']=bad['conditions']
            with self.assertRaises(ValueError):approve_pilot(bad,cap,current,'L1024_covmax')


class RunnerTests(unittest.TestCase):
    def test_cli_contract_all_phases(self):
        m=runner();common=['--model','facebook/opt-2.7b','--dtype','float16','--device','cuda:0','--cache-device','cuda',
            '--capacity-mib','512','--seed','42','--order-seed','4102','--independent-run-id','run2','--output-dir','unused']
        a=m.parse_args(['--phase','pilot','--prompt-lengths','512,1024','--coverages','0,.5,max','--warmup-runs','1','--measured-runs','3']+common)
        self.assertEqual(len(a.lengths)*len(a.coverage_values),6)
        a=m.parse_args(['--phase','breakdown','--pilot-inputs','pilot','--prompt-lengths','1024','--coverages','0,.5,max']+common)
        self.assertEqual(a.lengths,[1024])
        m.parse_args(['--phase','latency','--pilot-inputs','pilot','--warmup-runs','3','--measured-runs','20']+common)
        m.parse_args(['--phase','analyze','--inputs','a','b','c','--breakdown-input','diag','--output-dir','x'])
        for args in (['--phase','latency'],['--phase','breakdown','--prompt-lengths','512','--pilot-inputs','p'],['--phase','analyze','--inputs','a','b','c']):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):m.parse_args(args+['--output-dir','x'])

    def test_analyze_cpu_only_input_identity_and_failed_startup(self):
        m=runner()
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'Runtime',side_effect=AssertionError('OPT loading forbidden')):
            paths=[]
            for i in range(3):
                p=Path(tmp)/str(i);p.mkdir();manifest,rows=session(str(i));ref=p/'impact_references.json';ref.write_text('{}')
                manifest['impact_reference_artifact_sha256']=hashlib.sha256(ref.read_bytes()).hexdigest()
                (p/'manifest.json').write_text(json.dumps(manifest));(p/'cache_path_raw_pairs.json').write_text(json.dumps(rows));paths.append(str(p))
            d=Path(tmp)/'diag';d.mkdir();dm,dr=diagnostic(session('d')[0]);ref=d/'impact_references.json';ref.write_text('{}')
            dm['impact_reference_artifact_sha256']=hashlib.sha256(ref.read_bytes()).hexdigest()
            (d/'manifest.json').write_text(json.dumps(dm));(d/'cache_path_breakdown.json').write_text(json.dumps(dr))
            out=Path(tmp)/'analysis'
            self.assertEqual(m.main(['--phase','analyze','--inputs',*paths,'--breakdown-input',str(d),'--output-dir',str(out)]),0)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['GPU_VALIDATION'],'AGGREGATED_GPU_RUNS');self.assertFalse(manifest['analysis_process']['gpu_inference_performed'])
            self.assertEqual(manifest['aggregated_experimental_config']['dtype'],'float16')
            with self.assertRaises(FileExistsError):m.main(['--phase','analyze','--inputs',*paths,'--breakdown-input',str(d),'--output-dir',str(out)])
            failed=Path(tmp)/'failed'
            with redirect_stderr(io.StringIO()):self.assertEqual(m.main(['--phase','pilot','--output-dir',str(failed)]),1)
            f=json.loads((failed/'manifest.json').read_text());self.assertEqual(f['execution_status'],'FAILED');self.assertEqual(f['GPU_VALIDATION'],'NOT_RUN')
            self.assertEqual(len(f['conditions']),6);self.assertIn('traceback',f['failure'])

    def test_balanced_six_arm_schedule_and_complete_counts(self):
        from semcache.diagnostics.r1b3_break_even import schedule
        a=schedule(ARMS,20,4);self.assertEqual(a,schedule(ARMS,20,4))
        for position in range(6):
            counts=[sum(order[position]==arm for order in a) for arm in ARMS]
            self.assertLessEqual(max(counts)-min(counts),1)
        self.assertTrue(all(len(set(order))==6 for order in a))

    def test_gpu_runner_orchestration_artifacts_with_scalar_fixture(self):
        m=runner();h=m.helpers()
        base_m,base_rows=session('template');policy=base_m['conditions']['L1024_covmax']['policy_proofs']
        class Runtime:
            def __init__(self,a):
                self.a=a;self.provenance={'gpu':{'gpu_uuid':'GPU-same'}};self.meta={'revision':'SCALAR_FIXTURE_ONLY'}
                self.storage='cuda:0';self.weights='weights';self.namespace='namespace'
                parameter=NS(element_size=lambda:2,numel=lambda:1)
                self.model=NS(parameters=lambda:iter([parameter]),config=NS(max_position_embeddings=2048,num_hidden_layers=2,num_attention_heads=2,hidden_size=16,vocab_size=64))
                self.adapter=NS(layers=[0,1]);self.encoder=NS(metadata={'fixture':True})
            def setup_identity(self):return 'weights'
            def prepare(self,n):
                self.ids=list(range(n));self.snapshot='source';self.source_hashes={'q':'hash'};self.length=n
                self.catalog=NS(resident_bytes=1024);self.costs={};self.preflight={}
            def guard(self,*a):pass
            def model_guard(self,*a):pass
        def physical(r):
            reused=0 if r.coverage==0 else 510 if r.coverage==.5 else 1023
            return {'audits':{'RAW_SEMCACHE_QKV':{'coverage':{'reused_tokens':reused,'actual_coverage':reused/1024}}}}
        def refs(r):
            r.reference_digests=base_m['conditions']['L1024_covmax']['reference_digests']
            records={b:dict(policy=policy[b],impact_values_digest=b+'values') for b in policy}
            return {b:{'digest':r.reference_digests[b]} for b in policy},{},records
        def pairs(r,a,identity,save):
            _,rows=session(a.independent_run_id,r.coverage,count=a.measured_runs)
            save(rows=[dict(row,**identity,benchmark_phase=a.phase) for row in rows if row['phase']=='measured' or row['pair_index']<a.warmup_runs])
        def breakdown(r,a,identity,save):
            data=[]
            for repeat in range(a.measured_runs):
                for arm in ARMS:data.append(dict(identity,arm=arm,repeat=repeat,instrumentation='breakdown',timing_boundary_verified=True,
                    components=[{'component':'fixture_boundary','host_ms':1.,'cuda_ms':None}],full_relative_quality={'allclose':False}))
            save(breakdown=data)
        mem={'gpu':dict(free_bytes=18*1024**3,reserved_bytes=6*1024**3,allocated_bytes=5*1024**3,peak_allocated_bytes=9*1024**3)}
        rec=Mock();rec.finish.return_value={'status':'MEASURED'}
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'helpers',return_value=h),patch.object(m,'Runtime',Runtime),\
             patch.object(h,'backend_audit',side_effect=lambda r:{'arm':r.arm}),patch.object(h,'monitor',return_value={'status':'GPU_MONITOR_VERIFIED','other_processes':[]}),\
             patch.object(m,'memory',return_value=mem),patch.object(m,'physical_gate',side_effect=physical),\
             patch.object(m,'capture_references',side_effect=refs),patch.object(m,'validate_replay',return_value={'fixture_passed':True}),\
             patch.object(m,'gate',return_value={'fixture_passed':True}),patch.object(m,'paired',side_effect=pairs),\
             patch.object(m,'collect_breakdown',side_effect=breakdown),patch.object(m,'PilotPeakRecorder',return_value=rec):
            pilot=Path(tmp)/'pilot'
            self.assertEqual(m.main(['--phase','pilot','--prompt-lengths','1024','--coverages','0,.5,max','--warmup-runs','1','--measured-runs','2','--output-dir',str(pilot)]),0)
            p=json.loads((pilot/'capacity_pilot.json').read_text());self.assertEqual(p['experiment'],'R2-C4');self.assertEqual(p['execution_status'],'COMPLETE')
            latency=Path(tmp)/'latency'
            self.assertEqual(m.main(['--phase','latency','--prompt-lengths','1024','--coverages','0,.5,max','--pilot-inputs',str(pilot),
                '--independent-run-id','run1','--measured-runs','20','--output-dir',str(latency)]),0)
            lm=json.loads((latency/'manifest.json').read_text());self.assertEqual(lm['GPU_VALIDATION'],'EXECUTED')
            self.assertTrue(all(c['measured_pairs']==20 and c['status']=='VALID' for c in lm['conditions'].values()))
            diag=Path(tmp)/'breakdown'
            self.assertEqual(m.main(['--phase','breakdown','--pilot-inputs',str(pilot),'--warmup-runs','1','--measured-runs','1','--output-dir',str(diag)]),0)
            dm=json.loads((diag/'manifest.json').read_text())
            self.assertTrue(all(c['status']=='DIAGNOSED' and c['timing_boundary_verified'] for c in dm['conditions'].values()))
            self.assertTrue((diag/'cache_path_breakdown.csv').exists());self.assertFalse((diag/'cache_path_raw_pairs.json').exists())
            self.assertFalse(dm['correctness_qualified_speedup']);self.assertFalse(dm['production_speedup_claimed'])


if __name__=='__main__':unittest.main()
