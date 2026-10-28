"""Synthetic capacity/control-flow tests; none represents a GPU experiment."""
import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch,Mock

from semcache.diagnostics.r2c2_profile_memory import (GIB,PILOT_MEMORY_SCHEMA,
    PilotPeakRecorder,ProfileMemoryRejected,pilot_evidence,preflight,failure_status,validation_report)
from semcache.metrics.r1 import digest
from test_r2c1_benchmark import runner

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    measured=dict(schema=PILOT_MEMORY_SCHEMA,status='MEASURED',baseline_allocated_bytes=5*GIB,
        peak_allocated_bytes=9*GIB,allocation_increase_bytes=4*GIB,total_bytes=24*GIB,samples=10)
    condition=dict(length=1024,requested_coverage='max',status='CAPACITY_VALIDATED',
        structural_valid=True,condition_identity='source-tokens-adapter',strict_pass=False,pilot_memory=measured)
    manifest=dict(experiment='R2-C1',execution_status='COMPLETE',environment_identity='same-env',
        final_weights_verified=True,GPU_VALIDATION='EXECUTED',hardware_identity={'uuid':'GPU-actual-3090'},
        code={'sha256':'code-sha'},implementation_sha256={'runner.py':'sha'},model_weight_identity='weights',
        settings={'adapter':'user_a','atol':1e-3,'rtol':1e-3,'output_attentions':True},
        benchmark_config=dict(phase='pilot',model='facebook/opt-2.7b',dtype='float16',capacity_mib=512,cache_device='cuda',seed=42),
        conditions={'L1024_covmax':condition})
    pilot=dict(experiment='R2-C1',execution_status='COMPLETE',environment_identity='same-env',
        model_weights_verified=True,conditions=copy.deepcopy(manifest['conditions']))
    current=copy.deepcopy(manifest); current['benchmark_config']['phase']='profile'
    gpu=dict(allocated_bytes=5*GIB,reserved_bytes=6*GIB,free_bytes=18*GIB,total_bytes=24*GIB)
    return manifest,pilot,current,gpu


def write_pilot(directory,manifest,pilot):
    for filename,value in [('manifest.json',manifest),('capacity_pilot.json',pilot)]:
        (directory/filename).write_text(json.dumps(value))


class PolicyTests(unittest.TestCase):
    def evidence(self):
        m,p,c,gpu=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            write_pilot(Path(tmp),m,p)
            return pilot_evidence(tmp,c,1024,'max'),gpu

    def test_guarded_approves_where_conservative_rejects_and_keeps_both(self):
        evidence,gpu=self.evidence()
        conservative={'estimated_additional_bytes':20*GIB}
        original=copy.deepcopy(conservative)
        baseline=preflight('conservative',conservative,gpu)
        guarded=preflight('pilot_guarded',conservative,gpu,evidence)
        self.assertFalse(baseline['approved']); self.assertTrue(guarded['approved'])
        self.assertEqual(guarded['estimated_additional_required_bytes'],7*GIB)
        self.assertEqual(guarded['guarded']['profiler_allowance_bytes'],GIB)
        self.assertEqual(guarded['guarded']['safety_reserve_bytes'],2*GIB)
        self.assertEqual(guarded['available_with_allocator_bytes'],19*GIB)
        self.assertEqual(guarded['conservative'],original)
        self.assertEqual(conservative,original)
        self.assertEqual(len(evidence['files_sha256']),2)

    def test_guarded_insufficient_memory_and_custom_margins(self):
        evidence,gpu=self.evidence(); gpu['free_bytes']=5*GIB
        a=preflight('pilot_guarded',{'estimated_additional_bytes':20*GIB},gpu,evidence)
        self.assertFalse(a['approved']); self.assertEqual(a['reason'],'INSUFFICIENT_GPU_HEADROOM')
        gpu['free_bytes']=18*GIB
        a=preflight('pilot_guarded',{'estimated_additional_bytes':20*GIB},gpu,evidence,2*GIB,3*GIB)
        self.assertEqual(a['estimated_additional_required_bytes'],9*GIB)

    def test_missing_bad_current_or_pilot_measurements_fail_closed(self):
        evidence,gpu=self.evidence()
        for missing in ('allocated_bytes','reserved_bytes','free_bytes','total_bytes'):
            bad=dict(gpu); del bad[missing]
            self.assertFalse(preflight('pilot_guarded',{'estimated_additional_bytes':1},bad,evidence)['approved'])
        for mutation in ('missing','negative','inconsistent','zero','total'):
            bad=copy.deepcopy(evidence)
            if mutation=='missing': bad['memory'].pop('baseline_allocated_bytes')
            if mutation=='negative': bad['memory']['peak_allocated_bytes']=-1
            if mutation=='inconsistent': bad['memory']['allocation_increase_bytes']=1
            if mutation=='zero': bad['memory']['baseline_allocated_bytes']=0
            if mutation=='total': bad['memory']['total_bytes']=48*GIB
            self.assertFalse(preflight('pilot_guarded',{'estimated_additional_bytes':1},gpu,bad)['approved'])
        a=preflight('pilot_guarded',{'estimated_additional_bytes':1},gpu,None,evidence_error='old Pilot missing evidence')
        self.assertFalse(a['approved']); self.assertIn('old Pilot',a['reason'])

    def test_default_conservative_uses_unchanged_estimate(self):
        _,_,_,gpu=fixture()
        for required in (19*GIB-1,19*GIB,19*GIB+1):
            a=preflight('conservative',{'estimated_additional_bytes':required},gpu)
            self.assertEqual(a['approved'],required<=gpu['free_bytes']+gpu['reserved_bytes']-gpu['allocated_bytes'])
            self.assertIsNone(a['guarded'])

    def test_pilot_compatibility_and_approval_rejections(self):
        for mutation in ('gpu','missing_gpu','model','dtype','adapter','weights','code','implementation','capacity',
                         'length','coverage','unapproved','structural','incomplete','not_pilot','missing_memory',
                         'old_schema','bad_peak','bad_increase','no_gpu_run','unverified_weights','artifact_mismatch'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                m,p,c,_=fixture()
                if mutation=='gpu': m['hardware_identity']['uuid']='GPU-other'
                elif mutation=='missing_gpu': m['hardware_identity']['uuid']=None
                elif mutation in ('model','dtype','capacity'): m['benchmark_config'][{'capacity':'capacity_mib'}.get(mutation,mutation)]='different'
                elif mutation=='adapter': m['settings']['adapter']='user_b'
                elif mutation=='weights': m['model_weight_identity']='other'
                elif mutation=='code': m['code']['sha256']='old-code'
                elif mutation=='implementation': m['implementation_sha256']={'runner.py':'old'}
                elif mutation=='incomplete': m['execution_status']='COMPLETE_WITH_FAILURES'
                elif mutation=='not_pilot': m['benchmark_config']['phase']='latency'
                elif mutation=='no_gpu_run': m['GPU_VALIDATION']='NOT_RUN'
                elif mutation=='unverified_weights': p['model_weights_verified']=False
                else:
                    cond=m['conditions']['L1024_covmax']
                    if mutation=='length': cond['length']=512
                    if mutation=='coverage': cond['requested_coverage']=.5
                    if mutation=='unapproved': cond['status']='OOM'
                    if mutation=='structural': cond['structural_valid']=False
                    if mutation=='missing_memory': cond.pop('pilot_memory')
                    if mutation=='old_schema': cond['pilot_memory']['schema']='historical-estimate'
                    if mutation=='bad_peak': cond['pilot_memory']['peak_allocated_bytes']=4*GIB
                    if mutation=='bad_increase': cond['pilot_memory']['allocation_increase_bytes']=None
                    if mutation!='artifact_mismatch': p['conditions']=copy.deepcopy(m['conditions'])
                    else: p['conditions']['L1024_covmax']['condition_identity']='other'
                write_pilot(Path(tmp),m,p)
                with self.assertRaises((ValueError,TypeError)):
                    pilot_evidence(tmp,c,1024,'max')

    def test_strict_fail_does_not_reject_valid_capacity_pilot(self):
        evidence,_=self.evidence()
        self.assertEqual(evidence['case_id'],'L1024_covmax')

    def test_peak_recorder_keeps_source_peak_across_request_resets(self):
        state=dict(peak=12*GIB,allocated=6*GIB,resets=0)
        def reset(device=None): state.update(peak=state['allocated'],resets=state['resets']+1)
        cuda=NS(reset_peak_memory_stats=reset,max_memory_allocated=lambda d:state['peak'],
            memory_allocated=lambda d:state['allocated'],current_device=lambda:0)
        class Device:
            def __init__(self,v): self.index=int(v.split(':')[1]) if ':' in v else None
            def __eq__(self,v): return self.index==v.index
        with patch.dict('sys.modules',{'torch':NS(cuda=cuda,device=Device)}):
            recorder=PilotPeakRecorder('cuda:0')
            recorder.begin({'gpu':dict(allocated_bytes=5*GIB,total_bytes=24*GIB)})
            cuda.reset_peak_memory_stats('cuda:0')
            state.update(peak=8*GIB)
            cuda.reset_peak_memory_stats('cuda:0')
            measured=recorder.finish()
            self.assertEqual(measured['peak_allocated_bytes'],12*GIB)
            self.assertEqual(measured['allocation_increase_bytes'],7*GIB)
            self.assertGreaterEqual(measured['samples'],4)
            self.assertIs(cuda.reset_peak_memory_stats,reset)
            recorder.begin({'gpu':dict(allocated_bytes=5*GIB,total_bytes=24*GIB)},measured)
            self.assertEqual(recorder.finish()['peak_allocated_bytes'],12*GIB)

    def test_preflight_real_torch_oom_and_trace_classifications(self):
        class OOM(MemoryError): pass
        with patch.dict('sys.modules',{'torch':NS(cuda=NS(OutOfMemoryError=OOM))}):
            self.assertEqual(failure_status(OOM('allocator exhausted')),'OOM')
            self.assertEqual(failure_status(MemoryError('screen only')),'MEMORY_PREFLIGHT_FAILED')
            self.assertEqual(failure_status(RuntimeError('CUDA out of memory')),'OOM')
            self.assertEqual(failure_status(RuntimeError('TRACE_BUDGET_EXCEEDED: preserved')),'TRACE_FAILED')
            self.assertEqual(failure_status(RuntimeError('parity')),'INVALID_EXECUTION')

    def test_cli_profile_only_and_conservative_default(self):
        r=runner()
        base=['--output-dir','unused']
        a=r.parse_args(['--phase','pilot']+base)
        self.assertEqual(a.profile_memory_policy,'conservative')
        self.assertEqual(a.profile_memory_allowance_gib,1.)
        for args in (['--phase','latency','--profile-memory-policy','pilot_guarded'],
                     ['--phase','pilot','--profile-memory-policy','pilot_guarded'],
                     ['--phase','analyze','--profile-memory-safety-gib','1'],
                     ['--phase','profile','--profile-memory-safety-gib','nan'],
                     ['--phase','profile','--profile-memory-allowance-gib','0']):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): r.parse_args(list(args)+base)
        a=r.parse_args(['--phase','profile','--prompt-lengths','1024','--coverages','0,max',
            '--pilot-inputs','fresh','--profile-memory-policy','pilot_guarded']+base)
        self.assertEqual(a.lengths,[1024]); self.assertEqual(a.coverage_values,[0.,'max'])

    def test_validation_prints_original_case_memory_not_bare_assertion(self):
        m,_,_,_=fixture()
        m['execution_status']='COMPLETE_WITH_FAILURES'
        m['conditions']['L1024_covmax'].update(status='MEMORY_PREFLIGHT_FAILED',failure='INSUFFICIENT_GPU_HEADROOM',
            memory_preflight_audit={'policy':'pilot_guarded','estimated_additional_required_bytes':7*GIB},
            traceback='original traceback')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); (path/'manifest.json').write_text(json.dumps(m))
            report=validation_report(path)
            self.assertFalse(report['valid']); failed=report['failed_cases']['L1024_covmax']
            self.assertEqual(failed['failure'],'INSUFFICIENT_GPU_HEADROOM')
            self.assertEqual(failed['traceback'],'original traceback')
            self.assertEqual(failed['memory_preflight_audit']['estimated_additional_required_bytes'],7*GIB)

    def test_validation_rejects_missing_profiler_and_hash_or_gpu_mismatch(self):
        m,_,_,_=fixture(); m['benchmark_config']['phase']='profile'
        m['conditions']['L1024_covmax']['status']='VALID'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); (path/'manifest.json').write_text(json.dumps(m))
            self.assertFalse(validation_report(path)['valid'])
            from semcache.diagnostics.r2c1_workloads import MODES
            (path/'profiling_status.json').write_text(json.dumps({'L1024_covmax_'+mode:{'status':'PROFILE_COLLECTED'} for mode in MODES}))
            self.assertTrue(validation_report(path,'code-sha','actual-3090')['valid'])
            self.assertFalse(validation_report(path,'old','other')['valid'])

    def test_validator_cli_prints_memory_even_after_code_change(self):
        spec=importlib.util.spec_from_file_location('c2_validator',ROOT/'scripts/91_validate_r2c2_results.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp); m,_,_,_=fixture()
            m['execution_status']='COMPLETE_WITH_FAILURES'
            m['conditions']['L1024_covmax'].update(status='OOM',failure='CUDA out of memory: original',
                memory_preflight_audit={'guarded':{'estimated_additional_bytes':7*GIB}})
            (directory/'manifest.json').write_text(json.dumps(m))
            identity=directory/'identity.json'
            identity.write_text(json.dumps({'code':{'sha256':'changed'}}))
            stdout=io.StringIO(); stderr=io.StringIO()
            with redirect_stdout(stdout),redirect_stderr(stderr):
                code=module.main(['validate','--identity',str(identity),'--directory',str(directory)])
            self.assertEqual(code,1)
            self.assertIn('code changed',stderr.getvalue())
            self.assertIn('CUDA out of memory: original',stdout.getvalue())
            self.assertIn('memory_preflight_audit',stdout.getvalue())


class RunnerTests(unittest.TestCase):
    def test_guarded_profile_flow_and_failure_artifacts(self):
        for failure in (None,'preflight','missing_file','OOM','TRACE','pilot'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                r=runner(); output=Path(tmp)/'profile'; pilot=Path(tmp)/'pilot'; pilot.mkdir()
                (pilot/'capacity_pilot.json').write_text('{}')
                if failure=='missing_file': (pilot/'capacity_pilot.json').unlink()
                gpu=fixture()[3]
                class Runtime:
                    def __init__(self,a):
                        self.provenance={'gpu':{'gpu_uuid':'GPU-actual-3090'}}; self.meta={'revision':'fixture'}
                        self.storage='cuda:0'; self.weights='weights'; self.namespace='namespace'; self.a=a
                        parameter=NS(element_size=lambda:2,numel=lambda:1)
                        self.model=NS(parameters=lambda:iter([parameter]),config=NS(max_position_embeddings=2048,
                            num_hidden_layers=32,num_attention_heads=32,hidden_size=2560,vocab_size=50272))
                        self.encoder=NS(metadata={},model=NS(parameters=lambda:iter([parameter])),max_length=512,
                            tokenizer=lambda text,**kw:{'input_ids':list(range(1024))})
                    def setup_identity(self): return 'weights'
                    def prepare(self,n):
                        self.ids=list(range(n)); self.length=n; self.snapshot='s'; self.source_hashes={'q':'hash'}
                        self.catalog=NS(resident_bytes=480*1024**2); self.costs={}; self.preflight={}; self.text='fixture'
                        if failure=='OOM': raise RuntimeError('CUDA out of memory: test actual allocator failure')
                    def invoke(self,mode): return None,{}
                    def guard(self,*args): pass
                    def model_guard(self,*args): pass
                condition_id=digest(dict(ids=list(range(1024)),source_hashes={'q':'hash'},weights='weights',
                    namespace='namespace',coverage='max',storage='cuda:0'))
                evidence=dict(condition_identity=condition_id,memory=fixture()[1]['conditions']['L1024_covmax']['pilot_memory'])
                def collect(*args):
                    if failure=='TRACE': raise RuntimeError('TRACE_BUDGET_EXCEEDED: test trace preserved')
                    return {'status':'PROFILE_COLLECTED'},[],[]
                def lookup(*args):
                    if failure in ('preflight','missing_file'): raise ValueError('Pilot missing valid measurement')
                    return evidence
                validation=dict(audits={'RAW_SEMCACHE_QKV':{'coverage':{'reused_tokens':1023}}},quality={'raw':{'allclose':False}})
                recorder=Mock(); recorder.finish.return_value=evidence['memory']
                with patch.object(r,'Runtime',Runtime),patch.object(r,'backend_audit',return_value={}),\
                     patch.object(r,'memory',return_value={'gpu':gpu}),patch.object(r,'release_source'),\
                     patch.object(r,'monitor',return_value={'status':'GPU_MONITOR_VERIFIED'}),\
                     patch.object(r,'validate',return_value=validation),patch.object(r,'collect',side_effect=collect) as profiler,\
                     patch.object(r,'pilot_evidence',side_effect=lookup),patch.object(r,'PilotPeakRecorder',return_value=recorder),\
                     patch.object(r,'paired',side_effect=lambda runtime,args,identity,save:save(rows=[])),redirect_stderr(io.StringIO()):
                    args=['--phase','pilot' if failure=='pilot' else 'profile','--prompt-lengths','1024',
                        '--coverages','max','--output-dir',str(output)]
                    if failure!='pilot': args+=['--pilot-inputs',str(pilot),'--profile-memory-policy','pilot_guarded']
                    code=r.main(args)
                manifest=json.loads((output/'manifest.json').read_text())
                case=manifest['conditions']['L1024_covmax']
                if failure is None:
                    self.assertEqual(code,0); self.assertEqual(profiler.call_count,3)
                    self.assertFalse(case['strict_pass']); self.assertEqual(case['status'],'VALID')
                elif failure=='pilot':
                    self.assertEqual(case['status'],'CAPACITY_VALIDATED')
                    self.assertEqual(case['pilot_memory'],evidence['memory'])
                    recorder.begin.assert_called_once(); recorder.finish.assert_called_once()
                    current=copy.deepcopy(manifest); current['benchmark_config']['phase']='profile'
                    approved=pilot_evidence(output,current,1024,'max')
                    self.assertEqual(approved['memory']['allocation_increase_bytes'],4*GIB)
                else:
                    expected={'preflight':'MEMORY_PREFLIGHT_FAILED','missing_file':'MEMORY_PREFLIGHT_FAILED','OOM':'OOM','TRACE':'TRACE_FAILED'}[failure]
                    self.assertEqual(case['status'],expected); self.assertIn('traceback',case)
                    self.assertIn('memory_at_failure',case)
                    self.assertEqual(manifest['execution_status'],'FAILED' if failure=='TRACE' else 'COMPLETE_WITH_FAILURES')
                self.assertEqual(case['preflight']['source_qkv_backing_bytes'],480*1024**2)
                self.assertEqual(case['memory_preflight_audit']['policy'],'conservative' if failure=='pilot' else 'pilot_guarded')
                self.assertTrue(manifest['settings']['output_attentions'])
                self.assertFalse(manifest['correctness_qualified_speedup'])

    def test_launcher_only_remaining_profiles_one_allocation(self):
        text=(ROOT/'jobs/r2c2_profile_remaining.sbatch').read_text()
        self.assertIn('--phase pilot --prompt-lengths 512,1024',text)
        self.assertEqual(text.count('--phase profile --prompt-lengths 1024'),2)
        self.assertNotIn('--phase profile --prompt-lengths 512',text)
        self.assertNotIn('export CUDA_VISIBLE_DEVICES',text)
        self.assertNotIn('#SBATCH --array',text)
        self.assertIn('91_validate_r2c2_results.py validate',text)
        self.assertNotIn('AssertionError',text)


if __name__=='__main__': unittest.main()
