"""Synthetic CPU schemas/numerics; injected differences are not GPU findings."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import digest,digest_bytes
from semcache.edgelora.r3a_artifacts import ROOT,read,write,pack,unpack
from semcache.edgelora.r3a_tiny import prepare_tiny
from semcache.edgelora.r5_i5_evidence import seal
from semcache.edgelora.r5_i5r2_tensors import Evidence,file_sha
from semcache.edgelora.r5_i5r2bs1_plan import historical,prepare,validate_plan,schedule,suffix,layout_storage,fresh,CORE
from semcache.edgelora.r5_i5r2bs1_runtime import Store,measure
from semcache.edgelora.r5_i5r2bs1_analysis import analyze,compare_bytes,promote,verdicts


def historical_fixture(root,*,full=False,strided=False):
    base=prepare_tiny(root/'base');pm=read(base/'r3a_prepare_manifest.json');run=root/'historical_run';run.mkdir()
    plan_root=root/'historical_plan';plan_root.mkdir();review=root/'historical_review';review.mkdir()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    frozen='src/semcache/edgelora/r3b_runtime.py';inventory={frozen:file_sha(ROOT/frozen)}
    src=dict(commit=commit,files=inventory)
    tasks=[dict(request_id=rid,user_id='User_01',adapter_id='Adapter_A',token_ids=[1]*n) for rid,n in (('S',64),('F',256))]
    block=dict(dataset='qasper',tasks=tasks);config=dict(fixture=True,source_commit=commit,sources=inventory,block=block,
        i5r2b=dict(source=src),base_prepared=str(base),experiment_prepared=str(plan_root))
    old=dict(blocks=[block],i5r2b=config['i5r2b']);old['experiment_id']=digest(old)
    write(plan_root/'i5_workload_manifest.json',old);seal(plan_root)
    config.update(experiment_id=old['experiment_id'],experiment_manifest_sha256=file_sha(plan_root/'i5_output_manifest.json'))
    write(run/'i5_execution_config.json',config);write(run/'i5_block_status.json',{'status':'COMPLETE'})
    prefix=pack([.125+(i%4)/16 for i in range(64*4)],'float16')
    backend=dict(TF32_matmul=False,FP16_reduced_precision_reduction=True,float32_matmul_precision='highest',cudnn_TF32=False,
        cudnn_benchmark=False,cudnn_deterministic=False,deterministic_algorithms=True,cpu_threads=4,autocast_enabled=False,CUBLAS_WORKSPACE_CONFIG=':4096:8')
    directory=run/'FULL_RECOMPUTE/Adapter_A/es';directory.mkdir(parents=True);entries={};contexts={}
    for rid,n in (('S',64),('F',256)):
        contexts[rid]=dict(actual_context=dict(input_token_ids=[1]*n,past_length=0,pass_index=0,positions=list(range(n)),attention_mask=[1]*n),
            backend_before=backend,backend_after=backend)
        for role in 'qkv':
            for boundary in ('projection_input','base'):
                data=suffix(prefix,4,42) if full and rid=='F' and boundary=='projection_input' else prefix
                retained=len(data)//8;name=rid+'_'+boundary+'_'+role+'.bin';(directory/name).write_bytes(data)
                entries[rid+'/'+boundary+'_'+role]=dict(status='CAPTURED',file=name,sha256=digest_bytes(data),dtype='float16',shape=[1,retained,4],device='cpu',
                    observed_shape=[1,n,4],token_axis=1,semantic_positions=[0,retained],observed_stride=[n*8,8,2] if strided else [n*4,4,1],
                    original_contiguous=not strided,storage_offset=1 if strided else 0)
    cap=dict(status='COMPLETE',entries=entries,contexts=contexts);write(directory/'i5r2b_capture.json',cap);write(directory/'i5r2b_backend.json',backend)
    write(run/'FULL_RECOMPUTE/Adapter_A/identity.json',dict(model_identity=pm['identity'],tokenizer={'TEST_FIXTURE_ONLY':True}))
    write(run/'FULL_RECOMPUTE/gpu_identity.json',dict(TEST_FIXTURE_ONLY=True,execution_gpu_uuid=None))
    seal(run)
    write(review/'i5r2b_capture_inventory.json',dict(config=config['i5r2b'],source_commit=commit,
        independent_I5_audit=dict(status='PASS',TEST_FIXTURE_ONLY=True),inventories={'FULL_RECOMPUTE/ES':cap}))
    write(review/'i5r2b_first_divergence.json',dict(TEST_FIXTURE_ONLY=True,GPU_EVIDENCE_COMPLETE='CPU_FIXTURE_ONLY'))
    write(review/'i5r2b_output_manifest.json',dict(files={p.name:file_sha(p) for p in review.iterdir()}))
    return run,review,base,commit


def synthetic_run(experiment,path,*,perturb=True):
    """Actual stdlib CPU linear arithmetic plus explicit injected FP16 difference."""
    plan=validate_plan(experiment);pe=Evidence(experiment);pe.verify();out=fresh(path);store=Store(out);rows=[];inputs={};weights={}
    for r in 'qkv':
        raw=pack([1. if i//4==i%4 else 0. for i in range(16)],'float16');bias=pack([.0625]*4,'float16')
        saved={name:store.tensor('weights/'+r+'_'+name+'.bin',data,shape,'float16') for name,data,shape in
               (('weight',raw,[4,4]),('bias',bias,[4]))}
        original={k:{f:v for f,v in item.items() if f in ('shape','dtype','sha256')} for k,item in saved.items()}
        converted={k:dict(shape=v['shape'],dtype='float32',sha256=digest_bytes(promote(raw if k=='weight' else bias))) for k,v in original.items()}
        weights[r]=dict(module='SYNTHETIC_CPU_SCHEMA',module_class='Linear',before=original,after=original,saved=saved,
            NATIVE_FP16_temporary_parameter_hashes=original,DERIVED_FP32_temporary_parameter_hashes=converted)
        for precision,dtype in (('NATIVE_FP16','float16'),('DERIVED_FP32','float32')):
            for control in plan['roles'][r]['layouts']:
                group=r+'/'+precision+'/'+control;native=pe.tensor('',plan['blobs'][r+'/input_long']);data=native if dtype=='float16' else promote(native)
                width=2 if dtype=='float16' else 4;xs={64:data[:64*4*width],256:data};inputs[group]={}
                for n in (64,256):
                    old=plan['roles'][r]['captures'][('source' if n==64 else 'target')+'_projection_input']['item']
                    strides=[n*4,4,1] if control=='CONTIGUOUS' else old['observed_stride'];offset=0 if control=='CONTIGUOUS' else old['storage_offset']
                    inputs[group][str(n)]=store.tensor('inputs/'+group.replace('/','_')+'_'+str(n)+'.bin',xs[n],[1,n,4],dtype,
                        dict(shape=[1,n,4],dtype=dtype,stride=strides,storage_offset=offset,contiguous=control=='CONTIGUOUS',device='cpu'))
                def verify(n):return dict(input_sha256=digest_bytes(xs[n]),input_prefix_sha256=digest_bytes(xs[64]),weight=original,
                    active_parameters=original if dtype=='float16' else converted,gpu_uuid='TEST_FIXTURE_ONLY',backend={k:plan['historical_backend'][k] for k in CORE})
                def invoke(n):
                    result=[v+.0625 for v in unpack(xs[n],dtype)]
                    if perturb and n==256 and dtype=='float16':result[0]+=.015625
                    return pack(result,dtype)
                def save(data,step):return store.tensor('outputs/'+group.replace('/','_')+'_'+str(step['length'])+'_'+str(step['round'])+'.bin',data,[1,step['length'],4],dtype)
                rows.extend(dict(x,role=r,precision=precision,control=control) for x in measure(plan['schedule'],invoke,verify,save))
    store.put('i5r2bs1_plan.json',plan);store.put('i5r2bs1_records.json',rows);store.put('i5r2bs1_inputs.json',inputs);store.put('i5r2bs1_weights.json',weights)
    store.put('i5r2bs1_status.json',dict(status='COMPLETE',GPU_NOT_RUN=True,TEST_FIXTURE_ONLY=True,experiment_id=plan['experiment_id'],source_commit=plan['source']['commit'],
        model_load_count=1,production_forward_calls=0,LoRA_calls=0,TCP_frames=0,cache_reads=0,generation_tokens=0))
    store.put('i5r2bs1_backend.json',dict(effective=plan['historical_backend'],after=plan['historical_backend'],gpu={'execution_gpu_uuid':'TEST_FIXTURE_ONLY'}))
    store.writer.reconcile();seal(out);return out


class Numerics(unittest.TestCase):
    def test_frozen_tolerance_and_ULP_known_perturbation(self):
        a=pack([1.,-1.,0.],'float16');b=pack([1.0009765625,-1.0009765625,-0.],'float16')
        m=compare_bytes(a,b,'float16');self.assertEqual(m['FP16_ULP']['max'],1);self.assertEqual(m['FP16_ULP']['nonzero'],2)
        self.assertFalse(compare_bytes(pack([1.],'float16'),pack([1.03125],'float16'),'float16')['activation_tolerance_pass'])
    def test_promoted_FP32_preserves_exact_numeric_values(self):
        p=pack([.125,-.5,1.],'float16');self.assertEqual(unpack(promote(p),'float32'),unpack(p,'float16'))
    def test_balanced_repeats_and_warmups(self):
        s=schedule();self.assertEqual(sum(x['warmup'] for x in s),4)
        self.assertEqual([x['length'] for x in s[:4]],[64,256,256,64])
        for n in (64,256):self.assertEqual(sum(not x['warmup'] and x['length']==n for x in s),6)
        for bad in (0,5,100):
            with self.assertRaises(ValueError):schedule(bad)
    def test_suffix_is_deterministic_and_never_modifies_prefix(self):
        p=pack([.5]*256,'float16');self.assertEqual(suffix(p,4)[:512],p)
        self.assertEqual(suffix(p,4),suffix(p,4));self.assertNotEqual(suffix(p,4,43),suffix(p,4,42))
    def test_layout_storage_bound_overlap_and_offset(self):
        self.assertEqual(layout_storage([1,64,4],[512,8,2],1),512)
        for stride,offset in (([1,0,1],0),([1,4,1],-1),([1,1000000,1],0)):
            with self.assertRaises(ValueError):layout_storage([1,64,4],stride,offset)
    def test_measure_invokes_once_and_detects_mutation(self):
        calls=[];saved=[]
        rows=measure(schedule(),lambda n:calls.append(n) or b'x',lambda n:dict(immutable=True),lambda out,step:saved.append(step) or {})
        self.assertEqual(len(calls),16);self.assertEqual(len(rows),12);self.assertEqual(len(saved),12)
        state=[0]
        def mutate(n):state[0]+=1;return b'x'
        with self.assertRaises(ValueError):measure(schedule(),mutate,lambda n:state[0],lambda *a:None)
    def test_classification_distinguishes_confounds_precision_and_no_difference(self):
        same=[dict(precision=p,metrics={'bitwise_equal':True}) for p in ('NATIVE_FP16','DERIVED_FP32')]
        cross=[dict(precision=p,metrics={'bitwise_equal':p=='DERIVED_FP32'}) for p in ('NATIVE_FP16','DERIVED_FP32')]
        result=verdicts(same,cross,fixture=True);self.assertEqual(result['case'],'CASE_C_PRECISION_SENSITIVE_SHAPE_EFFECT')
        self.assertEqual(result['SHAPE_DEPENDENCE_DEMONSTRATED'],'CPU_FIXTURE_ONLY');self.assertFalse(result['ROOT_CAUSE_CONFIRMED'])
        same[0]['metrics']['bitwise_equal']=False;self.assertIn('CONFOUNDED',verdicts(same,cross)['case'])
        same[0]['metrics']['bitwise_equal']=True;cross[1]['metrics']['bitwise_equal']=False;self.assertTrue(verdicts(same,cross)['case'].startswith('CASE_D'))
        cross[0]['metrics']['bitwise_equal']=True;self.assertTrue(verdicts(same,cross)['case'].startswith('CASE_E'))
        self.assertEqual(verdicts([],[])['case'],'CASE_F_INCOMPLETE')


class HistoricalAndAudit(unittest.TestCase):
    def inputs(self,root,**kw):
        run,review,base,commit=historical_fixture(root,**kw);experiment=root/'S1_plan'
        prepare(run,review,base,experiment,expected_commit=commit,fixture=True);return run,review,base,commit,experiment
    def test_real_reference_schema_and_missing_suffix_are_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            r,v,b,c,p=self.inputs(Path(d));x=validate_plan(p)
            self.assertFalse(x['roles']['q']['full_suffix_available']);self.assertEqual(x['roles']['q']['suffix_method'],'SYNTHETIC_XORSHIFT32_DYADIC_V1')
            before=file_sha(r/'i5_output_manifest.json');historical(r,v,expected_commit=c,fixture=True);self.assertEqual(before,file_sha(r/'i5_output_manifest.json'))
    def test_current_source_commit_change_rejects_sealed_plan(self):
        with tempfile.TemporaryDirectory() as d:
            *_,p=self.inputs(Path(d));plan=validate_plan(p)
            with patch('semcache.edgelora.r5_i5r2bs1_plan.source_identity',return_value=dict(plan['source'],commit='0'*40)):
                with self.assertRaisesRegex(ValueError,'source changed'):validate_plan(p)
    def test_available_full_suffix_is_used_without_synthetic_substitution(self):
        with tempfile.TemporaryDirectory() as d:
            *_,p=self.inputs(Path(d),full=True);plan=validate_plan(p)
            self.assertTrue(plan['roles']['q']['full_suffix_available']);self.assertEqual(plan['roles']['q']['suffix_method'],'RECORDED_FULL_INPUT')
    def test_noncontiguous_historical_layout_is_explicit_additional_condition(self):
        with tempfile.TemporaryDirectory() as d:
            *_,p=self.inputs(Path(d),strided=True);plan=validate_plan(p)
            self.assertEqual(plan['roles']['q']['layouts'],['CONTIGUOUS','HISTORICAL_STRIDED'])
    def test_missing_and_corrupt_historical_bytes_fail_closed(self):
        for remove in (False,True):
            with self.subTest(remove=remove),tempfile.TemporaryDirectory() as d:
                r,v,b,c=historical_fixture(Path(d));path=next(r.rglob('S_projection_input_q.bin'))
                path.unlink() if remove else path.write_bytes(b'bad')
                with self.assertRaises(ValueError):historical(r,v,expected_commit=c,fixture=True)
    def test_altered_source_inventory_and_wrong_commit_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            r,v,b,c=historical_fixture(Path(d))
            with self.assertRaises(ValueError):historical(r,v,expected_commit='0'*40,fixture=True)
            cfg=read(r/'i5_execution_config.json');cfg['sources'][next(iter(cfg['sources']))]='0'*64;write(r/'i5_execution_config.json',cfg);seal(r)
            with self.assertRaisesRegex(ValueError,'historical source'):historical(r,v,expected_commit=c,fixture=True)
    def test_unequal_historical_prefix_rejected_even_after_reseal(self):
        with tempfile.TemporaryDirectory() as d:
            r,v,b,c=historical_fixture(Path(d));path=next(r.rglob('F_projection_input_q.bin'));path.write_bytes(pack([0.]*256,'float16'));seal(r)
            with self.assertRaisesRegex(ValueError,'corrupt tensor'):historical(r,v,expected_commit=c,fixture=True)
            cap_path=path.parent/'i5r2b_capture.json';cap=read(cap_path);cap['entries']['F/projection_input_q']['sha256']=file_sha(path)
            write(cap_path,cap);seal(r)
            inv_path=v/'i5r2b_capture_inventory.json';inv=read(inv_path);inv['inventories']['FULL_RECOMPUTE/ES']=cap;write(inv_path,inv)
            write(v/'i5r2b_output_manifest.json',dict(files={p.name:file_sha(p) for p in v.iterdir() if p.name!='i5r2b_output_manifest.json'}))
            with self.assertRaisesRegex(ValueError,'prefixes unequal'):historical(r,v,expected_commit=c,fixture=True)
    def test_synthetic_full_analysis_is_deterministic_read_only_and_unqualified(self):
        with tempfile.TemporaryDirectory() as d:
            *_,p=self.inputs(Path(d));r=synthetic_run(p,Path(d)/'run');before=file_sha(r/'i5_output_manifest.json')
            a=analyze(p,r,Path(d)/'audit1');b=analyze(p,r,Path(d)/'audit2')
            self.assertEqual(a,b);self.assertTrue(a['GPU_NOT_RUN']);self.assertFalse(a['ROOT_CAUSE_CONFIRMED'])
            self.assertEqual(a['case'],'CASE_C_PRECISION_SENSITIVE_SHAPE_EFFECT');self.assertEqual(file_sha(r/'i5_output_manifest.json'),before)
            self.assertEqual(file_sha(Path(d)/'audit1/i5r2bs1_comparisons.json'),file_sha(Path(d)/'audit2/i5r2bs1_comparisons.json'))
    def test_missing_duplicate_or_changed_receipt_rejected(self):
        for change in ('missing','duplicate','weight','input','backend'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                *_,p=self.inputs(Path(d));r=synthetic_run(p,Path(d)/'run');rows=read(r/'i5r2bs1_records.json')
                if change=='missing':rows.pop()
                elif change=='duplicate':rows.append(rows[0])
                elif change=='weight':rows[0]['receipt']['active_parameters']['weight']['sha256']='0'*64
                elif change=='input':rows[0]['receipt']['input_prefix_sha256']='0'*64
                else:rows[0]['receipt']['backend']['TF32_matmul']=True
                write(r/'i5r2bs1_records.json',rows);seal(r)
                with self.assertRaises(ValueError):analyze(p,r,Path(d)/'bad_audit')
    def test_strided_control_is_audited_separately_from_shape_control(self):
        with tempfile.TemporaryDirectory() as d:
            *_,p=self.inputs(Path(d),strided=True);r=synthetic_run(p,Path(d)/'run')
            result=analyze(p,r,Path(d)/'audit')
            self.assertEqual(set(result['per_layout']),{'CONTIGUOUS','HISTORICAL_STRIDED'})
            comparisons=read(Path(d)/'audit/i5r2bs1_comparisons.json')['layout_matched_rounds']
            self.assertTrue(comparisons);self.assertTrue(all(x['metrics']['bitwise_equal'] for x in comparisons))
    def test_evidence_budget_and_file_bound_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            store=Store(Path(d),max_files=1);store.tensor('a.bin',b'\0\0',[1],'float16')
            with self.assertRaisesRegex(ValueError,'file bound'):store.tensor('b.bin',b'\0\0',[1],'float16')
            self.assertFalse((Path(d)/'b.bin').exists())
        with tempfile.TemporaryDirectory() as d:
            store=Store(Path(d),budget=64*2**20+1)
            with self.assertRaisesRegex(ValueError,'BUDGET'):store.tensor('a.bin',b'xx',[1],'float16')
    def test_invalid_tensor_size_and_unsafe_paths_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            store=Store(Path(d))
            with self.assertRaisesRegex(ValueError,'shape/dtype'):store.tensor('bad.bin',b'x',[1],'float16')
            with self.assertRaisesRegex(ValueError,'escapes'):store.tensor('../bad.bin',b'\0\0',[1],'float16')
    def test_safe_output_namespace_and_existing_directory(self):
        with self.assertRaises(ValueError):fresh(ROOT/'results/r5/i5r/controlled_hit/S1_PROHIBITED')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):fresh(d)
    def test_production_arithmetic_files_unchanged(self):
        for name in ('r3b_runtime.py','user_device.py','r5_i5_driver.py','r5_i5_cache.py','r5_i5_generation.py','r3a_artifacts.py'):
            p='src/semcache/edgelora/'+name
            committed=subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT)
            self.assertEqual(digest_bytes(committed),file_sha(ROOT/p))
    def test_job_wrapper_rejects_arrays_and_has_no_default_node(self):
        import os
        script=ROOT/'jobs/r5_i5r2bs1_linear.sh';self.assertNotIn('#SBATCH --nodelist=',script.read_text())
        variables=dict(os.environ,SLURM_JOB_ID='CPU_NEGATIVE_FIXTURE',SLURM_ARRAY_TASK_ID='1',S1_ENABLE_GPU='1',S1_NODE='moana-y3',
            S1_EXPERIMENT='UNUSED',S1_HISTORICAL_RUN='UNUSED',S1_HISTORICAL_REVIEW='UNUSED',S1_BASE_PREPARED='UNUSED',
            S1_SOURCE_COMMIT='UNUSED',S1_RUN_ROOT='UNUSED',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        result=subprocess.run(['bash',str(script)],env=variables,capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('arrays are forbidden',result.stderr)


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('torch','numpy')),'actual CPU Torch S1 helper test NOT_RUN; dependencies absent')
class ActualTorchCPU(unittest.TestCase):
    def test_real_CPU_Linear_and_strided_helpers_preserve_operands(self):
        import torch
        from semcache.edgelora.r5_i5r2bs1_runtime import parameters,reconstruct
        module=torch.nn.Linear(4,4).eval();before=parameters(module);long=torch.arange(1024,dtype=torch.float32).reshape(1,256,4)/1024
        strided=reconstruct(long,dict(observed_stride=[2048,8,2],storage_offset=1));self.assertTrue(torch.equal(long,strided));self.assertFalse(strided.is_contiguous())
        with torch.inference_mode():self.assertTrue(torch.allclose(module(long[:,:64]),module(long)[:,:64]))
        self.assertEqual(parameters(module),before)


if __name__=='__main__':unittest.main()
