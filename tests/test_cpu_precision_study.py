"""Deterministic TEST_FIXTURE_ONLY CPU contracts; no model downloads or GPU."""
from contextlib import nullcontext
import copy
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import ProtocolError,digest
from semcache.edgelora.r3a_artifacts import ROOT,THRESHOLDS,pack,read,sha,source_identity,unpack
from semcache.edgelora.cpu_precision_study import (CANDIDATES,account,bind_matched_probe,
    compare_logits,cpu_candidate,decide,fp64_pair)

SCRIPT=ROOT/'scripts/98_explore_r3a_cpu_precision.py'


def quantize(value,dtype):
    return float(value) if dtype=='torch.float64' else unpack(pack([value],dtype.removeprefix('torch.')),dtype.removeprefix('torch.'))[0]


class ScalarTensor:
    """Small scalar TEST double; explicitly NOT Torch/BLAS numerical evidence."""
    def __init__(self,values,shape,dtype='torch.float32',device='cpu'):
        self.values=[quantize(v,dtype) for v in values];self.shape=tuple(shape);self.ndim=len(shape)
        self.dtype=dtype;self.device=SimpleNamespace(type=device)
        assert len(self.values)==math.prod(shape)
    def to(self,dtype):return ScalarTensor(self.values,self.shape,dtype,self.device.type)
    def count_nonzero(self):return sum(v!=0. for v in self.values)
    def __mul__(self,number):return ScalarTensor([v*number for v in self.values],self.shape,self.dtype)


def scalar_linear(x,w):
    assert x.dtype==w.dtype and x.shape[-1]==w.shape[-1]
    width=x.shape[-1];outputs=w.shape[0];values=[]
    for i in range(len(x.values)//width):
        for j in range(outputs):
            total=0.
            for k in range(width):total=quantize(total+quantize(x.values[i*width+k]*w.values[j*width+k],x.dtype),x.dtype)
            values.append(total)
    return ScalarTensor(values,(*x.shape[:-1],outputs),x.dtype)


def scalar_modules():
    torch=ModuleType('torch');nn=ModuleType('torch.nn');functional=ModuleType('torch.nn.functional')
    torch.float16='torch.float16';torch.float32='torch.float32';torch.float64='torch.float64'
    torch.inference_mode=nullcontext
    torch.isfinite=lambda t:SimpleNamespace(all=lambda:all(math.isfinite(v) for v in t.values))
    torch.stack=lambda ts,dim=0:ScalarTensor([v for t in ts for v in t.values],(len(ts),*ts[0].shape),ts[0].dtype)
    functional.linear=scalar_linear;nn.functional=functional;torch.nn=nn
    return {'torch':torch,'torch.nn':nn,'torch.nn.functional':functional}


def scalar_inputs():
    # Exact cancellation exposes the staged FP32 rounding boundary. These are
    # synthetic numbers, not exported research weights or observed GPU values.
    x=ScalarTensor([1.]*128,[1,32,4],'torch.float16');matrices={}
    a=[1.,2.**-24,0.,0.,1.,0.,0.,0.]+[.125,0.,0.,0.]*6
    b=[2.**24,-2.**24,0.,0.,0.,0.,0.,0.]+[.1,0.,0.,0.,0.,0.,0.,0.]*3
    for role in 'qkv':
        matrices[0,role,'A']=ScalarTensor(a,[8,4]);matrices[0,role,'B']=ScalarTensor(b,[4,8])
    return x,matrices


class ScalarOperationTests(unittest.TestCase):
    def setUp(self):
        self.context=patch.dict(sys.modules,scalar_modules());self.context.start();self.addCleanup(self.context.stop)
        self.x,self.matrices=scalar_inputs()

    def test_baseline_calls_actual_frozen_function_and_identical_output(self):
        from semcache.edgelora.user_device import cpu_deltas
        expected=cpu_deltas(self.x,self.matrices,0,4,torch_backend=True)
        with patch('semcache.edgelora.user_device.cpu_deltas',wraps=cpu_deltas) as delegated:
            precision={};actual=cpu_candidate(self.x,self.matrices,0,4,CANDIDATES[0],precision_out=precision)
            delegated.assert_called_once_with(self.x,self.matrices,0,4,torch_backend=True,precision_out=precision)
        self.assertEqual(actual.values,expected.values);self.assertEqual(actual.shape,(3,1,32,4))
        self.assertEqual(actual.dtype,'torch.float32');self.assertEqual(set(precision),set('qkv'))
        self.assertTrue(all(p['scale']==1. and p['A_output_dtype']=='torch.float32' for p in precision.values()))

    def test_full_and_staged_have_exact_predeclared_rounding_order(self):
        full=cpu_candidate(self.x,self.matrices,0,4,CANDIDATES[1])
        staged=cpu_candidate(self.x,self.matrices,0,4,CANDIDATES[2])
        self.assertEqual(full.values[0],1.);self.assertEqual(staged.values[0],0.)
        self.assertEqual(full.dtype,staged.dtype);self.assertEqual(full.device.type,'cpu')
        events=[]
        def cast(t,dtype):events.append(('cast',dtype));return t.to('torch.'+dtype)
        def linear(t,w):events.append(('linear',t.dtype,w.dtype));return scalar_linear(t,w)
        def scale(t,value):events.append(('scale',value));return t*value
        fp64_pair(self.x,self.matrices[0,'q','A'],self.matrices[0,'q','B'],CANDIDATES[2],cast=cast,linear=linear,scale=scale)
        self.assertEqual(events,[('cast','float64'),('cast','float64'),('linear','torch.float64','torch.float64'),
            ('cast','float32'),('cast','float64'),('cast','float64'),('linear','torch.float64','torch.float64'),
            ('scale',1.),('cast','float32')])

    def test_all_candidates_preserve_original_hidden_and_fp32_matrices(self):
        before=copy.deepcopy({k:t.values for k,t in self.matrices.items()});hidden=list(self.x.values)
        for candidate in CANDIDATES:cpu_candidate(self.x,self.matrices,0,4,candidate)
        self.assertEqual(before,{k:t.values for k,t in self.matrices.items()});self.assertEqual(hidden,self.x.values)
        self.assertTrue(all(t.dtype=='torch.float32' for t in self.matrices.values()))

    def test_wrong_dtype_shape_device_zero_weights_nonfinite_and_unknown_method_fail(self):
        for case in ('dtype','shape','device','zero','nan','hidden_nan','candidate'):
            x,matrices=scalar_inputs();method=CANDIDATES[0]
            if case=='dtype':matrices[0,'q','A'].dtype='torch.float64'
            if case=='shape':matrices[0,'q','A'].shape=(4,8)
            if case=='device':matrices[0,'q','B'].device.type='cuda'
            if case=='zero':matrices[0,'q','B'].values=[0.]*32
            if case=='nan':matrices[0,'q','A'].values[0]=float('nan')
            if case=='hidden_nan':x.values[0]=float('nan')
            if case=='candidate':method='CPU_FP16_UNDECLARED'
            with self.subTest(case=case),self.assertRaises(ProtocolError):cpu_candidate(x,matrices,0,4,method)

    def test_zero_hidden_is_valid_but_does_not_permit_zero_adapter(self):
        self.x.values=[0.]*128
        for candidate in CANDIDATES:self.assertFalse(any(cpu_candidate(self.x,self.matrices,0,4,candidate).values))


def records_fixture():
    return [dict(layer=layer,role=role,hidden=dict(sha256=digest(('TEST_FIXTURE_ONLY',layer,'hidden'))),
        same_native_hidden=layer==0,native_qkv_sha_equal=False) for layer in range(32) for role in 'qkv']


class ReportGates(unittest.TestCase):
    def test_full_logit_bytes_sha_shape_dtype_and_frozen_metrics(self):
        original=pack([float(i%7) for i in range(224)],'float32')
        result=compare_logits(original,original,[1,32,7])
        self.assertTrue(result['parity']['passed']);self.assertEqual(result['parity']['thresholds'],THRESHOLDS)
        self.assertEqual(result['logits']['bytes'],896);self.assertEqual(result['logits']['dtype'],'float32')
        changed=pack([float(i%7)+.1 for i in range(224)],'float32')
        bad=compare_logits(changed,original,[1,32,7]);self.assertFalse(bad['parity']['passed'])
        self.assertNotEqual(result['logits']['sha256'],bad['logits']['sha256'])
        with self.assertRaises(ProtocolError):compare_logits(changed[:-4],original,[1,32,7])
        with self.assertRaises(ProtocolError):compare_logits(changed,original,[])

    def test_all96_and_first_actual_divergence_accounting(self):
        records=records_fixture();result=account(records)
        self.assertEqual(result['qkv_outputs'],96);self.assertEqual(result['first_hidden_difference'],dict(layer=1,role='q'))
        self.assertEqual(result['first_qkv_difference'],dict(layer=0,role='q'))
        for mutation in ('missing','duplicate','hidden'):
            changed=copy.deepcopy(records)
            if mutation=='missing':changed.pop()
            if mutation=='duplicate':changed[1]=copy.deepcopy(changed[0])
            if mutation=='hidden':changed[1]['hidden']['sha256']=digest('TEST_FIXTURE_ONLY different hidden')
            with self.subTest(mutation=mutation),self.assertRaises(ProtocolError):account(changed)

    def test_matched_numeric_metrics_only_after_component_hash_binding(self):
        actual=dict(hidden=dict(sha256='h'),base=dict(sha256='b'),cpu_delta=dict(sha256='d'),qkv=dict(sha256='q'))
        probe=dict(cpu_delta=dict(sha256='d'),assembled=dict(sha256='q'),fp16_flip_count=2,
                   assembled_vs_native=dict(max_abs=.001),delta_vs_native_gpu=dict(max_abs=1e-8))
        native=dict(hidden=dict(sha256='h'),base=dict(sha256='b'),candidates={CANDIDATES[0]:probe})
        matched=bind_matched_probe(actual,native,CANDIDATES[0]);self.assertEqual(matched['fp16_flip_count'],2)
        actual['hidden']['sha256']='downstream different hidden'
        unmatched=bind_matched_probe(actual,native,CANDIDATES[0]);self.assertIsNone(unmatched['numeric_diff'])
        self.assertIsNone(unmatched['fp16_flip_count']);self.assertFalse(unmatched['same_hidden'])
        actual['hidden']['sha256']='h';actual['cpu_delta']['sha256']='different delta'
        with self.assertRaisesRegex(ProtocolError,'BINDING_MISMATCH'):bind_matched_probe(actual,native,CANDIDATES[0])

    def test_strict_failure_stop_and_local_pass_never_promote_tcp_or_production(self):
        candidates={n:dict(execution_status='COMPLETE',parity=dict(passed=False,thresholds=dict(THRESHOLDS)),
                           accounting=dict(qkv_outputs=96)) for n in CANDIDATES}
        failed=decide(True,candidates);self.assertEqual(failed['conclusion'],'NO_CPU_PRECISION_CANDIDATE_PASSED')
        candidates[CANDIDATES[1]]['parity']['passed']=True
        passed=decide(True,candidates);self.assertEqual(passed['conclusion'],'CANDIDATE_LOCAL_PARITY_PASS')
        self.assertFalse(passed['physical_tcp_parity_validated']);self.assertFalse(passed['PRIMARY_PERFORMANCE_ELIGIBLE'])
        self.assertEqual(decide(False,{})['conclusion'],'REFERENCE_NONREPRODUCIBLE')
        candidates[CANDIDATES[0]]['parity']['thresholds']['max_abs']=1.
        with self.assertRaises(ProtocolError):decide(True,candidates)
        with self.assertRaises(ProtocolError):decide(True,{})


class CLIAndLifecycle(unittest.TestCase):
    @staticmethod
    def cli():
        spec=importlib.util.spec_from_file_location('precision_cli',SCRIPT)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

    def test_help_dry_run_and_no_overwrite_without_model_imports(self):
        with tempfile.TemporaryDirectory(prefix='precision_study_TEST_FIXTURE_ONLY_',dir=ROOT/'results/r3/a') as temporary:
            output=Path(temporary)/'dry'
            command=[sys.executable,str(SCRIPT),'--phase','dry-run','--output-dir',str(output)]
            result=subprocess.run(command,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
            report=read(output/'r3a_cpu_precision_results.json');self.assertEqual(report['gpu_findings_status'],'NOT_RUN')
            self.assertFalse(report['original_151730_evidence_loaded']);self.assertFalse(report['gpu_inference_performed'])
            before={p.name:sha(p) for p in output.iterdir()}
            result=subprocess.run(command,capture_output=True,text=True);self.assertNotEqual(result.returncode,0)
            self.assertEqual(before,{p.name:sha(p) for p in output.iterdir()})
        result=subprocess.run([sys.executable,str(SCRIPT),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0);self.assertNotIn('--prepared',result.stdout)
        code="import semcache.edgelora.cpu_precision_study; import sys; assert not any(n in sys.modules for n in ('torch','transformers','peft','numpy'))"
        result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_run_opt_in_and_historical_inputs_rejected(self):
        import contextlib,io
        module=self.cli()
        for command in (['--phase','run','--output-dir','x'],['--phase','run','--enable-gpu-probes','--output-dir','x','--prepared','old'],
                        ['--phase','dry-run','--enable-gpu-probes','--output-dir','x']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):module.parse_args(command)

    def test_failed_run_is_reported_without_imputed_results(self):
        with tempfile.TemporaryDirectory(prefix='precision_study_TEST_FIXTURE_ONLY_',dir=ROOT/'results/r3/a') as temporary:
            output=Path(temporary)/'failure';env=dict(os.environ);env.pop('SLURM_JOB_ID',None)
            result=subprocess.run([sys.executable,str(SCRIPT),'--phase','run','--enable-gpu-probes','--output-dir',str(output)],
                env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);report=read(output/'r3a_cpu_precision_results.json')
            self.assertEqual(report['execution_status'],'FAILED');self.assertEqual(report['gpu_findings_status'],'NOT_RUN')
            self.assertTrue(all(v['execution_status']=='NOT_RUN' for v in report['candidates'].values()))
            self.assertIn('Slurm',report['reason'])

    def test_sequential_child_exit_and_timeout_cleanup(self):
        module=self.cli()
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            pid=module.run_process([sys.executable,'-c','print("TEST_FIXTURE_ONLY child")'],directory/'success',timeout=3.)
            life=read(directory/'success/lifecycle.json');self.assertTrue(life['all_children_exited'])
            self.assertEqual(life['pid'],pid);self.assertEqual(life['returncode'],0)
            with self.assertRaises(TimeoutError):
                module.run_process([sys.executable,'-c','import time; time.sleep(60)'],directory/'timeout',timeout=.1)
            life=read(directory/'timeout/lifecycle.json');self.assertTrue(life['all_children_exited'])
            self.assertTrue(life['forced_termination']);self.assertIsNotNone(life['returncode'])

    def test_prepare_like_separate_group_descendant_is_cleaned(self):
        module=self.cli()
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            code="import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); time.sleep(60)"
            with self.assertRaises(TimeoutError):module.run_process([sys.executable,'-c',code],directory,timeout=.3)
            life=read(directory/'lifecycle.json');self.assertTrue(life['tracked_descendants']['all_stopped'])
            self.assertTrue(life['tracked_descendants']['pid_start_ticks'])

    def test_one_shot_schedule_fixed_three_and_native_failure_blocks_them(self):
        """Mock orchestration only; never launches model/GPU/reference preparation."""
        import contextlib,io
        from semcache.edgelora.r3a_artifacts import write
        module=self.cli()
        for scenario in ('all_candidates_fail','native_error','candidate_error'):
            native_fails=scenario=='native_error'
            with self.subTest(scenario=scenario),tempfile.TemporaryDirectory(
                    prefix='precision_study_TEST_FIXTURE_ONLY_',dir=ROOT/'results/r3/a') as temporary:
                output=Path(temporary)/'mock';steps=[];identity=dict(TEST_FIXTURE_ONLY=True,versions={})
                pm=dict(TEST_FIXTURE_ONLY=True,gpu=dict(uuid='TEST_FIXTURE_ONLY_UUID'),identity=identity,
                    identity_sha256=digest(identity),code=source_identity(),preparation_pid=12)
                ref=dict(logits_sha256=digest('TEST_FIXTURE_ONLY logits'),logits_shape=[1,32,7],logits_dtype='float32')
                am=dict(matrix_sha256=digest('TEST_FIXTURE_ONLY A/B'))
                def process(command,directory,**kwargs):
                    steps.append('prepare');directory.mkdir()
                    # The real prepare API is checked here; the actual old
                    # runner is never called by this synthetic contract test.
                    self.assertIn('95_run_r3a_distributed_no_reuse.py',command[2])
                    self.assertEqual(command[command.index('--phase')+1],'prepare')
                    self.assertEqual(command[command.index('--prompt-lengths')+1],'32')
                    prepared=output/'prepare';prepared.mkdir()
                    write(prepared/'r3a_prepare_manifest.json',pm)
                    write(prepared/'r3a_preflight.json',dict(allocation_environment=dict(SLURM_JOB_ID='TEST_FIXTURE_ONLY')))
                    return 11
                def child(kind,plan,root,env):
                    name=plan.get('candidate',kind);steps.append(name)
                    if kind=='allocation':
                        return dict(process=dict(pid=10,gpu=dict(uuid='TEST_FIXTURE_ONLY_UUID'),versions={},affinity=[0]))
                    if kind=='native':
                        if native_fails:raise RuntimeError('REFERENCE_NONREPRODUCIBLE TEST_FIXTURE_ONLY')
                        (root/'native').mkdir();write(root/'native/worker_result.json',dict(TEST_FIXTURE_ONLY=True,execution_status='COMPLETE'))
                        return dict(native_exact=True,process=dict(pid=13))
                    if scenario=='candidate_error':raise RuntimeError('TEST_FIXTURE_ONLY candidate child failed')
                    return dict(TEST_FIXTURE_ONLY=True,execution_status='COMPLETE',process=dict(pid=14+CANDIDATES.index(name)),
                        parity=dict(passed=False,thresholds=dict(THRESHOLDS),max_abs=.1,relative_l2=.01,
                            tolerance_violations=1,last_token_kl=0.,last_token_argmax_agreement=True),
                        logits=dict(sha256=digest('TEST_FIXTURE_ONLY candidate')),accounting=dict(qkv_outputs=96))
                with patch.object(module,'run_process',side_effect=process),patch.object(module,'child',side_effect=child),\
                     patch.object(module,'verify_prepared',return_value=(pm,ref,am)),\
                     patch.object(module.importlib.util,'find_spec',return_value=object()),\
                     patch.dict(os.environ,dict(SLURM_JOB_ID='TEST_FIXTURE_ONLY',CUDA_VISIBLE_DEVICES='TEST_FIXTURE_ONLY')),\
                     contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    code=module.main(['--phase','run','--enable-gpu-probes','--output-dir',str(output)])
                report=read(output/'r3a_cpu_precision_results.json')
                if native_fails:
                    self.assertEqual(code,1);self.assertEqual(steps,['allocation','prepare','native'])
                    self.assertEqual(report['decision']['conclusion'],'REFERENCE_NONREPRODUCIBLE')
                    self.assertTrue(all(r['execution_status']=='NOT_RUN' for r in report['candidates'].values()))
                elif scenario=='candidate_error':
                    self.assertEqual(code,1);self.assertEqual(steps,['allocation','prepare','native',CANDIDATES[0]])
                    self.assertEqual(report['candidates'][CANDIDATES[0]]['execution_status'],'FAILED')
                    self.assertEqual(report['candidates'][CANDIDATES[1]]['execution_status'],'NOT_RUN')
                else:
                    self.assertEqual(code,0);self.assertEqual(steps,['allocation','prepare','native',*CANDIDATES])
                    self.assertEqual(report['decision']['conclusion'],'NO_CPU_PRECISION_CANDIDATE_PASSED')

    def test_source_glob_frozen_and_new_launcher_has_one_allocation_no_node_pin(self):
        files=source_identity()['files'];self.assertEqual(len(files),15)
        self.assertFalse(any('cpu_precision' in name or '/98_' in name for name in files))
        launcher=(ROOT/'jobs/r3a_cpu_precision_one_shot.sbatch').read_text()
        self.assertIn('#SBATCH --partition=batch_eebme_ugrad',launcher);self.assertIn('#SBATCH --gres=gpu:1',launcher)
        self.assertNotIn('--nodelist',launcher);self.assertNotIn('moana-',launcher)
        self.assertNotIn('scripts/97_',launcher);self.assertNotIn('--phase pilot',launcher)
        self.assertIn('HF_HUB_CACHE="$HF_HOME/hub"',launcher)
        self.assertEqual(THRESHOLDS,dict(atol=1e-3,rtol=1e-3,max_abs=1e-2,relative_l2=1e-3,last_token_kl=1e-5,last_token_argmax_agreement=True))


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Torch absent: actual CPU BLAS/GEMM tests NOT_RUN')
class ActualTorchCPU(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch;torch.manual_seed(42)
        self.x=torch.randn(1,32,4,dtype=torch.float16)
        self.matrices={(0,r,k):torch.randn(*shape,dtype=torch.float32) for r in 'qkv' for k,shape in (('A',(8,4)),('B',(4,8)))}

    def test_actual_baseline_bitwise_equals_original_cpu_path(self):
        from semcache.edgelora.user_device import cpu_deltas
        t=self.torch;before={k:v.clone() for k,v in self.matrices.items()}
        expected=cpu_deltas(self.x,self.matrices,0,4,torch_backend=True)
        actual=cpu_candidate(self.x,self.matrices,0,4,CANDIDATES[0])
        self.assertTrue(t.equal(actual,expected));self.assertEqual(actual.dtype,t.float32)
        self.assertTrue(all(t.equal(before[k],v) for k,v in self.matrices.items()))

    def test_actual_fp64_candidates_equal_exact_explicit_operation_specs(self):
        import torch.nn.functional as F
        t=self.torch;before={k:v.clone() for k,v in self.matrices.items()}
        for candidate in CANDIDATES[1:]:
            actual=cpu_candidate(self.x,self.matrices,0,4,candidate);expected=[]
            for role in 'qkv':
                middle=F.linear(self.x.float().double(),self.matrices[0,role,'A'].double())
                if candidate==CANDIDATES[2]:middle=middle.float().double()
                expected.append((F.linear(middle,self.matrices[0,role,'B'].double())*1.).float())
            self.assertTrue(t.equal(actual,t.stack(expected,dim=0)))
        self.assertTrue(all(t.equal(before[k],v) for k,v in self.matrices.items()))


@unittest.skipUnless(os.environ.get('SEMCACHE_R3A_PRECISION_GPU')=='1','explicit new-allocation GPU study opt-in required; GPU NOT_RUN')
class AllocatedGPU(unittest.TestCase):
    def test_one_shot(self):
        output=os.environ['SEMCACHE_R3A_PRECISION_OUTPUT']
        result=subprocess.run([sys.executable,str(SCRIPT),'--phase','run','--enable-gpu-probes','--output-dir',output],
            capture_output=True,text=True,timeout=8000.)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        report=read(Path(output)/'r3a_cpu_precision_results.json');self.assertEqual(report['gpu_findings_status'],'COMPLETE')
        self.assertFalse(report['PRIMARY_PERFORMANCE_ELIGIBLE'])


if __name__=='__main__':unittest.main()
