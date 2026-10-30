"""True localhost CPU KV-generation fixture; no OPT/GPU claim or downloads."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

from semcache.edgelora.r3a_artifacts import ROOT,read,sha
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.r4a_evidence import audit_run,load_run

SCRIPT=ROOT/'scripts/102_run_r4a_opt_full_e2e.py'


def temporary():
    base=ROOT/'results/r4/a';base.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(prefix='TEST_FIXTURE_ONLY_',dir=base)


class Contracts(unittest.TestCase):
    def test_no_heavy_import_help_dry_run_and_no_overwrite(self):
        command=[sys.executable,'-c',"import semcache.edgelora.r4a_runtime; import sys; assert not any(n in sys.modules for n in ('torch','transformers','peft','sacrebleu'))"]
        r=subprocess.run(command,capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')))
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(subprocess.run([sys.executable,str(SCRIPT),'--help'],capture_output=True).returncode,0)
        with temporary() as tmp:
            command=[sys.executable,str(SCRIPT),'--phase','dry-run','--output-dir',tmp]
            r=subprocess.run(command,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
            report=read(Path(tmp)/'r4a_results.json');self.assertFalse(report['gpu_inference_performed'])
            before={p.name:sha(p) for p in Path(tmp).iterdir()}
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(before,{p.name:sha(p) for p in Path(tmp).iterdir()})

    def test_gpu_missing_allocation_rejects_before_ud_or_cuda(self):
        with temporary() as tmp:
            env=dict(os.environ);env.pop('SLURM_JOB_ID',None)
            r=subprocess.run([sys.executable,str(SCRIPT),'--phase','run','--enable-gpu-experiment','--prepared',str(Path(tmp)/'absent_prepare'),
                '--output-dir',str(Path(tmp)/'failure')],env=env,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);report=read(Path(tmp)/'failure/r4a_results.json')
            self.assertEqual(report['execution_status'],'FAILED');self.assertFalse(report['gpu_inference_performed'])
            self.assertFalse((Path(tmp)/'failure/es').exists())

    def test_missing_dependencies_are_explicit_without_loading_gpu_or_installing(self):
        missing=[n for n in ('torch','numpy','transformers','peft','sacrebleu') if importlib.util.find_spec(n) is None]
        if not missing:self.skipTest('all dependencies installed; no fake GPU allocation is executed')
        with temporary() as tmp:
            env=dict(os.environ,SLURM_JOB_ID='TEST_FIXTURE_ONLY_PREFLIGHT',CUDA_VISIBLE_DEVICES='0')
            r=subprocess.run([sys.executable,str(SCRIPT),'--phase','run','--enable-gpu-experiment',
                '--prepared',str(Path(tmp)/'absent'),'--output-dir',str(Path(tmp)/'failed')],env=env,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);report=read(Path(tmp)/'failed/r4a_results.json')
            self.assertIn('MISSING_DEPENDENCY',report['reason']);self.assertFalse(report['gpu_inference_performed'])
            self.assertFalse((Path(tmp)/'failed/es').exists())


class PhysicalGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=temporary();cls.addClassCleanup(cls.temp.cleanup);cls.output=Path(cls.temp.name)/'physical'
        r=subprocess.run([sys.executable,'-u',str(SCRIPT),'--phase','fixture','--test-fixture-only','--output-dir',str(cls.output)],
            capture_output=True,text=True,timeout=120.)
        if r.returncode:raise AssertionError(r.stdout+r.stderr)

    def test_real_two_process_prefill_hit_plus_two_decode_passes_and_all_gates(self):
        report,plan,worker=load_run(self.output);rows=jsonl(self.output/'r4a_requests.jsonl');tokens=jsonl(self.output/'r4a_tokens.jsonl')
        self.assertEqual(len(rows),15);self.assertEqual(len(tokens),45);self.assertEqual(report['audit']['real_socket_forward_passes'],42)
        self.assertTrue(report['TEST_FIXTURE_ONLY']);self.assertFalse(report['gpu_inference_performed'])
        decision=read(self.output/'r4a_decision.json')
        for field in ('R4A_PHYSICAL_E2E_PASS','R4A_RAW_QKV_E2E_FIDELITY_PASS','R4A_TTFT_BENEFIT_OBSERVED','R4A_E2E_BENEFIT_OBSERVED',
            'NATIVE_PEFT_STRICT_PARITY_PASS','correctness_qualified_speedup'):self.assertFalse(decision[field])
        for row in rows:
            selected=[s for s in tokens if s['request_id']==row['request_id']]
            self.assertEqual(row['generated_tokens'],3);self.assertEqual(row['decode_forward_count'],2)
            self.assertEqual([s['context']['input_length'] for s in selected],[32,1,1])
            self.assertEqual([s['past']['sequence_length'] for s in selected],[32,33,34])
            self.assertEqual(row['e2e_wall_ms'],(row['request_end_monotonic_ns']-row['request_start_monotonic_ns'])/1e6)
            if row['mode']=='NO_REUSE':self.assertEqual(row['cache_lookup_ms'],0.);self.assertEqual(row['cache_events'],[])
            if row['mode']=='WARM_HIT':
                self.assertEqual(selected[0]['hidden_frames'],0);self.assertEqual(selected[0]['delta_frames'],0)
                self.assertEqual(selected[0]['ud_cpu_lora_ms'],0.);self.assertEqual(row['prefill_reused_qkv_outputs'],96)
                self.assertEqual(sum(s['hidden_frames'] for s in selected),64)  # decode traffic remains!
            else:self.assertEqual(sum(s['hidden_frames'] for s in selected),96)
        ready=read(self.output/'es/r4a_ud_ready.json');self.assertNotEqual(ready['pid'],worker['pid'])
        for pid in (ready['pid'],worker['pid']):
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
            probe.settimeout(.2);self.assertNotEqual(probe.connect_ex(('127.0.0.1',ready['port'])),0)
        self.assertTrue(read(self.output/'es/lifecycle.json')['tracked_descendants']['all_stopped'])
        self.assertEqual(read(self.output/'r4a_quality.json')['bleu_vs_ground_truth']['status'],'NOT_ESTIMABLE_NO_GOLD_REFERENCES')

    def test_independent_read_only_analysis_same_decision_and_no_input_mutations(self):
        before={str(p.relative_to(self.output)):sha(p) for p in self.output.rglob('*') if p.is_file()}
        with temporary() as tmp:
            r=subprocess.run([sys.executable,str(SCRIPT),'--phase','analyze','--inputs',str(self.output),'--output-dir',tmp],
                capture_output=True,text=True,timeout=60.)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            self.assertEqual(read(Path(tmp)/'r4a_decision.json'),read(self.output/'r4a_decision.json'))
        self.assertEqual(before,{str(p.relative_to(self.output)):sha(p) for p in self.output.rglob('*') if p.is_file()})

    def test_missing_or_corrupt_artifacts_and_rehashed_decode_frame_forgery_fail(self):
        plan=read(self.output/'es/es_plan.json');worker=read(self.output/'es/es_result.json')
        with temporary() as tmp:
            copy=Path(tmp)/'copy';shutil.copytree(self.output,copy)
            rows=jsonl(copy/'es/r4a_tokens.jsonl');rows[1]['hidden_frames']=0
            (copy/'es/r4a_tokens.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):audit_run(copy,plan,worker)
            with self.assertRaises(ValueError):load_run(copy)
            (copy/'es/r4a_tokens.jsonl').unlink()
            with self.assertRaises((ValueError,FileNotFoundError)):load_run(copy)


@unittest.skipUnless(os.environ.get('SEMCACHE_R4A_GPU')=='1','OPT-2.7B R4-A GPU NOT_RUN; explicit one-allocation opt-in required')
class AllocatedGPU(unittest.TestCase):
    def test_real_opt_pilot(self):
        output=Path(os.environ['SEMCACHE_R4A_OUTPUT']);output.mkdir(parents=True,exist_ok=False)
        commands=[['scripts/95_run_r3a_distributed_no_reuse.py','--phase','prepare','--output-dir',str(output/'base')],
            [str(SCRIPT),'--phase','prepare','--enable-gpu-experiment','--base-prepared',str(output/'base'),'--output-dir',str(output/'prepare')],
            [str(SCRIPT),'--phase','run','--enable-gpu-experiment','--prepared',str(output/'prepare/native'),'--output-dir',str(output/'run')]]
        for c in commands:
            r=subprocess.run([sys.executable,*c],capture_output=True,text=True,timeout=8000.);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        report,_,_=load_run(output/'run');self.assertFalse(report['TEST_FIXTURE_ONLY']);self.assertTrue(report['gpu_inference_performed'])


if __name__=='__main__':unittest.main()
