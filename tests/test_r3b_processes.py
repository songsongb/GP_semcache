"""True localhost CPU TEST_FIXTURE_ONLY integration; never GPU evidence."""
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

from semcache.edgelora.r3a_artifacts import ROOT,read,sha,write
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.r3b_analysis import audit_run

SCRIPT=ROOT/'scripts/99_run_r3b_raw_qkv_one_shot.py'


class CLIContracts(unittest.TestCase):
    def test_help_dry_run_no_heavy_import_and_no_overwrite(self):
        base=ROOT/'results/r3/b';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='raw_qkv_one_shot_TEST_FIXTURE_ONLY_',dir=base) as temporary:
            output=Path(temporary)/'dry'
            command=[sys.executable,str(SCRIPT),'--phase','dry-run','--output-dir',str(output)]
            r=subprocess.run(command,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
            report=read(output/'r3b_one_shot_results.json');self.assertEqual(report['gpu_findings_status'],'NOT_RUN')
            self.assertFalse(report['actual_network_socket_used']);self.assertTrue(all(v['status']=='NOT_RUN' for v in report['stages'].values()))
            before={p.name:sha(p) for p in output.iterdir()}
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(before,{p.name:sha(p) for p in output.iterdir()})
        r=subprocess.run([sys.executable,str(SCRIPT),'--help'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0);self.assertNotIn('--prepared',r.stdout)
        code="import semcache.edgelora.r3b_runtime; import sys; assert not any(n in sys.modules for n in ('torch','transformers','peft'))"
        r=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_production_missing_allocation_fails_before_any_listener(self):
        base=ROOT/'results/r3/b';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='raw_qkv_one_shot_TEST_FIXTURE_ONLY_',dir=base) as temporary:
            output=Path(temporary)/'failed';env=dict(os.environ);env.pop('SLURM_JOB_ID',None)
            r=subprocess.run([sys.executable,str(SCRIPT),'--phase','run','--enable-gpu-experiment','--output-dir',str(output)],
                capture_output=True,text=True,env=env)
            self.assertNotEqual(r.returncode,0);report=read(output/'r3b_one_shot_results.json')
            self.assertEqual(report['execution_status'],'FAILED');self.assertFalse(report['gpu_inference_performed'])
            self.assertFalse((output/'es/r3a_ud_ready.json').exists())


class PhysicalFixture(unittest.TestCase):
    def setUp(self):
        base=ROOT/'results/r3/b';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='raw_qkv_one_shot_TEST_FIXTURE_ONLY_',dir=base);self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name)/'fixture'

    def run_fixture(self):
        return subprocess.run([sys.executable,'-u',str(SCRIPT),'--phase','fixture','--test-fixture-only',
            '--warmup-runs','1','--measured-runs','2','--output-dir',str(self.output)],capture_output=True,text=True,timeout=90.)

    def cleanup_assertions(self):
        ready=read(self.output/'es/r3a_ud_ready.json');lifecycle=read(self.output/'es/lifecycle.json')
        self.assertTrue(lifecycle['all_children_exited']);self.assertTrue(lifecycle['tracked_descendants']['all_stopped'])
        for pid in (ready['pid'],lifecycle['pid']):
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as client:
            client.settimeout(.2);self.assertNotEqual(client.connect_ex(('127.0.0.1',ready['port'])),0)

    def test_actual_two_process_tcp_all96_reuse_stage_gates_and_raw_measurements(self):
        r=self.run_fixture();self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        report=read(self.output/'r3b_one_shot_results.json');self.assertTrue(report['TEST_FIXTURE_ONLY'])
        self.assertFalse(report['gpu_inference_performed']);self.assertTrue(report['actual_network_socket_used'])
        self.assertTrue(all(v['status']=='PASS' for v in report['stages'].values()))
        rows=jsonl(self.output/'r3b_requests.jsonl');self.assertEqual(len(rows),12)
        for row in rows:
            self.assertNotEqual(row['es_pid'],row['ud_pid']);self.assertTrue(row['cache_bitwise_passed'])
            if row['mode']=='WARM_HIT':
                self.assertEqual(row['hidden_frames']+row['delta_frames'],0);self.assertEqual(row['ud_cpu_lora_ms'],0.)
                self.assertEqual(row['reused_qkv_outputs'],96)
            else:self.assertEqual(row['hidden_frames'],32);self.assertEqual(row['delta_frames'],32)
        self.assertEqual(report['audit']['summary']['by_mode']['WARM_HIT']['samples'],2)
        self.cleanup_assertions()

    def test_independent_audit_detects_rehashed_counts_native_failure_not_cache_failure(self):
        r=self.run_fixture();self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        pm=read(self.output/'prepare/r3a_prepare_manifest.json');worker=read(self.output/'es/r3b_worker_result.json')
        rows=jsonl(self.output/'es/r3b_requests.jsonl');rows[2]['hidden_frames']=1
        path=self.output/'es/r3b_requests.jsonl'
        import json
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with self.assertRaises(ValueError):audit_run(self.output/'es',pm,worker,1,2)
        self.cleanup_assertions()

    def private_fixture(self,*,native_fail=False,capacity=1024*1024):
        """Fresh TEST_FIXTURE_ONLY data only; no real reference is rewritten."""
        from semcache.edgelora.r3a_tiny import prepare_tiny
        from semcache.edgelora.r3a_artifacts import pack,unpack
        spec=importlib.util.spec_from_file_location('r3b_test_cli',SCRIPT);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        root=Path(self.temp.name);prepared=prepare_tiny(root/'prepare');self.output=root
        if native_fail:
            ref_path=prepared/'r3a_reference.json';ref=read(ref_path);logits_path=prepared/ref['logits_file']
            values=unpack(logits_path.read_bytes(),'float32');values[0]+=1.
            logits_path.write_bytes(pack(values,'float32'));ref['logits_sha256']=sha(logits_path);write(ref_path,ref)
            pm_path=prepared/'r3a_prepare_manifest.json';pm=read(pm_path)
            pm['artifact_hashes'][ref_path.name]=sha(ref_path);pm['artifact_hashes'][logits_path.name]=sha(logits_path);write(pm_path,pm)
        es=root/'es';es.mkdir()
        plan=dict(prepared=str(prepared),prepared_manifest_sha256=sha(prepared/'r3a_prepare_manifest.json'),output=str(es),
            fixture=True,device='cpu',warmup=1,measured=2,capacity_bytes=capacity,sources=module.sources())
        write(es/'r3b_plan.json',plan)
        command=[sys.executable,str(SCRIPT),'--worker','es','--test-fixture-only','--plan',str(es/'r3b_plan.json')]
        try:module.process_runner()(command,es,timeout=60.)
        except ValueError:pass  # expected physical-worker failure is audited below
        return read(es/'r3b_worker_result.json'),read(prepared/'r3a_prepare_manifest.json')

    def test_synthetic_native_failure_preserved_but_cache_bitwise_gates_pass(self):
        worker,pm=self.private_fixture(native_fail=True)
        self.assertEqual(worker['execution_status'],'COMPLETE')
        audit=audit_run(self.output/'es',pm,worker,1,2)
        self.assertEqual(audit['native_strict_passes'],0);self.assertEqual(audit['native_strict_failures'],12)
        self.assertTrue(audit['output_cache_bitwise_all_pass']);self.cleanup_assertions()

    def test_admission_failure_stops_before_b2_b3_and_cleans_workers(self):
        worker,pm=self.private_fixture(capacity=6143)
        self.assertEqual(worker['execution_status'],'FAILED')
        self.assertEqual(worker['stages']['B1']['status'],'FAIL')
        self.assertEqual(worker['stages']['B2']['status'],'NOT_RUN');self.assertEqual(worker['stages']['B3']['status'],'NOT_RUN')
        self.assertIn('admission',worker['reason']);self.cleanup_assertions()


@unittest.skipUnless(os.environ.get('SEMCACHE_R3B_GPU')=='1','OPT-2.7B GPU R3-B NOT_RUN; explicit Slurm allocation/opt-in required')
class GPUOptIn(unittest.TestCase):
    def test_physical_gpu(self):
        output=os.environ['SEMCACHE_R3B_OUTPUT']
        r=subprocess.run([sys.executable,str(SCRIPT),'--phase','run','--enable-gpu-experiment',
            '--output-dir',output],capture_output=True,text=True,timeout=6000.)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        report=read(Path(output)/'r3b_one_shot_results.json');self.assertFalse(report['TEST_FIXTURE_ONLY'])
        self.assertTrue(all(v['status']=='PASS' for v in report['stages'].values()))


if __name__=='__main__':unittest.main()
