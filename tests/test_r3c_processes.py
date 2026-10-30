"""Real CPU TCP/process fixtures; never evidence of OPT/GPU inference."""
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
from unittest.mock import patch

from semcache.edgelora.r3a_artifacts import ROOT,read,sha,write
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.r3c_analysis import audit_model
from semcache.edgelora.r3c_process import spawn,ready,reap

SCRIPT=ROOT/'scripts/100_run_r3c_network_one_shot.py'


def cli():
    spec=importlib.util.spec_from_file_location('r3c_test_cli',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def temporary():
    base=ROOT/'results/r3/c';base.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(prefix='network_one_shot_TEST_FIXTURE_ONLY_',dir=base)


class Contracts(unittest.TestCase):
    def test_help_dry_run_no_overwrite_no_heavy_dependencies(self):
        with temporary() as tmp:
            output=Path(tmp)/'dry';command=[sys.executable,str(SCRIPT),'--phase','dry-run','--output-dir',str(output)]
            r=subprocess.run(command,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
            report=read(output/'r3c_one_shot_results.json')
            self.assertEqual(report['gpu_findings_status'],'NOT_RUN');self.assertFalse(report['actual_network_socket_used'])
            hashes={p.name:sha(p) for p in output.iterdir()}
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(hashes,{p.name:sha(p) for p in output.iterdir()})
        self.assertEqual(subprocess.run([sys.executable,str(SCRIPT),'--help'],capture_output=True).returncode,0)
        code="import semcache.edgelora.r3c_runtime; import sys; assert not any(n in sys.modules for n in ('torch','numpy','transformers','peft'))"
        r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')))
        self.assertEqual(r.returncode,0,r.stderr)

    def test_missing_allocation_stops_before_listeners_or_model(self):
        with temporary() as tmp:
            env=dict(os.environ);env.pop('SLURM_JOB_ID',None)
            r=subprocess.run([sys.executable,str(SCRIPT),'--phase','run','--enable-gpu-experiment','--output-dir',tmp],
                capture_output=True,text=True,env=env)
            self.assertNotEqual(r.returncode,0);report=read(Path(tmp)/'r3c_one_shot_results.json')
            self.assertEqual(report['execution_status'],'FAILED');self.assertFalse(report['gpu_inference_performed'])
            self.assertFalse((Path(tmp)/'proxy').exists());self.assertFalse((Path(tmp)/'prepare').exists())

    def test_forced_worker_cleanup_closes_listener_and_pid(self):
        with temporary() as tmp:
            output=Path(tmp);child=spawn('echo',dict(fixture=True,sources=cli().sources()),output)
            try:info=ready(child,output/'echo_ready.json')
            finally:life=reap(child,output/'forced_lifecycle.json')
            self.assertTrue(life['forced_termination']);self.assertTrue(life['all_children_exited'])
            with self.assertRaises(ProcessLookupError):os.kill(info['pid'],0)
            with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
                probe.settimeout(.2);self.assertNotEqual(probe.connect_ex(('127.0.0.1',info['port'])),0)

    def test_invalid_calibration_stops_before_prepare_and_preserves_raw_failure(self):
        with temporary() as tmp:
            module=cli();invalid=[dict(condition=dict(id='C00',rtt_ms=0,es_to_ud_mbps=None,ud_to_es_mbps=None),
                status='INVALID_IMPAIRMENT')]
            with patch('semcache.edgelora.r3c_calibration.validate_calibration',return_value=invalid):
                status=module.main(['--phase','fixture','--test-fixture-only','--rtt-ms','0',
                    '--bandwidth-mbps','unlimited','--hit-fractions','0,1','--output-dir',tmp])
            self.assertEqual(status,1);output=Path(tmp);report=read(output/'r3c_one_shot_results.json')
            self.assertEqual(report['execution_status'],'FAILED');self.assertIn('INVALID_IMPAIRMENT',report['reason'])
            self.assertFalse((output/'prepare').exists());self.assertFalse((output/'es').exists())
            self.assertFalse(report['gpu_inference_performed']);self.assertTrue((output/'calibration/calibration_probes.jsonl').stat().st_size)
            self.assertEqual(read(output/'r3c_break_even_decision.json')['status'],'NOT_ESTIMABLE')
            self.assertTrue(read(output/'proxy/proxy_lifecycle.json')['threads_stopped'])
            for kind,directory in (('echo','calibration'),('proxy','proxy')):
                pid=read(output/directory/(kind+'_parent_lifecycle.json'))['pid']
                with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_synthetic_native_failure_is_separate_from_real_socket_cache_equality(self):
        from semcache.edgelora.r3a_tiny import prepare_tiny
        from semcache.edgelora.r3a_artifacts import pack,unpack
        def synthetic_reference(output):
            prepared=prepare_tiny(output);ref=read(prepared/'r3a_reference.json')
            data=prepared/ref['logits_file'];values=unpack(data.read_bytes(),'float32');values[0]+=1
            data.write_bytes(pack(values,'float32'));ref['logits_sha256']=sha(data);write(prepared/'r3a_reference.json',ref)
            pm=read(prepared/'r3a_prepare_manifest.json')
            pm['artifact_hashes'][ref['logits_file']]=sha(data)
            pm['artifact_hashes']['r3a_reference.json']=sha(prepared/'r3a_reference.json')
            write(prepared/'r3a_prepare_manifest.json',pm);return prepared
        with temporary() as tmp:
            with patch('semcache.edgelora.r3a_tiny.prepare_tiny',side_effect=synthetic_reference):
                result=cli().main(['--phase','fixture','--test-fixture-only','--rtt-ms','0',
                    '--bandwidth-mbps','unlimited','--hit-fractions','0,1','--output-dir',tmp])
            self.assertEqual(result,0);output=Path(tmp);report,_=cli().load_run(output)
            self.assertTrue(report['TEST_FIXTURE_ONLY']);self.assertFalse(report['native_peft_strict_parity_passed'])
            self.assertTrue(all(r['cache_bitwise_passed'] and not r['native_parity']['passed'] for r in jsonl(output/'r3c_requests.jsonl')))
            self.assertFalse(report['PRIMARY_PERFORMANCE_ELIGIBLE']);self.assertEqual(report['audit']['status'],'PASS')


class PhysicalGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=temporary();cls.addClassCleanup(cls.temp.cleanup);cls.output=Path(cls.temp.name)/'fixture'
        cls.run_result=subprocess.run([sys.executable,'-u',str(SCRIPT),'--phase','fixture','--test-fixture-only',
            '--rtt-ms','0,10','--bandwidth-mbps','unlimited','--hit-fractions','0,1','--output-dir',str(cls.output)],
            capture_output=True,text=True,timeout=150.)
        if cls.run_result.returncode:raise AssertionError(cls.run_result.stdout+cls.run_result.stderr)

    def test_persistent_three_processes_actual_grid_all96_and_zero_hit_is_real_cache_miss(self):
        report,plan=cli().load_run(self.output);rows=jsonl(self.output/'r3c_requests.jsonl')
        self.assertTrue(report['TEST_FIXTURE_ONLY']);self.assertFalse(report['gpu_inference_performed'])
        self.assertEqual(len(rows),60);self.assertEqual(report['audit']['rows'],60)
        self.assertEqual({r['proxy_pid'] for r in rows},{report['model_result']['proxy_pid']})
        self.assertEqual(report['model_result']['proxy_pid'],plan['calibration']['proxy_pid'])
        for row in rows:
            self.assertEqual(len(row['qkv_sha256']),96);self.assertTrue(row['cache_bitwise_passed'])
            self.assertEqual(len({row['es_pid'],row['ud_pid'],row['proxy_pid']}),3)
            if row['mode']=='WARM_HIT':
                self.assertEqual(row['application_wire_bytes'],0);self.assertEqual(row['ud_cpu_lora_ms'],0.)
                self.assertEqual(row['reused_qkv_outputs'],96)
            else:
                self.assertEqual(row['hidden_frames'],32);self.assertEqual(row['delta_frames'],32)
                self.assertEqual(len(row['proxy_frames']),64)
            if row['arm']=='CACHE_00_OF_8':
                self.assertEqual(row['mode'],'COLD_MISS');self.assertTrue(row['cache_admitted'])
                self.assertGreater(row['cache_insert_ms'],0.)
        for relative in ('es/r3a_ud_ready.json','proxy/proxy_ready.json'):
            info=read(self.output/relative)
            with self.assertRaises(ProcessLookupError):os.kill(info['pid'],0)
            for field in ('port','data_port','control_port'):
                if field in info:
                    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
                        probe.settimeout(.2);self.assertNotEqual(probe.connect_ex(('127.0.0.1',info[field])),0)
        self.assertTrue(read(self.output/'es/lifecycle.json')['tracked_descendants']['all_stopped'])
        self.assertTrue(read(self.output/'proxy/proxy_lifecycle.json')['threads_stopped'])

    def test_read_only_reanalysis_exact_contrasts_and_sha_tamper_rejection(self):
        with temporary() as tmp:
            r=subprocess.run([sys.executable,str(SCRIPT),'--phase','analyze','--inputs',str(self.output),'--output-dir',tmp],
                capture_output=True,text=True,timeout=30.)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            self.assertEqual(read(Path(tmp)/'r3c_break_even_decision.json'),read(self.output/'r3c_break_even_decision.json'))
        with temporary() as tmp:
            copy=Path(tmp)/'copy';shutil.copytree(self.output,copy)
            (copy/'r3c_requests.jsonl').write_text('CORRUPTED\n')
            with self.assertRaises(ValueError):cli().load_run(copy)

    def test_independent_audit_rejects_counts_hashes_gates_and_wrong_calibration(self):
        original_rows=jsonl(self.output/'es/r3c_requests.jsonl')
        plan=read(self.output/'es/es_plan.json');result=read(self.output/'es/es_result.json')
        with temporary() as tmp:
            copy=Path(tmp)/'copy';shutil.copytree(self.output,copy)
            for field,value in (('delta_frames',31),('ud_cpu_lora_ms',-1),('logits_sha256','0'*64),
                                ('native_parity',dict(passed=False)),('impairment_status','INVALID_IMPAIRMENT')):
                rows=json.loads(json.dumps(original_rows));rows[0][field]=value
                (copy/'es/r3c_requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.subTest(field=field),self.assertRaises((ValueError,KeyError)):audit_model(copy,plan,result)
            (copy/'es/r3c_requests.jsonl').write_bytes((self.output/'es/r3c_requests.jsonl').read_bytes())
            bad=json.loads(json.dumps(plan));bad['calibration']['conditions'][0]['status']='INVALID_IMPAIRMENT'
            with self.assertRaises(ValueError):audit_model(copy,bad,result)
            bad=json.loads(json.dumps(plan));bad['reference_identity']['gpu_uuid']='WRONG_GPU'
            with self.assertRaises(ValueError):audit_model(copy,bad,result)
            (copy/'es/r3a_ud_wire_trace.jsonl').unlink()
            with self.assertRaises((ValueError,FileNotFoundError)):audit_model(copy,plan,result)


@unittest.skipUnless(os.environ.get('SEMCACHE_R3C_GPU')=='1',
    'OPT-2.7B R3-C GPU NOT_RUN; explicit Slurm allocation and opt-in required')
class GPUOptIn(unittest.TestCase):
    def test_gpu_one_allocation(self):
        output=Path(os.environ['SEMCACHE_R3C_OUTPUT'])
        r=subprocess.run([sys.executable,str(SCRIPT),'--phase','run','--enable-gpu-experiment','--output-dir',str(output)],
            capture_output=True,text=True,timeout=10000.)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        report,_=cli().load_run(output);self.assertFalse(report['TEST_FIXTURE_ONLY'])
        self.assertTrue(report['gpu_inference_performed'])


if __name__=='__main__':unittest.main()
