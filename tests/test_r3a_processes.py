"""Independent OS process + real localhost TCP integration, stdlib tiny model."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

from semcache.edgelora.r3a_artifacts import ROOT,pack,read,sha,unpack,write
from semcache.edgelora.r3a_evidence import jsonl,validate_run
from semcache.edgelora.r3a_tiny import prepare_tiny

SCRIPT=ROOT/'scripts/95_run_r3a_distributed_no_reuse.py'


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.prepared=prepare_tiny(self.root/'prepare')

    def run_phase(self,phase,name,extra=()):
        out=self.root/name
        cmd=[sys.executable,'-u',str(SCRIPT),'--phase',phase,'--prepared',str(self.prepared),'--output-dir',str(out),*extra]
        if phase!='analyze':cmd+=['--device','cpu']
        run=subprocess.run(cmd,capture_output=True,text=True,timeout=40.)
        return run,out

    def assert_cleanup(self,out):
        lifecycle=read(out/'r3a_lifecycle.json')
        self.assertTrue(lifecycle['all_children_exited'])
        for child in lifecycle['children']:
            with self.assertRaises(ProcessLookupError):os.kill(child['pid'],0)
        port=read(out/'r3a_ud_ready.json')['port']
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.settimeout(.2)
            self.assertNotEqual(s.connect_ex(('127.0.0.1',port)),0)

    def test_full_tiny_pipeline_true_socket_pids_counts_hashes_and_analysis(self):
        run,pilot=self.run_phase('pilot','pilot')
        self.assertEqual(run.returncode,0,run.stderr)
        validated=validate_run(pilot,self.prepared,phase='pilot')
        env=validated['environment'];self.assertNotEqual(env['es']['pid'],env['ud']['pid'])
        self.assertNotIn(os.getpid(),(env['es']['pid'],env['ud']['pid']))
        self.assertFalse(env['ud']['user_device_cuda_used']);self.assertFalse(env['ud']['full_model_loaded'])
        self.assertEqual(env['es']['adapter_parameters_on_es'],0)
        self.assertEqual(len(jsonl(pilot/'r3a_wire_trace.jsonl')),4*32*2+4)
        for observation in validated['observations']:
            row=observation['row'];self.assertEqual(row['es_to_ud_frames'],32);self.assertEqual(row['ud_to_es_frames'],32)
            self.assertEqual(row['qkv_outputs_observed'],96)
            self.assertEqual(row['es_to_ud_payload_bytes'],32*4*4*4)
            self.assertEqual(row['ud_to_es_payload_bytes'],3*32*4*4*4)
            self.assertGreater(row['es_to_ud_wire_bytes'],row['es_to_ud_payload_bytes'])
        self.assert_cleanup(pilot)
        run,latency=self.run_phase('latency','latency',['--pilot-inputs',str(pilot)])
        self.assertEqual(run.returncode,0,run.stderr);self.assert_cleanup(latency)
        run,aggregate=self.run_phase('analyze','aggregate',['--pilot-inputs',str(pilot),'--inputs',str(latency)])
        self.assertEqual(run.returncode,0,run.stderr)
        d=read(aggregate/'r3a_decision.json')
        self.assertEqual(d['execution_status'],'COMPLETE');self.assertTrue(d['parity_passed'])
        self.assertTrue(d['actual_network_socket_used']);self.assertTrue(d['TEST_FIXTURE_ONLY'])
        self.assertFalse(d['gpu_inference_performed']);self.assertFalse(d['r2d_break_even_validated'])
        before={p.name:sha(p) for p in aggregate.iterdir()}
        rerun,_=self.run_phase('analyze','aggregate',['--pilot-inputs',str(pilot),'--inputs',str(latency)])
        self.assertNotEqual(rerun.returncode,0);self.assertEqual(before,{p.name:sha(p) for p in aggregate.iterdir()})

    def test_parity_failure_stops_on_first_warmup_and_cleans_all_workers(self):
        reference_path=self.prepared/'r3a_reference.json';reference=read(reference_path)
        logits=self.prepared/reference['logits_file'];values=unpack(logits.read_bytes(),'float32');values[0]+=1.
        logits.write_bytes(pack(values,'float32'));reference['logits_sha256']=sha(logits);write(reference_path,reference)
        path=self.prepared/'r3a_prepare_manifest.json';m=read(path)
        m['artifact_hashes'][logits.name]=sha(logits);m['artifact_hashes'][reference_path.name]=sha(reference_path);write(path,m)
        run,out=self.run_phase('pilot','failed_parity')
        self.assertNotEqual(run.returncode,0)
        self.assertEqual(read(out/'r3a_manifest.json')['execution_status'],'FAILED')
        d=read(out/'r3a_decision.json');self.assertFalse(d['parity_passed'])
        self.assertIn('PARITY_FAIL',(out/'r3a_es.stderr.log').read_text())
        records=jsonl(out/'r3a_iterations.jsonl')
        self.assertEqual(len(records),1);self.assertEqual(records[0]['row']['phase'],'warmup')
        self.assertFalse(records[0]['parity']['passed']);self.assert_cleanup(out)

    def test_failed_missing_or_stale_pilot_rejected_before_listener_start(self):
        run,out=self.run_phase('latency','no_pilot',['--pilot-inputs',str(self.root/'missing')])
        self.assertNotEqual(run.returncode,0);self.assertFalse((out/'r3a_ud_ready.json').exists())
        self.assertEqual(read(out/'r3a_manifest.json')['execution_status'],'FAILED')
        run,pilot=self.run_phase('pilot','pilot');self.assertEqual(run.returncode,0,run.stderr)
        m=read(pilot/'r3a_manifest.json');m['identity_sha256']='0'*64;write(pilot/'r3a_manifest.json',m)
        run,out=self.run_phase('latency','stale_pilot',['--pilot-inputs',str(pilot)])
        self.assertNotEqual(run.returncode,0);self.assertFalse((out/'r3a_ud_ready.json').exists())

    def test_raw_trace_tampering_is_rejected_by_analyzer(self):
        run,pilot=self.run_phase('pilot','pilot');self.assertEqual(run.returncode,0,run.stderr)
        trace=pilot/'r3a_wire_trace.jsonl';trace.write_bytes(trace.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'SHA'):validate_run(pilot,self.prepared)

    def test_production_prepare_fails_usefully_without_runtime_dependencies(self):
        import importlib.util
        if all(importlib.util.find_spec(n) for n in ('torch','transformers','peft')):
            self.skipTest('runtime installed; GPU preparation is explicitly opt-in')
        out=self.root/'not_executed_gpu'
        run=subprocess.run([sys.executable,str(SCRIPT),'--phase','prepare','--model','facebook/opt-2.7b',
            '--dtype','float16','--prompt-lengths','32','--device','cuda:0','--adapter','user_a','--seed','42',
            '--output-dir',str(out)],capture_output=True,text=True)
        self.assertNotEqual(run.returncode,0);self.assertIn('missing local runtime dependencies',run.stderr)
        for name in ('r3a_prepare_manifest.json','r3a_reference.json','r3a_adapter_manifest.json','r3a_preflight.json'):
            self.assertEqual(read(out/name)['execution_status'],'FAILED')
        self.assertFalse(read(out/'r3a_prepare_manifest.json')['gpu_inference_performed'])


@unittest.skipUnless(os.environ.get('SEMCACHE_R3A_GPU_PILOT')=='1','OPT-2.7B GPU pilot NOT_RUN; requires explicit allocation/opt-in')
class AllocatedGPUIntegration(unittest.TestCase):
    def test_real_opt27b_prepare_and_physical_pilot(self):
        root=Path(os.environ['SEMCACHE_R3A_GPU_OUTPUT']).resolve()
        self.assertFalse(root.exists(),'choose a new GPU test output path; failed traces must be retained')
        for phase,extra in (('prepare',['--model','facebook/opt-2.7b','--dtype','float16','--prompt-lengths','32',
            '--device','cuda:0','--adapter','user_a','--seed','42']),
            ('pilot',['--prepared',str(root/'prepare'),'--device','cuda:0','--warmup-runs','1','--measured-runs','3'])):
            run=subprocess.run([sys.executable,'-u',str(SCRIPT),'--phase',phase,*extra,'--output-dir',str(root/phase)],timeout=1200.)
            self.assertEqual(run.returncode,0,f'{phase} failed; retained evidence: {root/phase}')
        evidence=validate_run(root/'pilot',root/'prepare',phase='pilot')
        self.assertTrue(evidence['manifest']['gpu_inference_performed'])


if __name__=='__main__':unittest.main()
