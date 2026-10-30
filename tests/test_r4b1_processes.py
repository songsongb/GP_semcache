"""Physical independent-process CPU fixture and adversarial B1 audits."""
from copy import deepcopy
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

from semcache.edgelora.communication import ProtocolError
from semcache.edgelora.r3a_artifacts import ROOT,read,sha
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.r4b1_evidence import audit_run,load_run,replay_events

SCRIPT=ROOT/'scripts/103_run_r4b1_multi_entry.py'


def temporary():
    base=ROOT/'results/r4/b1';base.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(prefix='TEST_FIXTURE_ONLY_',dir=base)


class Contracts(unittest.TestCase):
    def test_help_lazy_import_dry_run_no_overwrite_and_no_gpu_fallback(self):
        code="import semcache.edgelora.r4b1_runtime; import sys; assert not any(n in sys.modules for n in ('torch','transformers','peft','numpy','sacrebleu'))"
        r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')))
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(subprocess.run([sys.executable,str(SCRIPT),'--help'],capture_output=True).returncode,0)
        with temporary() as tmp:
            command=[sys.executable,str(SCRIPT),'--phase','dry-run','--output-dir',tmp]
            r=subprocess.run(command,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
            before={p.name:sha(p) for p in Path(tmp).iterdir()}
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(before,{p.name:sha(p) for p in Path(tmp).iterdir()})
            env=dict(os.environ);env.pop('SLURM_JOB_ID',None)
            r=subprocess.run([sys.executable,str(SCRIPT),'--phase','run','--enable-gpu-experiment','--prepared',str(Path(tmp)/'absent'),
                '--output-dir',str(Path(tmp)/'failed')],env=env,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);report=read(Path(tmp)/'failed/r4b1_results.json')
            self.assertFalse(report['decision']['R4B1_PHYSICAL_E2E_PASS']);self.assertFalse((Path(tmp)/'failed/es').exists())

    def test_branch_preflight_and_dependency_gate_preserve_r4a_branch_contract(self):
        spec=importlib.util.spec_from_file_location('r4b1_test_cli',SCRIPT);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        env=dict(SLURM_JOB_ID='TEST_FIXTURE_ONLY',CUDA_VISIBLE_DEVICES='0')
        with patch.dict(os.environ,env),patch.object(module,'source_identity',return_value={'git_branch':'exp/raw-semcache-r4'}):
            with self.assertRaisesRegex(ProtocolError,'requires exp/raw-semcache-r4b'):module.gpu_preflight()
        with patch.dict(os.environ,env),patch.object(module,'source_identity',return_value={'git_branch':'exp/raw-semcache-r4b'}), \
                patch.object(module.importlib.util,'find_spec',return_value=None):
            with self.assertRaisesRegex(ProtocolError,'MISSING_DEPENDENCY'):module.gpu_preflight()


class PhysicalMultiEntry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=temporary();cls.addClassCleanup(cls.temp.cleanup);cls.output=Path(cls.temp.name)/'physical'
        r=subprocess.run([sys.executable,str(SCRIPT),'--phase','fixture','--test-fixture-only','--output-dir',str(cls.output)],
            capture_output=True,text=True,timeout=180.)
        if r.returncode:
            child=cls.output/'es/es_result.json'
            raise AssertionError(r.stdout+r.stderr+(child.read_text() if child.exists() else ''))

    def test_a_b_a_residency_bitwise_evidence_pressure_and_all_clocks(self):
        report,plan,worker=load_run(self.output);rows=jsonl(self.output/'es/r4b1_requests.jsonl');tokens=jsonl(self.output/'es/r4b1_tokens.jsonl')
        self.assertEqual(len(rows),24);self.assertEqual(len(tokens),72);self.assertEqual(report['audit']['fidelity_comparisons'],18)
        a,b,hit=rows[2:5];self.assertEqual([r['mode'] for r in (a,b,hit)],['COLD_MISS','COLD_MISS','WARM_HIT'])
        self.assertNotEqual(a['sample_id'],b['sample_id']);self.assertEqual(a['sample_id'],hit['sample_id'])
        self.assertEqual(hit['cache_source_request_id'],a['request_id']);self.assertEqual(b['cache_entries_after'],2)
        self.assertEqual(hit['cache_entries_after'],2);self.assertEqual(report['audit']['cache']['evictions'],3)
        refusal=read(self.output/'es/r4b1_hit_refusal.json');self.assertEqual(refusal['new_wire_frames'],0);self.assertFalse(refusal['recomputed'])
        decision=read(self.output/'r4b1_decision.json')
        for key in ('R4B1_PHYSICAL_E2E_PASS','R4B1_MULTI_ENTRY_RESIDENCY_PASS','R4B1_RAW_QKV_E2E_FIDELITY_PASS',
                'NATIVE_PEFT_STRICT_PARITY_PASS','PRIMARY_PERFORMANCE_ELIGIBLE','correctness_qualified_speedup'):self.assertFalse(decision[key])
        for row in rows:
            steps=[s for s in tokens if s['request_id']==row['request_id']]
            self.assertEqual([s['context']['input_length'] for s in steps],[32,1,1]);self.assertEqual([s['past']['sequence_length'] for s in steps],[32,33,34])
            self.assertEqual(row['e2e_wall_ms'],(row['request_end_monotonic_ns']-row['request_start_monotonic_ns'])/1e6)
            self.assertEqual(row['cache_bytes_after'],row['cache_after']['occupancy_bytes'])
            self.assertGreater(row['decode_accounting']['application_wire_bytes'],0)
            if row['mode']=='WARM_HIT':self.assertEqual(row['prefill_accounting']['application_wire_bytes'],0)
            else:self.assertGreater(row['prefill_accounting']['application_wire_bytes'],0)
            if row['mode']=='NO_REUSE':self.assertEqual(row['cache_before'],row['cache_after']);self.assertEqual(row['cache_events'],[])
        self.assertEqual(read(self.output/'r4b1_quality.json')['bleu_vs_ground_truth']['status'],'NOT_ESTIMABLE_NO_GOLD_REFERENCES')
        for pid in (worker['pid'],worker['ud_pid']):
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)
        info=read(self.output/'es/r4a_ud_ready.json')
        with socket.socket() as probe:
            probe.settimeout(.2);self.assertNotEqual(probe.connect_ex(('127.0.0.1',info['port'])),0)

    def test_readonly_analysis_repeats_decision_and_preserves_all_input_bytes(self):
        before={str(p.relative_to(self.output)):sha(p) for p in self.output.rglob('*') if p.is_file()}
        with temporary() as tmp:
            r=subprocess.run([sys.executable,str(SCRIPT),'--phase','analyze','--inputs',str(self.output),'--output-dir',tmp],
                capture_output=True,text=True,timeout=60.)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            self.assertEqual(read(Path(tmp)/'r4b1_decision.json'),read(self.output/'r4b1_decision.json'))
        self.assertEqual(before,{str(p.relative_to(self.output)):sha(p) for p in self.output.rglob('*') if p.is_file()})

    def test_capacity_lru_eviction_source_and_occupancy_forgery_rejected(self):
        report,plan,worker=load_run(self.output);events=jsonl(self.output/'es/r4b1_cache_events.jsonl')
        rows=jsonl(self.output/'es/r4b1_requests.jsonl');contexts=read(self.output/'es/r4b1_contexts.json')
        for field,value in (('occupancy_bytes',1),('released_bytes',1),('source_request_id','FOREIGN'),('context_sha256','0'*64)):
            altered=deepcopy(events);e=next(e for e in altered if e['kind']=='EVICT');e[field]=value
            with self.subTest(field=field),self.assertRaises(ProtocolError):replay_events(altered,worker['capacities'],contexts,rows)
        altered=deepcopy(events);e=next(e for e in altered if e['kind']=='HIT');e['source_request_id']='FOREIGN'
        with self.assertRaises(ProtocolError):replay_events(altered,worker['capacities'],contexts,rows)

    def test_rehashed_logit_decode_and_cache_event_tampering_fails_independent_audit(self):
        _,plan,worker=load_run(self.output)
        with temporary() as tmp:
            target=Path(tmp)/'copy';shutil.copytree(self.output,target)
            p=target/'es/r4b1_tokens.jsonl';rows=jsonl(p);rows[1]['positions']=[99]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ProtocolError):audit_run(target,plan,worker)
            shutil.copyfile(self.output/'es/r4b1_tokens.jsonl',p)
            p=target/'es/r4b1_cache_events.jsonl';events=jsonl(p);next(e for e in events if e['kind']=='HIT')['source_request_id']='FOREIGN'
            p.write_text(''.join(json.dumps(e)+'\n' for e in events))
            with self.assertRaises(ProtocolError):audit_run(target,plan,worker)
            with self.assertRaises(ProtocolError):load_run(target)


@unittest.skipUnless(os.environ.get('SEMCACHE_R4B1_GPU')=='1','RTX3090 B1 GPU NOT_RUN; explicit allocation/opt-in required')
class AllocatedGPU(unittest.TestCase):
    def test_actual_multi_entry_gpu_pilot(self):
        base=Path(os.environ['SEMCACHE_R4B1_BASE_PREPARED']);root=Path(os.environ['SEMCACHE_R4B1_OUTPUT'])
        commands=[['--phase','prepare','--enable-gpu-experiment','--base-prepared',str(base),'--output-dir',str(root/'prepare')],
            ['--phase','run','--enable-gpu-experiment','--prepared',str(root/'prepare/native'),'--output-dir',str(root/'run')],
            ['--phase','analyze','--inputs',str(root/'run'),'--output-dir',str(root/'analysis')]]
        for args in commands:
            r=subprocess.run([sys.executable,str(SCRIPT),*args],capture_output=True,text=True,timeout=8000.)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        report,_,_=load_run(root/'run');self.assertTrue(report['decision']['R4B1_MULTI_ENTRY_RESIDENCY_PASS'])
        self.assertFalse(report['decision']['correctness_qualified_speedup'])


if __name__=='__main__':unittest.main()
