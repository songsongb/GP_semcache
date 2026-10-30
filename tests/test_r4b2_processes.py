"""Physical localhost B2 CPU feasibility, independent audits and strict failures."""
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

from semcache.edgelora.communication import ProtocolError,digest_bytes
from semcache.edgelora.r3a_artifacts import ROOT,THRESHOLDS,pack,read,sha,unpack
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.r4b2_analysis import analyze
from semcache.edgelora.r4b2_evidence import audit_run,load_run
from semcache.edgelora.r4b2_prefix_runtime import load_inputs

SCRIPT=ROOT/'scripts/104_run_r4b2_shared_prefix.py'


def temporary():
    base=ROOT/'results/r4/b2';base.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(prefix='TEST_FIXTURE_ONLY_',dir=base)


class Contracts(unittest.TestCase):
    def test_lazy_help_dry_run_existing_output_refusal_and_required_gpu_optin(self):
        code="import semcache.edgelora.r4b2_prefix_runtime; import sys; assert not any(n in sys.modules for n in ('torch','transformers','peft','numpy','sacrebleu'))"
        r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')))
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(subprocess.run([sys.executable,str(SCRIPT),'--help'],capture_output=True).returncode,0)
        with temporary() as tmp:
            command=[sys.executable,str(SCRIPT),'--phase','dry-run','--output-dir',tmp]
            self.assertEqual(subprocess.run(command,capture_output=True).returncode,0)
            before={p.name:sha(p) for p in Path(tmp).iterdir()}
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(before,{p.name:sha(p) for p in Path(tmp).iterdir()})
            env=dict(os.environ);env.pop('R4B2_ENABLE_GPU',None)
            r=subprocess.run([sys.executable,str(SCRIPT),'--phase','run','--enable-gpu-experiment','--prepared',str(Path(tmp)/'absent'),
                '--output-dir',str(Path(tmp)/'failed')],env=env,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);report=read(Path(tmp)/'failed/r4b2_results.json')
            self.assertIn('R4B2_ENABLE_GPU=1',report['reason']);self.assertFalse((Path(tmp)/'failed/es').exists())

    def test_new_branch_gate_is_separate_and_missing_dependencies_never_fall_back(self):
        spec=importlib.util.spec_from_file_location('b2_test_cli',SCRIPT);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        env=dict(R4B2_ENABLE_GPU='1',SLURM_JOB_ID='TEST_FIXTURE_ONLY',CUDA_VISIBLE_DEVICES='0')
        with patch.dict(os.environ,env),patch.object(module,'source_identity',return_value={'git_branch':'exp/raw-semcache-r4b'}):
            with self.assertRaisesRegex(ProtocolError,'requires exp/raw-semcache-r4b2'):module.gpu_preflight()
        with patch.dict(os.environ,env),patch.object(module,'source_identity',return_value={'git_branch':'exp/raw-semcache-r4b2'}), \
                patch.object(module.importlib.util,'find_spec',return_value=None):
            with self.assertRaisesRegex(ProtocolError,'MISSING_DEPENDENCY'):module.gpu_preflight()


class PhysicalPrefix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=temporary();cls.addClassCleanup(cls.temp.cleanup);cls.output=Path(cls.temp.name)/'physical'
        r=subprocess.run([sys.executable,str(SCRIPT),'--phase','fixture','--test-fixture-only','--output-dir',str(cls.output)],
            capture_output=True,text=True,timeout=240.)
        if r.returncode:
            p=cls.output/'es/es_result.json';raise AssertionError(r.stdout+r.stderr+(p.read_text() if p.exists() else ''))

    def test_actual_p_cold_a_hit_b_hit_and_reuse_after_unrelated_no_hidden_prefill(self):
        report,plan,worker=load_run(self.output);rows=jsonl(self.output/'es/r4b2_requests.jsonl');tokens=jsonl(self.output/'es/r4b2_tokens.jsonl')
        self.assertEqual(len(rows),24);self.assertEqual(len(tokens),72);self.assertEqual(report['audit']['partial_hits'],9)
        self.assertEqual(report['audit']['standalone_prefix_sources'],3);self.assertEqual(report['audit']['negative_controls'],13)
        for offset in (0,8,16):
            cold,a,b,u,again=rows[offset+3:offset+8]
            self.assertEqual([r['mode'] for r in (cold,a,b,u,again)],['COLD_MISS','PARTIAL_PREFIX_HIT','PARTIAL_PREFIX_HIT','NO_REUSE','PARTIAL_PREFIX_HIT'])
            self.assertNotEqual(a['target_token_ids'],b['target_token_ids'])
            self.assertEqual(a['source'],b['source']);self.assertEqual(a['source'],again['source'])
            self.assertEqual(a['source']['source_request_id'],cold['request_id'])
            prefix=cold['prompt_tokens'];self.assertIn(prefix,(32,64,96))
            for row in (a,b,again):
                ss=[s for s in tokens if s['request_id']==row['request_id']]
                self.assertEqual([s['context']['input_length'] for s in ss],[128-prefix,1,1])
                self.assertEqual([s['context']['past_length'] for s in ss],[prefix,128,129])
                self.assertEqual([s['past']['sequence_length'] for s in ss],[128,129,130])
                self.assertEqual(row['prefix_projection_token_rows_skipped'],prefix*32*3)
                self.assertEqual(row['suffix_projection_token_rows_executed'],(128-prefix)*32*3)
                self.assertFalse(row['prefix_q_consumed']);self.assertEqual(row['cached_derived_kv_bytes'],0)
                self.assertEqual(row['request_initial_prefix_kv_bytes'],prefix*32*4*4*2)
                self.assertGreater(row['cache_lookup_ms'],0)
                ref=next(r for r in rows if r['request_id']==row['reference_request_id'])
                self.assertEqual(ref['prefill_accounting']['hidden_payload_bytes']-row['prefill_accounting']['hidden_payload_bytes'],prefix*32*4*4)
                self.assertEqual(ref['prefill_accounting']['delta_payload_bytes']-row['prefill_accounting']['delta_payload_bytes'],prefix*32*4*4*3)
        decision=read(self.output/'r4b2_decision.json');self.assertTrue(decision['fixture_checks_passed']);self.assertTrue(decision['fixture_strict_bitwise'])
        for key,value in decision.items():
            if key.endswith('_PASS') or key in ('PRIMARY_PERFORMANCE_ELIGIBLE','correctness_qualified_speedup','R4B2_TTFT_BENEFIT_OBSERVED','R4B2_E2E_BENEFIT_OBSERVED'):
                self.assertFalse(value,key)
        checks=read(self.output/'r4b2_fidelity.json');self.assertEqual(len(checks),12);self.assertTrue(all(c['strict_bitwise_pass'] for c in checks))
        negatives=read(self.output/'es/r4b2_negative_controls.json');self.assertTrue(all(n['data_frames']==0 and not n['recomputed'] for n in negatives))
        reuse=[e for e in jsonl(self.output/'es/r4b2_cache_events.jsonl') if e['kind']=='PHYSICAL_REUSE']
        self.assertTrue(all(e['raw_kv_projections_consumed']==64 and not e['cached_q_consumed'] and 'qkv_outputs' not in e for e in reuse))
        self.assertEqual(len(jsonl(self.output/'es/r4b2_failed_requests.jsonl')),12)
        for pid in (worker['pid'],worker['ud_pid']):
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)
        info=read(self.output/'es/r4b2_ud_ready.json')
        with socket.socket() as probe:
            probe.settimeout(.2);self.assertNotEqual(probe.connect_ex(('127.0.0.1',info['port'])),0)

    def test_readonly_analysis_same_decision_and_all_input_bytes_unchanged(self):
        before={str(p.relative_to(self.output)):sha(p) for p in self.output.rglob('*') if p.is_file()}
        with temporary() as tmp:
            r=subprocess.run([sys.executable,str(SCRIPT),'--phase','analyze','--inputs',str(self.output),'--output-dir',tmp],
                capture_output=True,text=True,timeout=120.)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            self.assertEqual(read(Path(tmp)/'r4b2_decision.json'),read(self.output/'r4b2_decision.json'))
        self.assertEqual(before,{str(p.relative_to(self.output)):sha(p) for p in self.output.rglob('*') if p.is_file()})

    def test_rehashed_initial_past_cache_events_and_raw_kv_corruption_fail_independent_audit(self):
        _,plan,worker=load_run(self.output)
        with temporary() as tmp:
            target=Path(tmp)/'copy';shutil.copytree(self.output,target);p=target/'es/r4b2_tokens.jsonl'
            rows=jsonl(p);next(s for s in rows if s['context']['phase']=='SUFFIX_PREFILL')['context']['past_length']=0
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ProtocolError):audit_run(target,plan,worker)
            shutil.copyfile(self.output/'es/r4b2_tokens.jsonl',p)
            p=target/'es/r4b2_cache_events.jsonl';events=jsonl(p);next(e for e in events if e['kind']=='HIT')['source_request_id']='FOREIGN'
            p.write_text(''.join(json.dumps(e)+'\n' for e in events))
            with self.assertRaises(ProtocolError):audit_run(target,plan,worker)
            with self.assertRaises(ProtocolError):load_run(target)
            shutil.copyfile(self.output/'es/r4b2_cache_events.jsonl',p)
            p=target/'es/r4b2_tokens.jsonl';records=jsonl(p);source=next(s for s in records if s['step']==0 and s['context']['mode']=='COLD_MISS')
            spec=source['kv'][0]['k'];tensor=target/'es'/spec['file'];vals=unpack(tensor.read_bytes(),spec['dtype']);vals[0]+=.01
            blob=pack(vals,spec['dtype']);tensor.write_bytes(blob);spec['sha256']=digest_bytes(blob)
            p.write_text(''.join(json.dumps(s)+'\n' for s in records))
            with self.assertRaisesRegex(ProtocolError,'Raw K/V transform'):audit_run(target,plan,worker)

    def test_numerical_deviation_cannot_be_promoted_to_strict_pass_even_when_tokens_match(self):
        _,plan,worker=load_run(self.output);_,suite,*_=load_inputs(plan['prepared'])
        with temporary() as tmp:
            target=Path(tmp)/'copy';shutil.copytree(self.output/'es',target);rows=jsonl(target/'r4b2_requests.jsonl');tokens=jsonl(target/'r4b2_tokens.jsonl')
            candidate=next(r for r in rows if r['mode']=='PARTIAL_PREFIX_HIT');p=target/candidate['logits_file'];values=unpack(p.read_bytes(),'float32')
            values[0]+=.00001;blob=pack(values,'float32');p.write_bytes(blob);candidate['logits_sha256']=digest_bytes(blob)
            decision,checks,_,_=analyze(target,suite,rows,tokens,fixture=True,audit={'status':'PASS'})
            changed=next(c for c in checks if c['request_id']==candidate['request_id'])
            self.assertFalse(changed['strict_bitwise_pass']);self.assertTrue(changed['generated_ids_equal'])
            self.assertGreater(changed['per_step_logits'][0]['original_threshold_metrics']['max_abs'],0)
            self.assertFalse(decision['fixture_strict_bitwise']);self.assertFalse(decision['R4B2_STRICT_BITWISE_FIDELITY_PASS'])
            self.assertEqual(decision['R4B2_NUMERICAL_FIDELITY_DIAGNOSTIC']['original_thresholds'],THRESHOLDS)
            self.assertFalse(decision['correctness_qualified_speedup'])

    def test_fixture_cannot_promote_report_or_decision_qualification_flags(self):
        with temporary() as tmp:
            target=Path(tmp)/'copy';shutil.copytree(self.output,target);path=target/'r4b2_results.json'
            report=read(path);report['PRIMARY_PERFORMANCE_ELIGIBLE']=True;path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ProtocolError,'qualification'):load_run(target)
            report=read(self.output/'r4b2_results.json');decision=read(target/'r4b2_decision.json')
            decision['R4B2_STRICT_BITWISE_FIDELITY_PASS']=True;(target/'r4b2_decision.json').write_text(json.dumps(decision))
            report['decision']=decision;report['output_hashes']['r4b2_decision.json']=sha(target/'r4b2_decision.json');path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ProtocolError,'CPU fixture'):load_run(target)


class BoundedExtension(unittest.TestCase):
    def test_explicit_cpu_g16_extension_is_bounded_and_uses_same_prefix_protocol(self):
        with temporary() as tmp:
            r=subprocess.run([sys.executable,str(SCRIPT),'--phase','fixture','--test-fixture-only','--target-length','64','--prefix-lengths','32',
                '--max-new-tokens','16','--output-dir',tmp],capture_output=True,text=True,timeout=240.)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            report,plan,worker=load_run(tmp);self.assertEqual(report['audit']['requests'],8);self.assertEqual(report['audit']['generation_steps'],128)
            self.assertTrue(report['decision']['fixture_strict_bitwise']);self.assertFalse(report['decision']['R4B2_PHYSICAL_E2E_PASS'])


@unittest.skipUnless(os.environ.get('SEMCACHE_R4B2_GPU')=='1','B2 RTX3090/OPT GPU NOT_RUN; explicit opt-in and allocation required')
class AllocatedGPU(unittest.TestCase):
    def test_actual_allocated_b2_pilot(self):
        base=Path(os.environ['SEMCACHE_R4B2_BASE_PREPARED']);root=Path(os.environ['SEMCACHE_R4B2_OUTPUT'])
        commands=[['--phase','prepare','--enable-gpu-experiment','--base-prepared',str(base),'--output-dir',str(root/'prepare')],
            ['--phase','run','--enable-gpu-experiment','--prepared',str(root/'prepare/prepared'),'--output-dir',str(root/'run')],
            ['--phase','analyze','--inputs',str(root/'run'),'--output-dir',str(root/'analysis')]]
        for args in commands:
            r=subprocess.run([sys.executable,str(SCRIPT),*args],capture_output=True,text=True,timeout=8000.)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        report,_,_=load_run(root/'run');self.assertTrue(report['decision']['R4B2_SUFFIX_ONLY_EXECUTION_PASS'])
        self.assertFalse(report['decision']['correctness_qualified_speedup'])


if __name__=='__main__':unittest.main()
