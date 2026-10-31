"""Complete application + actual independent TCP, and adversarial saved audits."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from semcache.edgelora.r3a_artifacts import ROOT,read,write
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.r5_analysis import analyze
from semcache.edgelora.r5_evidence import seal

COMMAND=[sys.executable,str(ROOT/'scripts/110_run_r5_semcache_e2e.py')]


class ProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.out=cls.root/'fixture'
        r=subprocess.run(COMMAND+['--phase','fixture','--test-fixture-only','--lifecycle-capacity','--output-dir',str(cls.out)],capture_output=True,text=True,timeout=240)
        if r.returncode:
            durable=ROOT/'results/r5/test_failures'/str(time.time_ns());durable.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(cls.out,durable)
            raise AssertionError(f'fixture failed; evidence preserved: {durable}\n{r.stderr}\n'+(cls.out/'es/stderr.log').read_text())
        cls.rows=jsonl(cls.out/'r5_generation_outputs.jsonl')
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def tamper(self,path,edit,pattern):
        path=self.out/path;before=path.read_bytes()
        try:
            values=jsonl(path);edit(values);path.write_text(''.join(json.dumps(v)+'\n' for v in values))
            with self.assertRaisesRegex((ValueError,KeyError),pattern):analyze(self.out)
        finally:path.write_bytes(before)

    def test_actual_full_path_hit_miss_eviction_decode_and_no_gpu_promotion(self):
        a=read(self.out/'r5_audit.json');d=read(self.out/'r5_decision.json')
        self.assertEqual(a['status'],'PASS');self.assertEqual(a['requests'],20);self.assertEqual(a['generated_steps'],60)
        self.assertEqual(a['hits'],4);self.assertGreaterEqual(a['evictions'],1)
        self.assertTrue(all(d[k] for k in ('R5_I0_CONTRACT_PASS','R5_I1_SEMANTIC_PATH_PASS','R5_I2_CACHE_LIFECYCLE_PASS','R5_I3_DISTRIBUTED_GENERATION_PASS','R5_CPU_FIXTURE_PASS')))
        self.assertFalse(d['R5_GPU_PHYSICAL_EXECUTION_PASS']);self.assertFalse(d['NATIVE_PEFT_STRICT_PARITY_PASS']);self.assertFalse(d['correctness_qualified_speedup'])
        self.assertTrue(all(r['generated_token_ids'] and r['es_pid']!=r['ud_pid'] for r in self.rows))

    def test_rejected_admission_preserves_completed_generation(self):
        rows=[r for r in self.rows if r['condition']=='INTEGRATED'];self.assertIsNone(rows[5]['source']);self.assertIsNone(rows[5]['admission'])
        self.assertEqual(rows[5]['status'],'COMPLETE');self.assertEqual(len(rows[5]['generated_token_ids']),3)
        self.assertEqual(rows[6]['decision'],'SEMANTIC_CANDIDATE_REJECTED');self.assertGreater(rows[6]['candidate_count'],0)

    def test_actual_cross_user_source_and_skipped_projection_payloads(self):
        by={r['request_id']:r for r in self.rows};hit=by['R5_01'];source=by[hit['source']['source_request_id']];baseline=by['N_R5_01']
        self.assertNotEqual(hit['user_id'],source['user_id']);self.assertEqual(hit['adapter_id'],source['adapter_id'])
        self.assertEqual(hit['skipped_prefix_projection_token_rows'],10*3*32)
        self.assertLess(hit['prefill']['hidden_payload_bytes'],baseline['prefill']['hidden_payload_bytes'])
        self.assertLess(hit['prefill']['delta_payload_bytes'],baseline['prefill']['delta_payload_bytes'])
        self.assertFalse(hit['cached_prefix_q_consumed'])

    def test_readonly_analysis_reports_byte_identical(self):
        before=(self.out/'r5_output_manifest.json').read_bytes();a=analyze(self.out,sealed=True);b=analyze(self.out,sealed=True);self.assertEqual(a,b)
        for suffix in ('a','b'):
            r=subprocess.run([sys.executable,str(ROOT/'scripts/111_analyze_r5_semcache_e2e.py'),'--inputs',str(self.out),'--output-dir',str(self.root/suffix)],capture_output=True,text=True,timeout=90)
            self.assertEqual(r.returncode,0,r.stderr)
        for name in ('r5_audit.json','r5_decision.json','r5_fidelity.json','r5_tcp_summary.json','r5_latency_summary.json','r5_limitations.json'):
            self.assertEqual((self.root/'a'/name).read_bytes(),(self.root/'b'/name).read_bytes());self.assertEqual((self.root/'a'/name).read_bytes(),(self.out/name).read_bytes())
        self.assertEqual(before,(self.out/'r5_output_manifest.json').read_bytes())

    def test_fake_hit_full_prefill_is_independently_rejected(self):
        def edit(xs):next(s for s in xs if s['context']['mode']=='PARTIAL_PREFIX_HIT')['executed_projection_token_rows']+=960
        self.tamper('es/r5_tokens.jsonl',edit,'hidden full-prefill')

    def test_tensor_owner_accounting_corruption_is_independently_rejected(self):
        self.tamper('r5_ownership.jsonl',lambda xs:xs[0].update(physical_bytes=1),'byte accounting')

    def test_directional_traffic_and_latency_tampering_is_rejected(self):
        self.tamper('r5_generation_outputs.jsonl',lambda xs:xs[0].update(es_to_ud_wire_bytes=1),'directional traffic')
        self.tamper('r5_generation_outputs.jsonl',lambda xs:xs[0].update(decode_ms=0),'timing summary')

    def test_tensor_bytes_and_decode_chain_are_independently_rejected(self):
        steps=jsonl(self.out/'es/r5_tokens.jsonl');tensor=self.out/'es'/steps[0]['qkv']['0.q']['file'];before=tensor.read_bytes()
        try:
            tensor.write_bytes(bytes([before[0]^1])+before[1:])
            with self.assertRaisesRegex(ValueError,'tensor evidence'):analyze(self.out)
        finally:tensor.write_bytes(before)
        def edit(xs):[s for s in xs if s['request_id']=='R5_01'][-1]['kv'][0]['k']['sha256']='0'*64
        self.tamper('es/r5_tokens.jsonl',edit,'reconstructed Decode KV')

    def test_forged_index_and_eviction_and_incomplete_tokens_are_rejected(self):
        def edit(xs):next(e for e in xs if e['kind']=='DISCOVERY' and e['candidates'])['candidates'][0]['metadata']['source_request_id']='FORGED'
        self.tamper('r5_semantic_events.jsonl',edit,'candidate metadata')
        def edit(xs):next(e for e in xs if e['kind']=='EVICT')['released_bytes']+=1
        self.tamper('r5_cache_events.jsonl',edit,'invalid release')
        self.tamper('r5_generation_outputs.jsonl',lambda xs:xs[0].update(generated_token_ids=[]),'incomplete generation')

    def test_saved_fixture_flag_cannot_be_promoted_even_when_resealed(self):
        path=self.out/'r5_decision.json';manifest=self.out/'r5_output_manifest.json';before=path.read_bytes();old=manifest.read_bytes()
        try:
            d=read(path);d['R5_GPU_PHYSICAL_EXECUTION_PASS']=True;write(path,d);seal(self.out)
            with self.assertRaisesRegex(ValueError,'stored decision'):analyze(self.out,sealed=True)
        finally:path.write_bytes(before);manifest.write_bytes(old)

    def test_no_overwrite_and_unprepared_adapter_not_counted(self):
        before=(self.out/'r5_results.json').read_bytes()
        r=subprocess.run(COMMAND+['--phase','fixture','--test-fixture-only','--lifecycle-capacity','--output-dir',str(self.out)],capture_output=True,text=True,timeout=15)
        self.assertNotEqual(r.returncode,0);self.assertEqual(before,(self.out/'r5_results.json').read_bytes())
        neg=read(self.out/'r5_negative_controls.json');self.assertFalse(neg['generated_request']);self.assertEqual(neg['tensor_frames'],0)


class LauncherTests(unittest.TestCase):
    def test_help_dry_run_and_no_silent_fixture_optin(self):
        with tempfile.TemporaryDirectory() as d:
            for args in (['--help'],['--phase','dry-run','--output-dir',d]):
                r=subprocess.run(COMMAND+args,capture_output=True,text=True,timeout=15);self.assertEqual(r.returncode,0,r.stderr)
            r=subprocess.run(COMMAND+['--phase','fixture','--output-dir',str(Path(d)/'bad')],capture_output=True,text=True,timeout=15)
            self.assertNotEqual(r.returncode,0);self.assertFalse((Path(d)/'bad').exists())

    def test_missing_gpu_optin_rejected_before_ud_model(self):
        with tempfile.TemporaryDirectory() as d:
            env=dict(os.environ);env.pop('R5_ENABLE_GPU',None);out=Path(d)/'run'
            r=subprocess.run(COMMAND+['--phase','run','--enable-gpu-experiment','--base-prepared',str(Path(d)/'missing'),'--output-dir',str(out)],env=env,capture_output=True,text=True,timeout=15)
            self.assertNotEqual(r.returncode,0);self.assertIn('R5_ENABLE_GPU',read(out/'r5_failure.json')['reason']);self.assertFalse((out/'es').exists())

    def test_missing_real_tinybert_never_uses_controlled_vectors(self):
        from semcache.edgelora.r5_runtime import encoder_for
        with self.assertRaisesRegex(ValueError,'MISSING_TINYBERT'):encoder_for(dict(fixture=False,semantic_revision=None),[])

    def test_resource_budget_failure_preserves_stage_and_partial_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'limited';r=subprocess.run(COMMAND+['--phase','fixture','--test-fixture-only','--lifecycle-capacity','--evidence-budget-bytes','100','--output-dir',str(out)],capture_output=True,text=True,timeout=60)
            self.assertNotEqual(r.returncode,0);self.assertTrue((out/'r5_output_manifest.json').exists())
            failure=read(out/'r5_failure.json');self.assertIn('EVIDENCE_BUDGET_EXCEEDED',failure['worker_failure']['reason'])
            self.assertFalse(read(out/'r5_decision.json')['R5_CPU_FIXTURE_PASS'])

    def test_scalable_and_quality_profiles_execute_same_policy_semantics(self):
        with tempfile.TemporaryDirectory() as d:
            for mode,profile in (('fixed','SCALABLE_EVALUATION'),('quality','SCALABLE_EVALUATION')):
                out=Path(d)/mode;r=subprocess.run(COMMAND+['--phase','fixture','--test-fixture-only','--lifecycle-capacity',
                    '--generation-mode',mode,'--evidence-profile',profile,'--output-dir',str(out)],capture_output=True,text=True,timeout=180)
                self.assertEqual(r.returncode,0,r.stderr)
                rows=jsonl(out/'r5_generation_outputs.jsonl');self.assertTrue(all(r['generated_token_ids'] for r in rows))
                self.assertFalse(read(out/'r5_decision.json')['R5_STRICT_BITWISE_FIDELITY_PASS'])
                if mode=='fixed':self.assertTrue(all(r['generated_tokens']==3 for r in rows))
                else:
                    self.assertTrue(any(r['termination']=='EOS' for r in rows))
                    for r in rows:
                        self.assertIn(r['termination'],('EOS','LENGTH_CAP'))
                        if r['termination']=='EOS':self.assertEqual(r['generated_token_ids'][-1],6)
                        else:self.assertEqual(r['generated_tokens'],3)
