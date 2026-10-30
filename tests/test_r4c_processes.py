"""Real independent CPU UD/TCP fixture and read-only artifact reproducibility."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from semcache.edgelora.r3a_artifacts import ROOT,read
from semcache.edgelora.r4c_analysis import analyze


class ProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.output=cls.root/'fixture'
        cls.command=[sys.executable,str(ROOT/'scripts/106_run_r4c_quality_e2e.py')]
        result=subprocess.run(cls.command+['--phase','fixture','--test-fixture-only','--fixture-policy','fixed','--output-dir',str(cls.output)],capture_output=True,text=True,timeout=240)
        if result.returncode:
            import shutil,time
            durable=ROOT/'results/r4/c/test_failures'/str(time.time_ns());durable.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(cls.output,durable)
            detail=(cls.output/'es/stderr.log').read_text() if (cls.output/'es/stderr.log').exists() else ''
            raise AssertionError('fixture failed; preserved at '+str(durable)+'\n'+result.stderr+'\n'+detail)
        cls.result=read(cls.output/'r4c_results.json')
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_physical_fixture_long_prompt_gold_and_false_gpu_flags(self):
        self.assertEqual(self.result['execution_status'],'COMPLETE');self.assertFalse(self.result['gpu_inference_performed'])
        self.assertEqual(self.result['audit']['requests'],9);self.assertEqual(self.result['audit']['status'],'PASS')
        self.assertGreater(self.result['audit']['strict_prefix_kv_byte_checks'],0)
        d=read(self.output/'r4c_decision.json')
        self.assertTrue(all(v is False for k,v in d.items() if k.endswith('_PASS') or k in ('PRIMARY_PERFORMANCE_ELIGIBLE','correctness_qualified_speedup')))
        pairs=[json.loads(l) for l in (self.output/'r4c_quality_pairs.jsonl').read_text().splitlines()];self.assertEqual(len(pairs),3)
        self.assertTrue(all('quality_no_reuse' in p and 'quality_hit' in p for p in pairs))

    def test_independent_analysis_reproducible_and_source_read_only(self):
        before=(self.output/'r4c_output_manifest.json').read_bytes();a=analyze(self.output,sealed=True);b=analyze(self.output,sealed=True)
        self.assertEqual(a,b);self.assertEqual(before,(self.output/'r4c_output_manifest.json').read_bytes())
        destination=self.root/'analysis';r=subprocess.run(self.command+['--phase','analyze','--inputs',str(self.output),'--output-dir',str(destination)],capture_output=True,text=True,timeout=60)
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(read(destination/'r4c_audit.json'),a['audit'])

    def test_hidden_full_prefill_wrong_evidence_detected(self):
        path=self.output/'es/r4c_tokens.jsonl';before=path.read_bytes()
        try:
            lines=[json.loads(s) for s in before.splitlines()];hit=next(s for s in lines if s['context']['mode']=='PARTIAL_PREFIX_HIT');hit['executed_projection_token_rows']+=128*96
            path.write_text(''.join(json.dumps(s)+'\n' for s in lines))
            with self.assertRaisesRegex(ValueError,'hidden full-prefill'):analyze(self.output)
        finally:path.write_bytes(before)

    def test_failure_and_nonempty_output_are_preserved(self):
        before=(self.output/'r4c_results.json').read_bytes();r=subprocess.run(self.command+['--phase','fixture','--test-fixture-only','--output-dir',str(self.output)],capture_output=True,text=True,timeout=15)
        self.assertNotEqual(r.returncode,0);self.assertEqual(before,(self.output/'r4c_results.json').read_bytes())

    def test_quality_eos_uses_actual_generated_count_and_receipt(self):
        out=self.root/'eos';r=subprocess.run(self.command+['--phase','fixture','--test-fixture-only','--output-dir',str(out)],capture_output=True,text=True,timeout=180)
        self.assertEqual(r.returncode,0,r.stderr)
        rows=[json.loads(s) for s in (out/'es/r4c_requests.jsonl').read_text().splitlines()]
        targets=[s for s in rows if s['mode']!='COLD_MISS'];self.assertTrue(all(s['termination']=='EOS' and s['generated_tokens']<s['policy']['max_new_tokens'] for s in targets))
        self.assertTrue(all(s['generated_text']=='' for s in targets))  # Fixture EOS is a skipped special token too.

    def test_cpu_flags_cannot_be_promoted_by_resealing_decision(self):
        from semcache.edgelora.r4c_evidence import seal_directory
        path=self.output/'r4c_decision.json';manifest=self.output/'r4c_output_manifest.json';before=path.read_bytes();old_manifest=manifest.read_bytes()
        try:
            decision=read(path);decision['R4C_PHYSICAL_E2E_PASS']=True;path.write_text(json.dumps(decision));seal_directory(self.output)
            with self.assertRaisesRegex(ValueError,'stored decision'):analyze(self.output,sealed=True)
        finally:path.write_bytes(before);manifest.write_bytes(old_manifest)

    def test_last_decode_kv_digest_is_independently_reconstructed(self):
        path=self.output/'es/r4c_tokens.jsonl';before=path.read_bytes()
        try:
            records=[json.loads(line) for line in before.splitlines()]
            last=[s for s in records if s['request_id']=='C0000_H'][-1];last['kv'][0]['k']['sha256']='0'*64
            path.write_text(''.join(json.dumps(s)+'\n' for s in records))
            with self.assertRaisesRegex(ValueError,'reconstructed Decode KV'):analyze(self.output)
        finally:path.write_bytes(before)


class LauncherTests(unittest.TestCase):
    def test_help_and_dry_run_need_no_heavy_dependencies(self):
        with tempfile.TemporaryDirectory() as d:
            command=[sys.executable,str(ROOT/'scripts/106_run_r4c_quality_e2e.py')]
            for args in (['--help'],['--phase','dry-run','--output-dir',d]):
                r=subprocess.run(command+args,capture_output=True,text=True,timeout=20);self.assertEqual(r.returncode,0,r.stderr)

    def test_gpu_without_optin_fails_before_ud_or_model(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';env=dict(os.environ);env.pop('R4C_ENABLE_GPU',None)
            r=subprocess.run([sys.executable,str(ROOT/'scripts/106_run_r4c_quality_e2e.py'),'--phase','run','--enable-gpu-experiment',
                '--prepared',str(Path(d)/'missing'),'--base-prepared',str(Path(d)/'missingbase'),'--output-dir',str(out)],capture_output=True,text=True,env=env,timeout=20)
            self.assertNotEqual(r.returncode,0);self.assertEqual(read(out/'r4c_failure.json')['stage'],'PREFLIGHT');self.assertFalse((out/'es').exists())

    def test_missing_real_dataset_has_stage_evidence_and_no_synthetic_substitute(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);registry=root/'registry.json';registry.write_text(json.dumps({'qasper':dict(path='missing.jsonl',revision='test-v1',sha256={})}))
            out=root/'out';r=subprocess.run([sys.executable,str(ROOT/'scripts/105_prepare_r4c_longbench.py'),'--validate-datasets-only','--datasets','qasper','--registry',str(registry),'--output-dir',str(out)],capture_output=True,text=True,timeout=20)
            self.assertNotEqual(r.returncode,0);result=read(out/'r4c_prepare_result.json')
            self.assertEqual(result['stage'],'DATASET_VALIDATION');self.assertIn('MISSING_DATASET',result['reason']);self.assertFalse((out/'r4c_prepared_prompts.jsonl').exists())
