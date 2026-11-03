"""Actual CPU processes/TCP with scalar fixture; no OPT/FP16/GPU qualification."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from semcache.edgelora.communication import digest
from semcache.edgelora.r3a_artifacts import ROOT,read,write
from semcache.edgelora.r5_i5_evidence import seal,jsonl,verify
from semcache.edgelora.r5_i5r2b_plan import prepare
import test_r5_i5r2b as unit


class Processes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.root=Path(os.environ.get('R5_I5R2B_TEST_OUTPUT_ROOT',cls.tmp.name));cls.root.mkdir(parents=True,exist_ok=True)
        # Retained evidence includes retained inputs, enabling a fresh audit.
        cls.multi,cls.old=unit.fixture_inputs(cls.root/'inputs')
        cls.plan=cls.root/'plan';plan=prepare(cls.old,cls.multi,cls.plan,fixture=True)
        cls.run_root=cls.root/'block_0_123'
        cmd=[sys.executable,str(ROOT/'scripts/131_run_r5_i5r2b.py'),'--experiment',str(cls.plan),'--multi-prepared',str(cls.multi),
            '--source-commit',plan['i5r2b']['source']['commit'],'--output-dir',str(cls.run_root),'--test-fixture-only']
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
        (cls.root/'process_stdout.log').write_text(r.stdout);(cls.root/'process_stderr.log').write_text(r.stderr)
        if r.returncode:raise AssertionError(str(cls.root)+'\n'+r.stdout+'\n'+r.stderr)
        cls.result=read(cls.run_root.with_name(cls.run_root.name+'_arithmetic_audit')/'i5r2b_first_divergence.json')
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_actual_fixture_pipeline_completed_and_HIT_retained(self):
        self.assertTrue(self.result['GPU_NOT_RUN']);self.assertEqual(self.result['GPU_EVIDENCE_COMPLETE'],'CPU_FIXTURE_ONLY')
        self.assertEqual(self.result['independent_cache_hit_receipts'],1);self.assertFalse(self.result['ROOT_CAUSE_CONFIRMED'])
    def test_prefix_only_ES_UD_values_have_integrity(self):
        audit=self.run_root.with_name(self.run_root.name+'_arithmetic_audit');inv=read(audit/'i5r2b_capture_inventory.json')
        self.assertTrue(all(x['status']=='PASS' for x in inv['integrity']))
        for cap in inv['inventories'].values():
            self.assertLessEqual(cap['bytes'],cap['byte_limit']);self.assertLessEqual(cap['files'],cap['file_limit'])
            self.assertTrue(all(x['semantic_positions']==[0,64] for x in cap['entries'].values() if x['status']=='CAPTURED'))
    def test_OFF_original_pipeline_preserves_tokens_and_logits(self):
        plan=read(self.plan/'i5_workload_manifest.json');normal=deepcopy(plan);normal.pop('experiment_id');normal.pop('i5r2b')
        normal['experiment_id']=digest(normal);dest=self.root/'off_plan';dest.mkdir();write(dest/'i5_workload_manifest.json',normal)
        write(dest/'i5_dataset_manifest.json',read(self.plan/'i5_dataset_manifest.json'));seal(dest)
        out=self.root/'off_run'
        cmd=[sys.executable,str(ROOT/'scripts/119_run_r5_i5.py'),'--experiment',str(dest),'--multi-prepared',str(self.multi),
            '--source-commit',plan['i5r2b']['source']['commit'],'--output-dir',str(out),'--test-fixture-only']
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=600);self.assertEqual(r.returncode,0,r.stderr)
        self.assertFalse(list(out.rglob('i5r2b_capture.json')))
        for c in ('FULL_RECOMPUTE','SEMANTIC_NO_REUSE','SEMCACHE_INTEGRATED'):
            before=jsonl(self.run_root/c/'generation.jsonl');after=jsonl(out/c/'generation.jsonl')
            self.assertEqual([(r['generated_token_ids'],r['generated_text'],r['logits_sha256']) for r in before],
                             [(r['generated_token_ids'],r['generated_text'],r['logits_sha256']) for r in after])


if __name__=='__main__':unittest.main()
