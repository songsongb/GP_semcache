"""Real CPU ES/UD processes and localhost TCP. Explicit fixture, never GPU evidence."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from semcache.edgelora.r3a_artifacts import ROOT,read
from semcache.edgelora.r3a_tiny import prepare_tiny
from semcache.edgelora.r4c_prepare import TinyTokenizer
from semcache.edgelora.r5_multi_artifacts import validate,tokenizer_identity
from semcache.edgelora.r5_i5_artifacts import prepare as adapters
from semcache.edgelora.r5_i5_evidence import verify,seal
from semcache.edgelora.r5_i5r1_plan import prepare
from semcache.edgelora.r5_i5r1_analysis import summarize
from test_r5_i5 import dataset
from test_r5_i5r1 import SPECIAL


class Processes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(os.environ.get('R5_I5R1_TEST_OUTPUT_ROOT',cls.tmp.name))
        if os.environ.get('R5_I5R1_TEST_INPUTS'):
            cls.run_root=Path(os.environ['R5_I5R1_TEST_INPUTS']);verify(cls.run_root)
            config=read(cls.run_root/'i5_execution_config.json');assert config['fixture'] is True
            cls.plan=Path(config['experiment_prepared']);cls.multi=Path(config['multi_prepared']);cls.root=cls.run_root.parent
            cls.result=read(cls.run_root.with_name(cls.run_root.name+'_i5r1_audit')/'i5r1_decision.json');return
        cls.root.mkdir(parents=True,exist_ok=True);base=prepare_tiny(cls.root/'base');cls.multi=adapters(base,cls.root/'adapters','expanded')
        m,pm,_,_=validate(cls.multi);data=dataset(cls.root,'qasper',1);cls.plan=cls.root/'plan'
        plan=prepare({'qasper':data},TinyTokenizer(),tokenizer_identity(pm),2048,SPECIAL,cls.plan,multi_prepared=cls.multi,
            fixture=True,profile='AMORTIZATION',groups=1,cases=('64:256',),adapters=('Adapter_A',),repeats=(2,),generation_tokens=2,
            sharing_mode='AUTHORIZED_CROSS_USER')
        cls.run_root=cls.root/'block_0_123';cmd=[sys.executable,str(ROOT/'scripts/127_run_r5_i5r1.py'),'--experiment',str(cls.plan),
            '--multi-prepared',str(cls.multi),'--source-commit',plan['i5r1']['source']['commit'],'--test-fixture-only','--output-dir',str(cls.run_root)]
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
        if result.returncode:raise AssertionError(f'{cls.root}\n{result.stdout}\n{result.stderr}')
        cls.result=read(cls.run_root.with_name(cls.run_root.name+'_i5r1_audit')/'i5r1_decision.json')
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_complete_legacy_and_r1_independent_audits(self):
        self.assertEqual(read(cls_path(self,'_audit')/'i5_independent_audit.json')['status'],'PASS')
        self.assertEqual(self.result['functional_assertions'],'PASS');self.assertTrue(self.result['GPU_NOT_RUN'])
        self.assertEqual(self.result['groups'][0]['decisions']['FUNCTIONAL_PHYSICAL_REUSE'],'CPU_FIXTURE_ONLY')
    def test_two_actual_fixture_hits_and_omitted_prefix_wire(self):
        hits=read(cls_path(self,'_i5r1_audit')/'i5r1_physical_hits.json')
        self.assertEqual(len(hits),2)
        for hit in hits:
            self.assertEqual(hit['actual_reused_tokens'],64);self.assertTrue(hit['functional_checks_pass'])
            self.assertGreater(hit['prefill_hidden_payload_bytes_omitted'],0)
            self.assertGreater(hit['prefill_delta_payload_bytes_omitted'],0)
            self.assertNotEqual(hit['source_user'],hit['target_user'])
    def test_costs_include_source_and_numerics_include_priming(self):
        audit=cls_path(self,'_i5r1_audit');costs=read(audit/'i5r1_cost_comparisons.json')[0]['costs']
        for c in costs.values():
            self.assertEqual(c['targets'],2);self.assertEqual(c['priming_sources'],1)
            self.assertGreater(c['process_lifecycle_ms'],c['source_to_last_target_span_ms'])
        numerical=read(audit/'i5r1_numerical_fidelity.json');self.assertEqual(len(numerical),9)
        self.assertEqual(sum(not n['scored'] for n in numerical),3)
        self.assertEqual(len(read(audit/'i5r1_quality_per_sample.json')),6)
        targets=[r for r in read(audit/'i5r1_request_metrics.json') if r['scored']]
        self.assertEqual(len(targets),6)
        self.assertTrue(all(r['generated_token_ids'] for r in targets))
        quality={(q['condition'],q['request_id']):q for q in read(audit/'i5r1_quality_per_sample.json')}
        for row in targets:
            self.assertEqual(quality[row['condition'],row['request_id']]['empty_output'],not bool(row['generated_text'].strip()))
            if not row['generated_text'].strip():
                self.assertEqual(row['generated_token_ids'],[6])
                self.assertEqual(quality[row['condition'],row['request_id']]['score'],0)
    def test_missing_duplicate_and_corrupt_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):summarize(self.plan,[],Path(d)/'missing')
            with self.assertRaisesRegex(ValueError,'duplicate'):summarize(self.plan,[self.run_root,self.run_root],Path(d)/'duplicate')
            copy=Path(d)/'corrupt';shutil.copytree(self.run_root,copy)
            p=next(copy.rglob('tensors/*.bin'));blob=p.read_bytes();p.write_bytes(bytes([blob[0]^1])+blob[1:])
            with self.assertRaises(ValueError):summarize(self.plan,[copy],Path(d)/'bad')
    def test_read_only_reanalysis(self):
        before=(self.run_root/'i5_output_manifest.json').read_bytes()
        with tempfile.TemporaryDirectory() as d:
            r=summarize(self.plan,[self.run_root],Path(d)/'again');self.assertEqual(r,self.result)
        self.assertEqual(before,(self.run_root/'i5_output_manifest.json').read_bytes());verify(self.run_root)


def cls_path(test,suffix):return test.run_root.with_name(test.run_root.name+suffix)


if __name__=='__main__':unittest.main()
