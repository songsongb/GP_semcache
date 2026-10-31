"""Real CPU processes/TCP, physical fixture KV, condition semantics and audit tampering."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from semcache.edgelora.communication import canonical
from semcache.edgelora.r3a_artifacts import ROOT,read
from semcache.edgelora.r3a_tiny import prepare_tiny
from semcache.edgelora.r4c_prepare import TinyTokenizer
from semcache.edgelora.r5_multi_artifacts import tokenizer_identity
from semcache.edgelora.r5_i5_artifacts import prepare
from semcache.edgelora.r5_i5_evidence import jsonl,seal,verify
from semcache.edgelora.r5_i5_workload import prepare_experiment,CONDITIONS
from semcache.edgelora.r5_i5_analysis import analyze
from semcache.edgelora.r5_perf_evidence import Writer
from test_r5_i5 import dataset


class Processes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(os.environ.get('R5_I5_TEST_OUTPUT_ROOT',cls.tmp.name))
        existing=os.environ.get('R5_I5_TEST_INPUTS')
        if existing:
            cls.run_root=Path(existing);cls.root=cls.run_root.parent;verify(cls.run_root)
            from semcache.edgelora.r5_i5_evidence import sources
            assert read(cls.run_root/'i5_execution_config.json')['sources']==sources()
            assert read(cls.run_root/'i5_execution_config.json')['fixture'] is True
            cls.report=analyze(cls.run_root);return
        cls.root.mkdir(parents=True,exist_ok=True);base=prepare_tiny(cls.root/'base')
        multi=prepare(base,cls.root/'adapters','expanded');pm=read(base/'r3a_prepare_manifest.json')
        data=dataset(cls.root,'coqa',3)
        prepare_experiment({'coqa':data},TinyTokenizer(),tokenizer_identity(pm),2048,
            dict(bos_token_id=0,eos_token_id=6,pad_token_id=0),cls.root/'plan',fixture=True,overrides={'coqa':3},
            budgets={'coqa':2},tracks=('CONTROLLED',),prefixes=(64,),repetitions=1,block_size=3,
            input_limit=192,allow_truncation=True)
        cls.run_root=cls.root/'run'
        cmd=[sys.executable,str(ROOT/'scripts/119_run_r5_i5.py'),'--test-fixture-only','--experiment',str(cls.root/'plan'),
             '--multi-prepared',str(multi),'--output-dir',str(cls.run_root)]
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
        if r.returncode:raise AssertionError(str(cls.run_root)+'\n'+r.stderr)
        cls.report=analyze(cls.run_root)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_a_has_no_semantic_encoder_or_calls(self):
        path=self.run_root/CONDITIONS[0]
        self.assertTrue(all(read(path/('Adapter_'+a)/'identity.json')['semantic_encoder'] is None for a in 'ABC'))
        self.assertFalse(jsonl(path/'events.jsonl'))
        self.assertTrue(all(r['semantic_encoding_ms']==0 and r['candidate_count']==0 and r['mode']=='NO_REUSE' for r in jsonl(path/'generation.jsonl')))
    def test_b_discovers_completed_sources_without_tensor_reuse(self):
        rows=jsonl(self.run_root/CONDITIONS[1]/'generation.jsonl')
        self.assertEqual(sum(r['candidate_count']>0 for r in rows if r['scored']),3)
        self.assertTrue(all(r['source'] is None and r['reused_prefix_tokens']==0 and r['cache_after']['occupancy_bytes']==0 for r in rows))
        self.assertTrue(any(r['shadow']['metadata_serialized_bytes']>2 for r in rows))
    def test_c_physically_reuses_all_adapters_and_suffix_only(self):
        rows=jsonl(self.run_root/CONDITIONS[2]/'generation.jsonl');targets=[r for r in rows if r['scored']]
        self.assertEqual([r['reused_prefix_tokens'] for r in targets],[64]*3)
        self.assertEqual({r['adapter_id'] for r in targets},{'Adapter_'+a for a in 'ABC'})
        self.assertTrue(all(not r['cached_prefix_q_consumed'] and r['skipped_prefix_projection_token_rows']>0 for r in targets))
    def test_independent_gates_and_sampled_scope(self):
        r=self.report
        self.assertEqual(r['independent_audit']['status'],'PASS');self.assertTrue(r['independent_audit']['shadow_index_equivalent'])
        self.assertTrue(r['decision']['GPU_NOT_RUN']);self.assertFalse(r['decision']['GPU_VERIFIED'])
        self.assertEqual(r['decision']['strict_sampled_hit_audit'],'PASS')
        self.assertFalse(r['decision']['correctness_qualified_speedup']);self.assertFalse(r['independent_audit']['strict_all_requests'])
    def test_source_cost_and_reference_scores_accounted(self):
        for c in CONDITIONS:
            s=self.report['latency_summary'][c]
            self.assertEqual(s['priming_requests'],3);self.assertEqual(s['target_requests'],3)
            self.assertGreater(s['amortized_ms_per_target'],s['target_wall_ms']/3)
        self.assertEqual(len(self.report['quality_per_sample']),9)
    def test_read_only_reanalysis(self):
        before=(self.run_root/'i5_output_manifest.json').read_bytes();analyze(self.run_root)
        self.assertEqual(before,(self.run_root/'i5_output_manifest.json').read_bytes());verify(self.run_root)
    def test_partial_block_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/'copy';shutil.copytree(self.run_root,copy)
            (copy/'i5_block_status.json').write_text('{"status":"FAILED"}');seal(copy)
            with self.assertRaisesRegex(ValueError,'partial/failed'):analyze(copy)
    def test_forged_route_rejected_after_resealing(self):
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/'copy';shutil.copytree(self.run_root,copy);condition=copy/CONDITIONS[2]
            rows=jsonl(condition/'generation.jsonl');rows[-1]['adapter_id']='Adapter_A'
            Writer(condition,self.report['config']['condition_budget_bytes']).bytes(condition/'generation.jsonl',b''.join(canonical(r)+b'\n' for r in rows))
            seal(copy)
            with self.assertRaises(ValueError):analyze(copy)
    def test_tensor_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/'copy';shutil.copytree(self.run_root,copy)
            p=next((copy/CONDITIONS[2]).rglob('tensors/*.bin'));b=p.read_bytes();p.write_bytes(bytes([b[0]^1])+b[1:])
            with self.assertRaises(ValueError):analyze(copy)
    def test_complete_summary_and_missing_duplicate_block_rejection(self):
        from semcache.edgelora.r5_i5_summary import summarize
        with tempfile.TemporaryDirectory() as d:
            out=summarize(self.root/'plan',[self.run_root],Path(d)/'summary')
            self.assertEqual(read(out/'i5_independent_audit.json')['complete_blocks'],1)
            self.assertTrue((out/'i5_plot_data.csv').is_file())
            import csv
            table=list(csv.DictReader((out/'i5_plot_data.csv').open()))
            self.assertTrue(all(r['TEST_FIXTURE_ONLY']=='True' for r in table))
            with self.assertRaisesRegex(ValueError,'MISSING_BLOCKS'):summarize(self.root/'plan',[],Path(d)/'missing')
            with self.assertRaisesRegex(ValueError,'duplicate block'):summarize(self.root/'plan',[self.run_root,self.run_root],Path(d)/'duplicate')


if __name__=='__main__':unittest.main()
