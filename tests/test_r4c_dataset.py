"""C0 uses synthetic gold only under explicit fixture scope."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from semcache.edgelora.communication import digest,digest_bytes
from semcache.edgelora.r4c_dataset import load_dataset,select
from semcache.edgelora.r4c_prepare import fixture_inputs,prepare_collection,load_prepared
from semcache.edgelora.r4c_prompts import FixtureTokenizer,TokenizerAdapter,prepare,prefix_of,validate_prepared,TEMPLATE_ID
from semcache.edgelora.r4c_workload import natural_discovery


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.registry,self.tok=fixture_inputs(self.root/'data',count=3)
        self.rows,self.manifest=load_dataset('qasper',self.registry['qasper'],fixture=True)

    def test_missing_file_has_acquisition_diagnostic(self):
        with self.assertRaisesRegex(ValueError,'MISSING_DATASET.*Acquire LongBench'):
            load_dataset('qasper',dict(path=self.root/'missing.jsonl'))

    def test_hash_revision_schema_and_synthetic_scope(self):
        self.assertEqual(self.rows[0]['source_record_sha256'],digest_bytes(Path(self.registry['qasper']['path']).read_text().splitlines()[0].encode()))
        for spec in (dict(self.registry['qasper'],revision='main'),dict(self.registry['qasper'],sha256={})):
            with self.assertRaises(ValueError):load_dataset('qasper',spec,fixture=True)
        with self.assertRaisesRegex(ValueError,'scope mismatch'):load_dataset('qasper',self.registry['qasper'])
        path=Path(self.registry['qasper']['path']);row=json.loads(path.read_text().splitlines()[0]);row['answers']=[];path.write_text(json.dumps(row)+'\n')
        spec=dict(self.registry['qasper'],sha256={path.name:digest_bytes(path.read_bytes())})
        with self.assertRaisesRegex(ValueError,'MISSING_GOLD'):load_dataset('qasper',spec,fixture=True)

    def test_stable_selection_and_explicit_exclusions(self):
        a,meta=select(self.rows,2,42);b,_=select(list(reversed(self.rows)),2,42)
        self.assertEqual(a,b);self.assertEqual(meta['requested'],2)
        rid='qasper:'+a[0]['original']['_id'];c,report=select(self.rows,2,42,cohort='independent',excluded_ids=[rid])
        self.assertNotIn(a[0],c);self.assertEqual(report['exclusions'][0]['reason'],'previous_cohort')
        with self.assertRaisesRegex(ValueError,'INSUFFICIENT'):select(self.rows,4,42)
        with self.assertRaisesRegex(ValueError,'INSUFFICIENT'):select(self.rows,1,42,min_context_tokens=100000,tokenizer=self.tok)
        lengths={r['original']['_id']:1 for r in self.rows}
        with self.assertRaisesRegex(ValueError,'INSUFFICIENT'):select(self.rows,1,42,input_lengths=lengths,min_input_tokens=129)

    def test_prompt_preserves_gold_question_instruction_and_records_truncation(self):
        row=self.rows[0];tok=TokenizerAdapter(FixtureTokenizer(),{'fixture':'char'},2048)
        p=prepare(row,tok,input_budget=256,max_new_tokens=32)
        self.assertLessEqual(len(p['token_ids']),256);self.assertEqual(p['attention_mask'],[1]*len(p['token_ids']))
        self.assertEqual(p['original']['answers'],row['original']['answers']);self.assertIn(row['original']['input'],p['prompt_text'])
        self.assertGreater(p['removed_context_tokens'],0);self.assertEqual(p['original_context_tokens'],p['retained_context_tokens']+p['removed_context_tokens'])
        self.assertEqual(p['prompt_template_id'],TEMPLATE_ID);self.assertEqual(p,prepare(row,tok,input_budget=256,max_new_tokens=32))
        self.assertEqual(validate_prepared(p),p);bad=deepcopy(p);bad['token_ids'][0]+=1
        with self.assertRaises(ValueError):validate_prepared(bad)
        news,_=load_dataset('multi_news',self.registry['multi_news'],fixture=True)
        summary=prepare(news[0],tok,max_new_tokens=64);self.assertIn(news[0]['original']['input'],summary['prompt_text'])

    def test_actual_model_positional_limit_and_extension_gates(self):
        with self.assertRaisesRegex(ValueError,'POSITIONAL_LIMIT'):prepare(self.rows[0],TokenizerAdapter(FixtureTokenizer(),{},270),max_new_tokens=32)
        with self.assertRaisesRegex(ValueError,'512-token'):prepare(self.rows[0],self.tok,input_budget=512)
        p=prepare(self.rows[0],self.tok,input_budget=512,max_new_tokens=64,enable_512=True)
        self.assertLessEqual(len(p['token_ids']),512)
        row=deepcopy(self.rows[0]);row['original']['input']='x'*300
        with self.assertRaisesRegex(ValueError,'question/instruction'):prepare(row,self.tok)

    def test_token_boundary_prefix_avoids_lossy_roundtrip(self):
        p=prepare(self.rows[0],self.tok);src=prefix_of(p,128)
        self.assertEqual(src['token_ids'],p['token_ids'][:128]);self.assertEqual(src['target_prepared_sha256'],p['prepared_record_sha256'])
        self.assertNotEqual(self.tok.encode(self.tok.decode(src['token_ids'])),src['token_ids'])
        with self.assertRaises(ValueError):prefix_of(p,len(p['token_ids']))

    def test_natural_discovery_is_separate_from_physical_hits(self):
        prompts=[dict(sample_id='A',token_ids=[1,2,3,4]),dict(sample_id='B',token_ids=[1,2,3,5]),dict(sample_id='C',token_ids=[9,1,2,3])]
        r=natural_discovery(prompts,minimum=2,layers=2,hidden=4,element_bytes=2)
        self.assertEqual(r['eligible_pairs'],1);self.assertEqual(r['matches'][0]['prefix_tokens'],3)
        self.assertEqual(r['matches'][0]['raw_qkv_capacity_bytes'],144);self.assertFalse(r['physical_hits_executed']);self.assertEqual(r['records_without_useful_overlap'],['C'])

    def test_collection_manifest_and_schedule_integrity(self):
        out=self.root/'prepared';out.mkdir();m,p,t=prepare_collection(self.registry,self.tok,out,count=1,fixture=True)
        a,b,c=load_prepared(out);self.assertEqual((a,b,c),(m,p,t));self.assertEqual(len(t),9)
        self.assertTrue(all(task['prompt']['token_ids']==p[i//3]['token_ids'][:128] for i,task in enumerate(t) if task['mode']=='COLD_MISS'))
        (out/'r4c_schedule.json').write_text('[]')
        with self.assertRaises(ValueError):load_prepared(out)

    def test_per_dataset_generation_budgets_override_defaults(self):
        self.registry['qasper']['max_new_tokens']=19;self.registry['2wikimqa']['max_new_tokens']=23;self.registry['multi_news']['max_new_tokens']=37
        out=self.root/'per_dataset';out.mkdir();manifest,prompts,tasks=prepare_collection(self.registry,self.tok,out,count=1,fixture=True)
        self.assertEqual(manifest['generation_budgets'],dict(qasper=19,**{'2wikimqa':23},multi_news=37))
        for prompt in prompts:self.assertEqual(prompt['max_new_tokens'],manifest['generation_budgets'][prompt['dataset']])
        self.assertTrue(all(t['policy']['max_new_tokens']==manifest['generation_budgets'][t['dataset']] for t in tasks if t['mode']!='COLD_MISS'))

    def test_runtime_fingerprints_actual_tokenizer_object(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from semcache.edgelora.r4c_prepare import opt_tokenizer
        actual=SimpleNamespace(get_vocab=lambda:{'actual':7})
        config=SimpleNamespace(model_type='opt',max_position_embeddings=2048,num_hidden_layers=2,hidden_size=16,to_dict=lambda:{'kind':'MOCK_CONFIG_ONLY'})
        fake=SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=lambda *a,**k:config),AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:self.fail('must inspect serving tokenizer')))
        identity=dict(model='facebook/opt-2.7b',base_revision='a'*40,tokenizer_source='facebook/opt-2.7b',tokenizer_revision='a'*40,dtype='float16')
        with patch.dict('sys.modules',{'transformers':fake}),patch('semcache.edgelora.r4c_prepare.tokenizer_configuration',side_effect=lambda t:{'is_actual':t is actual}):
            adapter=opt_tokenizer(dict(identity=identity),actual)
        self.assertIs(adapter.tokenizer,actual);self.assertTrue(adapter.identity['configuration']['is_actual'])

    @unittest.skipUnless(importlib.util.find_spec('datasets'),'datasets package unavailable; JSONL path independently tested')
    def test_local_hf_snapshot(self):
        from datasets import Dataset
        from semcache.edgelora.r4c_dataset import file_hashes
        path=self.root/'hf';Dataset.from_list([r['original'] for r in self.rows]).save_to_disk(str(path))
        rows,_=load_dataset('qasper',dict(path=str(path),revision='test-v1',format='hf_saved_dataset',sha256=file_hashes(path)),fixture=True)
        self.assertEqual(len(rows),3)
