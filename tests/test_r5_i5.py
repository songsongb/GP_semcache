"""I5 schema, gold metrics, context, conditions and immutable planning contracts."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock

from semcache.edgelora.communication import digest
from semcache.edgelora.r3a_artifacts import write,sha
from semcache.edgelora.r4c_prompts import FixtureTokenizer
from semcache.edgelora.r4c_prepare import TinyTokenizer
from semcache.edgelora.r5_i5_data import normalize,validate_sample,LABELS,load_dataset
from semcache.edgelora.r5_i5_quality import score,aggregate
from semcache.edgelora.r5_i5_workload import prepare_prompt,select,requests,resources,prepare_experiment,EligibilityError,CONDITIONS
from semcache.edgelora.r5_i5_evidence import fresh,seal,verify
from semcache.edgelora.r5_i5_analysis import distribution,paired,grouped_ci


def samples(name='coqa', count=4):
    mark=dict(TEST_FIXTURE_ONLY=True)
    if name=='coqa':
        return normalize(name,[dict(mark,id=f'c{i}',story='A real-schema fixture story. '*25,
            questions=[dict(turn_id=1,input_text='Where?'),dict(turn_id=2,input_text='Who?')],
            answers=[dict(turn_id=1,input_text='SECRET_CURRENT_ANSWER'),dict(turn_id=2,input_text='Alice')],
            additional_answers={'a':[dict(turn_id=1,input_text='here'),dict(turn_id=2,input_text='Alice')]}) for i in range(count)],fixture=True)
    if name=='snips':
        return normalize(name,[dict(mark,id=f's{i}',text=f'Play song number {i}',intent=LABELS[i%7]) for i in range(count)],fixture=True)
    if name=='multiwoz':
        return normalize(name,[dict(mark,dialogue_id=f'd{i}',turns=[dict(turn_id='0',speaker='USER',utterance='Need a cheap hotel',
            frames=[dict(service='hotel',state=dict(slot_values={'hotel-pricerange':['cheap']}))])]) for i in range(count)],fixture=True,ontology=['hotel-pricerange'])
    return normalize(name,[dict(mark,_id=f'l{i}',context='Long fixture passages. '*100,input='Where?',answers=['here'],length=2000) for i in range(count)],fixture=True)


def dataset(root,name='coqa',count=4):
    p=Path(root)/name;p.mkdir();rows=samples(name,count)
    (p/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(p/'i5_dataset_manifest.json',dict(dataset=name,version=rows[0]['version'],samples=len(rows),
        samples_sha256=sha(p/'samples.jsonl'),TEST_FIXTURE_ONLY=True))
    seal(p);return p


class DataQuality(unittest.TestCase):
    def test_all_schemas(self):
        for name in ('coqa','snips','multiwoz','qasper','2wikimqa','multi_news'):
            self.assertTrue(all(validate_sample(r) for r in samples(name)))
    def test_missing_gold_rejected(self):
        for name in ('coqa','snips','qasper'):
            r=samples(name)[0];r['references']=[]
            with self.assertRaises(ValueError):validate_sample(r)
    def test_unknown_version_rejected(self):
        r=samples('multiwoz')[0];r['version']='2.1'
        with self.assertRaises(ValueError):validate_sample(r)
    def test_coqa_no_current_answer_leak_and_prior_history(self):
        rows=samples();tok=FixtureTokenizer();p=prepare_prompt(rows[0],tok,2048,32)
        self.assertNotIn('SECRET_CURRENT_ANSWER',p['input_text'])
        self.assertIn('SECRET_CURRENT_ANSWER',prepare_prompt(rows[1],tok,2048,32)['input_text'])
    def test_coqa_f1_leave_one_out(self):
        r=samples()[0];r['references']=['red blue','red','red blue']
        self.assertAlmostEqual(score(r,'red','v')['score'],8/9)
        r['references']=['The cat'];self.assertEqual(score(r,'cat','v')['score'],1)
    def test_snips_strict_parser_macro_f1(self):
        rows=samples('snips',7);scores=[score(r,r['references'][0],'v') for r in rows]
        self.assertEqual(aggregate(scores)['macro_f1'],1)
        self.assertTrue(score(rows[0],'Intent: '+LABELS[0],'v')['invalid_prediction'])
        self.assertEqual(score(rows[0],LABELS[0].lower(),'v')['score'],0)
    def test_dst_correct_alternative_invalid_and_duplicate(self):
        r=samples('multiwoz')[0];r['references']['hotel-pricerange'].append('inexpensive')
        self.assertEqual(score(r,'{"hotel-pricerange":"INEXPENSIVE"}','v')['score'],1)
        for text in ('[]','{"other":"cheap"}','{"hotel-pricerange":[]}','{"hotel-pricerange":"cheap","hotel-pricerange":"cheap"}'):
            self.assertTrue(score(r,text,'v')['invalid_prediction'])
        self.assertEqual(score(r,'{}','v')['slot_fn'],1)
    def test_empty_dst_reference_valid(self):
        r=samples('multiwoz')[0];r['references']={}
        self.assertEqual(score(r,'{}','v')['score'],1)
        self.assertEqual(score(r,'','v')['score'],0)
    def test_longbench_qa_and_fixture_rouge_separated(self):
        self.assertEqual(score(samples('qasper')[0],'here','v')['score'],1)
        r=score(samples('multi_news')[0],'here','v')
        self.assertTrue(r['TEST_FIXTURE_ONLY'])
    def test_missing_snapshot_and_hash_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=dataset(d);self.assertEqual(len(load_dataset(p,fixture=True)[0]),8)
            with self.assertRaises(ValueError):load_dataset(p)
            (p/'samples.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):load_dataset(p,fixture=True)
    def test_real_scope_rejects_fixture(self):
        with self.assertRaises(ValueError):normalize('snips',[dict(id='1',text='play',intent='PlayMusic',TEST_FIXTURE_ONLY=True)])
    def test_multinews_missing_real_scorer_fails_closed(self):
        from unittest.mock import patch
        from semcache.edgelora.r5_i5_quality import validate_scorers
        with patch.dict('sys.modules',{'rouge':None}):
            with self.assertRaisesRegex(ValueError,'SCORER_UNAVAILABLE'):validate_scorers(['multi_news'])
    @unittest.skipUnless(importlib.util.find_spec('rouge'), 'rouge==1.0.1 absent; real ROUGE-L package unverified locally')
    def test_pinned_rouge_known_identical_reference(self):
        from semcache.edgelora.r5_i5_quality import validate_scorers
        validate_scorers(['multi_news']);s=samples('multi_news')[0]
        self.assertAlmostEqual(score(s,s['references'][0],'v')['score'],1.,places=6)


class ContextWorkload(unittest.TestCase):
    def test_context_reservation_and_head_tail(self):
        r=samples('qasper')[0];p=prepare_prompt(r,FixtureTokenizer(),512,32,allow_truncation=True)
        self.assertLessEqual(len(p['token_ids'])+32,512);self.assertTrue(p['truncated'])
        self.assertEqual(p['evaluation_label'],'LONGBENCH_DERIVED_CONTEXT_LIMITED_EVALUATION')
        self.assertEqual(p['sample']['references'],r['references']);self.assertIn(r['question'],p['input_text'])
        self.assertEqual(p['removed_context_tokens']+p['retained_context_tokens'],p['original_context_tokens'])
    def test_question_and_history_not_truncated(self):
        r=samples()[0];r['question']='x'*2048
        with self.assertRaisesRegex(ValueError,'FIXED_FIELDS'):prepare_prompt(r,FixtureTokenizer(),2048,32,allow_truncation=True)
    def test_no_silent_truncation(self):
        with self.assertRaisesRegex(ValueError,'CONTEXT'):prepare_prompt(samples('qasper')[0],FixtureTokenizer(),512,32)
    def test_deterministic_selection(self):
        rows=samples();kw=dict(count=4,seed=42,context_limit=2048,budget=4)
        a,_=select(rows,FixtureTokenizer(),**kw);b,_=select(list(reversed(rows)),FixtureTokenizer(),**kw)
        self.assertEqual([p['sample']['id'] for p in a],[p['sample']['id'] for p in b])
    def test_insufficient_data_explicit_report(self):
        with self.assertRaises(EligibilityError) as c:select(samples(),FixtureTokenizer(),count=100,seed=1,context_limit=2048,budget=4)
        self.assertEqual(c.exception.report['eligible'],8)
    def test_controlled_slice_and_adapter_routing(self):
        prompts=[prepare_prompt(s,FixtureTokenizer(),2048,4) for s in samples()[:3]]
        for p in prompts:p.update(strict=True,source_semantic_text='semantic only')
        ts=requests(prompts,track='CONTROLLED',prefix=64,special=dict(bos_token_id=0,eos_token_id=6,pad_token_id=0))
        self.assertEqual(len(ts),6)
        for src,target in zip(ts[::2],ts[1::2]):
            self.assertEqual(src['token_ids'],target['token_ids'][:64]);self.assertFalse(src['scored'])
            self.assertEqual(src['adapter_id'],target['adapter_id']);self.assertNotIn('expected_hit',target)
        from semcache.edgelora.r5_i5_workload import routing
        self.assertEqual([t['user_id'] for t in ts[::2]],[routing(p['sample'])['user_id'] for p in prompts])
    def test_natural_has_no_injected_source(self):
        p=prepare_prompt(samples('snips')[0],FixtureTokenizer(),2048,4);p['strict']=True
        ts=requests([p],track='NATURAL',prefix=None,special=dict(bos_token_id=0,eos_token_id=6,pad_token_id=0))
        self.assertEqual(len(ts),1);self.assertTrue(ts[0]['scored'])
    def test_resource_raw_cost_and_1024_capacity(self):
        task=dict(token_ids=list(range(1025)),policy=dict(max_new_tokens=32),scored=True,strict=False)
        r=resources([task],evidence_budget=16*2**30)
        self.assertEqual(r['raw_qkv_bytes_per_token'],491520)
        self.assertGreaterEqual(r['max_source_tokens_fitting'],1024)
    def test_explicit_plan_count_and_balanced_orders(self):
        with tempfile.TemporaryDirectory() as d:
            path=dataset(d,'snips',10)
            p=prepare_experiment({'snips':path},TinyTokenizer(),{'fixture':True},2048,
                dict(bos_token_id=0,eos_token_id=6,pad_token_id=0),Path(d)/'plan',fixture=True,budgets={'snips':2},block_size=5)
            self.assertEqual(len(p['blocks']),4)
            self.assertEqual(p['blocks'][0]['condition_order'],list(CONDITIONS))
            self.assertEqual(p['blocks'][2]['condition_order'],list(reversed(CONDITIONS)))
            self.assertEqual(p['blocks'][3]['condition_order'],list(reversed(p['blocks'][1]['condition_order'])))


class StatisticsEvidence(unittest.TestCase):
    def test_distribution_and_pair_direction(self):
        self.assertEqual(distribution([1,2,3,4])['median'],2.5)
        self.assertEqual(paired({'x':10},{'x':8},'x')['percentage_delta'],-20)
    def test_grouped_ci_not_request_pseudoreplication(self):
        self.assertFalse(grouped_ci([dict(absolute_delta=1,independent_block_group='a')]*100)['available'])
        self.assertTrue(grouped_ci([dict(absolute_delta=i,independent_block_group=str(i)) for i in range(3)])['available'])
    def test_shared_conversation_merges_blocks_and_repetitions(self):
        from semcache.edgelora.r5_i5_summary import dependency_clusters
        rows=[dict(group_id='conversation_a',independent_block_group='b0'),
              dict(group_id='conversation_a',independent_block_group='b1'),
              dict(group_id='conversation_b',independent_block_group='b1'),
              dict(group_id='independent',independent_block_group='b2')]
        result=dependency_clusters(rows)
        self.assertEqual(len({r['independent_block_group'] for r in result[:3]}),1)
        self.assertNotEqual(result[0]['independent_block_group'],result[-1]['independent_block_group'])
    def test_no_overwrite_and_extra_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=fresh(Path(d)/'run');write(p/'x.json',{'x':1});seal(p);verify(p)
            with self.assertRaises(ValueError):fresh(p)
            (p/'extra').write_text('x')
            with self.assertRaises(ValueError):verify(p)


class ProtocolRegression(unittest.TestCase):
    def test_extended_context_preserves_legacy_bound(self):
        from semcache.edgelora.r5_i5_protocol import context
        from semcache.edgelora.r4c_session import context as legacy
        from semcache.edgelora.r4c_workload import policy,termination
        special=dict(bos_token_id=0,eos_token_id=6,pad_token_id=0);p=policy(special,'quality',128)
        ids=[1]*1100
        c=context('T',0,ids,ids,0,p,'COLD_MISS','0'*64,context_limit=2048)
        self.assertEqual(c['input_length'],1100)
        with self.assertRaises(ValueError):legacy('T',0,ids,ids,0,p,'COLD_MISS','0'*64,context_limit=2048)
        self.assertEqual(termination(p,[6]),'EOS');self.assertEqual(termination(p,[1]*128),'LENGTH_CAP')
        with self.assertRaises(ValueError):context('T',0,[1]*2000,[1]*2000,0,p,'COLD_MISS','0'*64,context_limit=2048)
    def test_suffix_protocol_1024_source_and_completion(self):
        from semcache.edgelora.r5_i5_protocol import context,ForwardOrder,VERSION
        from semcache.edgelora.r4c_workload import policy
        p=policy(dict(bos_token_id=0,eos_token_id=6,pad_token_id=0),'fixed',1)
        ids=[1]*1024;order=ForwardOrder('0'*64,1,2048)
        c=context('S',0,ids,ids,0,p,'COLD_MISS','0'*64,context_limit=2048)
        h=dict(metadata={'r4c_forward':c},kind='HIDDEN',request_id='S.s00',layer_id=0,shape=[1,1024,4],dtype='float32')
        order.accept(h,dict(hidden_size=4,hidden_dtype='float32'));order.advance()
        order.control(dict(version=VERSION,cohort_sha256='0'*64,action='COMPLETE_REQUEST',parent_request_id='S',generated_token_ids=[1],termination='LENGTH_CAP'))
        grant=dict(entry_id='1'*64,source_request_id='S',prefix_length=1024,prefix_token_ids=ids,proof_sha256='2'*64)
        order.control(dict(grant,version=VERSION,cohort_sha256='0'*64,action='REGISTER_PREFIX'))
        c=context('T',0,[2],ids+[2],1024,p,'PARTIAL_PREFIX_HIT','0'*64,grant,context_limit=2048)
        self.assertEqual(c['input_length'],1);self.assertEqual(c['past_length'],1024)
    def test_missing_priming_capacity_rejected(self):
        task=dict(token_ids=[1]*1024,policy=dict(max_new_tokens=1),scored=False,strict=False)
        with self.assertRaisesRegex(ValueError,'CAPACITY'):resources([task],capacity=64*2**20,evidence_budget=16*2**30)


if __name__=='__main__':unittest.main()
