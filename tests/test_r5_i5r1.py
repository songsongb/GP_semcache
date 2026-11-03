"""CPU-only R1 workload, immutable provenance, cost scopes and safety controls."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import digest
from semcache.edgelora.r3a_artifacts import read,write
from semcache.edgelora.r3a_tiny import prepare_tiny
from semcache.edgelora.r4c_prepare import TinyTokenizer
from semcache.edgelora.r5_i5_artifacts import prepare as prepare_adapters
from semcache.edgelora.r5_multi_artifacts import validate,tokenizer_identity
from semcache.edgelora.r5_i5_evidence import seal,verify,fresh
from semcache.edgelora.r5_i5r1_plan import prepare,validate_plan,pair_checks,inspect,PROFILES
from semcache.edgelora.r5_i5r1_analysis import costs,decide
from test_r5_i5 import dataset
import test_r5_i5_controller as controller_tests

SPECIAL=dict(bos_token_id=0,eos_token_id=6,pad_token_id=0)


class Planning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.base=prepare_tiny(cls.root/'base');cls.multi=prepare_adapters(cls.base,cls.root/'adapters','expanded')
        cls.manifest,cls.pm,cls.adapters,cls.registry=validate(cls.multi)
        cls.data=dataset(cls.root,'qasper',6)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def make(self,path,**kw):
        return prepare({'qasper':self.data},TinyTokenizer(),tokenizer_identity(self.pm),2048,SPECIAL,path,
            multi_prepared=self.multi,fixture=True,**kw)
    def test_smoke_genuine_sliced_source_and_all_strict(self):
        with tempfile.TemporaryDirectory() as d:
            plan=self.make(Path(d)/'plan');b=plan['blocks'][0];s,t=b['tasks']
            self.assertEqual(s['token_ids'],t['token_ids'][:64]);self.assertGreater(len(t['token_ids']),64)
            self.assertEqual((s['role'],t['role']),('PRIMING_SOURCE','SCORED_TARGET'))
            self.assertTrue(all(x['strict'] for x in b['tasks']));self.assertEqual(t['policy']['max_new_tokens'],8)
            self.assertEqual(s['policy']['max_new_tokens'],1);self.assertEqual(s['user_id'],t['user_id'])
            self.assertEqual(s['adapter_id'],'Adapter_A');self.assertFalse(b['pair_checks'][0]['runtime_HIT_authorized_by_static_check'])
            self.assertEqual(digest(validate_plan(Path(d)/'plan')),digest(plan))
    def test_numeric_prefix_authoritative_despite_decode_reencode(self):
        with tempfile.TemporaryDirectory() as d:
            s,t=self.make(Path(d)/'plan')['blocks'][0]['tasks']
            self.assertNotEqual(TinyTokenizer().encode(s['input_text']),s['token_ids'])
            self.assertEqual(s['token_ids'],t['token_ids'][:64])
    def test_all_cases_and_condition_orders(self):
        with tempfile.TemporaryDirectory() as d:
            plan=self.make(Path(d)/'plan',profile='COMPARISON')
            self.assertEqual(len(plan['blocks']),12)
            self.assertEqual({t['adapter_id'] for b in plan['blocks'] for t in b['tasks']},set(self.adapters))
            self.assertGreater(len({tuple(b['condition_order']) for b in plan['blocks']}),1)
            for b in plan['blocks']:
                self.assertLessEqual(max(len(t['token_ids'])+t['policy']['max_new_tokens'] for t in b['tasks']),2048)
                self.assertEqual(len(b['tasks']),2)
    def test_all_three_real_schema_dataset_profile(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);registry={'qasper':self.data,**{name:dataset(p,name,3) for name in ('2wikimqa','multi_news')}}
            plan=prepare(registry,TinyTokenizer(),tokenizer_identity(self.pm),2048,SPECIAL,p/'plan',multi_prepared=self.multi,
                fixture=True,profile='DATASETS')
            self.assertEqual(len(plan['blocks']),18)
            self.assertEqual({b['dataset'] for b in plan['blocks']},set(registry))
            self.assertTrue(all(b['pair_checks'][0]['exact_full_prefix'] for b in plan['blocks']))
    def test_repeat_targets_share_one_real_source(self):
        with tempfile.TemporaryDirectory() as d:
            plan=self.make(Path(d)/'plan',profile='AMORTIZATION')
            self.assertEqual(len(plan['blocks']),9)
            for b in plan['blocks']:
                self.assertEqual(sum(not t['scored'] for t in b['tasks']),1)
                self.assertEqual(len(b['tasks'])-1,b['i5r1_target_repeats'])
                self.assertEqual(len({t['request_id'] for t in b['tasks']}),len(b['tasks']))
    def test_deterministic_selection_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.make(Path(d)/'a');b=self.make(Path(d)/'b');self.assertEqual(a,b)
            info=inspect(Path(d)/'a');self.assertEqual(info['valid_array'],'0-0%1')
            self.assertEqual(info['maximum_physical_requests_per_condition'],2)
    def test_authorized_cross_user_and_stable_source_groups(self):
        with tempfile.TemporaryDirectory() as d:
            plan=self.make(Path(d)/'plan',profile='COMPARISON',cases=('64:256',),sharing_mode='AUTHORIZED_CROSS_USER')
            for b in plan['blocks']:
                s,t=b['tasks'];self.assertNotEqual(s['user_id'],t['user_id']);self.assertEqual(s['adapter_id'],t['adapter_id'])
                self.assertEqual(len(pair_checks(b,self.registry,2048)),1)
    def test_invalid_prefix_order_and_adapter_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.make(Path(d)/'plan')['blocks'][0]
            for mutation in ('prefix','length','order','adapter'):
                x=deepcopy(b)
                if mutation=='prefix':x['tasks'][0]['token_ids'][0]=9000
                elif mutation=='length':x['tasks'][0]['token_ids']=x['tasks'][1]['token_ids'][:]
                elif mutation=='order':x['tasks'].reverse()
                else:x['tasks'][1]['adapter_id']='Adapter_B'
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):pair_checks(x,self.registry,2048)
    def test_missing_real_dataset_never_substituted(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'DATASET_NOT_READY'):self.make(Path(d)/'plan',datasets=('multi_news',))
    def test_corrupt_inventory_and_source_changes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'plan';self.make(path)
            (path/'i5r1_protocol.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'inventory'):validate_plan(path)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'plan';plan=self.make(path)
            with patch('semcache.edgelora.r5_i5r1_plan.source_identity',return_value=dict(plan['i5r1']['source'],commit='0'*40)):
                with self.assertRaisesRegex(ValueError,'pinned source'):validate_plan(path)
    def test_no_overwrite_and_narrow_new_namespace(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan';self.make(p)
            with self.assertRaises(ValueError):self.make(p)
        from semcache.edgelora.r3a_artifacts import ROOT
        with self.assertRaises(ValueError):fresh(ROOT/'results/r5/fidelity/I5R1_MUST_NOT_WRITE')


class Safety(unittest.TestCase):
    setUpClass=classmethod(controller_tests.Controller.setUpClass.__func__)
    tearDownClass=classmethod(controller_tests.Controller.tearDownClass.__func__)
    computed=controller_tests.Controller.computed;manager=controller_tests.Controller.manager;req=controller_tests.Controller.req;source=controller_tests.Controller.source
    def test_same_user_and_authorized_cross_user_real_fixture_consumption(self):
        for user in ('User_01','User_02'):
            o,c,i=self.manager();g=self.source(o,c,i);_,found=c.decide(self.req('T',[1,2,3,4,5],i,'target',user=user))
            self.assertEqual(found[1]['entry_id'],g['entry_id'])
    def test_unauthorized_same_adapter_user_rejected(self):
        o,c,i=self.manager();self.source(o,c,i)
        c.registry=deepcopy(c.registry);c.registry['users']['User_02']['allow_cross_user_reuse']=False
        e,found=c.decide(self.req('T',[1,2,3,4,5],i,'target',user='User_02'))
        self.assertIsNone(found);self.assertEqual(e['candidate_reasons'][0]['rejection'],'SHARING_NOT_AUTHORIZED')
    def test_cross_adapter_never_gets_foreign_tensors(self):
        o,c,i=self.manager();self.source(o,c,i)
        with self.assertRaises(ValueError):c.decide(self.req('T',[1,2,3,4,5],i,'target',user='User_03'))
    def test_equal_length_and_nonprefix_subsequence_are_denied(self):
        for ids in ([1,2,3,4],[9,1,2,3,4]):
            o,c,i=self.manager();self.source(o,c,i);e,found=c.decide(self.req('T',ids,i,'target'))
            self.assertTrue(e['candidates']);self.assertIsNone(found)
    def test_missing_evicted_and_incomplete_sources(self):
        o,c,i=self.manager();g=self.source(o,c,i);o.remove(g['entry_id'],'EVICT','unit')
        _,found=c.decide(self.req('T',[1,2,3,4,5],i,'target'));self.assertIsNone(found)
        req=self.req('U',[1,2,3,4],i);e,_=c.decide(req)
        with self.assertRaises(ValueError):c.admit(req,e['assignment'],None,None,completed=False)
    def test_position_and_mask_contracts_remain_strict(self):
        for key in ('positions','attention_mask'):
            o,c,i=self.manager();self.source(o,c,i);r=self.req('T',[1,2,3,4,5],i,'target')
            r[key][0]=77;copy=dict(r);copy.pop('request_sha256');r['request_sha256']=digest(copy)
            with self.assertRaises(ValueError):c.decide(r)


class CostScopes(unittest.TestCase):
    def test_nonempty_and_empty_output_quality_are_distinct(self):
        from semcache.edgelora.r5_i5_quality import score
        from test_r5_i5 import samples
        sample=samples('qasper',1)[0]
        good=score(sample,'here','I5_PROMPTS_V1');empty=score(sample,'','I5_PROMPTS_V1')
        self.assertEqual(good['score'],1);self.assertFalse(good['empty_output'])
        self.assertEqual(empty['score'],0);self.assertTrue(empty['empty_output'])
    def test_warm_target_is_not_amortized_sequence(self):
        rows=[dict(scored=False,arrival_ns=0,completion_ns=100_000_000,application_e2e_ms=100,cache_admission_eviction_ms=40),
              dict(scored=True,arrival_ns=110_000_000,completion_ns=130_000_000,application_e2e_ms=20,cache_admission_eviction_ms=0)]
        c=costs(rows,10,5,dict(execution_start_ns=0,execution_end_ns=150_000_000,process_wall_ms=150))
        self.assertEqual(c['warm_target_mean_ms'],20);self.assertEqual(c['amortized_sequence_ms_per_target'],130)
        self.assertEqual(c['inter_request_gap_ms'],10);self.assertEqual(c['process_ms_per_target'],150)
    def test_cpu_and_numerical_failure_cannot_qualify_speedup(self):
        args=dict(functional=True,cache_integrity=True,numerical=False,tokens=True,nonempty=True,warm_delta=-10,amortized_delta=5)
        d=decide(fixture=False,**args);self.assertEqual(d['NUMERICAL_FIDELITY'],'FAIL');self.assertEqual(d['WARM_TARGET_BENEFIT'],'PASS')
        self.assertEqual(d['AMORTIZED_E2E_BENEFIT'],'FAIL');self.assertEqual(d['CORRECTNESS_QUALIFIED_SPEEDUP'],'UNESTABLISHED')
        self.assertEqual(decide(fixture=True,**args)['FUNCTIONAL_PHYSICAL_REUSE'],'CPU_FIXTURE_ONLY')


if __name__=='__main__':unittest.main()
