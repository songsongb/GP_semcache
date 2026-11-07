"""CPU shadow-policy tests; no physical reuse evidence is fabricated."""
import copy,importlib.util,unittest
from unittest.mock import patch
from semcache.edgelora.r5_mstage_bridge import MStageShadow,shadow_controller_factory
from semcache.edgelora.r5_i5r3b_transport import predicted_request
from semcache.edgelora.r5_multi_artifacts import registry_for
from semcache.edgelora.r4c_workload import policy

class Shadow(unittest.TestCase):
    def setUp(self):
        from test_r5_i5r3b import Transport
        self.assets=Transport();self.assets.setUp();self.users=registry_for(['Adapter_A','Adapter_B','Adapter_C'],'expanded')
        self.events=[];r=self.req('S',[1,2,3,4,5]);self.bridge=MStageShadow(r['physical_identity'],self.users,trace=self.events.append)
    def req(self,rid,ids,user='User_01',adapter='Adapter_A'):
        t=dict(request_id=rid,token_ids=ids,input_text='fixture',user_id=user,adapter_id=adapter,dataset='qasper',sample_id=rid,request_sequence_number=0,policy=policy(dict(bos_token_id=2,eos_token_id=2,pad_token_id=1),'quality',2))
        return predicted_request(t,self.assets.pm,self.assets.specs[adapter],self.assets.tokenizer)
    def source(self,ids=(1,2,3,4,5)):
        self.bridge.begin(self.req('S',list(ids)),{'cluster_id':0})
        with patch('semcache.edgelora.r5_mstage_bridge.actual_attention_impact',return_value=2.):self.bridge.complete(completed=True,attentions=['CPU_FIXTURE_ONLY'])
    def test_same_and_shifted_candidates_no_grants(self):
        self.source();r=self.bridge.begin(self.req('T',[9,1,2,3,4,5],user='User_02'),{'cluster_id':0})
        self.assertTrue(any('SHIFTED_POSITION' in x['rejections'] for x in r['candidates']));self.assertTrue(r['selected_spans'])
        self.assertEqual(self.bridge.physical_hits,0);self.assertEqual(self.bridge.skipped_rows,0)
        with self.assertRaises(NotImplementedError):self.bridge.physical_grant()
    def test_same_position_and_overlap_resolution(self):
        self.source();r=self.bridge.begin(self.req('T',[1,2,3,4,5,6]),{'cluster_id':0})
        self.assertEqual(r['selected_spans'],[[0,3]]);self.assertFalse(r['candidates'][0]['rejections'])
    def test_duplicate_window_admission_and_provenance(self):
        self.source((1,1,1,1,1));self.assertEqual(len(self.bridge.cache.entries),1)
        entry=next(iter(self.bridge.cache.entries.values()));self.assertEqual(entry.qkv_metadata['source_request_id'],'S');self.assertIsNone(entry.tensors)
        r=self.bridge.begin(self.req('T',[1]*8),{'cluster_id':0});self.assertEqual(r['selected_spans'],[[0,3],[3,6]])
    def test_cluster_separation(self):
        self.source();r=self.bridge.begin(self.req('T',[1,2,3,4,5]),{'cluster_id':1});self.assertEqual(r['candidates'],[])
    def test_cross_adapter_and_physical_identity_rejected(self):
        for req in (self.req('X',[1,2,3],user='User_11',adapter='Adapter_B'),):
            with self.assertRaises(ValueError):self.bridge.begin(req,{'cluster_id':0})
    def test_cross_user_permission(self):
        self.source();self.users['users']['User_02']['allow_cross_user_reuse']=False
        r=self.bridge.begin(self.req('T',[1,2,3],user='User_02'),{'cluster_id':0});self.assertFalse(r['selected_spans'])
    def test_missing_actual_attention_is_not_zero_importance(self):
        self.bridge.begin(self.req('S',[1,2,3]),{'cluster_id':0});self.assertEqual(self.bridge.complete(completed=True),[]);self.assertFalse(self.bridge.cache.entries)
        self.assertEqual(self.events[-1]['reason'],'ACTUAL_ATTENTION_PROBABILITIES_NOT_RECORDED')
    def test_incomplete_duplicate_and_dispose(self):
        self.bridge.begin(self.req('S',[1,2,3]),{'cluster_id':0})
        with self.assertRaises(ValueError):self.bridge.complete(completed=False)
        self.bridge.complete(completed=True)
        with self.assertRaises(ValueError):self.bridge.begin(self.req('S',[1,2,3]),{'cluster_id':0})
        self.bridge.dispose();self.assertFalse(self.bridge.cache.entries)
    def test_fais_age_appearance_vs_actual_frequency(self):
        self.source();entry=next(iter(self.bridge.cache.entries.values()));self.assertEqual(entry.frequency,0);self.assertEqual(entry.impact,2.)
        self.bridge.begin(self.req('T',[1,2,3,4,5]),{'cluster_id':0});self.assertEqual(entry.frequency,0);self.assertEqual(entry.age(self.bridge.cache.now),1)
        self.assertGreater(self.bridge.metrics.frequencies[entry.key],1)
        self.assertAlmostEqual(self.bridge.chu_preview(entry,4.),2.4);self.assertEqual(entry.impact,2.)
        self.assertTrue(all(not x['physical_hit'] and x['skipped_wire_bytes']==0 for x in self.events))
    def test_pbr_observed_history_only(self):
        self.source();entry=next(iter(self.bridge.cache.entries.values()));self.bridge.metrics.history.append(0,'observed',2,{entry.key:6.})
        self.bridge.metrics.pbr(0);self.assertEqual(entry.impact,4.);self.assertEqual(entry.frequency,0)
    def test_capacity_eviction_and_no_tensor_claim(self):
        self.bridge.cache.capacity_bytes=3*3*32*2560*2
        self.source();self.assertLessEqual(self.bridge.cache.logical_cache_bytes,self.bridge.cache.capacity_bytes)
        self.assertEqual(self.bridge.cache.physical_tensor_bytes,0);self.assertTrue(any(x['kind']=='MSTAGE_EVICT' for x in self.events))
    def test_admission_denial(self):
        from semcache.cache.admission import AdmissionPolicy
        self.bridge.cache.admission=AdmissionPolicy(threshold=1.)
        self.source();self.assertFalse(self.bridge.cache.entries)
    def test_entry_bound_and_no_stale_lookup(self):
        self.source(tuple(range(100)));self.assertLessEqual(len(self.bridge.cache.entries),64)
        self.assertNotIn((0,(0,1,2)),self.bridge.cache.entries)
        r=self.bridge.begin(self.req('T',[0,1,2]),{'cluster_id':0});self.assertFalse(r['candidates'])
    def test_same_position_different_causal_history(self):
        self.source();r=self.bridge.begin(self.req('T',[9,2,3,4,5]),{'cluster_id':0})
        self.assertTrue(any('CAUSAL_PREFIX_DIFFERS' in c['rejections'] and 'SHIFTED_POSITION' not in c['rejections'] for c in r['candidates']))
    def test_factory_restored_on_exception(self):
        from types import SimpleNamespace
        fake=SimpleNamespace(I5Controller=object)
        with self.assertRaises(RuntimeError):
            with shadow_controller_factory(fake):self.assertIsNot(fake.I5Controller,object);raise RuntimeError('fixture')
        self.assertIs(fake.I5Controller,object)
    def test_actual_i5_shadow_controller_integration(self):
        from types import SimpleNamespace
        import test_r5_i5_controller as fixture
        from semcache.edgelora.r5_i5_controller import I5Controller
        fixture.Controller.setUpClass()
        try:
            helper=fixture.Controller();runtime=SimpleNamespace(I5Controller=I5Controller)
            with shadow_controller_factory(runtime,trace=self.events.append),patch.object(fixture,'I5Controller',side_effect=lambda *a,**k:runtime.I5Controller(*a,**k)):
                owner,c,identity=helper.manager('SEMANTIC_NO_REUSE');req=helper.req('S',[1,2,3,4],identity)
                event,found=c.decide(req);self.assertIsNone(found)
                c.admit(req,event['assignment'],None,None,completed=True)
                self.assertFalse(c.mstage.cache.entries);self.assertEqual(owner.bytes,0)
                self.assertTrue(any(x['kind']=='MSTAGE_ADMISSION_UNAVAILABLE' for x in self.events))
            self.assertIs(runtime.I5Controller,I5Controller)
        finally:fixture.Controller.tearDownClass()
    @unittest.skipUnless(importlib.util.find_spec('torch'),'actual attention reduction NOT_RUN: Torch unavailable')
    def test_actual_attention_values(self):
        import torch
        self.bridge.begin(self.req('S',[1,2,3]),{'cluster_id':0})
        self.bridge.complete(completed=True,attentions=[torch.ones(1,1,3,3)/3])
        self.assertTrue(self.bridge.cache.entries)

if __name__=='__main__':unittest.main()
