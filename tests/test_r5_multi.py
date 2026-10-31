"""Multi-adapter preparation/authorization/cache tests; no ML dependency."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest

from semcache.semantic.encoder import ControlledEncoder
from semcache.edgelora.communication import digest
from semcache.edgelora.r3a_artifacts import read,validate_prepared,pack
from semcache.edgelora.r3a_tiny import prepare_tiny,linear,f32
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.r4a_fixture import TinyBackend
from semcache.edgelora.r4b2_kv import verify_raw_kv
from semcache.edgelora.r4c_prepare import TinyTokenizer
from semcache.edgelora.r4c_session import context
from semcache.edgelora.r5_contracts import request
from semcache.edgelora.r5_semantic_index import SemanticIndex
from semcache.edgelora.r5_workload import generation_policy
from semcache.edgelora.r5_multi_artifacts import prepare,validate,load,runtime_view,physical,resolve,sharing,validate_registry
from semcache.edgelora.r5_multi_controller import MultiOwner,MultiController,public_result
from semcache.edgelora.r5_multi_session import VERSION,SessionGuard
from semcache.edgelora.r5_multi_analysis import fidelity,gates,HISTORY
from semcache.edgelora.r5_multi_runtime import workload


class MultiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.base=prepare_tiny(cls.root/'base');cls.prepared=prepare(cls.base,cls.root/'adapters')
        cls.manifest,cls.pm,cls.adapters,cls.registry=validate(cls.prepared)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def setUp(self):self.scratch=tempfile.TemporaryDirectory();self.work=Path(self.scratch.name)
    def tearDown(self):self.scratch.cleanup()

    def manager(self,aid='Adapter_A',capacity=4*3*32*4*4,registry=None):
        registry=registry or self.registry;spec=self.adapters[aid];view=runtime_view(self.pm,spec)
        ident,contract=physical(view,self.manifest['tokenizer'],'cpu','TEST_FIXTURE_ONLY',{'scope':'TEST_FIXTURE_ONLY','output_attentions':False},spec,registry)
        index=SemanticIndex(ControlledEncoder({t:[1.,0.] for t in ('source','target','other')}))
        owner=MultiOwner(view,'pinned_prepare','TEST_FIXTURE_ONLY','cpu',capacity,index=index,tokenizer=self.manifest['tokenizer'],sources={'fixture':'unit'},
            fixture=True,contract=contract,observed_cache_type=tuple,spec=spec,registry=registry)
        return owner,MultiController(owner,index,ident,registry=registry),ident
    def req(self,rid,ids,ident,user='User_01',text='source'):
        return request(dict(request_id=rid,user_id=user,adapter_id=ident['adapter_id'],input_text=text,dataset='TEST_FIXTURE_ONLY',sample_id=rid,request_sequence_number=0),
                       ids,generation_policy(TinyTokenizer(),budget=2),ident,fixture=True)
    def computed(self,ids,aid='Adapter_A'):
        base=read(self.base/'r3a_tiny_base.json');matrices=load(self.prepared,self.adapters[aid],torch_backend=False);blocks={};h=base['hidden']
        def project(l,r,x,native):
            delta=linear(linear(x,matrices[l,r,'A'],h,8),matrices[l,r,'B'],8,h)
            vals=[f32(a+b) for a,b in zip(native(),delta)];blocks.setdefault(l,{})[r]=ArrayTensor(vals,[1,len(ids),h]);return vals
        out=TinyBackend(base).forward(ids,[1]*len(ids),None,project,True)
        blocks={l:tuple(v[r] for r in 'qkv') for l,v in blocks.items()}
        return blocks,verify_raw_kv(blocks,out.past_key_values,1,runtime_view(self.pm,self.adapters[aid])['identity'],'cpu')
    def source(self,owner,controller,ident,rid='S',ids=(1,2,3,4)):
        req=self.req(rid,list(ids),ident);event,_=controller.decide(req);blocks,proof=self.computed(list(ids))
        return controller.admit(req,event['assignment'],blocks,proof,completed=True)

    def test_actual_distinct_matrix_fingerprints_and_coverage(self):
        fps={s['matrix_sha256'] for s in self.adapters.values()}
        self.assertEqual(len(fps),2);self.assertNotIn(self.pm['identity']['adapter_matrix_sha256'],fps)
        for s in self.adapters.values():
            self.assertEqual(len(load(self.prepared,s,torch_backend=False)),32*6);self.assertEqual(s['rank'],8)
            self.assertEqual(s['scale'],1.);self.assertFalse(s['trained_adapter'])
    def test_reproducible_independent_preparation(self):
        other=prepare(self.base,self.work/'other');_,_,specs,_=validate(other)
        self.assertEqual({a:s['matrix_sha256'] for a,s in specs.items()},{a:s['matrix_sha256'] for a,s in self.adapters.items()})
    def test_expanded_third_adapter_and_thirty_user_assignment(self):
        path=prepare(self.base,self.work/'expanded',stage='expanded');m,_,specs,registry=validate(path)
        self.assertEqual(len(specs),3);self.assertEqual(len({s['matrix_sha256'] for s in specs.values()}),3)
        self.assertEqual(len(registry['users']),30);self.assertEqual(resolve(registry,'User_21'),'Adapter_C')
        self.assertEqual(specs['Adapter_C']['source_seed'],155921)
        self.assertEqual(specs['Adapter_A']['matrix_sha256'],self.adapters['Adapter_A']['matrix_sha256'])
        rows=workload(registry,fixture=True);self.assertEqual(len(rows),30)
        self.assertEqual([r['adapter_id'] for r in rows[:6]],['Adapter_A','Adapter_B','Adapter_C']*2)
        self.assertEqual(len({r['user_id'] for r in rows}),30)
        self.assertTrue(all(len([r for r in rows if r['adapter_id']==aid])==10 for aid in specs))
    def test_unsupported_scale_or_rank_is_not_loaded(self):
        for key,value in (('rank',4),('scale',2.),('matrix_dtype','float16')):
            spec=deepcopy(self.adapters['Adapter_A']);spec[key]=value
            with self.assertRaisesRegex(ValueError,'LoRA contract'):load(self.prepared,spec,torch_backend=False)
    def test_missing_or_combined_projection_role_is_rejected_before_loading(self):
        for role in ('','qk','kv','unknown'):
            spec=deepcopy(self.adapters['Adapter_A']);spec['matrices'][0]['role']=role
            with self.assertRaisesRegex(ValueError,'layout/coverage'):load(self.prepared,spec,torch_backend=False)
    def test_new_contract_is_not_forged_legacy_preparation(self):
        with self.assertRaises(FileNotFoundError):validate_prepared(self.prepared)
        self.assertEqual(validate_prepared(self.base)[0]['identity']['adapter'],'user_a')
    def test_corrupt_manifest_rejected(self):
        copied=self.work/'copy';shutil.copytree(self.prepared,copied);path=copied/'Adapter_A.json';path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaises(ValueError):validate(copied)
    def test_incorrect_matrix_loading_rejected(self):
        copied=self.work/'copy';shutil.copytree(self.prepared,copied)
        (copied/'Adapter_A.bin').write_bytes((copied/'Adapter_B.bin').read_bytes())
        with self.assertRaisesRegex(ValueError,'fingerprint'):load(copied,self.adapters['Adapter_A'],torch_backend=False)
    def test_registry_wrong_or_unknown_user_adapter_and_corruption(self):
        for user,claimed in (('User_01','Adapter_B'),('unknown','Adapter_A'),('User_01','unprepared')):
            with self.assertRaisesRegex(ValueError,'UNAUTHORIZED'):resolve(self.registry,user,claimed)
        bad=deepcopy(self.registry);bad['users']['User_01']['adapter_id']='Adapter_B'
        with self.assertRaisesRegex(ValueError,'corrupt'):validate_registry(bad,self.adapters)
    def test_same_adapter_cross_user_hit_is_automatic(self):
        owner,controller,ident=self.manager();grant=self.source(owner,controller,ident)
        event,found=controller.decide(self.req('T',[1,2,3,4,5],ident,user='User_02',text='target'))
        self.assertEqual(event['outcome'],'PHYSICAL_PREFIX_HIT');self.assertEqual(found[1]['entry_id'],grant['entry_id'])
    def test_wrong_user_rejected_before_semantic_encoding(self):
        owner,controller,ident=self.manager();before=controller.index.clusterer.queries
        with self.assertRaisesRegex(ValueError,'UNAUTHORIZED'):controller.decide(self.req('BAD',[1,2,3,4],ident,user='User_03'))
        self.assertEqual(controller.index.clusterer.queries,before);self.assertEqual(owner.bytes,0)
    def test_cross_adapter_owner_cannot_consume_identical_tokens(self):
        owner,controller,ident=self.manager();grant=self.source(owner,controller,ident)
        _,_,other=self.manager('Adapter_B');owner.active_request=self.req('BAD',[1,2,3,4,5],other,user='User_03')
        with self.assertRaisesRegex(ValueError,'UNAUTHORIZED_CACHE_NAMESPACE'):owner.selected(grant['entry_id'],{'token_ids':[1,2,3,4,5]},'BAD')
        self.assertEqual(sharing(self.registry,'User_01','User_03','Adapter_A','Adapter_B'),'CROSS_ADAPTER_REUSE_FORBIDDEN')
    def test_missing_cross_user_permission_rejects_candidate(self):
        registry=deepcopy(self.registry);registry['users']['User_02']['allow_cross_user_reuse']=False
        registry['sha256']=digest({k:v for k,v in registry.items() if k!='sha256'})
        owner,controller,ident=self.manager(registry=registry);self.source(owner,controller,ident)
        event,found=controller.decide(self.req('T',[1,2,3,4,5],ident,user='User_02',text='target'))
        self.assertIsNone(found);self.assertEqual(event['candidate_reasons'][0]['rejection'],'SHARING_NOT_AUTHORIZED')
    def test_different_sharing_groups_cannot_share_with_same_adapter(self):
        registry=deepcopy(self.registry);registry['users']['User_02']['share_group']='private'
        self.assertEqual(sharing(registry,'User_01','User_02','Adapter_A','Adapter_A'),'SHARING_NOT_AUTHORIZED')
    def test_corrupt_cached_kv_fails_closed_and_invalidates_own_index(self):
        owner,controller,ident=self.manager();grant=self.source(owner,controller,ident)
        entry=next(iter(owner.entries[grant['entry_id']].pool.entries.values()));entry.tensors[0][1].values[0]+=1.
        with self.assertRaisesRegex(ValueError,'INVALID_CACHE_INTEGRITY'):controller.decide(self.req('CORRUPT',[1,2,3,4,5],ident,user='User_02',text='target'))
        self.assertNotIn(grant['entry_id'],controller.index.entries);self.assertEqual(owner.bytes,0)
    def test_session_guard_rejects_wrong_adapter_and_mapping_changes(self):
        _,_,a=self.manager();_,_,b=self.manager('Adapter_B');req=self.req('R',[1,2,3,4],a)
        ctx=context('R',0,[1,2,3,4],[1,2,3,4],0,req['generation_policy'],'NO_REUSE','a'*64)
        route=dict(schema=VERSION,registry_sha256=self.registry['sha256'],request=req)
        guard=SessionGuard(self.registry,b)
        with self.assertRaisesRegex(ValueError,'identity mismatch'):guard.accept(route,ctx)
        self.assertFalse(guard.requests)
        guard=SessionGuard(self.registry,a);guard.accept(route,ctx)
        req2=self.req('R',[1,2,3,4],a,user='User_02');route['request']=req2
        with self.assertRaisesRegex(ValueError,'mapping changed'):guard.accept(route,ctx)
    def test_eviction_invalidates_index_and_causes_miss(self):
        owner,controller,ident=self.manager();first=self.source(owner,controller,ident)
        self.source(owner,controller,ident,rid='Q',ids=(2,3,4,5))
        self.assertNotIn(first['entry_id'],owner.entries);self.assertNotIn(first['entry_id'],controller.index.entries)
        event,found=controller.decide(self.req('OLD',[1,2,3,4,5],ident,text='target'))
        self.assertIsNone(found);self.assertNotEqual(event['outcome'],'PHYSICAL_PREFIX_HIT')
    def test_public_result_does_not_expose_foreign_or_own_cache_metadata(self):
        row=dict(request_id='R',status='COMPLETE',generated_token_ids=[1],generated_text='own',termination='LENGTH_CAP',
                 source='private',candidate_count=3,rejected_cross_adapter_candidates=9,adapter_id='private')
        self.assertEqual(set(public_result(row)),{'request_id','status','generated_token_ids','generated_text','termination'})
    def test_numerical_failure_reporting_preserves_historical_classification(self):
        _,_,ident=self.manager();common=dict(adapter_id='Adapter_A',physical_identity=ident,logits_shape=[1,1,3],generated_token_ids=[2],generated_text='x')
        (self.work/'a').write_bytes(pack([0.,1.,2.],'float32'));(self.work/'b').write_bytes(pack([.02,1.02,2.02],'float32'))
        row=fidelity(dict(common,request_id='N_R',logits_file='a'),dict(common,request_id='R',logits_file='b'),self.work,self.work)
        self.assertTrue(row['generated_tokens_equal']);self.assertFalse(row['frozen_tolerance_pass'])
        self.assertEqual(HISTORY['frozen_numerical_tolerance'],'FAIL_R5_03_R5_08');self.assertFalse(gates()['correctness_qualified_speedup'])


if __name__=='__main__':unittest.main()
