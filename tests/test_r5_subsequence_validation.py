"""D5 synthetic CPU integrity fixtures; no GPU or performance conclusions."""
import copy,importlib.util,multiprocessing,unittest,tempfile,threading,contextlib,io
from pathlib import Path
from unittest.mock import patch
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.communication import digest,Channel,connect,header
from semcache.edgelora.r5_subsequence_physical import PhysicalOwner,entry_proof,EXACT
from semcache.edgelora.r5_subsequence_integrity import EAGER,SCOPED,MODES
from semcache.edgelora.r5_subsequence_profile import Profiler
import test_r5_sparse_physical as fixture

class Session:
    def correction(self,l,r,x,positions):
        if not positions:return None
        if isinstance(x,ArrayTensor):return ArrayTensor([0.]*(len(positions)*4),[1,len(positions),4])
        import torch
        return torch.zeros((1,len(positions),4))

class Validation(unittest.TestCase):
    def setUp(self):self.f=fixture.Physical();self.f.setUp()
    def x(self,n,value=1.,torch_backend=False):
        if not torch_backend:return ArrayTensor([value]*(n*4),[1,n,4])
        import torch
        return torch.full((1,n,4),value,dtype=torch.float32)
    def native(self,x):
        return ArrayTensor([2*v for v in x.values],x.shape) if isinstance(x,ArrayTensor) else 2*x
    def project_all(self,owner,n,*,late_mismatch=False,torch_backend=False):
        for layer in range(2):
            for role in 'qkv':owner.project(layer,role,self.x(n,2. if late_mismatch and layer else 1.,torch_backend),self.native,Session())
    def complete(self,owner):
        # Explicit synthetic impact fixture isolates validation, not attention math.
        with patch('semcache.edgelora.r5_subsequence_physical.actual_attention_impact',return_value=1.):return owner.complete(completed=True,attentions=['TEST_FIXTURE_ONLY']*2)
    def prepared(self,mode=SCOPED,torch_backend=False):
        owner=PhysicalOwner(self.f.ident,self.f.registry,validation_mode=mode);owner.begin(self.f.req('S',[1,2,3,4,5,6]),{'cluster_id':0})
        self.project_all(owner,6,torch_backend=torch_backend);self.complete(owner)
        owner.begin(self.f.req('T',[1,2,3,4,5,6,7,8],user='User_02'),{'cluster_id':0})
        return owner,owner.active['selected'][0].entry
    def test_untouched_modes_have_identical_values_rows_and_selection(self):
        results=[]
        for mode in MODES:
            owner,_=self.prepared(mode);selected=[(h.window.start,h.window.end) for h in owner.active['selected']]
            self.project_all(owner,8,late_mismatch=True);result=self.complete(owner);results.append((selected,result))
        self.assertEqual(results[0][0],results[1][0]);self.assertEqual(results[0][1]['physical_projection_rows_reused'],18)
        for a,b in zip(results[0][1]['records'],results[1][1]['records']):
            self.assertEqual(a['output'],b['output']);self.assertEqual(a['fresh_positions'],b['fresh_positions']);self.assertEqual(a['reused_positions'],b['reused_positions'])
        eager,scoped=[r[1]['integrity_validation'] for r in results]
        self.assertEqual(eager['full_entry_proof_count'],10);self.assertEqual(eager['consumed_tensor_proof_count'],0)
        self.assertEqual(scoped['full_entry_proof_count'],2);self.assertEqual(scoped['consumed_tensor_proof_count'],6)
        self.assertTrue(scoped['records_expired']);self.assertEqual(scoped['integrity_check_failures'],0)
    def test_original_aggregate_and_fingerprints_share_one_hash_pass(self):
        owner,entry=self.prepared(EAGER);expected=entry_proof(entry);fingerprints={};profile=Profiler()
        with profile.activate('ONE_PASS'):actual=entry_proof(entry,fingerprints=fingerprints)
        self.assertEqual(actual,expected);self.assertEqual(set(fingerprints),{'0','1'})
        self.assertEqual(sum(r['count'] for r in profile.finish()['components'] if r['component']=='tensor_proof'),6)
    def test_scalar_autoregressive_fixture_logits_and_tokens_unchanged(self):
        import random
        from semcache.edgelora.r3b_fixture import forward
        from semcache.edgelora.r3a_tiny import linear
        from semcache.edgelora.r5_subsequence_generation import generate_loop
        rng=random.Random(42);values=lambda n:[rng.uniform(-.2,.2) for _ in range(n)]
        base=dict(hidden=4,vocabulary=16,embedding=values(16*4),head=values(16*4),layers=[{role:values(16) for role in 'qkv'} for _ in range(2)])
        results=[]
        for mode in MODES:
            owner=PhysicalOwner(self.f.ident,self.f.registry,validation_mode=mode)
            def run(req):
                owner.begin(req,{'cluster_id':0})
                def step(index,generated,past):
                    ids=req['token_ids']+generated
                    def project(layer,role,x,native):
                        if index:return native()  # Explicit all-fresh scalar decode reference.
                        tensor=ArrayTensor(x,[1,len(ids),4])
                        return owner.project(layer,role,tensor,lambda fresh:ArrayTensor(linear(fresh.values,base['layers'][layer][role],4,4),fresh.shape),Session()).values
                    logits=forward(ids,base,project)
                    return logits[-16:],None,{'TEST_FIXTURE_ONLY':True}
                return generate_loop(req,step,lambda tokens:self.complete(owner),lambda tokens:' '.join(map(str,tokens)))
            run(self.f.req('S',[1,2,3,4,5,6]));target=self.f.req('T',[1,2,3,4,5,6,7,8])
            target['generation_policy'].update(mode='fixed',max_new_tokens=3,stop_at_eos=False);target.pop('request_sha256');target['request_sha256']=digest(target)
            results.append(run(target))
        self.assertEqual(results[0]['logits_blob'],results[1]['logits_blob']);self.assertEqual(results[0]['generated_token_ids'],results[1]['generated_token_ids'])
        self.assertGreater(results[1]['completion']['physical_projection_rows_reused'],0)
    def test_corruption_before_authorization_all_roles(self):
        for role in range(3):
            with self.subTest(role=role):
                owner,entry=self.prepared();entry.tensors[0][role].values[0]+=1
                with self.assertRaisesRegex(ValueError,'integrity failure'):owner.valid(entry)
                self.assertFalse(owner.validation.open);self.assertFalse(owner.validation.verified)
    def test_corruption_after_authorization_all_roles(self):
        for role in range(3):
            with self.subTest(role=role):
                owner,entry=self.prepared();owner.valid(entry);entry.tensors[0][role].values[0]+=1
                with self.assertRaisesRegex(ValueError,'consumed cached tensor integrity failure'):owner.validation.consume(entry,0,'qkv'[role])
                self.assertFalse(owner.validation.verified)
    def test_tensor_replacement_even_equal_values_rejected(self):
        owner,entry=self.prepared();owner.valid(entry);old=entry.tensors[0];entry.tensors[0]=(old[0].clone(),old[1],old[2])
        with self.assertRaisesRegex(ValueError,'identity changed'):owner.valid(entry)
    def test_equal_byte_replacement_before_request_verification_rejected(self):
        for kind in ('tensor','entry','proof'):
            with self.subTest(kind=kind):
                owner,entry=self.prepared();owner.abort()
                if kind=='tensor':ts=entry.tensors[0];entry.tensors[0]=(ts[0].clone(),ts[1],ts[2])
                elif kind=='entry':entry=copy.deepcopy(entry);owner.cache.entries[entry.key]=entry
                else:entry.tensors[0][0].values[0]+=1;owner.proofs[entry.key]=entry_proof(entry)
                owner.begin(self.f.req('PRE_AUTH',[1,2,3,4,5,6,7,8]),{'cluster_id':0})
                with self.assertRaises(ValueError):owner.valid(entry)
    def test_metadata_policy_registry_and_physical_tampering(self):
        for field in ('metadata','positions','size','policy','registry','physical','metadata_root','tensor_root'):
            with self.subTest(field=field):
                owner,entry=self.prepared();owner.valid(entry)
                if field=='metadata':entry.qkv_metadata['source_request_id']='CHANGED'
                elif field=='positions':entry.positions=(1,4)
                elif field=='size':entry.size_bytes+=4
                elif field=='policy':entry.impact+=1
                elif field=='registry':owner.bridge.registry['D5_BAD']=True
                elif field=='metadata_root':entry.qkv_metadata=dict(entry.qkv_metadata)
                elif field=='tensor_root':entry.tensors=dict(entry.tensors)
                else:entry.qkv_metadata['physical_identity']=dict(self.f.ident,adapter_matrix_fingerprint='c'*64)
                with self.assertRaises(ValueError):owner.valid(entry)
    def test_resident_entry_replacement_same_key_rejected(self):
        owner,entry=self.prepared();owner.valid(entry);replacement=copy.deepcopy(entry);owner.cache.entries[entry.key]=replacement
        with self.assertRaises(ValueError):owner.valid(replacement)
    def test_eviction_disposal_and_abort_expire_records(self):
        for action in ('eviction','dispose','abort'):
            with self.subTest(action=action):
                owner,entry=self.prepared();owner.valid(entry)
                if action=='eviction':del owner.cache.entries[entry.key]
                elif action=='dispose':owner.dispose()
                else:owner.abort()
                with self.assertRaises(ValueError):owner.valid(entry)
                self.assertFalse(owner.validation.verified)
    def test_failure_cannot_resurrect_authorization(self):
        owner,entry=self.prepared();owner.valid(entry);old=entry.tensors[0][0].values[0];entry.tensors[0][0].values[0]+=1
        with self.assertRaises(ValueError):owner.validation.consume(entry,0,'q')
        entry.tensors[0][0].values[0]=old
        with self.assertRaises(ValueError):owner.validation.consume(entry,0,'q')
    def test_new_request_revalidates_resident_entry(self):
        owner,entry=self.prepared();owner.valid(entry);old_epoch=owner.validation.epoch;owner.abort()
        entry.tensors[0][0].values[0]+=1;owner.begin(self.f.req('NEW',[1,2,3,4,5,6,7,8]),{'cluster_id':0})
        self.assertGreater(owner.validation.epoch,old_epoch)
        with self.assertRaises(ValueError):owner.valid(entry)
        self.assertEqual(owner.validation.metrics['full_entry_proof_count'],1)
    def test_old_verification_record_cannot_grant_a_new_request(self):
        owner,entry=self.prepared();owner.valid(entry);stale=owner.validation.verified[entry.key];owner.abort()
        owner.begin(self.f.req('NEW_EPOCH',[1,2,3,4,5,6,7,8]),{'cluster_id':0});owner.validation.verified[entry.key]=stale
        with self.assertRaisesRegex(ValueError,'stale request'):owner.valid(entry)
        self.assertFalse(owner.validation.verified)
    def test_later_layer_mutation_before_consumption_detected(self):
        owner,entry=self.prepared();owner.valid(entry);owner.validation.consume(entry,0,'q');entry.tensors[1][1].values[0]+=1
        with self.assertRaises(ValueError):owner.validation.consume(entry,1,'k')
    def test_unconsumed_corruption_limitation_is_explicit(self):
        owner,entry=self.prepared();owner.valid(entry);entry.tensors[1][2].values[0]+=1
        owner.validation.consume(entry,0,'q')  # Actual consumed bytes remain authenticated.
        self.assertEqual(owner.validation.report()['unconsumed_content_after_authorization'],'NOT_CONTINUOUSLY_REHASHED')
        with self.assertRaises(ValueError):owner.validation.consume(entry,1,'v')
    def test_private_snapshot_has_no_cache_alias(self):
        owner,entry=self.prepared();owned,receipt=owner.validation.consume(entry,0,'q');original=list(owned.values)
        entry.tensors[0][0].values[0]+=1;self.assertEqual(owned.values,original);self.assertTrue(receipt['private_snapshot'])
    def test_chu_pbr_remain_legal_after_records_expire(self):
        owner,_=self.prepared();owner.bridge.pbr_interval=1;self.project_all(owner,8);result=self.complete(owner)
        self.assertTrue(result['integrity_validation']['records_expired']);self.assertFalse(owner.validation.verified)
        owner.begin(self.f.req('AFTER_POLICY',[1,2,3,4,5,6,7,8]),{'cluster_id':0})
        for hit in owner.active['selected']:owner.valid(hit.entry)
    def test_no_hit_and_all_rejected_windows(self):
        owner,_=self.prepared();owner.abort();owner.begin(self.f.req('NONE',[9,10,11,12,13,14,15,16]),{'cluster_id':0})
        self.project_all(owner,8);result=self.complete(owner);self.assertEqual(result['physical_projection_rows_reused'],0)
        self.assertEqual(result['integrity_validation']['full_entry_proof_count'],0)
        owner,_=self.prepared()
        for layer in range(2):
            for role in 'qkv':owner.project(layer,role,self.x(8,2.),self.native,Session())
        result=self.complete(owner);self.assertEqual(result['physical_projection_rows_reused'],0);self.assertEqual(result['integrity_validation']['consumed_tensor_proof_count'],0)
    def test_foreign_thread_api_is_rejected_not_supported_concurrency(self):
        owner,entry=self.prepared();owner.valid(entry);errors=[]
        def invoke():
            try:owner.validation.consume(entry,0,'q')
            except ValueError as exc:errors.append(str(exc))
        thread=threading.Thread(target=invoke);thread.start();thread.join()
        self.assertEqual(len(errors),1);self.assertIn('foreign-thread',errors[0]);self.assertFalse(owner.validation.verified)
    def test_validation_mode_cli_dispatch_and_no_implicit_switch(self):
        from semcache.edgelora.r3a_artifacts import ROOT
        spec=importlib.util.spec_from_file_location('d5_cli',ROOT/'scripts/152_r5_subsequence_diagnostics.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        args=['--experiment','p','--run','r','--audit','a','--output-dir','o']
        with patch('semcache.edgelora.r5_subsequence_diagnostics.prepare_revalidation',return_value={'TEST_FIXTURE_ONLY':True}) as prepare,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args+['--prepare-revalidation','--source-commit','new','--validation-mode',SCOPED]),0)
            self.assertEqual(prepare.call_args.kwargs['validation_mode'],SCOPED)
        with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):cli.main(args+['--validation-mode',SCOPED])
        with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):cli.main(args+['--compare-validation'])
        other=['--verification','v','--other-experiment','op','--other-run','or','--other-audit','oa','--other-verification','ov']
        with patch('semcache.edgelora.r5_subsequence_integrity.compare_validation',return_value={'TEST_FIXTURE_ONLY':True}) as compare,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args+['--compare-validation',*other]),0);self.assertEqual(compare.call_args.kwargs['other_run'],'or')
    def test_fresh_sealed_d5_plan_preserves_original_tasks(self):
        from semcache.edgelora import r5_subsequence_diagnostics as d
        from semcache.edgelora.r3a_artifacts import ROOT,write
        from semcache.edgelora.r5_i5r2_tensors import Evidence
        from semcache.edgelora.r5_i5_evidence import seal
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';old.mkdir();original=dict(tasks=[{'request_id':'S','token_ids':[1,2,3]}],source={'commit':'a'*40},multi=str(root/'multi'),dataset=str(root/'dataset'))
            write(old/'subsequence_plan.json',original);seal(old);before=Evidence(old).verify();new={'commit':'c'*40}
            test_bytes=(ROOT/'tests/test_r5_subsequence_validation.py').read_bytes()
            with patch.object(d,'history',return_value=(original,None,None,{'TEST_FIXTURE_ONLY':True})),patch('semcache.edgelora.r5_i5r1_plan.source_identity',return_value=new),patch.object(d,'diagnostic_source_files',return_value={}),patch.object(d.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=test_bytes)):
                d.prepare_revalidation(old,root/'raw',root/'audit',root/'new',new['commit'],validation_mode=SCOPED)
            ev=Evidence(root/'new');ev.verify();p=ev.json('subsequence_plan.json')
            self.assertEqual(p['tasks'],original['tasks']);self.assertEqual(p['integrity_validation']['mode'],SCOPED);self.assertEqual(Evidence(old).verify(),before)
    def test_failed_projection_aborts_scope_without_fallback(self):
        owner,entry=self.prepared();owner.valid(entry);entry.tensors[0][0].values[0]+=1
        with self.assertRaises(ValueError):owner.project(0,'q',self.x(8),self.native,Session())
        self.assertIsNone(owner.active);self.assertFalse(owner.validation.verified);self.assertIsNone(owner.bridge.pending)
    def test_independent_value_authorization_replay_and_tamper(self):
        from semcache.edgelora.r4c_evidence import Evidence as Writer
        from semcache.edgelora.r5_i5r2_tensors import Evidence
        from semcache.edgelora.r5_i5_evidence import seal
        from semcache.edgelora.r5_subsequence_audit import replay_validation
        for mode in MODES:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'es').mkdir();writer=Writer(root/'es');owner=PhysicalOwner(self.f.ident,self.f.registry,evidence=writer,validation_mode=mode)
                source=self.f.req('S',[1,2,3,4,5,6]);owner.begin(source,{'cluster_id':0});self.project_all(owner,6);src=self.complete(owner)
                target=self.f.req('T',[1,2,3,4,5,6,7,8],user='User_02');matching=owner.begin(target,{'cluster_id':0});self.project_all(owner,8,late_mismatch=True);physical=self.complete(owner)
                seal(root);ev=Evidence(root);ev.verify();ev.registry=self.f.registry;sources={'S':(source,{(r['layer'],r['role']):r for r in src['records']})}
                result=dict(condition='SEMCACHE_SUBSEQUENCE',matching=matching,assignment={'cluster_id':0},completion={'physical':physical})
                replay_validation(ev,target,result,sources)
                for field in ('count','expired','epoch','proof','consumed'):
                    with self.subTest(field=field):
                        bad=copy.deepcopy(result);report=bad['completion']['physical']['integrity_validation']
                        if field=='count':report['full_entry_proof_count']+=1
                        elif field=='expired':report['records_expired']=False
                        elif field=='epoch':report['request_sha256']='0'*64
                        elif field=='proof' and mode==SCOPED:
                            auth=report['authorizations'][0];auth['tensors']['0'][0]['sha256']='0'*64
                            auth['authorization_id']=digest({k:v for k,v in auth.items() if k!='authorization_id'})
                        elif field=='consumed' and mode==SCOPED:bad['completion']['physical']['records'][0]['consumed'][0]['integrity_validation']['actual']['sha256']='0'*64
                        else:continue
                        with self.assertRaises(ValueError):replay_validation(ev,target,bad,sources)
    @unittest.skipUnless(importlib.util.find_spec('torch'),'native tensor alias/data mutation NOT_RUN: Torch absent')
    def test_torch_inplace_data_and_view_alias_mutations(self):
        import torch
        for role in range(3):
            for change in ('inplace','data','alias','replace_data','stride','numpy_alias'):
                with self.subTest(role=role,change=change):
                    owner,entry=self.prepared(torch_backend=True);owner.valid(entry);t=entry.tensors[1][role];version=t._version
                    if change=='inplace':t.add_(1)
                    elif change=='data':t.data.add_(1);self.assertEqual(t._version,version)
                    elif change=='alias':t.detach().view(-1)[0]+=1
                    elif change=='stride':t.as_strided_((1,3,4),(12,1,3))
                    elif change=='numpy_alias':t.detach().numpy().reshape(-1)[0]+=1;self.assertEqual(t._version,version)
                    else:t.data=t.data.clone()
                    with self.assertRaises(ValueError):owner.validation.consume(entry,1,'qkv'[role])
    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),'native OPT validation-mode parity NOT_RUN: dependencies absent')
    def test_native_cpu_generation_modes_logits_tokens_and_rows_match(self):
        import torch
        from transformers import OPTConfig,OPTForCausalLM
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.semantic.encoder import ControlledEncoder
        from semcache.edgelora.r5_semantic_index import SemanticIndex
        from semcache.edgelora.r5_subsequence_generation import generate
        from test_r5_subsequence_generation import torch_ud_child
        results=[]
        for mode in MODES:
            ctx=multiprocessing.get_context('spawn');parent,kid=ctx.Pipe();proc=ctx.Process(target=torch_ud_child,args=(kid,self.f.ident,self.f.registry,self.f.matrices));proc.start();channel=None
            try:
                self.assertTrue(parent.poll(20));channel=Channel(connect(parent.recv()),'fixture_session',self.f.ident['adapter_matrix_fingerprint'])
                torch.manual_seed(42);cfg=OPTConfig(vocab_size=32,hidden_size=4,word_embed_proj_dim=4,ffn_dim=16,num_hidden_layers=2,num_attention_heads=2,max_position_embeddings=64,dropout=0.,attention_dropout=0.);cfg._attn_implementation='eager'
                model=OPTForCausalLM(cfg).eval();adapter=OPTModelAdapter(model);owner=PhysicalOwner(self.f.ident,self.f.registry,validation_mode=mode);index=SemanticIndex(ControlledEncoder({'fixture':[1.,0.]}))
                class Tokenizer:
                    def decode(self,ids,**kwargs):return ' '.join(map(str,ids))
                source=self.f.req('S',[1,2,3,4,5,6]);target=self.f.req('T',[1,2,3,4,5,6,7,8],user='User_02')
                target['generation_policy'].update(mode='fixed',max_new_tokens=3,stop_at_eos=False);target.pop('request_sha256');target['request_sha256']=digest(target)
                generate(model,adapter,Tokenizer(),source,channel,owner,index,condition='SEMCACHE_SUBSEQUENCE')
                results.append(generate(model,adapter,Tokenizer(),target,channel,owner,index,condition='SEMCACHE_SUBSEQUENCE'))
                channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE');self.assertTrue(parent.poll(20));parent.recv()
            finally:
                if channel:channel.sock.close()
                proc.join(10)
                if proc.is_alive():proc.terminate();proc.join()
                parent.close()
            self.assertEqual(proc.exitcode,0)
        self.assertEqual(results[0]['logits_blob'],results[1]['logits_blob']);self.assertEqual(results[0]['generated_token_ids'],results[1]['generated_token_ids'])
        self.assertEqual(results[0]['completion']['physical']['physical_projection_rows_reused'],results[1]['completion']['physical']['physical_projection_rows_reused'])
        self.assertGreater(results[1]['completion']['physical']['physical_projection_rows_reused'],0)

if __name__=='__main__':unittest.main()
