"""CPU generation/streaming contracts; fixture outputs are not OPT GPU results."""
import unittest,copy,importlib.util,multiprocessing
from unittest.mock import patch
from semcache.edgelora.r5_subsequence_generation import generate_loop,StreamingImpact
from semcache.edgelora.r5_sparse_qkv import SparseOrder,envelope,SparseSession,finish_request
from semcache.edgelora.communication import digest,Channel,connect,header
from semcache.edgelora.r3b_fixture import ArrayTensor
import test_r5_sparse_physical as fixture_module

def torch_ud_child(pipe,identity,registry,matrices):
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from semcache.edgelora.user_device import configure_cpu
    from semcache.edgelora.communication import listener
    from semcache.edgelora.r5_sparse_qkv import serve_sparse
    configure_cpu(True)
    import torch
    matrices={k:torch.tensor(v,dtype=torch.float32).reshape((8,4) if k[2]=='A' else (4,8)) for k,v in matrices.items()}
    sock=listener();pipe.send(sock.getsockname()[1]);connection,_=sock.accept();sock.close()
    channel=Channel(connection,'fixture_session',identity['adapter_matrix_fingerprint']);rows=[]
    try:serve_sparse(channel,identity,registry,matrices,fixture=False,trace=rows.append);pipe.send(rows)
    finally:connection.close();pipe.close()

class Generation(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture_module.Physical();self.fixture.setUp();self.req=self.fixture.req('GEN',[1,2,3,4,5,6])
        p=self.req['generation_policy'];p.update(mode='fixed',max_new_tokens=3,stop_at_eos=False)
        self.req.pop('request_sha256');self.req['request_sha256']=digest(self.req)
    def test_greedy_past_lifecycle_and_completion_order(self):
        seen=[]
        def forward(step,tokens,past):
            self.assertEqual(len(tokens),step);self.assertEqual(past,None if not step else step)
            seen.append(('forward',step));return [0.,1.,0.,2.],step+1,{'TEST_FIXTURE_ONLY':True}
        def complete(tokens):self.assertEqual(tokens,[3,3,3]);seen.append(('complete',3));return {'completed':True}
        r=generate_loop(self.req,forward,complete,lambda ids:'fixture')
        self.assertEqual(r['generated_token_ids'],[3,3,3]);self.assertEqual(seen[-1],('complete',3));self.assertEqual(r['logits_shape'],[1,3,4])
        self.assertEqual(r['application_e2e_ms'],(r['completion_ns']-r['arrival_ns'])/1e6)
    def test_eos_and_nonfinite_fail_closed(self):
        req=copy.deepcopy(self.req);req['generation_policy'].update(mode='quality',stop_at_eos=True)
        r=generate_loop(req,lambda *a:([0.,0.,5.],None,{}),lambda t:{},str);self.assertEqual(r['generated_token_ids'],[2]);self.assertEqual(r['termination'],'EOS')
        with self.assertRaises(ValueError):generate_loop(req,lambda *a:([float('nan')],None,{}),lambda t:self.fail('completed on nonfinite'),str)
    def test_decode_history_absolute_positions(self):
        order=SparseOrder(self.fixture.ident,self.fixture.registry)
        for i in range(2):order.accept(envelope(self.req,i,list(range(6)),'P0'))
        c=envelope(self.req,0,[0],'P1',[3]);self.assertEqual(c['absolute_positions'],[6]);self.assertEqual(c['full_length'],1)
        order.accept(c);order.accept(envelope(self.req,1,[0],'P1',[3]))
        with self.assertRaises(ValueError):order.accept(envelope(self.req,0,[0],'P2',[9,3]))
    def test_streaming_requires_all_real_layers_and_budget(self):
        from semcache.semantic.subsequence import SubsequenceExtractor
        class Prob:
            shape=(1,2,6,6)
            def numel(self):return 72
            def element_size(self):return 4
            def detach(self):return self
            def double(self):return self
        windows=SubsequenceExtractor(3).extract(self.req['token_ids']);impact=StreamingImpact(self.req,2,windows,max_layer_bytes=1024)
        active={'req':self.req,'windows':windows}
        with patch('semcache.edgelora.r5_subsequence_generation.actual_attention_impact',return_value=1.):impact.observe(0,Prob())
        with self.assertRaises(ValueError):impact.values_for(active)
        with patch('semcache.edgelora.r5_subsequence_generation.actual_attention_impact',return_value=2.):impact.observe(1,Prob())
        self.assertTrue(all(x==3. for x in impact.values_for(active).values()));self.assertEqual(impact.peak_bytes,288)
        with self.assertRaises(ValueError):StreamingImpact(self.req,2,windows,max_layer_bytes=128).observe(0,Prob())
    def test_real_sparse_decode_and_completion_tcp(self):
        ctx=multiprocessing.get_context('fork');parent,kid=ctx.Pipe();process=ctx.Process(target=fixture_module.child,args=(kid,self.fixture.ident,self.fixture.registry,self.fixture.matrices));process.start();ch=None
        try:
            self.assertTrue(parent.poll(10));ch=Channel(connect(parent.recv()),'fixture_session',self.fixture.ident['adapter_matrix_fingerprint'])
            def forward(step,tokens,past):
                session=SparseSession(ch,self.req,f'GEN_s{step}',fixture=True,generated=tokens);n=1 if tokens else 6
                x=ArrayTensor([1.]*(n*4),[1,n,4])
                for l in range(2):
                    for role in 'qkv':self.assertEqual(session.correction(l,role,x,list(range(n))).shape,(1,n,4))
                session.complete();return [0.,0.,0.,4.],step+1,dict(receipts=session.receipts)
            r=generate_loop(self.req,forward,lambda tokens:finish_request(ch,self.req,tokens),str)
            self.assertEqual(r['generated_token_ids'],[3]*3)
            ch.send(header('STOP','shutdown',-1));ch.recv(kind='BYE');self.assertTrue(parent.poll(10));rows=parent.recv()
            complete=[x for x in rows if x.get('kind')=='COMPLETE_REQUEST'];self.assertEqual(len(complete),1)
            self.assertEqual(sum(x.get('cpu_projection_rows',0) for x in rows),6*(6+1+1))
        finally:
            if ch:ch.sock.close()
            process.join(10)
            if process.is_alive():process.terminate();process.join()
            parent.close()
        self.assertEqual(process.exitcode,0)
    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),'installed eager OPT streaming NOT_RUN: native dependencies absent')
    def test_installed_opt_streamed_impact_matches_original_reducer(self):
        import torch
        from transformers import OPTConfig,OPTForCausalLM
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.semantic.subsequence import SubsequenceExtractor
        from semcache.cache.metric_manager import actual_attention_impact
        cfg=OPTConfig(vocab_size=32,hidden_size=8,word_embed_proj_dim=8,ffn_dim=16,num_hidden_layers=2,num_attention_heads=2,max_position_embeddings=64,dropout=0.,attention_dropout=0.)
        cfg._attn_implementation='eager';model=OPTForCausalLM(cfg).eval();adapter=OPTModelAdapter(model)
        tokens=torch.tensor([self.req['token_ids']]);windows=SubsequenceExtractor(3).extract(self.req['token_ids'])
        with torch.inference_mode():baseline=model(tokens,output_attentions=True,use_cache=False)
        impact=StreamingImpact(self.req,2,windows)
        with torch.inference_mode(),impact.hooks(adapter):observed=model(tokens,output_attentions=True,use_cache=False)
        self.assertTrue(torch.equal(baseline.logits,observed.logits));self.assertTrue(all(x is None for x in observed.attentions))
        for w in windows:self.assertEqual(impact.values[w.start],actual_attention_impact(baseline.attentions,w.start,w.end))
    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),'native OPT sparse generation NOT_RUN: dependencies absent')
    def test_installed_opt_real_tcp_generation_and_admission(self):
        import torch
        from transformers import OPTConfig,OPTForCausalLM
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.edgelora.r5_subsequence_generation import generate
        from semcache.edgelora.r5_subsequence_physical import PhysicalOwner,APPROX
        from semcache.edgelora.r5_semantic_index import SemanticIndex
        from semcache.semantic.encoder import ControlledEncoder
        ctx=multiprocessing.get_context('spawn');parent,kid=ctx.Pipe();proc=ctx.Process(target=torch_ud_child,args=(kid,self.fixture.ident,self.fixture.registry,self.fixture.matrices));proc.start();ch=None
        try:
            self.assertTrue(parent.poll(10));ch=Channel(connect(parent.recv()),'fixture_session',self.fixture.ident['adapter_matrix_fingerprint'])
            cfg=OPTConfig(vocab_size=32,hidden_size=4,word_embed_proj_dim=4,ffn_dim=16,num_hidden_layers=2,num_attention_heads=2,max_position_embeddings=64,dropout=0.,attention_dropout=0.)
            cfg._attn_implementation='eager';model=OPTForCausalLM(cfg).eval();adapter=OPTModelAdapter(model)
            owner=PhysicalOwner(self.fixture.ident,self.fixture.registry);index=SemanticIndex(ControlledEncoder({'fixture':[1.,0.]}))
            class Tokenizer:
                def decode(self,ids,**kw):return ' '.join(map(str,ids))
            source=self.fixture.req('SOURCE',[1,2,3,4,5,6])
            a=generate(model,adapter,Tokenizer(),source,ch,owner,index,condition='SEMCACHE_SUBSEQUENCE',mode=APPROX,allow_approximate=True)
            self.assertEqual(a['generated_token_ids'].__len__(),1);self.assertTrue(owner.cache.entries)
            b=generate(model,adapter,Tokenizer(),self.req,ch,owner,index,condition='SEMCACHE_SUBSEQUENCE',mode=APPROX,allow_approximate=True)
            self.assertEqual(len(b['generated_token_ids']),3);self.assertGreater(b['completion']['physical']['physical_projection_rows_reused'],0)
            self.assertEqual(b['logits_shape'],[1,3,32]);self.assertEqual(b['completion']['impact']['peak_probability_bytes'],1*2*6*6*4)
            for condition,rid in (('FULL_RECOMPUTE','A_TEST'),('SEMANTIC_NO_REUSE','B_TEST')):
                req=copy.deepcopy(self.req);req['request_id']=rid;req.pop('request_sha256');req['request_sha256']=digest(req)
                local_owner=PhysicalOwner(self.fixture.ident,self.fixture.registry)
                result=generate(model,adapter,Tokenizer(),req,ch,local_owner,index,condition=condition)
                self.assertEqual(len(result['generated_token_ids']),3)
                if condition=='FULL_RECOMPUTE':self.assertIsNone(result['completion']['physical']);self.assertIsNone(result['completion']['impact'])
                else:self.assertEqual(result['completion']['physical']['physical_projection_rows_reused'],0)
            ch.send(header('STOP','shutdown',-1));ch.recv(kind='BYE');self.assertTrue(parent.poll(10));parent.recv()
        finally:
            if ch:ch.sock.close()
            proc.join(10)
            if proc.is_alive():proc.terminate();proc.join()
            parent.close()
        self.assertEqual(proc.exitcode,0)

if __name__=='__main__':unittest.main()
