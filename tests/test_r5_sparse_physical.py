"""Actual CPU TCP and owned ArrayTensor reuse; NOT GPU/numerical evidence."""
import copy,multiprocessing,os,socket,unittest,importlib.util
from unittest.mock import patch
from semcache.edgelora.communication import Channel,listener,connect,header,digest
from semcache.edgelora.r5_sparse_qkv import SparseSession,serve_sparse,gather,scatter,SparseOrder,envelope,audit_sparse_receipts
from semcache.edgelora.r5_subsequence_physical import PhysicalOwner,EXACT,APPROX
from semcache.edgelora.r5_i5_contracts import request,namespace
from semcache.edgelora.r5_multi_artifacts import runtime_view,registry_for
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.r4c_workload import policy

def child(pipe,identity,registry,matrices):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sock=listener();pipe.send(sock.getsockname()[1]);connection,_=sock.accept();sock.close()
    channel=Channel(connection,'fixture_session',identity['adapter_matrix_fingerprint']);rows=[]
    try:serve_sparse(channel,identity,registry,matrices,fixture=True,trace=rows.append);pipe.send(rows)
    finally:connection.close();pipe.close()

class Physical(unittest.TestCase):
    def setUp(self):
        self.registry=registry_for(['Adapter_A','Adapter_B','Adapter_C'],'expanded')
        pm={'identity':dict(model='fixture',base_revision='fixture',base_weight_sha256='a'*64,dtype='float32',hidden_dtype='float32',layers=2,hidden_size=4)}
        spec=dict(adapter_id='Adapter_A',matrix_sha256='b'*64,rank=8,matrix_dtype='float32')
        self.ident=namespace(runtime_view(pm,spec),{'fixture':'integer'},'cpu','CPU_FIXTURE',{'fixture':True})
        self.matrices={(l,r,k):([.01]*32) for l in range(2) for r in 'qkv' for k in ('A','B')}
    def req(self,rid,ids,user='User_01'):
        return request(dict(request_id=rid,user_id=user,adapter_id='Adapter_A',input_text='fixture',dataset='qasper',sample_id=rid,request_sequence_number=0),ids,policy(dict(bos_token_id=2,eos_token_id=2,pad_token_id=1),'quality',1),self.ident,fixture=True)
    def test_gather_scatter_and_missing_coverage(self):
        x=ArrayTensor(list(range(16)),[1,4,4]);fresh=gather(x,[0,3]);cached=gather(x,[1,2])
        self.assertEqual(scatter(fresh,[0,3],[(1,cached)],4,4,x).values,x.values)
        with self.assertRaises(ValueError):scatter(fresh,[0,3],[],4,4,x)
        with self.assertRaises(ValueError):gather(x,[3,0])
    def test_sparse_order_auth_and_stale_pass(self):
        req=self.req('S',[1,2,3]);order=SparseOrder(self.ident,self.registry)
        order.accept(envelope(req,0,[0,2],'P'));order.accept(envelope(req,1,[0,2],'P'))
        with self.assertRaises(ValueError):order.accept(envelope(req,0,[0,2],'P'))
        bad=self.req('X',[1,2,3],user='User_11')
        with self.assertRaises(ValueError):SparseOrder(self.ident,self.registry).accept(envelope(bad,0,[0],'Q'))
    def test_tcp_full_and_sparse_physical_consumption(self):
        ctx=multiprocessing.get_context('fork');parent,kid=ctx.Pipe();process=ctx.Process(target=child,args=(kid,self.ident,self.registry,self.matrices));process.start()
        channel=None
        try:
            self.assertTrue(parent.poll(10),'CPU UD did not become ready');port=parent.recv();channel=Channel(connect(port),'fixture_session',self.ident['adapter_matrix_fingerprint'])
            owner=PhysicalOwner(self.ident,self.registry)
            def execute(req,pid,reuse,mode=EXACT,allow_approximate=False):
                session=SparseSession(channel,req,pid,fixture=True);owner.begin(req,{'cluster_id':0},reuse=reuse,mode=mode,allow_approximate=allow_approximate)
                for layer in range(2):
                    x=ArrayTensor([float(t)+j*.1 for t in req['token_ids'] for j in range(4)],[1,len(req['token_ids']),4])
                    for role in 'qkv':owner.project(layer,role,x,lambda t:ArrayTensor([2*v for v in t.values],t.shape),session)
                session.complete()
                with patch('semcache.edgelora.r5_subsequence_physical.actual_attention_impact',return_value=2.):result=owner.complete(completed=True,attentions=['CPU_FIXTURE_ONLY']*2)
                return result,session
            source,_=execute(self.req('S',[1,2,3,4,5,6]),'SOURCE',False)
            self.assertEqual(source['physical_projection_rows_reused'],0);self.assertGreater(owner.cache.physical_tensor_bytes,0)
            target=self.req('T',[1,2,3,4,5,6,7,8],user='User_02');hit,hs=execute(target,'TARGET',True)
            full,fs=execute(self.req('B',[1,2,3,4,5,6,7,8],user='User_02'),'BASELN',False)
            self.assertEqual(hit['physical_projection_rows_reused'],36)
            self.assertEqual([r['output']['sha256'] for r in hit['records']],[r['output']['sha256'] for r in full['records']])
            self.assertEqual(sum(r['fresh_rows_per_role'] for r in hs.receipts),4)
            self.assertGreater(sum(r['hidden_wire_bytes']+r['delta_wire_bytes'] for r in fs.receipts),sum(r['hidden_wire_bytes']+r['delta_wire_bytes'] for r in hs.receipts))
            exact,_=execute(self.req('X',[9,2,3,4,5,6,7,8]),'CAUSAL',True)
            self.assertEqual(exact['physical_projection_rows_reused'],0)
            shifted,_=execute(self.req('Z',[0,1,2,3,4,5,6,7,8]),'SHIFTED',True,APPROX,True)
            self.assertGreater(shifted['physical_projection_rows_reused'],0);self.assertEqual(shifted['mode'],APPROX)
            entire,zero=execute(self.req('R',[0,1,2,3,4,5,6,7,8]),'ALLHIT',True,APPROX,True)
            self.assertEqual(entire['physical_projection_rows_reused'],54);self.assertEqual(sum(r['fresh_rows_per_role'] for r in zero.receipts),0)
            channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE');self.assertTrue(parent.poll(10));ud=parent.recv()
            self.assertEqual(sum(r['cpu_projection_rows'] for r in ud),36+12+48+48+18)
            hit_ud=[r for r in ud if r['context']['pass_id']=='TARGET']
            audited=audit_sparse_receipts(hit['records'],hs.receipts,hit_ud);self.assertEqual(audited['reused_projection_rows'],36)
            bad=copy.deepcopy(hit['records']);bad[0]['native_projection_rows']+=1
            with self.assertRaises(ValueError):audit_sparse_receipts(bad,hs.receipts,hit_ud)
            key=next(iter(owner.cache.entries));entry=owner.cache.entries[key];entry.tensors[0][0].values[0]+=1
            with self.assertRaises(ValueError):owner.valid(entry)
            owner.dispose()
            with self.assertRaises(ValueError):owner.valid(entry)
        finally:
            if channel:channel.sock.close()
            process.join(10)
            if process.is_alive():process.terminate();process.join()
            parent.close()
        self.assertEqual(process.exitcode,0)
    def test_approximate_opt_in_required_and_exact_context_diagnostic(self):
        owner=PhysicalOwner(self.ident,self.registry)
        with self.assertRaises(ValueError):owner.begin(self.req('S',[1,2,3]),{'cluster_id':0},mode=APPROX)
    def test_real_tcp_corrupted_payload_rejected(self):
        from semcache.edgelora.communication import encode
        from semcache.edgelora.r3a_artifacts import pack
        server=listener();raw=connect(server.getsockname()[1]);accepted,_=server.accept();server.close()
        channel=Channel(accepted,'fixture_session',self.ident['adapter_matrix_fingerprint'])
        try:
            h=dict(header('HIDDEN','X',0,'float32',[1,1,4]),session_id='fixture_session',adapter_sha256=self.ident['adapter_matrix_fingerprint'],sequence=0)
            packet,_=encode(h,pack([1.,2.,3.,4.],'float32'));raw.sendall(packet[:-1]+bytes([packet[-1]^1]))
            with self.assertRaises(ValueError):channel.recv()
        finally:raw.close();accepted.close()
    @unittest.skipUnless(importlib.util.find_spec('torch'),'native bare Linear path NOT_RUN: torch absent')
    def test_actual_bare_linear_hook(self):
        import torch
        from types import SimpleNamespace
        from semcache.edgelora.r5_subsequence_physical import projection_path,tensor_proof
        modules=[{r:torch.nn.Linear(4,4).eval() for r in 'qkv'} for _ in range(2)]
        adapter=SimpleNamespace(layers=[None,None],projection_modules=lambda i:modules[i]);owner=PhysicalOwner(self.ident,self.registry)
        class Session:
            calls=0
            def correction(s,layer,role,x,positions):s.calls+=1;return torch.zeros((1,len(positions),4)) if positions else None
            def complete(s):require=s.calls==6;assert require
        before=[tensor_proof(m.weight) for ms in modules for m in ms.values()]
        req=self.req('S',[1,2,3,4,5,6]);owner.begin(req,{'cluster_id':0});session=Session()
        with torch.inference_mode(),projection_path(adapter,owner,session):
            for layer in range(2):
                for role in 'qkv':modules[layer][role](torch.ones(1,6,4))
        owner.complete(completed=True,attentions=[torch.ones(1,1,6,6)/6]*2)
        self.assertGreater(owner.cache.physical_tensor_bytes,0)
        self.assertEqual(before,[tensor_proof(m.weight) for ms in modules for m in ms.values()]);self.assertTrue(all('forward' not in m.__dict__ for ms in modules for m in ms.values()))
    @unittest.skipUnless(importlib.util.find_spec('torch'),'native bare Linear/actual impact NOT_RUN: torch absent')
    def test_native_attention_reducer(self):
        import torch
        from semcache.cache.metric_manager import actual_attention_impact
        self.assertGreater(actual_attention_impact([torch.ones(1,1,3,3)/3],0,3),0)

if __name__=='__main__':unittest.main()
