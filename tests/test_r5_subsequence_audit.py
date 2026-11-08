"""CPU value replay/tamper tests, never GPU correctness claims."""
import copy,tempfile,unittest
from pathlib import Path
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.r3a_tiny import f32
from semcache.edgelora.r3a_artifacts import pack
from semcache.edgelora.r4c_evidence import Evidence as Writer
from semcache.edgelora.r5_i5r2_tensors import Evidence
from semcache.edgelora.r5_i5_evidence import seal
from semcache.edgelora.communication import digest_bytes
from semcache.edgelora.r5_sparse_qkv import envelope,gather
from semcache.edgelora.r5_subsequence_audit import replay_projection,replay_policy,replay_wire,request_costs
import test_r5_sparse_physical as fixture

class ValueAudit(unittest.TestCase):
    def setUp(self):self.fixture=fixture.Physical();self.fixture.setUp()
    def scenario(self,root):
        (root/'es').mkdir();(root/'ud').mkdir();writer=Writer(root/'es')
        source=self.fixture.req('S',[1,2,3]);req=self.fixture.req('T',[9,1,2,3,10],user='User_02')
        x=ArrayTensor(list(range(20)),[1,5,4]);positions=[0,4];base=gather(x,positions);base=ArrayTensor([2*v for v in base.values],base.shape)
        delta=ArrayTensor([.1]*8,[1,2,4]);fresh=[f32(a+b) for a,b in zip(base.values,delta.values)];cached=ArrayTensor([float(i+100) for i in range(12)],[1,3,4]);out=ArrayTensor(fresh[:4]+cached.values+fresh[4:],[1,5,4])
        src=writer.save(cached,'source_output');record=dict(layer=0,role='q',mode='APPROXIMATE_SUBSEQUENCE_TOTAL_QKV',input=writer.save(x,'input'),output=writer.save(out,'output'),fresh_positions=positions,fresh_base=writer.save(base,'base'),fresh_delta=writer.save(delta,'delta'),consumed=[dict(source_request_id='S',source_positions=[0,3],target_positions=[1,4],tensor=writer.save(cached,'cached'))])
        hidden=pack(gather(x,positions).values,'float32');deltas=pack([.1]*24,'float32');(root/'ud/hidden.bin').write_bytes(hidden);(root/'ud/delta.bin').write_bytes(deltas)
        ud=dict(context=envelope(req,0,positions,'P'),hidden_file='hidden.bin',delta_file='delta.bin',hidden_sha256=digest_bytes(hidden),delta_sha256=digest_bytes(deltas))
        seal(root);ev=Evidence(root);ev.verify();ev.registry=self.fixture.registry
        return ev,record,ud,req,{'S':(source,{(0,'q'):{'output':src}})}
    def test_saved_source_gather_scatter_values(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.scenario(Path(d));self.assertEqual(replay_projection(*args),3)
    def test_corrupt_payload_and_false_source_values(self):
        for part in ('ud/hidden.bin','ud/delta.bin','es/tensors/cached.bin'):
            with self.subTest(part=part),tempfile.TemporaryDirectory() as d:
                root=Path(d);args=self.scenario(root);data=(root/part).read_bytes();(root/part).write_bytes(bytes([data[0]^1])+data[1:])
                with self.assertRaises(ValueError):replay_projection(*args)
    def test_exact_claim_and_missing_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            args=list(self.scenario(Path(d)));args[1]['mode']='EXACT_CONTEXT_VERIFIED'
            with self.assertRaises(ValueError):replay_projection(*args)
            args[1]['mode']='APPROXIMATE_SUBSEQUENCE_TOTAL_QKV';args[-1]={}
            with self.assertRaises(ValueError):replay_projection(*args)
    def test_cross_adapter_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            args=list(self.scenario(Path(d)));source,records=args[-1]['S'];source=copy.deepcopy(source);source['adapter_id']='Adapter_B';args[-1]['S']=(source,records)
            with self.assertRaises(ValueError):replay_projection(*args)
    def test_physical_fais_chu_independent_replay(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from semcache.edgelora.r5_subsequence_physical import PhysicalOwner
        events=[];owner=PhysicalOwner(self.fixture.ident,self.fixture.registry,trace=events.append);reqs=[];results=[]
        class Session:
            def correction(self,l,r,x,positions):return ArrayTensor([0.]*(len(positions)*4),[1,len(positions),4]) if positions else None
        for rid in ('S','T'):
            req=self.fixture.req(rid,[1,2,3,4,5,6]);reqs.append(req);owner.begin(req,{'cluster_id':0})
            for layer in range(2):
                for role in 'qkv':owner.project(layer,role,ArrayTensor([1.]*24,[1,6,4]),lambda x:x.clone(),Session())
            with patch('semcache.edgelora.r5_subsequence_physical.actual_attention_impact',return_value=2.):physical=owner.complete(completed=True,attentions=['CPU_FIXTURE_ONLY']*2)
            results.append(dict(assignment={'cluster_id':0},completion=dict(physical=physical,impact={'records':[{'span_values':{str(i):1. for i in range(4)}} for _ in range(2)]})))
        ev=SimpleNamespace(jsonl=lambda *a,**k:events);replay_policy(ev,reqs,results)
        next(e for e in events if e['kind']=='CHU')['F']+=1
        with self.assertRaises(ValueError):replay_policy(ev,reqs,results)
    def test_independent_wire_endpoint_equality(self):
        send=dict(operation='send',complete=True,frame_sha256='a'*64,wire_bytes=100,header={'sequence':0})
        recv=dict(send,operation='recv');reply=dict(send,frame_sha256='b'*64,operation='recv');response=dict(reply,operation='send')
        self.assertEqual(replay_wire([send,reply],[recv,response])['frames'],2)
        bad=copy.deepcopy(recv);bad['header']['sequence']=1
        with self.assertRaises(ValueError):replay_wire([send,reply],[bad,response])
    def test_actual_request_cost_rows_not_unique_tokens(self):
        receipt=dict(fresh_rows_per_role=2,skipped_rows_per_role=3,sent={'wire_bytes':80},received={'wire_bytes':120,'header':{'metadata':{'cpu_lora_ms':1.5}}})
        result=dict(request_id='CPU_FIXTURE',sparse_receipts=[receipt]*2,completion={'physical':{'physical_projection_rows_reused':18,'costs':{'owner_completion_ms':5.}}})
        costs=request_costs(result);self.assertEqual(costs['native_projection_rows'],12);self.assertEqual(costs['physically_reused_projection_rows'],18)
        self.assertEqual(costs['es_to_ud_projection_wire_bytes'],160);self.assertEqual(costs['cpu_lora_ms'],3.)
        result['completion']['physical']['physical_projection_rows_reused']+=1
        with self.assertRaises(ValueError):request_costs(result)

if __name__=='__main__':unittest.main()
