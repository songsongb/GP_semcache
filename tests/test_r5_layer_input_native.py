"""Optional real Torch CPU bare-Linear test; tiny TEST_FIXTURE_ONLY, not OPT-2.7B."""
import importlib.util
import multiprocessing
import unittest

from semcache.edgelora.communication import Channel,connect,header
from semcache.edgelora.r5_layer_input_physical import LayerInputOwner,MODE
from semcache.edgelora.r5_layer_input_arithmetic import NativeReferenceArithmetic
from semcache.edgelora.r5_sparse_qkv import SparseSession,finish_request
import test_r5_layer_input_verified as fixtures


class Native(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                         'NativeReferenceArithmetic CPU NOT_RUN: Torch/Transformers absent')
    def test_native_full_shape_guard_real_cpu_ud(self):
        import torch
        from types import SimpleNamespace
        from test_r5_subsequence_generation import torch_ud_child
        fixture=fixtures.Verified('runTest');fixture.setUp()
        torch.manual_seed(42)
        modules=[{r:torch.nn.Linear(4,4).eval() for r in 'qkv'} for _ in range(2)]
        adapter=SimpleNamespace(layers=[None,None],projection_modules=lambda l:modules[l])
        arithmetic=NativeReferenceArithmetic(fixture.f.ident,adapter)
        owner=LayerInputOwner(fixture.f.ident,fixture.f.registry,arithmetic=arithmetic)
        ctx=multiprocessing.get_context('spawn');parent,kid=ctx.Pipe()
        process=ctx.Process(target=torch_ud_child,args=(kid,fixture.f.ident,fixture.f.registry,fixture.matrices))
        process.start();channel=None
        try:
            self.assertTrue(parent.poll(20),'native CPU UD unavailable')
            channel=Channel(connect(parent.recv()),'fixture_session',fixture.f.ident['adapter_matrix_fingerprint'])
            results=[]
            for rid,ids in (('SOURCE',[1,2,3,4,5,6]),('TARGET',[9,2,3,4,5,6])):
                req=fixture.f.req(rid,ids);owner.begin(req,{'cluster_id':0},mode=MODE)
                bound=owner.completion_channel(channel);session=SparseSession(bound,req,rid+'_PREFILL')
                attentions=[]
                with torch.inference_mode():
                    for l in range(2):
                        values=fixture.inputs(ids,l);x=torch.tensor(values.values,dtype=torch.float32).reshape(values.shape)
                        qkv={r:owner.project(l,r,x,modules[l][r].forward,session) for r in 'qkv'}
                        scores=(qkv['q'] @ qkv['k'].transpose(-1,-2))/2.
                        mask=torch.triu(torch.ones(len(ids),len(ids),dtype=torch.bool),diagonal=1)
                        attentions.append(torch.softmax(scores.masked_fill(mask,-torch.inf),dim=-1).unsqueeze(1))
                    session.complete();finish_request(bound,req,[3])
                    results.append(owner.complete(completed=True,attentions=attentions))
            self.assertGreater(results[1]['reused_projection_rows_by_category']['B'],0)
            self.assertEqual(results[1]['projection_rows_skipped'],0)
            self.assertEqual(results[1]['native_projection_rows'],36)
            channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE')
            self.assertTrue(parent.poll(20));parent.recv()
        finally:
            owner.dispose()
            if channel:channel.sock.close()
            process.join(10)
            if process.is_alive():process.terminate();process.join()
            parent.close()
        self.assertEqual(process.exitcode,0)


if __name__=='__main__':unittest.main()
