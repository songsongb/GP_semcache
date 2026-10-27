"""Prepared-cache ownership and timing contracts; fake fixtures are not GPU evidence."""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch
from semcache.diagnostics.r2b15_cost import PreparedTicket,invoke,CostBook
from semcache.diagnostics.r2b15_analysis import REQUEST_MODES,timing_summary,cost_decisions
from semcache.diagnostics.r2b1_prefix_cache import PrefixError
from semcache.diagnostics.r1b3_break_even import schedule
ROOT=Path(__file__).resolve().parents[1]


def runner():
    spec=importlib.util.spec_from_file_location('b15_runner_test',ROOT/'scripts/86_diagnose_r2b15_prefix.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class CostContracts(unittest.TestCase):
    def fake_runtime(self):
        from test_r2b1_prefix_cache import identity,FakeTensor
        import semcache.diagnostics.r2b1_prefix_cache as co
        stack=ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch.object(co,'clone_tensor',lambda t:t.clone()))
        stack.enter_context(patch.object(co,'tensor_hash',lambda t:str(t.data)))
        stack.enter_context(patch('semcache.metrics.r1_runtime.synchronize',lambda d:None))
        stack.enter_context(patch.dict('sys.modules',{'torch':NS(equal=lambda a,b:a.data==b.data)}))
        snapshot=co.PrefixSnapshot(((FakeTensor(),FakeTensor()),),identity(),[1,2,3],layers=1,heads=2,head_dim=4,source_path='PREFIX_SOURCE_PREFILL')
        class Runtime:
            p=3; n=4; ids=[1,2,3,4]; device='cpu'
            def identity(self): return identity()
            def check(self): return self.snapshot.check()
            def forward(self,path,audit=False):
                if path=='EXACT_PREFIX_WARM': self.snapshot.materialize(self.identity(),self.ids,position_start=self.p,mask=[1]*self.n)
                return {'mode':path,'snapshot':self.snapshot.identity_hash}
        r=Runtime(); r.snapshot=snapshot; return r
    def test_prepared_fresh_and_single_use_with_restore(self):
        r=self.fake_runtime(); original=r.snapshot.materialize
        a=PreparedTicket(r); b=PreparedTicket(r)
        self.assertIsNot(a.cache,b.cache)
        with a.use(): r.forward('EXACT_PREFIX_WARM')
        self.assertEqual(a.report()['materialize_handoffs'],1)
        self.assertEqual(r.snapshot.materialize,original)
        with self.assertRaises(PrefixError):
            with a.use(): r.forward('EXACT_PREFIX_WARM')
        result,ticket=invoke(r,'WARM_PREPARED_DIAGNOSTIC')
        self.assertEqual(ticket['classification'],'IDEAL_DIAGNOSTIC')
        self.assertEqual(r.snapshot.materialize,original)
    def test_only_clone_factory_are_moved_outside(self):
        import semcache.diagnostics.r2b1_prefix_cache as co
        r=self.fake_runtime()
        with patch.object(co,'clone_tensor',wraps=co.clone_tensor) as clone,patch.object(co,'validate_request',wraps=co.validate_request) as guard:
            r.forward('EXACT_PREFIX_WARM'); self.assertEqual(clone.call_count,2)
            ticket=PreparedTicket(r); self.assertEqual(clone.call_count,4)
            previous=guard.call_count
            with ticket.use(): r.forward('EXACT_PREFIX_WARM')
            self.assertEqual(clone.call_count,4); self.assertEqual(guard.call_count,previous+1)

    def test_mutation_namespace_missing_call_and_exception(self):
        r=self.fake_runtime(); t=PreparedTicket(r); t.cache[0][0].data[0]=99
        with self.assertRaises(PrefixError): t.check()
        self.assertEqual(t.state,'INVALIDATED')
        t=PreparedTicket(r); r.ids=[1,2,3,99]
        with self.assertRaises(PrefixError): t.check()
        r.ids=[1,2,3,4]; t=PreparedTicket(r); original=r.snapshot.materialize
        with self.assertRaisesRegex(RuntimeError,'forward failed'):
            with t.use():
                r.snapshot.materialize(r.identity(),r.ids,position_start=3,mask=[1]*4); raise RuntimeError('forward failed')
        self.assertEqual(t.state,'DISCARDED'); self.assertEqual(r.snapshot.materialize,original)
        t=PreparedTicket(r)
        with self.assertRaises(PrefixError):
            with t.use(): pass
    def test_paired_order_sign_and_forbidden_clock_mixing(self):
        rows=[dict(case='c',repeat=i,phase='measured',mode=m,wall_ms=v,instrumentation='minimal') for i in range(3) for m,v in zip(REQUEST_MODES,(20,30,15))]
        result=timing_summary(rows)
        self.assertEqual(next(r for r in result if r['mode']=='WARM_NORMAL')['full_minus_mode']['mean'],-10)
        self.assertEqual(result[0]['normal_minus_prepared']['mean'],15)
        self.assertFalse(result[0]['correctness_qualified_speedup'])
        self.assertEqual(cost_decisions(result)['c']['B-H1']['status'],'UNRESOLVED')
        for bad in ('breakdown','PREPARE_ONLY'):
            modified=[dict(r) for r in rows]; modified[0]['instrumentation' if bad=='breakdown' else 'mode']=bad
            with self.assertRaises(PrefixError): timing_summary(modified)
        with self.assertRaises(PrefixError): timing_summary(rows[:-1])
        order=schedule(REQUEST_MODES,3,701)
        for slot in range(3): self.assertEqual(set(o[slot] for o in order),set(REQUEST_MODES))
    def test_component_parent_and_unavailable_not_zero(self):
        book=CostBook('cpu'); book.host={'cache_materialize':20.,'kv_clone':10.}; book.calls={'cache_materialize':1,'kv_clone':2}
        rows={r['component']:r for r in book.export()}
        self.assertEqual(rows['kv_clone']['parent'],'cache_materialize')
        self.assertIsNone(rows['native_forward']['host_ms']); self.assertIsNone(rows['kv_clone']['cuda_interval_ms'])
        self.assertEqual(rows['return_cleanup']['status'],'UNSUPPORTED_ISOLATED_BOUNDARY')
    def test_monitoring_is_optional_and_excludes_self(self):
        import os
        from semcache.diagnostics.r2b15_cost import monitoring
        missing=monitoring(NS(gpu_state=lambda:{'gpu_uuid':None,'reason':'no monitoring'}))
        self.assertEqual(missing['monitoring_status'],'MONITORING_UNAVAILABLE')
        r=NS(gpu_state=lambda:{'gpu_uuid':'GPU-test','driver':'observed'})
        with patch('subprocess.run',return_value=NS(stdout=f'GPU-test, {os.getpid()}, own, 100\nGPU-test, 99, other, 100\nGPU-other, 98, unrelated, 100')):
            state=monitoring(r)
        self.assertEqual(state['other_processes'],['GPU-test, 99, other, 100'])

    def test_cli_schema_failure_collision(self):
        m=runner()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            with patch.object(m,'run',side_effect=PrefixError('REJECT_PREPARED_MUTATION','bad cache')):
                self.assertEqual(m.main(['--output-dir',str(root)]),1)
            manifest=json.loads((root/'manifest.json').read_text())
            self.assertEqual(manifest['implementation_status'],'IMPLEMENTED_LOCAL_ONLY'); self.assertEqual(manifest['execution_status'],'FAILED')
            for name in m.CSV: self.assertTrue((root/(name+'.csv')).exists())
            self.assertFalse(json.loads((root/'decision_summary.json').read_text())['correctness_qualified_speedup'])
            with self.assertRaises(FileExistsError): m.main(['--output-dir',str(root)])
        self.assertEqual(m.parse_args(['--phase','timing','--model','facebook/opt-2.7b','--dtype','float16','--output-dir','unused']).measured_runs,10)
        with self.assertRaises(SystemExit): m.parse_args(['--phase','timing','--dtype','float16','--measured-runs','3','--output-dir','unused'])

    def test_timing_runner_preserves_strict_fail(self):
        from test_r2b1_prefix_cache import identity as namespace
        m=runner()
        class Runtime:
            def __init__(self,a):
                self.provenance={'fixture':'FAKE_CONTROL_FLOW_ONLY'}; self.model=NS(modules=lambda:[])
                self.adapter=NS(layers=[object()]); self.device='cpu'; self.a=a; self.snapshot=NS(identity_hash='source',bytes=100)
                self.meta={'resolved_model_revision':'rev'}
            def prepare(self,n,p): self.n=n; self.p=p; self.ids=list(range(n)); self.source_report={'source_ms':999}; self.preflight={}
            def identity(self): return namespace()
            def check(self): return {'status':'FAKE'}
            def gpu_state(self): return {'status':'NOT_A_GPU_TEST'}
            def verify_weight_hashes(self): return {'status':'FAKE'}
        metric={'allclose':False,'correctness_status':'STRICT_FAIL'}
        def call(r,mode,**kwargs):
            value=dict(logits=0,wall_ms={'FULL_USE_CACHE':10,'WARM_NORMAL':20,'WARM_PREPARED_DIAGNOSTIC':15}[mode],
                cache_prepare_host_ms=1,request_clone_bytes=100,query_tokens=r.n if mode=='FULL_USE_CACHE' else r.n-r.p,
                output_kv_length=r.n,cache_api='FAKE',memory={})
            return value,{'classification':'IDEAL_DIAGNOSTIC'} if mode=='WARM_PREPARED_DIAGNOSTIC' else None
        with tempfile.TemporaryDirectory() as tmp,patch.dict('sys.modules',{'torch':NS(equal=lambda a,b:a==b)}),patch.object(m,'PrefixRuntime',Runtime),patch.object(m,'structural_gate',return_value={'status':'FAKE','quality':{'suffix':metric}}),patch.object(m,'invoke',call),patch.object(m,'compare_suffix',return_value={'suffix':metric,'last_position':metric}):
            root=Path(tmp)/'run'
            self.assertEqual(m.main(['--phase','timing','--cases','32:16','--device','cpu','--output-dir',str(root)]),0)
            manifest=json.loads((root/'manifest.json').read_text())
            self.assertEqual(manifest['execution_status'],'COMPLETE'); self.assertEqual(manifest['correctness_status'],'STRICT_FAIL')
            decision=json.loads((root/'decision_summary.json').read_text()); self.assertFalse(decision['correctness_qualified_speedup'])
            import csv
            with (root/'timing_paired_samples.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),39)
            self.assertTrue(all(r['instrumentation']=='minimal' for r in rows))



DEPS=all(importlib.util.find_spec(n) is not None for n in ('torch','transformers','peft'))
@unittest.skipUnless(DEPS,'Torch/Transformers/PEFT unavailable: actual prepared/timing/model tests NOT EXECUTED')
class ModelTests(unittest.TestCase):
    def exercise(self,device):
        import torch
        from test_r2b1_prefix_runtime import OfflineOPTTests
        from semcache.diagnostics.r2b15_cost import structural_gate,prepare_only
        r=OfflineOPTTests().runtime(device); r.prepare(32,16)
        self.assertTrue(structural_gate(r)['prepared_parity_exact'])
        normal,_=invoke(r,'WARM_NORMAL'); result,ticket=invoke(r,'WARM_PREPARED_DIAGNOSTIC')
        self.assertTrue(torch.equal(normal['logits'],result['logits']))
        self.assertEqual(ticket['input_length_before'],16); self.assertEqual(result['output_kv_length'],32)
        prepared=PreparedTicket(r); co=__import__('semcache.diagnostics.r2b1_prefix_cache',fromlist=['cache_pairs'])
        co.cache_pairs(prepared.cache)[0][0][0].data.add_(1)
        with self.assertRaises(PrefixError): prepared.check()
        before=r.source_forwards; book=CostBook(device); v=prepare_only(r,book)
        self.assertEqual(v['native_forward_calls'],0); self.assertEqual(before,r.source_forwards)
        self.assertGreater(book.clone_bytes,0)
        book=CostBook(device); measured,_=invoke(r,'WARM_NORMAL',book=book)
        self.assertTrue(torch.equal(normal['logits'],measured['logits']))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in r.model.modules()))
        r.check()
    def test_cpu(self): self.exercise('cpu')
    def test_cuda(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        self.exercise('cuda:0')

if __name__=='__main__': unittest.main()
