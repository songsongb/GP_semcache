"""Offline mapping/schema tests; real tensor and model evidence require Torch."""
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.batched_mixed_projection import PositionMapping,compare_projections
from semcache.diagnostics.r1b3_fast_selection import FastOccurrenceIndex
from semcache.semantic.hit_selection import CacheHit
from semcache.semantic.subsequence import Subsequence
from test_r1b3_fast_selection import fixture


def runner():
    spec=importlib.util.spec_from_file_location('executor_runner',ROOT/'scripts/76_benchmark_r1b3_batched_executor.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class OfflineTests(unittest.TestCase):
    def test_mapping_randomized_and_max_coverage(self):
        rng=random.Random(8128)
        for n in [128,256]+[rng.randrange(4,80) for _ in range(40)]:
            ids=[rng.randrange(5) for _ in range(n)]
            c,l=fixture([('first',ids,'a'),('second',ids,'a')]); index=FastOccurrenceIndex(c,l)
            for coverage in (0,.25,.5,.75,'max'):
                hits,mask,_=index.select(ids,0,'a','model','weights',l,coverage)
                plan=PositionMapping(index,hits,ids,'a','model','weights')
                self.assertEqual(plan.mask,mask); self.assertEqual(plan.reused,sum(mask))
                self.assertEqual([p for g in plan.groups for p in g['target']],[p for h in hits for p in range(h.window.start,h.window.end)])
                self.assertTrue(all(g['source']==g['target'] for g in plan.groups))
                self.assertEqual(plan.reused+len(plan.fresh),n)
                if coverage=='max': self.assertEqual(plan.reused,n//3*3)
                if coverage==0: self.assertEqual(plan.reused,0)

    def test_rejection_overlap_stale_prefix_namespace_adapter_and_view(self):
        ids=[1,2,3]*4
        for invalid in ('overlap','position','prefix','namespace','adapter','view','owner','dtype'):
            c,l=fixture([('one',ids,'a')]); index=FastOccurrenceIndex(c,l)
            hits=index.select(ids,0,'a','model','weights',l,'max')[0]
            target=list(ids); ns='model'; ad='a'
            if invalid=='overlap': hits=hits+hits[:1]
            if invalid=='position': hits=[CacheHit(Subsequence(hits[0].window.token_ids,3,6),hits[0].entry)]
            if invalid=='prefix': target[0]=99
            if invalid=='namespace': ns='other'
            if invalid=='adapter': ad='other'
            if invalid=='view': hits[0].entry.tensors[0][0].shape=(1,2,4)
            if invalid=='dtype': hits[0].entry.tensors[0][0].dtype='fp16'
            if invalid=='owner': c.sources['one']['qkv'][0][0].version+=1
            with self.assertRaises(ValueError): PositionMapping(index,hits,target,ad,ns,'weights')

    def test_multiple_groups_keep_order_and_storage(self):
        ids=[1,2,3]*4; c,l=fixture([('one',ids,'a'),('two',ids,'a')]); index=FastOccurrenceIndex(c,l)
        first=index.select(ids,0,'a','model','weights',l,'max')[0]
        second=c.entries[(0,tuple(ids[3:6]),'two',3,6,'a','model')]
        hits=[first[0],CacheHit(first[1].window,second),*first[2:]]
        plan=PositionMapping(index,hits,ids,'a','model','weights')
        self.assertEqual([g['query'] for g in plan.groups],['one','two'])
        self.assertIs(plan.groups[1]['backing'],c.sources['two']['qkv'])
        self.assertEqual(plan.groups[1]['source'],(3,4,5))
        self.assertEqual(len(set(p for g in plan.groups for p in g['target'])),12)

    def test_cli_failure_collision_resume_and_no_policy_edits(self):
        m=runner()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run'; args=['--profile','smoke','--device','cpu','--output-dir',str(root)]
            a=m.args(args); self.assertEqual(a.executor,'legacy'); self.assertEqual(a.occurrence_selector,'optimized')
            with patch.object(m,'execute',side_effect=RuntimeError('missing Torch')): self.assertEqual(m.main(args),1)
            for name in ('manifest.json','executor_parity.json','projection_audit.json','latency_comparison.csv',
                         'executor_breakdown.csv','memory_accounting.csv','correctness_comparison.csv','summary.json'):
                self.assertTrue((root/name).exists())
            with self.assertRaises(FileExistsError): m.main(args)
            with patch.object(m,'execute',side_effect=MemoryError('out of memory')): self.assertEqual(m.main(args+['--resume']),1)
            self.assertEqual(json.loads((root/'manifest.json').read_text())['status'],'OOM')
            with self.assertRaises(ValueError): m.main(args+['--resume','--seed','4'])
            with self.assertRaises(SystemExit): m.args(args+['--occurrence-selector','legacy'])

    def test_failed_qkv_cannot_pass_optimization_gate(self):
        m=runner()
        def row(e,passed):
            return dict(row_id=e,source_snapshot_id='s',phase='measured',repeat=0,length=128,storage='cpu',coverage='max',
                executor=e,instrumentation='minimal',target_warm_ms=30 if e=='legacy' else 20,selection_ms=2,
                full_request_ms=10,executor_parity=dict(passed=passed),vs_full=dict(logit_allclose=False),
                projection_rows={},selected_provenance=[],qkv_parity={},breakdown={})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            m.reports(root,[dict(rows=[row('legacy',True),row('batched',False)])])
            self.assertFalse(json.loads((root/'summary.json').read_text())['groups'][0]['warm_improvement'])
            m.reports(root,[dict(rows=[row('legacy',True),row('batched',True)])])
            result=json.loads((root/'summary.json').read_text())['groups'][0]
            self.assertTrue(result['warm_improvement']); self.assertEqual(result['strict_full_passes'],0)
            self.assertIsNone(result['executor_gpu_speedup'])


TORCH=importlib.util.find_spec('torch') is not None
@unittest.skipUnless(TORCH,'NOT EXECUTED: Torch absent; no real tensor/GPU parity claim')
class TensorTests(unittest.TestCase):
    def exercise(self,device,storage):
        import torch
        import semcache.edgelora.batched_mixed_projection as batched
        import semcache.edgelora.mixed_projection as legacy
        from semcache.diagnostics.r1b3_occurrences import OccurrenceCatalog
        torch.manual_seed(12)
        class Projection(torch.nn.Linear):
            def get_base_layer(self): return self
        modules={name:Projection(4,4).to(device).eval() for name in 'qkv'}
        adapter=NS(layers=[0],projection_modules=lambda l:modules)
        for n in (3,9,32,128,256):
            ids=[i%6 for i in range(n)]; layout={0:((4,torch.float32),)*3}
            c=OccurrenceCatalog('m','w',storage,10000000)
            source={0:tuple(torch.randn(1,n,4,device=device) for _ in 'qkv')}
            for query in ('one','two'): c.add_source(query,ids,0,'a',source,layout,prefill_complete=True)
            # Exercise real strided owners and regenerate matching occurrence views.
            for query in c.sources:
                c.sources[query]['qkv'][0]=tuple(torch.stack((t,t),dim=-1)[...,0] for t in c.sources[query]['qkv'][0])
            for e in c.entries.values():
                q=e.qkv_metadata['source_query_id']; s,t=e.positions
                e.tensors[0]=tuple(x[:,s:t].detach() for x in c.sources[q]['qkv'][0])
            index=FastOccurrenceIndex(c,layout)
            hidden=torch.randn(1,n,4,device=device)
            for fraction in (0,.25,.5,.75,'max'):
                hits,mask,_=index.select(ids,0,'a','m','w',layout,fraction)
                if len(hits)>1:
                    h=hits[1]; identity=(0,h.window.token_ids,'two',h.window.start,h.window.end,'a','m')
                    hits[1]=CacheHit(h.window,c.entries[identity])
                plan=PositionMapping(index,hits,ids,'a','m','w')
                with patch.object(legacy,'validate_projection'),patch.object(batched,'validate_projection'):
                    with legacy.mixed_projection_path(adapter,'a',hits,n) as a:
                        for role in 'qkv': modules[role](hidden)
                    with batched.batched_mixed_projection_path(adapter,'a',hits,n,mapping=plan) as b:
                        for role in 'qkv': modules[role](hidden)
                self.assertEqual(a.records,b.records); self.assertTrue(torch.equal(a.reused_mask,b.reused_mask))
                self.assertTrue(compare_projections(a.projections,b.projections,mask)['passed'])
                self.assertEqual(b.memory['gather_calls'],len(plan.groups)*3)
                self.assertEqual(b.memory['gather_conversion_live_bytes_upper_bound']>0,bool(hits))
                self.assertGreater(b.memory['index_tensor_bytes'],0)
                self.assertFalse(any('forward' in m.__dict__ for m in modules.values()))
            # A plan must retain owners, and must detect modification before use.
            del source
            with torch.no_grad(): c.sources['one']['qkv'][0][0].add_(1)
            with self.assertRaises(ValueError):
                with batched.batched_mixed_projection_path(adapter,'a',hits,n,mapping=plan): pass

    def test_cpu_strided_grouped_and_edge_cases(self): self.exercise('cpu','cpu')
    def test_cuda_residency(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable; NOT EXECUTED')
        for storage in ('cpu','cuda:0'): self.exercise('cuda:0',storage)

    def test_dtype_conversion_and_invalid_shape(self):
        import torch
        import semcache.edgelora.batched_mixed_projection as batched
        import semcache.edgelora.mixed_projection as legacy
        from semcache.diagnostics.r1b3_occurrences import OccurrenceCatalog
        class Projection(torch.nn.Linear):
            def get_base_layer(self): return self
        modules={r:Projection(4,4).double().eval() for r in 'qkv'}
        adapter=NS(layers=[0],projection_modules=lambda l:modules)
        ids=[1,2,3,1]; layout={0:((4,torch.float32),)*3}
        c=OccurrenceCatalog('m','w','cpu',100000)
        c.add_source('one',ids,0,'a',{0:tuple(torch.randn(1,4,4) for _ in 'qkv')},layout,prefill_complete=True)
        index=FastOccurrenceIndex(c,layout); hits,mask,_=index.select(ids,0,'a','m','w',layout,'max')
        plan=PositionMapping(index,hits,ids,'a','m','w'); hidden=torch.randn(1,4,4,dtype=torch.float64)
        with patch.object(legacy,'validate_projection'),patch.object(batched,'validate_projection'):
            with legacy.mixed_projection_path(adapter,'a',hits,4) as a:
                for r in 'qkv': modules[r](hidden)
            with batched.batched_mixed_projection_path(adapter,'a',hits,4,mapping=plan) as b:
                for r in 'qkv': modules[r](hidden)
            self.assertTrue(compare_projections(a.projections,b.projections,mask)['passed'])
            self.assertEqual(b.memory['conversion_copy_calls'],3)
            self.assertGreater(b.memory['conversion_allocation_bytes_total'],0)
            with self.assertRaises(ValueError):
                with batched.batched_mixed_projection_path(adapter,'a',hits,4,mapping=plan):
                    modules['q'](hidden[:,:3])
            self.assertFalse(any('forward' in m.__dict__ for m in modules.values()))

    def test_observer_original_parity_and_restore(self):
        import torch
        import semcache.edgelora.mixed_projection as legacy
        from semcache.edgelora.batched_mixed_projection import ExecutorBook,observe_legacy
        class Projection(torch.nn.Linear):
            def get_base_layer(self): return self
        modules={r:Projection(4,4).eval() for r in 'qkv'}; ad=NS(layers=[0],projection_modules=lambda l:modules)
        hidden=torch.randn(1,4,4); results=[]; book=ExecutorBook('cpu')
        from contextlib import nullcontext
        with patch.object(legacy,'validate_projection'):
            for context in (nullcontext(),observe_legacy(book)):
                with context,legacy.mixed_projection_path(ad,'a',[],4) as audit:
                    for r in 'qkv': modules[r](hidden)
                results.append(audit.projections)
        self.assertTrue(compare_projections(*results,[False]*4)['passed'])
        self.assertIsNone(sys.gettrace())
        self.assertEqual(book.export()['full_mixed_projection']['calls'],3)
        self.assertIsNone(book.export()['source_conversion']['host_ms'])


MODELS=all(importlib.util.find_spec(x) is not None for x in ('torch','transformers','peft'))
@unittest.skipUnless(MODELS,'NOT EXECUTED: Torch/Transformers/PEFT absent; real model parity required')
class ModelTests(unittest.TestCase):
    def exercise(self,device):
        import torch
        from contextlib import contextmanager
        from test_r1_raw_semcache import TinyModelTests
        from semcache.metrics.r1_runtime import engine_request
        from semcache.diagnostics.r1b3_occurrences import OccurrenceCatalog,capture_completed_source
        from semcache.edgelora.batched_mixed_projection import batched_mixed_projection_path
        from semcache.semantic.encoder import ControlledEncoder
        from semcache.semantic.intent_clusterer import IntentClusterer
        from semcache.cache.global_cache import GlobalCache
        from semcache.semcache_engine import SemCacheEngine
        import semcache.semcache_engine as em
        import semcache.edgelora.mixed_projection as mm
        from semcache.metrics.r1b1 import snapshot,compare_snapshots
        f=TinyModelTests(); f.setUp(); dtype=torch.float16 if device.startswith('cuda') else torch.float32
        f.model.to(device=device,dtype=dtype)
        ids=[2,3,4]*4; text=' '.join(map(str,ids)); cluster=IntentClusterer(1); cluster.initialize([[0.]])
        engine=SemCacheEngine(f.model,lambda _:dict(input_ids=ids),f.adapter,ControlledEncoder({text:[0.]}),cluster,
            GlobalCache(100000),impact_execution='token_precompute',capture_impact_values=True)
        with capture_completed_source() as captured: engine_request(engine,text,'user_a','source','SEMCACHE_LOOKUP_NO_REUSE',True)
        layout={l:((16,dtype),)*3 for l in range(2)}; original=mm.mixed_projection_path
        for storage in (['cpu',device] if device.startswith('cuda') else ['cpu']):
            c=OccurrenceCatalog('m','w',storage,100000); c.add_source('source',ids,0,'user_a',captured,layout,prefill_complete=True)
            index=FastOccurrenceIndex(c,layout)
            state=[(e,e.frequency,e.impact,e.created_at,e.last_access,e.updated_at) for e in c.entries.values()]
            for coverage in (0,.25,'max'):
                outputs=[]; audits=[]; policies=[]
                for executor in ('legacy','batched'):
                    for e,freq,impact,created,last,updated in state:
                        e.frequency,e.impact,e.created_at,e.last_access,e.updated_at=freq,impact,created,last,updated
                    target=runner().b1().clone_source_engine(engine); held={}
                    def select(h,n,t,u): return index.select(t,0,u,'m','w',layout,coverage)[:2]
                    @contextmanager
                    def context(ad,user,hits,n,**kw):
                        ctx=original(ad,user,hits,n,**kw) if executor=='legacy' else batched_mixed_projection_path(ad,user,hits,n,
                            mapping=PositionMapping(index,hits,ids,user,'m','w'),**kw)
                        with ctx as audit: yield audit
                        held['audit']=audit
                    with patch.object(em,'select_position_aligned_diagnostic',select),patch.object(mm,'mixed_projection_path',context):
                        result=engine_request(target,text,'user_a','target','SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC',False)
                    outputs.append(result['logits']); audits.append(held['audit']); policies.append(snapshot(target,result))
                self.assertTrue(torch.equal(*outputs)); self.assertEqual(audits[0].records,audits[1].records)
                self.assertTrue(compare_projections(audits[0].projections,audits[1].projections,audits[0].reused_mask.tolist())['passed'])
                self.assertTrue(compare_snapshots(*policies)['passed'])
    def test_cpu_tiny_opt(self): self.exercise('cpu')
    def test_gpu_tiny_opt(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable; NOT EXECUTED')
        self.exercise('cuda:0')


if __name__=='__main__': unittest.main()
