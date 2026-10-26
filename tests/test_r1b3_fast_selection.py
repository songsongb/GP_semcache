"""Differential selector tests. Fake tensor metadata is NOT model/GPU evidence."""
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r1b3_occurrences import OccurrenceCatalog
from semcache.diagnostics.r1b3_fast_selection import FastOccurrenceIndex,HostBook,profile_legacy,compare_selection,prefix_limit
from semcache.semantic.subsequence import SubsequenceExtractor
from semcache.cache.cache_entry import CacheEntry
from semcache.metrics.r1 import atomic_json,digest


class FakeTensor:
    """Only metadata/aliasing for CPU-only control-flow differential tests."""
    next_pointer=100000
    def __init__(self,n,width=4,owner=None,start=0):
        self.owner=owner or self; self.start=start; self.shape=(1,n,width); self.dtype='fp32'; self.device='cpu'
        if owner is None:
            self.version=0; self.pointer=FakeTensor.next_pointer; FakeTensor.next_pointer+=100000; self.bytes=n*width*4
    @property
    def _version(self): return self.owner.version
    def data_ptr(self): return self.owner.pointer+self.start*self.shape[2]*4
    def stride(self): return (self.owner.shape[1]*self.shape[2],self.shape[2],1)
    def untyped_storage(self): return NS(nbytes=lambda:self.owner.bytes)
    def detach(self): return self
    def __getitem__(self,index):
        s=index[1]; start=s.start or 0; end=s.stop if s.stop is not None else self.shape[1]
        return FakeTensor(end-start,self.shape[2],self.owner,self.start+start)


def fixture(sources):
    catalog=OccurrenceCatalog('model','weights','cpu',100000000)
    layout={0:((4,'fp32'),)*3,1:((4,'fp32'),)*3}
    for qid,ids,adapter in sources:
        ids=tuple(ids); qkv={l:tuple(FakeTensor(len(ids)) for _ in range(3)) for l in layout}
        catalog.sources[qid]=dict(ids=ids,qkv=qkv,adapter=adapter,cluster=0,layout=layout)
        for w in SubsequenceExtractor(3).extract(ids):
            identity=(0,w.token_ids,qid,w.start,w.end,adapter,'model')
            tensors={l:tuple(t[:,w.start:w.end] for t in q) for l,q in qkv.items()}
            e=CacheEntry(0,w.token_ids,(w.start,w.end),96,tensors=tensors,qkv_metadata=dict(
                source_query_id=qid,source_query_token_ids=ids,source_adapter=adapter,source_user=adapter,
                model_namespace='model',weight_identity='weights',occurrence_identity=identity,component_scope='total_qkv'))
            catalog.entries[identity]=e; catalog.index.setdefault((0,w.token_ids,w.start,w.end),[]).append(identity)
    return catalog,layout


def module():
    spec=importlib.util.spec_from_file_location('r1b3_b1',ROOT/'scripts/75_benchmark_r1b3_selection.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class OfflineTests(unittest.TestCase):
    def compare(self,catalog,layout,target,adapter='a',namespace='model',weights='weights',coverage='max'):
        index=FastOccurrenceIndex(catalog,layout)
        # Legacy only calls torch.device to normalize its metadata string here.
        # This mock is never used to claim execution/content validation.
        with patch.dict(sys.modules,{'torch':NS(device=lambda x:x)}):
            legacy=catalog.select(target,0,adapter,namespace,weights,layout,coverage)
            fast=index.select(target,0,adapter,namespace,weights,layout,coverage)
        self.assertTrue(compare_selection(legacy,fast)['passed'])
        return legacy,fast,index

    def test_randomized_differential(self):
        rng=random.Random(7103)
        for _ in range(80):
            n=rng.randrange(4,40); source=[rng.randrange(4) for _ in range(n)]
            second=list(source)
            if rng.random()<.7: second[rng.randrange(n)]=9
            catalog,layout=fixture([('z_first',source,'a'),('a_second',second,'a'),('other',source,'b')])
            targets=[source,source+[7,8],source[:rng.randrange(3,n+1)],second]
            for target in targets:
                for coverage in (0,.25,.5,.75,'max'):
                    self.compare(catalog,layout,target,coverage=coverage)

    def test_tie_order_coverage_128_256(self):
        for n,expected in ((128,126),(256,255)):
            ids=[i%54 for i in range(n)]
            c,l=fixture([('z_first',ids,'a'),('a_second',ids,'a')])
            legacy,fast,index=self.compare(c,l,ids)
            self.assertEqual(sum(fast[1]),expected)
            self.assertTrue(all(h.entry.qkv_metadata['source_query_id']=='z_first' for h in fast[0]))
            self.assertEqual(index.additional_qkv_bytes,0)
            self.assertGreater(index.additional_index_memory_bytes,0)

    def test_identity_prefix_lengths_and_positions(self):
        ids=[1,2,3,1,2,3,7,8,9]; c,l=fixture([('one',ids,'a')])
        for args in [('b','model','weights'),('a','wrong','weights'),('a','model','wrong')]:
            self.assertEqual(self.compare(c,l,ids,*args)[1][0],[])
        target=[9]+ids[1:]
        self.assertEqual(self.compare(c,l,target)[1][0],[])
        suffix=ids[:6]+[4,4,4]
        self.assertEqual(sum(self.compare(c,l,suffix)[1][1]),6)
        self.assertEqual(prefix_limit(ids,ids[:4]),4)
        self.assertEqual(prefix_limit(ids,[8]),0)

    def test_invalidation_not_silent_reuse(self):
        for change in ('metadata','mapping','owner','view','bucket'):
            c,l=fixture([('one',[1,2,3]*3,'a')]); index=FastOccurrenceIndex(c,l); e=next(iter(c.entries.values()))
            if change=='metadata': e.qkv_metadata['source_adapter']='other'
            if change=='mapping': e.tensors[0]=tuple(FakeTensor(3) for _ in range(3))
            if change=='owner': c.sources['one']['qkv'][0][0].version+=1
            if change=='view': e.tensors[0][0].shape=(1,2,4)
            if change=='bucket': next(iter(c.index.values())).clear()
            with self.assertRaises(ValueError): index.select([1,2,3]*3,0,'a','model','weights',l,'max')

    def test_profile_legacy_unchanged_and_scoped(self):
        c,l=fixture([('one',[1,2,3]*5,'a')]); ids=list(c.sources['one']['ids']); index=FastOccurrenceIndex(c,l)
        with patch.dict(sys.modules,{'torch':NS(device=lambda x:x)}):
            plain=c.select(ids,0,'a','model','weights',l,'max')
            book=HostBook()
            with profile_legacy(c,book): profiled=c.select(ids,0,'a','model','weights',l,'max')
            self.assertTrue(compare_selection(plain,profiled)['passed'])
            index.guard()
            fast_book=HostBook(); fast=index.select(ids,0,'a','model','weights',l,'max',fast_book)
            self.assertTrue(compare_selection(plain,fast)['passed'])
            with self.assertRaises(RuntimeError):
                with profile_legacy(c,book): raise RuntimeError('test')
            index.guard()
        self.assertIsNone(book.export()['position_filtering']['host_ms'])
        self.assertGreater(book.export()['tensor_backing_validation']['host_ms'],0)
        self.assertGreater(fast_book.export()['prefix_context_validation']['host_ms'],0)

    def test_cli_failure_artifacts_collision_resume(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'run'; argv=['--profile','smoke','--device','cpu','--output-dir',str(output)]
            self.assertEqual(m.args(argv).occurrence_selector,'legacy')
            with patch.object(m,'execute',side_effect=RuntimeError('missing model')): self.assertEqual(m.main(argv),1)
            for name in ('manifest.json','selection_parity.json','selection_breakdown.csv','latency_comparison.csv',
                         'coverage_comparison.csv','correctness_comparison.csv','index_storage_accounting.csv','summary.json'):
                self.assertTrue((output/name).exists())
            with self.assertRaises(FileExistsError): m.main(argv)
            with patch.object(m,'execute',side_effect=KeyboardInterrupt()): self.assertEqual(m.main(argv+['--resume']),1)
            with self.assertRaises(ValueError): m.main(argv+['--resume','--seed','999'])
            rows=[dict(row_id='one')]
            bundle=dict(identity='id',status='COMPLETE',rows=rows,checksum=digest(rows))
            atomic_json(output/'checkpoints/a.json',bundle)
            self.assertEqual(len(m.checkpoints(output,'id')),1)
            atomic_json(output/'checkpoints/b.json',bundle)
            with self.assertRaises(ValueError): m.checkpoints(output,'id')


    def test_report_gate_separates_fp16_failure_and_selector_parity(self):
        m=module()
        def row(selector,allclose=True):
            return dict(row_id=selector,source_snapshot_id='source',phase='measured',repeat=0,length=128,
                storage='cuda:0',coverage='max',selector=selector,instrumentation='minimal',
                selection_ms=10 if selector=='legacy' else 5,target_warm_ms=30 if selector=='legacy' else 25,
                full_request_ms=20,projection_rows=dict(skipped=126),catalog_bytes=100,
                selection_parity=dict(passed=True),vs_legacy=dict(logit_allclose=allclose),
                vs_full=dict(logit_allclose=False),breakdown={})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            m.reports(root,[dict(rows=[row('legacy'),row('optimized',False)])])
            report=json.loads((root/'summary.json').read_text())['groups'][0]
            self.assertFalse(report['selection_optimization_success'])
            m.reports(root,[dict(rows=[row('legacy'),row('optimized')])])
            report=json.loads((root/'summary.json').read_text())['groups'][0]
            self.assertTrue(report['selection_optimization_success'])
            self.assertEqual(report['strict_full_passes'],0)
            self.assertTrue(report['no_safety_claim'])


TORCH=importlib.util.find_spec('torch') is not None
@unittest.skipUnless(TORCH,'Torch absent; real tensor backing/content NOT EXECUTED')
class TensorTests(unittest.TestCase):
    def build(self,n):
        import torch
        c=OccurrenceCatalog('model','weights','cpu',10000000)
        ids=[i%54 for i in range(n)]; layout={0:((4,torch.float32),)*3}
        qkv={0:tuple(torch.arange(n*4,dtype=torch.float32).reshape(1,n,4)+10000*r for r in range(3))}
        c.add_source('source',ids,0,'a',qkv,layout,prefill_complete=True)
        return c,ids,layout

    def test_real_tensors_and_lifetime_128_256(self):
        import gc
        import torch
        for n in (128,256):
            c,ids,l=self.build(n); index=FastOccurrenceIndex(c,l)
            for coverage in (0,.25,.5,.75,'max'):
                a=c.select(ids,0,'a','model','weights',l,coverage)
                b=index.select(ids,0,'a','model','weights',l,coverage)
                self.assertTrue(compare_selection(a,b)['passed'])
                for x,y in zip(a[0],b[0]):
                    for t,u in zip(x.entry.tensors[0],y.entry.tensors[0]):
                        self.assertIs(t,u); self.assertTrue(torch.equal(t,u))
            del c; gc.collect()
            self.assertEqual(sum(index.select(ids,0,'a','model','weights',l,'max')[1]),n//3*3)

    def test_data_rebinding_and_inplace_mutation_invalidate(self):
        import torch
        for rebind in (False,True):
            c,ids,l=self.build(32); index=FastOccurrenceIndex(c,l)
            t=next(iter(c.entries.values())).tensors[0][0]
            if rebind: t.data=torch.zeros_like(t)
            else: t.add_(1)
            with self.assertRaises(ValueError): index.select(ids,0,'a','model','weights',l,'max')


MODELS=all(importlib.util.find_spec(x) is not None for x in ('torch','transformers','peft'))
@unittest.skipUnless(MODELS,'Torch/Transformers/PEFT absent; actual projection and GPU parity NOT EXECUTED')
class ModelTests(unittest.TestCase):
    def exercise(self,device):
        import torch
        from test_r1_raw_semcache import TinyModelTests
        from semcache.metrics.r1_runtime import engine_request
        from semcache.diagnostics.r1b3_occurrences import capture_completed_source,catalog_override
        from semcache.semantic.encoder import ControlledEncoder
        from semcache.semantic.intent_clusterer import IntentClusterer
        from semcache.cache.global_cache import GlobalCache
        from semcache.semcache_engine import SemCacheEngine
        from semcache.metrics.r1 import audit_projection_records
        from semcache.metrics.r1b1 import snapshot,compare_snapshots
        f=TinyModelTests(); f.setUp(); dtype=torch.float16 if device.startswith('cuda') else torch.float32
        f.model.to(device=device,dtype=dtype)
        ids=[2,3,4]*4; text=' '.join(map(str,ids)); cluster=IntentClusterer(1); cluster.initialize([[0.]])
        engine=SemCacheEngine(f.model,lambda _:dict(input_ids=ids),f.adapter,ControlledEncoder({text:[0.]}),cluster,GlobalCache(100000),impact_execution='token_precompute',capture_impact_values=True)
        with capture_completed_source() as captured: engine_request(engine,text,'user_a','source','SEMCACHE_LOOKUP_NO_REUSE',True)
        layout={l:((16,dtype),)*3 for l in range(2)}
        runner=module()
        for storage in (['cpu',device] if device.startswith('cuda') else ['cpu']):
            c=OccurrenceCatalog('m','w',storage,100000); c.add_source('source',ids,0,'user_a',captured,layout,prefill_complete=True)
            index=FastOccurrenceIndex(c,layout)
            for fraction in (0,.25,.5,.75,'max'):
                a=c.select(ids,0,'user_a','m','w',layout,fraction); b=index.select(ids,0,'user_a','m','w',layout,fraction)
                self.assertTrue(compare_selection(a,b)['passed'])
                state=[(e,e.frequency,e.impact,e.created_at,e.last_access,e.updated_at) for e in c.entries.values()]
                outputs=[]; policies=[]; audits=[]
                import semcache.semcache_engine as engine_module
                for method in (c.select,index.select):
                    for e,freq,impact,created,last,updated in state:
                        e.frequency,e.impact,e.created_at,e.last_access,e.updated_at=freq,impact,created,last,updated
                    target=runner.clone_source_engine(engine)
                    def select(h,n,t,u): return method(t,0,u,'m','w',layout,fraction)[:2]
                    with patch.object(engine_module,'select_position_aligned_diagnostic',select):
                        result=engine_request(target,text,'user_a','target','SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC',False)
                    n=result['summary']['reused_unique_token_count']
                    audits.append(audit_projection_records(list(result['projection_audit'].values()),12,2,n))
                    outputs.append(result['logits']); policies.append(snapshot(target,result)); index.guard()
                self.assertEqual(audits[0],audits[1]); self.assertTrue(torch.equal(outputs[0],outputs[1]))
                self.assertTrue(compare_snapshots(policies[0],policies[1])['passed'])
            # Standard in-place changes invalidate the source proof.
            with torch.no_grad(): next(iter(c.entries.values())).tensors[0][0].add_(1)
            with self.assertRaises(ValueError): index.guard()

    def test_cpu_model(self): self.exercise('cpu')
    def test_cuda_model(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable; NOT EXECUTED')
        self.exercise('cuda:0')


if __name__=='__main__': unittest.main()
