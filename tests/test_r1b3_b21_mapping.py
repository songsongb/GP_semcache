"""CPU-only descriptor/cache differential tests; FakeTensor is metadata only."""
import random, sys, unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'tests'))
from test_r1b3_fast_selection import fixture
from semcache.diagnostics.r1b3_fast_selection import FastOccurrenceIndex,compare_selection
from semcache.edgelora.batched_mapping_optimized import SourceBackingDescriptors,PreparedMappingCache,PreparedMapping,balanced_orders,paired_statistics
from semcache.semantic.hit_selection import CacheHit
from semcache.semantic.subsequence import Subsequence

class FakeIndexTensor:
    def __init__(self,values,device='cpu'): self.values=tuple(values); self.device=device
    def numel(self): return len(self.values)
    def element_size(self): return 8
    def to(self,device): return FakeIndexTensor(self.values,str(device))

class TorchStub:
    long='long'
    @staticmethod
    def tensor(values,dtype=None,device='cpu'): return FakeIndexTensor(values,str(device))

class MappingTests(unittest.TestCase):
    def selected(self,n=32,coverage='max'):
        ids=[i%7 for i in range(n)]; catalog,layout=fixture([('source',ids,'a'),('other',ids,'a')])
        index=FastOccurrenceIndex(catalog,layout); hits,mask,rows=index.select(ids,0,'a','model','weights',layout,coverage)
        return ids,catalog,layout,index,hits,mask

    def test_descriptor_and_mapping_exactness_randomized(self):
        rng=random.Random(2101)
        for n in [32,64,128,256]+[rng.randrange(6,70) for _ in range(20)]:
            ids=[rng.randrange(6) for _ in range(n)]; c,l=fixture([('source',ids,'a')]); ix=FastOccurrenceIndex(c,l)
            desc=SourceBackingDescriptors(ix); hits,mask,_=ix.select(ids,0,'a','model','weights',l,'max')
            p=PreparedMapping(desc,hits,ids,'a','model','weights','cpu','fp16')
            self.assertEqual(list(p.mask),[int(x) for x in mask]); self.assertEqual(p.reused,sum(mask))
            self.assertEqual(p.selected_sequence,tuple((h.entry.qkv_metadata['source_query_id'],*h.entry.positions,
                h.window.start,h.window.end,h.entry.token_ids,id(h.entry)) for h in hits))
            self.assertEqual(all(p.contiguous),bool(hits))
            if n in (128,256): self.assertEqual(p.reused,n//3*3)

    def test_no_partial_max_and_exact_target_cache(self):
        ids,c,l,ix,hits,mask=self.selected(32)
        desc=SourceBackingDescriptors(ix); cache=PreparedMappingCache(desc,capacity=3)
        with patch.dict(sys.modules,{'torch':TorchStub}):
            first,hit=cache.get(hits,ids,'a','model','weights','cpu','fp32',mode='gather')
            self.assertFalse(hit)
            again,hit=cache.get(hits,list(ids),'a','model','weights','cpu','fp32',mode='gather')
            self.assertTrue(hit); self.assertIs(first,again)
            altered=list(ids); altered[-1]=99
            # Zero-reuse target is still a distinct mapping key.
            zero,hit=cache.get([],altered,'a','model','weights','cpu','fp32',mode='gather')
            self.assertFalse(hit); self.assertEqual(zero.reused,0)
            self.assertEqual(cache.stats()['hits'],1); self.assertEqual(cache.stats()['misses'],2)
            self.assertGreater(cache.stats()['estimated_memory_bytes'],0)
            # Preparing the contiguous ablation adds only missing index tensors.
            again,hit=cache.get(hits,ids,'a','model','weights','cpu','fp32',mode='contiguous')
            self.assertTrue(hit); self.assertIn('gather',again.prepared_modes); self.assertIn('contiguous',again.prepared_modes)

    def test_validation_rejects_adapter_namespace_prefix_stale_source_and_view(self):
        for change in ('adapter','namespace','prefix','owner','view','entrymap'):
            ids,c,l,ix,hits,mask=self.selected(24); desc=SourceBackingDescriptors(ix)
            if change=='adapter': args=('other','model','weights',ids)
            elif change=='namespace': args=('a','other','weights',ids)
            elif change=='prefix':
                target=list(ids); target[0]=99; args=('a','model','weights',target)
            else: args=('a','model','weights',ids)
            if change=='owner': c.sources['source']['qkv'][0][0].owner.version+=1
            elif change=='view': hits[0].entry.tensors[0][0].shape=(1,2,4)
            elif change=='entrymap': hits[0].entry.tensors={}
            with self.assertRaises(ValueError):
                with patch.dict(sys.modules,{'torch':TorchStub}): PreparedMappingCache(desc).get(hits,args[3],args[0],args[1],args[2],'cpu','fp32')

    def test_selected_occurrence_order_multiple_source_groups(self):
        ids,c,l,ix,hits,mask=self.selected(18)
        h=hits[0]; other=c.entries[(0,h.entry.token_ids,'other',h.window.start,h.window.end,'a','model')]
        mix=[CacheHit(h.window,other),*hits[1:]]
        desc=SourceBackingDescriptors(ix); p=PreparedMapping(desc,mix,ids,'a','model','weights','cpu','fp16')
        self.assertEqual([g['query'] for g in p.groups],['other','source'])
        self.assertEqual(p.groups[0]['source'][0],h.entry.positions[0])
        self.assertEqual(p.groups[0]['target'][0],h.window.start)
        self.assertTrue(p.contiguous[0] is False or p.contiguous[0] is True)

    def test_order_balance_and_paired_statistics_keep_outlier(self):
        modes=('legacy','v1','v2','contiguous'); schedule=balanced_orders(modes,48,22)
        self.assertEqual(len(schedule),48)
        self.assertEqual({tuple(sorted(x)) for x in schedule},{tuple(sorted(modes))})
        counts={m:sum(order[0]==m for order in schedule) for m in modes}
        self.assertLessEqual(max(counts.values())-min(counts.values()),1)
        result=paired_statistics([10,10,500,10],[9,9,9,9],seed=1,bootstrap_samples=1000)
        self.assertEqual(result['max'],491); self.assertEqual(result['count'],4)
        self.assertEqual(result['optimized_win_count'],4); self.assertEqual(result['independent_run_count'],1)

    def test_source_descriptor_cost_separate(self):
        ids,c,l,ix,hits,mask=self.selected(16); d=SourceBackingDescriptors(ix)
        self.assertEqual(d.additional_qkv_bytes,0); self.assertGreaterEqual(d.build_ms,0); self.assertGreater(d.python_bytes_estimate,0)
        d.validate(hits)

if __name__=='__main__': unittest.main()

TORCH=__import__('importlib').util.find_spec('torch') is not None
@unittest.skipUnless(TORCH,'NOT EXECUTED: Torch absent; real tensor gather/scatter parity unavailable')
class TensorParityTests(unittest.TestCase):
    def run_device(self,device,storage):
        import torch
        from contextlib import contextmanager
        from semcache.diagnostics.r1b3_occurrences import OccurrenceCatalog
        from semcache.edgelora.batched_mixed_projection import PositionMapping as V1Map,batched_mixed_projection_path as V1
        from semcache.edgelora.batched_mapping_optimized import batched_v2_projection_path as V2
        import semcache.edgelora.mixed_projection as legacy_mod
        import semcache.edgelora.batched_mixed_projection as v1_mod
        import semcache.edgelora.batched_mapping_optimized as v2_mod
        from semcache.models.qkv_projection import extract_qkv_positions
        torch.manual_seed(17); n=32; ids=[i%7 for i in range(n)]; dtype=torch.float32
        layout={0:((4,dtype),)*3}; catalog=OccurrenceCatalog('model','weights',storage,1000000)
        qkv={0:tuple(torch.randn(1,n,4,device=device) for _ in 'qkv')}
        catalog.add_source('source',ids,0,'a',qkv,layout,prefill_complete=True)
        catalog.add_source('second',ids,0,'a',qkv,layout,prefill_complete=True)
        index=FastOccurrenceIndex(catalog,layout); descriptor=SourceBackingDescriptors(index); cache=PreparedMappingCache(descriptor,4)
        class Projection(torch.nn.Linear):
            def get_base_layer(self): return self
        modules={r:Projection(4,4,bias=False).to(device).eval() for r in 'qkv'}
        adapter=NS(layers=[object()],projection_modules=lambda _:modules); hidden=torch.randn(1,n,4,device=device)
        saved_validate=(legacy_mod.validate_projection,v1_mod.validate_projection,v2_mod.validate_projection)
        legacy_mod.validate_projection=v1_mod.validate_projection=v2_mod.validate_projection=lambda *a,**k:None
        try:
            for coverage in (0,.25,'max'):
                hits,mask,_=index.select(ids,0,'a','model','weights',layout,coverage)
                plans={}; audits={}
                for mode in ('legacy','v1','v2','contiguous'):
                    if mode=='legacy': ctx=legacy_mod.mixed_projection_path(adapter,'a',hits,n)
                    elif mode=='v1':
                        plan=V1Map(index,hits,ids,'a','model','weights'); ctx=V1(adapter,'a',hits,n,mapping=plan); plans[mode]=plan
                    else:
                        plan,cache_hit=cache.get(hits,ids,'a','model','weights',device,torch.float32,mode='contiguous' if mode=='contiguous' else 'gather')
                        plans[mode]=plan
                        ctx=V2(adapter,'a',hits,n,mapping=plan,contiguous=mode=='contiguous')
                    with ctx as audit:
                        for role in 'qkv': modules[role](hidden)
                    audits[mode]=audit
                for mode in ('v1','v2','contiguous'):
                    self.assertEqual(audits['legacy'].records,audits[mode].records)
                    self.assertTrue(torch.equal(audits['legacy'].reused_mask,audits[mode].reused_mask))
                    self.assertTrue(__import__('semcache.edgelora.batched_mixed_projection',fromlist=['compare_projections']).compare_projections(
                        audits['legacy'].projections,audits[mode].projections,mask)['passed'])
                    self.assertTrue(torch.equal(sum(audits['legacy'].projections[0][r] for r in 'qkv'),
                                                sum(audits[mode].projections[0][r] for r in 'qkv')))
                if coverage=='max': self.assertEqual(sum(mask),30)
                if coverage==0: self.assertEqual(sum(mask),0)
            # Recreated CacheHit wrappers retain a cache hit by occurrence sequence.
            again,hit=index.select(ids,0,'a','model','weights',layout,'max')
            value,cached=cache.get(again,ids,'a','model','weights',device,torch.float32)
            self.assertTrue(cached); self.assertEqual(value.reused,30)
            with torch.no_grad(): catalog.sources['source']['qkv'][0][0].add_(1)
            with self.assertRaises(ValueError): cache.get(again,ids,'a','model','weights',device,torch.float32)
        finally:
            legacy_mod.validate_projection,v1_mod.validate_projection,v2_mod.validate_projection=saved_validate
    def test_cpu_raw(self): self.run_device('cpu','cpu')
    def test_cuda_gpu_raw(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('NOT EXECUTED: CUDA unavailable')
        self.run_device('cuda:0','cuda:0')
    def test_cuda_model_cpu_cache(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('NOT EXECUTED: CUDA unavailable')
        self.run_device('cuda:0','cpu')

class RunnerAndAggregateTests(unittest.TestCase):
    def load(self,path,name):
        import importlib.util
        spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
    def test_runner_cli_artifacts_resume_and_default_legacy(self):
        import tempfile,json
        from unittest.mock import patch
        m=self.load('77_benchmark_r1b3_b21_setup.py','runner77')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'failed'; base=['--profile','smoke','--device','cpu','--cache-device','cpu','--executor','all','--output-dir',str(out)]
            self.assertEqual(m.args(base).modes,['legacy','v1','v2','contiguous'])
            self.assertEqual(m.args(['--profile','smoke','--device','cpu','--cache-device','cpu','--output-dir',str(Path(d)/'default')]).modes,['legacy'])
            with patch.object(m,'execute',side_effect=RuntimeError('Torch missing')): self.assertEqual(m.main(base),1)
            names=('manifest.json','mapping_parity.json','executor_parity.json','setup_breakdown.csv','latency_comparison.csv',
                'paired_differences.csv','gpu_state.json','memory_accounting.csv','correctness_comparison.csv','summary.json')
            self.assertTrue(all((out/x).exists() for x in names))
            self.assertEqual(json.loads((out/'manifest.json').read_text())['status'],'FAILED')
            self.assertEqual(json.loads((out/'summary.json').read_text())['status'],'FAILED')
            with self.assertRaises(FileExistsError): m.main(base)
            with patch.object(m,'execute',side_effect=MemoryError('out of memory')): self.assertEqual(m.main(base+['--resume']),1)
            self.assertEqual(json.loads((out/'manifest.json').read_text())['status'],'OOM')
            with self.assertRaises(SystemExit): m.args(['--profile','pilot','--device','cpu','--measured-runs','10','--output-dir','x'])
    def test_aggregator_run_cluster_bootstrap_and_collision(self):
        import tempfile,json
        from semcache.metrics.r1 import atomic_json,digest
        agg=self.load('78_aggregate_r1b3_b21.py','aggregate78')
        with tempfile.TemporaryDirectory() as d:
            roots=[]
            for idx,delta in enumerate((2.,4.,3.)):
                root=Path(d)/f'run{idx}'; (root/'checkpoints').mkdir(parents=True); roots.append(root)
                config=dict(dtype='float16',lengths=[128],cache_device='cuda',instrumentation='minimal',independent_run_id=f'r{idx}')
                manifest=dict(identity=f'id{idx}',status='COMPLETE',cache_classification='ORACLE_DIAGNOSTIC',config=config,
                    runtime=dict(gpu_uuid='GPU-X',model=dict(resolved_model_revision='abc')))
                atomic_json(root/'manifest.json',manifest)
                rows=[]
                for repeat in range(4):
                    for mode,latency in (('legacy',100.),('v2',100.-delta)):
                        rows.append(dict(row_id=f'{repeat}-{mode}',phase='measured',length=128,storage='cuda:0',coverage='max',
                            instrumentation='minimal',repeat=repeat,mode=mode,target_warm_ms=latency,
                            executor_parity=dict(passed=True)))
                atomic_json(root/'checkpoints'/'x.json',dict(identity=f'id{idx}',status='COMPLETE',rows=rows,checksum=digest(rows)))
            out=Path(d)/'aggregate'; command=['--run-dirs',','.join(map(str,roots)),'--output-dir',str(out),'--bootstrap-samples','100']
            self.assertEqual(agg.main(command),0)
            result=json.loads((out/'summary.json').read_text())['groups'][0]
            self.assertEqual(result['independent_run_count'],3); self.assertAlmostEqual(result['mean_difference_ms'],3.)
            self.assertTrue(result['reproducible_warm_improvement']); self.assertTrue((out/'overall_summary.csv').exists())
            with self.assertRaises(FileExistsError): agg.main(command)
