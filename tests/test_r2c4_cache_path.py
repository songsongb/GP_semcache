"""Metadata/control-flow tests; actual Tensor/OPT execution is dependency-gated."""
import copy
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from test_r1b3_fast_selection import fixture
from semcache.diagnostics.r1b3_fast_selection import FastOccurrenceIndex
from semcache.edgelora.batched_mapping_optimized import SourceBackingDescriptors,PreparedMappingCache
from semcache.diagnostics import r2c4_cache_path as cp
from semcache.diagnostics.r2c4_benchmark import equivalence
from semcache.diagnostics.r2c4_breakdown import flatten
from semcache.metrics.r1 import digest


class Device:
    def __init__(self,v):self.value=str(v);self.type=self.value.split(':')[0];self.index=int(self.value.split(':')[1]) if ':' in self.value else None
    def __str__(self):return self.value
    def __eq__(self,other):return str(self)==str(other)
class IndexTensor:
    def __init__(self,values,device='cpu'):
        self.values=tuple(values);self.device=str(device);self.dtype='long';self.shape=(len(values),);self._version=0
    def numel(self):return len(self.values)
    def element_size(self):return 8
    def to(self,device):return IndexTensor(self.values,device)
    def data_ptr(self):return id(self)
    def stride(self):return (1,)
    def untyped_storage(self):return NS(nbytes=lambda:len(self.values)*8)
    def detach(self):return self
    def cpu(self):return self
    def tolist(self):return list(self.values)
class TorchStub:
    long='long';float32='torch.float32';device=Device
    @staticmethod
    def tensor(values,dtype=None,device='cpu'):return IndexTensor(values,device)


def runtime(coverage='max'):
    ids=[i%7 for i in range(12)];catalog,layout=fixture([('source',ids,'user_a')])
    index=FastOccurrenceIndex(catalog,layout);descriptor=SourceBackingDescriptors(index);cache=PreparedMappingCache(descriptor)
    r=NS(ids=ids,index=index,catalog=catalog,descriptor=descriptor,caches={'contiguous':cache},adapter=NS(layers=[0,1]),
        a=NS(device='cpu',dtype='float32',capacity_mib=512),namespace='model',weights='weights',snapshot='source',storage='cpu',
        model_weight_identity='base_and_lora',execution_code_sha256='code',coverage=coverage,source_hashes={'q':'initial'},content='initial')
    r.guard=index.guard;r.canonical_hashes=lambda:{'q':r.content}
    r.selected_hits=lambda cov:index.select(ids,0,'user_a','model','weights',layout,cov)[0]
    r.binding=lambda base:dict(base_mode=base,token_ids=ids,model_namespace=r.namespace,source_snapshot=r.snapshot)
    return r


class Flow:
    def __init__(self,r):
        self.r=r;self.cache=r.caches['contiguous'];self.namespace=r.namespace;self.weights=r.weights
        self.device=r.a.device;self.dtype=TorchStub.float32;self.plan=None;self.completed=False
    def select(self,hits,ids,user):
        self.hits=tuple(hits);self.ids=tuple(ids);self.user=user
        if hits:self.plan,self.cache_hit=self.cache.get(hits,ids,user,self.namespace,self.weights,self.device,self.dtype,mode='contiguous',validate=False)
    @contextmanager
    def context(self,*args,**kwargs):
        if self.plan:self.plan.descriptor.validate(self.hits)
        self.audit=NS(projections={l:dict(zip('qkv',q)) for l,q in self.r.catalog.sources['source']['qkv'].items()})
        try:yield self.audit
        finally:self.completed=True
    def report(self):return {'completed':self.completed},{}


class PathTests(unittest.TestCase):
    def test_oracle_call_isolation_exact_boundary_zero_controls_and_restore(self):
        import semcache.semcache_engine as em
        original=(em.CPUWallTimer,Flow.select,Flow.context,PreparedMappingCache.get,SourceBackingDescriptors.validate)
        for coverage in (0.,.5,'max'):
            for arm in cp.RAW:
                with self.subTest(coverage=coverage,arm=arm),patch.dict('sys.modules',{'torch':TorchStub}),\
                     patch.object(cp,'tensor_sha',side_effect=lambda t:digest(getattr(t,'values',[]))):
                    r=runtime(coverage);r.b=NS(RequestMapping=Flow)
                    hits=r.selected_hits(coverage)
                    if hits:r.caches['contiguous'].get(hits,r.ids,'user_a','model','weights','cpu',TorchStub.float32,mode='contiguous')
                    lease=cp.OracleLease(r,arm) if arm!=cp.ARMS[2] else None
                    probe=cp.PathProbe(r,arm,lease,True);flow=Flow(r)
                    with probe.scoped():
                        timer=em.CPUWallTimer();timer.__enter__()
                        r.index.guard();flow.select(hits,r.ids,'user_a')
                        with flow.context():pass
                        timer.__exit__(None,None,None);flow.report()
                    if lease:lease.finish(flow)
                    with patch.object(cp,'tensor_sha',side_effect=lambda t: 'outside' if not probe.active else (_ for _ in ()).throw(AssertionError('audit inside timer'))):
                        self.assertEqual(len(probe.output_hashes()),6)
                    probe.active=True
                    with self.assertRaises(cp.MissingEvidence):probe.output_hashes()
                    probe.active=False
                    record={'reused_tokens':sum(h.window.end-h.window.start for h in hits),'mapping_cache_hit':True,'memory':{}}
                    audit=probe.audit(record)
                    self.assertTrue(audit['timing_boundary_verified'])
                    self.assertEqual(probe.count('fast_index_guard'),1)
                    if not hits:
                        self.assertEqual(audit['snapshot_intervention'],'NOT_APPLICABLE')
                        self.assertEqual(probe.count('prepared_mapping_get'),0)
                    elif arm in cp.MAPPING:
                        self.assertEqual(probe.count('prepared_mapping_get'),0)
                        self.assertEqual(probe.count('prepared_mapping_oracle_dispatch'),1)
                        self.assertTrue(audit['prebind_cache_hit']);self.assertIsNone(audit['mapping_cache_hit'])
                    self.assertEqual(original,(em.CPUWallTimer,Flow.select,Flow.context,PreparedMappingCache.get,SourceBackingDescriptors.validate))

    def test_mutated_source_content_stamps_and_indices_rejected(self):
        for mutation in ('content','owner','index'):
            with self.subTest(mutation=mutation),patch.dict('sys.modules',{'torch':TorchStub}),patch.object(cp,'tensor_sha',side_effect=lambda t:digest(getattr(t,'values',[]))):
                r=runtime();lease=cp.OracleLease(r,cp.ARMS[5])
                if mutation=='content':r.content='corrupt'
                if mutation=='owner':next(iter(r.catalog.sources['source']['qkv'].values()))[0].owner.version+=1
                if mutation=='index':lease.plan.index_tensors[('fresh',0)].values=(99,)
                flow=NS(hits=lease.hits,ids=r.ids,user='user_a',namespace=r.namespace,weights=r.weights,device='cpu',dtype=TorchStub.float32,plan=lease.plan)
                with self.assertRaises((ValueError,cp.ParityFailure,cp.MissingEvidence)):lease.finish(flow)

    def test_mapping_oracle_rejects_wrong_selection_and_adapter_namespace(self):
        with patch.dict('sys.modules',{'torch':TorchStub}),patch.object(cp,'tensor_sha',side_effect=lambda t:digest(getattr(t,'values',[]))):
            r=runtime();lease=cp.OracleLease(r,cp.ARMS[4])
            with self.assertRaises(cp.MissingEvidence):lease.match(list(reversed(lease.hits)))
            for field,value in [('adapter','user_b'),('namespace','other'),('weights','other'),('device','cuda:1'),('dtype','torch.float16'),('adapter',None)]:
                with self.subTest(field=field),self.assertRaises(cp.MissingEvidence):lease.match(lease.hits,**{field:value})
            lease.plan.groups[0]['target']=tuple(reversed(lease.plan.groups[0]['target']))
            with self.assertRaises(cp.MissingEvidence):cp.validate_plan(r,lease.plan,lease.hits)

    def test_failure_restores_scoped_methods_and_timer(self):
        import semcache.semcache_engine as em
        original=(em.CPUWallTimer,Flow.select,Flow.context,Flow.report,PreparedMappingCache.get,SourceBackingDescriptors.validate)
        with patch.dict('sys.modules',{'torch':TorchStub}):
            r=runtime();r.b=NS(RequestMapping=Flow)
            with self.assertRaises(RuntimeError):
                with cp.PathProbe(r,cp.ARMS[2]).scoped():raise RuntimeError('injected forward failure')
        self.assertEqual(original,(em.CPUWallTimer,Flow.select,Flow.context,Flow.report,PreparedMappingCache.get,SourceBackingDescriptors.validate))

    def test_policy_impact_qkv_logits_parity_fail_closed(self):
        base=dict(coverage_audit={},projection_audit={},selection_hash='same',selected_provenance=[],output_shape=[1,12,32],
            mapping={'same':True},policy={'history':'same'},impact_values_digest='values',reference_digest='ref',
            assembled_qkv_sha256={'0:q':{'sha256':'bytes','shape':[1,12,4],'dtype':'float32'}},
            impact_real_reduction_calls=0,impact_fallback_calls=0,impact_dispatch_calls=1)
        with patch.dict('sys.modules',{'torch':NS(equal=lambda a,b:a==b)}),patch('semcache.diagnostics.r2c4_benchmark.strict_metrics',return_value={'allclose':True}):
            self.assertTrue(equivalence([1],[1],base,base)['exact_qkv'])
            for field in ('policy','impact_values_digest','assembled_qkv_sha256','projection_audit','mapping'):
                with self.subTest(field=field),self.assertRaises(cp.ParityFailure):equivalence([1],[1],base,dict(base,**{field:'changed'}))
            with self.assertRaises(cp.ParityFailure):equivalence([1],[2],base,base)

    def test_breakdown_nested_accounting_no_unsupported_zero(self):
        record=dict(cache_path_host_breakdown=[dict(component='prepared_mapping_get',parent='request_mapping_select',host_ms=3.,cuda_ms=None)],
            cache_path={'snapshot_bypassed':True},reused_tokens=12,
            setup_breakdown={'source_snapshot_validation':dict(host_ms=None,cuda_ms=None,calls=None,status='UNAVAILABLE')},
            executor_breakdown={'full_mixed_projection':dict(host_ms=20.,cuda_ms=10.,calls=6,status='MEASURED'),
                'cached_scatter':dict(host_ms=2.,cuda_ms=1.,calls=6,status='MEASURED')})
        rows=flatten(record);by={r['component']:r for r in rows}
        self.assertEqual(by['cached_scatter']['parent'],'full_mixed_projection')
        self.assertEqual(by['source_snapshot_validation']['status'],'BYPASSED_ORACLE')
        self.assertIsNone(by['source_snapshot_validation']['cuda_ms'])
        self.assertNotIn('summed_total',by);self.assertTrue(by['full_mixed_projection']['do_not_sum_nested'])


TORCH=importlib.util.find_spec('torch') is not None
@unittest.skipUnless(TORCH,'Torch absent: actual QKV Tensor/executor oracle parity NOT_RUN')
class TensorTests(unittest.TestCase):
    def test_actual_cpu_qkv_index_and_source_corruption(self):
        import torch
        from semcache.diagnostics.r1b3_occurrences import OccurrenceCatalog
        from semcache.edgelora.batched_mapping_optimized import batched_v2_projection_path
        from contextlib import ExitStack
        torch.manual_seed(4);n=12;ids=[i%7 for i in range(n)];dtype=torch.float32
        layout={0:((4,dtype),)*3};c=OccurrenceCatalog('model','weights','cpu',1000000)
        c.add_source('source',ids,0,'user_a',{0:tuple(torch.randn(1,n,4) for _ in 'qkv')},layout,prefill_complete=True)
        ix=FastOccurrenceIndex(c,layout);d=SourceBackingDescriptors(ix);cache=PreparedMappingCache(d)
        r=NS(ids=ids,index=ix,catalog=c,descriptor=d,caches={'contiguous':cache},a=NS(device='cpu',dtype='float32',capacity_mib=512),
            snapshot='s',namespace='model',weights='weights',model_weight_identity='w',execution_code_sha256='code',storage='cpu',coverage='max')
        r.binding=lambda base:{'mode':base};r.guard=ix.guard
        r.canonical_hashes=lambda:{str(i):cp.tensor_sha(t) for i,t in enumerate(c.tensors)};r.source_hashes=r.canonical_hashes()
        r.selected_hits=lambda cov:ix.select(ids,0,'user_a','model','weights',layout,cov)[0]
        class Projection(torch.nn.Linear):
            def get_base_layer(self):return self
        modules={q:Projection(4,4,bias=False).eval() for q in 'qkv'};adapter=NS(layers=[0],projection_modules=lambda _:modules)
        hidden=torch.randn(1,n,4);lease=cp.OracleLease(r,cp.ARMS[5]);plan=lease.plan;hits=lease.hits
        import semcache.models.lora_decomposition as decomposition
        import semcache.edgelora.batched_mapping_optimized as executor
        with patch.object(decomposition,'validate_projection',return_value=None):
            with batched_v2_projection_path(adapter,'user_a',hits,n,mapping=plan,contiguous=True) as a:
                first={q:module(hidden) for q,module in modules.items()}
            original=d.validate
            with patch.object(d,'validate',return_value=0.):
                with batched_v2_projection_path(adapter,'user_a',hits,n,mapping=plan,contiguous=True) as b:
                    second={q:module(hidden) for q,module in modules.items()}
        self.assertTrue(all(torch.equal(first[q],second[q]) for q in 'qkv'))
        self.assertEqual(a.records,b.records)
        lease.finish(NS(hits=hits,ids=ids,user='user_a',namespace='model',weights='weights',device='cpu',dtype=dtype,plan=plan))
        with torch.no_grad():c.tensors[0].add_(1.)
        with self.assertRaises((ValueError,cp.ParityFailure)):lease.finish(NS(hits=hits,ids=ids,user='user_a',namespace='model',weights='weights',device='cpu',dtype=dtype,plan=plan))


if __name__=='__main__':unittest.main()
