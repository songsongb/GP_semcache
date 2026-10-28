"""Scalar Engine-policy fixtures and dependency-gated real reducer parity tests."""
from dataclasses import replace
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.cache.metric_manager import CacheMetricManager
from semcache.semantic.subsequence import SubsequenceExtractor
from semcache.metrics.r1 import digest
from semcache.diagnostics.r1b3_break_even import MissingEvidence,ParityFailure
from semcache.diagnostics.r2c3_impact_replay import (ARMS,BASES,ImpactReference,ImpactSession,canonical,policy_state)
from semcache.diagnostics.r2c3_benchmark import check_equivalence,paired


def engine():
    cache=GlobalCache(1024)
    return NS(cache=cache,metrics=CacheMetricManager(cache),clusterer=NS(queries=0,counts=[1],centroids=[[0.]],pending=[]),
        query_id='source',_admission_candidate_order=0,_admission_order=0,impact_execution='token_precompute',capture_impact_values=True)


class Fixture:
    def __init__(self,mode=BASES[0]):
        self.mode=mode; self.ids=[3,4,5,3,4,5]; self.windows=SubsequenceExtractor(3).extract(self.ids)
        self.runtime=NS(catalog=NS(entries={}))
        self.catalog_entry=CacheEntry(0,tuple(self.ids[:3]),(0,3),4,impact=10.)
        self.runtime.catalog.entries['occurrence']=self.catalog_entry
        self.attentions=(NS(shape=(1,1,6,6),dtype='float32',device='cpu'),)
        self.mask=NS(is_cuda=False,tolist=lambda:[True]*6)
        self.actual_calls=0
    def binding(self):
        return dict(token_ids=list(self.ids),base_mode=self.mode,source_snapshot='source',coverage='max',
            model_weight_identity='w',adapter='user_a',implementation_sha256='code')
    def compute(self,attentions,windows,mode,valid_attention_mask=None,**kwargs):
        self.actual_calls+=1
        return [float(i+(1 if self.mode==BASES[0] else 2)) for i,_ in enumerate(windows)]
    def request(self,e,*args,**kwargs):
        from semcache.cache.impact_execution import compute_window_values
        e.query_id='target'; e.clusterer.queries+=1
        keys=[(0,w.token_ids) for w in self.windows]; e.metrics.arrive(keys)
        values=compute_window_values(self.attentions,self.windows,'TOKEN_PRECOMPUTE',self.mask)
        impacts={}
        for key,value in zip(keys,values): impacts.setdefault(key,[]).append(value)
        per_query={k:sum(v)/len(v) for k,v in impacts.items()}
        e.metrics.history.append(0,'target',e.cache.now,per_query)
        events=[]
        if self.mode==BASES[1]:
            e.metrics.reused(self.catalog_entry,values[0])
            events.append(dict(event_type='CHU',new_I=self.catalog_entry.impact,frequency=self.catalog_entry.frequency))
        admission=[]
        for w,value,key in zip(self.windows,values,keys):
            if key in e.cache.entries: continue
            candidate=CacheEntry(0,w.token_ids,(w.start,w.end),4,impact=value)
            allowed=e.cache.admission.admit(e.cache.admission_metrics(candidate,e.metrics.frequencies[key],e.metrics.frequencies))
            admission.append(dict(key=key,allowed=allowed))
            events.append(dict(event_type='ADMISSION_SCORE',key=key,decision=allowed))
            e.cache.insert(candidate,e.metrics.frequencies[key],e.metrics.frequencies)
        summary=dict(token_ids=self.ids,cluster_id=0,impact_windows=[dict(start=w.start,end=w.end,
            token_ids=list(w.token_ids),value=v) for w,v in zip(self.windows,values)],
            admission_audit=admission,deduplication_audit=[],chu_update_count=int(self.mode==BASES[1]),
            pbr_updates_this_query=0,admitted_block_count=sum(a['allowed'] for a in admission),
            rejected_block_count=sum(not a['allowed'] for a in admission),impact_wall_ms=2.)
        return dict(summary=summary,events=events)
    def execute(self,reference=None):
        import semcache.metrics.r1_runtime as rt
        import semcache.cache.impact_execution as impact
        self.catalog_entry.frequency=0; self.catalog_entry.impact=10.
        self.catalog_entry.last_access=self.catalog_entry.updated_at=0
        with patch.object(rt,'engine_request',self.request),patch.object(impact,'compute_window_values',self.compute):
            session=ImpactSession(self.runtime,self.binding(),reference,reference.sha256 if reference else None)
            with session.scoped(): result=rt.engine_request(engine())
        return session,result


class ReplayTests(unittest.TestCase):
    def test_exact_values_policy_history_admission_and_catalog_chu(self):
        for mode in BASES:
            with self.subTest(mode=mode):
                f=Fixture(mode); on,result=f.execute(); reference=ImpactReference.capture(f.binding(),on)
                replay,rerun=f.execute(reference)
                self.assertEqual(f.actual_calls,1)
                self.assertEqual(on.policy,replay.policy)
                self.assertEqual(on.values,list(replay.values))
                self.assertEqual(result['summary']['admission_audit'],rerun['summary']['admission_audit'])
                self.assertEqual(replay.real_reduction_calls,0); self.assertEqual(replay.dispatch_calls,1)
                self.assertTrue(on.policy['history_and_catalog_included'])
                self.assertEqual(reference.export(f.binding())['candidate_windows'][0]['value_hex'],float(on.values[0]).hex())

    def test_modes_have_distinct_references_never_share_unrelated_values(self):
        a=Fixture(BASES[0]); b=Fixture(BASES[1])
        sa,_=a.execute(); sb,_=b.execute()
        ra=ImpactReference.capture(a.binding(),sa); rb=ImpactReference.capture(b.binding(),sb)
        self.assertNotEqual(ra.values,rb.values); self.assertNotEqual(ra.sha256,rb.sha256)
        with self.assertRaises(MissingEvidence): ImpactSession(b.runtime,b.binding(),ra,ra.sha256)

    def test_reference_provenance_and_value_tampering_rejected(self):
        f=Fixture(); session,_=f.execute(); ref=ImpactReference.capture(f.binding(),session)
        for binding in (dict(f.binding(),adapter='user_b'),dict(f.binding(),source_snapshot='other'),
                        dict(f.binding(),token_ids=[9]*6),dict(f.binding(),model_weight_identity='changed'),
                        dict(f.binding(),implementation_sha256='new')):
            with self.assertRaises(MissingEvidence): ref.verify(binding,ref.sha256)
        with self.assertRaises(MissingEvidence): replace(ref,values=(0.,)*4).verify(f.binding(),ref.sha256)
        with self.assertRaises(MissingEvidence): ref.verify(f.binding(),None)
        with self.assertRaises(MissingEvidence): ImpactReference.capture(f.binding(),NS(real_reduction_calls=0,dispatch_calls=1,fallback_calls=0))

    def test_window_order_length_and_mask_mismatch(self):
        for mutation in ('order','length','tokens','mask','attention'):
            with self.subTest(mutation=mutation):
                f=Fixture(); session,_=f.execute(); ref=ImpactReference.capture(f.binding(),session)
                if mutation=='order': f.windows.reverse()
                if mutation=='length': f.windows.pop()
                if mutation=='tokens': f.windows[0]=replace(f.windows[0],token_ids=(9,9,9))
                if mutation=='mask': f.mask=NS(is_cuda=False,tolist=lambda:[False]*6)
                if mutation=='attention': f.attentions=(NS(shape=(1,2,6,6),dtype='float32',device='cpu'),)
                with self.assertRaises(MissingEvidence): f.execute(ref)

    def test_policy_discrepancy_is_fatal(self):
        f=Fixture(); s,_=f.execute(); ref=ImpactReference.capture(f.binding(),s)
        original=f.request
        def altered(e,*a,**kw):
            result=original(e,*a,**kw); e.metrics.history.history[0][-1]['impacts'].clear(); return result
        f.request=altered
        with self.assertRaises(ParityFailure): f.execute(ref)

    def test_patches_restored_after_forward_failure_and_fallback(self):
        import semcache.metrics.r1_runtime as rt
        import semcache.cache.impact_execution as impact
        import semcache.semcache_engine as em
        from semcache.cache.attention_impact import PaperRowL2SumReducer
        originals=(rt.engine_request,impact.compute_window_values,impact._validate,em.actual_attention_impact,PaperRowL2SumReducer.reduce)
        f=Fixture(); on,_=f.execute(); ref=ImpactReference.capture(f.binding(),on)
        for kind in ('forward','fallback','validate'):
            with self.assertRaises((RuntimeError,ParityFailure,MissingEvidence)):
                with ImpactSession(f.runtime,f.binding(),ref,ref.sha256).scoped():
                    if kind=='fallback': em.actual_attention_impact(None,0,3)
                    elif kind=='validate': impact._validate(None,6)
                    else: raise RuntimeError('injected forward failure')
            self.assertEqual(originals,(rt.engine_request,impact.compute_window_values,impact._validate,em.actual_attention_impact,PaperRowL2SumReducer.reduce))

    def test_snapshot_keeps_logical_query_clocks(self):
        f=Fixture(); e=engine(); before=policy_state(e,f.runtime.catalog)
        e.cache.now+=1
        self.assertNotEqual(before,policy_state(e,f.runtime.catalog))
        self.assertNotEqual(canonical(0.),canonical(-0.))

    def test_output_mismatch_cannot_be_accepted(self):
        record=dict(coverage_audit={},projection_audit={},selection_hash='s',selected_provenance=[],output_shape=[1,6,2],
            policy={},impact_values_digest='i',reference_digest='r',impact_real_reduction_calls=1,impact_dispatch_calls=1,impact_fallback_calls=0)
        replay=dict(record,impact_real_reduction_calls=0)
        with patch.dict('sys.modules',{'torch':NS(equal=lambda a,b:a==b)}),patch('semcache.diagnostics.r2c3_benchmark.strict_metrics',return_value={'allclose':True}):
            self.assertTrue(check_equivalence(1,1,record,replay)['exact_logits'])
            with self.assertRaises(ParityFailure): check_equivalence(1,2,record,replay)
            with self.assertRaises(ParityFailure): check_equivalence(1,1,record,dict(replay,impact_real_reduction_calls=1))


TORCH=importlib.util.find_spec('torch') is not None
@unittest.skipUnless(TORCH,'Torch unavailable: real tensor reduction/Replay test NOT_RUN')
class RealTensorTests(unittest.TestCase):
    def test_actual_reducer_cpu_replay_values(self):
        import torch
        f=Fixture(); f.attentions=(torch.softmax(torch.randn(1,1,6,6),dim=-1),); f.mask=torch.ones(6,dtype=torch.bool)
        # Keep the policy fixture but execute the real existing reducer.
        from semcache.cache.impact_execution import compute_window_values
        f.compute=compute_window_values
        on,_=f.execute(); ref=ImpactReference.capture(f.binding(),on)
        replay,_=f.execute(ref)
        self.assertEqual(tuple(on.values),tuple(replay.values)); self.assertEqual(replay.real_reduction_calls,0)


MODELS=all(importlib.util.find_spec(n) is not None for n in ('torch','transformers','peft'))
@unittest.skipUnless(MODELS,'Torch/Transformers/PEFT unavailable: real tiny OPT/PEFT Engine parity NOT_RUN')
class RealEngineTests(unittest.TestCase):
    def test_actual_engine_cpu_on_replay_logit_policy_history(self):
        import torch
        from test_r1_raw_semcache import TinyModelTests
        from semcache.semantic.encoder import ControlledEncoder
        from semcache.semantic.intent_clusterer import IntentClusterer
        from semcache.semcache_engine import SemCacheEngine
        import semcache.metrics.r1_runtime as rt
        f=TinyModelTests(); f.setUp()
        text='2 3 4 5 6 7 8 9'; ids=list(map(int,text.split()))
        tokenizer=lambda text:{'input_ids':list(map(int,text.split()))}
        cluster=IntentClusterer(1,update_mode='buffered');cluster.initialize([[0.]])
        source=SemCacheEngine(f.model,tokenizer,f.adapter,ControlledEncoder({text:[0.]}),cluster,
            GlobalCache(100000),impact_execution='token_precompute',capture_impact_values=True)
        rt.engine_request(source,text,'user_a','source','SEMCACHE_LOOKUP_NO_REUSE',False)
        runtime=NS(catalog=NS(entries={}))
        def clone():
            memo={id(v):v for v in (source.model,source.tokenizer,source.adapter,source.encoder)}
            for entry in source.cache.entries.values():
                for qkv in entry.tensors.values():
                    for tensor in qkv: memo[id(tensor)]=tensor
            return copy.deepcopy(source,memo)
        for base,mode in zip(BASES,('SEMCACHE_LOOKUP_NO_REUSE','SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC')):
            binding=dict(base_mode=base,token_ids=ids,adapter='user_a',model='SYNTHETIC_TINY_OPT_NO_HUB')
            on=ImpactSession(runtime,binding)
            with on.scoped(): a=rt.engine_request(clone(),text,'user_a','target',mode,False)
            reference=ImpactReference.capture(binding,on)
            replay=ImpactSession(runtime,binding,reference,reference.sha256)
            with replay.scoped(): b=rt.engine_request(clone(),text,'user_a','target',mode,False)
            self.assertTrue(torch.equal(a['logits'],b['logits']))
            self.assertEqual(a['summary']['impact_windows'],b['summary']['impact_windows'])
            self.assertEqual(on.policy,replay.policy);self.assertEqual(replay.real_reduction_calls,0)


if __name__=='__main__': unittest.main()
