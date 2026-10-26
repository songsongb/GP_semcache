"""Offline provenance tests and explicitly optional tensor/model execution tests."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r1b3_occurrences import (occurrence_loss_audit,occurrence_reasons,
    OccurrenceCatalog,physical_bytes,capture_completed_source,catalog_override)
from semcache.semantic.subsequence import SubsequenceExtractor
from semcache.metrics.r1 import atomic_json,digest,audit_projection_records


def fixture(n):
    ids=[i%54 for i in range(n)]; seen=set(); entries=[]; admission=[]; dedup=[]
    for w in SubsequenceExtractor(3).extract(ids):
        k=(0,w.token_ids)
        if k in seen:
            dedup.append(dict(cache_key=k,source_start=w.start,source_end=w.end))
        else:
            seen.add(k)
            entries.append(dict(key=k,positions=(w.start,w.end),qkv_resident=True,metadata={},layers=[0]))
            admission.append(dict(cache_key=k,source_start=w.start,source_end=w.end,admission_decision=True,rejection_reason=None))
    return dict(source_ids=ids,cluster=0,query_id='synthetic_source',adapter='a',namespace='m',weight_identity='w',
        entries=entries,admission=admission,deduplication=dedup,evictions=[])


def runner():
    spec=importlib.util.spec_from_file_location('r1b3',ROOT/'scripts/74_diagnose_r1b3_occurrences.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class OfflineTests(unittest.TestCase):
    def test_repeated_128_256_fixture_explains_54_not_historical_claim(self):
        for n in (32,64,128,256):
            evidence=fixture(n); audit=occurrence_loss_audit(evidence,evidence['source_ids'],'a','m','w')
            self.assertEqual(audit['total_extracted_windows'],n-2)
            self.assertEqual(audit['predicted_legacy_max_reused_tokens'],30 if n==32 else 54)
            self.assertFalse(audit['historical_54_token_cause_verified'])
            if n>=64:
                row=audit['target_windows'][54]
                self.assertTrue(row['aligned_source_occurrence_exists'])
                self.assertTrue(row['prefix_token_ids_equal'])
                self.assertFalse(row['resident_aligned_qkv'])
                self.assertIn('DUPLICATE_KEY_OCCURRENCE_LOST',row['causes'])
                self.assertIn('POSITION_MISMATCH',row['causes'])

    def test_multiple_causes_capacity_admission_and_absence(self):
        ev=fixture(128); k=ev['entries'][0]['key']; ev['entries']=ev['entries'][1:]
        ev['admission'][0].update(admission_decision=False,rejection_reason='admission_score_at_or_below_threshold')
        ev['evictions']=[dict(cache_key=k)]
        audit=occurrence_loss_audit(ev,ev['source_ids'],'a','m','w')
        causes=audit['target_windows'][54]['causes']
        self.assertTrue({'CAPACITY_OR_EVICTION','ADMISSION_REJECTED','DUPLICATE_KEY_OCCURRENCE_LOST'}<=set(causes))
        self.assertEqual(audit['target_windows'][54]['primary_cause'],'CAPACITY_OR_EVICTION')
        ids=list(ev['source_ids']); ids[90]=999
        audit=occurrence_loss_audit(ev,ids,'wrong','wrong','w')
        self.assertIn('ADAPTER_OR_MODEL_MISMATCH',audit['target_windows'][91]['causes'])
        self.assertIn('PREFIX_CONTEXT_MISMATCH',audit['target_windows'][91]['causes'])
        self.assertIn('NO_MATCHING_SOURCE_OCCURRENCE',audit['target_windows'][90]['causes'])

    def test_prefix_and_identity_predicates(self):
        ids=[1,2,3,1,2,3,9]
        window=SubsequenceExtractor(3).extract(ids)[3]
        entry=NS(positions=(3,6),token_ids=(1,2,3),qkv_metadata=dict(source_query_token_ids=ids,
            source_adapter='a',model_namespace='m',weight_identity='w'))
        self.assertEqual(occurrence_reasons(entry,window,ids,'a','m','w'),([],True))
        different_suffix=ids[:-1]+[8]
        self.assertEqual(occurrence_reasons(entry,window,different_suffix,'a','m','w'),([],True))
        different_prefix=[8]+ids[1:]
        self.assertIn('PREFIX_CONTEXT_MISMATCH',occurrence_reasons(entry,window,different_prefix,'a','m','w')[0])
        entry.positions=(0,3)
        self.assertIn('POSITION_MISMATCH',occurrence_reasons(entry,window,ids,'a','m','w')[0])
        for user,namespace,weight in [('b','m','w'),('a','other','w'),('a','m','other')]:
            self.assertIn('ADAPTER_OR_MODEL_MISMATCH',occurrence_reasons(entry,window,ids,user,namespace,weight)[0])

    def test_catalog_patch_restores_on_failure(self):
        import semcache.semcache_engine as module
        original=module.select_position_aligned_diagnostic
        with self.assertRaises(RuntimeError):
            with catalog_override(None,None,'m','w',{},0): raise RuntimeError('test')
        self.assertIs(module.select_position_aligned_diagnostic,original)

    def test_checkpoint_checksum_collision_resume_failure_states(self):
        m=runner()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            argv=['--profile','smoke','--device','cpu','--output-dir',str(root)]
            with patch.object(m,'execute',side_effect=RuntimeError('CUDA out of memory')):
                self.assertEqual(m.main(argv),1)
            self.assertEqual(json.loads((root/'manifest.json').read_text())['status'],'OOM')
            with self.assertRaises(FileExistsError): m.main(argv)
            with patch.object(m,'execute',side_effect=KeyboardInterrupt()): self.assertEqual(m.main(argv+['--resume']),1)
            self.assertEqual(json.loads((root/'manifest.json').read_text())['status'],'INTERRUPTED')
            with self.assertRaises(ValueError): m.main(argv+['--resume','--seed','123'])
            b=dict(identity='x',status='COMPLETE',rows=[dict(row_id='a')],audits={})
            b.update(checksum=digest(b['rows']),audit_checksum=digest({}))
            atomic_json(root/'checkpoints/a.json',b)
            self.assertEqual(len(m.load_groups(root,'x')),1)
            atomic_json(root/'checkpoints/b.json',b)
            with self.assertRaises(ValueError): m.load_groups(root,'x')

    def test_no_existing_runtime_or_jobs_edits_required(self):
        import ast
        tree=ast.parse((ROOT/'scripts/74_diagnose_r1b3_occurrences.py').read_text())
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_source' for n in ast.walk(tree)))
        m=runner(); args=m.args(['--profile','smoke','--device','cpu','--output-dir','unused'])
        self.assertFalse(args.enable_catalog); self.assertEqual(args.instrumentation,'minimal')


TORCH=importlib.util.find_spec('torch') is not None
@unittest.skipUnless(TORCH,'Torch unavailable: tensor selection/residency NOT EXECUTED')
class TensorTests(unittest.TestCase):
    def build(self,n=128,storage='cpu',source_device='cpu'):
        import torch
        ids=fixture(n)['source_ids']; layout={0:((2,torch.float32),)*3}
        values={0:tuple(torch.arange(n*2,device=source_device,dtype=torch.float32).reshape(1,n,2)+10000*r for r in range(3))}
        catalog=OccurrenceCatalog('m','w',storage,10000000)
        catalog.add_source('source',ids,0,'a',values,layout,prefill_complete=True)
        return catalog,ids,values,layout

    def test_128_256_recovery_distinct_occurrence_values(self):
        import torch
        for n in (32,64,128,256):
            c,ids,values,layout=self.build(n)
            for coverage in (0,.25,.5,.75,'max'):
                chosen,mask,rows=c.select(ids,0,'a','m','w',layout,coverage)
                expected=n//3*3 if coverage=='max' else int(n*coverage//3)*3
                self.assertEqual(sum(mask),expected)
                self.assertEqual(len({i for h in chosen for i in range(h.window.start,h.window.end)}),expected)
                for h in chosen:
                    self.assertEqual(h.entry.positions,(h.window.start,h.window.end))
                    for role,t in enumerate(h.entry.tensors[0]): self.assertTrue(torch.equal(t,values[0][role][:,h.window.start:h.window.end]))
            self.assertEqual(len(c.entries),n-2)
            self.assertEqual(c.resident_bytes,n*2*3*4)
            self.assertEqual(physical_bytes([t for e in c.entries.values() for q in e.tensors.values() for t in q]),c.resident_bytes)
            if n>=64:
                a=c.index[(0,(0,1,2),0,3)][0]; b=c.index[(0,(0,1,2),54,57)][0]
                self.assertFalse(torch.equal(c.entries[a].tensors[0][0],c.entries[b].tensors[0][0]))

    def test_reject_mismatch_missing_corrupt_tensor_and_capacity(self):
        c,ids,values,layout=self.build()
        for adapter,namespace,weights in [('b','m','w'),('a','x','w'),('a','m','x')]:
            self.assertEqual(c.select(ids,0,adapter,namespace,weights,layout,'max')[0],[])
        other=list(ids); other[0]=999
        self.assertEqual(c.select(other,0,'a','m','w',layout,'max')[0],[])
        entry=next(iter(c.entries.values())); entry.tensors[0]=(values[0][0][:,3:6],*entry.tensors[0][1:])
        with self.assertRaises(ValueError): c.select(ids,0,'a','m','w',layout,'max')
        empty=OccurrenceCatalog('m','w','cpu',1)
        with self.assertRaises(MemoryError): empty.add_source('s',ids,0,'a',values,layout,prefill_complete=True)
        with self.assertRaises(ValueError): empty.add_source('s',ids,0,'a',values,layout,prefill_complete=False)

    def test_multiple_source_queries_are_distinct(self):
        c,ids,values,layout=self.build(64)
        shifted={l:tuple(t+500 for t in q) for l,q in values.items()}
        c.add_source('second',ids,0,'a',shifted,layout,prefill_complete=True)
        self.assertEqual(len(c.entries),124)
        self.assertEqual(len(c.index[(0,(0,1,2),0,3)]),2)
        self.assertEqual(c.resident_bytes,64*2*3*4*2)

    def test_gpu_storage_parity(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable; GPU storage NOT EXECUTED')
        a,ids,_,layout=self.build(storage='cpu',source_device='cuda:0')
        b,_,_,_=self.build(storage='cuda:0',source_device='cuda:0')
        ah=a.select(ids,0,'a','m','w',layout,'max')[0]; bh=b.select(ids,0,'a','m','w',layout,'max')[0]
        self.assertEqual([h.entry.positions for h in ah],[h.entry.positions for h in bh])
        for x,y in zip(ah,bh):
            for p,q in zip(x.entry.tensors[0],y.entry.tensors[0]): self.assertTrue(torch.equal(p,q.cpu()))


MODELS=all(importlib.util.find_spec(x) is not None for x in ('torch','transformers','peft'))
@unittest.skipUnless(MODELS,'Torch/Transformers/PEFT unavailable: actual executor NOT EXECUTED')
class ModelTests(unittest.TestCase):
    def exercise(self,device,dtype):
        import torch
        from test_r1_raw_semcache import TinyModelTests
        from semcache.semantic.encoder import ControlledEncoder
        from semcache.semantic.intent_clusterer import IntentClusterer
        from semcache.cache.global_cache import GlobalCache
        from semcache.semcache_engine import SemCacheEngine
        from semcache.metrics.r1_runtime import engine_request
        from semcache.metrics.r1b2 import coverage_override
        from semcache.metrics.r1b1 import snapshot,compare_snapshots
        f=TinyModelTests(); f.setUp(); f.model.to(device=device,dtype=dtype); ids=[2,3,4]*3; text=' '.join(map(str,ids))
        layout={i:((16,dtype),)*3 for i in range(2)}
        legacy=[]
        for capture_enabled in (False,True):
            cluster=IntentClusterer(1); cluster.initialize([[0.]])
            e=SemCacheEngine(f.model,lambda _:dict(input_ids=ids),f.adapter,ControlledEncoder({text:[0.]}),cluster,GlobalCache(100000),
                impact_execution='token_precompute',capture_impact_values=True)
            from contextlib import nullcontext
            with capture_completed_source() if capture_enabled else nullcontext({}) as projections:
                engine_request(e,text,'user_a','source','SEMCACHE_LOOKUP_NO_REUSE',True)
            with coverage_override('max'):
                out=engine_request(e,text,'user_a','target','SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC',False)
            legacy.append((snapshot(e,out),out['logits']))
            if capture_enabled:
                oracle_results={}
                for storage in (['cpu',device] if device.startswith('cuda') else ['cpu']):
                    c=OccurrenceCatalog('m','w',storage,100000)
                    c.add_source('source',ids,0,'user_a',projections,layout,prefill_complete=True)
                    for coverage in (0,.5,'max'):
                        with catalog_override(c,e,'m','w',layout,coverage):
                            result=engine_request(e,text,'user_a','target2','SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC',False)
                        n=result['summary']['reused_unique_token_count']
                        self.assertEqual(n,9 if coverage=='max' else int(9*coverage//3)*3)
                        audit_projection_records(list(result['projection_audit'].values()),9,2,n)
                        spans=[(p['source_start'],p['destination_start']) for p in result['summary']['reuse_block_provenance']]
                        if coverage in oracle_results:
                            previous=oracle_results[coverage]
                            self.assertEqual(previous[0],spans)
                            self.assertTrue(torch.allclose(previous[1],result['logits'],atol=1e-3,rtol=1e-3))
                        else: oracle_results[coverage]=(spans,result['logits'])
        self.assertTrue(compare_snapshots(legacy[0][0],legacy[1][0])['passed'])
        self.assertTrue(torch.equal(legacy[0][1],legacy[1][1]))


    def test_legacy_b2_parity_and_actual_catalog_skipping(self):
        import torch
        self.exercise('cpu',torch.float32)

    def test_cuda_actual_skipping(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable; actual GPU execution NOT EXECUTED')
        self.exercise('cuda:0',torch.float16)


if __name__=='__main__': unittest.main()
