"""CPU-only scalar tests; optional real tiny OPT CPU/CUDA execution tests."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.metrics.r1 import atomic_json, digest, audit_projection_records
from semcache.metrics.r1b2 import (filter_coverage, coverage_override, correctness_status, breakdown,
    summarize, load_checkpoints, reports, overhead, comparison_audits)
from semcache.semantic.hit_selection import CacheHit


def hit(start):
    tokens=tuple(range(start,start+3))
    return CacheHit(NS(start=start,end=start+3,token_ids=tokens),NS(token_ids=tokens,positions=(start,start+3)))


def row(mode='POSITION_ALIGNED_REUSE',instrumentation='minimal',coverage='max',status='STRICT_CORRECTNESS_FAIL',wall=8):
    full=mode=='FULL_RECOMPUTE'
    return dict(row_id=mode+instrumentation,pair_id='pair'+instrumentation,comparison_identity='identity',length=32,
        storage='RAW_CPU',instrumentation=instrumentation,mode=mode,requested_coverage=None if full else coverage,
        phase='measured',repeat=0,token_ids_sha256='tokens',adapter='user_a',request_wall_ms=10 if full else wall,
        correctness=dict(status='NO_REUSE' if full else status,logit_allclose=full or status=='STRICT_CORRECTNESS_PASS'),
        actual_coverage=0 if full else .75,reused_tokens=0 if full else 24,recomputed_tokens=32 if full else 8,
        selected_windows=0 if full else 8,native_projection_rows=192 if full else 48,skipped_projection_rows=0 if full else 144,
        source_build_ms=None if full else 30,source_signature='source',selection_signature='hits',policy_signature='policy',
        source_policy_signature='sourcepolicy',logits_sha256='logits',coverage_filter_host_ms=.01,spans=[],
        breakdown=breakdown(dict(request_wall_ms=wall,prefill_gpu_ms=6,mixed_executor_gpu_ms=2,native_qkv_gpu_ms=1),mode,instrumentation))


def runner():
    spec=importlib.util.spec_from_file_location('r1b2_runner',ROOT/'scripts/72_run_r1b2_latency.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class OfflineTests(unittest.TestCase):
    def test_coverage_deterministic_nested_and_rounding(self):
        candidates=[hit(i) for i in (0,3,6,9,12,15,18,21,24,27)]
        previous=[]
        for value,count in ((0,0),(.25,2),(.5,5),(.75,8),('max',10)):
            selected,mask=filter_coverage(list(reversed(candidates)),32,value)
            self.assertEqual(len(selected),count); self.assertEqual(sum(mask),count*3)
            self.assertEqual(selected[:len(previous)],previous); previous=selected
            for h in selected: self.assertEqual(h.entry.positions,(h.window.start,h.window.end))
        self.assertEqual(sum(filter_coverage(candidates,32,'max')[1]),30)  # max != 100%

    def test_reject_invalid_candidates_and_coverage(self):
        for candidates in ([hit(0),hit(0)],[hit(1),hit(3)],[hit(31)]):
            with self.assertRaises(ValueError): filter_coverage(candidates,32,'max')
        bad=hit(0); bad.entry.positions=(3,6)
        with self.assertRaises(ValueError): filter_coverage([bad],32,.5)
        for x in (-1,1.1,'nan','inf'):
            with self.assertRaises(ValueError): filter_coverage([],32,x)

    def test_guard_is_scoped_even_on_failure(self):
        import semcache.semcache_engine as module
        original=module.select_position_aligned_diagnostic
        with patch.object(module,'select_position_aligned_diagnostic',return_value=([hit(0)],[])) as fake:
            with self.assertRaises(RuntimeError):
                with coverage_override(0) as evidence:
                    selected,mask=module.select_position_aligned_diagnostic([],32,[],'user_a')
                    self.assertEqual(selected,[]); self.assertEqual(sum(mask),0)
                    self.assertEqual(evidence['candidate_windows'],1)
                    raise RuntimeError('test')
            self.assertIs(module.select_position_aligned_diagnostic,fake)
        self.assertIs(module.select_position_aligned_diagnostic,original)

    def test_row_audit_zero_and_fractions(self):
        for used in (0,6,15,24,30):
            records=[dict(layer=layer,tensor_type=role,native_positions=list(range(used,32)),
                native_projection_rows=32-used,reused_projection_rows=used,native_call_count=1)
                for layer in range(2) for role in 'qkv']
            result=audit_projection_records(records,32,2,used,list(range(used)))
            self.assertEqual(result['skipped_projection_rows_all_layers_roles'],used*6)
            self.assertEqual(result['native_projection_rows_all_layers_roles'],(32-used)*6)
        records[0]['native_positions'][0]=0
        with self.assertRaises(AssertionError): audit_projection_records(records,32,2,30,list(range(30)))

    def test_correctness_gate_and_break_even(self):
        self.assertEqual(correctness_status(0,True,True,True,True),'NO_REUSE')
        self.assertEqual(correctness_status(3,None,True,True,True),'UNVERIFIED')
        self.assertEqual(correctness_status(3,True,False,True,True),'STRICT_CORRECTNESS_FAIL')
        full=row('FULL_RECOMPUTE')
        for status,expected in [('STRICT_CORRECTNESS_FAIL',False),('STRICT_CORRECTNESS_PASS',True),('UNVERIFIED',False)]:
            result=next(x for x in summarize([full,row(status=status)]) if x['mode']!='FULL_RECOMPUTE')
            self.assertTrue(result['observed_break_even'])
            self.assertEqual(result['correctness_qualified_break_even'],expected)
            self.assertEqual(result['source_build_mean_ms'],30)
            self.assertEqual(result['request_mean_ms'],8)  # build not added!
            self.assertEqual(result['paired_delta_mean_ms'],-2)
        result=summarize([row('FULL_RECOMPUTE','breakdown'),row(instrumentation='breakdown',status='STRICT_CORRECTNESS_PASS')])
        self.assertFalse(any(r['observed_break_even'] for r in result))

    def test_parity_failure_disqualifies_reference_pass(self):
        minimal=row(status='STRICT_CORRECTNESS_PASS')
        detailed=row(instrumentation='breakdown',status='STRICT_CORRECTNESS_PASS')
        detailed['comparison_parity']=dict(status='FAIL',reference=minimal['row_id'])
        summary=summarize([row('FULL_RECOMPUTE'),minimal,row('FULL_RECOMPUTE','breakdown'),detailed])
        self.assertFalse(any(r['correctness_qualified_break_even'] for r in summary))

    def test_missing_full_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows=[row()]
            atomic_json(Path(d)/'checkpoints/a.json',dict(identity='id',status='COMPLETE',rows=rows,rows_sha256=digest(rows)))
            with self.assertRaises(ValueError): load_checkpoints(d,'id')

    def test_pair_identity_and_duplicates(self):
        with self.assertRaises(ValueError): summarize([row(),row()])
        bad=row(); bad['adapter']='user_b'
        with self.assertRaises(ValueError): summarize([row('FULL_RECOMPUTE'),bad])

    def test_timing_nesting_and_null(self):
        result={x['metric']:x for x in breakdown(dict(prefill_gpu_ms=10,mixed_executor_gpu_ms=4,
            native_qkv_gpu_ms=2,cache_policy_ms=6,cache_materialization_ms=3),'POSITION_ALIGNED_REUSE','breakdown')}
        self.assertEqual(result['remaining_transformer_gpu_ms']['value_ms'],6)
        self.assertEqual(result['remaining_transformer_gpu_ms']['status'],'DERIVED')
        self.assertEqual(result['cache_materialization_ms']['parent'],'cache_policy_ms')
        self.assertTrue(all(not r['additive_to_parent'] for r in result.values()))
        self.assertIsNone(result['cache_fetch_gpu_ms']['value_ms'])
        full={x['metric']:x for x in breakdown(dict(cache_fetch_gpu_ms=0),'FULL_RECOMPUTE','breakdown')}
        self.assertIsNone(full['cache_fetch_gpu_ms']['value_ms'])

    def test_reports_overhead_checkpoint_integrity(self):
        rows=[row('FULL_RECOMPUTE'),row(),row('FULL_RECOMPUTE','breakdown'),row(instrumentation='breakdown',wall=9)]
        self.assertEqual(overhead(rows)[1]['breakdown_minus_minimal_ms'],1)
        self.assertTrue(all(c['status']=='EXACT_PASS' for c in comparison_audits(rows)))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            atomic_json(root/'checkpoints'/'a.json',dict(identity='id',status='COMPLETE',rows=rows,rows_sha256=digest(rows)))
            self.assertEqual(load_checkpoints(root,'id'),rows)
            with self.assertRaises(ValueError): load_checkpoints(root,'wrong')
            reports(root,rows,aggregate=True)
            for file in ('overall_summary.csv','break_even_matrix.csv','bottleneck_ranking.csv','latency_breakdown.csv',
                         'correctness.json','coverage_audit.json','instrumentation_overhead.json'):
                self.assertTrue((root/file).exists())
            p=root/'checkpoints'/'a.json'; bundle=json.loads(p.read_text()); bundle['rows'][0]['length']=64
            atomic_json(p,bundle)
            with self.assertRaises(ValueError): load_checkpoints(root,'id')

    def test_cli_failure_resume_preflight_and_collision(self):
        m=runner()
        for error,status in [(RuntimeError('missing torch'),'FAILED'),(MemoryError('screen'),'MEMORY_PREFLIGHT_FAILED'),
                             (RuntimeError('CUDA out of memory'),'OOM'),(KeyboardInterrupt(),'INTERRUPTED')]:
            with tempfile.TemporaryDirectory() as d:
                output=Path(d)/'new'
                argv=['--profile','smoke','--device','cpu','--output-dir',str(output)]
                def fail(a,root,manifest,guard):
                    if isinstance(error,MemoryError): manifest['status']='MEMORY_PREFLIGHT_FAILED'
                    raise error
                with patch.object(m,'execute',side_effect=fail):
                    self.assertNotEqual(m.main(argv),0)
                self.assertEqual(json.loads((output/'tokens_32/manifest.json').read_text())['status'],status)
                with self.assertRaises(FileExistsError): m.main(argv)
                with patch.object(m,'execute',side_effect=fail): self.assertNotEqual(m.main(argv+['--resume']),0)
                with self.assertRaises(SystemExit): m.main(argv+['--resume','--seed','43'])

    def test_aggregate_incomplete_duplicates_and_collision(self):
        spec=importlib.util.spec_from_file_location('aggregate',ROOT/'scripts/73_aggregate_r1b2.py')
        m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'input'; root.mkdir(); rows=[row('FULL_RECOMPUTE'),row()]
            atomic_json(root/'manifest.json',dict(experiment='R1-B2',status='INTERRUPTED',identity='id',rows=2))
            atomic_json(root/'checkpoints/a.json',dict(identity='id',status='COMPLETE',rows=rows,rows_sha256=digest(rows)))
            argv=['--inputs',str(root),'--output-dir',str(Path(d)/'out')]
            with self.assertRaises(ValueError): m.main(argv)
            self.assertEqual(m.main(argv+['--allow-incomplete']),0)
            with self.assertRaises(FileExistsError): m.main(argv+['--allow-incomplete'])
            with self.assertRaises(ValueError): m.main(['--inputs',str(root),str(root),'--output-dir',str(Path(d)/'other')])


AVAILABLE=all(importlib.util.find_spec(x) is not None for x in ('torch','transformers','peft'))
@unittest.skipUnless(AVAILABLE,'torch/transformers/peft unavailable; real executor tests NOT EXECUTED')
class ModelTests(unittest.TestCase):
    def exercise(self,device,dtype):
        import torch
        from test_r1_raw_semcache import TinyModelTests
        from semcache.semantic.encoder import ControlledEncoder
        from semcache.semantic.intent_clusterer import IntentClusterer
        from semcache.cache.global_cache import GlobalCache
        from semcache.semcache_engine import SemCacheEngine
        from semcache.metrics.r1_runtime import engine_request
        from semcache.metrics.r1b1 import snapshot, compare_snapshots
        fixture=TinyModelTests(); fixture.setUp()
        model=fixture.model.to(device=device,dtype=dtype); adapter=fixture.adapter
        text='2 3 4 5 6 7 8 9 10'
        tokenizer=lambda text: {'input_ids':list(map(int,text.split()))}
        references={}
        for storage in (['cpu',device] if device.startswith('cuda') else ['cpu']):
            for detailed in (False,True):
                for coverage in (0,.25,.5,.75,'max'):
                    clusterer=IntentClusterer(1); clusterer.initialize([[0.]])
                    engine=SemCacheEngine(model,tokenizer,adapter,ControlledEncoder({text:[0.]}),clusterer,GlobalCache(100000),
                        storage_device=storage,impact_execution='token_precompute',capture_impact_values=True)
                    engine_request(engine,text,'user_a','source','SEMCACHE_LOOKUP_NO_REUSE',True)
                    with coverage_override(coverage) as evidence:
                        r=engine_request(engine,text,'user_a','target','SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC',detailed)
                    n=r['summary']['reused_unique_token_count']
                    target=evidence['candidate_windows'] if coverage=='max' else min(evidence['candidate_windows'],int(9*coverage//3))
                    self.assertEqual(n,target*3)
                    audit=audit_projection_records(list(r['projection_audit'].values()),9,2,n)
                    self.assertEqual(audit['skipped_projection_rows_all_layers_roles'],n*6)
                    state=snapshot(engine,r)
                    if coverage in references:
                        previous=references[coverage]
                        self.assertTrue(compare_snapshots(previous[0],state)['passed'])
                        self.assertTrue(torch.allclose(previous[1],r['logits'],atol=1e-3,rtol=1e-3))
                    else: references[coverage]=(state,r['logits'])

    def test_tiny_cpu_real_execution(self):
        import torch
        self.exercise('cpu',torch.float32)

    def test_tiny_cuda_real_execution(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable; GPU execution NOT EXECUTED')
        self.exercise('cuda:0',torch.float16)


if __name__=='__main__': unittest.main()
