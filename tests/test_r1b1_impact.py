"""Stdlib contracts plus optional synthetic attention / tiny OPT integration."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.cache.impact_execution import EXECUTIONS, MODES, compute_window_values
from semcache.cache.attention_impact import PaperRowL2SumReducer
from semcache.cache.global_cache import GlobalCache
from semcache.semcache_engine import SemCacheEngine
from semcache.metrics.r1b1 import R1B1Extension, compare_snapshots, pair_checks, summarize, snapshot
from semcache.diagnostics.impact import ATOL, RTOL, parity, measure_impact
from semcache.semantic.subsequence import Subsequence, SubsequenceExtractor


class Contracts(unittest.TestCase):
    def test_engine_plumbing_history_chu_pbr_without_tensor_execution(self):
        from test_m85_alignment import PrefillPolicyTest
        previous=None
        for execution in EXECUTIONS:
            fixture=PrefillPolicyTest()
            fixture.setUp()
            fixture.engine.impact_execution=execution
            fixture.engine.capture_impact_values=True
            fixture.engine.pbr_interval_queries=1
            with patch('semcache.cache.impact_execution.compute_window_values',return_value=[2.]) as compute, \
                 patch.object(fixture.engine.metrics.history,'append',wraps=fixture.engine.metrics.history.append) as append:
                result=fixture.query('SEMCACHE_PHYSICAL_REUSE')
                self.assertEqual(append.call_count,1)
                self.assertEqual(append.call_args.args[:3],(0,'test',1))
                self.assertEqual(compute.call_count,0 if execution=='legacy_blockwise' else 1)
                if execution!='legacy_blockwise':
                    self.assertEqual(compute.call_args.args[0],('literal_stub',))
            actual=snapshot(fixture.engine,result)
            self.assertTrue(actual['pbr_updates'])
            self.assertEqual(actual['windows'][0]['value'],2.)
            self.assertEqual(sum(e['event_type']=='CHU' for e in actual['events']),1)
            if previous is not None:
                self.assertTrue(compare_snapshots(previous,actual)['passed'])
            previous=actual

    def test_defaults_and_reducer_execution_are_separate(self):
        engine = SemCacheEngine(None,None,None,None,None,GlobalCache(10))
        self.assertEqual(engine.impact_execution,'legacy_blockwise')
        self.assertFalse(engine.capture_impact_values)
        for execution in EXECUTIONS[1:]:
            optimized = SemCacheEngine(None,None,None,None,None,GlobalCache(10),impact_execution=execution)
            self.assertIs(type(optimized.impact_reducer),PaperRowL2SumReducer)
            with self.assertRaises(ValueError):
                SemCacheEngine(None,None,None,None,None,GlobalCache(10),impact_execution=execution,
                               impact_reducer='reproduction_frobenius_mean')
        class Custom(PaperRowL2SumReducer):
            pass
        with self.assertRaises(ValueError):
            SemCacheEngine(None,None,None,None,None,GlobalCache(10),impact_execution='token_precompute',impact_reducer=Custom())

    def test_shared_math_has_no_history_or_diagnostic_identity(self):
        source=(ROOT/'src/semcache/cache/impact_execution.py').read_text()
        self.assertNotIn('impact_audit',source)
        self.assertNotIn('history.append',source)
        tree=ast.parse((ROOT/'src/semcache/semcache_engine.py').read_text())
        query=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='query')
        appends=[n for n in ast.walk(query) if isinstance(n,ast.Call) and ast.unparse(n.func)=='self.metrics.history.append']
        self.assertEqual(len(appends),1)
        self.assertEqual([ast.unparse(a) for a in appends[0].args],['cluster','query_id','self.cache.now','per_query'])

    def test_policy_boundary_never_hidden_by_numeric_tolerance(self):
        reference=dict(admission=dict(score=.3,decision=False),entries=[1],history=[dict(query_id='source',order=1)])
        candidate=copy.deepcopy(reference)
        candidate['admission']['score'] += 1e-13
        self.assertTrue(compare_snapshots(reference,candidate)['passed'])
        candidate['admission']['decision']=True
        result=compare_snapshots(reference,candidate)
        self.assertFalse(result['passed'])
        self.assertEqual(result['differences'][0]['kind'],'exact_decision')
        candidate=copy.deepcopy(reference)
        candidate['history'][0]['query_id']='impact_audit'
        self.assertFalse(compare_snapshots(reference,candidate)['passed'])
        candidate=copy.deepcopy(reference)
        candidate['entries']=[]
        self.assertFalse(compare_snapshots(reference,candidate)['passed'])

    def rows(self):
        rows=[]
        for algorithm in EXECUTIONS[:2]:
            for mode in ('FULL_RECOMPUTE','POSITION_ALIGNED_DIAGNOSTIC'):
                row=dict(pair_id=algorithm,phase='measured',repeat=0,storage='RAW_CPU',instrumentation='minimal',
                    requested_prompt_tokens=32,condition='same_user_exact',mode=mode,impact_execution=algorithm,
                    timing=dict(request_wall_ms=10. if mode=='FULL_RECOMPUTE' else 8.,attention_impact_ms=2.),
                    correctness=dict(allclose=True,nonzero_reuse_validated=mode!='FULL_RECOMPUTE'),
                    source_policy=None,target_policy=None,token_ids_sha256='x',adapter='a',reuse_provenance=[],
                    reused_tokens=3 if mode!='FULL_RECOMPUTE' else 0,coverage=.1,
                    projection_audit=dict(skipped_projection_rows_all_layers_roles=6),
                    resident_bytes_before=1,resident_bytes_after=1,cache_event_counts={'EVICT':0})
                rows.append(row)
        return rows

    def test_reports_pending_failure_retention_and_summary(self):
        rows=self.rows()
        self.assertTrue(all(c['passed'] for c in pair_checks(rows)))
        self.assertIsNone(pair_checks(rows[2:])[0]['passed'])
        summary=summarize(rows,pair_checks(rows))
        selected=next(r for r in summary if r['impact_execution']=='token_precompute' and r['mode']!='FULL_RECOMPUTE')
        self.assertEqual(selected['observed_full_request_ratio'],1.25)
        self.assertEqual(selected['request_wall_ms_std'],0.)
        rows[-1]['reused_tokens']=0
        self.assertFalse(pair_checks(rows)[-1]['passed'])
        extension=R1B1Extension()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            extension.write_reports(root,rows)
            self.assertEqual(len((root/'raw.jsonl').read_text().splitlines()),4)
            self.assertIn('POLICY_PARITY_FAILED',(root/'policy_parity.json').read_text())
            manifest={'status':'COMPLETE'}
            extension.finalize(SimpleNamespace(output_dir=root,impact_executions=','.join(EXECUTIONS[:2])),manifest,rows)
            self.assertEqual(manifest['status'],'POLICY_PARITY_FAILED')
            self.assertFalse(manifest['policy_equivalent_optimization'])

    def test_cli_isolation_failure_manifest_and_resume(self):
        script=str(ROOT/'scripts/70_run_r1b1_impact.py')
        result=subprocess.run([sys.executable,script,'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--impact-executions',result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            command=[sys.executable,script,'--output-dir',str(out),'--smoke','--warmup-runs','0','--measured-runs','1']
            bad=subprocess.run(command+['--impact-executions','token_precompute'],capture_output=True,text=True)
            self.assertEqual(bad.returncode,2)
            self.assertFalse(out.exists())
            if importlib.util.find_spec('torch') is None:
                failed=subprocess.run(command,capture_output=True,text=True)
                self.assertEqual(failed.returncode,1)
                manifest=json.loads((out/'manifest.json').read_text())
                self.assertEqual(manifest['experiment'],'R1-B1')
                self.assertIn('NOT EXECUTED',manifest['failure'])
                previous=(out/'manifest.json').read_bytes()
                subprocess.run(command,capture_output=True)
                self.assertEqual((out/'manifest.json').read_bytes(),previous)
                resumed=subprocess.run(command+['--resume'],capture_output=True,text=True)
                self.assertIn('NOT EXECUTED',resumed.stderr)


HAS_TORCH=importlib.util.find_spec('torch') is not None
HAS_MODELS=all(importlib.util.find_spec(n) is not None for n in ('torch','transformers','peft'))


@unittest.skipUnless(HAS_TORCH,'NOT EXECUTED: torch unavailable')
class AttentionTests(unittest.TestCase):
    def check_device(self,device):
        import torch
        generator=torch.Generator().manual_seed(123)
        for n in (4,9,17,64):
            for dtype in (torch.float16,torch.float32,torch.float64):
                tensors=[torch.rand(1,3,n,n,generator=generator,dtype=dtype).to(device) for _ in range(3)]
                windows=SubsequenceExtractor(3).extract([i%4 for i in range(n)])[::-1]
                mask=torch.rand(n,generator=generator)>.3
                reference=[PaperRowL2SumReducer().reduce(tensors,w.start,w.end,mask) for w in windows]
                for mode in MODES:
                    actual=compute_window_values(tensors,windows,mode,mask)
                    self.assertTrue(parity(reference,actual)['impact_values_within_tolerance'])
                    # Existing diagnostic uses the very same helper; finalize stays diagnostic-only.
                    diagnostic=measure_impact(tensors,windows,mode,mask,instrument=False)
                    self.assertEqual(actual,diagnostic['impact_values'])

    def test_randomized_order_masks_duplicates_cpu(self):
        self.check_device('cpu')

    def test_randomized_cuda(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('NOT EXECUTED: CUDA unavailable')
        self.check_device('cuda:0')

    def test_nonfinite_and_mask_interpretation(self):
        import torch
        a=torch.ones(1,2,7,7)
        w=[Subsequence((1,2,3),0,3)]
        a[:,:,0,6]=float('nan')  # future key
        a[:,:,6,:]=float('inf')  # unqueried row
        mask=torch.tensor([[1,0,1,1,1,1,1]])
        a[:,:,1,:]=float('nan')  # padding query
        for mode in MODES:
            expected=PaperRowL2SumReducer().reduce([a],0,3,mask)
            self.assertTrue(parity([expected],compute_window_values([a],w,mode,mask))['impact_values_within_tolerance'])
        a[:,:,0,0]=float('nan')
        for mode in MODES:
            with self.assertRaises(ValueError):
                compute_window_values([a],w,mode,mask)
        self.assertEqual(compute_window_values([a],[],'TOKEN_PRECOMPUTE'),[])


@unittest.skipUnless(HAS_MODELS,'NOT EXECUTED: torch/transformers/peft unavailable')
class EngineTests(unittest.TestCase):
    def setUp(self):
        from test_r1_raw_semcache import TinyModelTests
        TinyModelTests.setUp(self)

    def check_engine(self,device,storage):
        import torch
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.semantic.encoder import ControlledEncoder
        from semcache.semantic.intent_clusterer import IntentClusterer
        from semcache.metrics.r1_runtime import engine_request,native_request
        from semcache.metrics.r1 import ENGINE_MODES
        self.model.to(device)
        self.adapter=OPTModelAdapter(self.model)
        text='2 3 4 2 3 4 8 9'
        tokenizer=lambda x:{'input_ids':list(map(int,x.split()))}
        for capacity in (100000,2304):
            reference_stages=None
            for execution in EXECUTIONS:
                clusterer=IntentClusterer(1)
                clusterer.initialize([[0.]])
                engine=SemCacheEngine(self.model,tokenizer,self.adapter,ControlledEncoder({text:[0.]}),
                    clusterer,GlobalCache(capacity),storage_device=storage,impact_execution=execution,
                    capture_impact_values=True,pbr_interval_queries=1,history_lambda=2)
                stages=[]
                with patch.object(engine.metrics.history,'append',wraps=engine.metrics.history.append) as append:
                    for index,name in enumerate(('source','exact','repeat')):
                        mode='LOOKUP_NO_REUSE' if index==0 else 'POSITION_ALIGNED_DIAGNOSTIC'
                        full=native_request(self.model,tokenizer,self.adapter,text,'user_a',False)
                        result=engine_request(engine,text,'user_a',name,ENGINE_MODES[mode],False)
                        torch.testing.assert_close(full['logits'],result['logits'],atol=1e-6,rtol=1e-5)
                        stages.append(snapshot(engine,result))
                        if capacity==100000 and index:
                            self.assertGreater(result['summary']['reused_unique_token_count'],0)
                    self.assertEqual(append.call_count,3)
                    self.assertEqual([(c.args[1],c.args[2]) for c in append.call_args_list],[('source',1),('exact',2),('repeat',3)])
                self.assertEqual([q['query_id'] for q in stages[-1]['history'][0]['observations']],['exact','repeat'])
                self.assertTrue(any(s['pbr_updates'] for s in stages))
                if capacity==2304:
                    self.assertTrue(any(e['event_type']=='EVICT' for s in stages for e in s['events']))
                if reference_stages is None:
                    reference_stages=stages
                else:
                    comparison=compare_snapshots(reference_stages,stages)
                    self.assertTrue(comparison['passed'],comparison['differences'])

    def test_history_pbr_eviction_strict_logits_and_rows_cpu(self):
        self.check_engine('cpu','cpu')

    def test_cuda_cpu_and_gpu_residency(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('NOT EXECUTED: CUDA unavailable')
        for storage in ('cpu','cuda:0'):
            self.check_engine('cuda:0',storage)


if __name__=='__main__':
    unittest.main()
