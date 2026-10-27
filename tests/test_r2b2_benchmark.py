"""Runner/control tests are synthetic; GPU model tests explicitly dependency-gated."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch
from semcache.diagnostics import r2b2_benchmark as bench
from semcache.diagnostics.r2b2_profiler import monitor,require_profiler,ProfilerUnavailable
ROOT=Path(__file__).resolve().parents[1]


def runner():
    spec=importlib.util.spec_from_file_location('b2_runner_test',ROOT/'scripts/87_run_r2b2_paired_latency.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class BenchmarkTests(unittest.TestCase):
    def test_b1_normal_timer_body_unchanged(self):
        # Protect exact adapter/eval/input/materialize/native/sync sequencing.
        old=(ROOT/'src/semcache/diagnostics/r2b1_prefix_runtime.py').read_text()
        new=(ROOT/'src/semcache/diagnostics/r2b2_benchmark.py').read_text()
        def timed(text):
            tree=ast.parse(text)
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
            fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
            return ast.dump(next(n for n in fn.body if isinstance(n,ast.With)),include_attributes=False)
        self.assertEqual(timed(old),timed(new))

    def test_pair_identity_order_and_no_prefix(self):
        calls=[]; saves=[]
        runtime=NS(p=16,n=32,device='cpu',model=NS(modules=lambda:[]),adapter=NS(layers=[1,2]),
            check=lambda:None,ids=list(range(32)))
        def invoke(r,mode):
            calls.append((tuple(r.ids),r.p,mode))
            result=dict(logits=0,cache=None,wall_ms={'FULL_USE_CACHE':10.,'WARM_NORMAL':8.,'WARM_PREPARED_DIAGNOSTIC':7.}[mode],
                cache_prepare_host_ms=1 if mode=='WARM_NORMAL' else None,request_clone_bytes=100 if mode=='WARM_NORMAL' and r.p else 0,
                logit_shape=[1,r.n if mode=='FULL_USE_CACHE' else r.n-r.p,16],query_tokens=r.n if mode=='FULL_USE_CACHE' else r.n-r.p,
                output_kv_length=r.n,memory={})
            return result,{'classification':'IDEAL_DIAGNOSTIC'} if mode=='WARM_PREPARED_DIAGNOSTIC' else None
        identity=dict(independent_run_id='one',case_id='c',attention_arm='on',endpoint_policy='b1_all_logits',structural_strict_pass=False)
        with patch.object(bench,'invoke',invoke),patch.object(bench,'backend_audit',return_value={}),patch.object(bench,'parity',return_value={}),patch.object(bench,'quality',return_value={'strict':True}):
            rows=bench.run_pairs(runtime,NS(warmup_runs=3,measured_runs=6,order_seed=5),identity,lambda **kw:saves.append(kw))
            self.assertEqual(len(rows),9); self.assertEqual(len(calls),27)
            self.assertTrue(all(c[0]==tuple(range(32)) and c[1]==16 for c in calls))
            measured=[r for r in rows if r['phase']=='measured']
            for mode in bench.MODES:
                self.assertEqual(sum(r['execution_order'][0]==mode for r in measured),2)
            for r in rows:
                self.assertEqual(r['correctness_status'],'STRICT_FAIL')
                self.assertEqual(r['prepared_classification'],'IDEAL_DIAGNOSTIC_ONLY')
                self.assertEqual(r['normal_clone_bytes'],100); self.assertEqual(r['full_minus_normal_ms'],2)
            runtime.p=0
            rows=bench.run_pairs(runtime,NS(warmup_runs=3,measured_runs=6,order_seed=5),identity,lambda **kw:None)
            self.assertTrue(all(r['prepared_wall_ms'] is None and not r['cache_hit'] for r in rows))

    def test_prepared_mutation_rejected_using_frozen_guard(self):
        from test_r2b15_cost import CostContracts
        from semcache.diagnostics.r2b15_cost import PreparedTicket
        from semcache.diagnostics.r2b1_prefix_cache import PrefixError
        fixture=CostContracts()
        try:
            runtime=fixture.fake_runtime(); ticket=PreparedTicket(runtime)
            ticket.cache[0][0].data[0]=123
            with self.assertRaises(PrefixError): ticket.check()
            self.assertEqual(ticket.state,'INVALIDATED')
        finally: fixture.doCleanups()

    def test_request_change_cannot_enter_paired_samples(self):
        from semcache.diagnostics.r2b1_prefix_cache import PrefixError
        r=NS(ids=[1,2],n=2,p=1,model=NS(modules=lambda:[]),check=lambda:None)
        def changed(rt,mode): rt.ids[-1]=99; return {},None
        with patch.object(bench,'invoke',changed),patch.object(bench,'backend_audit',return_value={}):
            with self.assertRaisesRegex(PrefixError,'REJECT_PAIRED_IDENTITY'):
                bench.run_pairs(r,NS(warmup_runs=3,measured_runs=20,order_seed=5),{},lambda **kw:None)

    def test_unavailable_profiler_and_monitor_no_affinity_guess(self):
        with patch.dict('sys.modules',{'torch':NS()}):
            with self.assertRaises(ProfilerUnavailable): require_profiler('cuda:0')
            self.assertEqual(monitor('cuda:0')['status'],'GPU_MONITOR_UNVERIFIED')
        cuda=NS(get_device_properties=lambda d:NS(uuid='test-uuid',name='fake GPU'),memory_allocated=lambda d:10,memory_reserved=lambda d:20)
        with patch.dict('sys.modules',{'torch':NS(cuda=cuda)}),patch('subprocess.run',side_effect=[NS(stdout='GPU-test-uuid, 2, fake, driver, 1, 2, 3, 4, 5'),NS(stdout='')]) as run:
            state=monitor('cuda:0')
            self.assertEqual(state['status'],'GPU_MONITOR_VERIFIED')
            self.assertFalse(any('--id' in x for x in run.call_args_list[0].args[0]))
            self.assertEqual(state['values']['physical_index'],'2')

    def test_profiler_exclusive_attribution_does_not_add_parent_inclusive(self):
        from semcache.diagnostics.r2b2_profiler import cpu_costs
        parent=NS(name='r2b2::native_forward',cpu_parent=None)
        child=NS(name='r2b2::lm_head',cpu_parent=parent)
        event=NS(name='aten::mm',device_type='CPU',cpu_parent=child,input_shapes=[[1,16]],
            self_cpu_time_total=500.,cpu_time_total=9000.,self_device_time_total=100.,
            self_cpu_memory_usage=0,self_device_memory_usage=0)
        rows=cpu_costs([event,event])
        self.assertEqual(len(rows),1); self.assertEqual(rows[0]['component'],'lm_head')
        self.assertEqual(rows[0]['cpu_self_ms'],1.)
        self.assertEqual(rows[0]['device_self_attributed_ms'],.2)
        self.assertEqual(rows[0]['calls'],2)

    def test_cli_failure_preserves_traceback_and_collision(self):
        m=runner()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run'
            with patch.object(m,'run',side_effect=RuntimeError('fixture failure')):
                self.assertEqual(m.main(['--output-dir',str(path)]),1)
            manifest=json.loads((path/'manifest.json').read_text())
            self.assertEqual(manifest['execution_status'],'FAILED'); self.assertIn('fixture failure',manifest['traceback'])
            self.assertFalse(manifest['CORRECTNESS_QUALIFIED_SPEEDUP'])
            with self.assertRaises(FileExistsError): m.main(['--output-dir',str(path)])
        self.assertEqual(m.parse_args(['--phase','latency','--output-dir','unused']).measured_runs,20)
        with self.assertRaises(SystemExit): m.parse_args(['--phase','latency','--measured-runs','10','--output-dir','unused'])

    def test_complete_runner_and_artifacts_strict_fail_stays_fail(self):
        from test_r2b1_prefix_cache import identity as namespace
        m=runner()
        class Runtime:
            def __init__(self,a):
                self.device='cpu'; self.a=a; self.provenance={'fixture':'MOCK_CONTROL_FLOW_ONLY'}
                self.meta={'resolved_model_revision':'fake'}
            def configure(self,arm,endpoint):
                self.arm=arm; self.endpoint={'effective':'b1_all_logits'}; self.backend={'effective':'eager'}
            def prepare(self,n,p):
                self.n=n; self.p=p; self.ids=list(range(n)); self.snapshot=NS(identity_hash='source') if p else None
                self.source_report={'prefix_prefill_ms':1234}; self.preflight={}
            def identity(self): return namespace()
            def check(self): return {'status':'UNCHANGED_FAKE'}
            def verify_weight_hashes(self): return {'status':'UNCHANGED_FAKE'}
        gate=dict(quality={'strict':False},full={'rows':[]},normal={'rows':[]},cache_hit=True,
            canonical={'status':'UNCHANGED_FAKE'},output={},native_endpoint_parity='NOT_APPLICABLE')
        def pairs(runtime,a,identity,save):
            from test_r2b2_analysis import session
            _,rows=session(a.independent_run_id,count=a.measured_runs,strict=False)
            rows=[dict(r,**identity) for r in rows]
            save(rows=rows)
            return rows
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'B2Runtime',Runtime),patch.object(m,'structural_gate',return_value=gate),patch.object(m,'run_pairs',pairs),patch.object(m,'monitor',return_value={'status':'GPU_MONITOR_UNVERIFIED'}):
            path=Path(tmp)/'run'
            self.assertEqual(m.main(['--cases','32:16','--output-dir',str(path)]),0)
            manifest=json.loads((path/'manifest.json').read_text())
            self.assertEqual(manifest['execution_status'],'COMPLETE')
            self.assertEqual(manifest['correctness_status'],'STRICT_CORRECTNESS_FAIL')
            self.assertFalse(manifest['GPU_SMOKE_VALIDATED'])
            self.assertFalse(manifest['CORRECTNESS_QUALIFIED_SPEEDUP'])
            self.assertTrue((path/'latency_raw_pairs.csv').exists())
            self.assertFalse((path/'profiling_cpu_summary.csv').exists())
            rows=json.loads((path/'latency_raw_pairs.json').read_text())
            self.assertEqual({r['attention_arm'] for r in rows},{'on','off'})
            self.assertTrue(all(r['full_wall_ms']==10 for r in rows))


DEPS=all(importlib.util.find_spec(n) is not None for n in ('torch','transformers','peft'))
@unittest.skipUnless(DEPS,'Torch/Transformers/PEFT unavailable: model/API/attention/parity NOT EXECUTED')
class ModelTests(unittest.TestCase):
    def exercise(self,device):
        from test_r2b1_prefix_runtime import OfflineOPTTests
        original=OfflineOPTTests().runtime(device)
        runtime=bench.B2Runtime.__new__(bench.B2Runtime); runtime.__dict__.update(original.__dict__)
        runtime.configure('on','b1'); runtime.prepare(32,16)
        # Same model, source snapshot and token IDs against the frozen B1 body.
        from semcache.diagnostics.r2b1_prefix_runtime import PrefixRuntime
        import torch
        for path in ('FULL_USE_CACHE','EXACT_PREFIX_WARM'):
            baseline=PrefixRuntime.forward(runtime,path); current=runtime.forward(path)
            self.assertTrue(torch.equal(baseline['logits'],current['logits']))
            self.assertTrue(bench.exact_cache(baseline['cache'],current['cache']))
        for arm in ('on','off'):
            runtime.configure(arm,'auto')
            for p in (0,16,31):
                runtime.prepare(32,p)
                gate=bench.structural_gate(runtime)
                self.assertTrue(gate['STRUCTURAL_REUSE_VALID'])
                self.assertEqual(gate['normal']['query_tokens'],32-p)
                self.assertEqual(gate['full']['query_tokens'],32)
                self.assertEqual(gate['output']['full_attention']['returned'],arm=='on')
                if p: self.assertTrue(gate['NORMAL_PREPARED_PARITY']['final_kv_exact'])
        runtime.verify_weight_hashes()
    def test_cpu_random_opt(self): self.exercise('cpu')
    def test_cuda_random_opt(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        self.exercise('cuda:0')


if __name__=='__main__': unittest.main()
