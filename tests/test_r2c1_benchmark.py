import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch
from semcache.diagnostics import r2c1_benchmark as bench
from semcache.diagnostics.r2c1_workloads import MODES
ROOT=Path(__file__).resolve().parents[1]


def runner():
    spec=importlib.util.spec_from_file_location('c1_test_runner',ROOT/'scripts/90_run_r2c1_long_context_qkv.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Contracts(unittest.TestCase):
    def test_modes_reuse_frozen_runtime_and_timing_disclosure(self):
        from semcache.diagnostics.r2a_runtime import CostRuntime
        self.assertTrue(issubclass(bench.Runtime,CostRuntime))
        self.assertIn('deepcopy',bench.BOUNDARIES['excluded'])
        self.assertIn('ORACLE_DIAGNOSTIC',bench.BOUNDARIES['raw_classification'])
        self.assertIn('not pooled',bench.BOUNDARIES['prefix_reference'])

    def test_canonical_content_or_namespace_change_rejected(self):
        r=bench.Runtime.__new__(bench.Runtime)
        r.ids=[1,2,3]; r.snapshot='s'; r.namespace='n'; r.weights='w'
        r.expected_binding=(tuple(r.ids),'s','n','w'); r.source_hashes={'q':'original'}
        r.model_guard=lambda *a:None; r.index=NS(guard=lambda:None)
        r.canonical_hashes=lambda:{'q':'changed'}
        with self.assertRaises(Exception): r.guard(True)
        r.namespace='other'
        with self.assertRaises(Exception): r.guard()

    def test_output_policy_and_all_reuse_no_fresh_lora(self):
        runtime=NS(ids=list(range(6)),length=6,adapter=NS(layers=[1]),model=NS(config=NS(vocab_size=64)))
        observer=bench.OutputAudit(runtime)
        observer.roots=[dict(ids=runtime.ids,use_cache=False,output_attentions=True)]
        observer.heads=[dict(input=[1,6,8],output=[1,6,64])]
        self.assertTrue(observer.report(6)['output_matched_within_c1'])
        observer.heads[0]['output']=[1,1,64]
        with self.assertRaises(Exception): observer.report(6)

    def test_three_way_pairing_sign_and_no_instrumentation(self):
        calls=[]
        def invoke(mode):
            calls.append(mode)
            return 0,dict(request_ms=dict(zip(MODES,(20.,30.,25.)))[mode],wrapper_invoke_wall_ms=40.,
                coverage_audit={'reused_tokens':30},memory={})
        r=NS(model=NS(modules=lambda:[]),length=32,guard=lambda *a:None,invoke=invoke)
        with patch.object(bench,'strict_metrics',return_value={'allclose':False}):
            rows=bench.paired(r,NS(phase='latency',warmup_runs=3,measured_runs=21,order_seed=4),{'case_id':'test'},lambda **kw:None)
        self.assertEqual(len(calls),72)
        measured=[r for r in rows if r['phase']=='measured']
        for mode in MODES: self.assertEqual(sum(r['execution_order'][0]==mode for r in measured),7)
        self.assertTrue(all(r['delta_full_raw_ms']==-5 and r['delta_miss_raw_ms']==5 and not r['strict_pass'] for r in rows))

    def test_profiler_unavailable_and_actual_v2_ranges_cleanup(self):
        from semcache.diagnostics.r2c1_profiler import require_profiler,ProfilerUnavailable,extra_ranges
        from contextlib import nullcontext
        from semcache.cache.global_cache import GlobalCache
        original=GlobalCache.insert; labels=[]
        with patch.dict('sys.modules',{'torch':NS()}):
            with self.assertRaises(ProfilerUnavailable): require_profiler('cuda:0')
        with patch.dict('sys.modules',{'torch':NS(profiler=NS(record_function=lambda name:labels.append(name) or nullcontext()))}),patch.object(GlobalCache,'insert',return_value='sentinel'):
            with extra_ranges(): self.assertEqual(GlobalCache.insert(None,'entry'),'sentinel')
        self.assertIs(GlobalCache.insert,original)
        self.assertIn('r2a::cache_policy_insert_inclusive',labels)

    def test_runner_partial_capacity_and_strict_fail_provenance(self):
        from test_r2c1_analysis import session
        m=runner()
        class Runtime:
            def __init__(self,a):
                self.a=a; self.provenance={'gpu':{'gpu_uuid':None}}; self.meta={'revision':'FAKE'}
                self.storage='cpu'; self.weights='w'; self.namespace='n'
                p=NS(element_size=lambda:4,numel=lambda:1)
                self.model=NS(parameters=lambda:iter([p]),config=NS(max_position_embeddings=2048,num_hidden_layers=2,num_attention_heads=2,hidden_size=8,vocab_size=64))
                self.encoder=NS(metadata={'fixture':'CONTROL_ONLY'},model=NS(parameters=lambda:iter([p])),max_length=512,tokenizer=lambda text,**kw:{'input_ids':list(range(600))})
            def setup_identity(self): return 'hash'
            def prepare(self,n):
                if n==1024: raise MemoryError('test capacity')
                self.length=n; self.ids=list(range(n)); self.snapshot='snapshot'; self.source_hashes={'0':'hash'}
                self.catalog=NS(resident_bytes=128); self.costs={'source_prefill_ms':999}; self.preflight={}; self.text='fixture'
            def guard(self,*a): pass
            def model_guard(self,*a): pass
        def validation(r):
            return dict(audits={'RAW_SEMCACHE_QKV':{'coverage':{'reused_tokens':510}}},quality={'RAW':{'allclose':False}})
        def pairs(r,a,identity,save):
            _,rows=session(a.independent_run_id,count=a.measured_runs)
            rows=[dict(row,**identity,benchmark_phase=a.phase) for row in rows]
            save(rows=rows); return rows
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'Runtime',Runtime),patch.object(m,'backend_audit',return_value={'effective':'eager'}),patch.object(m,'memory',return_value={'gpu':{},'host':{}}),patch.object(m,'monitor',return_value={'status':'GPU_MONITOR_UNVERIFIED'}),patch.object(m,'validate',validation),patch.object(m,'paired',pairs):
            out=Path(tmp)/'pilot'
            self.assertEqual(m.main(['--phase','pilot','--prompt-lengths','512,1024','--coverages','max','--device','cpu','--cache-device','cpu','--output-dir',str(out)]),0)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['execution_status'],'COMPLETE_WITH_FAILURES')
            self.assertEqual(manifest['conditions']['L1024_covmax']['status'],'MEMORY_PREFLIGHT_FAILED')
            self.assertEqual(manifest['GPU_VALIDATION'],'NOT_RUN'); self.assertFalse(manifest['correctness_qualified_speedup'])
            self.assertEqual(len(json.loads((out/'latency_raw_pairs.json').read_text())),3)
            audit=json.loads((out/'workload_audit.json').read_text())['L512_covmax']
            self.assertTrue(audit['semantic_truncated']); self.assertFalse(audit['model_truncated'])
            with self.assertRaises(FileExistsError): m.main(['--output-dir',str(out)])

    def test_aggregate_gpu_status_comes_from_inputs(self):
        from test_r2c1_analysis import session
        m=runner()
        with tempfile.TemporaryDirectory() as tmp:
            inputs=[]
            for i in range(3):
                path=Path(tmp)/str(i); path.mkdir(); manifest,rows=session(str(i))
                (path/'manifest.json').write_text(json.dumps(manifest)); (path/'latency_raw_pairs.json').write_text(json.dumps(rows)); inputs.append(str(path))
            out=Path(tmp)/'aggregate'
            self.assertEqual(m.main(['--phase','analyze','--inputs',*inputs,'--output-dir',str(out)]),0)
            saved=json.loads((out/'manifest.json').read_text())
            self.assertEqual(saved['GPU_VALIDATION'],'AGGREGATED_GPU_RUNS')
            self.assertFalse(saved['analysis_process']['gpu_inference_performed'])
            self.assertEqual(saved['input_run_validation']['completed_gpu_sessions'],3)
            self.assertEqual(saved['aggregated_experimental_config']['dtype'],['float16'])

    def test_cli_requires_pilot_for_latency_and_long_extension(self):
        m=runner()
        for args in [['--phase','latency'],['--phase','pilot','--prompt-lengths','1536'],['--phase','pilot','--prompt-lengths','4096']]:
            with self.assertRaises(SystemExit): m.parse_args(args+['--output-dir','unused'])


DEPS=all(importlib.util.find_spec(n) is not None for n in ('torch','transformers','peft'))
@unittest.skipUnless(DEPS,'Torch/Transformers/PEFT unavailable: native model/LoRA/output audit NOT_RUN')
class ModelTest(unittest.TestCase):
    def test_native_opt_head_lora_audit_cpu(self):
        import torch
        from test_r2b1_prefix_runtime import OfflineOPTTests
        r=OfflineOPTTests().runtime('cpu'); r.ids=list(range(32)); r.length=32
        observer=bench.OutputAudit(r)
        with observer.hooks(),torch.inference_mode():
            r.model(input_ids=torch.tensor([r.ids]),use_cache=False,output_attentions=True)
        self.assertEqual(observer.report(0)['head']['output'],[1,32,64])


if __name__=='__main__': unittest.main()
