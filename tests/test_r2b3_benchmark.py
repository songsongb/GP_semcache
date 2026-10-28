import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch
from semcache.diagnostics import r2b3_benchmark as b3
from semcache.diagnostics import r2b2_benchmark as b2
ROOT=Path(__file__).resolve().parents[1]


def runner():
    spec=importlib.util.spec_from_file_location('b3_runner_test',ROOT/'scripts/88_run_r2b3_length_sweep.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Contracts(unittest.TestCase):
    def test_reference_relative_strict_rule_and_scoped_callback(self):
        # Opposite torch.isclose argument direction would accept this value.
        self.assertFalse(b3.strict_scalar([1.],[1.002001])['allclose'])
        self.assertTrue(b3.strict_scalar([1.],[1.0019])['allclose'])
        original=b2.quality
        try:
            with b3.numeric_contract():
                self.assertIs(b2.quality,b3.quality)
                raise RuntimeError('exit')
        except RuntimeError: pass
        self.assertIs(b2.quality,original)
        self.assertIs(b3.Runtime,b2.B2Runtime)

    def test_canonical_mutation_rejected_and_prepared_single_use(self):
        from test_r2b15_cost import CostContracts
        from semcache.diagnostics.r2b15_cost import PreparedTicket
        from semcache.diagnostics.r2b1_prefix_cache import PrefixError
        fixture=CostContracts()
        try:
            r=fixture.fake_runtime(); ticket=PreparedTicket(r)
            with ticket.use(): r.forward('EXACT_PREFIX_WARM')
            with self.assertRaises(PrefixError): ticket.check()
            r.snapshot.pairs[0][0].data[0]=99
            with self.assertRaises(PrefixError): r.check()
        finally: fixture.doCleanups()

    def test_normal_prepared_logits_and_final_kv_parity(self):
        from test_r2b1_prefix_cache import FakeTensor
        from semcache.diagnostics.r2b1_prefix_cache import PrefixError
        a=FakeTensor(); b=a.clone()
        normal=dict(logits=FakeTensor(),cache=((a,a),),query_tokens=1)
        prepared=dict(logits=normal['logits'].clone(),cache=((b,b),),query_tokens=1)
        with patch.dict('sys.modules',{'torch':NS(equal=lambda x,y:x.data==y.data)}):
            self.assertTrue(b2.parity(normal,prepared)['final_kv_exact'])
            b.data[0]=99
            with self.assertRaises(PrefixError): b2.parity(normal,prepared)
            b.data[0]=a.data[0]; prepared['logits'].data[0]=99
            with self.assertRaises(PrefixError): b2.parity(normal,prepared)

    def test_cli_context_pilot_requirement_and_collision_failure(self):
        m=runner()
        for argv in [['--phase','latency'],['--phase','profile','--pilot-inputs','pilot'],['--cases','2048:2047'],
            ['--phase','analyze','--inputs','a','--model','facebook/opt-125m']]:
            with self.assertRaises(SystemExit): m.parse_args(argv+['--output-dir','unused'])
        a=m.parse_args(['--phase','latency','--pilot-inputs','pilot','--output-dir','unused'])
        self.assertEqual(a.measured_runs,20); self.assertEqual(a.attention_arm,'off')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'failed'
            with patch.object(m,'run',side_effect=RuntimeError('injected failure')):
                self.assertEqual(m.main(['--output-dir',str(path)]),1)
            manifest=json.loads((path/'manifest.json').read_text())
            self.assertEqual(manifest['execution_status'],'FAILED')
            self.assertIn('injected failure',manifest['failure']['traceback'])
            with self.assertRaises(FileExistsError): m.main(['--output-dir',str(path)])

    def test_pilot_partial_failure_preserves_valid_pairs(self):
        from test_r2b1_prefix_cache import identity as namespace
        m=runner()
        class Runtime:
            def __init__(self,a):
                self.provenance={}; self.device='cpu'; self.a=a; self.model=NS(config=NS(max_position_embeddings=2048))
            def identity(self): return namespace()
            def configure(self,arm,endpoint):
                self.arm=arm; self.backend={'effective':'eager'}
                self.endpoint={'effective':'b1_all_logits'}
            def prepare(self,n,p):
                if n==1024: raise MemoryError('pilot capacity')
                self.n=n; self.p=p; self.ids=list(range(n)); self.snapshot=NS(identity_hash='source')
                self.source_report={'cache_bytes':256,'prefix_prefill_ms':123,'snapshot_ms':2}; self.preflight={}
            def check(self): return {'status':'UNCHANGED_FAKE'}
            def verify_weight_hashes(self): return {'status':'UNCHANGED_FAKE'}
        g=dict(quality={'strict':False},FULL_RELATIVE_STRICT_CORRECTNESS='STRICT_FAIL',
            full={'rows':[]},normal={'rows':[]},cache_hit=True,output={})
        def pairs(rt,a,identity,save):
            from test_r2b3_analysis import session
            _,rows=session(a.independent_run_id,count=a.measured_runs)
            rows=[dict(r,**identity,memory={}) for r in rows]
            save(context={'case':'L128_P64_off','phase':'measured'},rows=rows)
            return rows
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'Runtime',Runtime),patch.object(m,'memory',return_value={'host':{},'gpu':{}}),patch.object(m,'monitor',return_value={'status':'GPU_MONITOR_UNVERIFIED'}),patch.object(m,'gate',return_value=g),patch.object(m,'run_pairs',pairs),patch.object(m,'prepare_only',return_value={'wall_ms':2.}):
            path=Path(tmp)/'pilot'
            self.assertEqual(m.main(['--phase','pilot','--device','cpu','--cases','128:64,1024:960','--output-dir',str(path)]),0)
            manifest=json.loads((path/'manifest.json').read_text())
            self.assertEqual(manifest['execution_status'],'COMPLETE_WITH_FAILURES')
            self.assertEqual(manifest['conditions']['L128_P64_off']['status'],'CAPACITY_VALIDATED')
            self.assertEqual(manifest['conditions']['L1024_P960_off']['status'],'MEMORY_PREFLIGHT_FAILED')
            self.assertIn('pilot capacity',manifest['conditions']['L1024_P960_off']['traceback'])
            saved=json.loads((path/'latency_raw_pairs.json').read_text())
            self.assertEqual(len(saved),3)
            self.assertTrue(all(r['correctness_status']=='STRICT_FAIL' and not r['output_matched'] for r in saved))
            self.assertFalse(manifest['correctness_qualified_speedup'])

    def test_analyze_manifest_derived_from_benchmark_inputs(self):
        from test_r2b3_analysis import session
        m=runner()
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for i in range(3):
                p=Path(tmp)/str(i); p.mkdir(); manifest,rows=session(str(i))
                (p/'manifest.json').write_text(json.dumps(manifest)); (p/'latency_raw_pairs.json').write_text(json.dumps(rows)); paths.append(str(p))
            out=Path(tmp)/'analysis'
            self.assertEqual(m.main(['--phase','analyze','--inputs',*paths,'--output-dir',str(out)]),0)
            result=json.loads((out/'manifest.json').read_text())
            self.assertNotIn('benchmark_config',result)
            self.assertEqual(result['aggregated_experimental_config']['dtype'],['float16'])
            self.assertEqual(result['aggregated_experimental_config']['model'],['facebook/opt-2.7b'])
            self.assertTrue((out/'break_even_summary.csv').exists())


DEPS=all(importlib.util.find_spec(n) is not None for n in ('torch','transformers','peft'))
@unittest.skipUnless(DEPS,'Torch/Transformers/PEFT unavailable: real model/Cache/FP16/GPU NOT_RUN')
class ModelTests(unittest.TestCase):
    def test_cpu_native_pairing_and_strict_metrics(self):
        import torch
        from test_r2b1_prefix_runtime import OfflineOPTTests
        old=OfflineOPTTests().runtime('cpu')
        rt=b3.Runtime.__new__(b3.Runtime); rt.__dict__.update(old.__dict__); rt.configure('off','b1')
        for p in (0,16,31):
            rt.prepare(32,p); g=b3.gate(rt)
            self.assertTrue(g['STRUCTURAL_REUSE_VALID'])
            self.assertEqual(g['normal']['query_tokens'],32-p)
        q=b3.strict_metrics(torch.tensor([[[1.]]]),torch.tensor([[[1.002001]]]))
        self.assertFalse(q['allclose'])

    def test_cuda_native_gate(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable: GPU NOT_RUN')
        from test_r2b1_prefix_runtime import OfflineOPTTests
        old=OfflineOPTTests().runtime('cuda:0')
        rt=b3.Runtime.__new__(b3.Runtime); rt.__dict__.update(old.__dict__); rt.configure('off','b1')
        rt.prepare(32,31); self.assertTrue(b3.gate(rt)['STRUCTURAL_REUSE_VALID'])


if __name__=='__main__': unittest.main()
