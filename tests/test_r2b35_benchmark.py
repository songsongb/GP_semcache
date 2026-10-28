import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch
from semcache.diagnostics import r2b35_benchmark as bench
from semcache.diagnostics.r2b1_prefix_cache import PrefixError
ROOT=Path(__file__).resolve().parents[1]


def runner():
    spec=importlib.util.spec_from_file_location('b35_runner_test',ROOT/'scripts/89_run_r2b35_output_matched.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class BenchmarkTests(unittest.TestCase):
    def test_same_session_four_paths_order_and_strict_fail(self):
        calls=[]
        r=NS(ids=list(range(32)),n=32,p=31,arm='off',snapshot=NS(identity_hash='source'),model=NS(modules=lambda:[]),check=lambda:None)
        def execute(mode):
            calls.append((mode,tuple(r.ids),r.p,r.snapshot.identity_hash))
            return dict(wall_ms=dict(zip(bench.MODES,[20.,10.,12.,9.]))[mode],cache_prepare_host_ms=1.,request_clone_bytes=256,
                logit_shape=[1,1,64],memory={})
        r.execute=execute
        q=dict(axis_a={'FULL':{'allclose':True},'WARM_NORMAL':{'allclose':True}},matched={'allclose':False},legacy_suffix={'allclose':False})
        with patch.object(bench,'backend_audit'),patch.object(bench,'pair_quality',return_value=q):
            rows=bench.run_pairs(r,NS(warmup_runs=3,measured_runs=20,order_seed=3,phase='latency'),
                dict(case_id='L32_P31',gate_full_warm_strict=False,gate_native_equivalent=True),lambda **kw:None)
        self.assertEqual(len(calls),23*4)
        self.assertTrue(all(c[1:]==(tuple(range(32)),31,'source') for c in calls))
        measured=[v for v in rows if v['phase']=='measured']
        for mode in bench.MODES: self.assertEqual(sum(v['execution_order'][0]==mode for v in measured),5)
        self.assertTrue(all(not v['full_warm_strict'] and not v['correctness_qualified_speedup'] for v in measured))

    def test_cache_mutation_and_single_use_frozen_contract(self):
        from test_r2b15_cost import CostContracts
        from semcache.diagnostics.r2b15_cost import PreparedTicket
        fixture=CostContracts()
        try:
            r=fixture.fake_runtime(); a=PreparedTicket(r); b=PreparedTicket(r)
            self.assertIsNot(a.cache,b.cache)
            r.snapshot.pairs[0][0].data[0]=99
            with self.assertRaises(PrefixError): r.check()
        finally: fixture.doCleanups()

    def test_runner_failure_collision_cli_and_approval_required(self):
        m=runner()
        for phase in ('pilot','latency','profile'):
            with self.assertRaises(SystemExit): m.parse_args(['--phase',phase,'--output-dir','unused'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fail'
            with patch.object(m,'run',side_effect=PrefixError('OUTPUT_MATCHED_UNAVAILABLE','unknown casting')):
                self.assertEqual(m.main(['--output-dir',str(path)]),1)
            saved=json.loads((path/'manifest.json').read_text())
            self.assertEqual(saved['output_matching'],'OUTPUT_MATCHED_UNAVAILABLE'); self.assertEqual(saved['execution_status'],'FAILED')
            self.assertIn('unknown casting',saved['failure']['traceback']); self.assertEqual(saved['GPU_VALIDATION'],'NOT_RUN')
            with self.assertRaises(FileExistsError): m.main(['--output-dir',str(path)])

    def test_analysis_manifest_from_inputs_without_model_defaults(self):
        from test_r2b35_analysis import session
        m=runner()
        with tempfile.TemporaryDirectory() as tmp:
            inputs=[]
            for i in range(3):
                directory=Path(tmp)/str(i); directory.mkdir(); manifest,rows=session(str(i))
                (directory/'manifest.json').write_text(json.dumps(manifest)); (directory/'latency_raw_pairs.json').write_text(json.dumps(rows)); inputs.append(str(directory))
            out=Path(tmp)/'aggregate'
            self.assertEqual(m.main(['--phase','analyze','--inputs',*inputs,'--output-dir',str(out)]),0)
            result=json.loads((out/'manifest.json').read_text())
            self.assertNotIn('benchmark_config',result)
            self.assertEqual(result['aggregated_experimental_config']['model'],['facebook/opt-2.7b'])
            self.assertFalse(result['correctness_qualified_speedup'])

    def test_pilot_oom_preserves_prior_valid_pairs_and_numeric_fail(self):
        """Pure control fixture: never evidence of model/GPU correctness or speed."""
        from test_r2b1_prefix_cache import identity
        from test_r2b35_analysis import session
        m=runner()
        class Runtime:
            def __init__(self,a):
                self.a=a; self.provenance={}; self.device='cpu'
                self.model=NS(config=NS(max_position_embeddings=2048))
            def setup_matched(self): return {'status':'FAKE_CONTROL_FIXTURE'}
            def identity(self): return identity()
            def configure(self,arm,policy): self.arm=arm
            def prepare(self,n,p):
                if n==512: raise MemoryError('test capacity')
                self.n=n; self.p=p; self.ids=list(range(n)); self.snapshot=NS(identity_hash='source')
                self.source_report={}; self.preflight={}
            def check(self): return {'status':'UNCHANGED_FAKE'}
            def verify_weight_hashes(self): return {'status':'UNCHANGED_FAKE'}
        g=dict(output_matching='OUTPUT_MATCHED_VALID',native_endpoint_equivalent=True,axis_a={},
            axis_b={'matched':{'allclose':False},'legacy_suffix':{'allclose':False}},canonical={},
            audits={mode:dict(endpoint={'lora':[]},physical={}) for mode in bench.MODES})
        def pairs(rt,a,ident,save):
            _,rows=session(a.independent_run_id,count=a.measured_runs)
            rows=[dict(r,**ident,benchmark_phase=a.phase) for r in rows]
            save(rows=rows); return rows
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'Runtime',Runtime),patch.object(m,'gate',return_value=g),patch.object(m,'run_pairs',pairs),patch.object(m,'memory',return_value={}),patch.object(m,'monitor',return_value={'status':'GPU_MONITOR_UNVERIFIED'}),patch.object(m,'prepare_only',return_value={'wall_ms':1.}):
            smoke=Path(tmp)/'smoke'; smoke.mkdir()
            # Deliberately mocked approval metadata, confined to temporary test files.
            sargs=m.parse_args(['--output-dir',str(Path(tmp)/'metadata')]); metadata,_=m.initialize(sargs)
            approval=dict(metadata,experiment='R2-B3.5',execution_status='COMPLETE',GPU_VALIDATION='EXECUTED',native_endpoint_equivalent=True,benchmark_config={'phase':'smoke'})
            (smoke/'manifest.json').write_text(json.dumps(approval))
            out=Path(tmp)/'pilot'
            self.assertEqual(m.main(['--phase','pilot','--cases','128:127,512:511','--smoke-inputs',str(smoke),'--device','cpu','--output-dir',str(out)]),0)
            saved=json.loads((out/'manifest.json').read_text())
            self.assertEqual(saved['execution_status'],'COMPLETE_WITH_FAILURES')
            self.assertEqual(saved['conditions']['L128_P127_off']['status'],'CAPACITY_VALIDATED')
            self.assertFalse(saved['conditions']['L128_P127_off']['full_warm_strict'])
            self.assertEqual(saved['conditions']['L512_P511_off']['status'],'MEMORY_PREFLIGHT_FAILED')
            self.assertEqual(len(json.loads((out/'latency_raw_pairs.json').read_text())),3)
            self.assertEqual(saved['GPU_VALIDATION'],'NOT_RUN'); self.assertFalse(saved['correctness_qualified_speedup'])


if __name__=='__main__': unittest.main()
