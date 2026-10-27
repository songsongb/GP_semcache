"""CPU-only FULL-relative arithmetic and artifact/workload invariants."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from semcache.diagnostics.r1b3_break_even import (amortized,summarize_pairs,schedule,target_variants,workload_target,hierarchical,create_run,MissingEvidence)
ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/name)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Tokenizer:
    all_special_ids=[0]
    def __len__(self): return 40
    def decode(self,ids,**kwargs): return ','.join(map(str,ids))
    def __call__(self,text): return {'input_ids':[int(x) for x in text.split(',')]}


class BreakEvenTests(unittest.TestCase):
    def test_full_not_legacy_amortization_and_strict_equality(self):
        self.assertEqual(amortized(80,100,120,10)['status'],'NO_FINITE_BREAK_EVEN_UNDER_MEASURED_STEADY_STATE')
        self.assertEqual(amortized(80,80,80,0)['n'],None)
        # N*10 > 20+12+(N-1)*8 -> 2N>24 -> N=13.
        self.assertEqual(amortized(10,8,12,20)['n'],13)
        self.assertEqual(amortized(10,8,None,0)['status'],'MISSING_COLD_MAPPING_EVIDENCE')
        for full,warm,cold,source in [(10,9,12,3),(100,20,80,0),(3,2,1,0)]:
            n=amortized(full,warm,cold,source)['n']
            self.assertGreater(n*full,source+cold+(n-1)*warm)
            if n>1: self.assertLessEqual((n-1)*full,source+cold+(n-2)*warm)

    def rows(self):
        base=dict(length=128,requested_coverage='max',actual_coverage=126/128,workload='warm',executor='contiguous',instrumentation='minimal',
            source_snapshot_id='source',source_costs=dict(incremental_build_ms=30.,source_prefill_ms=100.),executor_parity=True,
            correctness_status='STRICT_FAIL',reused_tokens=126)
        return [dict(base,phase='warmup',full_ms=80.,reuse_ms=60.,mapping_cache_hit=False),
                *[dict(base,phase='measured',full_ms=80.,reuse_ms=50.,mapping_cache_hit=True) for _ in range(3)]]

    def test_strict_fail_and_breakdown_cannot_qualify(self):
        rows=self.rows(); r=summarize_pairs(rows)[0]
        self.assertTrue(r['observed_break_even']); self.assertFalse(r['correctness_qualified_break_even'])
        self.assertEqual(r['amortized_existing_source']['incremental_source_ms'],30.)
        self.assertEqual(r['amortized_dedicated_source']['incremental_source_ms'],130.)
        for x in rows: x['instrumentation']='breakdown'; x['correctness_status']='STRICT_PASS'
        r=summarize_pairs(rows)[0]; self.assertFalse(r['observed_break_even']); self.assertFalse(r['correctness_qualified_break_even'])
        for x in rows: x['instrumentation']='minimal'; x['executor_parity']=False
        self.assertFalse(summarize_pairs(rows)[0]['observed_break_even'])

    def test_cold_target_is_real_token_change_prefix_safe_and_unique(self):
        ids=[1,2,3,4,5,6,7,8]; t=Tokenizer()
        cold=target_variants(t,ids,t.decode(ids),10,3,lambda text,x:x[:-1]==ids[:-1])
        self.assertEqual(len({tuple(x['ids']) for x in cold}),10)
        self.assertTrue(all(x['ids'][:-1]==ids[:-1] and x['ids']!=ids for x in cold))
        warm=dict(ids=ids,text=t.decode(ids))
        self.assertIs(workload_target('mixed',0,warm,cold),warm)
        self.assertIs(workload_target('mixed',1,warm,cold),cold[1])
        self.assertIs(workload_target('cold',0,warm,cold),cold[0])
        with self.assertRaises(ValueError): target_variants(t,[1,2,3],'',2,3)
        with self.assertRaises(MissingEvidence): target_variants(t,ids,'',50,3)

    def test_balanced_order_separate_seed_and_bootstrap_keeps_outlier(self):
        orders=schedule(['full','legacy','contiguous'],30,101)
        for slot in range(3): self.assertEqual({m:sum(o[slot]==m for o in orders) for m in orders[0]},dict(full=10,legacy=10,contiguous=10))
        self.assertNotEqual(orders,schedule(['full','legacy','contiguous'],30,102))
        result=hierarchical({'a':[1,2,300],'b':[1,2,3],'c':[1,2,3]},samples=200)
        self.assertGreater(result['mean'],30); self.assertEqual(result,hierarchical({'a':[1,2,300],'b':[1,2,3],'c':[1,2,3]},samples=200))

    def test_output_collision_and_cli_formal_counts(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'run'; manifest=create_run(ROOT,out,dict(seed=1),'test')
            self.assertEqual(manifest['execution_status'],'RUNNING')
            with self.assertRaises(FileExistsError): create_run(ROOT,out,dict(seed=1),'test')
        runner=load('80_benchmark_r1b3_b3_break_even.py')
        a=runner.parse_args(['--output-dir','unused','--profile','formal','--coverages','0,.25,.5,.75,max','--order-seed','103'])
        self.assertEqual(a.measured_runs,30); self.assertEqual(a.coverage_values,[0.,.25,.5,.75,'max'])
        with self.assertRaises(SystemExit): runner.parse_args(['--output-dir','unused','--profile','formal','--measured-runs','10'])

    def test_aggregator_rejects_incomplete_and_corrupt_evidence(self):
        m=load('81_aggregate_r1b3_b3.py')
        with tempfile.TemporaryDirectory() as root:
            p=Path(root); (p/'manifest.json').write_text(json.dumps(dict(execution_status='FAILED',classification='ORACLE_DIAGNOSTIC')))
            with self.assertRaises(ValueError): m.load_run(p)

    def test_aggregate_full_relative_ci_strict_fail_and_order_identity(self):
        from semcache.metrics.r1 import atomic_json,digest
        m=load('81_aggregate_r1b3_b3.py')
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for i in range(3):
                path=Path(directory)/str(i); path.mkdir(); paths.append(path)
                manifest=dict(identity=str(i),classification='ORACLE_DIAGNOSTIC',execution_status='COMPLETE',
                    config=dict(independent_run_id=str(i),order_seed=101+i,measured_runs=30,seed=42),
                    code=dict(sha256='synthetic-test'),cases={'case':dict(fixtures={})},runtime=dict(model={'revision':'test'},
                        weight_identity='weights',encoder={},gpu={'gpu_uuid':'test-GPU'},torch='test',cuda='test'))
                atomic_json(path/'manifest.json',manifest)
                rows=[dict(row_id=str(j),independent_run_id=str(i),order_seed=101+i,phase='measured',repeat=j,
                    length=128,requested_coverage='max',workload='warm',executor='contiguous',instrumentation='minimal',
                    full_ms=80.,reuse_ms=70.,paired_difference_ms=10.,executor_parity=True,qkv_validation='PASS',
                    correctness_status='STRICT_FAIL',reused_tokens=126) for j in range(30)]
                atomic_json(path/'checkpoints'/'case.json',dict(identity=str(i),status='COMPLETE',rows=rows,checksum=digest(rows)))
            output=Path(directory)/'aggregate'
            command=['--run-dirs',*map(str,paths),'--output-dir',str(output),'--bootstrap-samples','100']
            self.assertEqual(m.main(command),0)
            result=json.loads((output/'break_even_decisions.json').read_text())['groups'][0]
            self.assertTrue(result['reproducible_break_even']); self.assertFalse(result['correctness_qualified_break_even'])
            self.assertEqual(result['bootstrap']['mean'],10.)
            with self.assertRaises(FileExistsError): m.main(command)
            manifest=json.loads((paths[2]/'manifest.json').read_text()); manifest['config']['order_seed']=101
            atomic_json(paths[2]/'manifest.json',manifest)
            with self.assertRaises(ValueError): m.main(command)

    def test_runner_startup_failure_keeps_manifest_and_required_artifacts(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from unittest.mock import patch
        artifact_sets={
            '79_diagnose_r1b3_b3_fp16.py':('baseline_repeatability.json','first_divergence.csv',
                'layerwise_qkv_comparison.csv','precision_control.csv','logit_correctness.csv','root_cause_summary.json'),
            '80_benchmark_r1b3_b3_break_even.py':('latency_comparison.csv','paired_differences.csv','coverage_sweep.csv',
                'cold_warm_mapping.csv','source_cost_accounting.csv','memory_accounting.csv','gpu_state.json','break_even_summary.json','summary.json')}
        with tempfile.TemporaryDirectory() as root:
            for name,artifacts in artifact_sets.items():
                runner=load(name); out=Path(root)/name
                with patch.object(runner,'Runtime',side_effect=ModuleNotFoundError('test: Torch unavailable')),redirect_stderr(StringIO()):
                    self.assertEqual(runner.main(['--output-dir',str(out)]),1)
                manifest=json.loads((out/'manifest.json').read_text())
                self.assertEqual(manifest['execution_status'],'FAILED')
                self.assertEqual(manifest['correctness_status'],'UNVERIFIED')
                self.assertIn('Traceback',manifest['traceback'])
                self.assertTrue(all((out/a).exists() for a in artifacts))
                with self.assertRaises(FileExistsError): runner.main(['--output-dir',str(out)])

    def test_b21_runtime_interfaces_resolve_without_torch(self):
        from semcache.diagnostics.r1b3_b3_runtime import b21
        reference=b21(); clone,a3=reference.runtime_helpers()
        self.assertEqual(Path(clone.__code__.co_filename).name,'75_benchmark_r1b3_selection.py')
        self.assertTrue(callable(a3.quality)); self.assertTrue(callable(reference.RequestMapping))


if __name__=='__main__': unittest.main()
