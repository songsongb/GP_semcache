"""CPU-only analysis/export contract, scope separation and no heavy imports."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from r2d_fixtures import ROOT, make_all, make_m9a, read, sha256, write
from semcache.diagnostics.r2d_evidence import EvidenceAudit, load_m9a, load_c4
from semcache.diagnostics.r2d_analysis import analyze

BANDS = [10.,50.,100.,200.,500.,1000.]
RTTS = [0.,1.,10.,30.,50.]
SCRIPT = ROOT/'scripts/94_run_r2d_edgelora_break_even.py'


def runner():
    spec = importlib.util.spec_from_file_location('r2d_runner_test',SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.m9path, self.c4path = make_all(self.root)
        audit = EvidenceAudit(ROOT)
        self.m9 = load_m9a(self.m9path,audit)
        self.c4 = load_c4(self.c4path,audit)

    def test_no_cross_cohort_addition_strict_failure_oracle_and_copy_counters(self):
        sweep, summary, decision, compat = analyze(self.m9,self.c4,BANDS,RTTS)
        self.assertEqual(len(sweep),120)
        self.assertEqual(compat['status'],'NOT_COMPARABLE')
        self.assertFalse(decision['primary_claim_eligible'])
        for key in ('real_network_measured','full_generation_e2e_measured','correctness_qualified_speedup'):
            self.assertFalse(decision[key])
        c4other = copy.deepcopy(self.c4)
        for c in c4other['conditions']:
            c['comparisons']['baseline_minus_no_reuse_ms']['observed_equal_session_mean_ms'] += 1_000_000
        other = analyze(self.m9,c4other,BANDS,RTTS)[0]
        historical = [r for r in sweep if r['evidence_class'] == 'HISTORICAL_M9A']
        self.assertEqual(historical,[r for r in other if r['evidence_class'] == 'HISTORICAL_M9A'])
        r = historical[0]
        self.assertAlmostEqual(r['payload_only_reuse_total_ms'],r['reuse_es_ms']+r['reuse_ud_ms']+
                               r['reuse_control_ms']+r['reuse_transfer_ms'])
        self.assertNotEqual(r['saved_payload_bytes'],502824960)
        local = next(c for c in summary['r2c4_conditional_illustrative'] if c['actual_reused_tokens'] == 1023)
        self.assertFalse(local['full_relative_strict']['RAW_REPLAY_BASELINE'])
        self.assertFalse(local['oracle_diagnostics_deployable'])
        threshold = next(t for t in local['thresholds'] if t['bandwidth_mbps'] == 100)
        self.assertAlmostEqual(threshold['required_avoided_payload_bytes'],536.28*100/.008)
        self.assertEqual(local['bandwidth_root_status'],'NOT_ESTIMABLE')

    def test_zero_reuse_noise_and_no_invented_distributed_savings(self):
        sweep, summary, _, _ = analyze(self.m9,self.c4,BANDS,RTTS)
        zero = next(c for c in summary['r2c4_conditional_illustrative'] if c['actual_reused_tokens'] == 0)
        self.assertEqual(zero['status'],'NOT_APPLICABLE')
        self.assertLess(zero['comparisons']['baseline_minus_no_reuse_ms']['observed_equal_session_mean_ms'],0)
        self.assertTrue(all(t['required_avoided_payload_bytes'] is None for t in zero['thresholds']))
        for row in sweep:
            if row['evidence_class'] == 'R2C4_CONDITIONAL_ILLUSTRATIVE':
                self.assertNotIn('saved_payload_bytes',row)
                self.assertNotIn('hypothetical_reuse_minus_no_reuse_ms',row)
            elif row['rtt_ms']:
                self.assertIsNone(row['actual_rtt_adjusted_delta_ms'])
                self.assertEqual(row['actual_rtt_status'],'NOT_ESTIMABLE')
                # Any nonzero fresh-token batch keeps the same assumed RTT count.
                self.assertEqual(row['hypothetical_no_reuse_exchanges'],row['hypothetical_reuse_exchanges'])

    def test_separate_nested_cpu_gpu_timers_never_summed_into_totals(self):
        before = analyze(self.m9,self.c4,BANDS,RTTS)
        altered = copy.deepcopy(self.c4)
        altered['breakdown'][0]['components'][0].update(host_ms=10_000_000,cuda_ms=999_999)
        altered['breakdown'][0]['instrumented_request_wall_ms'] = 99_000_000
        after = analyze(self.m9,altered,BANDS,RTTS)
        self.assertEqual(before[0],after[0])
        self.assertEqual(before[1]['historical_m9a'],after[1]['historical_m9a'])
        self.assertTrue(all(c['do_not_sum_nested'] for c in after[1]['r2c4_separate_diagnostic_components']))
        self.assertTrue(all(not c['network_bytes'] for c in after[1]['r2c4_logical_copy_counters']))

    def test_bandwidth_winners_negative_results_roots_and_out_of_domain(self):
        costly = copy.deepcopy(self.m9)
        costly['records'][0]['row']['semcache_control_ms'] = 1000.
        sweep, summary, _, _ = analyze(costly,self.c4,BANDS,RTTS)
        rows = [r for r in sweep if r['evidence_class'] == 'HISTORICAL_M9A' and r['rtt_ms'] == 0]
        self.assertEqual({r['payload_only_winner'] for r in rows},{'REUSE','NO_REUSE'})
        self.assertTrue(any(r['payload_only_speedup_percent'] < 0 for r in rows))
        root = summary['historical_m9a'][0]['roots'][0]['root']
        self.assertEqual(root['status'],'FINITE_ROOT')
        self.assertEqual(root['winning_region'],'BELOW_ROOT')
        only = analyze(costly,self.c4,[1000.],[0.])[1]['historical_m9a'][0]['roots'][0]['root']
        self.assertTrue(only['root_outside_tested_domain'])
        zero = load_m9a(make_m9a(self.root/'zero',zero=True),EvidenceAudit(ROOT))
        zs = analyze(zero,self.c4,BANDS,RTTS)[1]['historical_m9a'][0]
        self.assertEqual(zs['ledger']['saved']['total_network_bytes'],0)

    def command(self,out):
        return [sys.executable,'-u',str(SCRIPT),'--phase','analyze','--m9a-summary',str(self.m9path),
            '--c4-aggregate-dir',str(self.c4path),'--bandwidth-mbps','10,50,100,200,500,1000',
            '--rtt-ms','0,1,10,30,50','--output-dir',str(out)]

    def test_exact_cli_contract_all_artifacts_hashes_no_heavy_imports_or_network(self):
        out = self.root/'empty'
        out.mkdir()
        cmd = self.command(out)
        guard = '''import importlib.abc, runpy, socket, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch','transformers','datasets','numpy') or 'cachegen' in name.lower():
            raise AssertionError('forbidden import: '+name)
sys.meta_path.insert(0, Block())
def forbidden(*a, **k): raise AssertionError('network forbidden')
class ForbiddenSocket(socket.socket):
    def __new__(cls, *a, **k): raise AssertionError('network forbidden')
socket.socket = ForbiddenSocket
socket.create_connection = forbidden
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        frozen = [ROOT/'scripts/93_run_r2c4_cache_path.py',ROOT/'src/semcache/system_cost/model.py']
        old = {str(p):sha256(p) for p in frozen}
        run = subprocess.run([sys.executable,'-c',guard,*cmd[2:]],capture_output=True,text=True,cwd='/tmp')
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(old,{str(p):sha256(p) for p in frozen})
        manifest = read(out/'r2d_manifest.json')
        self.assertEqual(manifest['execution_status'],'COMPLETE')
        self.assertTrue(manifest['TEST_FIXTURE_ONLY'])
        self.assertFalse(manifest['gpu_inference_performed'])
        self.assertFalse(manifest['network_executed'])
        self.assertEqual(set(p.name for p in out.iterdir()),{
            'r2d_manifest.json','r2d_evidence_audit.json','r2d_bandwidth_rtt_sweep.csv',
            'r2d_break_even_summary.json','r2d_decision.json'})
        for name, expected in manifest['output_hashes'].items():
            self.assertEqual(sha256(out/name),expected)
            self.assertGreater((out/name).stat().st_size,0)
        with (out/'r2d_bandwidth_rtt_sweep.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),120)
        decision = read(out/'r2d_decision.json')
        self.assertEqual(decision['classification'],'ANALYTICAL_DIAGNOSTIC')
        self.assertFalse(decision['analytical_competitiveness']['supported_within_m9a'])
        snapshots = {p.name:sha256(p) for p in out.iterdir()}
        again = subprocess.run(cmd,capture_output=True,text=True)
        self.assertNotEqual(again.returncode,0)
        self.assertEqual(snapshots,{p.name:sha256(p) for p in out.iterdir()})

    def test_failed_evidence_exports_failed_manifest_and_visible_invalid_row(self):
        self.m9path.write_text('{"schema": "wrong"}')
        out = self.root/'failed'
        run = subprocess.run(self.command(out),capture_output=True,text=True)
        self.assertNotEqual(run.returncode,0)
        self.assertEqual(read(out/'r2d_manifest.json')['execution_status'],'FAILED')
        self.assertEqual(read(out/'r2d_decision.json')['validity_status'],'NOT_ESTIMABLE')
        self.assertIn('NOT_ESTIMABLE',(out/'r2d_bandwidth_rtt_sweep.csv').read_text())

    def test_cli_help_and_invalid_grids_without_gpu_phases(self):
        help_run = subprocess.run([sys.executable,str(SCRIPT),'--help'],capture_output=True,text=True)
        self.assertEqual(help_run.returncode,0,help_run.stderr)
        self.assertIn('--c4-aggregate-dir',help_run.stdout)
        m = runner()
        for text, positive in (('0',True),('nan',True),('inf',False),('1,1',True),('-1',False),('',True)):
            with self.assertRaises(ValueError): m.grid(text,positive=positive)


if __name__ == '__main__':
    unittest.main()
