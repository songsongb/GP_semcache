import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from semcache.edgelora.r3a_artifacts import ROOT,read,write
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.r5_perf_analysis import analyze

COMMAND=[sys.executable,str(ROOT/'scripts/112_run_r5_balanced_benchmark.py')]


class BalancedProcesses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.out=cls.root/'balanced'
        r=subprocess.run(COMMAND+['--phase','fixture','--test-fixture-only','--block-budget-bytes',str(256*2**20),'--output-dir',str(cls.out)],capture_output=True,text=True,timeout=600)
        if r.returncode:
            p=ROOT/'results/r5/performance/test_failures'/str(time.time_ns());p.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(cls.out,p)
            raise AssertionError(str(p)+'\n'+r.stderr)
        cls.report=analyze(cls.out)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_actual_abba_fresh_processes_and_automatic_hits(self):
        a=self.report;self.assertEqual(a['audit']['requests'],40);self.assertEqual([b['condition'] for b in a['blocks']],list('ABBA'))
        self.assertEqual([b['physical_hits'] for b in a['blocks']],[0,4,4,0])
        self.assertEqual([b['admissions'] for b in a['blocks']],[0,4,4,0]);self.assertEqual([b['evictions'] for b in a['blocks']],[0,1,1,0])
        self.assertFalse(a['decision']['R5_I3P_GPU_EXECUTION_PASS']);self.assertFalse(a['decision']['R5_I3P_PERFORMANCE_QUALIFIED'])
        self.assertTrue(a['decision']['R5_I3P_STATE_ISOLATION_PASS']);self.assertTrue(a['decision']['R5_I3P_EVIDENCE_ACCOUNTING_PASS'])

    def test_fidelity_negative_control_and_insufficient_independent_repetitions(self):
        a=self.report;self.assertEqual(len(a['paired_requests']),20);self.assertEqual(len(a['negative_controls']),12)
        self.assertTrue(all(p['token_agreement'] for p in a['paired_requests']))
        self.assertEqual(a['statistics']['independent_repetitions'],1)
        self.assertIsNone(a['statistics']['workload_e2e_delta']['ci95'])
        self.assertFalse(a['decision']['correctness_qualified_speedup'])

    def test_budget_scan_and_post_forward_timing_are_observed(self):
        steps=self.report['steps'];self.assertEqual(len(steps),80)
        self.assertTrue(all('evidence_budget_check' in s['exclusive_forward_ms'] and 'forward_preflight' in s['exclusive_forward_ms'] for s in steps))
        self.assertTrue(all(s['post_forward_ms']>=0 and s['step_through_json_ms']>=s['wall_to_availability_ms'] for s in steps))
        self.assertTrue(all(s['evidence']['recursive_scans']<s['evidence']['budget_checks'] for s in steps if s['step']>0))

    def test_independent_analysis_deterministic_and_read_only(self):
        before=(self.out/'r5_output_manifest.json').read_bytes();self.assertEqual(self.report,analyze(self.out))
        for name in ('one','two'):
            r=subprocess.run([sys.executable,str(ROOT/'scripts/113_analyze_r5_balanced_benchmark.py'),'--inputs',str(self.out),'--output-dir',str(self.root/name)],capture_output=True,text=True,timeout=120)
            self.assertEqual(r.returncode,0,r.stderr)
        for name in ('r5_perf_audit.json','r5_perf_decision.json','r5_perf_statistics.json','r5_perf_fidelity.json','r5_perf_drift.json'):
            self.assertEqual((self.root/'one'/name).read_bytes(),(self.root/'two'/name).read_bytes())
        self.assertEqual(before,(self.out/'r5_output_manifest.json').read_bytes())

    def test_unexpected_file_and_state_leak_and_missing_blocks_fail_audit(self):
        foreign=self.out/'blocks/B000/es/foreign';foreign.write_bytes(b'x')
        try:
            with self.assertRaisesRegex(ValueError,'accounting'):analyze(self.out,sealed=False)
        finally:foreign.unlink()
        path=self.out/'blocks/B000/r5_initial_state.json';before=path.read_bytes()
        try:
            state=read(path);state['cluster']['queries']=1;write(path,state)
            with self.assertRaises(ValueError):analyze(self.out,sealed=False)
        finally:path.write_bytes(before)
        path=self.out/'r5_perf_execution_order.json';before=path.read_bytes()
        try:
            write(path,read(path)[:-1])
            with self.assertRaisesRegex(ValueError,'incomplete'):analyze(self.out,sealed=False)
        finally:path.write_bytes(before)


class LauncherTests(unittest.TestCase):
    def test_late_failure_revokes_previously_computed_qualification(self):
        from importlib.util import spec_from_file_location,module_from_spec
        from types import SimpleNamespace
        from unittest.mock import patch
        from semcache.edgelora.r5_perf_analysis import flags
        spec=spec_from_file_location('r5p_failure_test',ROOT/'scripts/112_run_r5_balanced_benchmark.py')
        m=module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'failure';claimed={k:True for k in flags()}
            # Failure-path orchestration control, no forward/TCP/GPU execution.
            with patch.object(m,'spawn',return_value=(SimpleNamespace(pid=123),[],[])), \
                 patch.object(m,'finish',return_value=dict(pid=123,returncode=0,all_children_exited=True,forced_termination=False,pipe_errors=[])), \
                 patch('semcache.edgelora.r5_perf_analysis.analyze',return_value=dict(decision=claimed)), \
                 patch('semcache.edgelora.r5_perf_analysis.export'), \
                 patch.object(m,'verify_sources',side_effect=ValueError('late source verification failure')):
                self.assertEqual(m.main(['--phase','fixture','--test-fixture-only','--output-dir',str(out)]),1)
            self.assertTrue(all(v is False for v in read(out/'r5_perf_decision.json').values()))
            self.assertEqual(read(out/'r5_perf_results.json')['execution_status'],'FAILED')

    def test_help_dry_run_missing_gpu_optins_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            r=subprocess.run(COMMAND+['--help'],capture_output=True,text=True,timeout=20);self.assertEqual(r.returncode,0,r.stderr)
            args=['--phase','dry-run','--output-dir',str(root/'dry')]
            self.assertEqual(subprocess.run(COMMAND+args,capture_output=True,timeout=20).returncode,0)
            self.assertNotEqual(subprocess.run(COMMAND+args,capture_output=True,timeout=20).returncode,0)
            env=dict(os.environ);env.pop('R5P_ENABLE_GPU',None)
            r=subprocess.run(COMMAND+['--phase','run','--enable-gpu-experiment','--base-prepared',str(root/'absent'),'--output-dir',str(root/'run')],env=env,capture_output=True,text=True,timeout=20)
            self.assertNotEqual(r.returncode,0);self.assertFalse((root/'run/blocks').exists())
            self.assertFalse(read(root/'run/r5_perf_decision.json')['R5_I3P_GPU_EXECUTION_PASS'])
