"""Actual pilot/30-user TCP generation and adversarial independent I4A audits."""
from copy import deepcopy
from importlib.util import module_from_spec,spec_from_file_location
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import canonical
from semcache.edgelora.r3a_artifacts import ROOT,read,write
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.r5_evidence import verify
from semcache.edgelora.r5_multi_analysis import analyze,HISTORY,gates
from semcache.edgelora.r5_perf_evidence import Writer

COMMAND=[sys.executable,str(ROOT/'scripts/116_run_r5_multiuser.py')]


class MultiProcesses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(os.environ.get('R5M_TEST_OUTPUT_ROOT',cls.tmp.name)).resolve()
        cls.root.mkdir(parents=True,exist_ok=True);cls.paths={};cls.reports={}
        spec=spec_from_file_location('r5m_test_runner',ROOT/'scripts/116_run_r5_multiuser.py');runner=module_from_spec(spec);spec.loader.exec_module(runner)
        for stage in ('pilot','expanded'):
            existing=os.environ.get('R5M_TEST_'+stage.upper()+'_INPUTS')
            if existing:
                dest=Path(existing).resolve();verify(dest)
                if read(dest/'multi_sources.json')!=runner.sources():raise AssertionError('existing fixture execution sources differ')
                if read(dest/'multi_config.json')['fixture'] is not True:raise AssertionError('GPU evidence cannot stand in for a CPU test')
            else:
                dest=cls.root/stage
                r=subprocess.run(COMMAND+['--phase','fixture','--test-fixture-only','--stage',stage,'--condition-budget-bytes',str(512*2**20),
                    '--generation-budget','2','--output-dir',str(dest)],capture_output=True,text=True,timeout=900)
                if r.returncode:raise AssertionError(str(dest)+'\n'+r.stderr)
            cls.paths[stage]=dest;cls.reports[stage]=analyze(dest)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_two_adapter_pilot_actual_user_sharing_and_foreign_rejection(self):
        p=self.reports['pilot'];m=p['metrics']['INTEGRATED']['global_metrics']
        self.assertEqual((m['requests'],m['logical_users'],m['actual_ud_processes'],m['physical_hits'],m['admissions']),(4,4,2,2,2))
        self.assertGreater(m['rejected_cross_adapter_candidates'],0)
        rows={r['request_id']:r for r in jsonl(self.paths['pilot']/'INTEGRATED/generation.jsonl')}
        self.assertEqual(rows['M_A_01']['source']['source_request_id'],'M_A_00')
        self.assertIsNone(rows['M_B_00']['source']);self.assertEqual(rows['M_B_01']['source']['source_request_id'],'M_B_00')

    def test_three_adapters_thirty_logical_users_and_post_eviction_misses(self):
        p=self.reports['expanded'];m=p['metrics']['INTEGRATED']['global_metrics']
        self.assertEqual((m['requests'],m['logical_users'],m['actual_ud_processes'],m['physical_hits'],m['admissions'],m['eviction_count']),(30,30,3,12,12,3))
        self.assertEqual(m['physical_hit_rate'],.4)
        for aid,stats in p['metrics']['INTEGRATED']['per_adapter'].items():
            self.assertEqual((stats['logical_users'],stats['physical_hits'],stats['eviction_count']),(10,4,1))
            self.assertLessEqual(stats['peak_cache_bytes'],stats['cache_capacity_bytes'])
        rows={r['request_id']:r for r in jsonl(self.paths['expanded']/'INTEGRATED/generation.jsonl')}
        for a in 'ABC':
            self.assertIsNone(rows[f'M_{a}_05']['source'])
            self.assertEqual(rows[f'M_{a}_06']['decision'],'SEMANTIC_CANDIDATE_REJECTED')

    def test_actual_distinct_adapter_corrections_and_fresh_ud_processes(self):
        for stage,p in self.reports.items():
            pids=[];sessions=[]
            for condition,data in p['audit']['conditions'].items():
                corrections=set();hidden=set()
                for aid,entry in data['adapters'].items():
                    pids.append(entry['ud_pid']);sessions.append(entry['session_id'])
                    corrections.add(entry['first_layer_delta']);hidden.add(entry['first_layer_hidden'])
                self.assertEqual(len(hidden),1);self.assertEqual(len(corrections),2 if stage=='pilot' else 3)
            self.assertEqual(len(set(pids)),len(pids));self.assertEqual(len(set(sessions)),len(sessions))

    def test_completed_matched_generation_frozen_tolerances_and_separate_gates(self):
        for stage,p in self.reports.items():
            for key in gates():
                if key not in ('R5_I4A_GPU_EXECUTION_PASS','NATIVE_PEFT_STRICT_PARITY_PASS','correctness_qualified_speedup'):
                    self.assertTrue(p['decision'][key],(stage,key))
            self.assertFalse(p['decision']['R5_I4A_GPU_EXECUTION_PASS']);self.assertFalse(p['decision']['correctness_qualified_speedup'])
            self.assertTrue(all(r['generated_tokens_equal'] and r['generated_text_equal'] and r['frozen_tolerance_pass'] for r in p['fidelity']))
            self.assertEqual(p['limitations']['historical'],HISTORY)

    def test_client_results_private_namespace_and_no_foreign_metadata(self):
        for stage,root in self.paths.items():
            self.assertEqual(root.stat().st_mode&0o777,0o700)
            for path in (root/'INTEGRATED/client_results').glob('*.json'):
                self.assertEqual(set(read(path)),{'request_id','status','generated_token_ids','generated_text','termination'})
            self.assertEqual(read(root/'INTEGRATED/negative_controls.json')['wire_frames'],0)

    def test_missing_gpu_optin_rejected_before_any_es_worker(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'blocked';env=dict(os.environ);env.pop('R5M_ENABLE_GPU',None)
            r=subprocess.run(COMMAND+['--phase','run','--enable-gpu-experiment','--multi-prepared',str(self.paths['pilot']/'multi_prepare'),
                '--output-dir',str(out)],env=env,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(r.returncode,0);self.assertIn('R5M_ENABLE_GPU',read(out/'multi_failure.json')['reason'])
            self.assertFalse((out/'DISTRIBUTED_NO_REUSE').exists());self.assertFalse(read(out/'multi_decision.json')['R5_I4A_GPU_EXECUTION_PASS'])

    def test_independent_read_only_reports_deterministic(self):
        source=self.paths['expanded'];before=(source/'r5_output_manifest.json').read_bytes()
        for label in ('analysis_one','analysis_two'):
            r=subprocess.run([sys.executable,str(ROOT/'scripts/117_analyze_r5_multiuser.py'),'--inputs',str(source),
                '--output-dir',str(self.root/label)],capture_output=True,text=True,timeout=180)
            self.assertEqual(r.returncode,0,r.stderr)
        for path in (self.root/'analysis_one').glob('multi_*.json'):
            if path.name=='multi_results.json':continue
            self.assertEqual(path.read_bytes(),(self.root/'analysis_two'/path.name).read_bytes())
        self.assertEqual(before,(source/'r5_output_manifest.json').read_bytes())

    def test_forged_foreign_adapter_identity_rejected_after_accounting_is_updated(self):
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/'copy';shutil.copytree(self.paths['pilot'],copy)
            condition=copy/'INTEGRATED';path=condition/'Adapter_A/identity.json';identity=read(path)
            identity['physical_identity']['adapter_id']='Adapter_B'
            writer=Writer(condition,read(copy/'multi_config.json')['condition_budget_bytes']);writer.write(path,identity)
            with self.assertRaisesRegex(ValueError,'physical adapter'):analyze(copy,sealed=False)

    def test_stale_session_and_foreign_ownership_rejected_after_valid_accounting(self):
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/'copy';shutil.copytree(self.paths['pilot'],copy);condition=copy/'INTEGRATED'
            writer=Writer(condition,read(copy/'multi_config.json')['condition_budget_bytes'])
            path=condition/'Adapter_A/ud/plan.json';plan=read(path);old=deepcopy(plan);plan['session_id']='stale_session';writer.write(path,plan)
            with self.assertRaisesRegex(ValueError,'UD startup identity'):analyze(copy,sealed=False)
            writer.write(path,old)
            path=condition/'ownership.jsonl';rows=jsonl(path);rows[0]['adapter_id']='Adapter_B'
            writer.bytes(path,b''.join(canonical(r)+b'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'ownership/admission'):analyze(copy,sealed=False)

    def test_corrupt_cached_tensor_and_inventory_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/'copy';shutil.copytree(self.paths['pilot'],copy)
            es=copy/'INTEGRATED/Adapter_A/es';item=jsonl(es/'r5_tokens.jsonl')[0]['qkv']['0.k'];path=es/item['file']
            before=path.read_bytes();path.write_bytes(bytes([before[0]^1])+before[1:])
            with self.assertRaises(ValueError):analyze(copy)
            with self.assertRaises(ValueError):analyze(copy,sealed=False)


class Launcher(unittest.TestCase):
    def test_help_dry_run_no_overwrite_and_missing_gpu_optin(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertEqual(subprocess.run(COMMAND+['--help'],capture_output=True,timeout=20).returncode,0)
            args=['--phase','dry-run','--stage','expanded','--output-dir',str(root/'dry')]
            self.assertEqual(subprocess.run(COMMAND+args,capture_output=True,timeout=20).returncode,0)
            self.assertNotEqual(subprocess.run(COMMAND+args,capture_output=True,timeout=20).returncode,0)
            self.assertEqual(read(root/'dry/multi_preflight.json')['logical_users'],30)

    def test_late_failure_revokes_gates(self):
        spec=spec_from_file_location('r5m_late_failure',ROOT/'scripts/116_run_r5_multiuser.py');runner=module_from_spec(spec);spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);source=path/'source';source.mkdir();write(source/'r5_output_manifest.json',{'files':{}})
            with patch.object(runner,'analyze',return_value=dict(decision={k:True for k in gates()})),patch.object(runner,'export'),patch.object(runner,'sha',side_effect=ValueError('late identity failure')):
                self.assertEqual(runner.main(['--phase','analyze','--inputs',str(source),'--output-dir',str(path/'out')]),1)
            self.assertTrue(all(v is False for v in read(path/'out/multi_decision.json').values()))


if __name__=='__main__':unittest.main()
