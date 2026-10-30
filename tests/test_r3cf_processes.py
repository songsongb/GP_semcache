"""Actual CPU localhost fixtures with original transport; never GPU/151893 data."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from semcache.edgelora.r3a_artifacts import ROOT,read,sha
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.r3cf_evidence import audit_physical,legacy_reanalysis,load_focused

SCRIPT=ROOT/'scripts/101_run_r3c_break_even_focused.py'


def cli():
    spec=importlib.util.spec_from_file_location('focused_test_cli',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def temporary(prefix='focused_break_even_TEST_FIXTURE_ONLY_'):
    base=ROOT/'results/r3/c';base.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix,dir=base)


class Contracts(unittest.TestCase):
    def test_help_dry_run_and_non_destructive_output(self):
        with temporary() as tmp:
            command=[sys.executable,str(SCRIPT),'--phase','dry-run','--output-dir',tmp]
            run=subprocess.run(command,capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr)
            report=read(Path(tmp)/'r3cf_results.json');self.assertEqual(report['gpu_findings_status'],'NOT_RUN')
            self.assertFalse(report['actual_network_socket_used']);self.assertEqual(report['budget']['model_request_bound'],126)
            before={p.name:sha(p) for p in Path(tmp).iterdir()}
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(before,{p.name:sha(p) for p in Path(tmp).iterdir()})
        help_run=subprocess.run([sys.executable,str(SCRIPT),'--help'],capture_output=True,text=True)
        self.assertEqual(help_run.returncode,0);self.assertNotIn('--hit-fractions',help_run.stdout)
        code="import semcache.edgelora.r3cf_runtime; import sys; assert not any(n in sys.modules for n in ('torch','numpy','transformers','peft'))"
        run=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')))
        self.assertEqual(run.returncode,0,run.stderr)

    def test_missing_legacy_report_no_fetch_or_synthetic_substitution(self):
        with temporary() as tmp:
            out=Path(tmp)/'missing';missing=Path(tmp)/'original_SERAPH_not_mounted'
            run=subprocess.run([sys.executable,str(SCRIPT),'--phase','reanalyze','--legacy-input',str(missing),
                '--output-dir',str(out)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            report=read(out/'r3cf_legacy_reanalysis.json');self.assertEqual(report['execution_status'],'NOT_RUN_MISSING_INPUT')
            self.assertNotIn('arms',report);self.assertFalse((out/'es').exists())

    def test_missing_allocation_fails_before_any_socket(self):
        with temporary() as tmp:
            env=dict(os.environ);env.pop('SLURM_JOB_ID',None)
            run=subprocess.run([sys.executable,str(SCRIPT),'--phase','run','--enable-gpu-experiment','--output-dir',tmp],
                env=env,capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0);report=read(Path(tmp)/'r3cf_results.json')
            self.assertEqual(report['execution_status'],'FAILED');self.assertFalse(report['gpu_inference_performed'])
            self.assertFalse((Path(tmp)/'proxy').exists())

    def test_invalid_calibration_retains_failures_and_cleans_proxy_before_prepare(self):
        with temporary() as tmp:
            with patch('semcache.edgelora.r3c_calibration.validate_calibration',return_value=[dict(status='INVALID_IMPAIRMENT')]):
                status=cli().main(['--phase','fixture','--test-fixture-only','--output-dir',tmp])
            self.assertEqual(status,1);report=read(Path(tmp)/'r3cf_results.json')
            self.assertIn('INVALID_IMPAIRMENT',report['reason']);self.assertFalse((Path(tmp)/'prepare').exists())
            self.assertTrue(read(Path(tmp)/'proxy/proxy_lifecycle.json')['threads_stopped'])
            for kind,directory in (('proxy','proxy'),('echo','calibration')):
                life=read(Path(tmp)/directory/(kind+'_parent_lifecycle.json'))
                with self.assertRaises(ProcessLookupError):os.kill(life['pid'],0)


class PhysicalFocused(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=temporary();cls.addClassCleanup(cls.temp.cleanup);cls.output=Path(cls.temp.name)/'physical'
        run=subprocess.run([sys.executable,'-u',str(SCRIPT),'--phase','fixture','--test-fixture-only','--output-dir',str(cls.output)],
            capture_output=True,text=True,timeout=180.)
        if run.returncode:raise AssertionError(run.stdout+run.stderr)

    def test_all126_independent_processes_real_frames_and_auxiliary_exclusion(self):
        report,plan=load_focused(self.output);rows=jsonl(self.output/'r3cf_requests.jsonl')
        self.assertTrue(report['TEST_FIXTURE_ONLY']);self.assertFalse(report['gpu_inference_performed'])
        self.assertEqual(len(rows),126);self.assertEqual(sum(r['phase']=='measured' for r in rows),96)
        self.assertEqual(sum(r['phase']=='aux_measured' for r in rows),16)
        self.assertEqual(report['audit']['completed_ud_requests'],101)
        self.assertEqual(sum(r['hidden_frames'] for r in rows),3232)
        self.assertTrue(all(len(r['qkv_sha256'])==96 and r['cache_bitwise_passed'] for r in rows))
        self.assertEqual(len({r['es_pid'] for r in rows}),1);self.assertEqual(len({r['ud_pid'] for r in rows}),1)
        self.assertEqual(len({r['proxy_pid'] for r in rows}),1)
        self.assertEqual(plan['calibration']['proxy_pid'],rows[0]['proxy_pid'])
        for row in rows:
            if row['mode']=='WARM_HIT':
                self.assertEqual((row['hidden_frames'],row['delta_frames'],row['ud_cpu_lora_ms']),(0,0,0))
                self.assertEqual(row['reused_qkv_outputs'],96)
            else:self.assertEqual((row['hidden_frames'],row['delta_frames']),(32,32))
        for path in ('es/r3a_ud_ready.json','proxy/proxy_ready.json'):
            info=read(self.output/path)
            with self.assertRaises(ProcessLookupError):os.kill(info['pid'],0)
            for key in ('port','data_port','control_port'):
                if key in info:
                    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as client:
                        client.settimeout(.2);self.assertNotEqual(client.connect_ex(('127.0.0.1',info[key])),0)

    def test_independent_analysis_same_decision_and_input_hashes_unchanged(self):
        before={str(p.relative_to(self.output)):sha(p) for p in self.output.rglob('*') if p.is_file()}
        with temporary() as tmp:
            run=subprocess.run([sys.executable,str(SCRIPT),'--phase','analyze','--inputs',str(self.output),'--output-dir',tmp],
                capture_output=True,text=True,timeout=60.)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertEqual(read(Path(tmp)/'r3cf_break_even_decision.json'),read(self.output/'r3cf_break_even_decision.json'))
        self.assertEqual(before,{str(p.relative_to(self.output)):sha(p) for p in self.output.rglob('*') if p.is_file()})

    def test_rehashed_invalid_rows_schedule_counts_or_metadata_cannot_pass_audit(self):
        plan=read(self.output/'es/es_plan.json');result=read(self.output/'es/es_result.json')
        rows=jsonl(self.output/'es/r3cf_requests.jsonl')
        with temporary() as tmp:
            copy=Path(tmp)/'copy';shutil.copytree(self.output,copy)
            for key,value in (('delta_frames',31),('analysis_role','AUXILIARY'),('schedule_sha256','0'*64),
                              ('native_parity',dict(passed=False)),('logits_sha256','0'*64)):
                altered=json.loads(json.dumps(rows));altered[0][key]=value
                (copy/'es/r3cf_requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in altered))
                with self.subTest(key=key),self.assertRaises(ValueError):audit_physical(copy,plan,result)
            (copy/'r3cf_requests.jsonl').write_text('CORRUPTED\n')
            with self.assertRaises(ValueError):load_focused(copy)


class ActualLegacyFormat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A genuine original-100 tiny CPU run, explicitly unrelated to 151893.
        cls.temp=temporary('network_one_shot_TEST_FIXTURE_ONLY_r3cf_');cls.addClassCleanup(cls.temp.cleanup)
        cls.output=Path(cls.temp.name)/'legacy'
        run=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/100_run_r3c_network_one_shot.py'),
            '--phase','fixture','--test-fixture-only','--output-dir',str(cls.output)],capture_output=True,text=True,timeout=180.)
        if run.returncode:raise AssertionError(run.stdout+run.stderr)

    def test_original_228_fixture_retrospective_is_read_only_and_never_confirmed(self):
        before={str(p.relative_to(self.output)):sha(p) for p in self.output.rglob('*') if p.is_file()}
        with self.assertRaises(ValueError):legacy_reanalysis(self.output)
        result=legacy_reanalysis(self.output,allow_fixture=True)
        self.assertTrue(result['TEST_FIXTURE_ONLY']);self.assertEqual(result['independent_request_ids'],228)
        self.assertEqual(result['interpretation'],'RETROSPECTIVE_EXPLORATORY_ONLY');self.assertEqual(len(result['arms']),20)
        singles=[p for p in result['arms'] if p['n_hit']==1];self.assertEqual(len(singles),4)
        self.assertTrue(all(p['single_hit_degenerate'] and not p['uncertainty_adequate_for_diagnostic_bracket'] for p in singles))
        self.assertEqual(result['original_decision'],read(self.output/'r3c_break_even_decision.json'))
        self.assertTrue(all(p['original_unconditional_csv_row'] for p in result['arms']))
        self.assertEqual(before,{str(p.relative_to(self.output)):sha(p) for p in self.output.rglob('*') if p.is_file()})


@unittest.skipUnless(os.environ.get('SEMCACHE_R3CF_GPU')=='1','R3-C-F OPT/GPU NOT_RUN; explicit Slurm opt-in required')
class GPUOptIn(unittest.TestCase):
    def test_gpu_126(self):
        output=Path(os.environ['SEMCACHE_R3CF_OUTPUT'])
        run=subprocess.run([sys.executable,str(SCRIPT),'--phase','run','--enable-gpu-experiment','--output-dir',str(output)],
            capture_output=True,text=True,timeout=10000.)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        report,_=load_focused(output);self.assertTrue(report['gpu_inference_performed']);self.assertFalse(report['TEST_FIXTURE_ONLY'])


if __name__=='__main__':unittest.main()
