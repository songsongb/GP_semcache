"""Actual CPU ES/UD TCP fixture and independent adversarial forensic reports."""
from importlib.util import spec_from_file_location, module_from_spec
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from semcache.edgelora.r3a_artifacts import ROOT, read, write
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.r5_evidence import seal, verify
from semcache.edgelora.r5_fidelity import TARGETS, gates
from semcache.edgelora.r5_fidelity_analysis import analyze

COMMAND=[sys.executable,str(ROOT/'scripts/114_run_r5_fidelity_forensics.py')]


class ForensicProcesses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.root=Path(os.environ.get('R5F_TEST_OUTPUT_ROOT',cls.tmp.name)).resolve()
        if cls.root!=Path(cls.tmp.name):
            if cls.root.exists() and any(cls.root.iterdir()):raise AssertionError('refusing nonempty retained test output')
            cls.root.mkdir(parents=True,exist_ok=True)
        cls.out=cls.root/'fixture'
        r=subprocess.run(COMMAND+['--phase','fixture','--test-fixture-only','--generation-budget','2',
            '--block-budget-bytes',str(256*2**20),'--output-dir',str(cls.out)],capture_output=True,text=True,timeout=900)
        if r.returncode:
            dest=ROOT/'results/r5/fidelity/test_failures'/str(time.time_ns());dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(cls.out,dest)
            raise AssertionError(str(dest)+'\n'+r.stderr)
        cls.report=analyze(cls.out)
        cls.reference=cls.root/'unchanged_reference'
        r=subprocess.run([sys.executable,str(ROOT/'scripts/110_run_r5_semcache_e2e.py'),'--phase','fixture','--test-fixture-only',
            '--generation-budget','2','--lifecycle-capacity','--output-dir',str(cls.reference)],capture_output=True,text=True,timeout=600)
        if r.returncode:raise AssertionError('unchanged reference fixture failed: '+r.stderr)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_actual_causal_sources_automatic_hits_and_cpu_scope(self):
        a=self.report;self.assertEqual(a['audit']['requests'],20)
        self.assertEqual({d['target_request_id']:d['source_request_id'] for d in a['dependencies']},
            dict(R5_03='R5_00',R5_08='R5_07',R5_01='R5_00',R5_09='R5_04'))
        self.assertTrue(a['decision']['R5_I3F_INDEPENDENT_AUDIT_PASS'])
        self.assertFalse(a['decision']['R5_I3F_GPU_FORENSICS_COMPLETE']);self.assertFalse(a['decision']['R5_I3F_CPU_TESTS_PASS'])
        self.assertFalse(a['decision']['NATIVE_PEFT_STRICT_PARITY_PASS']);self.assertFalse(a['decision']['correctness_qualified_speedup'])

    def test_reconstruction_is_independently_bitwise_verified(self):
        a=self.report;self.assertTrue(a['decision']['R5_I3F_CACHE_INTEGRITY_PASS'])
        self.assertEqual(len(a['reconstruction']),4*32*2*6)
        self.assertTrue(all(r['metrics']['bitwise_equal'] for r in a['reconstruction']))
        self.assertEqual([a['audit']['wire'][k]['status'] for k in ('A','B')],['PASS','PASS'])

    def test_full_target_suffix_decode_frozen_gates_and_tokens(self):
        a=self.report;self.assertEqual(len(a['tolerances']),8)
        self.assertTrue(a['decision']['R5_I3F_FROZEN_NUMERICAL_TOLERANCE_PASS'])
        self.assertTrue(a['decision']['R5_I3F_GENERATION_TOKEN_FIDELITY_PASS'])
        self.assertTrue(all(r['root_cause']=='ROOT_CAUSE_UNRESOLVED' and not r['skipped_decode'] for r in a['divergence']))
        self.assertFalse(a['decision']['R5_I3F_FIRST_DIVERGENCE_IDENTIFIED'])

    def test_passive_instrumentation_preserves_unchanged_production_fixture(self):
        reference={r['request_id']:r for r in jsonl(self.reference/'r5_generation_outputs.jsonl')}
        for condition in ('A','B'):
            es=self.out/'blocks'/condition/'es'
            for row in jsonl(es.parent/'r5_generation_outputs.jsonl'):
                old=reference[row['request_id']]
                self.assertEqual(row['generated_token_ids'],old['generated_token_ids'])
                self.assertEqual(row['generated_text'],old['generated_text'])
                self.assertEqual((es/row['logits_file']).read_bytes(),(self.reference/'es'/old['logits_file']).read_bytes())

    def test_independent_reports_deterministic_and_inputs_immutable(self):
        before=(self.out/'r5_output_manifest.json').read_bytes()
        self.assertEqual(self.report,analyze(self.out))
        for name in ('one','two'):
            r=subprocess.run([sys.executable,str(ROOT/'scripts/115_analyze_r5_fidelity_forensics.py'),
                '--inputs',str(self.out),'--output-dir',str(self.root/name)],capture_output=True,text=True,timeout=180)
            self.assertEqual(r.returncode,0,r.stderr)
        for path in (self.root/'one').glob('fidelity_*.json'):
            if path.name=='fidelity_results.json':continue
            self.assertEqual(path.read_bytes(),(self.root/'two'/path.name).read_bytes(),path.name)
        self.assertEqual(before,(self.out/'r5_output_manifest.json').read_bytes());verify(self.out)

    def test_corrupted_extra_and_missing_evidence_fail_independent_audit(self):
        es=self.out/'blocks/B/es';item=jsonl(es/'fidelity_reconstruction.jsonl')[0]['tensors']['L0.reconstructed_k']
        path=es/item['file'];before=path.read_bytes()
        try:
            path.write_bytes(b'x'+before[1:])
            with self.assertRaises(ValueError):analyze(self.out)
            with self.assertRaises(ValueError):analyze(self.out,sealed=False)
        finally:path.write_bytes(before)
        extra=es/'foreign.bin';extra.write_bytes(b'extra')
        try:
            with self.assertRaises(ValueError):analyze(self.out,sealed=False)
        finally:extra.unlink()
        path=es/'fidelity_steps.jsonl';before=path.read_bytes()
        try:
            path.write_bytes(b'')
            with self.assertRaises(ValueError):analyze(self.out,sealed=False)
        finally:path.write_bytes(before)


class Launcher(unittest.TestCase):
    def test_sealed_failed_run_cannot_be_promoted_by_reanalysis(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write(root/'fidelity_results.json',dict(execution_status='FAILED'));seal(root)
            with self.assertRaisesRegex(ValueError,'incomplete or failed'):analyze(root)

    def test_help_dry_run_optins_and_output_collision(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            r=subprocess.run(COMMAND+['--help'],capture_output=True,text=True,timeout=20);self.assertEqual(r.returncode,0)
            args=['--phase','dry-run','--output-dir',str(root/'dry')]
            self.assertEqual(subprocess.run(COMMAND+args,capture_output=True,timeout=20).returncode,0)
            self.assertNotEqual(subprocess.run(COMMAND+args,capture_output=True,timeout=20).returncode,0)
            env=dict(os.environ);env.pop('R5F_ENABLE_GPU',None)
            r=subprocess.run(COMMAND+['--phase','run','--enable-gpu-experiment','--base-prepared',str(root/'missing'),
                '--output-dir',str(root/'run')],env=env,capture_output=True,text=True,timeout=20)
            self.assertNotEqual(r.returncode,0);self.assertFalse((root/'run/blocks').exists())
            self.assertTrue(all(v is False for v in read(root/'run/fidelity_decision.json').values()))

    def test_late_failure_revokes_already_computed_gates(self):
        spec=spec_from_file_location('r5f_failure_test',ROOT/'scripts/114_run_r5_fidelity_forensics.py');m=module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'src';src.mkdir();write(src/'r5_output_manifest.json',dict(files={}))
            out=Path(d)/'out'
            with patch('semcache.edgelora.r5_fidelity_analysis.analyze',return_value=dict(decision={k:True for k in gates()})), \
                 patch('semcache.edgelora.r5_fidelity_analysis.export'),patch.object(m,'sha',side_effect=ValueError('late manifest read failure')):
                self.assertEqual(m.main(['--phase','analyze','--inputs',str(src),'--output-dir',str(out)]),1)
            self.assertTrue(all(v is False for v in read(out/'fidelity_decision.json').values()))
            self.assertEqual(read(out/'fidelity_results.json')['execution_status'],'FAILED')
