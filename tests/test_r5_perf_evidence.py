from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.r5_perf_evidence import Writer,PerformanceEvidence,RESERVE,accounted_modules
from semcache.edgelora.r5_perf_analysis import accounting_audit


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.writer=Writer(self.root,128*2**20,create=True)

    def test_incremental_append_replace_and_independent_reconciliation(self):
        w=self.writer;w.bytes(self.root/'a',b'abc',exclusive=True);w.bytes(self.root/'a',b'def',append=True)
        w.bytes(self.root/'b',b'12345');w.bytes(self.root/'b',b'x')
        self.assertEqual(w.status()['data_bytes'],7);self.assertEqual(w.reconcile()['files'],2)
        self.assertEqual(accounting_audit(self.root)['data_bytes'],7)

    def test_multiprocess_log_growth_and_pipe_sized_writes(self):
        code='from semcache.edgelora.r5_perf_evidence import Writer; from pathlib import Path; import sys; p=Path(sys.argv[1]); w=Writer(p,128*2**20); [w.bytes(p/sys.argv[2],b"x"*101,append=True) for i in range(25)]'
        children=[subprocess.Popen([sys.executable,'-c',code,str(self.root),str(i)],env=dict(os.environ,PYTHONPATH='src')) for i in range(2)]
        observations=[]
        while any(p.poll() is None for p in children):
            observations.append(self.writer.status()['data_bytes'])
        self.assertTrue(all(p.wait(timeout=30)==0 for p in children));self.assertEqual(self.writer.status()['data_bytes'],5050)
        self.assertEqual(observations,sorted(observations))
        self.writer.reconcile();self.assertEqual(accounting_audit(self.root)['data_files'],2)

    def test_overflow_fails_closed_and_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);w=Writer(p,RESERVE+5,create=True);w.bytes(p/'one',b'1234')
            with self.assertRaisesRegex(ValueError,'BUDGET'):w.bytes(p/'two',b'xx')
            self.assertEqual((p/'one').read_bytes(),b'1234');self.assertFalse((p/'two').exists())
            with self.assertRaisesRegex(ValueError,'FAILED'):w.status()

    def test_unexpected_missing_and_corrupted_files_are_detected(self):
        for action in ('extra','missing','corrupt'):
            with self.subTest(action=action),tempfile.TemporaryDirectory() as d:
                p=Path(d);w=Writer(p,128*2**20,create=True);w.bytes(p/'one',b'1234')
                if action=='extra':(p/'foreign').write_bytes(b'not accounted')
                if action=='missing':(p/'one').unlink()
                if action=='corrupt':(p/'one').write_bytes(b'4321')
                with self.assertRaisesRegex(ValueError,'MISMATCH'):w.reconcile()

    def test_bounded_checks_do_not_scan_every_decode_and_profiles_keep_digests(self):
        es=self.root/'es';es.mkdir();e=PerformanceEvidence(es,self.writer,reconcile_every=4)
        with patch.object(self.writer,'reconcile',wraps=self.writer.reconcile) as scan:
            for _ in range(7):e.check()
            self.assertEqual(scan.call_count,1)
        tensor=ArrayTensor([1.,2.,3.,4.],[1,1,4]);compact=e.save(tensor,'compact')
        self.assertNotIn('file',compact);e.profile='STRICT_AUDIT';strict=e.save(tensor,'strict')
        self.assertEqual({k:v for k,v in strict.items() if k!='file'},compact)
        self.assertEqual((es/strict['file']).stat().st_size,16);self.writer.reconcile()

    def test_optin_io_injection_is_restored_and_does_not_modify_frozen_files(self):
        from semcache.edgelora import r3a_artifacts,r4c_generation
        original=r4c_generation.append
        with accounted_modules(self.writer):
            r4c_generation.append(self.root/'record.jsonl',dict(value=1))
            r3a_artifacts.write(self.root/'meta.json',dict(value=2))
        self.assertIs(r4c_generation.append,original);self.writer.reconcile()

    def test_collision_and_corrupt_append_never_silently_resume(self):
        self.writer.bytes(self.root/'one',b'1234',exclusive=True)
        with self.assertRaisesRegex(ValueError,'collision'):self.writer.bytes(self.root/'one',b'abc',exclusive=True)

    def test_profiled_strict_and_scalable_preserve_actual_tiny_forward_math(self):
        # Numerical unit control; independent TCP is covered by the ABBA fixture.
        from types import SimpleNamespace
        from semcache.edgelora.r3a_tiny import prepare_tiny,linear
        from semcache.edgelora.r3a_artifacts import read,load_matrices
        from semcache.edgelora.r4a_fixture import TinyBackend
        from semcache.edgelora.r4c_generation import QualityDriver
        from semcache.edgelora.r5_perf_timing import ProfiledDriver
        from semcache.edgelora.r5_workload import generation_policy
        from semcache.edgelora.r4c_prepare import TinyTokenizer
        base=prepare_tiny(self.root/'base');pm=read(base/'r3a_prepare_manifest.json');am=read(base/'r3a_adapter_manifest.json')
        matrices=load_matrices(base,am,torch_backend=False);ids=[1,2,3,4];outputs=[]
        class Session:
            def __init__(self,*a,**k):pass
            def correction(self,l,r,x):return linear(linear(x,matrices[l,r,'A'],4,8),matrices[l,r,'B'],8,4)
            def complete(self):pass
        for i,kind in enumerate(('original','strict','scalable')):
            path=self.root/str(i);path.mkdir();tiny=TinyBackend(read(base/'r3a_tiny_base.json'),matrices)
            cls=QualityDriver if kind=='original' else ProfiledDriver
            d=cls(pm['identity'],'cpu',tiny=tiny,channel=SimpleNamespace(records=[]));d.configure(path);d.sample_id='unit'
            if kind!='original':
                w=Writer(path/'accounted',128*2**20,create=True)
                d.output=w.root/'es';d.output.mkdir();d.diagnostics_path=d.output/'diagnostics.jsonl'
                d.evidence=PerformanceEvidence(d.output,w,profile='STRICT_AUDIT' if kind=='strict' else 'SCALABLE_PERFORMANCE',perf=d.perf)
            with patch('semcache.edgelora.r4c_generation.DynamicSession',Session):
                out,record,blocks,values,blob=d.c_forward('UNIT',0,ids,ids,None,generation_policy(TinyTokenizer()),'NO_REUSE','a'*64)
            outputs.append((out.logits,blob,record['raw_kv_proof']))
        self.assertEqual(outputs[0],outputs[1]);self.assertEqual(outputs[0],outputs[2])

