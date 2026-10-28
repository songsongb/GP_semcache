from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from semcache.diagnostics.r2b3_capacity import approved,failure_record,host_memory,release_failed_case


class CapacityTests(unittest.TestCase):
    def test_exact_approval_requires_finished_weights_and_environment(self):
        c=dict(status='CAPACITY_VALIDATED',structural_valid=True,cache_integrity_valid=True,condition_identity='tokens')
        p=dict(experiment='R2-B3',phase='pilot',execution_status='COMPLETE',model_weights_verified=True,
            environment_identity='same',conditions={'L128_P64_off':c})
        self.assertEqual(approved([p],'L128_P64','off','same'),c)
        for key,arm,identity in [('L256_P192','off','same'),('L128_P64','on','same'),('L128_P64','off','other')]:
            with self.assertRaises(ValueError): approved([p],key,arm,identity)
        with self.assertRaises(ValueError): approved([dict(p,execution_status='RUNNING')],'L128_P64','off','same')
        c['status']='OOM'
        with self.assertRaises(ValueError): approved([p],'L128_P64','off','same')

    def test_failure_classes_and_host_memory_unavailable(self):
        for exc,status in [(MemoryError('capacity'),'MEMORY_PREFLIGHT_FAILED'),(RuntimeError('CUDA out of memory'),'OOM'),(RuntimeError('bad mask'),'INVALID_EXECUTION')]:
            try: raise exc
            except Exception as caught: r=failure_record(caught,{'L':1024})
            self.assertEqual(r['status'],status); self.assertIn(type(exc).__name__,r['traceback'])
            self.assertFalse(r['valid_performance_evidence'])
        with patch('pathlib.Path.read_text',side_effect=OSError('unavailable')):
            h=host_memory(); self.assertIsNone(h['rss_bytes']); self.assertEqual(h['status'],'UNAVAILABLE')

    def test_cleanup_only_explicit_after_case_failure(self):
        r=NS(device='cpu',snapshot=object(),source_logits=object())
        release_failed_case(r); self.assertIsNone(r.snapshot); self.assertIsNone(r.source_logits)


if __name__=='__main__': unittest.main()
