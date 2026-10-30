"""CPU-only Raw-QKV ownership, real SemCache policy/index and exact guards."""
import copy
import importlib.util
import unittest

from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.semantic.matcher import ExactTokenMatcher
from semcache.edgelora.communication import ProtocolError
from semcache.edgelora.r3b_cache import ExactRawCache,invalid_guard_checks
from semcache.edgelora.r3b_fixture import ArrayTensor


def context():
    return dict(token_ids=[1,2,3,4],positions=[0,1,2,3],adapter='user_a',
        identity=dict(length=4,hidden_size=4,layers=32,hidden_dtype='float32',base_weight_sha256='TEST_FIXTURE_ONLY'),
        context=dict(prefix_token_ids=[],causal=True),component_scope='total_qkv')


def blocks():
    return {layer:tuple(ArrayTensor([float(layer+i+1)]*16,[1,4,4]) for i in range(3)) for layer in range(32)}


class ExactCacheTests(unittest.TestCase):
    def setUp(self):self.context=context();self.cache=ExactRawCache(self.context,6144,'cpu',fixture=True);self.blocks=blocks()
    def insert(self):return self.cache.insert(self.context,self.blocks,'TEST_SOURCE_COMPLETE',completed=True)

    def test_actual_semcache_classes_owned_total_qkv_and_full_exact_index(self):
        self.assertIsInstance(self.cache.pool,GlobalCache);self.assertIsInstance(self.cache.matcher,ExactTokenMatcher)
        self.assertIsNone(self.cache.lookup(self.context));self.assertTrue(self.insert())
        entry=self.cache.lookup(self.context);self.assertIsInstance(entry,CacheEntry)
        self.assertEqual(len(entry.tensors),32);self.assertEqual(entry.physical_tensor_bytes,6144)
        self.assertEqual(self.cache.bytes,6144);self.assertEqual(entry.qkv_metadata['component_scope'],'total_qkv')
        self.blocks[0][0].values[0]=999.
        self.assertEqual(entry.tensors[0][0].values[0],1.)
        clone=entry.tensors[0][0].clone();clone.values[0]=777.
        self.assertEqual(entry.tensors[0][0].values[0],1.)
        self.cache.reused(entry);self.assertEqual(entry.frequency,1)
        self.assertTrue(any(e['kind']=='ADMISSION' and e['allowed'] for e in self.cache.events))

    def test_capacity_rejection_allocates_nothing_and_reset_releases_storage(self):
        too_small=ExactRawCache(self.context,6143,'cpu',fixture=True)
        self.assertFalse(too_small.insert(self.context,self.blocks,'source',completed=True));self.assertEqual(too_small.bytes,0)
        self.assertTrue(self.insert());self.cache.reset('cold precondition')
        self.assertEqual(self.cache.bytes,0);self.assertIsNone(self.cache.lookup(self.context))

    def test_partial_and_uncompleted_capture_never_admitted(self):
        for complete,captures in ((False,self.blocks),(True,{0:self.blocks[0]}),(True,{**self.blocks,0:(self.blocks[0][0],)})):
            with self.subTest(complete=complete,layers=len(captures)),self.assertRaises(ProtocolError):
                self.cache.insert(self.context,captures,'unsafe',completed=complete)
        self.assertEqual(self.cache.bytes,0)

    def test_nonfinite_wrong_shape_dtype_and_foreign_device_rejected(self):
        for case in ('nonfinite','shape','dtype','device'):
            values=blocks();tensor=values[0][0]
            if case=='nonfinite':tensor.values[0]=float('nan')
            if case=='shape':tensor.shape=(1,2,8)
            if case=='dtype':tensor.dtype='float16'
            if case=='device':tensor.device='cuda:7'
            with self.subTest(case=case),self.assertRaises(ProtocolError):self.cache.insert(self.context,values,'unsafe',completed=True)
        self.assertEqual(self.cache.bytes,0)

    def test_context_model_adapter_tokens_positions_prefix_and_metadata_fail_closed(self):
        self.insert();guards=invalid_guard_checks(self.cache)
        self.assertEqual(len(guards),9);self.assertTrue(all(g['rejected'] for g in guards))
        self.assertTrue(all(g['real_alternative_adapter_or_model_executed'] is False for g in guards))
        self.assertIsNotNone(self.cache.lookup(self.context))
        for field,value in (('token_ids',[9,2,3,4]),('positions',[1,2,3,4]),('adapter','user_b')):
            changed=copy.deepcopy(self.context);changed[field]=value
            with self.subTest(field=field):self.assertIsNone(self.cache.lookup(changed))

    def test_real_content_corruption_rejected_by_hash_not_only_metadata(self):
        self.insert();entry=self.cache.lookup(self.context);entry.tensors[7][2].values[0]+=1.
        self.assertIsNone(self.cache.lookup(self.context))
        self.assertEqual(self.cache.events[-1]['kind'],'REJECT')

    def test_duplicate_does_not_overwrite_provenance_or_allocate(self):
        self.insert();entry=self.cache.lookup(self.context)
        self.assertFalse(self.cache.insert(self.context,blocks(),'DIFFERENT_SOURCE',completed=True))
        self.assertIs(self.cache.lookup(self.context),entry)
        self.assertEqual(entry.qkv_metadata['source_request_id'],'TEST_SOURCE_COMPLETE')

    def test_cpu_storage_is_only_allowed_for_explicit_fixture(self):
        with self.assertRaises(ProtocolError):ExactRawCache(self.context,6144,'cpu')

    def test_actual_hit_branch_does_not_call_any_miss_projection_or_delta(self):
        from semcache.edgelora.r3b_runtime import total_projection
        self.insert();entry=self.cache.lookup(self.context)
        def forbidden():raise AssertionError('HIT executed base projection or UD delta')
        for layer in range(32):
            for role in 'qkv':
                output=total_projection(entry,layer,role,forbidden)
                self.assertEqual(output.fixture_bytes(),entry.tensors[layer]['qkv'.index(role)].fixture_bytes())
                self.assertIsNot(output.values,entry.tensors[layer]['qkv'.index(role)].values)

    def test_existing_semcache_frontend_and_policy_numerical_regressions(self):
        from semcache.evaluation.system_validation import validate_frontend,validate_policy
        for records in (validate_frontend(),validate_policy()):self.assertTrue(all(r['passed'] for r in records))


@unittest.skipUnless(importlib.util.find_spec('torch') is not None and importlib.util.find_spec('numpy') is not None,
                     'actual Torch tensor ownership/hash unit test NOT_RUN; Torch/NumPy absent')
class TorchOwnership(unittest.TestCase):
    def test_actual_torch_cache_entry_owns_storage_and_detects_mutation(self):
        import torch
        ctx=context();cache=ExactRawCache(ctx,6144,'cpu',fixture=True)
        tensors={i:tuple(torch.full((1,4,4),float(i+j),dtype=torch.float32) for j in range(3)) for i in range(32)}
        self.assertTrue(cache.insert(ctx,tensors,'CPU_TENSOR_UNIT',completed=True));entry=cache.lookup(ctx)
        tensors[0][0].add_(100.)
        self.assertIsNotNone(cache.lookup(ctx));self.assertNotEqual(entry.tensors[0][0].data_ptr(),tensors[0][0].data_ptr())
        entry.tensors[0][0].add_(1.)
        self.assertIsNone(cache.lookup(ctx))


if __name__=='__main__':unittest.main()
