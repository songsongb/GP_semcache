"""Policy/control-plane tests; mocks never qualify actual GPU inference."""
import inspect
import unittest
from copy import deepcopy
from unittest.mock import Mock,patch
from semcache.edgelora.gpu_identity import validate_gpu,STRICT_PREPARATION_GPU,SAME_COMPATIBLE_GPU_CLASS
from semcache.edgelora.r5_i5_gpu import receipt,verify_receipt,match_execution,diagnostics
from semcache.edgelora.r4a_runtime import setup

class GPUPolicy(unittest.TestCase):
    def setUp(self):
        self.prepared=dict(uuid='preparation-real-uuid',name='NVIDIA GeForce RTX 3090')
        self.current=dict(uuid='execution-real-uuid',name='NVIDIA GeForce RTX 3090',device='cuda:0')
    def test_strict_default_and_equal_uuid(self):
        self.assertEqual(inspect.signature(setup).parameters['gpu_identity_policy'].default,STRICT_PREPARATION_GPU)
        self.assertEqual(validate_gpu(self.prepared,dict(self.current,uuid=self.prepared['uuid']),'cuda:0')['gpu_uuid_matched'],True)
    def test_legacy_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'fresh allocated'):validate_gpu(self.prepared,self.current,'cuda:0')
    def test_i5_mismatch_recorded_without_mutating_preparation(self):
        before=deepcopy(self.prepared);r=validate_gpu(self.prepared,self.current,'cuda:0',SAME_COMPATIBLE_GPU_CLASS)
        self.assertFalse(r['gpu_uuid_matched']);self.assertEqual(self.prepared,before)
        self.assertEqual(r['execution_gpu_uuid'],self.current['uuid'])
    def test_unavailable_uuid_rejected(self):
        for value in ('',' ','UNAVAILABLE','None',None):
            with self.assertRaises(ValueError):validate_gpu(self.prepared,dict(self.current,uuid=value),'cuda:0',SAME_COMPATIBLE_GPU_CLASS)
    def test_wrong_class_or_device_rejected(self):
        for fields in ({'name':'NVIDIA A100'},{'name':'RTX 3090 Ti'},{'device':'cuda:1'}):
            with self.assertRaises(ValueError):validate_gpu(self.prepared,dict(self.current,**fields),'cuda:0',SAME_COMPATIBLE_GPU_CLASS)
    def test_opt_in_not_available_to_legacy_plan(self):
        with self.assertRaisesRegex(ValueError,'I5-only'):setup({},gpu_identity_policy=SAME_COMPATIBLE_GPU_CLASS)
    def test_receipt_preserves_both_ids_and_rejects_restore(self):
        pm={'gpu':self.prepared};r=receipt(pm,self.current,SAME_COMPATIBLE_GPU_CLASS)
        self.assertEqual(r['preparation_gpu_uuid'],self.prepared['uuid']);self.assertTrue(r['physical_kv_created_in_current_execution'])
        verify_receipt(r,pm,self.current,SAME_COMPATIBLE_GPU_CLASS)
        with self.assertRaises(ValueError):verify_receipt(dict(r,historical_live_kv_restored=True),pm,self.current,SAME_COMPATIBLE_GPU_CLASS)
        with self.assertRaises(ValueError):verify_receipt(dict(r,base_fingerprint_verified=False),pm,self.current,SAME_COMPATIBLE_GPU_CLASS)
    def test_execution_identity_matched_across_conditions(self):
        match_execution(self.current['uuid'],self.current['uuid'])
        with self.assertRaisesRegex(ValueError,'changed'):match_execution(self.current['uuid'],'foreign-execution-uuid')
    def test_physical_namespace_uses_execution_uuid(self):
        from semcache.edgelora.r5_contracts import namespace
        model=dict(model='facebook/opt-2.7b',base_revision='a'*40,base_weight_sha256='b'*64,adapter='Adapter_A',adapter_matrix_sha256='c'*64,hidden_dtype='float16',layers=32,hidden_size=2560,rank=8)
        n=namespace({'identity':model},{'tokenizer_revision':'d'*40},'cuda:0',self.current['uuid'],{'test':'contract'})
        self.assertEqual(n['gpu_uuid'],self.current['uuid']);self.assertNotEqual(n['gpu_uuid'],self.prepared['uuid'])
    def test_no_cuda_diagnostic_is_not_checked(self):
        import sys
        torch=Mock();torch.cuda.is_available.return_value=False
        with patch.dict(sys.modules,{'torch':torch}):r=diagnostics({'gpu':self.prepared})
        self.assertFalse(r['eligible']);self.assertFalse(r['allocated_gpu_checked'])
    def test_fixture_never_claims_real_gpu(self):
        r=receipt({},None,SAME_COMPATIBLE_GPU_CLASS,fixture=True)
        self.assertIsNone(r['execution_gpu_uuid']);self.assertIsNone(r['gpu_class_compatible'])
        self.assertFalse(r['physical_kv_created_in_current_execution'])
    def test_restore_plan_rejected(self):
        from semcache.edgelora.r5_i5_gpu import reject_restore
        for plan in ({'historical_live_kv_restored':True},{'restore_cache_from':'old_job'},{'cache_checkpoint':'old.bin'}):
            with self.assertRaises(ValueError):reject_restore(plan)
    def test_setup_mismatch_allowed_but_model_fingerprint_still_enforced(self):
        import sys
        from contextlib import ExitStack
        import importlib
        for module in ('semcache.edgelora.r3a_runtime','semcache.models.model_adapter','semcache.system_cost.es_base_profile'):
            importlib.import_module(module)
        identity=dict(model='facebook/opt-2.7b',layers=32,hidden_size=2560,dtype='float16',seed=42,versions={'test':'pinned'},base_weight_sha256='expected')
        pm=dict(identity=identity,gpu=self.prepared,TEST_FIXTURE_ONLY=False)
        for fingerprint in ('expected','corrupt'):
            with ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules,{'torch':Mock()}))
                for target,value in [('semcache.edgelora.r4a_runtime.validate_prepared',(pm,None,None)),
                    ('semcache.edgelora.r3a_runtime.runtime_setup',None),('semcache.edgelora.r3a_runtime.memory',self.current),
                    ('semcache.edgelora.r3a_runtime.load_bare',(Mock(),Mock(),{})),('semcache.edgelora.r4a_runtime.versions',identity['versions']),
                    ('semcache.edgelora.r4a_runtime.full_base_fingerprint',{'sha256':fingerprint}),
                    ('semcache.system_cost.es_base_profile.assert_no_lora',None),('semcache.models.model_adapter.OPTModelAdapter',Mock()),
                    ('semcache.edgelora.r4a_runtime.ForwardDriver',Mock()),('os.sched_setaffinity',None)]:
                    stack.enter_context(patch(target,return_value=value))
                plan=dict(base_prepared='mock',fixture=False,device='cuda:0',i5_gpu_policy_scope='R5_I5')
                if fingerprint=='corrupt':
                    with self.assertRaisesRegex(ValueError,'fingerprint'):setup(plan,gpu_identity_policy=SAME_COMPATIBLE_GPU_CLASS)
                else:self.assertEqual(setup(plan,gpu_identity_policy=SAME_COMPATIBLE_GPU_CLASS)[-1]['uuid'],self.current['uuid'])
    def test_tokenizer_revision_mismatch_still_rejected(self):
        from semcache.edgelora.r3a_runtime import load_bare
        identity=dict(model='facebook/opt-2.7b',tokenizer_source='facebook/opt-2.7b',base_revision='a'*40,tokenizer_revision='b'*40)
        meta=dict(resolved_model_revision='a'*40,resolved_tokenizer_revision='c'*40,tokenizer_source_id='facebook/opt-2.7b')
        import importlib
        loader=importlib.import_module('semcache.models.loader')
        with patch.object(loader,'load_model',return_value=(Mock(),Mock(),meta)):
            with self.assertRaisesRegex(ValueError,'snapshots mismatch'):load_bare(identity,'cuda:0')
