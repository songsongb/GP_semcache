from copy import deepcopy
import unittest

from semcache.edgelora.communication import digest
from semcache.edgelora.r5_contracts import request,validate_request,gates
from semcache.edgelora.r5_workload import workload,generation_policy
from semcache.edgelora.r4c_prepare import TinyTokenizer


class ContractTests(unittest.TestCase):
    def make(self):
        r=workload(fixture=True)[0];tok=TinyTokenizer()
        identity=dict(schema='R5_PHYSICAL_IDENTITY_V1',adapter_id='user_a',model='TEST_FIXTURE_ONLY',model_revision='TEST_FIXTURE_ONLY',
            base_fingerprint='a'*64,adapter_matrix_fingerprint='b'*64,opt_contract='c'*64,rank=8,layers=32,hidden_size=4,
            dtype='float32',device='cpu',gpu_uuid='TEST_FIXTURE_ONLY',tokenizer={'scope':'TEST_FIXTURE_ONLY'},
            attention_context='standalone causal; all-ones mask; positions from zero')
        return request(r,tok.encode(r['input_text']),generation_policy(tok),identity,fixture=True)

    def test_version_content_hash_and_independent_logical_user(self):
        r=self.make();self.assertEqual(validate_request(r),r)
        r['user_id']='different';r.pop('request_sha256');r['request_sha256']=digest(r)
        self.assertEqual(validate_request(r)['adapter_id'],'user_a')

    def test_required_schema_tokens_positions_and_tampering_fail(self):
        for key,value in (('schema','old'),('input_text','changed'),('token_ids',[2]),('positions',[1]*10),('attention_mask',[0]*10),('request_id','../escape')):
            r=self.make();r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_request(r)
        r=self.make();r.pop('sample_id')
        with self.assertRaises(ValueError):validate_request(r)

    def test_adapter_and_matrix_identity_are_distinct_from_user(self):
        r=self.make();r['adapter_id']='other';r.pop('request_sha256');r['request_sha256']=digest(r)
        with self.assertRaisesRegex(ValueError,'unprepared adapter'):validate_request(r)

    def test_fixture_gates_never_default_to_gpu_or_native_pass(self):
        flags=gates(True)
        self.assertFalse(flags['R5_GPU_PHYSICAL_EXECUTION_PASS']);self.assertFalse(flags['NATIVE_PEFT_STRICT_PARITY_PASS'])
        self.assertFalse(flags['correctness_qualified_speedup'])

    def test_physical_fingerprints_and_layout_are_required(self):
        for key,value in (('base_fingerprint','missing'),('adapter_matrix_fingerprint','other'),('hidden_size',0),('rank',4)):
            r=self.make();r['physical_identity'][key]=value;r.pop('request_sha256');r['request_sha256']=digest(r)
            with self.subTest(key=key),self.assertRaises(ValueError):validate_request(r)
