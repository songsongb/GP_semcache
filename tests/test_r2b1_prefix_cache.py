"""Cache/namespace contracts run without model packages; Torch tests are explicit."""
from dataclasses import replace
from contextlib import ExitStack
import hashlib
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import semcache.diagnostics.r2b1_prefix_cache as cache


def identity():
    return cache.PrefixIdentity('opt','r'*40,'base','tok','t'*40,'user_a','adapter','torch.float32','cpu')


class FakeTensor:
    def __init__(self,length=3,data=None):
        self.shape=(1,2,length,4); self.dtype='torch.float32'; self.device='cpu'; self._version=0
        self.data=list(range(8*length)) if data is None else list(data)
    def clone(self): return FakeTensor(self.shape[2],self.data)
    def stride(self): return (8*self.shape[2],4*self.shape[2],4,1)
    def data_ptr(self): return id(self.data)
    def numel(self): return len(self.data)
    def element_size(self): return 4
    def untyped_storage(self): return NS(nbytes=lambda:self.numel()*4)


class FakeDynamic:
    def __init__(self,pairs=()): self.pairs=pairs
    def to_legacy_cache(self): return self.pairs
    @classmethod
    def from_legacy_cache(cls,pairs): return cls(pairs)
    def get_seq_length(self,*a): return self.pairs[0][0].shape[2]


class NamespaceTests(unittest.TestCase):
    def test_exact_prefix_and_independent_suffix_allowed(self):
        a=identity()
        r=cache.validate_request(a,a,[1,2,3],[1,2,3,9],mask=[1]*4,position_start=3)
        self.assertEqual(r['suffix_length'],1)
        self.assertTrue(cache.validate_request(a,a,[1,2],[1,2,99,98])['cache_hit'])
    def test_each_namespace_and_semantic_substring_rejected(self):
        a=identity()
        for field,value in [('model_id','other'),('model_revision','other'),('base_weights','other'),('tokenizer_id','other'),
            ('tokenizer_revision','other'),('adapter','user_b'),('adapter_weights','other'),('dtype','other'),('device','other'),
            ('backend','sdpa'),('position_policy','other'),('mask_policy','other'),('eval_mode',False),('output_attentions',False)]:
            with self.subTest(field=field),self.assertRaises(cache.PrefixError):
                cache.validate_request(a,replace(a,**{field:value}),[1,2],[1,2,3])
        for kw in (dict(source_start=1),dict(position_start=0),dict(mask=[1]),dict(mask=[1,0,1])):
            with self.assertRaises(cache.PrefixError): cache.validate_request(a,a,[1,2],[1,2,3],**kw)
        with self.assertRaisesRegex(cache.PrefixError,'REJECT_PREFIX_TOKEN_MISMATCH'):
            cache.validate_request(a,a,[1,2],[2,1,3])
        for bad in ([True,2,3],[1.,2,3],[1,2,-3]):
            with self.assertRaisesRegex(cache.PrefixError,'REJECT_TOKEN_IDS'):
                cache.validate_request(a,a,[1,2],bad)
        for prefix in ([],[1,2,3]):
            with self.assertRaises(cache.PrefixError): cache.validate_request(a,a,prefix,[1,2,3])


class FakeCacheTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(cache,'clone_tensor',lambda t:t.clone()))
        self.stack.enter_context(patch.object(cache,'tensor_hash',lambda t:hashlib.sha256(str(t.data).encode()).hexdigest()))
    def snapshot(self,past=None):
        return cache.PrefixSnapshot(past if past is not None else ((FakeTensor(),FakeTensor()),),identity(),[1,2,3],layers=1,heads=2,head_dim=4,source_path='PREFIX_SOURCE_PREFILL')
    def test_source_clone_request_clone_and_repeat(self):
        original=((FakeTensor(),FakeTensor()),); s=self.snapshot(original)
        self.assertNotEqual(original[0][0].data_ptr(),s.pairs[0][0].data_ptr())
        for _ in range(3):
            request=s.materialize(identity(),[1,2,3,4],position_start=3,mask=[1]*4)
            self.assertEqual(request[0][0].shape[2],3)
            request[0][0].data[0]=999
            self.assertEqual(s.check()['status'],'UNCHANGED')
        self.assertEqual(s.bytes,2*3*8*4)
    def test_canonical_mutation_even_without_version_change(self):
        s=self.snapshot(); s.pairs[0][0].data[0]=999
        with self.assertRaisesRegex(cache.PrefixError,'REJECT_SOURCE_CACHE_MUTATION'): s.check()
        s=self.snapshot(); s.prefix_ids=(9,2,3)
        with self.assertRaises(cache.PrefixError): s.check()
    def test_layout_and_source_provenance(self):
        for value in (((FakeTensor(2),FakeTensor(2)),),((FakeTensor(),),)):
            with self.assertRaises(cache.PrefixError): self.snapshot(value)
        with self.assertRaises(cache.PrefixError):
            cache.PrefixSnapshot(((FakeTensor(),FakeTensor()),),identity(),[1,2,3],layers=1,heads=2,head_dim=4,source_path='FULL_USE_CACHE')
    def test_source_copy_corruption_rejected(self):
        with patch.object(cache,'clone_tensor',lambda t:FakeTensor(t.shape[2],[v+1 for v in t.data])):
            with self.assertRaisesRegex(cache.PrefixError,'REJECT_SOURCE_COPY_MISMATCH'): self.snapshot()

    def test_dynamic_container_fresh_on_every_request(self):
        self.stack.enter_context(patch.dict('sys.modules',{'transformers.cache_utils':NS(DynamicCache=FakeDynamic)}))
        s=self.snapshot(FakeDynamic(((FakeTensor(),FakeTensor()),)))
        a=s.materialize(identity(),[1,2,3,4],position_start=3,mask=[1]*4)
        a.pairs=((FakeTensor(4),FakeTensor(4)),)
        b=s.materialize(identity(),[1,2,3,4],position_start=3,mask=[1]*4)
        self.assertIsNot(a,b); self.assertEqual(b.get_seq_length(),3); s.check()
        with self.assertRaises(cache.PrefixError): cache.cache_pairs(object())
        with patch.object(FakeDynamic,'get_seq_length',return_value=99):
            with self.assertRaisesRegex(cache.PrefixError,'REJECT_CACHE_LENGTH'):
                self.snapshot(FakeDynamic(((FakeTensor(),FakeTensor()),)))


class APITests(unittest.TestCase):
    def test_signature_capabilities_and_missing_contract(self):
        def legacy(input_ids=None,attention_mask=None,past_key_values=None,use_cache=True,return_dict=True,output_attentions=True): pass
        def modern(input_ids=None,attention_mask=None,past_key_values=None,use_cache=True,return_dict=True,output_attentions=True,position_ids=None,cache_position=None,logits_to_keep=0): pass
        pos=NS(offset=2,forward=lambda attention_mask,past_key_values_length=0:None)
        for method,optional in [(legacy,[]),(modern,['position_ids','cache_position'])]:
            api=cache.OPTCacheAPI(NS(forward=method),NS(forward=method,embed_positions=pos))
            self.assertEqual(api.optional,optional)
        with self.assertRaisesRegex(cache.PrefixError,'UNSUPPORTED_API'):
            cache.OPTCacheAPI(NS(forward=lambda input_ids:None),NS(forward=legacy,embed_positions=pos))


try:
    import torch
except ImportError: torch=None


@unittest.skipIf(torch is None,'Torch unavailable: actual KV cloning/storage tests NOT EXECUTED')
class TensorCacheTests(unittest.TestCase):
    def exercise(self,device):
        a=replace(identity(),device=device)
        source=tuple((torch.randn(1,2,3,4,device=device),torch.randn(1,2,3,4,device=device)) for _ in range(2))
        s=cache.PrefixSnapshot(source,a,[1,2,3],layers=2,heads=2,head_dim=4,source_path='PREFIX_SOURCE_PREFILL')
        for _ in range(3):
            request=s.materialize(a,[1,2,3,4],position_start=3,mask=[1]*4)
            self.assertEqual(s.inspect_request(request)['status'],'INDEPENDENT_EQUAL')
            request[0][0].add_(5); s.check()
        s.pairs[0][0].data.add_(1)
        with self.assertRaises(cache.PrefixError): s.check()
    def test_cpu(self): self.exercise('cpu')
    def test_cuda(self):
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        self.exercise('cuda:0')

if __name__=='__main__': unittest.main()
