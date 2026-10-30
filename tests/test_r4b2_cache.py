"""Causal fixture proofs, owned Raw storage, exact matching and numerical guards."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from semcache.edgelora.communication import ProtocolError,digest
from semcache.edgelora.r3a_artifacts import load_matrices,pack,read,sha
from semcache.edgelora.r3a_tiny import f32,linear,prepare_tiny
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.r4a_fixture import TinyBackend
from semcache.edgelora.r4b2_analysis import tensor_metrics
from semcache.edgelora.r4b2_cache import PrefixCache
from semcache.edgelora.r4b2_kv import attention_kv,construct_past,describe_api,spec,verify_raw_kv
from semcache.edgelora.r4b2_workload import make_suite


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);base=prepare_tiny(Path(self.temp.name)/'base')
        self.pm=read(base/'r3a_prepare_manifest.json');self.prepared_sha=sha(base/'r3a_prepare_manifest.json')
        self.matrices=load_matrices(base,read(base/'r3a_adapter_manifest.json'),torch_backend=False)
        self.model=TinyBackend(read(base/'r3a_tiny_base.json'),self.matrices);self.cache=self.manager(2**20)
        self.prefix=[1,2,3,4]*8
    def manager(self,capacity):
        return PrefixCache(self.pm,self.prepared_sha,'TEST_FIXTURE_ONLY','cpu',capacity,fixture=True,
            tokenizer={'fixture':'integer'},sources={'fixture':'unit'},contract={'scope':'TEST_FIXTURE_ONLY'},observed_cache_type=tuple)
    def prompt(self,ids):return dict(token_ids=list(ids),attention_mask=[1]*len(ids))
    def computed(self,ids):
        blocks={};h=self.model.hidden
        def project(l,r,x,native):
            delta=linear(linear(x,self.matrices[l,r,'A'],h,8),self.matrices[l,r,'B'],8,h)
            values=[f32(a+b) for a,b in zip(native(),delta)]
            blocks.setdefault(l,{})[r]=ArrayTensor(values,[1,len(ids),h]);return values
        out=self.model.forward(ids,[1]*len(ids),project=project)
        blocks={l:tuple(b[r] for r in 'qkv') for l,b in blocks.items()}
        proof=verify_raw_kv(blocks,out.past_key_values,1,self.pm['identity'],'cpu');return out,blocks,proof
    def admit(self,ids,rid):
        out,blocks,proof=self.computed(ids)
        grant=self.cache.admit(self.prompt(ids),blocks,proof,rid,completed=True)
        self.assertIsNotNone(grant);return grant

    def test_one_source_two_suffixes_and_intervening_request_use_private_actual_kv(self):
        source=self.admit(self.prefix,'P_COLD')
        for suffix in ([2,3,4,5]*24,[5,4,3,2]*24):
            target=self.prefix+suffix;entry,grant,past=self.cache.find(self.prompt(target),'TARGET')
            self.assertEqual(grant['entry_id'],source['entry_id']);self.assertEqual(grant['source_request_id'],'P_COLD')
            self.assertFalse(self.cache.proofs[grant['entry_id']]['q_consumed'])
            cached_raw=spec(entry.tensors[0][1]);prefix_hash=spec(past[0][0]);self.assertEqual(prefix_hash['shape'],[1,1,32,4])
            out=self.model.forward(suffix,[1]*128,past);full=self.model.forward(target,[1]*128)
            self.assertEqual(out.logits,full.logits[-len(suffix)*self.model.vocab:])
            for token in (3,4,2):
                target.append(token);out=self.model.forward([token],[1]*len(target),out.past_key_values)
                full=self.model.forward(target,[1]*len(target));self.assertEqual(out.logits,full.logits[-self.model.vocab:])
            self.assertEqual(spec(entry.tensors[0][1]),cached_raw)
            self.model.forward([5]*128,[1]*128)  # Unrelated full request cannot rebind source provenance.
        self.assertEqual(len(self.cache.entries),1)
        self.assertEqual(self.cache.bytes,3*32*32*4*4)

    def test_longest_prefix_is_explicit_standalone_source_not_a_longer_source_slice(self):
        a=self.admit(self.prefix,'P32');b=self.admit(self.prefix*2,'P64')
        _,grant,_=self.cache.find(self.prompt(self.prefix*3+[2]*32),'TARGET')
        self.assertEqual(grant['entry_id'],b['entry_id']);self.assertEqual(grant['prefix_length'],64)
        with self.assertRaises(ProtocolError):self.cache.find(self.prompt(self.prefix*3+[2]*32),'TARGET',expected_length=32)
        with self.assertRaises(ValueError):self.cache.find(self.prompt(self.prefix),'NO_EMPTY_SUFFIX')
        self.assertEqual(len(self.cache.entries),2)

    def test_identity_mask_positions_offsets_and_unsupported_lengths_reject(self):
        self.admit(self.prefix,'P');target=self.prompt(self.prefix+[5]*96)
        changed=deepcopy(target);changed['token_ids'][12]+=1
        for ids in (changed['token_ids'],[5]+target['token_ids'][:-1],[5]*128):
            with self.assertRaises(ValueError):self.cache.find(self.prompt(ids),'BAD')
        with self.assertRaises(ProtocolError):self.cache.find(target,'BAD',cohort='0'*64)
        with self.assertRaises(ProtocolError):self.cache.find(target,'BAD',expected_length=16)
        with self.assertRaises(ProtocolError):self.cache.find(dict(target,attention_mask=[0]+[1]*127),'BAD')
        with self.assertRaises(ProtocolError):self.cache.find(dict(target,positions=[99]*128),'BAD')
        with self.assertRaises(ProtocolError):self.cache.find(dict(target,prompt_sha256='0'*64),'BAD')
        key=next(iter(self.cache.entries));self.cache.proofs[key]['prefix_length']=64
        with self.assertRaises(ProtocolError):self.cache.find(target,'CORRUPT_PROOF')

    def test_mutated_tokenizer_model_adapter_weights_device_and_source_namespace_never_hit(self):
        self.admit(self.prefix,'P');target=self.prompt(self.prefix+[5]*96)
        mutations=[(self.cache.tokenizer,'fixture','OTHER_TOKENIZER'),(self.cache.sources,'fixture','OTHER_SOURCE'),
            (self.cache._pm['identity'],'adapter','OTHER_ADAPTER'),(self.cache._pm['identity'],'adapter_matrix_sha256','0'*64),
            (self.cache._pm['identity'],'base_weight_sha256','1'*64),(self.cache._pm['identity'],'model','OTHER_MODEL')]
        for obj,key,value in mutations:
            saved=obj[key];obj[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ProtocolError,'namespace changed'):self.cache.find(target,'BAD_NAMESPACE')
            obj[key]=saved
        saved=self.cache.device;self.cache.device='cuda:99'
        with self.assertRaisesRegex(ProtocolError,'namespace changed'):self.cache.find(target,'BAD_DEVICE')
        self.cache.device=saved;self.assertIsNotNone(self.cache.find(target,'RESTORED')[0])

    def test_lru_eviction_releases_real_storage_and_denies_oversize_or_incomplete_source(self):
        self.cache=self.manager(2*3*32*32*4*4)
        a=self.admit(self.prefix,'A');b=self.admit([2,3,4,5]*8,'B')
        old_b=next(iter(self.cache.entries[b['entry_id']].pool.entries.values()))
        self.cache.find(self.prompt(self.prefix+[5]*96),'A_ACCESS');self.admit([3,4,5,1]*8,'C')
        self.assertIsNone(old_b.tensors);self.assertNotIn(b['entry_id'],self.cache.entries)
        with self.assertRaises(ValueError):self.cache.find(self.prompt([2,3,4,5]*8+[1]*96),'EVICTED')
        self.assertTrue(all(e['occupancy_bytes']<=self.cache.capacity_bytes for e in self.cache.events))
        _,blocks,proof=self.computed(self.prefix*3)
        self.assertIsNone(self.cache.admit(self.prompt(self.prefix*3),blocks,proof,'TOO_BIG',completed=True))
        with self.assertRaises(ProtocolError):self.cache.admit(self.prompt(self.prefix*3),blocks,proof,'INCOMPLETE',completed=False)
        with self.assertRaises(ProtocolError):self.cache.admit(self.prompt(self.prefix[:16]),blocks,proof,'UNSUPPORTED',completed=True)

    def test_corrupt_tensor_and_wrong_raw_to_actual_kv_proof_are_fatal(self):
        out,blocks,proof=self.computed(self.prefix);bad=deepcopy(proof);bad['layers'][0]['actual_k']['sha256']='0'*64
        with self.assertRaises(ProtocolError):self.cache.admit(self.prompt(self.prefix),blocks,bad,'BAD',completed=True)
        out.past_key_values[0][0].values[0]+=1.
        with self.assertRaisesRegex(ProtocolError,'ARCHITECTURAL_BLOCKER'):verify_raw_kv(blocks,out.past_key_values,1,self.pm['identity'],'cpu')
        grant=self.admit(self.prefix,'GOOD');entry=next(iter(self.cache.entries[grant['entry_id']].pool.entries.values()))
        entry.tensors[0][0].values[0]+=1.
        with self.assertRaises(ValueError):self.cache.find(self.prompt(self.prefix+[5]*96),'CORRUPTED')
        self.assertEqual(self.cache.bytes,0)

    def test_head_transform_unknown_cache_api_and_numerical_diagnostics(self):
        t=ArrayTensor(list(range(16)),[1,2,8]);blocks={0:(t,t,t)};pairs=attention_kv(blocks,2,8)
        self.assertEqual(pairs[0][0].values,[0,1,2,3,8,9,10,11,4,5,6,7,12,13,14,15])
        self.assertEqual(list(pairs[0][0].shape),[1,2,2,4])
        self.assertIsNot(pairs[0][0].values,t.values)
        with self.assertRaisesRegex(ProtocolError,'ARCHITECTURAL_BLOCKER'):describe_api(object())
        metric=tensor_metrics(pack([.01,1.,2.],'float32'),pack([0.,1.,2.],'float32'),'float32')
        self.assertEqual(metric['unequal_elements'],1);self.assertGreater(metric['max_abs'],0);self.assertGreater(metric['relative_l2'],0)


class WorkloadTests(unittest.TestCase):
    def test_actual_tokenizer_interface_checks_counts_boundary_and_suffix_difference(self):
        class Tokenizer:
            bos_token_id=0;eos_token_id=1;pad_token_id=2
            backend_tokenizer=type('UnitBackend',(),{'to_str':lambda self:'TEST_ONLY_BYTE_TOKENIZER'})()
            def __call__(self,text,**kwargs):
                assert kwargs==dict(add_special_tokens=False,truncation=False,padding=False)
                ids=[ord(c)+10 for c in text];return dict(input_ids=ids,attention_mask=[1]*len(ids))
            def decode(self,ids,**kwargs):return ''.join(chr(i-10) for i in ids)
        suite=make_suite(Tokenizer());self.assertEqual(len(suite['conditions']),3)
        for c in suite['conditions']:
            ps={p['sample_id']:p for p in suite['prompts']};p=ps[c['source']]['token_ids'];a,b=[ps[k]['token_ids'] for k in c['targets']]
            self.assertEqual(len(a),128);self.assertEqual(a[:len(p)],b[:len(p)]);self.assertNotEqual(a[len(p):],b[len(p):])
        self.assertEqual(make_suite(fixture=True,new_tokens=16)['policy']['max_new_tokens'],16)
        with self.assertRaises(ProtocolError):make_suite(fixture=True,prefix_lengths=(16,))


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),'actual HF OPT CPU NOT_RUN; dependencies absent, no download')
class ActualHFCPU(unittest.TestCase):
    def test_loaded_opt_raw_kv_public_factory_and_suffix_decode(self):
        import torch
        from transformers import OPTConfig,OPTForCausalLM
        torch.manual_seed(42);config=OPTConfig(vocab_size=67,hidden_size=16,word_embed_proj_dim=16,ffn_dim=32,num_hidden_layers=2,
            num_attention_heads=2,max_position_embeddings=160,dropout=0.,attention_dropout=0.,layerdrop=0.)
        config._attn_implementation='eager';model=OPTForCausalLM(config).eval();ids=[3+i%60 for i in range(128)];blocks={};handles=[]
        for l,layer in enumerate(model.model.decoder.layers):
            for role in 'qkv':
                def hook(module,args,out,l=l,role=role):blocks.setdefault(l,{})[role]=out.detach().clone()
                handles.append(getattr(layer.self_attn,role+'_proj').register_forward_hook(hook))
        with torch.inference_mode():source=model(input_ids=torch.tensor([ids[:64]]),attention_mask=torch.ones(1,64,dtype=torch.long),use_cache=True)
        for h in handles:h.remove()
        blocks={l:tuple(b[r] for r in 'qkv') for l,b in blocks.items()};identity=dict(layers=2,hidden_size=16,hidden_dtype='float32')
        proof=verify_raw_kv(blocks,source.past_key_values,2,identity,'cpu');past=construct_past(attention_kv(blocks,2,16),type(source.past_key_values),proof['cache_api'])
        with torch.inference_mode():
            partial=model(input_ids=torch.tensor([ids[64:]]),attention_mask=torch.ones(1,128,dtype=torch.long),past_key_values=past,use_cache=True)
            full=model(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,128,dtype=torch.long),use_cache=True)
        from semcache.edgelora.r3a_artifacts import check_parity
        metric=check_parity(partial.logits[0,-1].tolist(),full.logits[0,-1].tolist(),[1,1,67])
        self.assertTrue(metric['passed'])  # Original thresholds, never a fabricated strict bitwise qualification.
        for _ in range(2):
            token=int(partial.logits[0,-1].argmax());ids.append(token)
            with torch.inference_mode():
                partial=model(input_ids=torch.tensor([[token]]),attention_mask=torch.ones(1,len(ids),dtype=torch.long),
                    past_key_values=partial.past_key_values,use_cache=True)
                full=model(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long),use_cache=True)
            self.assertTrue(check_parity(partial.logits[0,-1].tolist(),full.logits[0,-1].tolist(),[1,1,67])['passed'])


if __name__=='__main__':unittest.main()
