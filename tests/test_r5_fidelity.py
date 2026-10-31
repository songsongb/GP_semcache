"""CPU alignment, numerical and adversarial evidence tests (no ML dependencies)."""
from copy import deepcopy
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

from semcache.edgelora.communication import digest, digest_bytes
from semcache.edgelora.r3a_artifacts import pack
from semcache.edgelora.r4c_session import context
from semcache.edgelora.r4c_workload import policy
from semcache.edgelora.r5_fidelity import align, compare, compare_bytes, first_divergence, select, tensor_blob
from semcache.edgelora.r5_fidelity_analysis import compare_steps, indexed, reconstruction, validate_masked_scores, validate_observations
from semcache.edgelora.r5_fidelity_capture import PassiveCapture, descriptor, local_assignments

POLICY=policy(dict(bos_token_id=0,eos_token_id=6,pad_token_id=0),'fixed',4)
COHORT='a'*64
GRANT=dict(prefix_length=2,entry_id='b'*64,source_request_id='source',proof_sha256='c'*64)


def full(tokens=(1,2,3,4),step=0,generated=5):
    return context('full',step,list(tokens) if not step else [generated],list(tokens),0 if not step else len(tokens)+step-1,
                   POLICY,'NO_REUSE',COHORT)


def hit(step=0,generated=5):
    return context('hit',step,[3,4] if not step else [generated],[1,2,3,4],2 if not step else 4+step-1,
                   POLICY,'PARTIAL_PREFIX_HIT',COHORT,GRANT)


class Alignment(unittest.TestCase):
    def test_source_prefix_and_suffix_positions(self):
        self.assertEqual(align(full((1,2)),full(),kind='source_prefix',prefix=2),((0,2),(0,2)))
        self.assertEqual(align(full(),hit(),kind='matched',prefix=2),((2,4),(0,2)))

    def test_wrong_source_tokenization_or_boundary_rejected(self):
        for source,p in ((full((2,1)),2),(full((1,2)),3),(full((1,2,3)),2)):
            with self.assertRaises(ValueError):align(source,full(),kind='source_prefix',prefix=p)

    def test_wrong_positions_cache_positions_masks_rejected(self):
        for field,value in (('positions',[0,1]),('cache_positions',[0,1]),('attention_mask',[1,1,0,1])):
            b=hit();b[field]=value
            with self.assertRaises(ValueError):align(full(),b,kind='matched',prefix=2)

    def test_decode_only_while_actual_histories_align(self):
        self.assertEqual(align(full(step=1),hit(step=1),kind='matched',prefix=2,history_a=[5],history_b=[5]),((0,1),(0,1)))
        for a,b in (([5],[4]),([],[5])):
            with self.assertRaisesRegex(ValueError,'histories'):align(full(step=1),hit(step=1),kind='matched',prefix=2,history_a=a,history_b=b)
        with self.assertRaises(ValueError):align(full(step=1),hit(step=1,generated=4),kind='matched',prefix=2,history_a=[5],history_b=[5])

    def test_different_step_and_policy_rejected(self):
        with self.assertRaises(ValueError):align(full(),hit(step=1),kind='matched',prefix=2)
        b=hit();b['policy']=policy(dict(bos_token_id=0,eos_token_id=6,pad_token_id=0),'fixed',3)
        with self.assertRaises(ValueError):align(full(),b,kind='matched',prefix=2)


class NumericalEvidence(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.count=0
    def tearDown(self):self.tmp.cleanup()
    def item(self,values,shape,**kwargs):
        self.count+=1;name=f'{self.count}.bin';blob=pack(values,'float32');(self.root/name).write_bytes(blob)
        return descriptor(dict(file=name,shape=shape,dtype='float32',device='cpu',sha256=digest_bytes(blob)),**kwargs)

    def test_known_absolute_relative_cosine_and_hash_results(self):
        a=pack([3,4],'float32');b=pack([3,0],'float32');m=compare_bytes(a,b,'float32')
        self.assertEqual(m['max_abs'],4);self.assertEqual(m['relative_l2'],.8);self.assertEqual(m['cosine_similarity'],.6)
        self.assertFalse(m['bitwise_equal']);self.assertEqual(m['differing_values'],1);self.assertTrue(m['finite'])

    def test_zero_norm_and_signed_zero_bitwise_difference(self):
        m=compare_bytes(pack([0.],'float32'),pack([-0.],'float32'),'float32')
        self.assertIsNone(m['cosine_similarity']);self.assertEqual(m['max_abs'],0);self.assertFalse(m['bitwise_equal'])
        first=first_divergence([dict(step=0,operation='L0.raw_k',metrics=m)])
        self.assertEqual(first['type'],'BIT_PATTERN_ONLY')

    def test_nonfinite_misaligned_and_wrong_sized_values_rejected(self):
        for blob in (struct.pack('<f',float('nan')),struct.pack('<f',float('inf')),b'x'):
            with self.assertRaises(ValueError):compare_bytes(blob,blob,'float32')
        with self.assertRaises(ValueError):compare_bytes(b'',b'','float16')

    def test_actual_masked_negative_infinity_is_preserved_and_scoped(self):
        a=struct.pack('<4f',0,float('-inf'),1,2);b=struct.pack('<4f',0,-65504,1,2)
        m=compare_bytes(a,a,'float32',allow_negative_infinity=True)
        self.assertTrue(m['bitwise_equal']);self.assertFalse(m['finite']);self.assertEqual(m['max_abs'],0)
        mismatch=compare_bytes(a,b,'float32',allow_negative_infinity=True)
        self.assertIsNone(mismatch['max_abs']);self.assertEqual(mismatch['nonfinite_pattern_mismatches'],1)
        path=self.root/'masked.bin';path.write_bytes(a)
        item=descriptor(dict(file=path.name,shape=[1,2,2],dtype='float32',device='cpu',sha256=digest_bytes(a),allow_negative_infinity=True),
                        layout='HQK',token_axis=1,key_axis=2)
        step=dict(context=full((1,2)),tensors={'L0.attention_masked_scores':item})
        validate_masked_scores(self.root,step)
        bad=struct.pack('<4f',float('-inf'),0,1,2);path.write_bytes(bad);item['sha256']=digest_bytes(bad)
        with self.assertRaisesRegex(ValueError,'outside future'):validate_masked_scores(self.root,step)

    def test_head_layout_suffix_and_query_key_alignment(self):
        a=self.item(list(range(16)),[1,2,4,2],layout='BHTD',token_axis=2)
        blob,shape=select(self.root,a,(2,4));self.assertEqual(shape,[1,2,2,2])
        self.assertEqual(blob,pack([4,5,6,7,12,13,14,15],'float32'))
        scores=self.item(list(range(16)),[1,4,4],layout='HQK',token_axis=1,key_axis=2)
        blob,shape=select(self.root,scores,(0,2),2)
        self.assertEqual(shape,[1,2,2]);self.assertEqual(blob,pack([0,1,4,5],'float32'))

    def test_incompatible_shape_dtype_layout_and_position_ranges(self):
        a=self.item([1,2],[1,2,1]);b=self.item([1,2],[1,1,2])
        with self.assertRaisesRegex(ValueError,'shapes'):compare(self.root,a,self.root,b)
        for field,value in (('dtype','float16'),('layout','BHTD'),('token_axis',2)):
            other=dict(a,**{field:value})
            with self.assertRaises(ValueError):compare(self.root,a,self.root,other)
        with self.assertRaises(ValueError):select(self.root,a,(1,3))

    def test_injected_prefix_and_suffix_perturbation_localizes_correct_layer(self):
        # Different final logits alone are insufficient: an earlier captured Q
        # perturbation must determine the first-divergence record.
        a=self.item([1,2,3,4],[1,4,1]);b=self.item([3,4.25],[1,2,1])
        same=self.item([3,4],[1,2,1])
        left=dict(context=full(),step=0,tensors={'L2.raw_q':a,'L10.layer_output':a})
        right=dict(context=hit(),step=0,tensors={'L2.raw_q':b,'L10.layer_output':same})
        records=compare_steps(self.root,left,self.root,right,kind='matched',prefix=2)
        first=first_divergence(records);self.assertEqual(first['layer'],2);self.assertEqual(first['operation'],'L2.raw_q')
        self.assertEqual(first['metrics']['max_abs'],.25)
        self.assertEqual(first_divergence([r for r in records if r['operation']=='L10.layer_output'])['status'],'NO_OBSERVED_DIVERGENCE')

    def test_parallel_branches_recorded_as_peers(self):
        m=compare_bytes(pack([1],'float32'),pack([2],'float32'),'float32')
        records=[dict(step=0,operation=op,metrics=m) for op in ('L12.layer_input','L2.delta_q','L2.base_q')]
        first=first_divergence(records);self.assertEqual(first['layer'],2)
        self.assertEqual(set(first['causal_peers']),{'L2.delta_q','L2.base_q'})

    def test_corrupted_missing_and_escaped_evidence_fails(self):
        t=self.item([1,2],[1,2,1]);path=self.root/t['file'];path.write_bytes(pack([1,3],'float32'))
        with self.assertRaisesRegex(ValueError,'corrupted'):tensor_blob(self.root,t)
        path.unlink()
        with self.assertRaises(FileNotFoundError):tensor_blob(self.root,t)
        with self.assertRaisesRegex(ValueError,'escape'):tensor_blob(self.root,dict(t,file='../outside.bin'))

    def test_duplicate_indices_and_incorrect_layer_indices_fail(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):indexed([dict(i=0),dict(i=0)],('i',))
        t=self.item([1,2],[1,2,1]);kv=self.item([1,2],[1,1,2,1],layout='BHTD',token_axis=2)
        tensors={f'L0.{op}':kv if op.startswith('attention_kv') else t for op in
                 ('layer_input','layer_output','raw_q','raw_k','raw_v','attention_kv_k','attention_kv_v')}
        step=dict(tensors=tensors,context=full((1,2)),actual_layer_context={})
        validate_observations(step,1,True)
        bad=deepcopy(step);bad['tensors']['L3.raw_q']=bad['tensors'].pop('L0.raw_q')
        with self.assertRaises(ValueError):validate_observations(bad,1,True)

    def test_actual_mask_causal_perturbation_detected(self):
        t=self.item([1,2],[1,2,1]);kv=self.item([1,2],[1,1,2,1],layout='BHTD',token_axis=2)
        tensors={f'L0.{op}':kv if op.startswith('attention_kv') else t for op in
                 ('layer_input','layer_output','raw_q','raw_k','raw_v','attention_kv_k','attention_kv_v')}
        step=dict(tensors=tensors,context=full((1,2)),actual_layer_context={'0':{'attention_mask':dict(shape=[1,1,2,2],values=[0,-65504,0,0])}})
        validate_observations(step,1,True)
        step['actual_layer_context']['0']['attention_mask']['values'][1]=0
        with self.assertRaisesRegex(ValueError,'causal'):validate_observations(step,1,True)

    def test_corrupted_derived_kv_detected_even_with_valid_file_hash(self):
        raw=self.item([1,2,3,4,5,6,7,8],[1,2,4])
        transformed=self.item([1,2,5,6,3,4,7,8],[1,2,2,2],layout='BHTD',token_axis=2)
        tensors={f'L0.raw_{r}':raw for r in 'kv'}
        tensors.update({f'L0.attention_kv_{r}':transformed for r in 'kv'})
        source=dict(request_id='source',tensors=tensors)
        target=dict(tensors={f'L0.input_past_{r}':transformed for r in 'kv'})
        proof=dict(test='proof');grant=dict(source_request_id='source',prefix_length=2,proof_sha256=digest(proof))
        evidence=dict(grant=grant,source_proof=proof,proof_sha256=digest(proof),tensors={})
        for r in 'kv':
            for kind in ('cached_raw','derived','reconstructed'):
                evidence['tensors'][f'L0.{kind}_{r}']=raw if kind=='cached_raw' else transformed
        records=reconstruction(self.root,source,target,self.root,source,evidence,1,2,4)
        self.assertTrue(all(r['metrics']['bitwise_equal'] for r in records))
        evidence['tensors']['L0.reconstructed_k']=self.item([1,2,5,6,3,4,7,9],[1,2,2,2],layout='BHTD',token_axis=2)
        with self.assertRaisesRegex(ValueError,'corrupted cached'):reconstruction(self.root,source,target,self.root,source,evidence,1,2,4)


class AssignmentSites(unittest.TestCase):
    def test_installed_source_sites_are_discovered_from_assignments(self):
        # Source discovery only; these operations are never executed in a fixture.
        def attention(self):
            attn_weights = torch.bmm(query_states,key_states)
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.nn.functional.softmax(attn_weights,dim=-1)
            return attn_weights
        sites=local_assignments(attention,True)
        self.assertEqual([s[2] for _,s in sorted(sites.items())],['attention_scores','attention_masked_scores','attention_probabilities'])

    def test_passive_hooks_share_code_but_preserve_layer_indices_and_cleanup(self):
        from semcache.edgelora.r3b_fixture import ArrayTensor
        from semcache.edgelora.r4c_evidence import Evidence
        class Tensor(ArrayTensor):
            def reshape(self,*shape):
                shape=list(shape)
                if -1 in shape:
                    index=shape.index(-1);known=1
                    for n in shape:
                        if n!=-1:known*=n
                    shape[index]=len(self.values)//known
                return Tensor(self.values,shape)
            def __add__(self,other):return Tensor([a+b for a,b in zip(self.values,other.values)],self.shape)
        def bmm(a,b):return Tensor([1,2,3,4],[1,2,2])
        def softmax(a,dim):return Tensor([.25,.75,.5,.5],[1,2,2])
        class Module:
            def __init__(self):self.pre=[];self.post=[]
            def register(self,collection,hook):
                collection.append(hook)
                return SimpleNamespace(remove=lambda:collection.remove(hook))
            def register_forward_pre_hook(self,hook,with_kwargs):return self.register(self.pre,hook)
            def register_forward_hook(self,hook,with_kwargs):return self.register(self.post,hook)
            def forward(self,hidden_states):return hidden_states
            def __call__(self,x):
                for hook in self.pre:self_unused=hook(self,(x,),{});assert self_unused is None
                out=self.forward(x)
                for hook in self.post:self_unused=hook(self,(x,),{},out);assert self_unused is None
                return out
        class Attention(Module):
            def __init__(self):
                super().__init__();self.roles={r:Module() for r in 'qkv'};self.out_proj=Module();self.num_heads=1
            def forward(self,hidden_states,attention_mask=None):
                key_states=hidden_states;value_states=hidden_states
                attn_weights = bmm(hidden_states,key_states)
                causal_mask=bmm(hidden_states,key_states)
                attn_weights = attn_weights + causal_mask
                attn_weights = softmax(attn_weights,dim=-1)
                for module in self.roles.values():module(hidden_states)
                return self.out_proj(hidden_states),None,None
        class Layer(Module):
            def __init__(self):
                super().__init__();self.self_attn=Attention();self.self_attn_layer_norm=Module()
                self.final_layer_norm=Module();self.fc1=Module();self.fc2=Module()
            def forward(self,hidden_states):
                residual=hidden_states
                hidden_states=self.self_attn_layer_norm(hidden_states)
                hidden_states=self.self_attn(hidden_states)[0]
                hidden_states = residual + hidden_states
                residual=hidden_states
                hidden_states=self.final_layer_norm(hidden_states)
                hidden_states=self.fc2(self.fc1(hidden_states))
                hidden_states = residual + hidden_states
                return (hidden_states,)
        layers=[Layer(),Layer()];adapter=SimpleNamespace(layers=layers,projection_modules=lambda i:layers[i].self_attn.roles)
        with tempfile.TemporaryDirectory() as d:
            head=Module();driver=SimpleNamespace(identity={'hidden_size':1},heads=1,model=SimpleNamespace(lm_head=head),evidence=Evidence(Path(d)))
            cap=PassiveCapture(driver,'fixture',2)
            with cap.hooks(adapter):
                for i,layer in enumerate(layers):
                    tensor=Tensor([i+1,i+2],[1,2,1]);result=layer(tensor)[0]
                    self.assertEqual(result.values,[4*(i+1),4*(i+2)]);head(result)
            self.assertNotEqual(cap.tensors['L0.layer_input']['sha256'],cap.tensors['L1.layer_input']['sha256'])
            self.assertEqual(cap.tensors['L0.raw_q']['sha256'],cap.tensors['L0.projection_input']['sha256'])
            for layer in range(2):
                for op in ('attention_scores','attention_masked_scores','attention_probabilities','attention_residual','ffn_residual','used_attention_kv_k','used_attention_kv_v'):
                    self.assertIn(f'L{layer}.{op}',cap.tensors)
            for layer in layers:
                for module in (layer,layer.self_attn,layer.fc1,layer.fc2,layer.self_attn_layer_norm,layer.final_layer_norm,*layer.self_attn.roles.values(),layer.self_attn.out_proj):
                    self.assertFalse(module.pre or module.post)
            self.assertFalse(head.post)


if __name__=='__main__':unittest.main()
