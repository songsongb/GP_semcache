"""Decorated original-method tracing without Torch/GPU or inference substitution."""
from functools import wraps
import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semcache.edgelora.r3a_artifacts import pack
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.r4c_evidence import Evidence
from semcache.edgelora.r5_fidelity_capture import PassiveCapture, local_assignments, original_method


WRAPPER_CALLS=[]


def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Exactly the failing deprecation-wrapper shape: no local named self.
        WRAPPER_CALLS.append((wrapper.__code__,tuple(sys._getframe().f_locals)))
        return func(*args, **kwargs)
    return wrapper


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


def bmm(a,b):return Tensor([a.values[0],0,a.values[0],a.values[1]],[1,2,2])
def softmax(a,dim):return Tensor([1,0,.5,.5],[1,2,2])


class Module:
    def __init__(self):self.pre=[];self.post=[]
    def register(self,collection,hook):
        collection.append(hook)
        return SimpleNamespace(remove=lambda:collection.remove(hook))
    def register_forward_pre_hook(self,hook,with_kwargs):return self.register(self.pre,hook)
    def register_forward_hook(self,hook,with_kwargs):return self.register(self.post,hook)
    def forward(self,hidden_states):return hidden_states
    def __call__(self,x):
        for hook in self.pre:assert hook(self,(x,),{}) is None
        out=self.forward(x)
        for hook in self.post:assert hook(self,(x,),{},out) is None
        return out


class Attention(Module):
    def __init__(self):
        super().__init__();self.roles={r:Module() for r in 'qkv'};self.out_proj=Module();self.num_heads=1
        self.fail=False;self.omit_kv=False

    @decorator
    @decorator
    def forward(self,hidden_states,attention_mask=None):
        if self.fail:raise RuntimeError('injected original attention error')
        key_states=hidden_states;value_states=hidden_states
        attn_weights = bmm(hidden_states,key_states)
        causal_mask=Tensor([0,0,0,0],[1,2,2])
        attn_weights = attn_weights + causal_mask
        attn_weights = softmax(attn_weights,dim=-1)
        for module in self.roles.values():module(hidden_states)
        if self.omit_kv:key_states=None
        return self.out_proj(hidden_states),None,None


class Layer(Module):
    def __init__(self):
        super().__init__();self.self_attn=Attention();self.self_attn_layer_norm=Module()
        self.final_layer_norm=Module();self.fc1=Module();self.fc2=Module();self.child=None

    @decorator
    def forward(self,hidden_states):
        if self.child is not None:hidden_states=self.child(hidden_states)[0]
        residual=hidden_states
        hidden_states=self.self_attn_layer_norm(hidden_states)
        hidden_states=self.self_attn(hidden_states)[0]
        hidden_states = residual + hidden_states
        residual=hidden_states
        hidden_states=self.final_layer_norm(hidden_states)
        hidden_states=self.fc2(self.fc1(hidden_states))
        hidden_states = residual + hidden_states
        return (hidden_states,)


class DecoratedTracer(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);WRAPPER_CALLS.clear()
        self.layers=[Layer() for _ in range(32)];self.head=Module()
        self.adapter=SimpleNamespace(layers=self.layers,projection_modules=lambda i:self.layers[i].self_attn.roles)
        self.driver=SimpleNamespace(identity={'hidden_size':1},heads=1,model=SimpleNamespace(lm_head=self.head),evidence=Evidence(self.root))
        self.capture=PassiveCapture(self.driver,'decorated',2)
        self.trace_before=sys.gettrace();self.assertIsNone(self.trace_before)

    def tearDown(self):
        self.assertIs(sys.gettrace(),self.trace_before)
        self.assert_clean()
        self.tmp.cleanup()

    def assert_clean(self):
        for layer in self.layers:
            for module in (layer,layer.self_attn,layer.self_attn.out_proj,*layer.self_attn.roles.values(),
                           layer.self_attn_layer_norm,layer.final_layer_norm,layer.fc1,layer.fc2):
                self.assertFalse(module.pre or module.post)
        self.assertFalse(self.head.pre or self.head.post)

    def run_layers(self):
        outputs=[]
        for i,layer in enumerate(self.layers):
            result=layer(Tensor([i+1,i+2],[1,2,1]))[0];self.head(result);outputs.append(result.fixture_bytes())
        return outputs

    def tensor_bytes(self,key):return (self.root/self.capture.tensors[key]['file']).read_bytes()

    def test_decorated_original_code_matches_analyzed_assignment_sites(self):
        original=original_method(self.layers[0].self_attn.forward)
        self.assertIs(original,inspect.unwrap(Attention.forward))
        self.assertIsNot(original.__code__,Attention.forward.__code__)
        sites=local_assignments(self.layers[0].self_attn.forward,True)
        self.assertEqual(sites,local_assignments(original,True))
        executable_lines={line for _,_,line in original.__code__.co_lines() if line is not None}
        self.assertTrue(set(sites)<=executable_lines)
        self.assertEqual({s[2] for s in sites.values()},
                         {'attention_scores','attention_masked_scores','attention_probabilities'})

    def test_wrapper_without_self_original_captures_all_32_layers_and_unchanged_results(self):
        expected=self.run_layers();WRAPPER_CALLS.clear()
        with self.capture.hooks(self.adapter):actual=self.run_layers()
        self.assertEqual(actual,expected)
        self.assertEqual(len(WRAPPER_CALLS),32*3)
        self.assertTrue(all('self' not in names for _,names in WRAPPER_CALLS))
        for i in range(32):
            for op in ('attention_scores','attention_masked_scores','attention_probabilities','attention_residual',
                       'ffn_residual','used_attention_kv_k','used_attention_kv_v'):
                self.assertIn(f'L{i}.{op}',self.capture.tensors)
            self.assertEqual(self.tensor_bytes(f'L{i}.attention_scores'),pack([i+1,0,i+1,i+2],'float32'))
            self.assertEqual(self.tensor_bytes(f'L{i}.attention_residual'),pack([2*(i+1),2*(i+2)],'float32'))
            self.assertEqual(self.tensor_bytes(f'L{i}.ffn_residual'),expected[i])
            site=self.capture.sites[f'L{i}.attention_scores']
            self.assertEqual(site['file'],original_method(Attention.forward).__code__.co_filename)
            self.assertIn(site['line'],[s[0] for s in local_assignments(Attention.forward,True).values()])

    def test_nested_frames_sharing_code_retain_their_layer_and_residual_state(self):
        for a,b in zip(self.layers,self.layers[1:]):a.child=b
        expected=self.layers[0](Tensor([1,2],[1,2,1]))[0].fixture_bytes()
        with self.capture.hooks(self.adapter):
            result=self.layers[0](Tensor([1,2],[1,2,1]))[0];self.head(result)
        self.assertEqual(result.fixture_bytes(),expected)
        for i in range(32):
            factor=4**(31-i)
            self.assertEqual(self.tensor_bytes(f'L{i}.attention_residual'),pack([2*factor,4*factor],'float32'))
            self.assertEqual(self.tensor_bytes(f'L{i}.ffn_residual'),pack([4*factor,8*factor],'float32'))

    def test_original_exception_preserved_and_all_hooks_and_tracing_removed(self):
        self.layers[1].self_attn.fail=True
        with self.assertRaisesRegex(RuntimeError,'injected original attention error'):
            with self.capture.hooks(self.adapter):self.run_layers()
        self.assertIs(sys.gettrace(),self.trace_before);self.assert_clean()
        self.assertIn('L0.attention_probabilities',self.capture.tensors)
        self.assertNotIn('L1.used_attention_kv_k',self.capture.tensors)
        self.layers[1].self_attn.fail=False
        self.assertEqual(self.layers[1](Tensor([2,3],[1,2,1]))[0].values,[8,12])

    def test_trace_callback_failure_cleans_up_and_keeps_partial_evidence(self):
        self.layers[0].self_attn.omit_kv=True
        with self.assertRaisesRegex(ValueError,'FORENSIC_OPT_FRAME: missing required L0.key_states'):
            with self.capture.hooks(self.adapter):self.run_layers()
        self.assertIs(sys.gettrace(),self.trace_before);self.assert_clean()
        self.assertIn('L0.attention_scores',self.capture.tensors)

    def test_unexecuted_wrapped_original_fails_instead_of_skipping_required_observations(self):
        @wraps(original_method(Attention.forward))
        def bypass(*args,**kwargs):return args[1],None,None
        self.layers[0].self_attn.forward=bypass.__get__(self.layers[0].self_attn,Attention)
        with self.assertRaisesRegex(ValueError,'FORENSIC_OPT_FRAME: missing original OPT method calls/returns'):
            with self.capture.hooks(self.adapter):self.run_layers()

    def test_opaque_wrapper_without_original_identity_fails_informatively_and_cleans_up(self):
        def opaque(*args,**kwargs):return args[1],None,None
        self.layers[0].self_attn.forward=opaque.__get__(self.layers[0].self_attn,Attention)
        with self.assertRaisesRegex(ValueError,'FORENSIC_OPT_FRAME:.*no positional self'):
            with self.capture.hooks(self.adapter):self.run_layers()

    def test_unknown_instance_of_shared_original_method_fails_informatively(self):
        foreign=Layer()
        with self.assertRaisesRegex(ValueError,'FORENSIC_OPT_FRAME: cannot identify original-method self'):
            with self.capture.hooks(self.adapter):foreign(Tensor([1,2],[1,2,1]))

    def test_missing_analyzed_assignment_is_not_silently_ignored(self):
        real=local_assignments
        def undiscoverable(method,attention):return real(method,attention) if attention else {}
        with patch('semcache.edgelora.r5_fidelity_capture.local_assignments',side_effect=undiscoverable):
            with self.assertRaisesRegex(ValueError,'FORENSIC_OPT_FRAME: missing required tensor observations at L0'):
                with self.capture.hooks(self.adapter):self.run_layers()


if __name__=='__main__':unittest.main()
