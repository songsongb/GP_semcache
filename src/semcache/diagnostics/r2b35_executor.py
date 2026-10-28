"""Opt-in last-hidden-state head, retaining the real PEFT wrapper and B2 timer."""
import ast
from contextlib import contextmanager
import hashlib
import inspect
import textwrap
from unittest.mock import patch
from semcache.diagnostics.r2b2_benchmark import B2Runtime
from semcache.diagnostics.r2b1_prefix_cache import PrefixError

MODES=('LEGACY_FULL','LEGACY_WARM_NORMAL','MATCHED_FULL','MATCHED_WARM_NORMAL')


def native_contract(source):
    """Fail closed on unrecognized output postprocessing or hidden-state transforms."""
    tree=ast.parse(textwrap.dedent(source))
    fn=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)))
    expected=ast.dump(ast.parse('self.lm_head(outputs[0]).contiguous()',mode='eval').body)
    logits=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='logits' for t in n.targets)]
    outputs=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='outputs' for t in n.targets)]
    head_calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='lm_head']
    if len(logits)!=1 or ast.dump(logits[0].value)!=expected or len(head_calls)!=1:
        raise PrefixError('OUTPUT_MATCHED_UNAVAILABLE','native head must be exactly lm_head(outputs[0]).contiguous(); no inferred casts/postprocessing')
    if len(outputs)!=1 or not isinstance(outputs[0].value,ast.Call) or ast.unparse(outputs[0].value.func)!='self.model.decoder':
        raise PrefixError('OUTPUT_MATCHED_UNAVAILABLE','unsupported native decoder/hidden postprocessing')
    # Reject further writes to outputs/logits (e.g. += or slice assignment).
    for node in ast.walk(fn):
        if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store) and node.id in ('logits','outputs'):
            if node not in [t for n in logits+outputs for t in n.targets]:
                raise PrefixError('OUTPUT_MATCHED_UNAVAILABLE','additional output mutation in native forward')
        if isinstance(node,(ast.Assign,ast.AugAssign)):
            for target in node.targets if isinstance(node,ast.Assign) else [node.target]:
                if not isinstance(target,ast.Name) and any(isinstance(n,ast.Name) and n.id in ('outputs','logits') for n in ast.walk(target)):
                    raise PrefixError('OUTPUT_MATCHED_UNAVAILABLE','hidden/output slice mutation')
    final=fn.body[-1]
    if not isinstance(final,ast.Return) or not isinstance(final.value,ast.Call) or ast.unparse(final.value.func)!='CausalLMOutputWithPast':
        raise PrefixError('OUTPUT_MATCHED_UNAVAILABLE','unknown native output construction')
    fields={kw.arg:ast.unparse(kw.value) for kw in final.value.keywords}
    expected_fields=dict(logits='logits',past_key_values='outputs.past_key_values',hidden_states='outputs.hidden_states',attentions='outputs.attentions')
    if any(fields.get(k)!=v for k,v in expected_fields.items()): raise PrefixError('OUTPUT_MATCHED_UNAVAILABLE','native output casts/transforms are unsupported')
    return dict(native_head='self.lm_head(outputs[0]).contiguous()',dtype_policy='native head dtype; no added cast',
        source_sha256=hashlib.sha256(source.encode()).hexdigest())


def capability(runtime):
    base=runtime.model.get_base_model()
    if base is not runtime.base or type(base).__name__!='OPTForCausalLM' or not hasattr(runtime.model,'peft_config'):
        raise PrefixError('OUTPUT_MATCHED_UNAVAILABLE','controlled PEFT OPTForCausalLM required')
    for config in runtime.model.peft_config.values():
        if getattr(config,'is_prompt_learning',False) or str(config.peft_type) not in ('PeftType.LORA','LORA') or set(config.target_modules)!=set(('q_proj','k_proj','v_proj')) or getattr(config,'modules_to_save',None):
            raise PrefixError('OUTPUT_MATCHED_UNAVAILABLE','only existing vanilla QKV-only LoRA fixture is verified')
    decoder=base.get_decoder(); head=base.get_output_embeddings()
    if decoder is not runtime.decoder or decoder is not base.model.decoder or head is not base.lm_head:
        raise PrefixError('OUTPUT_MATCHED_UNAVAILABLE','native decoder/head aliases do not match runtime')
    if type(head).__name__!='Linear' or hasattr(head,'lora_A'):
        raise PrefixError('OUTPUT_MATCHED_UNAVAILABLE','only verified fixture QKV LoRA and native Linear head supported')
    source=inspect.getsource(inspect.unwrap(type(base).forward)); contract=native_contract(source)
    from transformers.modeling_outputs import CausalLMOutputWithPast
    decoder_parameters=inspect.signature(decoder.forward).parameters
    required=('input_ids','attention_mask','past_key_values','use_cache','output_attentions','return_dict')
    if any(n not in decoder_parameters for n in required):
        raise PrefixError('OUTPUT_MATCHED_UNAVAILABLE','decoder keyword/cache contract missing')
    report=dict(status='CAPABILITY_DETECTED_PENDING_EXECUTION_AUDIT',**contract,
        native_class=type(base).__name__,peft_class=type(runtime.model).__name__,decoder_class=type(decoder).__name__,
        decoder_signature=str(inspect.signature(decoder.forward)),head_class=type(head).__name__,head_shape=list(head.weight.shape),
        head_dtype=str(head.weight.dtype),head_device=str(head.weight.device),
        decoder_is_native=True,head_is_native=True,head_adapter=False,
        peft_forward_source_sha256=hashlib.sha256(inspect.getsource(type(runtime.model).forward).encode()).hexdigest(),
        numerical_equivalence='UNVERIFIED_UNTIL_GATE',wrapper='actual PEFT model call retained')
    return report,decoder,head,set(decoder_parameters),CausalLMOutputWithPast


def decoder_then_head(decoder,head,output_factory,allowed,kwargs):
    """No all-logit call occurs here. Select hidden state BEFORE the actual head."""
    values=dict(kwargs)
    for name in ('labels','logits_to_keep','num_logits_to_keep'):
        value=values.pop(name,None)
        if value is not None and not (name!='labels' and value==0):
            raise PrefixError('OUTPUT_MATCHED_UNAVAILABLE','labels or nonzero native head option not supported')
    for name in list(values):
        if name not in allowed:
            if values[name] is not None: raise PrefixError('OUTPUT_MATCHED_UNAVAILABLE','unknown decoder argument '+name)
            del values[name]
    outputs=decoder(**values)
    hidden=outputs[0]
    if hidden.ndim!=3 or hidden.shape[0]!=1 or hidden.shape[1]<1:
        raise PrefixError('OUTPUT_MATCHED_INVALID','decoder hidden shape invalid')
    logits=head(hidden[:,-1:,:]).contiguous()
    return output_factory(logits=logits,past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,attentions=outputs.attentions)


class Runtime(B2Runtime):
    def setup_matched(self):
        self.matched_capability,self.matched_decoder,self.matched_head,self.decoder_parameters,self.output_factory=capability(self)
        self.binding=(id(self.base),id(self.matched_decoder),id(self.matched_head))
        return self.matched_capability

    @contextmanager
    def matched_scope(self):
        if self.binding!=(id(self.model.get_base_model()),id(self.base.get_decoder()),id(self.base.get_output_embeddings())):
            raise PrefixError('OUTPUT_MATCHED_INVALID','decoder/head reference changed')
        def forward(*args,**kwargs):
            if args: raise PrefixError('OUTPUT_MATCHED_UNAVAILABLE','unexpected positional native Forward arguments')
            return decoder_then_head(self.matched_decoder,self.matched_head,self.output_factory,self.decoder_parameters,kwargs)
        # Preserve B2 shape validation for a one-logit output without changing its
        # input builder. PEFT wrapper still invokes the real adapted native model.
        with patch.object(self.base,'forward',forward),patch.object(self,'endpoint',dict(self.endpoint,effective='native_last')):
            yield

    def execute(self,mode,audit=False):
        if mode not in MODES: raise ValueError(mode)
        path='FULL_USE_CACHE' if mode.endswith('FULL') else 'EXACT_PREFIX_WARM'
        if mode.startswith('LEGACY'): return super().forward(path,audit=audit)
        with self.matched_scope(): result=super().forward(path,audit=audit)
        result['endpoint']=dict(effective='matched_last_hidden_head',output_matched=True,
            implementation='actual PEFT wrapper, adapted native decoder and native Linear head')
        return result
