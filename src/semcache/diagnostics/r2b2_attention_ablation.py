"""Native endpoint capability and explicit eager attention-output contracts."""
import hashlib
import inspect
from semcache.diagnostics.r2b1_prefix_cache import PrefixError


def endpoint_policy(options,requested):
    if requested not in ('b1','auto','native-last'): raise ValueError(requested)
    option=next((x for x in ('logits_to_keep','num_logits_to_keep') if x in options),None)
    if requested=='native-last' and option is None:
        raise PrefixError('UNSUPPORTED_ENDPOINT','native OPT signature has no explicit last-token option')
    last=requested!='b1' and option is not None
    return dict(requested=requested,effective='native_last' if last else 'b1_all_logits',
        native_option=option if last else None,output_matched=last,
        materialization_confounded=not last,
        reason='explicit native option; output shape and last-logit parity must be verified' if last else
        'Full L versus Warm S logits; not output-matched serving; no custom LM head')


class B2API:
    def __init__(self,original,arm,endpoint):
        if arm not in ('on','off'): raise ValueError(arm)
        self.original=original; self.arm=arm; self.endpoint=endpoint
    def __getattr__(self,name): return getattr(self.original,name)
    def kwargs(self,*args,**kwargs):
        out=self.original.kwargs(*args,**kwargs)
        out['output_attentions']=self.arm=='on'
        if self.endpoint['native_option']: out[self.endpoint['native_option']]=1
        return out


def backend_audit(runtime):
    effective=getattr(runtime.model.config,'_attn_implementation',None)
    modules=[]
    for layer in runtime.adapter.layers:
        attention=layer.self_attn
        name=type(attention).__name__
        configured=getattr(getattr(attention,'config',runtime.model.config),'_attn_implementation',effective)
        # Older OPT uses an eager-specific class, newer OPT dispatches via config.
        if effective!='eager' or configured!='eager' or 'sdpa' in name.lower() or 'flash' in name.lower():
            raise PrefixError('BACKEND_CONFOUNDED',f'expected eager, observed {effective}/{configured}/{name}')
        source=inspect.getsource(type(attention).forward)
        if name!='OPTAttention' or not any(x in source for x in ('torch.bmm','eager_attention_forward')):
            raise PrefixError('UNVERIFIED_BACKEND','unsupported attention implementation; inspect installed source')
        modules.append(dict(class_name=name,config=configured,forward_sha256=hashlib.sha256(source.encode()).hexdigest()))
    return dict(requested='eager',effective=effective,fallback=False,status='EAGER_SOURCE_AND_CONFIG_VERIFIED',
        layers=modules,output_attentions=runtime.arm=='on',
        limitation='OFF suppresses returned weights; eager may still compute probabilities internally; no skipped-attention claim')


def validate_attention_output(attentions,arm,layers,heads,query,total):
    if arm=='off':
        if attentions is not None: raise PrefixError('REJECT_ATTENTION_OUTPUT','OFF unexpectedly returns attention weights')
        return dict(returned=False,shapes=[],internal_probabilities='NOT_INFERRED_FROM_OUTPUT')
    if attentions is None or len(attentions)!=layers or any(tuple(a.shape)!=(1,heads,query,total) for a in attentions):
        raise PrefixError('MISSING_FORWARD_EVIDENCE','ON output attention shape mismatch')
    return dict(returned=True,shapes=[list(a.shape) for a in attentions],internal_probabilities='RETURNED')
