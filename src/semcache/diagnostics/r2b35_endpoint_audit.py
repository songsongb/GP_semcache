"""Untimed evidence for real one-token head, active adapters and unchanged decoder."""
from contextlib import contextmanager
from semcache.diagnostics.r2b1_prefix_cache import PrefixError
from semcache.diagnostics.r2b2_benchmark import exact_cache
from semcache.diagnostics.r2b3_benchmark import strict_metrics


def validate_shapes(heads,decoder_calls,query,hidden,vocab,matched):
    expected=1 if matched else query
    if decoder_calls!=1 or len(heads)!=1 or heads[0]['input']!=[1,expected,hidden] or heads[0]['output']!=[1,expected,vocab]:
        raise PrefixError('OUTPUT_MATCHED_INVALID','need exactly one decoder and one head call, with actual expected shapes')
    return dict(status='OUTPUT_SHAPE_MATCHED' if matched else 'LEGACY_ALL_LOGITS',head_calls=1,decoder_calls=1,head=heads[0],
        hidden_all_token_head_calls=0 if matched else None)


class EndpointAudit:
    def __init__(self,runtime,mode): self.r=runtime; self.mode=mode; self.heads=[]; self.decoder_calls=0; self.lora=[]; self.last_hidden=None
    @contextmanager
    def hooks(self):
        import torch
        handles=[]; pending=[]
        def before_head(m,args):
            x=args[0]; pending.append(dict(input=list(x.shape),input_stride=list(x.stride()),input_dtype=str(x.dtype),input_device=str(x.device)))
            self.last_hidden=x[:,-1:,:].detach().cpu().clone()
        def after_head(m,args,out): self.heads.append(dict(pending.pop(),output=list(out.shape),dtype=str(out.dtype),device=str(out.device)))
        def decoder(m,args,kwargs): self.decoder_calls+=1
        try:
            handles.append(self.r.matched_decoder.register_forward_pre_hook(decoder,with_kwargs=True))
            handles.append(self.r.matched_head.register_forward_pre_hook(before_head))
            handles.append(self.r.matched_head.register_forward_hook(after_head))
            for layer in range(len(self.r.adapter.layers)):
                for role,module in self.r.adapter.projection_modules(layer).items():
                    if module.disable_adapters or module.merged or list(module.active_adapters)!=[self.r.a.adapter]:
                        raise PrefixError('OUTPUT_MATCHED_INVALID','LoRA disabled/merged/wrong adapter')
                    def observed(m,args,out,i=layer,r=role,owner=module):
                        self.lora.append(dict(layer=i,role=r,adapter=self.r.a.adapter,shape=list(out.shape),
                            scaling=owner.scaling[self.r.a.adapter],delta_nonzero=bool(torch.count_nonzero(out)),
                            module_identity=id(owner),lora_b_identity=id(m)))
                    handles.append(module.lora_B[self.r.a.adapter].register_forward_hook(observed))
            yield self
        finally:
            for h in handles: h.remove()
    def report(self):
        matched=self.mode.startswith('MATCHED'); query=self.r.n if self.mode.endswith('FULL') else self.r.n-self.r.p
        result=validate_shapes(self.heads,self.decoder_calls,query,int(self.r.matched_head.weight.shape[1]),self.r.model.config.vocab_size,matched)
        expected={(i,r) for i in range(len(self.r.adapter.layers)) for r in 'qkv'}
        if len(self.lora)!=len(expected) or {(v['layer'],v['role']) for v in self.lora}!=expected:
            raise PrefixError('OUTPUT_MATCHED_INVALID','actual active LoRA contribution calls missing/duplicated')
        if any(v['shape'][1]!=query for v in self.lora): raise PrefixError('OUTPUT_MATCHED_INVALID','LoRA processed wrong query count')
        result.update(lora=self.lora,active_adapter=self.r.a.adapter,query_tokens=query)
        return result


def metrics(reference,actual):
    result=strict_metrics(reference,actual)
    ref=reference.double(); diff=(actual.double()-ref).abs()
    meaningful=ref.abs()>1e-12
    result.update(mean_abs_error=float(diff.mean()),max_abs_error=float(diff.max()),
        max_relative_error=float((diff[meaningful]/ref.abs()[meaningful]).max()) if bool(meaningful.any()) else None,
        relative_error_definition='abs(diff)/abs(reference) where abs(reference)>1e-12')
    return result


def gate(runtime):
    import torch
    from semcache.diagnostics.r2b35_executor import MODES
    runtime.check(); outputs={}; audits={}; hidden={}
    for mode in MODES:
        observer=EndpointAudit(runtime,mode)
        with observer.hooks(): output=runtime.execute(mode,audit=True)
        audits[mode]=dict(endpoint=observer.report(),physical=output['audit'],
            prefix_check=output['output_prefix_check'],incoming_cache=output['prepared_input_audit'])
        hidden[mode]=observer.last_hidden; outputs[mode]=output
        runtime.check()
    axis_a={}
    for kind in ('FULL','WARM_NORMAL'):
        legacy=outputs['LEGACY_'+kind]; matched=outputs['MATCHED_'+kind]
        same_hidden=torch.equal(hidden['LEGACY_'+kind],hidden['MATCHED_'+kind])
        same_kv=exact_cache(legacy['cache'],matched['cache'])
        if not same_hidden or not same_kv:
            raise PrefixError('OUTPUT_MATCHED_INVALID','native/matched decoder hidden or final KV differs; not attributable to head shape')
        q=metrics(legacy['logits'][:,-1:,:],matched['logits'])
        axis_a[kind]=dict(metrics=q,final_kv_exact=same_kv,decoder_last_hidden_exact=same_hidden,
            status='NATIVE_ENDPOINT_EQUIVALENCE_PASS' if q['allclose'] else 'NATIVE_ENDPOINT_EQUIVALENCE_FAIL',
            adapter_or_decoder_mismatch_suspected=False,head_shape_numerical_difference_observed=q['max_abs_error']>0)
    matched=metrics(outputs['MATCHED_FULL']['logits'],outputs['MATCHED_WARM_NORMAL']['logits'])
    suffix=metrics(outputs['LEGACY_FULL']['logits'][:,runtime.p:,:],outputs['LEGACY_WARM_NORMAL']['logits'])
    last=metrics(outputs['LEGACY_FULL']['logits'][:,-1:,:],outputs['LEGACY_WARM_NORMAL']['logits'][:,-1:,:])
    return dict(output_matching='OUTPUT_MATCHED_VALID',audits=audits,axis_a=axis_a,
        native_endpoint_equivalent=all(v['metrics']['allclose'] for v in axis_a.values()),
        axis_b=dict(matched=matched,legacy_suffix=suffix,legacy_last=last,
            status='FULL_WARM_STRICT_PASS' if matched['allclose'] else 'FULL_WARM_STRICT_FAIL'),
        canonical=runtime.check(),correctness_qualified_speedup=False)
