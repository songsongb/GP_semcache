"""Read-only chronological OPT/PEFT hooks; bounded CPU reference, no tensor dumps."""
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
from semcache.diagnostics.r2b1_prefix_cache import PrefixError,cache_pairs,tensor_hash
from semcache.diagnostics.r1b3_fp16_divergence import tensor_metrics,logit_metrics
from semcache.diagnostics.r2b15_analysis import alignment,first_findings


def token_view(value,kind):
    if kind=='tokens' and value.ndim==2: return value.unsqueeze(0)  # batch1 flattened OPT FFN
    return value


def aligned_reference(reference,actual,kind,comparison,prefix,total):
    n=prefix if comparison=='FULL_PREFIX_vs_SOURCE' else total-prefix
    start=0 if comparison=='FULL_PREFIX_vs_SOURCE' else prefix
    end=start+n
    kv=prefix if comparison=='FULL_PREFIX_vs_SOURCE' else total
    alignment(tuple(reference.shape),kind,start,end,kv)
    ref=reference[:,start:end] if kind=='tokens' else reference[:,:,start:end,:kv]
    if tuple(ref.shape)!=tuple(actual.shape):
        raise PrefixError('REJECT_ALIGNMENT',f'aligned {tuple(ref.shape)} != actual {tuple(actual.shape)}; broadcasting forbidden')
    return ref,list(range(start,end))


def observed_metrics(ref,value,positions,kind):
    import torch
    # Shared metric uses token axis 1; attention uses query axis 2. Remap explicit
    # coordinates rather than passing a misleading head index as token position.
    if ref.dtype!=value.dtype: raise PrefixError('REJECT_ALIGNMENT','dtype differs across matched observations')
    result=tensor_metrics(ref,value,positions if kind=='tokens' else None)
    for name in ('first_nonzero','first_tolerance_failure'):
        point=result.get(name)
        if point and kind=='attention': point['token_index']=positions[point['index'][2]]
    # torch.equal treats +0 and -0 as equal; keep actual representation equality separate.
    a=ref.contiguous().view(torch.uint8); b=value.contiguous().view(torch.uint8)
    result['bitwise_equal']=bool(torch.equal(a,b))
    changed=(a!=b).nonzero()
    if len(changed):
        coords=[int(x) for x in changed[0]]; byte=coords[-1]%ref.element_size(); coords[-1]//=ref.element_size()
        result['first_bitwise_element']=dict(index=coords,byte_within_element=byte,token_index=positions[coords[1 if kind=='tokens' else 2]])
    else: result['first_bitwise_element']=None
    return result


class Recorder:
    def __init__(self,runtime,layers,*,reference=None,comparison=None,limit_mib=1024):
        self.r=runtime; self.layers=tuple(layers); self.reference=reference; self.comparison=comparison
        self.limit=limit_mib*1024**2; self.bytes=0; self.snapshots={}; self.metadata={}; self.rows=[]; self.order=0; self.seen=set()
        self.unsupported=['attention score before softmax: no stable native module boundary',
            'inline residual add: bracketed by layer/LN boundaries; no arithmetic replacement',
            'kernel identity: not collected']
    def put(self,key,value,kind='tokens'):
        native_shape=list(value.shape); native_stride=list(value.stride())
        value=token_view(value,kind)
        if key in self.seen: raise PrefixError('MISSING_NUMERIC_EVIDENCE','repeated hook '+str(key))
        self.seen.add(key); order=self.order; self.order+=1
        info=dict(shape=list(value.shape),stride=list(value.stride()),native_shape=native_shape,native_stride=native_stride,dtype=str(value.dtype),device=str(value.device))
        cpu=value.detach().to('cpu',copy=True)
        if self.reference is None:
            self.bytes+=cpu.numel()*cpu.element_size()
            if self.bytes>self.limit: raise MemoryError(f'CPU numeric reference budget {self.bytes}>{self.limit}; choose fewer --layers')
            self.snapshots[key]=cpu; self.metadata[key]=dict(info,kind=kind,execution_order=order)
        else:
            if key not in self.reference.snapshots: raise PrefixError('MISSING_NUMERIC_EVIDENCE','missing Full reference '+str(key))
            ref,positions=aligned_reference(self.reference.snapshots[key],cpu,kind,self.comparison,self.r.p,self.r.n)
            metric=observed_metrics(ref,cpu,positions,kind)
            self.rows.append(dict(comparison=self.comparison,layer=key[0],module=key[1],observation=key[2],execution_order=order,
                actual=info,full=self.reference.metadata[key],aligned_shape=list(ref.shape),absolute_token_positions=positions,
                coordinate_kind=kind,capture_scope='attention module return; not softmax execution timestamp' if kind=='attention' else 'module boundary',**metric))
    @contextmanager
    def hooks(self):
        handles=[]; expected=set()
        def install(module,key,input_=False,kind='tokens'):
            expected.add(key)
            if input_:
                def before(m,args,kwargs): self.put(key,args[0] if args else kwargs['hidden_states'],kind)
                handles.append(module.register_forward_pre_hook(before,with_kwargs=True))
            else:
                def after(m,args,out): self.put(key,out[0] if isinstance(out,tuple) else out,kind)
                handles.append(module.register_forward_hook(after))
        try:
            with ExitStack() as stack:
                d=self.r.decoder
                install(d.embed_tokens,(-1,'token_embedding','output')); install(d.embed_positions,(-1,'position_embedding','output'))
                for i in self.layers:
                    layer=self.r.adapter.layers[i]
                    install(layer,(i,'layer','input'),True); install(layer,(i,'layer','output'))
                    for name in ('self_attn_layer_norm','final_layer_norm'):
                        module=getattr(layer,name,None)
                        if module is None: self.unsupported.append(f'layer {i} {name} absent'); continue
                        install(module,(i,name,'input'),True); install(module,(i,name,'output'))
                    install(layer.self_attn,(i,'attention','input'),True); install(layer.self_attn,(i,'attention','output'))
                    probs=(i,'attention_probabilities','output'); expected.add(probs)
                    def attn(m,args,out,key=probs):
                        if not isinstance(out,tuple) or len(out)<2 or out[1] is None:
                            raise PrefixError('MISSING_NUMERIC_EVIDENCE','eager output attentions missing')
                        self.put(key,out[1],'attention')
                    handles.append(layer.self_attn.register_forward_hook(attn))
                    install(layer.self_attn.out_proj,(i,'attention_context','input'),True)
                    install(layer.self_attn.out_proj,(i,'attention_out_proj','output'))
                    for role,module in self.r.adapter.projection_modules(i).items():
                        install(module,(i,role+'_qkv','input'),True); install(module,(i,role+'_total','output'))
                        install(module.get_base_layer(),(i,role+'_base','output'))
                        delta=(i,role+'_lora_delta','output')
                        if hasattr(module,'lora_B') and self.r.a.adapter in module.lora_B:
                            expected.add(delta)
                            def save(m,args,out,key=delta,scale=module.scaling[self.r.a.adapter]): self.put(key,out*scale)
                            handles.append(module.lora_B[self.r.a.adapter].register_forward_hook(save))
                        else: self.unsupported.append(f'layer {i} {role} LoRA delta unavailable')
                    for name in ('fc1','fc2'):
                        module=getattr(layer,name); install(module,(i,name,'input'),True); install(module,(i,name,'output'))
                    activation=getattr(layer,'activation_fn',None)
                    if activation is not None:
                        key=(i,'ffn_activation','output')
                        if hasattr(activation,'register_forward_hook'): install(activation,key)
                        elif callable(activation):
                            expected.add(key)
                            def call(*args,_fn=activation,_key=key,**kwargs):
                                out=_fn(*args,**kwargs); self.put(_key,out); return out
                            stack.enter_context(patch.object(layer,'activation_fn',call))
                        else: self.unsupported.append(f'layer {i} activation not hookable')
                if getattr(d,'final_layer_norm',None) is not None:
                    install(d.final_layer_norm,(-1,'decoder_final_ln','input'),True); install(d.final_layer_norm,(-1,'decoder_final_ln','output'))
                install(self.r.base.lm_head,(-1,'lm_head','input'),True); install(self.r.base.lm_head,(-1,'lm_head','output'))
                yield self
                if self.seen!=expected: raise PrefixError('MISSING_NUMERIC_EVIDENCE',f'hook coverage missing={expected-self.seen}')
                if self.reference is not None and self.seen!=self.reference.seen:
                    raise PrefixError('MISSING_NUMERIC_EVIDENCE','Full/Source/Warm checkpoint sets differ')
        finally:
            for handle in handles: handle.remove()


def cache_comparisons(runtime,full,source,warm):
    result=[]
    f,_,_=cache_pairs(full['cache']); s,_,_=cache_pairs(source['cache']); h,_,_=cache_pairs(warm['cache'])
    for i,(fp,sp,hp) in enumerate(zip(f,s,h)):
        for role,a,b,c in zip('kv',fp,sp,hp):
            for label,left,right in [('FULL_PREFIX_vs_SOURCE_KV',a[:,:,:runtime.p],b),('SOURCE_vs_WARM_PAST',b,c[:,:,:runtime.p])]:
                # Move [B,H,T,D] -> [B,T,H*D] for the shared token-index metrics.
                positions=list(range(runtime.p)); ref=left.transpose(1,2).reshape(1,runtime.p,-1).detach().cpu()
                value=right.transpose(1,2).reshape(1,runtime.p,-1).detach().cpu()
                result.append(dict(layer=i,role=role,comparison=label,source_shape=list(left.shape),actual_shape=list(right.shape),
                    source_stride=list(left.stride()),actual_stride=list(right.stride()),absolute_token_positions=positions,
                    metric_coordinates='[batch,token,flattened_head_dim]',native_head_dim=int(left.shape[-1]),
                    **observed_metrics(ref,value,positions,'tokens')))
    return result


def projection_probe(runtime,reference,rows):
    """Actual one-layer Full input vs prefix-only GEMM, never builds a KV source."""
    import torch
    from types import SimpleNamespace
    from semcache.diagnostics.r1b3_gemm_shape import Layer0Inputs,capture_call,parameter_stamps,assert_parameters
    from semcache.metrics.r1 import digest
    candidates=[r for r in rows if r['comparison']=='FULL_PREFIX_vs_SOURCE' and not r['exact_equal'] and
        r['module'] in [role+suffix for role in 'qkv' for suffix in ('_base','_total','_lora_delta')]]
    if not candidates: return [],dict(status='NOT_TRIGGERED',reason='no observed prefix projection difference')
    layer=min(candidates,key=lambda r:r['execution_order'])['layer']
    view=SimpleNamespace(layers=[runtime.adapter.layers[layer]],projection_modules=lambda _:runtime.adapter.projection_modules(layer))
    capture=Layer0Inputs(view)
    with capture.hooks(): replay=runtime.forward('FULL_USE_CACHE')
    results=[]; stamps=parameter_stamps(runtime.adapter)
    try:
        for role,module in runtime.adapter.projection_modules(layer).items():
            x=capture.inputs[role]; key=(layer,role+'_qkv','input')
            if not torch.equal(x.detach().cpu(),reference.snapshots[key]): raise PrefixError('REJECT_PROBE_INPUT','Full replay hidden differs from numeric reference')
            index=torch.arange(runtime.p,device=x.device); prefix=x.index_select(1,index)
            whole,full_info=capture_call(module,x,runtime.a.adapter); small,part_info=capture_call(module,prefix,runtime.a.adapter)
            for component,value in small.items():
                results.append(dict(comparison='SAME_INPUT_FULL_SHAPE_vs_PREFIX_SHAPE',classification='NON_SERVING_DIAGNOSTIC',layer=layer,role=role,component=component,
                    identical_prefix_input=bool(torch.equal(x[:,:runtime.p],prefix)),actual_full_call=full_info,actual_prefix_call=part_info,
                    weight_identity=runtime.provenance['base_weight_identity'],adapter_weight_identity=runtime.adapter_hashes[runtime.a.adapter],
                    actual_source_input_exact=next(row['exact_equal'] for row in rows if row['comparison']=='FULL_PREFIX_vs_SOURCE' and row['layer']==layer and row['module']==role+'_qkv' and row['observation']=='input'),
                    **observed_metrics(whole[component][:,:runtime.p],value,list(range(runtime.p)),'tokens')))
        assert_parameters(stamps)
    finally: capture.inputs.clear()
    return results,dict(status='EXECUTED',layer=layer,kernel_mechanism_identified=False,meaning='same-input shape sensitivity only; no unique causal kernel attribution')


def numeric_case(runtime,layers,limit_mib,probe=True,progress=None):
    import torch
    runtime.check(); a=runtime.forward('FULL_USE_CACHE'); b=runtime.forward('FULL_USE_CACHE')
    repeat=logit_metrics(a['logits'],b['logits']); full_plain=a['logits']; del a,b
    reference=Recorder(runtime,layers,limit_mib=limit_mib)
    try:
        with reference.hooks(): full=runtime.forward('FULL_USE_CACHE',audit=True)
        if not torch.equal(full_plain,full['logits']): raise PrefixError('REJECT_INSTRUMENTATION_PARITY','hooked Full differs from plain Full; repeatability unresolved')
        source_rec=Recorder(runtime,layers,reference=reference,comparison='FULL_PREFIX_vs_SOURCE')
        try:
            with source_rec.hooks(): source=runtime.forward('PREFIX_SOURCE_PREFILL',audit=True)
        finally:
            if progress: progress(source_rec.rows)
        runtime.snapshot.inspect_output(source['cache'],runtime.p)
        if [tensor_hash(t) for pair in cache_pairs(source['cache'])[0] for t in pair]!=runtime.snapshot.hashes:
            raise PrefixError('REJECT_INSTRUMENTATION_PARITY','hooked Source cache differs bitwise from canonical source')
        plain=runtime.forward('EXACT_PREFIX_WARM')
        warm_rec=Recorder(runtime,layers,reference=reference,comparison='FULL_SUFFIX_vs_WARM')
        try:
            with warm_rec.hooks(): warm=runtime.forward('EXACT_PREFIX_WARM',audit=True)
        finally:
            if progress: progress(source_rec.rows+warm_rec.rows)
        if not torch.equal(plain['logits'],warm['logits']): raise PrefixError('REJECT_INSTRUMENTATION_PARITY','hooked Warm differs from plain Warm')
        rows=source_rec.rows+warm_rec.rows
        probes,probe_info=projection_probe(runtime,reference,rows) if probe else ([],dict(status='DISABLED'))
        kv=cache_comparisons(runtime,full,source,warm)
        quality=dict(full_repeat=repeat,source_boundary=logit_metrics(full['logits'][:,runtime.p-1:runtime.p],source['logits'][:,-1:]),
            suffix=logit_metrics(full['logits'][:,runtime.p:],warm['logits']),last_position=logit_metrics(full['logits'][:,-1:],warm['logits'][:,-1:]))
        claims=dict(
            fixed_input_projection_shape_sensitivity=dict(status='SUPPORTED' if probes and any(not r['exact_equal'] for r in probes) else 'CONTRADICTED' if probes else 'UNRESOLVED',
                scope='one-layer identical-input/weights Full versus prefix-only shape; not unique cause or kernel identification'),
            actual_prefix_kv_difference=dict(status='SUPPORTED' if any(not r['bitwise_equal'] for r in kv if r['comparison']=='FULL_PREFIX_vs_SOURCE_KV') else 'CONTRADICTED',
                scope='actual independent Source KV versus Full prefix KV; comparison only, not cache construction'),
            full_repeat_variation=dict(status='CONTRADICTED' if repeat['exact_equal'] else 'SUPPORTED',scope='two uninstrumented Full calls only'),
            unique_initial_cause=dict(status='UNRESOLVED',reason='observed boundaries and one projection probe do not isolate every LN/attention arithmetic or CUDA kernel'))
        summary=dict(first_findings=first_findings(rows),quality=quality,hypotheses=claims,probe=probe_info,captured_layers=layers,
            unsupported=reference.unsupported,reference_cpu_bytes=reference.bytes,
            scope='first observable checkpoint only; prefix and suffix branches remain separate',
            causal_status='UNRESOLVED; prefix KV and suffix computation shape both contribute to warm comparison',kernel_mechanism_identified=False)
        audits=dict(full=full['audit'],source=source['audit'],warm=warm['audit'],canonical=runtime.check())
        return rows,kv,probes,summary,audits
    finally:
        reference.snapshots.clear(); reference.metadata.clear()

