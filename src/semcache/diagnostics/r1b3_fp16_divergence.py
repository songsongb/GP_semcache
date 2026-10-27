"""Correctness-only streaming hook comparisons; no production projection changes."""
from contextlib import contextmanager
import math
from semcache.diagnostics.r1b1_d1 import error as d1_error
from semcache.diagnostics.r1b3_break_even import MissingEvidence


def scalar_metrics(reference,actual,atol=1e-3,rtol=1e-3):
    """CPU-only specification of scalar metrics; tensor allclose is measured by Torch."""
    if len(reference)!=len(actual) or not reference: raise ValueError('Equal nonempty vectors required')
    if not all(math.isfinite(x) for x in [*reference,*actual]): raise ValueError('Nonfinite values')
    differences=[abs(a-b) for a,b in zip(reference,actual)]
    failed=[i for i,(a,b) in enumerate(zip(reference,actual)) if abs(a-b)>atol+rtol*abs(b)]
    norm=math.sqrt(sum(x*x for x in reference)); diffnorm=math.sqrt(sum(x*x for x in differences))
    return dict(max_abs_error=max(differences),mean_abs_error=sum(differences)/len(differences),
        relative_l2_error=diffnorm/norm if norm else (0. if not diffnorm else None),
        tolerance_violations=len(failed),tolerance_violation_fraction=len(failed)/len(reference),
        first_tolerance_failure=failed[0] if failed else None,exact_equal=not any(differences),allclose=not failed)


def tensor_metrics(reference,actual,positions=None):
    import torch
    result=d1_error(reference,actual,1e-3,1e-3,positions)
    a,b=reference.detach().cpu(),actual.detach().cpu()
    diff=(a.double()-b.double()).abs()
    violations=int((~torch.isclose(a,b,atol=1e-3,rtol=1e-3)).sum())
    result.update(mean_abs_error=float(diff.mean()) if diff.numel() else 0.,tolerance_violations=violations,
        tolerance_violation_fraction=violations/diff.numel() if diff.numel() else 0.,elements=diff.numel())
    return result


def logit_metrics(reference,actual,top_k=5):
    import torch
    result=tensor_metrics(reference,actual)
    a,b=reference[0,-1].double(),actual[0,-1].double(); k=min(top_k,a.numel())
    values,indices=a.topk(k); error=float((a-b).abs().max())
    loga=torch.log_softmax(a,dim=-1); logb=torch.log_softmax(b,dim=-1)
    margin=float(values[0]-values[1]) if k>=2 else None
    result.update(last_argmax_agreement=int(a.argmax())==int(b.argmax()),last_position_kl=float((loga.exp()*(loga-logb)).sum()),
        top_k_ids=indices.tolist(),top_k_reference_logits=values.tolist(),top1_top2_margin=margin,
        last_max_abs_error=error,margin_exceeds_twice_max_error=margin>2*error if margin is not None else None,
        correctness_status='STRICT_PASS' if result['allclose'] else 'STRICT_FAIL')
    return result


def first_divergences(rows):
    # Preserve observed hook order; q/k/v execution order need not be alphabetical.
    groups={}
    for row in rows: groups.setdefault((row.get('length'),row['comparison'],row['layer']),[]).append(row)
    return [dict(length=k[0],comparison=k[1],layer=k[2],
        first_nonzero=next((r for r in group if r.get('exact_equal') is False),None),
        first_tolerance_failure=next((r for r in group if r.get('allclose') is False),None)) for k,group in groups.items()]


class StreamingAudit:
    """Retain one CPU reference only; candidates are compared and released at hooks."""
    def __init__(self,adapter,user,layers,comparison,reference=None,retain=False,limit_mib=1024):
        self.adapter=adapter; self.user=user; self.layers=tuple(layers); self.comparison=comparison
        self.reference=reference; self.retain=retain; self.limit=limit_mib*1024**2
        self.snapshots={}; self.rows=[]; self.bytes=0; self.seen=set()
    @contextmanager
    def hooks(self,fresh=None):
        handles=[]; seen_before=set(self.seen)
        def put(layer,stage,role,value):
            value=value[0] if isinstance(value,tuple) else value
            key=(layer,stage,role)
            if key in self.seen: raise MissingEvidence(f'Duplicate hook execution {key}')
            self.seen.add(key); tensor=value.detach().to('cpu',copy=True)
            positions=None
            if self.reference is not None:
                if key not in self.reference: raise MissingEvidence(f'Missing reference hook {key}')
                ref=self.reference[key]
                if stage in ('base_projection','lora_delta') and fresh is not None:
                    positions=fresh
                    if ref.shape!=tensor.shape: ref=ref[:,fresh]
                self.rows.append(dict(comparison=self.comparison,event=len(self.rows),layer=layer,stage=stage,role=role,
                    reference_shape=list(ref.shape),actual_shape=list(tensor.shape),positions='fresh_only' if positions is not None else 'all',
                    **tensor_metrics(ref,tensor,positions)))
            if self.retain:
                self.bytes+=tensor.numel()*tensor.element_size()
                if self.bytes>self.limit: raise MemoryError(f'CPU reference snapshot budget exceeded: {self.bytes}>{self.limit}')
                self.snapshots[key]=tensor
        def before(layer,stage,role):
            return lambda m,args,kwargs:put(layer,stage,role,args[0] if args else kwargs['hidden_states'])
        def after(layer,stage,role): return lambda m,args,out:put(layer,stage,role,out)
        try:
            base=self.adapter.model.get_base_model() if hasattr(self.adapter.model,'peft_config') else self.adapter.model
            decoder=base.model.decoder
            for name in ('embed_tokens','embed_positions'):
                handles.append(getattr(decoder,name).register_forward_hook(after(-1,name,'')))
            for i in self.layers:
                layer=self.adapter.layers[i]
                handles.append(layer.register_forward_pre_hook(before(i,'layer_input',''),with_kwargs=True))
                handles.append(layer.register_forward_hook(after(i,'layer_output','')))
                handles.append(layer.self_attn.register_forward_hook(after(i,'attention_output','')))
                for role,module in self.adapter.projection_modules(i).items():
                    handles.append(module.register_forward_pre_hook(before(i,'qkv_input',role),with_kwargs=True))
                    handles.append(module.register_forward_hook(after(i,'total_qkv',role)))
                    handles.append(module.get_base_layer().register_forward_hook(after(i,'base_projection',role)))
                    # B's actual output, scaled exactly as vanilla PEFT does; not an independent LoRA recomputation.
                    handles.append(module.lora_B[self.user].register_forward_hook(
                        lambda m,args,out,i=i,r=role,scale=module.scaling[self.user]:put(i,'lora_delta',r,out*scale)))
            yield self
            expected={(i,s,r) for i in self.layers for s in ('qkv_input','total_qkv','base_projection','lora_delta') for r in 'qkv'}
            expected|={(i,s,'') for i in self.layers for s in ('layer_input','layer_output','attention_output')}
            expected|={(-1,'embed_tokens',''),(-1,'embed_positions','')}
            if fresh==[]: expected={k for k in expected if k[1] not in ('base_projection','lora_delta')}
            if expected!=self.seen-seen_before: raise MissingEvidence(f'Hook coverage mismatch: missing={expected-(self.seen-seen_before)}')
        finally:
            for handle in handles: handle.remove()


def shape_probe(adapter,reference,hits,length,user,layers):
    """Reuse D1's controlled full/subset PEFT/base/A/B computations one layer at a time."""
    from types import SimpleNamespace
    from semcache.diagnostics.r1b1_d1 import shape_probes
    rows=[]
    for layer in layers:
        view=SimpleNamespace(model=adapter.model,layers=[adapter.layers[layer]],projection_modules=lambda _,i=layer:adapter.projection_modules(i))
        captured={(0,role+'_input'):reference[(layer,'qkv_input',role)] for role in 'qkv'}
        captured.update({(0,role):reference[(layer,'total_qkv',role)] for role in 'qkv'})
        for row in shape_probes(view,captured,hits,length,user,1e-3,1e-3):
            row.update(layer=layer,comparison='SAME_HIDDEN_FULL_VS_SUBSET',probe=True)
            rows.append(row)
    return rows


def hypotheses(comparisons,rows,probes,control=None):
    full=comparisons['FULL_A_vs_FULL_B']; source=comparisons['SOURCE_FULL_vs_TARGET_FULL']
    source_qkv=[r for r in rows if r['comparison']=='SOURCE_FULL_vs_TARGET_FULL' and r['stage']=='total_qkv']
    exact_source=bool(source_qkv) and all(r['exact_equal'] for r in source_qkv)
    shape=any(not r['exact_equal'] for r in probes)
    trajectory=[dict(comparison=r['comparison'],layer=r['layer'],stage=r['stage'],role=r['role'],max_abs_error=r['max_abs_error'])
        for r in rows if r['comparison'] in ('TARGET_FULL_vs_LEGACY','TARGET_FULL_vs_CONTIGUOUS') and r['stage'] in ('layer_input','total_qkv','attention_output','layer_output')]
    propagated=any(r['layer']>0 and r['stage']=='layer_input' and r['max_abs_error']>0 for r in trajectory)
    return dict(H1=dict(status='SUPPORTED_BY_CONTROLLED_PROBE' if shape and full['exact_equal'] and exact_source else 'UNRESOLVED',
                    shape_difference_observed=shape,full_shape_injection_exact=control.get('exact_equal') if control else None),
        H2=dict(status='NOT_OBSERVED_IN_CAPTURED_LAYERS' if exact_source else
            'SOURCE_TARGET_QKV_DIFFERENCE_OBSERVED' if source_qkv else 'MISSING_EVIDENCE',source_logits_exact=source['exact_equal']),
        H3=dict(status='SAME_ADAPTER_AND_PREFIX_ENFORCED; DIFFERENT_ADAPTER_CONTROL_REPORTED_SEPARATELY'),
        H4=dict(status='DOWNSTREAM_HIDDEN_DIFFERENCES_OBSERVED' if propagated else 'NOT_OBSERVED_IN_CAPTURED_LAYERS',
            trajectory=trajectory,causal_attribution='trajectory alone does not isolate a unique numerical cause'),
        H5=dict(status='NOT_OBSERVED_IN_TWO_FULL_RUNS' if full['exact_equal'] else 'OBSERVED_FULL_REPEAT_VARIATION'),
        conclusion='Diagnostic evidence only; no universal FP16 or production-safety conclusion')
