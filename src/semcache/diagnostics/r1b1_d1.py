"""Correctness-only D1 controls. No production executor or policy changes.

CPU captures exist only in process memory; output artifacts contain scalar
errors/indices, never hidden tensors, attention arrays or weights.
"""
from contextlib import contextmanager, nullcontext
import math

EXPECTED_MOVES = [(3, 6, 54, 57), (6, 9, 57, 60), (9, 12, 60, 63)]


def provenance_audit(natural, strict):
    def spans(row):
        records = row['reuse_provenance']
        occupied = []
        for p in records:
            ss,se,ds,de = (p[k] for k in ('source_start','source_end','destination_start','destination_end'))
            if se-ss != 3 or de-ds != 3:
                raise ValueError('Expected exact-w3 spans')
            if p['token_ids'] != row['input_token_ids'][ds:de]:
                raise ValueError('Provenance tokens do not match destination input')
            if p['token_ids'] != row['input_token_ids'][ss:se]:
                raise ValueError('Exact-repeat provenance tokens do not match source input')
            if p.get('source_adapter')!=row['adapter'] or p.get('destination_adapter')!=row['adapter']:
                raise ValueError('Exact-repeat provenance adapter mismatch')
            occupied.extend(range(ds,de))
        if len(occupied)!=len(set(occupied)) or len(occupied)!=row['reused_tokens']:
            raise ValueError('Executed coverage disagrees with nonoverlap provenance')
        return [(p['source_start'],p['source_end'],p['destination_start'],p['destination_end']) for p in records]
    for field in ('input_token_ids','adapter','token_ids_sha256','impact_execution','storage','repeat','instrumentation'):
        if natural[field] != strict[field]:
            raise ValueError('Natural/strict artifact pair differs: '+field)
    a,b=spans(natural),spans(strict)
    mismatches=[p for p in a if p[0]!=p[2]]
    strict_bad=[p for p in b if p[0]!=p[2]]
    is64=natural['requested_prompt_tokens']==64
    return dict(natural_reused=natural['reused_tokens'],strict_reused=strict['reused_tokens'],
        mismatched_windows=mismatches,mismatched_tokens=sum(p[3]-p[2] for p in mismatches),
        context_audit=[dict(source_start=ss,destination_start=ds,
            prefix_token_ids_equal=natural['input_token_ids'][:ss]==natural['input_token_ids'][:ds])
            for ss,se,ds,de in mismatches],
        strict_mismatched_windows=strict_bad,
        expected_64_case_verified=(sorted(mismatches)==EXPECTED_MOVES and not strict_bad
            and natural['reused_tokens']==63 and strict['reused_tokens']==54) if is64 else None,
        selected_guard_equals_strict=[p for p in a if p[0]==p[2]]==b,
        reported_natural_quality=natural['correctness'],reported_strict_quality=strict['correctness'],
        evidence='input artifact only; no GPU reproduction implied')


def spans(hits):
    return [dict(source_start=h.entry.positions[0],source_end=h.entry.positions[1],
                 destination_start=h.window.start,destination_end=h.window.end,
                 token_ids=list(h.window.token_ids)) for h in hits]


def guarded_selections(hits, n, ids, user):
    from semcache.semantic.hit_selection import select_nonoverlapping, select_position_aligned_diagnostic
    natural=select_nonoverlapping(hits,n)[0]
    return dict(NATURAL=natural,
        POSITION_GUARD=select_nonoverlapping([h for h in hits if h.entry.positions[0]==h.window.start],n)[0],
        SELECTED_POSITION_GUARD=[h for h in natural if h.entry.positions[0]==h.window.start],
        STRICT=select_position_aligned_diagnostic(hits,n,ids,user)[0])


def error(reference, candidate, atol, rtol, positions=None):
    """FP64 diagnostics; allclose threshold is exactly the artifact's tolerance."""
    import torch
    if reference.shape!=candidate.shape:
        raise ValueError('Diagnostic tensor shape mismatch')
    a,b=(x.detach().to(device='cpu',dtype=torch.float64) for x in (reference,candidate))
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        raise ValueError('Nonfinite diagnostic tensor')
    d=(a-b).abs()
    flat=int(d.argmax()) if d.numel() else None
    coords=list(torch.unravel_index(torch.tensor(flat),d.shape)) if flat is not None else []
    coords=[int(i) for i in coords]
    first=(d!=0).nonzero()
    # Match the existing torch.allclose(reference, candidate) operand order;
    # torch's relative threshold uses the second operand, not the FULL tensor.
    failing=(~torch.isclose(reference.cpu(),candidate.cpu(),atol=atol,rtol=rtol)).nonzero()
    def coordinate(values):
        if len(values)==0:
            return None
        index=[int(i) for i in values[0]]
        token=index[1] if len(index)>=3 else index[0]
        return dict(index=index,token_index=positions[token] if positions is not None else token,
                    absolute_error=float(d[tuple(index)]),reference_value=float(a[tuple(index)]),
                    relative_error=float(d[tuple(index)]/a[tuple(index)].abs()) if a[tuple(index)]!=0 else None)
    norm=float(a.norm())
    return dict(max_abs_error=float(d.max()) if d.numel() else 0.,
        relative_l2_error=float(d.norm())/norm if norm else (0. if not bool(d.any()) else None),
        max_error_index=coords,exact_equal=bool(torch.equal(reference.cpu(),candidate.cpu())),
        allclose=bool(torch.allclose(reference.cpu(),candidate.cpu(),atol=atol,rtol=rtol)),
        first_nonzero=coordinate(first),first_tolerance_failure=coordinate(failing),atol=atol,rtol=rtol)


@contextmanager
def capture(adapter):
    """Install after mixed context validation. Hooks only copy, never reproject."""
    values,handles={},[]
    def put(key,value):
        if key in values:
            raise RuntimeError('One forward per capture context')
        values[key]=value.detach().to('cpu',copy=True)
    try:
        for i,layer in enumerate(adapter.layers):
            handles.append(layer.register_forward_pre_hook(lambda m,a,i=i:put((i,'hidden_in'),a[0])))
            handles.append(layer.register_forward_hook(lambda m,a,o,i=i:put((i,'hidden_out'),o[0] if isinstance(o,tuple) else o)))
            for role,module in adapter.projection_modules(i).items():
                handles.append(module.register_forward_pre_hook(lambda m,a,i=i,r=role:put((i,r+'_input'),a[0])))
                handles.append(module.register_forward_hook(lambda m,a,o,i=i,r=role:put((i,r),o)))
        yield values
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def full_shape_injection(adapter,hits):
    """Native full projection then overwrite cached rows. Correctness only."""
    handles=[]
    try:
        for i in range(len(adapter.layers)):
            for role,module in adapter.projection_modules(i).items():
                def inject(m,args,output,i=i,role=role):
                    result=output.clone()
                    for hit in hits:
                        result[:,hit.window.start:hit.window.end]=hit.entry.tensors[i]['qkv'.index(role)].to(result)
                    return result
                handles.append(module.register_forward_hook(inject))
        yield
    finally:
        for handle in handles:
            handle.remove()


def forward(model, adapter, ids, user, hits=None, control='mixed'):
    import torch
    from semcache.edgelora.mixed_projection import mixed_projection_path
    model.set_adapter(user)
    model.eval()
    n=len(ids)
    device=next(model.parameters()).device
    inputs=dict(input_ids=torch.tensor([ids],device=device),use_cache=False,output_attentions=True)
    whole_inputs,handles={},[]
    fresh=[p for p in range(n) if not any(h.window.start<=p<h.window.end for h in (hits or []))]
    def full_fresh(i,role,module,subset,native):
        return native(whole_inputs[(i,role)]).index_select(1,torch.tensor(fresh,device=device))
    context=(nullcontext(None) if hits is None else full_shape_injection(adapter,hits)
             if control=='injection' else mixed_projection_path(adapter,user,hits,n,
                fresh_projection=full_fresh if control=='full_assembly' else None))
    try:
        with torch.inference_mode(), context as audit:
            if control=='full_assembly':
                for i in range(len(adapter.layers)):
                    for role,module in adapter.projection_modules(i).items():
                        handles.append(module.register_forward_pre_hook(
                            lambda m,a,i=i,r=role:whole_inputs.__setitem__((i,r),a[0])))
            with capture(adapter) as captured:
                output=model(**inputs)
            logits=output.logits.detach().cpu()
            records=list(audit.records.values()) if audit is not None else []
        return dict(logits=logits,captured=captured,projection_records=records)
    finally:
        for handle in handles:
            handle.remove()


def layer_errors(reference,candidate,atol,rtol):
    order=('hidden_in','q_input','q','k_input','k','v_input','v','hidden_out')
    rows=[]
    for layer in sorted({k[0] for k in reference}):
        for role in order:
            rows.append(dict(layer=layer,role=role,**error(reference[(layer,role)],candidate[(layer,role)],atol,rtol)))
    firsts=[]
    for layer in sorted({r['layer'] for r in rows}):
        group=[r for r in rows if r['layer']==layer]
        firsts.append(dict(layer=layer,first_nonzero=next((r for r in group if not r['exact_equal']),None),
            first_tolerance_failure=next((r for r in group if not r['allclose']),None),
            first_qkv_nonzero=next((r for r in group if r['role'] in 'qkv' and not r['exact_equal']),None),
            first_qkv_tolerance_failure=next((r for r in group if r['role'] in 'qkv' and not r['allclose']),None)))
    return rows,firsts


def shape_probes(adapter,captured,hits,n,user,atol,rtol):
    """Same inputs for each comparison; isolate base, LoRA A/B and total PEFT."""
    import torch
    positions=[p for p in range(n) if not any(h.window.start<=p<h.window.end for h in hits)]
    if not positions:
        return []
    device=next(adapter.model.parameters()).device
    index=torch.tensor(positions,device=device)
    rows=[]
    with torch.inference_mode():
        for i in range(len(adapter.layers)):
            for role,module in adapter.projection_modules(i).items():
                h=captured[(i,role+'_input')].to(device)
                cast=module._cast_input_dtype(h,module.lora_A[user].weight.dtype)
                a_full=module.lora_A[user](cast)
                for component,fn,inputs in (('total_peft',module,h),('base',module.get_base_layer(),h),
                    ('lora_A',module.lora_A[user],cast),('lora_B_same_A_input',module.lora_B[user],a_full)):
                    full=fn(inputs).index_select(1,index)
                    subset=fn(inputs.index_select(1,index))
                    row=dict(layer=i,role=role,component=component,full_rows=n,subset_rows=len(positions),
                        input_dtype=str(inputs.dtype),full_input_stride=list(inputs.stride()),
                        **error(full,subset,atol,rtol,positions))
                    if component=='total_peft':
                        actual=captured[(i,role)][:,positions]
                        row['observed_output_vs_recomputed_subset']=error(subset,actual,atol,rtol,positions)
                        row['observed_output_vs_full_shape']=error(full,actual,atol,rtol,positions)
                    rows.append(row)
    return rows


def cache_errors(hits,source_capture,reference_capture,atol,rtol,strict_capture=None):
    aggregates={}
    for h in hits:
        for layer,qkv in h.entry.tensors.items():
            for role,tensor in zip('qkv',qkv):
                ss,se=h.entry.positions
                ds,de=h.window.start,h.window.end
                references=[('cache_vs_source_execution',source_capture[(layer,role)][:,ss:se]),
                            ('cache_vs_target_full',reference_capture[(layer,role)][:,ds:de])]
                if strict_capture is not None:
                    references.append(('cache_vs_mixed_assembled_rows',strict_capture[(layer,role)][:,ds:de]))
                for label,reference in references:
                    value=error(reference,tensor,atol,rtol)
                    key=(layer,role,label)
                    row=aggregates.setdefault(key,dict(layer=layer,role=role,comparison=label,windows=0,
                        nonexact_windows=0,failed_windows=0,max_abs_error=0.,first_nonexact_window=None))
                    row['windows']+=1
                    row['nonexact_windows']+=not value['exact_equal']
                    row['failed_windows']+=not value['allclose']
                    row['max_abs_error']=max(row['max_abs_error'],value['max_abs_error'])
                    if not value['exact_equal'] and row['first_nonexact_window'] is None:
                        row['first_nonexact_window']=dict(source_start=ss,destination_start=ds,**value)
    return list(aggregates.values())
