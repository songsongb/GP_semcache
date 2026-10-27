"""C1 correctness interventions. Existing mapping and assembly executors are unchanged.

Class-forward interception is scoped to a single, single-threaded diagnostic
forward. It changes the input rows seen by native PEFT only, not its arithmetic.
"""
from contextlib import contextmanager, ExitStack
import hashlib
from types import SimpleNamespace
from unittest.mock import patch
from semcache.metrics.r1 import digest
from semcache.diagnostics.r1b3_break_even import MissingEvidence
from semcache.diagnostics.r1b3_fp16_divergence import tensor_metrics


def prefix_layers(value, count):
    result=[]
    for token in value.split(','):
        n=count if token.strip()=='all' else int(token)
        if not 0<=n<=count: raise ValueError('Replay prefix exceeds loaded model layers')
        if n not in result: result.append(n)
    return result


def row_cost(length, fresh, layers, full_layers):
    full=set(full_layers)
    if length<1 or not 0<=fresh<=length or layers<1 or any(i<0 or i>=layers for i in full):
        raise ValueError('Invalid projection row geometry')
    native=3*(len(full)*length+(layers-len(full))*fresh)
    return dict(actual_native_projection_rows=native,
        actual_skipped_projection_rows=3*layers*length-native,
        reused_data_tokens=length-fresh,full_shape_layers=sorted(full),
        baseline_wrapper_skipped_rows=3*layers*(length-fresh))


def metadata(tensor):
    return dict(shape=list(tensor.shape),stride=list(tensor.stride()),dtype=str(tensor.dtype),
        device=str(tensor.device),storage_offset=tensor.storage_offset(),data_ptr=tensor.data_ptr())


def tensor_sha(tensor):
    import torch
    value=tensor.detach().to('cpu').contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def parameter_stamps(adapter):
    # Parameters support version counters even when the forward uses inference_mode.
    return tuple((p,p.data_ptr(),tuple(p.shape),tuple(p.stride()),str(p.dtype),str(p.device),p._version)
        for i in range(len(adapter.layers)) for m in adapter.projection_modules(i).values() for p in m.parameters())


def assert_parameters(stamps):
    for p,pointer,shape,stride,dtype,device,version in stamps:
        if (p.data_ptr(),tuple(p.shape),tuple(p.stride()),str(p.dtype),str(p.device),p._version)!=(pointer,shape,stride,dtype,device,version):
            raise MissingEvidence('Projection weights changed during correctness control')


class Layer0Inputs:
    """Retain only actual Layer-0 QKV inputs on their original device/stride."""
    def __init__(self,adapter): self.adapter=adapter; self.inputs={}; self.info={}
    @contextmanager
    def hooks(self):
        handles=[]
        try:
            for role,module in self.adapter.projection_modules(0).items():
                def before(m,args,kwargs,r=role):
                    x=args[0] if args else kwargs['hidden_states']
                    if r in self.inputs: raise MissingEvidence('Layer-0 input captured twice')
                    self.inputs[r]=x.detach(); self.info[r]=metadata(x)
                handles.append(module.register_forward_pre_hook(before,with_kwargs=True))
            yield
            if set(self.inputs)!=set('qkv'): raise MissingEvidence('Missing Layer-0 projection input')
        finally:
            for h in handles: h.remove()


class NativeShapeIntervention:
    """Make selected layers compute native PEFT on full input, return fresh rows.

    Patch the PEFT class before executor entry, allowing the unchanged executor
    to bind this native callable. Install input pre-hooks only AFTER executor
    validation. No module instance forward is installed by this intervention.
    """
    def __init__(self,adapter,full_layers,user='user_a'):
        self.adapter=adapter; self.full_layers=set(full_layers); self.user=user
        self.modules={id(m):(i,r,m) for i in range(len(adapter.layers)) for r,m in adapter.projection_modules(i).items()}
        if not self.full_layers<=set(range(len(adapter.layers))): raise ValueError('Invalid intervention layers')
        self.whole={}; self.fresh=None; self.records={}; self.stamps=parameter_stamps(adapter)
        self.active=False; self.stack=None
    def __enter__(self):
        from semcache.models.lora_decomposition import validate_projection
        for _,_,module in self.modules.values(): validate_projection(module,self.user)
        self.stack=ExitStack()
        try:
            for cls in {type(m) for _,_,m in self.modules.values()}:
                original=cls.forward
                def dispatch(module,hidden,*args,_native=original,**kwargs):
                    import torch
                    item=self.modules.get(id(module))
                    if item is None: return _native(module,hidden,*args,**kwargs)
                    layer,role,_=item; key=(layer,role)
                    if key in self.records: raise MissingEvidence('More than one native projection call in a replay')
                    if self.fresh is None or key not in self.whole: raise MissingEvidence('Missing actual full projection input')
                    whole=self.whole[key]
                    index=torch.tensor(self.fresh,device=whole.device,dtype=torch.long)
                    if not torch.equal(whole.index_select(1,index),hidden): raise MissingEvidence('Fresh input differs from actual full hidden selection')
                    controlled=layer in self.full_layers
                    call_input=whole if controlled else hidden
                    info=dict(layer=layer,role=role,controlled=controlled,logical_subset_input=metadata(hidden),native_input=metadata(call_input),
                        native_rows=call_input.shape[1],input_values_match=True)
                    output=_native(module,call_input,*args,**kwargs)
                    info['native_output']=metadata(output)
                    selected=output.index_select(1,index) if controlled else output
                    info['returned_output']=metadata(selected); self.records[key]=info
                    self.whole.pop(key)
                    return selected
                self.stack.enter_context(patch.object(cls,'forward',dispatch))
            self.active=True
            return self
        except BaseException:
            self.stack.close(); raise
    def __exit__(self,*exc):
        try: return self.stack.__exit__(*exc)
        finally:
            self.active=False; self.whole.clear(); assert_parameters(self.stamps)
    def observer(self,audit=None):
        @contextmanager
        def observe(fresh):
            if not fresh: raise MissingEvidence('Counterfactual replay requires at least one fresh row')
            self.fresh=tuple(fresh); handles=[]
            try:
                for i,role,module in self.modules.values():
                    def before(m,args,kwargs,key=(i,role)):
                        self.whole[key]=args[0] if args else kwargs['hidden_states']
                    handles.append(module.register_forward_pre_hook(before,with_kwargs=True))
                with audit.hooks(fresh,self.full_layers) if audit else _null(): yield
            finally:
                for h in handles: h.remove()
                self.whole.clear()
        return observe
    def accounting(self,length,fresh):
        if len(self.records)!=len(self.modules): raise MissingEvidence('Missing native projection call ledger')
        expected=row_cost(length,fresh,len(self.adapter.layers),self.full_layers)
        actual=sum(r['native_rows'] for r in self.records.values())
        if actual!=expected['actual_native_projection_rows']: raise MissingEvidence('Actual native row count disagrees with intervention')
        return dict(expected,native_calls=list(self.records.values()),
            native_call_count=len(self.records),assembly='UNCHANGED_B21_CONTIGUOUS',
            intervention='original PEFT full forward then select fresh output; correctness only')


@contextmanager
def _null(): yield


class ReplayAudit:
    """B3 streaming comparisons with per-layer full/subset child-output alignment."""
    def __init__(self,adapter,reference,layers,comparison,user='user_a'):
        self.adapter=adapter; self.reference=reference; self.layers=tuple(layers)
        self.comparison=comparison; self.user=user; self.rows=[]
        self.layer0_totals={}
        self.gemm_calls=[]
    @contextmanager
    def hooks(self,fresh,full_layers):
        handles=[]; seen=set()
        def put(layer,stage,role,value):
            key=(layer,stage,role)
            if key in seen or key not in self.reference: raise MissingEvidence('Duplicate/missing replay reference: '+str(key))
            seen.add(key); value=value[0] if isinstance(value,tuple) else value
            actual=value.detach().to('cpu',copy=True); ref=self.reference[key]; positions=None
            if layer==0 and stage=='total_qkv': self.layer0_totals[role]=actual
            if stage in ('base_projection','lora_delta') and layer not in full_layers:
                positions=fresh; ref=ref[:,fresh]
            self.rows.append(dict(comparison=self.comparison,event=len(self.rows),layer=layer,stage=stage,role=role,
                actual_output=metadata(value),full_native_layer=layer in full_layers,
                **tensor_metrics(ref,actual,positions)))
        def pre(i,stage,r): return lambda m,args,kwargs:put(i,stage,r,args[0] if args else kwargs['hidden_states'])
        def post(i,stage,r): return lambda m,args,out:put(i,stage,r,out)
        pending={}
        def gemm_pre(i,role,component):
            def record(m,args):
                key=(i,role,component)
                if key in pending: raise MissingEvidence('Duplicate GEMM component call')
                value=dict(layer=i,role=role,component=component,input=metadata(args[0]),weight=metadata(m.weight))
                pending[key]=value; self.gemm_calls.append(value)
            return record
        def gemm_post(i,role,component):
            return lambda m,args,out:pending[(i,role,component)].update(output=metadata(out))
        try:
            base=self.adapter.model.get_base_model() if hasattr(self.adapter.model,'peft_config') else self.adapter.model
            for name in ('embed_tokens','embed_positions'):
                handles.append(getattr(base.model.decoder,name).register_forward_hook(post(-1,name,'')))
            for i in self.layers:
                layer=self.adapter.layers[i]
                handles.append(layer.register_forward_pre_hook(pre(i,'layer_input',''),with_kwargs=True))
                handles.append(layer.register_forward_hook(post(i,'layer_output','')))
                handles.append(layer.self_attn.register_forward_hook(post(i,'attention_output','')))
                for role,m in self.adapter.projection_modules(i).items():
                    handles.append(m.register_forward_pre_hook(pre(i,'qkv_input',role),with_kwargs=True))
                    handles.append(m.register_forward_hook(post(i,'total_qkv',role)))
                    for component,child in (('base',m.get_base_layer()),('lora_A',m.lora_A[self.user]),('lora_B',m.lora_B[self.user])):
                        handles.append(child.register_forward_pre_hook(gemm_pre(i,role,component)))
                        handles.append(child.register_forward_hook(gemm_post(i,role,component)))
                    handles.append(m.get_base_layer().register_forward_hook(post(i,'base_projection',role)))
                    handles.append(m.lora_B[self.user].register_forward_hook(lambda mod,args,out,i=i,r=role,scale=m.scaling[self.user]:put(i,'lora_delta',r,out*scale)))
            yield
            expected={(-1,'embed_tokens',''),(-1,'embed_positions','')}
            expected|={(i,s,'') for i in self.layers for s in ('layer_input','layer_output','attention_output')}
            expected|={(i,s,r) for i in self.layers for s in ('qkv_input','base_projection','lora_delta','total_qkv') for r in 'qkv'}
            if seen!=expected: raise MissingEvidence('Incomplete replay hook coverage')
        finally:
            for h in handles: h.remove()


def capture_call(module,hidden,user='user_a',select_positions=None):
    """Actual native outputs; copy base BEFORE PEFT can add delta in-place."""
    import torch
    from semcache.models.lora_decomposition import validate_projection
    validate_projection(module,user)
    values={}; calls=[]; handles=[]
    index=torch.tensor(select_positions,device=hidden.device,dtype=torch.long) if select_positions is not None else None
    def select(value): return value.index_select(1,index) if index is not None else value
    def before(component):
        return lambda m,args: calls.append(dict(component=component,input=metadata(args[0]),weight=metadata(m.weight)))
    def after(component,scale=1.):
        def save(m,args,out):
            if component in values: raise MissingEvidence('Repeated component call')
            # The scaling expression follows vanilla PEFT; source output is unmodified.
            value=out if component=='base' else out*scale
            selected=select(value)
            values[component]=selected.detach().to('cpu',copy=True)
            calls[-1]['output']=metadata(out)
            calls[-1]['selected_component_output']=metadata(selected)
        return save
    try:
        for name,child in (('base',module.get_base_layer()),('lora_A',module.lora_A[user]),('lora_delta',module.lora_B[user])):
            handles.append(child.register_forward_pre_hook(before(name)))
            if name!='lora_A': handles.append(child.register_forward_hook(after(name,module.scaling[user] if name=='lora_delta' else 1.)))
            else: handles.append(child.register_forward_hook(lambda m,args,out:calls[-1].update(output=metadata(out))))
        with torch.inference_mode(): output=module(hidden)
        selected=select(output)
        values['total']=selected.detach().to('cpu',copy=True)
        return values,dict(input=metadata(hidden),native_output=metadata(output),returned_output=metadata(selected),gemm_calls=calls,
            selection_device=str(hidden.device) if index is not None else None)
    finally:
        for h in handles: h.remove()


def fp32_call(module,hidden,user='user_a'):
    """Functional FP32 control, upcasting the SAME stored weights, never mutating them."""
    import torch
    import torch.nn.functional as F
    calls=[]
    def linear(name,x,m):
        weight=m.weight.detach().float(); bias=m.bias.detach().float() if m.bias is not None else None
        result=F.linear(x,weight,bias)
        calls.append(dict(component=name,input=metadata(x),weight=metadata(weight),output=metadata(result)))
        return result
    old_precision=torch.get_float32_matmul_precision()
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    try:
        # This precision control applies only to the functional FP32 probe.
        # FP16 reduction settings and model tensors are never modified.
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        with torch.inference_mode():
            x=hidden.float(); base=linear('base',x,module.get_base_layer())
            a=linear('lora_A',x,module.lora_A[user]); delta=linear('lora_delta',a,module.lora_B[user])*module.scaling[user]
            total=base+delta
    finally:
        torch.set_float32_matmul_precision(old_precision)
        if torch.backends.cuda.matmul.allow_tf32!=old_tf32: torch.backends.cuda.matmul.allow_tf32=old_tf32
        if torch.get_float32_matmul_precision()!=old_precision:
            raise MissingEvidence('FP32 probe did not restore float32 matmul precision')
    return {k:v.detach().to('cpu',copy=True) for k,v in dict(base=base,lora_delta=delta,total=total).items()},dict(input=metadata(x),native_output=metadata(total),gemm_calls=calls,
        reference_kind='functional FP32 with upcast stored base/LoRA weights, dropout eval; not original checkpoint FP32 model',
        fp32_tf32_allowed=False,precision_settings_restored=True)


def shape_controls(adapter,inputs,fresh,user='user_a'):
    import torch
    if not fresh: raise MissingEvidence('No fresh rows for shape control')
    rows=[]; full_values={}; stamps=parameter_stamps(adapter)
    for role,m in adapter.projection_modules(0).items():
        hidden=inputs[role]; index=torch.tensor(fresh,device=hidden.device,dtype=torch.long)
        subset=hidden.index_select(1,index)
        weights={n:tensor_sha(p) for n,p in m.named_parameters()}
        common=dict(layer=0,role=role,adapter=user,projection_weight_identity=digest(weights),
            actual_full_hidden=metadata(hidden),hidden_sha256=tensor_sha(hidden),fresh_positions=list(fresh),
            subset_values_equal_full_selection=bool(torch.equal(hidden.index_select(1,index),subset)))
        full,full_info=capture_call(m,hidden,user); full_values[role]=full
        controls={'FULL_NATIVE':(full,full_info)}
        controls['FULL_NATIVE_REPEAT']=capture_call(m,hidden,user)
        controls['SUBSET_NATIVE']=capture_call(m,subset,user)
        controls['FULL_THEN_SELECT']=capture_call(m,hidden,user,select_positions=fresh)
        f32,f32_info=fp32_call(m,hidden,user); f32subset,f32sub_info=fp32_call(m,subset,user)
        for label,(values,info) in controls.items():
            for component,value in values.items():
                whole=label in ('FULL_NATIVE','FULL_NATIVE_REPEAT')
                ref=full[component] if whole else full[component][:,fresh]
                rows.append(dict(common,control=label,component=component,native_call=info,
                    reference_scope='all' if whole else 'fresh',precision=str(hidden.dtype),
                    **tensor_metrics(ref,value,None if whole else fresh)))
        for component in full:
            rows.append(dict(common,control='FP32_FULL_VS_SUBSET',component=component,native_call=f32sub_info,full_call=f32_info,
                precision='torch.float32',reference_scope='fresh',**tensor_metrics(f32[component][:,fresh],f32subset[component],fresh)))
            rows.append(dict(common,control='FP32_REFERENCE_VS_NATIVE_FULL',component=component,native_call=full_info,full_call=f32_info,
                precision=str(hidden.dtype)+' vs upcast-weight FP32',reference_scope='all',
                **tensor_metrics(f32[component],full[component].float())))
    assert_parameters(stamps)
    return rows,full_values


def classify_h1(rows,controls,replays,total_layers,audit_all):
    """Conservative scalar-only evidence gates, testable without CUDA."""
    def all_exact(label,component=None):
        selected=[r for r in rows if r['control']==label and (component is None or r['component']==component)]
        expected={(r,c) for r in 'qkv' for c in (('base','lora_delta','total') if component is None else (component,))}
        return {(r['role'],r['component']) for r in selected}==expected and all(r['exact_equal'] for r in selected)
    subset=[r for r in rows if r['control']=='SUBSET_NATIVE']
    subset_complete={(r['role'],r['component']) for r in subset}=={(r,c) for r in 'qkv' for c in ('base','lora_delta','total')}
    supported=(all_exact('FULL_NATIVE_REPEAT') and all_exact('FULL_THEN_SELECT')
               and subset_complete and all(r.get('subset_values_equal_full_selection') is True for r in subset)
               and controls.get('shape_weight_identity_unchanged') and any(not r['exact_equal'] for r in subset) and controls.get('layer0_input_exact')
               and controls.get('full_repeat_exact') and controls.get('source_target_qkv_exact') and controls.get('cache_backing_exact'))
    full_replay=next((r for r in replays if r['full_shape_layers']==list(range(total_layers))),None)
    assembly_exact=all_exact('FULL_SHAPE_ASSEMBLY','total')
    causal=bool(supported and assembly_exact and audit_all and controls.get('subset_assembly_matches_ordinary')
        and controls.get('ordinary_strict_fail') and full_replay
        and full_replay['logits']['exact_equal'] and full_replay['actual_skipped_projection_rows']==0
        and full_replay['mapping_and_source_unchanged'] and full_replay['weight_identity_unchanged'])
    role_attribution=[]
    for role in 'qkv':
        base=[r for r in subset if r['role']==role and r['component']=='base']
        delta=[r for r in subset if r['role']==role and r['component']=='lora_delta']
        role_attribution.append(dict(role=role,base_difference_observed=bool(base) and any(not r['exact_equal'] for r in base),
            lora_delta_exact=bool(delta) and all(r['exact_equal'] for r in delta),
            initial_base_only_evidence=bool(base and delta) and any(not r['exact_equal'] for r in base) and all(r['exact_equal'] for r in delta)))
    precision=[]
    for role in 'qkv':
        for component in ('base','lora_delta','total'):
            a=next((r for r in subset if r['role']==role and r['component']==component),None)
            b=next((r for r in rows if r['control']=='FP32_FULL_VS_SUBSET' and r['role']==role and r['component']==component),None)
            if a and b: precision.append(dict(role=role,component=component,stored_precision_max_error=a['max_abs_error'],
                fp32_max_error=b['max_abs_error'],fp32_exact=b['exact_equal'],reduced_error_in_fp32=b['max_abs_error']<a['max_abs_error']))
    return dict(H1_A_supported=bool(supported),H1_B_assembly_control_exact=assembly_exact,
        H1_C_precision_controls=precision,H1_D_base_lora_attribution=role_attribution,
        causal_contribution_isolated=causal,kernel_mechanism_identified=False,
        kernel_evidence_status='NOT_COLLECTED; shapes/strides do not identify a cuBLAS algorithm or CUDA kernel',
        causal_gate='ordinary strict fail + all-layer full-native exact + unchanged source/weights/mapping/assembly + repeatability + complete layer audit',
        conclusion='CAUSAL_CONTRIBUTION_ISOLATED_FOR_CONTROLLED_FIXTURE' if causal else 'SHAPE_SENSITIVITY_SUPPORTED' if supported else 'UNRESOLVED_OR_NOT_OBSERVED')


def assembly_controls(runtime,inputs,fresh,full_values,hits):
    """Run Layer 0 through the ACTUAL existing Contiguous executor, not a copy.

    A one-layer adapter view is used only to satisfy the existing single-forward
    executor audit. The full original catalog/index/mapping and source QKV remain
    referenced. No target projection is admitted to any source cache.
    """
    import torch
    adapter=SimpleNamespace(model=runtime.model,layers=[runtime.adapter.layers[0]],
        projection_modules=lambda _:runtime.adapter.projection_modules(0))
    rows=[]; reports={}; assembled={}
    for label,full_layers in (('SUBSET_ASSEMBLY',set()),('FULL_SHAPE_ASSEMBLY',{0})):
        diagnostic={}
        flow=runtime.b.RequestMapping('contiguous',runtime.index,runtime.caches['contiguous'],runtime.namespace,
            runtime.weights,runtime.a.device,getattr(torch,runtime.a.dtype),diagnostic)
        flow.select(hits,runtime.ids,'user_a')
        with NativeShapeIntervention(adapter,full_layers) as intervention:
            with flow.context(runtime.original,adapter,'user_a',hits,len(runtime.ids)):
                with intervention.observer()(fresh):
                    for role,module in adapter.projection_modules(0).items():
                        values,info=capture_call(module,inputs[role])
                        if label=='SUBSET_ASSEMBLY': assembled[role]=values['total']
                        for component,actual in values.items():
                            positions=fresh if component!='total' and not full_layers else None
                            ref=full_values[role][component][:,fresh] if positions is not None else full_values[role][component]
                            rows.append(dict(layer=0,role=role,control=label,component=component,precision=runtime.a.dtype,
                                native_call=info,reference_scope='fresh' if positions is not None else 'all',
                                **tensor_metrics(ref,actual,positions)))
            evidence,_=flow.report()
            cost=intervention.accounting(len(runtime.ids),len(fresh))
        runtime.index.guard()
        reports[label]=dict(mapping=evidence,ledger=cost,source_snapshot=runtime.snapshot,
            source_is_actual_prefill=True,assembly_implementation='existing batched_v2_projection_path(contiguous=True)' if hits else 'B21 NO_REUSE_NATIVE')
    if reports['SUBSET_ASSEMBLY']['mapping']!=reports['FULL_SHAPE_ASSEMBLY']['mapping']:
        raise MissingEvidence('Assembly control mapping differs from subset mapping')
    return rows,reports,assembled
