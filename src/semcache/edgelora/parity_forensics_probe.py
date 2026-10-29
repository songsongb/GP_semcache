"""Opt-in numerical probes, excluded from production source_identity().

Native PEFT hooks only observe outputs. The independent ES diagnostic path is
never a production pilot/latency path. No tensor library loads at import time.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import time
import uuid

from .communication import Channel, connect, digest_bytes, header, require
from .r3a_artifacts import (THRESHOLDS, append, check_parity, full_base_fingerprint, load_matrices,
    process_environment, read, sha, tensor_bytes, tensor_from_bytes, unpack, validate_prepared, versions, write)
from .parity_forensics_audit import CLASSIFICATION


def tensor_info(tensor):
    import torch
    cpu=tensor.detach().cpu()
    require(bool(torch.isfinite(cpu).all()),'nonfinite diagnostic tensor')
    return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),device=str(tensor.device),
        stride=list(tensor.stride()),norm=float(cpu.double().norm()),sha256=digest_bytes(tensor_bytes(cpu)))


def tensor_metrics(candidate,reference):
    import torch
    require(candidate.shape==reference.shape and candidate.numel()>0,'diagnostic tensor shape mismatch')
    a=candidate.detach().cpu().contiguous();b=reference.detach().cpu().contiguous()
    require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),'nonfinite tensor comparison')
    ad,bd=a.double(),b.double();diff=(ad-bd).abs();norm=float(bd.norm())
    bitwise=None
    if a.dtype==b.dtype:
        bitwise=int((a.view(torch.uint8).reshape(a.numel(),a.element_size())!=
                     b.view(torch.uint8).reshape(b.numel(),b.element_size())).any(dim=1).sum())
    return dict(max_abs=float(diff.max()),relative_l2=float(diff.norm())/max(norm,1e-30),
        unequal_elements=int((ad!=bd).sum()),bitwise_unequal_elements=bitwise,
        elementwise_threshold_failures=int((diff>THRESHOLDS['atol']+THRESHOLDS['rtol']*bd.abs()).sum()),
        reference_norm=norm,candidate_norm=float(ad.norm()),elements=a.numel(),
        candidate_dtype=str(candidate.dtype),reference_dtype=str(reference.dtype),
        candidate_device=str(candidate.device),reference_device=str(reference.device),shape=list(a.shape),
        threshold_scope='frozen atol/rtol applied to components descriptively; production logit gate is unchanged')


def logit_metrics(candidate,reference,shape):
    result=check_parity(candidate,reference,shape)
    result['unequal_elements']=sum(a!=b for a,b in zip(candidate,reference))
    return result


def setup(plan):
    """Verify the immutable preparation before importing/initializing CUDA."""
    require(plan.get('enable_gpu_probes') is True,'explicit --enable-gpu-probes required')
    pm,ref,am=validate_prepared(plan['prepared'])
    require(pm.get('TEST_FIXTURE_ONLY') is not True,'GPU forensic probe requires original production preparation, not a fixture')
    require(versions()==pm['identity']['versions'],'runtime versions differ from original reference; restore matching environment')
    from .r3a_runtime import runtime_setup,memory
    import torch
    threads=plan['offline']['process_metadata']['original_ud_cpu']['effective_threads']
    require(type(threads) is int and 1<=threads<=4,'original effective CPU thread evidence missing')
    available=sorted(os.sched_getaffinity(0))
    require(len(available)>=threads,'allocation has fewer CPUs than original UD thread count')
    os.sched_setaffinity(0,available[:threads]);torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    runtime_setup(plan['device']);torch.manual_seed(pm['identity']['seed'])
    gpu=memory(plan['device']);original=plan['offline']['process_metadata']['original_es_gpu']
    require(original and original.get('uuid') not in (None,'UNAVAILABLE') and gpu['uuid']==original['uuid'],
            'matching original GPU UUID required for controlled attribution')
    return pm,ref,am,dict(pid=os.getpid(),parent_pid=os.getppid(),environment=process_environment(),versions=versions(),
        gpu=gpu,torch_threads=torch.get_num_threads(),effective_affinity=sorted(os.sched_getaffinity(0)),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        classification=CLASSIFICATION,PRIMARY_PERFORMANCE_ELIGIBLE=False)


def load_native(plan,pm,am):
    """Construct a wrapper, overwrite EVERY A/B from the original verified blob.

    Constructor initialization is never used for inference. No fixture generator
    is called. The whole bare base state and every exported matrix are rehashed.
    """
    import torch
    from peft import LoraConfig,get_peft_model
    from .r3a_runtime import load_bare
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.models.lora_decomposition import validate_projection
    identity=pm['identity'];base,tokenizer,meta=load_bare(identity,plan['device'])
    require(full_base_fingerprint(base)['sha256']==identity['base_weight_sha256'],'base weights differ from original')
    matrices=load_matrices(plan['prepared'],am,torch_backend=True)
    model=get_peft_model(base,LoraConfig(r=8,lora_alpha=8,lora_dropout=0.,bias='none',
        target_modules=['q_proj','k_proj','v_proj'],task_type='CAUSAL_LM'),adapter_name='user_a',autocast_adapter_dtype=True)
    model.set_adapter('user_a');model.eval();adapter=OPTModelAdapter(model)
    records={(r['layer'],r['role'],r['matrix']):r for r in am['matrices']}
    with torch.no_grad():
        for layer in range(len(adapter.layers)):
            for role,module in adapter.projection_modules(layer).items():
                validate_projection(module,'user_a')
                require(module.scaling['user_a']==am['scale']==1.,'native LoRA scale mismatch')
                for name,part in (('A',module.lora_A['user_a']),('B',module.lora_B['user_a'])):
                    src=matrices[layer,role,name];record=records[layer,role,name]
                    require(part.bias is None and part.weight.dtype==src.dtype==torch.float32 and
                            list(part.weight.shape)==record['shape'],'actual native A/B dtype/shape mismatch')
                    part.weight.copy_(src.to(part.weight.device))
                    require(digest_bytes(tensor_bytes(part.weight))==record['sha256'],'actual native matrix SHA differs from original')
    require(full_base_fingerprint(model.get_base_model(),peft_wrapped=True)['sha256']==identity['base_weight_sha256'],
            'base changed during diagnostic PEFT attachment')
    return model,adapter,matrices


@contextmanager
def passive_hashes(adapter):
    records={};handles=[];order=[]
    def hook(layer,role):
        def observe(module,args,out):
            key=f'{layer}.{role}';require(key not in records,'repeated native projection')
            order.append(key)
            records[key]=dict(hidden_sha256=digest_bytes(tensor_bytes(args[0])),
                              native_output_sha256=digest_bytes(tensor_bytes(out)))
            return None
        return observe
    try:
        for layer in range(len(adapter.layers)):
            for role,module in adapter.projection_modules(layer).items():handles.append(module.register_forward_hook(hook(layer,role)))
        yield records,order
        require(len(records)==3*len(adapter.layers),'native control did not observe all QKV outputs')
    finally:
        for handle in handles:handle.remove()


def _native_forward(model,identity,device,output,stage):
    import torch
    write(output/'gpu_execution.json',dict(gpu_inference_performed=True,stage=stage,pid=os.getpid()))
    with torch.inference_mode():
        result=model(input_ids=torch.tensor([identity['token_ids']],device=device),use_cache=False,output_attentions=True)
        torch.cuda.synchronize(device)
    raw=tensor_bytes(result.logits.float());shape=list(result.logits.shape);dtype=str(result.logits.dtype)
    del result
    return unpack(raw,'float32'),shape,digest_bytes(raw),dtype


@contextmanager
def same_hidden_probes(adapter,matrices,identity,selected,output,*,fp64_control=False,capture_qkv=False):
    import torch
    import torch.nn.functional as F
    from semcache.models.lora_decomposition import projection_parts
    from .user_device import cpu_deltas
    records=[];handles=[];state={};cpu_layer={};enabled=True;order=[]
    def part_hook(key,kind):
        def observe(module,args,out):
            if enabled:state.setdefault(key,{})[kind]=out.detach().clone()
            return None
        return observe
    def projection_hook(layer,role):
        def observe(module,args,out):
            nonlocal enabled
            key=f'{layer}.{role}';order.append(key)
            if layer not in selected:
                return None
            x=args[0];observed=state.pop(key)
            native_base=observed['base'];native_delta=observed['B']*module.scaling['user_a']
            if not cpu_layer:
                cpu_hidden=tensor_from_bytes(tensor_bytes(x),identity['hidden_dtype'],list(x.shape))
                precision={};bundle=cpu_deltas(cpu_hidden,matrices,layer,identity['hidden_size'],torch_backend=True,precision_out=precision)
                raw=tensor_bytes(bundle);decoded=tensor_from_bytes(raw,identity['delta_dtype'],list(bundle.shape))
                cpu_layer.update(layer=layer,hidden=x.detach().clone(),cpu_hidden=cpu_hidden,bundle=bundle,decoded=decoded,
                    raw_sha256=digest_bytes(raw),serialization_byte_identity=tensor_bytes(decoded)==raw,roles=set(),precision=precision)
            require(cpu_layer['layer']==layer and torch.equal(cpu_layer['hidden'],x),'native QKV inputs differ; bundled comparison unsafe')
            cpu_layer['roles'].add(role);idx='qkv'.index(role)
            cpu_delta=cpu_layer['bundle'][idx];returned=cpu_layer['decoded'][idx].to(x.device)
            enabled=False
            try:base,gpu_delta,gpu_combined=projection_parts(module,x,'user_a')
            finally:enabled=True
            cpu_combined=(base+returned).to(base.dtype)
            observed_sum=(native_base+native_delta).to(native_base.dtype)
            cpu_a=F.linear(cpu_layer['cpu_hidden'].to(matrices[layer,role,'A'].dtype),matrices[layer,role,'A'])
            comparisons=dict(native_vs_gpu_reconstruction=tensor_metrics(gpu_combined,out),
                native_vs_observed_parts_sum=tensor_metrics(observed_sum,out),
                observed_vs_recomputed_base=tensor_metrics(base,native_base),
                observed_vs_recomputed_gpu_delta=tensor_metrics(gpu_delta,native_delta),
                gpu_delta_vs_cpu_delta=tensor_metrics(cpu_delta,native_delta),
                gpu_A_vs_cpu_A=tensor_metrics(cpu_a,observed['A']),
                native_vs_cpu_assembled=tensor_metrics(cpu_combined,out),
                cpu_delta_vs_unpacked_cpu=tensor_metrics(cpu_layer['decoded'][idx],cpu_delta),
                cpu_delta_vs_unpacked_cuda=tensor_metrics(returned,cpu_delta))
            record=dict(layer=layer,role=role,call_index=len(order)-1,comparisons=comparisons,
                hidden=tensor_info(x),native_output=tensor_info(out),native_base=tensor_info(native_base),
                independent_gpu_base=tensor_info(base),native_gpu_delta=tensor_info(native_delta),cpu_delta=tensor_info(cpu_delta),
                gpu_reconstructed=tensor_info(gpu_combined),predicted_cpu_assembled=tensor_info(cpu_combined),
                wire_roundtrip_delta=tensor_info(returned),cpu_precision=cpu_layer['precision'][role],
                fp16_output_flip_count=int((gpu_combined.to(torch.float16)!=cpu_combined.to(torch.float16)).sum()),
                serialization_byte_identity=cpu_layer['serialization_byte_identity'],
                bundle_shape=list(cpu_layer['bundle'].shape),bundle_role_order=list('qkv'),
                bundle_payload_sha256=cpu_layer['raw_sha256'],native_forward_output_replaced=False,
                instrumentation_excluded_from_performance=True)
            if fp64_control:
                a=matrices[layer,role,'A'].double();b=matrices[layer,role,'B'].double()
                control=F.linear(F.linear(cpu_layer['cpu_hidden'].double(),a),b).float()
                final=(base+control.to(base.device)).to(base.dtype)
                record['fp64_cpu_control']=dict(label='DIAGNOSTIC_CPU_FP64_FROM_ORIGINAL_FP32_WEIGHTS_NOT_PRODUCTION',
                    delta_vs_gpu=tensor_metrics(control,native_delta),final_vs_native=tensor_metrics(final,out))
            if capture_qkv and layer in (0,1):
                name=f'verified_candidate_qkv_L{layer}_{role}.bin'
                (output/name).write_bytes(tensor_bytes(out))
                record['optional_qkv_capture']=dict(file=name,sha256=sha(output/name),shape=list(out.shape),
                    dtype=str(out.dtype).removeprefix('torch.'),qualified=False)
            records.append(record);append(output/'same_hidden_projection_metrics.jsonl',record)
            if cpu_layer['roles']==set('qkv'):cpu_layer.clear()
            return None
        return observe
    try:
        for layer in range(len(adapter.layers)):
            for role,module in adapter.projection_modules(layer).items():
                key=f'{layer}.{role}'
                if layer in selected:
                    for kind,part in (('base',module.get_base_layer()),('A',module.lora_A['user_a']),('B',module.lora_B['user_a'])):
                        handles.append(part.register_forward_hook(part_hook(key,kind)))
                handles.append(module.register_forward_hook(projection_hook(layer,role)))
        yield records,order
        require(not state and not cpu_layer and len(records)==3*len(selected),'incomplete passive probe coverage')
    finally:
        for handle in handles:handle.remove()
        state.clear();cpu_layer.clear()


def native_worker(plan,output,*,components):
    pm,ref,am,metadata=setup(plan);identity=pm['identity']
    model,adapter,matrices=load_native(plan,pm,am)
    original=unpack((Path(plan['prepared'])/ref['logits_file']).read_bytes(),ref['logits_dtype'])
    controls=[];previous=None;repeat=None
    for number in range(2):
        with passive_hashes(adapter) as (hashes,order):
            values,shape,blob_sha,dtype=_native_forward(model,identity,plan['device'],output,f'native_control_{number}')
        require(shape==ref['logits_shape'] and dtype==ref['native_logits_dtype'],'native endpoint shape/dtype changed')
        mismatches=[dict(projection=k,field=f,original=ref['projections'][k][f],observed=v[f])
            for k,v in hashes.items() for f in ('hidden_sha256','native_output_sha256') if v[f]!=ref['projections'][k][f]]
        controls.append(dict(label=f'{output.name}:native_control_{number}',vs_original=logit_metrics(values,original,shape),
            logits_sha256=blob_sha,original_logits_sha256=ref['logits_sha256'],projection_hashes=hashes,
            original_projection_mismatches=mismatches,role_order=order))
        if previous is not None:repeat=logit_metrics(values,previous,shape)
        previous=values
    projections=[];instrumented=None
    if components:
        with same_hidden_probes(adapter,matrices,identity,set(plan['layers']),output,
                fp64_control=plan['cpu_fp64_control'],capture_qkv=plan['capture_qkv_reference']) as (projections,order):
            values,shape,blob_sha,dtype=_native_forward(model,identity,plan['device'],output,'native_passive_same_hidden_probes')
        instrumented=dict(vs_original=logit_metrics(values,original,shape),vs_control=logit_metrics(values,previous,shape),
                          logits_sha256=blob_sha,role_order=order)
        for p in projections:
            key=f"{p['layer']}.{p['role']}"
            p['matches_original_native_hidden']=p['hidden']['sha256']==ref['projections'][key]['hidden_sha256']
            p['matches_original_native_output']=p['native_output']['sha256']==ref['projections'][key]['native_output_sha256']
            p['matches_original_gpu_delta']=p['native_gpu_delta']['sha256']==ref['projections'][key]['delta_sha256']
            if 'optional_qkv_capture' in p:
                p['optional_qkv_capture']['qualified']=(p['matches_original_native_hidden'] and p['matches_original_native_output'] and
                    instrumented['vs_original']['unequal_elements']==0 and all(c['vs_original']['unequal_elements']==0 for c in controls))
    require(full_base_fingerprint(model.get_base_model(),peft_wrapped=True)['sha256']==identity['base_weight_sha256'],
            'base state changed during passive diagnostic')
    # Rehash actual GPU A/B again; passive observation must never mutate them.
    entries={(r['layer'],r['role'],r['matrix']):r for r in am['matrices']}
    for layer in range(len(adapter.layers)):
        for role,module in adapter.projection_modules(layer).items():
            for kind,part in (('A',module.lora_A['user_a']),('B',module.lora_B['user_a'])):
                require(digest_bytes(tensor_bytes(part.weight))==entries[layer,role,kind]['sha256'],'native A/B changed during diagnostic')
    return dict(execution_status='COMPLETE',process_metadata=metadata,controls=controls,repeat_comparison=repeat,
        instrumented_forward=instrumented,projections=projections,original_weights_verified=True,
        new_adapter_generated=False,performance_timing_eligible=False,raw_tensor_dumps=bool(plan['capture_qkv_reference']))


@contextmanager
def diagnostic_es_path(adapter,session,matrices,identity,output,captures):
    """Independent diagnostic-only routing. It cannot be passed to runner 95.

    Real UD replies drive the forward. Local CPU matrices only VERIFY the wire;
    they never supply the returned correction. No A/B is loaded on the ES GPU.
    """
    import torch
    from .user_device import cpu_deltas
    originals=[];handles=[];records=[];bundles={};pending={};order=[]
    def replace(layer,role,native):
        def forward(x,*args,**kwargs):
            require(not args and not kwargs,'unsupported diagnostic projection arguments')
            delta=session.correction(layer,role,x)
            if not bundles:
                cpu_x=tensor_from_bytes(tensor_bytes(x),identity['hidden_dtype'],list(x.shape))
                expected=cpu_deltas(cpu_x,matrices,layer,identity['hidden_size'],torch_backend=True)
                bundles.update(layer=layer,expected=expected,hidden=x.detach().clone(),roles=set())
            require(bundles['layer']==layer and torch.equal(bundles['hidden'],x),'diagnostic hidden input changed')
            base=native(x);assembled=(base+delta).to(base.dtype)
            idx='qkv'.index(role);local=bundles['expected'][idx]
            expected_assembly=(base+local.to(base.device)).to(base.dtype)
            order.append(dict(layer=layer,role=role))
            pending[layer,role]=dict(layer=layer,role=role,hidden=tensor_info(x),base=tensor_info(base),
                received_delta=tensor_info(delta),local_cpu_delta=tensor_info(local),
                assembled=tensor_info(assembled),cpu_verification_comparison=tensor_metrics(delta,local),
                wire_matches_local_cpu=tensor_bytes(delta)==tensor_bytes(local),
                assembly_vs_local_cpu=tensor_metrics(assembled,expected_assembly),
                local_cpu_computation_scope='diagnostic verifier only; real CPU UD reply drives output; no GPU A/B on ES')
            bundles['roles'].add(role)
            if bundles['roles']==set('qkv'):bundles.clear()
            return assembled
        return forward
    def observe(layer,role):
        def hook(module,args,out):
            record=pending.pop((layer,role));observed=tensor_info(out)
            record['observed_output']=observed
            record['observed_output_matches_assembly']=observed['sha256']==record['assembled']['sha256']
            capture=captures.get(f'{layer}.{role}')
            if capture and capture['qualified']:
                path=Path(capture['directory'])/capture['file'];require(sha(path)==capture['sha256'],'optional reference capture SHA mismatch')
                native=tensor_from_bytes(path.read_bytes(),capture['dtype'],capture['shape'])
                record['optional_native_qkv_magnitude']=tensor_metrics(out,native)
                record['optional_native_qkv_magnitude']['same_hidden']=record['hidden']['sha256']==capture['hidden_sha256']
            records.append(record);append(output/'diagnostic_es_qkv.jsonl',record)
            return None
        return hook
    try:
        for layer in range(len(adapter.layers)):
            for role,module in adapter.projection_modules(layer).items():
                require(type(module) is torch.nn.Linear and 'forward' not in module.__dict__ and
                        not module._forward_hooks and not module._forward_pre_hooks,'bare uninstrumented ES projection required')
                originals.append(module);module.forward=replace(layer,role,module.forward)
                handles.append(module.register_forward_hook(observe(layer,role)))
        yield records,order
        session.complete();require(not pending and not bundles,'unfinished ES assembly')
    finally:
        for h in handles:h.remove()
        for module in originals:del module.forward
        pending.clear();bundles.clear()


def es_worker(plan,output):
    pm,ref,am,metadata=setup(plan);identity=pm['identity']
    from .r3a_runtime import load_bare
    from .edge_server import ProjectionSession
    from .collaborative_inference import launch,wait_ready,cleanup
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.system_cost.es_base_profile import assert_no_lora
    model,tokenizer,model_meta=load_bare(identity,plan['device']);adapter=OPTModelAdapter(model);assert_no_lora(model,adapter)
    require(full_base_fingerprint(model)['sha256']==identity['base_weight_sha256'],'ES diagnostic base fingerprint mismatch')
    matrices=load_matrices(plan['prepared'],am,torch_backend=True)
    captures={}
    native_report=plan.get('native_report')
    if native_report:
        nr=read(native_report)
        for p in nr.get('projections',[]):
            if 'optional_qkv_capture' in p:
                captures[f"{p['layer']}.{p['role']}"]=dict(p['optional_qkv_capture'],directory=str(Path(native_report).parent),hidden_sha256=p['hidden']['sha256'])
    session_id=uuid.uuid4().hex
    ud_plan=dict(prepared=plan['prepared'],output=str(output),session_id=session_id)
    write(output/'ud_plan.json',ud_plan)
    children=[];channel=None;records=[];result={}
    try:
        child=launch('ud',output/'ud_plan.json',output);children.append(('ud',child))
        ready=wait_ready(child[0],output/'r3a_ud_ready.json')
        channel=Channel(connect(ready['port']),session_id,am['matrix_sha256'],trace=lambda r:append(output/'diagnostic_es_wire.jsonl',r))
        channel.send(header('HELLO','handshake',-1,pid=os.getpid(),identity_sha256=pm['identity_sha256']))
        response,_,_=channel.recv(kind='READY',request_id='handshake',layer_id=-1)
        require(response['metadata']['pid']==ready['pid']==child[0].pid,'diagnostic UD PID mismatch')
        session=ProjectionSession(channel,identity,'diagnostic_000000',torch_backend=True)
        import torch
        write(output/'gpu_execution.json',dict(gpu_inference_performed=True,stage='separate_diagnostic_ES_forward',pid=os.getpid()))
        with torch.inference_mode(),diagnostic_es_path(adapter,session,matrices,identity,output,captures) as (records,order):
            outputs=model(input_ids=torch.tensor([identity['token_ids']],device=plan['device']),use_cache=False,output_attentions=True)
            torch.cuda.synchronize(plan['device'])
        require(str(outputs.logits.dtype)==ref['native_logits_dtype'] and list(outputs.logits.shape)==ref['logits_shape'],
                'diagnostic ES endpoint dtype/shape differs from native reference')
        values=unpack(tensor_bytes(outputs.logits.float()),'float32');del outputs
        original=unpack((Path(plan['prepared'])/ref['logits_file']).read_bytes(),ref['logits_dtype'])
        for p in records:
            key=f"{p['layer']}.{p['role']}";original_p=ref['projections'][key]
            p['original_native_hidden_sha256']=original_p['hidden_sha256']
            p['original_native_qkv_sha256']=original_p['native_output_sha256']
            p['hidden_matches_original_native']=p['hidden']['sha256']==original_p['hidden_sha256']
            p['output_matches_original_native']=p['observed_output']['sha256']==original_p['native_output_sha256']
            layer_trace=session.trace[p['layer']]
            p['sent_hidden_sha256']=layer_trace['hidden_payload_sha256']
            p['received_bundled_delta_sha256']=layer_trace['delta_payload_sha256']
        # A failed logit comparison is an observation here, not benchmark admission.
        result=dict(execution_status='COMPLETE',process_metadata=metadata,ud_pid=ready['pid'],
            actual_network_socket_used=True,actual_remote_network_measured=False,
            gpu_lora_on_es=False,cpu_verification_on_es=True,placement_scope='diagnostic only; real UD reply drives each correction',
            vs_original_logits=logit_metrics(values,original,ref['logits_shape']),projections=records,role_order=order,
            layers=session.trace,expected_layer_count=identity['layers'],observed_qkv_outputs=len(records),
            original_es_failure_reproduced=not logit_metrics(values,original,ref['logits_shape'])['passed'])
        channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE',request_id='shutdown',layer_id=-1)
        child[0].wait(timeout=10.);require(child[0].returncode==0,'diagnostic UD exited unsuccessfully')
        ud=read(output/'r3a_ud_result.json')
        require(ud['user_device_cuda_used'] is False and ud['cpu']['effective_threads']==
            plan['offline']['process_metadata']['original_ud_cpu']['effective_threads'],'diagnostic UD placement/thread mismatch')
        result['ud_process_metadata']=dict(pid=ud['pid'],cpu=ud['cpu'],environment=ud['environment'],user_device_cuda_used=False)
        return result
    finally:
        if channel is not None:channel.close()
        result['cleanup']=cleanup(children)
        write(output/'diagnostic_es_partial.json',result)
