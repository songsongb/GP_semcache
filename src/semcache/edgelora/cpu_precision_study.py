"""Three predeclared CPU LoRA precision candidates; diagnostic use only.

This file deliberately does not match the production r3a_*.py identity glob.
Tensor/model dependencies are imported only by explicitly enabled workers.
"""
from contextlib import contextmanager
import os
from pathlib import Path

from .communication import digest_bytes, require
from .r3a_artifacts import (ENDPOINT, THRESHOLDS, append, check_parity, full_base_fingerprint,
    load_matrices, process_environment, read, sha, tensor_bytes, tensor_from_bytes,
    unpack, validate_prepared, versions, write)

CLASSIFICATION = 'R3A_PARITY_DIAGNOSTIC'
CANDIDATES = ('CPU_FP32_BASELINE', 'CPU_FP64_FULL', 'CPU_FP64_STAGED')
KEYS = {f'{layer}.{role}' for layer in range(32) for role in 'qkv'}
SERIALIZATION = 'contiguous row-major little-endian float32; whole all-token logits'


def fp64_pair(x, a, b, candidate, *, cast, linear, scale):
    """Shared, testable operation order. Callers supply CPU tensor operations."""
    require(candidate in CANDIDATES[1:], 'FP64 pair requires a predeclared FP64 candidate')
    intermediate = linear(cast(x, 'float64'), cast(a, 'float64'))
    if candidate == 'CPU_FP64_STAGED':
        intermediate = cast(cast(intermediate, 'float32'), 'float64')
    return cast(scale(linear(intermediate, cast(b, 'float64')), 1.0), 'float32')


def cpu_candidate(values, matrices, layer, hidden, candidate, *, precision_out=None):
    """Only CPU A/B matmuls. Baseline delegates directly to the frozen UD API."""
    import torch
    import torch.nn.functional as F
    from .user_device import cpu_deltas
    require(candidate in CANDIDATES and type(layer) is int and layer >= 0, 'invalid candidate/layer')
    require(values.device.type == 'cpu' and values.dtype == torch.float16 and values.ndim == 3 and
            values.shape[0] == 1 and values.shape[1] > 0 and values.shape[2] == hidden and
            bool(torch.isfinite(values).all()), 'CPU hidden dtype/shape/finite gate failed')
    for role in 'qkv':
        for kind, shape in (('A', [8, hidden]), ('B', [hidden, 8])):
            weight = matrices[layer, role, kind]
            require(weight.device.type == 'cpu' and weight.dtype == torch.float32 and list(weight.shape) == shape and
                    bool(torch.isfinite(weight).all()) and bool(weight.count_nonzero()),
                    'original nonzero CPU FP32 A/B shape/dtype/finite gate failed')
    with torch.inference_mode():
        if candidate == CANDIDATES[0]:
            # Do not replace this call by a numerically similar reconstruction.
            result = cpu_deltas(values, matrices, layer, hidden, torch_backend=True, precision_out=precision_out)
        else:
            outputs = []
            for role in 'qkv':
                a, b = matrices[layer, role, 'A'], matrices[layer, role, 'B']
                result_role = fp64_pair(values.to(torch.float32), a, b, candidate,
                    cast=lambda t, dtype: t.to(getattr(torch, dtype)), linear=F.linear, scale=lambda t, s: t*s)
                outputs.append(result_role)
                if precision_out is not None:
                    precision_out[role] = dict(hidden_dtype=str(values.dtype), input_to_A='torch.float64',
                        original_A_B_dtype='torch.float32', A_compute='torch.float64',
                        A_rounding='float32 then float64' if candidate == CANDIDATES[2] else 'none before B',
                        B_compute='torch.float64', returned_delta_dtype=str(result_role.dtype), scale=1.0,
                        scale_python_type='float', all_A_B_operations_device='cpu')
            result = torch.stack(outputs, dim=0)
    require(result.device.type == 'cpu' and result.dtype == torch.float32 and
            list(result.shape) == [3]+list(values.shape) and bool(torch.isfinite(result).all()),
            'CPU candidate returned invalid delta')
    return result


def account(records):
    require(len(records) == 96 and {f"{r['layer']}.{r['role']}" for r in records} == KEYS,
            'all 32 layers / 96 distinct QKV records required')
    for layer in range(32):
        block = [r for r in records if r['layer'] == layer]
        require(len({r['hidden']['sha256'] for r in block}) == 1, 'Q/K/V hidden inputs differ')
    return dict(layers=32, qkv_outputs=96,
        first_hidden_difference=next((dict(layer=r['layer'], role=r['role']) for r in records
                                      if not r['same_native_hidden']), None),
        first_qkv_difference=next((dict(layer=r['layer'], role=r['role']) for r in records
                                   if not r['native_qkv_sha_equal']), None),
        native_qkv_hashes_equal=sum(r['native_qkv_sha_equal'] for r in records))


def bind_matched_probe(actual, native, candidate):
    """Never assign same-input magnitudes to a changed incoming hidden."""
    same = actual['hidden']['sha256'] == native['hidden']['sha256']
    probe = native['candidates'][candidate]
    if not same:
        return dict(status='NOT_APPLICABLE_CHANGED_HIDDEN', same_hidden=False,
            fp16_flip_count=None, numeric_diff=None,
            interpretation='downstream changed input; native-hidden probe is a separate counterfactual')
    require(actual['base']['sha256'] == native['base']['sha256'] and
            actual['cpu_delta']['sha256'] == probe['cpu_delta']['sha256'] and
            actual['qkv']['sha256'] == probe['assembled']['sha256'],
            'MATCHED_INPUT_PROBE_BINDING_MISMATCH: CPU process/base/observer numerics require separate diagnosis')
    return dict(status='MATCHED_INPUT_MEASURED_AND_HASH_BOUND', same_hidden=True,
        fp16_flip_count=probe['fp16_flip_count'], numeric_diff=probe['assembled_vs_native'],
        delta_numeric_diff=probe['delta_vs_native_gpu'],
        interpretation='native-worker same-hidden measurement bound to actual candidate hidden/base/delta/output bytes')


def decide(native_exact, candidates):
    if not native_exact:
        return dict(conclusion='REFERENCE_NONREPRODUCIBLE', candidate_local_parity_pass=[],
                    next_action='stop; no candidate comparison is qualified')
    require(set(candidates) == set(CANDIDATES) and len(candidates) == 3, 'exactly three completed candidates required')
    for name in CANDIDATES:
        r = candidates[name]
        require(r['execution_status'] == 'COMPLETE' and r['parity']['thresholds'] == THRESHOLDS and
                r['accounting']['qkv_outputs'] == 96, 'incomplete candidate or changed strict thresholds')
    passed = [name for name in CANDIDATES if candidates[name]['parity']['passed']]
    # Multiple metrics are retained; there is no unsupported scalar "closest" ranking.
    return dict(conclusion='CANDIDATE_LOCAL_PARITY_PASS' if passed else 'NO_CPU_PRECISION_CANDIDATE_PASSED',
        candidate_local_parity_pass=passed,
        next_action='separate future physical CPU-UD/TCP strict validation; no production promotion' if passed else
                    'stop this three-candidate study; retain the negative result without further search',
        physical_tcp_parity_validated=False, PRIMARY_PERFORMANCE_ELIGIBLE=False)


def compare_logits(raw, reference_raw, shape):
    require(isinstance(shape,(list,tuple)) and len(shape)==3 and shape[0]==1 and
            all(type(v) is int and v>0 for v in shape),'invalid all-token logit shape')
    require(len(raw) == len(reference_raw) == 4*shape[0]*shape[1]*shape[2], 'full logit bytes/shape mismatch')
    return dict(logits=dict(sha256=digest_bytes(raw), bytes=len(raw), shape=shape, dtype='float32', serialization=SERIALIZATION),
                parity=check_parity(unpack(raw, 'float32'), unpack(reference_raw, 'float32'), shape))


def configure(plan, *, reference_required=True):
    require(plan.get('enable_gpu_probes') is True, 'explicit GPU study opt-in required')
    from .r3a_artifacts import ROOT
    for name, expected in plan['diagnostic_source_hashes'].items():
        p = (ROOT/name).resolve()
        require(p.is_relative_to(ROOT) and sha(p) == expected, 'diagnostic source changed during study')
    if reference_required:
        pm, ref, am = validate_prepared(plan['prepared'])
        require(pm.get('TEST_FIXTURE_ONLY') is not True and pm['identity']['model'] == 'facebook/opt-2.7b' and
                pm['identity']['endpoint'] == ENDPOINT and pm['identity']['layers'] == 32 and pm['identity']['length'] == 32,
                'fresh production OPT-2.7B L32 native preparation required')
        require(versions() == pm['identity']['versions'], 'same-cohort runtime versions changed')
        require(sha(Path(plan['prepared'])/'r3a_prepare_manifest.json') == plan['prepared_manifest_sha256'], 'prepare identity changed')
    from .r3a_runtime import runtime_setup, memory
    import torch
    available = sorted(os.sched_getaffinity(0))
    affinity = plan.get('cpu_affinity', available[:4])
    require(affinity and set(affinity).issubset(available), 'allocation CPU affinity unavailable')
    os.sched_setaffinity(0, affinity); torch.set_num_threads(min(4,len(affinity))); torch.set_num_interop_threads(1)
    runtime_setup(plan['device'])
    gpu = memory(plan['device'])
    require(gpu['uuid'] not in (None, '', 'UNAVAILABLE'), 'physical allocated GPU UUID unavailable')
    metadata = dict(pid=os.getpid(), parent_pid=os.getppid(), gpu=gpu, versions=versions(),
        environment=process_environment(), affinity=sorted(os.sched_getaffinity(0)), threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(), torch_config_sha256=digest_bytes(torch.__config__.show().encode()),
        torch_config_telemetry_only=True, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision())
    if not reference_required:return metadata
    require(gpu['uuid'] == plan['allocated_gpu_uuid'] == pm['gpu']['uuid'], 'same-allocation GPU UUID mismatch')
    torch.manual_seed(pm['identity']['seed'])
    return pm, ref, am, metadata


def forward(model, plan, output, ref, stage):
    import torch
    write(output/'gpu_execution.json', dict(gpu_inference_performed=True, stage=stage, pid=os.getpid()))
    with torch.inference_mode():
        result = model(input_ids=torch.tensor([ref['identity']['token_ids']], device=plan['device']),
                       use_cache=False, output_attentions=True)
        torch.cuda.synchronize(plan['device'])
    require(str(result.logits.dtype) == ref['native_logits_dtype'] and list(result.logits.shape) == ref['logits_shape'],
            'full native endpoint shape/dtype changed')
    raw = tensor_bytes(result.logits.float())
    reference_raw = (Path(plan['prepared'])/ref['logits_file']).read_bytes()
    result = compare_logits(raw, reference_raw, ref['logits_shape'])
    result['native_logits_dtype'] = ref['native_logits_dtype']
    return result


def verify_matrices(matrices, am):
    entries = {(r['layer'], r['role'], r['matrix']):r for r in am['matrices']}
    require(set(matrices) == set(entries) and all(digest_bytes(tensor_bytes(t)) == entries[k]['sha256']
            for k,t in matrices.items()), 'original CPU A/B mutated or mismatched')


def native_worker(plan, output):
    """One independent native forward, passive hooks, bounded per-layer state."""
    import torch
    from .parity_forensics_probe import load_native, tensor_info, tensor_metrics
    pm, ref, am, metadata = configure(plan)
    model, adapter, matrices = load_native(plan, pm, am)
    state = {}; bundle = {}; records = []; handles = []
    def part(key, kind):
        def observe(module, args, out):
            state.setdefault(key, {})[kind] = out.detach().clone()
            return None
        return observe
    def projection(layer, role):
        def observe(module, args, out):
            key=f'{layer}.{role}'; x=args[0]; observed=state.pop(key); base=observed['base']
            gpu_delta=observed['B']*module.scaling['user_a']
            require(str(x.dtype)=='torch.float16' and base.dtype==torch.float16 and gpu_delta.dtype==torch.float32,
                    'native precision path changed')
            require(tensor_bytes((base+gpu_delta).to(base.dtype)) == tensor_bytes(out), 'native GPU assembly mismatch')
            if not bundle:
                cpu_x=tensor_from_bytes(tensor_bytes(x),'float16',list(x.shape)); choices={};precisions={}
                for candidate in CANDIDATES:
                    precision={};choices[candidate]=cpu_candidate(cpu_x,matrices,layer,2560,candidate,precision_out=precision)
                    precisions[candidate]=precision
                bundle.update(layer=layer, hidden=x.detach().clone(), choices=choices, precisions=precisions, roles=set())
            require(bundle['layer']==layer and role not in bundle['roles'] and torch.equal(x,bundle['hidden']), 'native QKV bundle identity failed')
            record=dict(layer=layer,role=role,hidden=tensor_info(x),base=tensor_info(base),native_qkv=tensor_info(out),
                classification=CLASSIFICATION,PRIMARY_PERFORMANCE_ELIGIBLE=False,
                native_gpu_delta=tensor_info(gpu_delta),same_native_hidden=True,native_qkv_sha_equal=True,candidates={})
            matching=(record['hidden']['sha256']==ref['projections'][key]['hidden_sha256'] and
                    record['native_qkv']['sha256']==ref['projections'][key]['native_output_sha256'] and
                    record['native_gpu_delta']['sha256']==ref['projections'][key]['delta_sha256'])
            if not matching:write(output/'native_mismatch.json',dict(projection=key,observed=record,prepared=ref['projections'][key]))
            require(matching,
                    'REFERENCE_NONREPRODUCIBLE: original native hidden/QKV/delta differs')
            for candidate in CANDIDATES:
                delta=bundle['choices'][candidate]['qkv'.index(role)]; assembled=(base+delta.to(x.device)).to(base.dtype)
                record['candidates'][candidate]=dict(cpu_delta=tensor_info(delta),assembled=tensor_info(assembled),
                    delta_vs_native_gpu=tensor_metrics(delta,gpu_delta),assembled_vs_native=tensor_metrics(assembled,out),
                    fp16_flip_count=int((assembled!=out).sum()),precision=bundle['precisions'][candidate][role],
                    measurement_scope='SAME_NATIVE_HIDDEN_CPU_COUNTERFACTUAL; not the candidate propagated forward')
            records.append(record);append(output/'native_projection_probes.jsonl',record)
            bundle['roles'].add(role)
            if bundle['roles']==set('qkv'):bundle.clear()
            return None
        return observe
    try:
        for layer in range(32):
            for role,module in adapter.projection_modules(layer).items():
                key=f'{layer}.{role}'
                for kind,submodule in (('base',module.get_base_layer()),('A',module.lora_A['user_a']),('B',module.lora_B['user_a'])):
                    handles.append(submodule.register_forward_hook(part(key,kind)))
                handles.append(module.register_forward_hook(projection(layer,role)))
        metrics=forward(model,plan,output,ref,'independent_native_control')
        require(not state and not bundle,'incomplete native observations')
    finally:
        for handle in handles:handle.remove()
        state.clear();bundle.clear()
    write(output/'native_logit_comparison.json',metrics)
    require(metrics['logits']['sha256']==ref['logits_sha256'] and metrics['parity']['passed'], 'REFERENCE_NONREPRODUCIBLE: full native logits differ')
    verify_matrices(matrices,am)
    require(full_base_fingerprint(model.get_base_model(),peft_wrapped=True)['sha256']==pm['identity']['base_weight_sha256'], 'native base mutated')
    entries={(r['layer'],r['role'],r['matrix']):r for r in am['matrices']}
    for layer in range(32):
        for role,module in adapter.projection_modules(layer).items():
            for kind,part_module in (('A',module.lora_A['user_a']),('B',module.lora_B['user_a'])):
                require(digest_bytes(tensor_bytes(part_module.weight))==entries[layer,role,kind]['sha256'], 'native GPU A/B mutated')
    return dict(execution_status='COMPLETE',native_exact=True,process=metadata,projections=records,accounting=account(records),**metrics)


@contextmanager
def candidate_path(adapter, matrices, candidate, native, ref, output):
    import torch
    from .parity_forensics_probe import tensor_info, tensor_metrics
    from .user_device import cpu_deltas
    native_map={f"{r['layer']}.{r['role']}":r for r in native['projections']}
    originals=[];handles=[];bundle={};pending={};records=[];baseline_checks=[]
    def replacement(layer,role,original):
        def routed(x,*args,**kwargs):
            require(not args and not kwargs and x.dtype==torch.float16 and list(x.shape)==[1,32,2560], 'candidate hidden shape/dtype changed')
            if not bundle:
                cpu_x=tensor_from_bytes(tensor_bytes(x),'float16',list(x.shape));precision={}
                deltas=cpu_candidate(cpu_x,matrices,layer,2560,candidate,precision_out=precision)
                bundle.update(layer=layer,hidden=x.detach().clone(),deltas=deltas,precision=precision,roles=set())
                if candidate==CANDIDATES[0] and layer==0:
                    direct=cpu_deltas(cpu_x,matrices,layer,2560,torch_backend=True)
                    equal=tensor_bytes(deltas)==tensor_bytes(direct)
                    baseline_checks.append(dict(layer=layer,bundle_bitwise_equal_to_original_cpu_deltas=equal,
                        shape=list(deltas.shape),dtype=str(deltas.dtype),original_function='user_device.cpu_deltas(torch_backend=True)'))
                    require(equal,'baseline does not reproduce frozen CPU delta path')
            require(bundle['layer']==layer and role not in bundle['roles'] and torch.equal(x,bundle['hidden']), 'stale candidate QKV input')
            delta=bundle['deltas']['qkv'.index(role)]  # CPU FP32; no GPU A/B matmul
            base=original(x);returned=delta.to(x.device)
            require(base.dtype==torch.float16 and returned.dtype==torch.float32,'candidate ES add precision changed')
            predicted=(base.detach().clone()+returned.detach().clone()).to(base.dtype)
            pending[layer,role]=(x,base,delta,predicted,bundle['precision'][role])
            bundle['roles'].add(role)
            if bundle['roles']==set('qkv'):bundle.clear()
            return (base+returned).to(base.dtype)
        return routed
    def observer(layer,role):
        def observe(module,args,out):
            x,base,delta,predicted,precision=pending.pop((layer,role));key=f'{layer}.{role}'
            record=dict(layer=layer,role=role,hidden=tensor_info(x),base=tensor_info(base),cpu_delta=tensor_info(delta),
                classification=CLASSIFICATION,PRIMARY_PERFORMANCE_ELIGIBLE=False,
                qkv=tensor_info(out),precision=precision,assembly_comparison=tensor_metrics(out,predicted),
                same_native_hidden=digest_bytes(tensor_bytes(x))==ref['projections'][key]['hidden_sha256'],
                native_qkv_sha_equal=digest_bytes(tensor_bytes(out))==ref['projections'][key]['native_output_sha256'])
            require(tensor_bytes(out)==tensor_bytes(predicted),'actual candidate QKV differs from predicted assembly')
            record['matched_input_comparison']=bind_matched_probe(record,native_map[key],candidate)
            record['separate_native_hidden_counterfactual']=native_map[key]['candidates'][candidate]
            records.append(record);append(output/'candidate_projection_trace.jsonl',record)
            return None
        return observe
    try:
        for layer in range(32):
            for role,module in adapter.projection_modules(layer).items():
                require(type(module) is torch.nn.Linear and 'forward' not in module.__dict__ and
                        not module._forward_hooks and not module._forward_pre_hooks,'bare uninstrumented Linear required')
                originals.append(module);module.forward=replacement(layer,role,module.forward)
                handles.append(module.register_forward_hook(observer(layer,role)))
        yield records,baseline_checks
        require(not bundle and not pending,'incomplete candidate forward');account(records)
    finally:
        for handle in handles:handle.remove()
        for module in originals:del module.forward
        bundle.clear();pending.clear()


def candidate_worker(plan, output):
    import torch
    from .r3a_runtime import load_bare
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.system_cost.es_base_profile import assert_no_lora
    candidate=plan['candidate'];require(candidate in CANDIDATES,'unsupported candidate')
    pm,ref,am,metadata=configure(plan);model,tokenizer,meta=load_bare(pm['identity'],plan['device']);adapter=OPTModelAdapter(model)
    assert_no_lora(model,adapter)
    require(full_base_fingerprint(model)['sha256']==pm['identity']['base_weight_sha256'],'candidate bare weights differ')
    matrices=load_matrices(plan['prepared'],am,torch_backend=True)
    require(sha(plan['native_report'])==plan['native_report_sha256'],'native control report SHA changed')
    native=read(plan['native_report']);require(native['native_exact'] is True,'matching native control required')
    with torch.inference_mode(),candidate_path(adapter,matrices,candidate,native,ref,output) as (records,baseline_checks):
        metrics=forward(model,plan,output,ref,candidate)
    assert_no_lora(model,adapter);verify_matrices(matrices,am)
    require(full_base_fingerprint(model)['sha256']==pm['identity']['base_weight_sha256'],'candidate base mutated')
    return dict(execution_status='COMPLETE',candidate=candidate,process=metadata,projections=records,accounting=account(records),
        baseline_component_checks=baseline_checks,gpu_lora_on_candidate=False,all_A_B_operations_device='cpu',
        original_A_B_verified_after=True,physical_tcp_parity_validated=False,actual_network_socket_used=False,**metrics)
