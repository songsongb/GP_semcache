"""Additive single-user system model, not paper Eq.18 overlap or measured E2E."""
from .calibration import calibration_latency
from .common import tagged_ms
from .memory import ud_memory
from .network import communication
from .profiles import es_components, reuse_gate, validate_row, pair_key, validate_policy_match


def compare_request(native, lookup, physical, calibration, dims, *, bandwidth_mbps=200,
                    hidden_element_bytes=None, delta_element_bytes=None,
                    boundary_transfers=True, es_compute_policy='require-base-only',
                    allow_invalid_reuse=False):
    for row in (native, lookup, physical):
        validate_row(row)
    if native['mode'] != 'NATIVE_NO_CACHE' or lookup['mode'] != 'SEMCACHE_LOOKUP_NO_REUSE':
        raise ValueError('Need native and lookup-no-reuse baselines')
    if physical['mode'] not in ('SEMCACHE_PHYSICAL_REUSE', 'SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC'):
        raise ValueError('Need a physical-reuse profile')
    if pair_key(native) != pair_key(lookup) or pair_key(native) != pair_key(physical):
        raise ValueError('Cannot compare profiles with different request/model identities')
    validate_policy_match(lookup, physical)
    if native['model_id'] != dims.model_id:
        raise ValueError('Profile/config model mismatch')
    gate = reuse_gate(physical, allow_invalid_reuse)
    n, reused = physical['prompt_tokens'], physical['reused_tokens']
    dtype_bytes = {'float16': 2, 'bfloat16': 2, 'float32': 4}
    dtype = str(native['dtype']).removeprefix('torch.')
    if hidden_element_bytes is None:
        if dtype not in dtype_bytes:
            raise ValueError('Unknown activation dtype: supply explicit --hidden-element-bytes')
        hidden_element_bytes = dtype_bytes[dtype]
    if delta_element_bytes is None:
        delta_element_bytes = hidden_element_bytes  # explicit wire-choice; CPU math remains FP32
    net_args = dict(hidden_element_bytes=hidden_element_bytes, delta_element_bytes=delta_element_bytes,
                    bandwidth_mbps=bandwidth_mbps, boundary_transfers=boundary_transfers)
    edge_net = communication(dims, n, **net_args)
    reuse_net = communication(dims, n, reused, **net_args)
    saved_bytes = edge_net['request']['total_network_bytes'] - reuse_net['request']['total_network_bytes']
    m8_saved = physical.get('saved_communication_bytes')
    if m8_saved is not None and m8_saved != saved_bytes:
        raise ValueError('M8 saved_communication_bytes disagrees with tensor byte accounting; '
                         'check config dimensions and --hidden-element-bytes/--delta-element-bytes')
    edge_es, reuse_es = es_components(native, physical, es_compute_policy)
    edge_ud = calibration_latency(calibration, dims, n)
    reuse_ud = calibration_latency(calibration, dims, n-reused)
    # Use a measured exclusive residual, not a sum of overlapping M8 timers.
    # Tokenization is excluded symmetrically from both paths; prefill is retained.
    control = max(0., physical['request_wall_ms'] - physical['prefill_wall_ms'] - physical['tokenization_ms'])
    components = dict(es_compute_ms=edge_es, ud_lora_ms=edge_ud,
        network_ms=tagged_ms(edge_net['request']['total_network_ms'], 'SIMULATED', 'serialized tensor transfer model'),
        semcache_control_ms=tagged_ms(control, 'MEASURED', 'M8 request_wall - prefill_wall - tokenization'),
        semcache_es_ms=reuse_es, semcache_ud_ms=reuse_ud,
        semcache_network_ms=tagged_ms(reuse_net['request']['total_network_ms'], 'SIMULATED', 'remaining tensor transfer model'))
    edge = sum(components[k]['value_ms'] for k in ('es_compute_ms', 'ud_lora_ms', 'network_ms'))
    reuse = sum(components[k]['value_ms'] for k in ('semcache_control_ms', 'semcache_es_ms', 'semcache_ud_ms', 'semcache_network_ms'))
    compute_saved = edge_es['value_ms'] + edge_ud['value_ms'] - reuse_es['value_ms'] - reuse_ud['value_ms']
    network_saved = components['network_ms']['value_ms'] - components['semcache_network_ms']['value_ms']
    dependencies = tuple(c['provenance'] for c in components.values())
    for key, value in [('edge_lora_total_ms', edge), ('semcache_total_ms', reuse),
                       ('system_delta_ms', edge-reuse), ('compute_saved_ms', compute_saved),
                       ('communication_saved_ms', network_saved)]:
        components[key] = tagged_ms(value, 'SIMULATED', 'M9-A additive model arithmetic',
                                    signed=True, dependencies=dependencies)
    components['reuse_overhead_ms'] = tagged_ms(control, 'MEASURED',
        'exclusive SemCache control residual; GPU transfer/merge stays inside semcache_es_ms')
    # Preserve M8 source timings as measurements, including unused lookup baseline
    # and QKV diagnostics. They are NOT summed again into either modeled total.
    source_fields = ('request_wall_ms', 'prefill_wall_ms', 'prefill_gpu_ms',
                     'mixed_qkv_execution_ms', 'qkv_execution_ms')
    sources = {}
    for label, row in [('native', native), ('lookup_no_reuse', lookup), ('physical_reuse', physical)]:
        sources[label] = {key: tagged_ms(row[key], 'MEASURED', f'M8 {label}.{key}')
                          for key in source_fields if row.get(key) is not None}
    flat = dict(model=dims.model_id, hidden_size=dims.hidden_size, layers=dims.layers,
        rank=dims.rank, prompt_tokens=n, reused_tokens=reused, fresh_tokens=n-reused,
        reuse_ratio=reused/n, experiment_id=native['experiment_id'],
        query_id=native['query_id'], user_id=native['user_id'], repeat_index=native['repeat_index'],
        source_prompt_hash=native['prompt_token_ids_sha256'], bandwidth_mbps=bandwidth_mbps,
        hidden_element_bytes=hidden_element_bytes, delta_element_bytes=delta_element_bytes,
        es_compute_policy=es_compute_policy,
        es_profile_gpu=native.get('gpu_name'), es_profile_hostname=native.get('hostname'),
        es_profile_hardware_provenance='MEASURED',
        selected_action='REUSE' if reuse < edge else 'RECOMPUTE',
        decision_provenance='RESEARCH_EXTENSION', decision_applied_to_inference=False,
        communication_saved_bytes=saved_bytes, m8_saved_communication_bytes=m8_saved,
        m8_saved_bytes_checked=m8_saved is not None, **gate)
    for key, component in components.items():
        flat[key] = component['value_ms']
        flat[key + '_provenance'] = component['provenance']
    for prefix, net in [('edge', edge_net), ('semcache', reuse_net)]:
        for key, value in net['request'].items():
            if key.endswith('_bytes') or key.endswith('_ms'):
                flat[f'{prefix}_{key}'] = value
                flat[f'{prefix}_{key}_provenance'] = 'SIMULATED'
    memory = ud_memory(dims, n, parameter_bytes=calibration.get('parameter_element_bytes', 4),
                       temporary_element_bytes=calibration.get('temporary_element_bytes', 4))
    flat.update(ud_memory_usage_bytes=memory['estimated_usage_bytes'],
                ud_memory_capacity_bytes=memory['ud_memory_capacity_bytes'], ud_memory_fits=memory['fits'],
                ud_memory_provenance='SIMULATED')
    return dict(row=flat, components=components, es_source_timings=sources,
        communication=dict(edge_lora=edge_net, semcache=reuse_net), ud_memory=memory,
        scope='single_user_additive_prefill_model_not_measured_end_to_end',
        reproduction_choices=[
            'No ES/UD/network overlap; differs from paper max-overlap equation',
            'One-layer CPU calibration extrapolated to all layers at exact fresh-token count',
            'CPU FP32 LoRA; explicit wire precision; conversion cost not modeled',
            'Input embedding, output logits, tokenization and OS/runtime costs on UD excluded',
            'M8 local input/output computations remain in ES prefill proxy; placement is not emulated',
            'No protocol latency, queueing, contention, network serialization or Cloud',
            'Measured cache transfer/merge work remains in ES prefill, not counted twice in control',
            'PEFT proxy retains ES GPU LoRA while adding calibrated UD LoRA; not a base-only measurement'
                if es_compute_policy == 'peft-prefill-proxy' else 'Explicit base-only ES measurements required'])
