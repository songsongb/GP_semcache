"""CPU-only evidence gates for the diagnostic all-layer CPU-delta intervention.

No model imports, inferred tensor magnitudes, or production admission changes.
"""
from pathlib import Path

from .communication import digest, json_loads, require
from .r3a_artifacts import ENDPOINT, THRESHOLDS, sha
from .parity_forensics_audit import CLASSIFICATION

SERIALIZATION = 'little-endian contiguous row-major IEEE754 float32; native logits promoted without arithmetic'
EXPECTED_KEYS = {f'{layer}.{role}' for layer in range(32) for role in 'qkv'}


def identity_record(prepared, pilot, pm, ref, offline):
    observations=[json_loads(line) for line in (Path(pilot)/'r3a_iterations.jsonl').read_bytes().splitlines() if line.strip()]
    return dict(prepared_manifest_sha256=sha(Path(prepared)/'r3a_prepare_manifest.json'),
        pilot_manifest_sha256=sha(Path(pilot)/'r3a_manifest.json'),
        identity_sha256=pm['identity_sha256'], identity=pm['identity'],
        reference_logits_sha256=ref['logits_sha256'], reference_file_sha256=sha(Path(prepared)/'r3a_reference.json'),
        reference_logits_shape=ref['logits_shape'],
        adapter_matrix_sha256=pm['identity']['adapter_matrix_sha256'],
        production_source_sha256=pm['code']['sha256'],
        gpu_uuid=offline['process_metadata']['original_es_gpu']['uuid'],
        cpu_affinity=offline['process_metadata']['original_ud_cpu']['effective_affinity'],
        cpu_threads=offline['process_metadata']['original_ud_cpu']['effective_threads'],
        pilot_parity={k:offline['parity'][k] for k in ('max_abs','relative_l2','tolerance_violations',
            'last_token_kl','last_token_argmax_agreement','passed')},
        pilot_layer_payload_hashes=[dict(layer=l['layer_id'],hidden=l['hidden_payload_sha256'],delta=l['delta_payload_sha256'])
            for l in observations[0]['layers']],
        thresholds=dict(THRESHOLDS))


def validate_identity(record, expected):
    require(record == expected, 'causal run/prepared/reference/GPU/CPU identity mismatch')
    require(record['thresholds'] == THRESHOLDS, 'frozen thresholds changed')
    identity = record['identity']
    require(identity['layers'] == 32 and identity['length'] == 32 and
            identity['model'] == 'facebook/opt-2.7b' and identity['dtype'] == 'float16' and
            identity['hidden_dtype'] == 'float16' and identity['delta_dtype'] == 'float32' and
            identity['adapter'] == 'user_a' and identity['rank'] == 8 and identity['hidden_size'] == 2560 and
            identity['endpoint'] == ENDPOINT and len(identity['token_ids']) == 32 and
            identity['token_ids_sha256'] == digest(identity['token_ids']),
            'causal attribution requires the original OPT-2.7B L32 FP16 rank8 endpoint')
    require(record['identity_sha256'] == digest(identity), 'identity digest mismatch')
    require(record['gpu_uuid'] not in (None, '', 'UNAVAILABLE'), 'original GPU UUID missing')


def projection_map(records):
    result = {}
    for record in records:
        key = f"{record['layer']}.{record['role']}"
        require(key in EXPECTED_KEYS and key not in result, 'duplicate/invalid causal projection')
        result[key] = record
    require(set(result) == EXPECTED_KEYS, 'all 32 layers / 96 distinct QKV outputs are required')
    for layer in range(32):
        require(len({result[f'{layer}.{role}']['hidden']['sha256'] for role in 'qkv'}) == 1,
                'Q/K/V hidden input identity differs within a layer')
    return result


def descriptor_valid(d):
    require(d['dtype'] == 'float32' and d['serialization'] == SERIALIZATION and
            len(d['shape']) == 3 and d['shape'][:2] == [1, 32] and
            type(d['shape'][2]) is int and d['shape'][2] > 0 and
            d['bytes'] == 4*32*d['shape'][2] and len(d['sha256']) == 64 and
            all(c in '0123456789abcdef' for c in d['sha256']), 'invalid canonical full-logit descriptor')
    return d


def _controls_exact(worker):
    return control_failure(worker) is None


def control_failure(worker):
    controls = worker.get('controls', [])
    for c in controls:
        descriptor_valid(c['logits'])
        require(c['logits']['shape'] == worker['comparison_identity']['reference_logits_shape'],
                'native control logit shape differs from original')
    reproduced = (len(controls) == 3 and any(c['label'] == 'native_no_hooks' for c in controls) and
        all(c['vs_original']['passed'] is True and c['vs_original']['unequal_elements'] == 0 and
            c['logits']['sha256'] == worker['comparison_identity']['reference_logits_sha256'] and
            not c.get('original_projection_mismatches') for c in controls) and
        all(set(c['projection_hashes']) == EXPECTED_KEYS for c in controls if c['label'] != 'native_no_hooks'))
    if not reproduced:return 'REFERENCE_NONREPRODUCIBLE'
    if worker.get('all_gpu_reconstructions_exact') is not True or worker.get('gpu_reconstruction_count') != 96:
        return 'GPU_RECONSTRUCTION_MISMATCH'
    return None


def audit_prior_forensics(directory, evidence, offline, pm):
    """Read existing 96 evidence; do not upgrade its absent full-logit capture."""
    directory = Path(directory).resolve()
    report = evidence.read(directory/'r3a_parity_forensics.json')
    require(report['execution_status'] == report['gpu_findings_status'] == 'COMPLETE' and
            report['classification'] == CLASSIFICATION and report['PRIMARY_PERFORMANCE_ELIGIBLE'] is False,
            'completed diagnostic-only original forensic report required')
    require(report['prepared_manifest_sha256'] == offline['prepared_manifest_sha256'] and
            report['production_source_identity']['files'] == pm['code']['files'], 'prior forensic preparation/source mismatch')
    for field in ('prepared_manifest_sha256', 'pilot_manifest_sha256', 'reference_identity', 'parity', 'actual_schedule'):
        require(report['offline_audit'][field] == offline[field], 'prior forensic audit identity mismatch: '+field)
    # Original diagnostic sources must also be preserved, rather than trusting a new observer silently.
    from .r3a_artifacts import ROOT
    require(report.get('diagnostic_source_hashes'), 'prior forensic diagnostic source hashes missing')
    for name, expected in report['diagnostic_source_hashes'].items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT), 'invalid prior diagnostic source path')
        evidence.file(path, expected)
    require(report.get('diagnostic_output_hashes'), 'prior forensic output hashes missing')
    require({f'{folder}/{name}' for folder in ('native-a','native-b','es') for name in ('worker_result.json','lifecycle.json')}.
            issubset(report['diagnostic_output_hashes']), 'prior required worker hashes missing')
    for name, expected in report['diagnostic_output_hashes'].items():
        path = (directory/name).resolve()
        require(path.is_relative_to(directory), 'prior evidence path escape')
        evidence.file(path, expected)
    for key, folder in (('native_a', 'native-a'), ('native_b', 'native-b'), ('es_replay', 'es')):
        worker = evidence.read(directory/folder/'worker_result.json')
        require(worker == {k:v for k,v in report[key].items() if k != 'independent_process_cleanup'},
                'prior embedded/worker result mismatch: '+key)
        life = evidence.read(directory/folder/'lifecycle.json')
        require(life['all_children_exited'] is True and all(c['returncode'] == 0 and not c['forced_termination']
                for c in life['children']), 'prior worker did not exit cleanly')
    for key in ('native_a', 'native_b'):
        native = report[key]
        require(native['process_metadata']['gpu']['uuid'] == offline['process_metadata']['original_es_gpu']['uuid'] and
                native['process_metadata']['effective_affinity'] == offline['process_metadata']['original_ud_cpu']['effective_affinity'],
                'prior native GPU UUID/CPU affinity mismatch')
        require(len(native['controls']) == 2 and all(c['vs_original']['passed'] is True and
            c['vs_original']['unequal_elements'] == 0 and c['logits_sha256'] == pm['artifact_hashes']['r3a_reference_logits.bin'] and
            set(c['projection_hashes']) == EXPECTED_KEYS and not c['original_projection_mismatches'] for c in native['controls']) and
            native['repeat_comparison']['unequal_elements'] == 0, 'prior native controls are not exact')
        if native.get('instrumented_forward'):
            require(native['instrumented_forward']['vs_control']['unequal_elements'] == 0,
                    'prior native instrumentation changed outputs')
    replay = report['es_replay']
    require(replay['actual_network_socket_used'] is True and replay['gpu_lora_on_es'] is False and
            replay['ud_process_metadata']['user_device_cuda_used'] is False and
            replay['process_metadata']['gpu']['uuid'] == offline['process_metadata']['original_es_gpu']['uuid'],
            'prior ES placement/GPU identity mismatch')
    require(replay['process_metadata']['effective_affinity'] == replay['ud_process_metadata']['cpu']['effective_affinity'] ==
            offline['process_metadata']['original_ud_cpu']['effective_affinity'] and
            replay['ud_process_metadata']['cpu']['effective_threads'] == offline['process_metadata']['original_ud_cpu']['effective_threads'],
            'prior replay CPU affinity/thread mismatch; cannot claim matched operation')
    projection_map(replay['projections'])
    require(all(p['wire_matches_local_cpu'] is True and p['observed_output_matches_assembly'] is True
                for p in replay['projections']), 'prior replay consistency failed')
    return dict(report_sha256=sha(directory/'r3a_parity_forensics.json'), directory=str(directory),
        native_reproducible=True, replay=replay, prior_selected_probe_count=len(report['native_a']['projections']),
        original_full_logit_bitwise_comparison='NOT_ESTIMABLE_PRIOR_REPLAY_DID_NOT_STORE_LOGIT_SHA256',
        limit='old strict metrics and QKV hashes cannot reconstruct its missing full-logit byte stream')


def compare_projections(local, replay):
    a, b = projection_map(local), projection_map(replay)
    rows = []
    # Use observed execution order, which OPT need not make q,k,v lexicographic.
    for p in local:
        key = f"{p['layer']}.{p['role']}";other = b[key]
        same_hidden = p['hidden']['sha256'] == other['hidden']['sha256']
        fields = dict(hidden=(p['hidden'], other['hidden']), base=(p['base'], other['base']),
            cpu_delta=(p['local_cpu_delta'], other['received_delta']),
            qkv=(p['observed_output'], other['observed_output']))
        matches = {name: left['sha256'] == right['sha256'] and left['shape'] == right['shape'] and
            left['dtype'] == right['dtype'] for name,(left,right) in fields.items()}
        first_field = next((f for f in ('hidden','base','cpu_delta','qkv') if not matches[f]), None)
        rows.append(dict(layer=p['layer'], role=p['role'], same_hidden=same_hidden, matches=matches,
            first_different_component=first_field,
            hashes={k:dict(local=v[0]['sha256'], replay=v[1]['sha256']) for k,v in fields.items()},
            numeric_magnitude='NOT_ESTIMABLE_HASH_ONLY' if first_field else 'ZERO_IDENTICAL_BYTES',
            matched_input_arithmetic_claim_allowed=same_hidden,
            interpretation='matched-input comparison' if same_hidden else 'downstream input divergence; no matched-input attribution'))
        if other.get('local_oracle_numeric_comparisons'):
            rows[-1]['measured_numeric_comparisons']=other['local_oracle_numeric_comparisons']
    require([f"{p['layer']}.{p['role']}" for p in local] == [f"{p['layer']}.{p['role']}" for p in replay],
            'observed projection role order differs')
    return dict(rows=rows, qkv_hashes_matched=sum(r['matches']['qkv'] for r in rows),
        all_components_match=all(all(r['matches'].values()) for r in rows),
        first_difference=next((r for r in rows if r['first_different_component']), None))


def assess(native_a, native_b, local, replay, expected, prior_replay=None):
    """A sufficiency gate, not a production correctness/speed or necessity claim."""
    for worker in (native_a, native_b, local, replay):
        require(worker['execution_status'] == 'COMPLETE' and worker['classification'] == CLASSIFICATION and
                worker['PRIMARY_PERFORMANCE_ELIGIBLE'] is False, 'invalid diagnostic worker')
        validate_identity(worker['comparison_identity'], expected)
        require(worker['process_metadata']['gpu']['uuid'] == expected['gpu_uuid'], 'observed GPU UUID mismatch')
    pids = [w['process_metadata']['pid'] for w in (native_a, native_b, local, replay)]
    require(len(set(pids)) == 4, 'independent GPU child PIDs required')
    if not (_controls_exact(native_a) and _controls_exact(native_b)):
        return dict(conclusion=control_failure(native_a) or control_failure(native_b), sufficiency_established=False,
            reason='native no-hook/passive/reconstruction controls are not all exact; stop causal attribution')
    for worker in (local, replay):
        descriptor_valid(worker['logits'])
        require(worker['logits']['shape'] == expected['reference_logits_shape'], 'original full-logit shape mismatch')
        require(worker['vs_original_logits']['thresholds'] == THRESHOLDS, 'strict gate provenance changed')
        require(worker['weights_verified_after'] is True and worker['gpu_lora_on_es'] is False,
                'bare-base original weights/separation not verified')
        require(worker['cpu_runtime'] == local['cpu_runtime'], 'current CPU BLAS/configuration/affinity mismatch')
        for p in projection_map(worker['projections']).values():
            require(p['observed_output_matches_assembly'] is True, 'ES/local module output differs from predicted assembly')
    require(all(w['process_metadata']['cpu_runtime'] == local['cpu_runtime'] for w in (native_a,native_b)),
            'native and intervention CPU runtime fingerprints differ')
    require(local['actual_network_socket_used'] is False and local['distinct_ud_process_used'] is False and
            local['cpu_delta_is_forward_source'] is True and replay['actual_network_socket_used'] is True and
            replay['distinct_ud_process_used'] is True and replay['ud_process_metadata']['user_device_cuda_used'] is False and
            replay['ud_process_metadata']['pid'] not in pids and
            replay['ud_process_metadata']['cpu_runtime'] == local['cpu_runtime'], 'intervention/UD placement mismatch')
    require(replay['wire_accounting']['hidden_frames'] == replay['wire_accounting']['delta_frames'] == 32 and
            [l['layer_id'] for l in replay['layers']] == list(range(32)), '32 physical request/reply exchanges not verified')
    contrast = compare_projections(local['projections'], replay['projections'])
    same_logits = local['logits'] == replay['logits']
    strict_fail = not local['vs_original_logits']['passed'] and not replay['vs_original_logits']['passed']
    assembly_ok = all(p['wire_matches_local_cpu'] for p in replay['projections'])
    prior = compare_projections(local['projections'], prior_replay['projections']) if prior_replay else None
    prior_reproduced = bool(prior and prior['all_components_match'] and
                           replay['vs_original_logits'] == prior_replay['vs_original_logits'])
    pilot_metrics_equal = all(replay['vs_original_logits'][k] == v for k,v in expected['pilot_parity'].items())
    pilot_payloads_equal = ([dict(layer=l['layer_id'],hidden=l['hidden_payload_sha256'],delta=l['delta_payload_sha256'])
                            for l in replay['layers']] == expected['pilot_layer_payload_hashes'])
    native_points = projection_map(native_a['same_hidden_projections'])
    bindings=[]
    for p in local['projections']:
        key=f"{p['layer']}.{p['role']}";n=native_points[key]
        bindings.append(dict(layer=p['layer'],role=p['role'],same_hidden=p['hidden']['sha256']==n['hidden']['sha256'],
            same_base=p['base']['sha256']==n['independent_gpu_base']['sha256'],
            same_cpu_delta=p['local_cpu_delta']['sha256']==n['cpu_delta']['sha256'],
            matches_predicted_cpu_assembly=p['observed_output']['sha256']==n['predicted_cpu_assembled']['sha256'],
            differs_from_native=p['observed_output']['sha256']!=n['native_output']['sha256'],
            same_native_hidden_component_metrics=n['comparisons'],fp16_output_flip_count=n['fp16_output_flip_count']))
    first_native=next((b for b in bindings if b['differs_from_native']),None)
    first_isolated=bool(first_native and all(first_native[k] for k in
        ('same_hidden','same_base','same_cpu_delta','matches_predicted_cpu_assembly')) and first_native['fp16_output_flip_count']>0)
    enough = (same_logits and contrast['all_components_match'] and strict_fail and assembly_ok and
              prior_reproduced and pilot_metrics_equal and pilot_payloads_equal and first_isolated)
    return dict(conclusion='CPU_DELTA_SUFFICIENT_FOR_MATCHED_REPLAY' if enough else 'UNRESOLVED',
        sufficiency_established=enough, full_logits_bitwise_matched=same_logits,
        canonical_serialization=SERIALIZATION, qkv_hashes_matched=contrast['qkv_hashes_matched'], expected_qkv_outputs=96,
        local_vs_replay=contrast, local_vs_prior_replay=prior,
        prior_replay_tensor_and_strict_metrics_reproduced=prior_reproduced,
        original_pilot_payload_hashes_reproduced=pilot_payloads_equal,original_pilot_strict_metrics_reproduced=pilot_metrics_equal,
        local_vs_native_bindings=bindings,first_local_vs_native_qkv_difference=first_native,
        first_cpu_numeric_effect_isolated=first_isolated,
        local_strict_parity_passed=local['vs_original_logits']['passed'], replay_strict_parity_passed=replay['vs_original_logits']['passed'],
        original_failed_trace_bitwise_status='NOT_ESTIMABLE_PRIOR_FULL_LOGIT_SHA_NOT_RECORDED',
        interpretation=('Under verified matched conditions, local CPU FP32 delta computation without TCP or an independent UD '
            'reproduced all 96 physical replay QKV outputs and full serialized logits exactly, while native/GPU reconstruction '
            'matched the original. CPU deltas are sufficient for this matched replay divergence; transport is not needed to reproduce it.'
            if enough else 'No sufficiency attribution. Inspect first tensor/input difference, observer/environment and strict controls.'),
        limits=['sufficiency, not necessity or a claim that every CPU/GPU difference flips FP16 output',
            'local CPU oracle is a diagnostic counterfactual, not the physical UD architecture',
            'a prior replay without a full-logit SHA cannot be proved bitwise identical by approximate metrics',
            'same endpoint/untrained adapter only; no production fix, correctness-qualified baseline or performance claim'])
