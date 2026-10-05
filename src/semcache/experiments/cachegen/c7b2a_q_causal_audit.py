"""Four-case cached TOTAL-Q causal audit; read-only C7-B2 storage/runtime reuse."""
import argparse
import copy
import csv
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from . import c7b2_q_quality as gate
from . import c7b2_runtime as runtime
from . import c7b_q_capture as b0
from . import c7b_q_profiles as qcodec
from .c6_quality import generation_metrics, logit_metrics

require = b0.require
MODES = ('KV_BASELINE', 'Q24', 'Q32', 'Q_ZERO_COUNTERFACTUAL')
SELECTION_RULE = 'first canonical frozen32 episode in stable selection order for each history_depth=1,2,3,4'
# Predeclared measurement tolerances, not compression-quality acceptance gates.
LOGIT_MAX_ABS_TOLERANCE = 1e-6
LOGIT_MEAN_ABS_TOLERANCE = 1e-7
FRESH_ATTENTION_ABS_TOLERANCE = 1e-6
SAFETY_COUNTERS = ('runtime_q_profile_fit_count', 'runtime_storage_kv_cdf_fit_count',
                   'transport_encode_calls', 'transport_decode_calls')


def select_cases(episodes):
    require(len(episodes) == 32 and len({e['episode_id'] for e in episodes}) == 32,
            'Canonical frozen32 required; no reselection')
    selected = [next((i for i, e in enumerate(episodes) if e['history_depth'] == d), None)
                for d in (1, 2, 3, 4)]
    require(None not in selected, 'Missing canonical history depth')
    return selected


def verify_b2(root, prepared, args):
    m = b0.read(root/'manifest.json')
    gate.b3.check_fields(m, dict(stage='C7-B2', status='COMPLETE', baseline_matches_c6b3_2=True,
        runtime_q_profile_fit_count=0, runtime_storage_kv_cdf_fit_count=0,
        transport_compression_enabled=False, executed_modes=list(runtime.MODES),
        model=b0.MODEL, model_revision=b0.REVISION, tokenizer_revision=b0.REVISION,
        prompt_version=b0.VERSION, adapter_hashes=b0.WEIGHTS, q_profile_hashes=gate.Q_SHAS,
        kv_profile_sha256=gate.PROFILE_SHA, frozen32_selection_sha256=gate.b3.SELECTION_SHA,
        adapter_freeze_decision_sha256=gate.FREEZE_SHA,
        c7b1_manifest_sha256=b0.sha(args.q_calibration_root/'manifest.json'),
        source_sha256=gate.SOURCE_SHA, semantic_sha256=gate.SEMANTIC_SHA,
        physical_safety_contract=gate.CONTRACT, cached_payload='TOTAL_QKV',
        q_profile_frozen=False, selected_q_candidate=None, training_performed=False), 'C7-B2 prerequisite')
    require(all(m.get('counters', {}).get(k) == 0 for k in SAFETY_COUNTERS), 'B2 fitting/transport evidence missing')
    require(all(m.get('input_hashes', {}).get(k) == v for k, v in prepared.input_hashes.items()),
            'B2 inputs differ from current canonical chain')
    b0.verify_files(m['input_hashes'])
    require('per_case.csv' in m.get('output_hashes', {}), 'B2 per-case evidence unbound')
    files = gate.bound_outputs(root, m)
    files[str((root/'manifest.json').resolve())] = b0.sha(root/'manifest.json')
    # Verify the SAME canonical continuation actually recorded by B2 for the four
    # selected cases. Reading all CSV rows does not execute all frozen32 cases.
    with (root/'per_case.csv').open(newline='') as stream:
        saved = list(csv.DictReader(stream))
    for index in select_cases(prepared.episodes):
        e = prepared.episodes[index]
        canonical = prepared.official[index]['generated_token_ids']
        for mode in runtime.MODES:
            matches = [c for c in saved if c['episode_id'] == e['episode_id'] and c['mode'] == mode]
            require(len(matches) == 1, 'Missing/duplicate selected B2 case')
            c = matches[0]
            for key, value in e.items():
                actual = json.loads(c[key]) if isinstance(value, (dict, list)) else c.get(key)
                expected = value if isinstance(value, (dict, list)) else ('' if value is None else str(value))
                require(actual == expected, 'B2 frozen episode changed: '+key)
            require(c['logical_event_hash'] == b0.digest(e) and
                    json.loads(c['teacher_forced_canonical_token_ids']) == canonical and
                    c['teacher_forced_canonical_sha256'] == b0.digest(canonical), 'B2 canonical continuation changed')
    return files


def fingerprint(tensor):
    """Bit fingerprint includes dtype/shape and contiguous CPU tensor bytes."""
    import torch
    t = tensor.detach().cpu().contiguous()
    require(torch.isfinite(t).all().item(), 'Nonfinite audit tensor')
    h = hashlib.sha256(str((str(t.dtype), tuple(t.shape))).encode())
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def difference(raw, other, *, fingerprints=False):
    require(raw.shape == other.shape and raw.numel() > 0, 'Mismatched/empty audit tensors')
    result = qcodec.metrics(raw, other)
    a, b = fingerprint(raw), fingerprint(other)
    result['exact_equal'] = a == b
    if fingerprints:
        result.update(raw_sha256=a, comparison_sha256=b)
    return result


@contextmanager
def capture_mixed(adapter, sequence_length):
    """Install AFTER mixed_projection_path, with no hooks on Q/K/V modules.

    Wrappers clone each actual returned projection before attention sees it.
    out_proj hooks capture actual attention output before decoder residual/dropout;
    decoder hooks capture post-layer hidden state (not mislabeled post-attention).
    No extra projection/attention computation and no tensor mutation occurs.
    """
    captured, originals, handles = {}, [], []

    def save(layer, role, output):
        value = output[0] if isinstance(output, tuple) else output
        require(value.ndim == 3 and value.shape[:2] == (1, sequence_length), 'Unexpected OPT trace layout')
        require(role not in captured.setdefault(layer, {}), 'Repeated traced layer execution')
        captured[layer][role] = value.detach().cpu().clone()

    def wrap(layer, role, native):
        def forward(*args, **kwargs):
            value = native(*args, **kwargs)
            save(layer, role, value)
            return value
        return forward

    def hook(layer, role):
        return lambda module, args, output: save(layer, role, output)

    try:
        for layer, block in enumerate(adapter.layers):
            for role, module in adapter.projection_modules(layer).items():
                require('forward' in module.__dict__, 'Capture must follow real mixed-path installation')
                originals.append((module, module.forward))
                module.forward = wrap(layer, role, module.forward)
            handles.append(block.self_attn.out_proj.register_forward_hook(hook(layer, 'attention_output')))
            handles.append(block.register_forward_hook(hook(layer, 'post_layer_hidden')))
        yield captured
        require(len(captured) == len(adapter.layers) and all(set(r) ==
            {'q', 'k', 'v', 'attention_output', 'post_layer_hidden'} for r in captured.values()), 'Incomplete internal trace')
    finally:
        for handle in handles:
            handle.remove()
        for module, original in originals:
            module.forward = original


class AuditBackend(runtime.Backend):
    """Only audit forwards add passive captures; source/greedy use B2 unchanged."""
    tracing = False
    source_start = None

    def prepare(self, episode, mode):
        self.source_start = episode['source_start']
        try:
            return super().prepare(episode, mode)
        finally:
            self.source_start = None

    def forward(self, ids, user, hits, transport, *, use_cache=False, past=None, capture=False):
        if not self.tracing:
            output, projections = super().forward(ids, user, hits, transport,
                use_cache=use_cache, past=past, capture=capture)
            if capture and self.source_start is not None:
                start = self.source_start
                self.raw_source_q = {l: r['q'][:, start:start+3].detach().cpu().clone()
                                     for l, r in projections.items()}
            return output, projections
        import torch
        from semcache.models.task_adapters import activate_task_user
        from semcache.edgelora.mixed_projection import mixed_projection_path
        require(transport is False and not use_cache and past is None and not capture,
                'Trace only the established canonical teacher-forced forward')
        activate_task_user(self.model, user)
        require(not any(p.requires_grad for p in self.model.parameters()), 'Frozen model required')
        with torch.inference_mode(), mixed_projection_path(self.adapter, user, hits, len(ids)) as audit:
            with capture_mixed(self.adapter, len(ids)) as trace:
                output = self.model(input_ids=torch.tensor([ids], device=self.args.device), use_cache=False)
        self.last_trace = trace
        self.last_reuse = runtime.projection_audit(audit, len(self.adapter.layers), len(ids), hits)
        return output, None

    def traced_teacher(self, context, episode, canonical):
        self.tracing = True
        try:
            logits = self.teacher_logits(context, episode, canonical)
        finally:
            self.tracing = False
        trace = self.last_trace
        del self.last_trace
        return logits, trace, self.last_reuse


class _ZeroQView:
    def __init__(self, original):
        import torch
        self.original = original
        self.tensors = {l: (torch.zeros_like(q), k, v) for l, (q, k, v) in original.tensors.items()}

    def __getattr__(self, name):
        return getattr(self.original, name)


def zero_context(context):
    """Diagnostic temporary lookup view only; resident entry and K/V untouched."""
    from semcache.semantic.hit_selection import CacheHit
    require(len(context.hits) == 1, 'Exactly one selected w=3 HIT required')
    hit = context.hits[0]
    view = _ZeroQView(hit.entry)
    changed = copy.copy(context)
    changed.hits = [CacheHit(hit.window, view, hit.utility)]
    return changed


def injection_check(trace, baseline, hits, reuse, baseline_reuse):
    import torch
    require(reuse == baseline_reuse, 'INVALID: HIT mask/native skipping differs')
    positions = reuse['hit_positions']
    require(len(positions) == 3 and len(hits) == 1, 'INVALID: single w=3 HIT required')
    window = hits[0].window
    fresh = [i for i in range(trace[0]['q'].shape[1]) if i not in positions]
    layers = []
    for l, values in trace.items():
        record = dict(layer=l)
        for i, role in enumerate('qkv'):
            actual = values[role][:, window.start:window.end]
            expected = hits[0].entry.tensors[l][i].to(actual)
            record[role+'_injection_verified'] = fingerprint(actual) == fingerprint(expected)
            if role != 'q':
                record[role+'_rows_equal'] = fingerprint(actual) == fingerprint(baseline[l][role][:, window.start:window.end])
        record['fresh_q_rows_equal'] = fingerprint(values['q'][:, fresh]) == fingerprint(baseline[l]['q'][:, fresh])
        require(all(v for k, v in record.items() if k != 'layer'),
                'INVALID: mixed projection injection/fresh Q/KV mismatch: '+str(record))
        # Record fingerprints of the actual tensor returned to OPT attention.
        record['q_hit_sha256'] = fingerprint(values['q'][:, window.start:window.end])
        record['q_fresh_sha256'] = fingerprint(values['q'][:, fresh])
        require(torch.isfinite(values['attention_output']).all().item(), 'Nonfinite attention')
        layers.append(record)
    return dict(hit_mask_equal=True, hit_positions=positions, fresh_q_rows_equal=True,
        k_rows_equal=True, v_rows_equal=True, q_injection_verified=True, per_layer=layers,
        reuse_audit=reuse)


def internal_effect(baseline, zero, positions):
    layers = []
    fresh = [i for i in range(baseline[0]['q'].shape[1]) if i not in positions]
    for l, reference in baseline.items():
        attention = difference(reference['attention_output'][:, positions], zero[l]['attention_output'][:, positions])
        hidden = difference(reference['post_layer_hidden'][:, positions], zero[l]['post_layer_hidden'][:, positions])
        untouched = difference(reference['attention_output'][:, fresh], zero[l]['attention_output'][:, fresh])
        layers.append(dict(layer=l, hit_attention=attention, hit_post_layer_hidden=hidden,
            fresh_attention_max_abs_diff=untouched['max_absolute_error'], fresh_attention_exact_equal=untouched['exact_equal'],
            fresh_attention_numerically_negligible=untouched['max_absolute_error'] <= FRESH_ATTENTION_ABS_TOLERANCE,
            fresh_post_layer_hidden=difference(reference['post_layer_hidden'][:, fresh], zero[l]['post_layer_hidden'][:, fresh])))
    # No failure for small fresh-row floating-point differences. The required
    # positive causal evidence is a changed hit attention output or hidden state.
    changed = any(r['hit_attention']['max_absolute_error'] > 0 or
                  r['hit_post_layer_hidden']['max_absolute_error'] > 0 for r in layers)
    require(changed, 'INVALID: Q_ZERO has no hit-row internal causal effect')
    return dict(hit_row_causal_effect_verified=changed, per_layer=layers,
        fresh_attention_unchanged=all(r['fresh_attention_exact_equal'] for r in layers),
        fresh_attention_numerically_negligible=all(r['fresh_attention_numerically_negligible'] for r in layers))


def continuation_effect(reference, candidate):
    import math
    result = logit_metrics(reference, candidate)
    result['top5_agreement'] = result.pop('top5_overlap')
    require(all(math.isfinite(v) for v in result.values()), 'Nonfinite continuation metrics')
    result['exact_equal'] = fingerprint(reference) == fingerprint(candidate)
    result['numerically_negligible'] = (result['max_abs_logit_difference'] <= LOGIT_MAX_ABS_TOLERANCE and
        result['mean_abs_logit_difference'] <= LOGIT_MEAN_ABS_TOLERANCE)
    return result


def validate_distortion(records):
    for candidate in ('Q24', 'Q32'):
        require(any(not r['exact_equal'] for r in records[candidate]),
                'INVALID: '+candidate+' decoded Q identical to raw Q in every layer')


def safety(counts):
    require(all(counts[k] == 0 for k in SAFETY_COUNTERS), 'INVALID: runtime fitting/transport forbidden')


def run(args, prepared, selected, reports, manifest):
    require(selected == select_cases(prepared.episodes), 'Exactly four canonical depth-stratified cases required')
    backend = AuditBackend(args, prepared.profiles, prepared.q_backend)
    backend.rows = prepared.rows
    require(len(backend.adapter.layers) == 32, '32 OPT projection layers required')
    manifest.update(model_inference_performed=True, counters=backend.counts,
        model_tokenizer_provenance=backend.metadata, storage_source=backend.storage_source,
        capture_points=dict(q='mixed projection return before OPT query scaling/head split',
            attention='OPT self_attn.out_proj return before decoder residual/dropout', hidden='OPT decoder layer return'))
    for index in selected:
        e = prepared.episodes[index]
        canonical = list(prepared.official[index]['generated_token_ids'])
        require(0 < len(canonical) <= 160, 'Invalid canonical continuation')
        baseline = backend.prepare(e, runtime.MODES[0])
        raw = backend.raw_source_q
        require(set(raw) == set(range(32)) and all(t.shape == (1, 3, 2560) for t in raw.values()), 'Source TOTAL Q shape changed')
        import torch
        require(all(t.dtype == torch.float16 and t.device.type == 'cpu' for t in raw.values()), 'CPU FP16 source TOTAL Q required')
        require(all(fingerprint(raw[l]) == fingerprint(baseline.entry.q_tensors[l]) for l in raw),
                'INVALID: raw resident Q differs from captured source TOTAL Q')
        identity = backend.event_identity(baseline, e)
        require(identity == b0.digest(e), 'INVALID: logical HIT changed')
        base_logits, base_trace, base_reuse = backend.traced_teacher(baseline, e, canonical)
        case_injection = dict(episode_id=e['episode_id'], modes={})
        case_injection['modes']['KV_BASELINE'] = injection_check(base_trace, base_trace, baseline.hits, base_reuse, base_reuse)
        case_injection['modes']['KV_BASELINE']['raw_q_resident_after_insert'] = True
        reports['injection_audit']['cases'].append(case_injection)
        distortion = dict(episode_id=e['episode_id'], candidates={})
        reports['q_distortion']['cases'].append(distortion)
        continuation = dict(episode_id=e['episode_id'], canonical_token_ids=canonical,
            canonical_sha256=b0.digest(canonical), comparisons={})
        reports['continuation_causal_effect']['cases'].append(continuation)
        base_greedy = backend.greedy(baseline, e) if args.greedy else None
        if base_greedy:
            gate.baseline_case_check(dict(generated_token_ids=base_greedy[0], generated_text=base_greedy[1]), prepared.baseline[index])
        for label, mode in zip(MODES[1:3], runtime.MODES[1:]):
            context = backend.prepare(e, mode)
            require(backend.event_identity(context, e) == identity and context.entry.q_tensors is None and
                    context.accounting['raw_q_resident_after_insert'] is False, 'INVALID: compressed Q retains raw Q/logical HIT changed')
            require(context.entry.compressed_kv.bitstream == baseline.entry.compressed_kv.bitstream,
                    'INVALID: K/V frames changed')
            require(all(fingerprint(backend.raw_source_q[l]) == fingerprint(raw[l]) for l in raw), 'INVALID: source Q changed between modes')
            decoded = context.hits[0].entry.tensors
            distortion['candidates'][label] = [dict(layer=l, **difference(raw[l], decoded[l][0], fingerprints=True)) for l in raw]
            logits, trace, reuse = backend.traced_teacher(context, e, canonical)
            evidence = injection_check(trace, base_trace, context.hits, reuse, base_reuse)
            evidence.update(raw_q_resident_after_insert=False)
            case_injection['modes'][label] = evidence
            continuation['comparisons'][label] = continuation_effect(base_logits, logits)
            if args.greedy:
                tokens, text = backend.greedy(context, e)
                continuation['comparisons'][label]['greedy'] = dict(generated_token_ids=tokens, generated_text=text,
                    **generation_metrics(base_greedy[0], tokens))
            del trace, logits, context
        validate_distortion(distortion['candidates'])
        distortion['nonidentical_layer_counts'] = {label: sum(not r['exact_equal'] for r in records)
            for label, records in distortion['candidates'].items()}
        zero = zero_context(baseline)
        require(backend.event_identity(zero, e) == identity, 'INVALID: counterfactual HIT changed')
        zero_logits, zero_trace, reuse = backend.traced_teacher(zero, e, canonical)
        case_injection['modes'][MODES[3]] = injection_check(zero_trace, base_trace, zero.hits, reuse, base_reuse)
        case_injection['q24_injection_verified'] = case_injection['q32_injection_verified'] = case_injection['qzero_injection_verified'] = True
        case_injection.update(hit_mask_equal=True, same_reuse_mask_qkv=True, fresh_q_rows_equal=True,
            k_rows_equal=True, v_rows_equal=True)
        internal = dict(episode_id=e['episode_id'], **internal_effect(base_trace, zero_trace, reuse['hit_positions']))
        reports['internal_causal_effect']['cases'].append(internal)
        effect = continuation_effect(base_logits, zero_logits)
        if args.greedy:
            tokens, text = backend.greedy(zero, e)
            effect['greedy'] = dict(generated_token_ids=tokens, generated_text=text, **generation_metrics(base_greedy[0], tokens))
            continuation['baseline_greedy'] = dict(generated_token_ids=base_greedy[0], generated_text=base_greedy[1])
        continuation['comparisons'][MODES[3]] = effect
        continuation['causal_isolation_supported'] = (internal['hit_row_causal_effect_verified'] and
            effect['numerically_negligible'] and effect['top1_agreement'] == effect['top5_agreement'] == 1 and
            (not args.greedy or effect['greedy']['exact_generation']))
        safety(backend.counts)
        require(backend.event_identity(baseline, e) == identity and baseline.entry.q_tensors is not None,
                'INVALID: counterfactual mutated resident baseline')
        del base_trace, zero_trace, base_logits, zero_logits, baseline, zero, raw
        print('Completed C7-B2A '+e['episode_id'], flush=True)
    for label in ('Q24', 'Q32'):
        records = [r for c in reports['q_distortion']['cases'] for r in c['candidates'][label]]
        reports['q_distortion'].setdefault('aggregate', {})[label] = qcodec.aggregate_metrics(records)
    reports['q_distortion']['layer_blocks_per_candidate'] = 128
    require(backend.counts['source_forward_count'] == backend.counts['storage_decode_count'] == 12 and
        backend.counts['teacher_forced_forward_count'] == 16 and
        backend.counts['q_encode_count'] == backend.counts['q_decode_count'] == 8, 'INVALID: bounded audit execution count changed')
    safety(backend.counts)
    b0.verify_files(prepared.input_hashes)
    manifest.update(status='COMPLETE', causal_isolation_supported=all(c['causal_isolation_supported'] for c in
        reports['continuation_causal_effect']['cases']))


def write_reports(root, reports, manifest):
    for name, value in reports.items():
        b0.write(root/(name+'.json'), dict(status=manifest['status'], **value))
    valid = manifest['status'] == 'COMPLETE'
    answer = lambda value: str(value) if valid else 'Not established; audit INVALID. See manifest error/partial evidence.'
    internal = reports['internal_causal_effect']['cases']
    continuation = reports['continuation_causal_effect']['cases']
    lines = ['# C7-B2A causal sanity audit', '',
        'Exactly four cases: '+SELECTION_RULE+'.',
        'Episodes: '+', '.join(c['episode_id'] for c in reports['selected_cases']['cases'])+'.', '',
        '1. Decoded Q24/Q32 different from raw TOTAL Q in every case? '+answer(True),
        '2. Decoded tensors actually injected into the selected HIT rows? '+answer(True),
        '3. Zeroing cached Q changed hit-row attention/internal states? '+answer(True),
        '4. Fresh-row attention remained numerically unchanged? '+answer(all(c['fresh_attention_numerically_negligible'] for c in internal)),
        '5. Canonical teacher-forced continuation remained numerically unchanged? '+answer(all(c['comparisons'][MODES[3]]['numerically_negligible'] for c in continuation)),
        '   Greedy generation: '+('measured in continuation_causal_effect.json' if manifest['greedy_executed'] else 'NOT_RUN (optional --greedy; avoids extra autoregressive forwards).'),
        '6. Compressed Q is genuinely consumed but causally isolated here? '+answer(manifest.get('causal_isolation_supported')), '',
        'If isolation is false after a valid audit, propagation was measured; inspect the per-layer and continuation evidence. If INVALID, the zero downstream difference needs further debugging.',
        'Interpretation is limited to this tested SemCache causal-generation path: perturbing reused cached Q altered hit-row computation and did/did not propagate to fresh continuation under fixed cached K/V.',
        'The exact w=3 HIT and TOTAL Q/K/V payload remain unchanged. This does not justify removing Q. No Q profile is automatically frozen.',
        'Numeric tolerances are declared in manifest.json before execution; fresh-attention rounding noise alone does not invalidate the audit.', '']
    (root/'summary.md').write_text('\n'.join(lines))
    manifest['output_hashes'] = {p.name: b0.sha(p) for p in root.iterdir() if p.is_file() and p.name != 'manifest.json'}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    defaults = dict(adapter_root='results/cachegen/c6b3/b1_full',
        freeze_decision='results/cachegen/c6b3/b1_full_freeze_decision.json',
        plan_dir='results/cachegen/c6b3/multiwoz_plan_v2', source='results/workloads/multiwoz.jsonl',
        semantic='results/workloads/c6b3_multiwoz_history_semantic.jsonl',
        q_calibration_root='results/cachegen/c7/b_q_calibration_retry2', kv_profile=gate.PROFILE,
        storage_src='/data/khuss/repos/GP_semcache/.c6_storage_src/src',
        c6b3_baseline_root='results/cachegen/c6b3/b2_frozen32',
        c7b2_root='results/cachegen/c7/b2_q_quality_gate_frozen32', output_root='results/cachegen/c7/b2a_q_causal_audit')
    for name, default in defaults.items():
        p.add_argument('--'+name.replace('_', '-'), type=Path, default=Path(default))
    p.add_argument('--device', default='cuda:0'); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--greedy', action='store_true', help='Also execute existing bounded 160-token greedy helper for four cases/four modes')
    args = p.parse_args(argv)
    require(args.seed == 42 and args.device.startswith('cuda:'), 'Frozen seed42 and one CUDA device required')
    require(Path('/data/khuss').is_dir() and args.output_root.resolve().is_relative_to(Path('/data/khuss')),
            'GPU audit execution requires SERAPH with output under /data/khuss')
    require(not args.output_root.exists(), 'New C7-B2A output root required; no overwrite')
    b0.seraph_paths(args.output_root)
    args.profile_path = args.kv_profile; args.max_sequence_length = 384; args.max_new_tokens = 160
    prepared = gate.prepare(args)  # READ frozen32/provenance; never rerun/reselect 32 cases.
    selected = select_cases(prepared.episodes)
    prepared.input_hashes.update(verify_b2(args.c7b2_root, prepared, args))
    reports = {name: dict(cases=[]) for name in ('q_distortion', 'injection_audit', 'internal_causal_effect', 'continuation_causal_effect')}
    reports['selected_cases'] = dict(selection_rule=SELECTION_RULE, frozen32_selection_sha256=gate.b3.SELECTION_SHA,
        cases=[dict(prepared.episodes[i], canonical_selection_index=i) for i in selected])
    reports['continuation_causal_effect'].update(canonical_provenance=prepared.provenance['teacher_forced_canonical_source'],
        policy='same C7-B2 imported FULL_RECOMPUTE canonical continuation; prompt + canonical[:-1]',
        greedy_status='REQUESTED' if args.greedy else 'NOT_RUN')
    manifest = dict(stage='C7-B2A', status='STARTING', research_scope='four-case cached TOTAL-Q causal sanity audit',
        audit_cases=4, selection_rule=SELECTION_RULE, episode_ids=[prepared.episodes[i]['episode_id'] for i in selected],
        executed_modes=list(MODES), physical_safety_contract=gate.CONTRACT, cached_payload='TOTAL_QKV',
        model=b0.MODEL, model_revision=b0.REVISION, tokenizer_revision=b0.REVISION, prompt_version=b0.VERSION,
        adapter_hashes=b0.WEIGHTS, q_profile_hashes=gate.Q_SHAS, kv_profile_sha256=gate.PROFILE_SHA,
        c7b2_manifest_sha256=b0.sha(args.c7b2_root/'manifest.json'), frozen32_selection_sha256=gate.b3.SELECTION_SHA,
        input_hashes=prepared.input_hashes, runtime_q_profile_fit_count=0, runtime_storage_kv_cdf_fit_count=0,
        transport_encode_calls=0, transport_decode_calls=0, transport_compression_enabled=False,
        q_profile_frozen=False, selected_q_candidate=None, training_performed=False, model_inference_performed=False,
        greedy_executed=args.greedy, tolerances=dict(logit_max_abs=LOGIT_MAX_ABS_TOLERANCE,
            logit_mean_abs=LOGIT_MEAN_ABS_TOLERANCE, fresh_attention_max_abs=FRESH_ATTENTION_ABS_TOLERANCE),
        isolation_rule='hit-row internal change AND negligible continuation logits AND unchanged top1/top5; greedy exact if requested',
        counterfactual_policy='zero only cached Q in temporary HIT view immediately before mixed reuse; resident and K/V unchanged',
        source_q_copies='audit CPU working copies only; not resident cache representation', git=b0.git())
    b0.write(args.output_root/'manifest.json', manifest)
    try:
        run(args, prepared, selected, reports, manifest)
    except Exception as exc:
        manifest.update(status='INVALID', error=str(exc), causal_isolation_supported=None)
        raise
    finally:
        for key in SAFETY_COUNTERS:
            manifest[key] = manifest.get('counters', {}).get(key, 0)
        write_reports(args.output_root, reports, manifest)
