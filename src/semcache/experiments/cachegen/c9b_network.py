"""CPU-only payload-bandwidth / frozen storage-codec break-even analysis."""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean

from . import c7b3_q_freeze as provenance
from . import c8b_quality as capacity
from . import c9a_latency as local
from . import c9b_transport as transport

require = provenance.require
BANDWIDTHS = (10, 50, 100, 500, 1000)
SPEEDUPS = (1, 10, 50, 100, 500, 1000)
REUSE_COUNTS = (1, 8, 32)
REQUIRED_INPUTS = ('target_latency_raw.csv', 'target_latency_per_case.csv', 'target_latency_summary.json',
    'hit_miss_latency.json', 'paired_latency.json', 'source_build_summary.json')
CLASSIFICATION_RULE = ('Per compressed policy/budget: SYSTEM_COMPETITIVE only if its mean current-codec '
    'payload-only estimate matches/beats every relevant comparator at all five tested bandwidths; '
    'NETWORK_CONDITIONAL if it does so at some tested bandwidths; NOT_LATENCY_COMPETITIVE otherwise. '
    'Stage SYSTEM requires all four contexts to be SYSTEM; stage CONDITIONAL requires at least one '
    'context competitive at some tested point. No claim outside 10–1000 Mbps; exact roots separately reported.')
NATIVE_RULE = ('native_backend_required means at least one tested compressed context requires a finite '
    'factor greater than 1 to break even. It is an optimization requirement flag, not proof that only '
    'a native language can provide it. Infeasible non-decode floors are reported separately.')


def read_measurements(path):
    rows = provenance.read_csv(path)
    for row in rows:
        for field in local.COMPONENTS:
            try: row[field] = float(row[field])
            except (KeyError, ValueError, TypeError) as exc: raise ValueError('Malformed C9-A timing field: '+field) from exc
        require(row.get('hit') in ('True', 'False'), 'Malformed measured HIT flag')
        row['hit'] = row['hit'] == 'True'
        row['retained_source_episode_id'] = row.get('retained_source_episode_id') or None
        for field in ('episode_index', 'repeat', 'prompt_tokens', 'native_projection_rows_skipped_per_role_per_layer'):
            try: row[field] = int(row[field])
            except (KeyError, ValueError, TypeError) as exc: raise ValueError('Malformed C9-A timing field: '+field) from exc
    return rows


def prepare(args):
    prepared = capacity.verify_capacity(args)
    local.verify_c8b(args, prepared)  # Provenance/reuse evidence only; no quality/model calls.
    expected = dict(stage='C9-A', status='COMPLETE', recommendation='C9_A_READY_FOR_NETWORK_ACCOUNTING',
        c8_hit_vectors_reproduced=True, latency_decomposition_consistent=True,
        conditions=list(local.CONDITIONS), budgets_raw_entry_equivalent=list(local.BUDGETS), policies=list(local.POLICIES),
        reference_mode=local.REFERENCE, repeats_per_target=3, warmup_forwards=5, dtype='float16',
        physical_safety_contract=provenance.CONTRACT, cached_payload='TOTAL_QKV',
        model_inference_performed=True, training_performed=False, quality_evaluated_in_c9=False,
        network_latency_evaluated=False, transport_compression_enabled=False,
        source_build_cost_in_primary_target_latency=False, timing_method=local.TIMING_METHOD,
        c7b3_freeze_sha256=provenance.sha(args.freeze_decision),
        c8b_manifest_sha256=provenance.sha(args.c8b_root/'manifest.json'),
        q24_profile_sha256=provenance.Q_SHA, kv_profile_sha256=provenance.KV_SHA,
        frozen32_selection_sha256=provenance.SELECTION_SHA, **prepared.frozen['model_namespace'],
        **{k: 0 for k in capacity.SAFETY})
    manifest, hashes = provenance.load_manifest(args.c9a_root, expected, required=REQUIRED_INPUTS)
    inputs = manifest.get('input_hashes')
    require(isinstance(inputs, dict) and all(inputs.get(p) == h for p, h in prepared.input_hashes.items()), 'C9-A upstream input chain differs')
    provenance.verify_files(inputs)
    require(manifest.get('c8a_artifact_hashes') == {name: provenance.sha(args.c8a_root/name)
        for name in ('manifest.json', 'summary.json', 'residency_trace.json', 'per_event.csv')}, 'C9-A C8-A bindings differ')
    require(isinstance(manifest.get('cuda_synchronization_policy'), str) and manifest['cuda_synchronization_policy'], 'Missing synchronized timing evidence')
    counters = manifest.get('counters', {})
    provenance.fields(counters, local.expected_execution_counts(prepared.expected), 'C9-A execution counts')
    tolerance = manifest.get('tie_tolerance_ms')
    require(type(tolerance) in (float, int) and math.isfinite(tolerance) and tolerance > 0, 'Invalid measured timer tolerance')
    raw = read_measurements(args.c9a_root/'target_latency_raw.csv')
    cases = local.per_case(raw, prepared.episodes, prepared.expected, tolerance)
    # Rebuild medians and summary diagnostics from the hash-verified raw repeats.
    saved_cases = provenance.read_csv(args.c9a_root/'target_latency_per_case.csv')
    require(saved_cases == [capacity.csv_form(row) for row in cases], 'C9-A saved per-case medians differ from raw measurements')
    summaries = local.summaries(cases)
    provenance.fields(provenance.read(args.c9a_root/'target_latency_summary.json'),
        dict(status='COMPLETE', primary_metric='target_total_ms', conditions=summaries), 'C9-A timing summary')
    require(provenance.read(args.c9a_root/'hit_miss_latency.json') == summaries, 'C9-A HIT/MISS timing summary differs')
    require(provenance.read(args.c9a_root/'paired_latency.json') == local.paired(cases, tolerance), 'C9-A paired timing differs')
    prepared.input_hashes.update(inputs); prepared.input_hashes.update(hashes)
    prepared.c9_manifest, prepared.args, prepared.tie_tolerance_ms = manifest, args, tolerance
    prepared.local_cases, prepared.raw = cases, raw
    prepared.source = provenance.read(args.c9a_root/'source_build_summary.json')
    require(prepared.source.get('source_build_cost_in_primary_target_latency') is False, 'Source build must remain secondary')
    prepared.transport, prepared.transport_summary = transport.audit_transport(prepared, raw, cases)
    provenance.verify_files(prepared.input_hashes)
    return prepared


def finite(*values):
    require(all(type(v) in (float, int) and math.isfinite(v) for v in values), 'Nonfinite/nonnumeric analytical input')


def transfer_ms(payload_bytes, bandwidth_mbps):
    finite(payload_bytes, bandwidth_mbps)
    require(payload_bytes >= 0 and bandwidth_mbps > 0, 'Nonnegative bytes / positive bandwidth required')
    return payload_bytes*.008/bandwidth_mbps


def accelerated_local(total_ms, decode_ms, speedup):
    finite(total_ms, decode_ms, speedup)
    require(0 <= decode_ms <= total_ms and speedup >= 1, 'Invalid decode floor / acceleration factor')
    return total_ms-decode_ms+decode_ms/speedup


def network_root(delta_local_ms, delta_bytes):
    """Exact mean/per-target equation, including inverse-bandwidth tradeoffs."""
    finite(delta_local_ms, delta_bytes)
    result = dict(delta_local_ms=delta_local_ms, delta_bytes=delta_bytes,
        break_even_bandwidth_mbps=None, winning_bandwidth_region=None)
    if (delta_local_ms > 0 and delta_bytes < 0) or (delta_local_ms < 0 and delta_bytes > 0):
        value = -.008*delta_bytes/delta_local_ms
        if math.isfinite(value) and value > 0:
            return dict(result, break_even_bandwidth_mbps=value, reason='POSITIVE_FINITE_BREAK_EVEN',
                winning_bandwidth_region='BELOW_BREAK_EVEN' if delta_local_ms > 0 else 'ABOVE_BREAK_EVEN')
        return dict(result, reason='NO_POSITIVE_FINITE_ROOT')
    if delta_local_ms == delta_bytes == 0: return dict(result, reason='TIED_AT_ALL_BANDWIDTHS')
    if delta_local_ms <= 0 and delta_bytes <= 0:
        return dict(result, reason='A_FASTER_AT_ALL_FINITE_BANDWIDTHS', winning_bandwidth_region='ALL_POSITIVE_BANDWIDTHS')
    return dict(result, reason='A_SLOWER_AT_ALL_FINITE_BANDWIDTHS', winning_bandwidth_region='NONE')


def codec_root(offset_ms, decode_coefficient_ms):
    """Solve offset + coefficient/S <=0, S>=1; supports negative fair deltas."""
    finite(offset_ms, decode_coefficient_ms)
    base = dict(offset_ms=offset_ms, decode_coefficient_ms=decode_coefficient_ms,
        required_speedup=None, maximum_winning_speedup=None,
        infinite_speedup_limit_delta_ms=offset_ms)
    if offset_ms+decode_coefficient_ms <= 0:
        maximum = -decode_coefficient_ms/offset_ms if offset_ms > 0 else None
        return dict(base, status='ALREADY_BREAKS_EVEN', required_speedup=1.,
            maximum_winning_speedup=maximum, acceleration_can_erode_advantage=maximum is not None)
    if offset_ms < 0 and decode_coefficient_ms > 0:
        factor = decode_coefficient_ms/(-offset_ms)
        if math.isfinite(factor): return dict(base, status='FINITE_CODEC_SPEEDUP_REQUIRED', required_speedup=factor)
    return dict(base, status='NO_FINITE_CODEC_SPEEDUP_CAN_BREAK_EVEN',
        reason='non-decode/network floor cannot be beaten by shrinking positive decode work' if decode_coefficient_ms >= 0
        else 'uniform acceleration removes a decode-time advantage; no S>=1 can win')


def root_distribution(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return dict(valid_target_count=len(values), undefined_target_count=len(rows)-len(values),
        **local.stats(values), min=min(values) if values else None, max=max(values) if values else None)


def comparisons():
    return [(budget, before, after) for budget in local.BUDGETS for before, after in
        [(local.REFERENCE, capacity.condition(budget, 'RAW_QKV')),
         (local.REFERENCE, capacity.condition(budget, 'KV_COMP')),
         (local.REFERENCE, capacity.condition(budget, 'Q24_KV_COMP')),
         (capacity.condition(budget, 'RAW_QKV'), capacity.condition(budget, 'KV_COMP')),
         (capacity.condition(budget, 'KV_COMP'), capacity.condition(budget, 'Q24_KV_COMP')),
         (capacity.condition(budget, 'RAW_QKV'), capacity.condition(budget, 'Q24_KV_COMP'))]]


def analyzed_cases(prepared):
    indexed = {(r['condition'], r['episode_id']): r for r in prepared.transport}
    rows = []
    for row in prepared.local_cases:
        payload = indexed[row['condition'], row['episode_id']]
        require(row['hit'] == payload['hit'] and row['retained_source_episode_id'] == payload['resident_source_episode_id'], 'Transport/measured HIT mismatch')
        accelerated_local(row['target_total_ms'], row['storage_decode_ms'], 1)
        rows.append(dict(row, transport_bytes=payload['total_transport_bytes']))
    return rows


def bandwidth_analysis(cases, tolerance):
    sweep, summary, pairs, roots = [], [], [], []
    by_key = {(r['condition'], r['episode_id']): r for r in cases}
    ids = [r['episode_id'] for r in cases if r['condition'] == local.REFERENCE]
    for bandwidth in BANDWIDTHS:
        for row in cases:
            network = transfer_ms(row['transport_bytes'], bandwidth)
            sweep.append(dict(condition=row['condition'], episode_id=row['episode_id'], hit=row['hit'],
                bandwidth_mbps=bandwidth, transport_bytes=row['transport_bytes'],
                measured_local_ms=row['target_total_ms'], analytical_transport_ms=network,
                analytical_e2e_ms=row['target_total_ms']+network, estimate='ANALYTICAL_END_TO_END_ESTIMATE'))
        for name in local.CONDITIONS:
            group = [r for r in sweep if r['bandwidth_mbps'] == bandwidth and r['condition'] == name]
            summary.append(dict(condition=name, bandwidth_mbps=bandwidth, cases=len(group),
                analytical_e2e_ms=local.stats([r['analytical_e2e_ms'] for r in group]),
                mean_transport_ms=mean(r['analytical_transport_ms'] for r in group),
                mean_local_ms=mean(r['measured_local_ms'] for r in group), estimate='ANALYTICAL_END_TO_END_ESTIMATE'))
        for budget, before, after in comparisons():
            differences = []
            for episode_id in ids:
                a, b = by_key[after, episode_id], by_key[before, episode_id]
                dl, db = a['target_total_ms']-b['target_total_ms'], a['transport_bytes']-b['transport_bytes']
                dn = .008*db/bandwidth
                differences.append(dict(episode_id=episode_id, delta_local_ms=dl, delta_transport_bytes=db,
                    delta_transport_ms=dn, delta_e2e_ms=dl+dn))
            pairs.append(dict(budget='B'+str(budget), before=before, after=after, bandwidth_mbps=bandwidth,
                **{'mean_'+k: mean(r[k] for r in differences) for k in
                    ('delta_local_ms', 'delta_transport_bytes', 'delta_transport_ms', 'delta_e2e_ms')},
                targets_faster=sum(r['delta_e2e_ms'] < -tolerance for r in differences),
                targets_slower=sum(r['delta_e2e_ms'] > tolerance for r in differences),
                targets_tied=sum(abs(r['delta_e2e_ms']) <= tolerance for r in differences), per_target=differences))
    for budget, before, after in comparisons():
        per_target = [dict(episode_id=i, **network_root(by_key[after, i]['target_total_ms']-by_key[before, i]['target_total_ms'],
            by_key[after, i]['transport_bytes']-by_key[before, i]['transport_bytes'])) for i in ids]
        aggregate = network_root(mean(r['delta_local_ms'] for r in per_target), mean(r['delta_bytes'] for r in per_target))
        roots.append(dict(budget='B'+str(budget), before=before, after=after, aggregate_mean=aggregate,
            per_target_distribution=root_distribution(per_target, 'break_even_bandwidth_mbps'),
            per_target=per_target))
    return sweep, summary, pairs, roots


def codec_analysis(cases, tolerance):
    by_key = {(r['condition'], r['episode_id']): r for r in cases}
    ids = [r['episode_id'] for r in cases if r['condition'] == local.REFERENCE]
    sweep, solved, fair = [], [], []
    for budget in local.BUDGETS:
        raw, kv, q = [capacity.condition(budget, p) for p in local.POLICIES]
        for name in (kv, q):
            for bandwidth in BANDWIDTHS:
                comparators = [local.REFERENCE, raw]+([kv] if name == q else [])
                for before in comparators:
                    per_target = []
                    for i in ids:
                        a, b = by_key[name, i], by_key[before, i]
                        offset = a['target_total_ms']-a['storage_decode_ms']-b['target_total_ms']+.008*(a['transport_bytes']-b['transport_bytes'])/bandwidth
                        per_target.append(dict(episode_id=i, **codec_root(offset, a['storage_decode_ms'])))
                    solved.append(dict(budget='B'+str(budget), after=name, before=before, bandwidth_mbps=bandwidth,
                        model='A_DECODE_ACCELERATED_COMPARATOR_UNCHANGED',
                        evidence_role='SECONDARY_ASYMMETRIC_SENSITIVITY' if before == kv else 'PRIMARY_VS_UNCOMPRESSED',
                        aggregate_mean=codec_root(mean(r['offset_ms'] for r in per_target), mean(r['decode_coefficient_ms'] for r in per_target)),
                        per_target_distribution=root_distribution(per_target, 'required_speedup'), per_target=per_target))
                for speedup in SPEEDUPS:
                    estimates = []
                    for i in ids:
                        a = by_key[name, i]
                        value = accelerated_local(a['target_total_ms'], a['storage_decode_ms'], speedup)+transfer_ms(a['transport_bytes'], bandwidth)
                        full, r, k = by_key[local.REFERENCE, i], by_key[raw, i], by_key[kv, i]
                        estimates.append(dict(e2e_ms=value, accelerated_local_ms=accelerated_local(a['target_total_ms'], a['storage_decode_ms'], speedup),
                            delta_vs_FULL=value-full['target_total_ms']-transfer_ms(full['transport_bytes'], bandwidth),
                            delta_vs_RAW=value-r['target_total_ms']-transfer_ms(r['transport_bytes'], bandwidth),
                            delta_vs_uniform_KV=value-accelerated_local(k['target_total_ms'], k['storage_decode_ms'], speedup)-transfer_ms(k['transport_bytes'], bandwidth)))
                    sweep.append(dict(condition=name, budget='B'+str(budget), bandwidth_mbps=bandwidth, codec_speedup=speedup,
                        estimate='HYPOTHETICAL_ACCELERATION', analytical_e2e_ms=local.stats([r['e2e_ms'] for r in estimates]),
                        mean_accelerated_local_ms=mean(r['accelerated_local_ms'] for r in estimates),
                        mean_delta_vs_FULL=mean(r['delta_vs_FULL'] for r in estimates),
                        mean_delta_vs_RAW=mean(r['delta_vs_RAW'] for r in estimates),
                        mean_delta_vs_same_speedup_KV=mean(r['delta_vs_uniform_KV'] for r in estimates)))
        for bandwidth in BANDWIDTHS:
            per_target = []
            for i in ids:
                a, b = by_key[q, i], by_key[kv, i]
                offset = (a['target_total_ms']-a['storage_decode_ms'])-(b['target_total_ms']-b['storage_decode_ms'])+.008*(a['transport_bytes']-b['transport_bytes'])/bandwidth
                per_target.append(dict(episode_id=i, **codec_root(offset, a['storage_decode_ms']-b['storage_decode_ms'])))
            fair.append(dict(budget='B'+str(budget), before=kv, after=q, bandwidth_mbps=bandwidth,
                model='SAME_UNIFORM_SPEEDUP_BOTH_DECODE_PATHS', evidence_role='PRIMARY_NATIVE_BACKEND_COMPARISON',
                aggregate_mean=codec_root(mean(r['offset_ms'] for r in per_target), mean(r['decode_coefficient_ms'] for r in per_target)),
                per_target_distribution=root_distribution(per_target, 'required_speedup'), per_target=per_target))
    return sweep, dict(estimate='HYPOTHETICAL_ACCELERATION', comparator_unchanged=solved, same_speedup_q24_vs_kv=fair)


def source_amortization(source):
    rows = source.get('conditions', [])
    by_name = provenance.unique(rows, 'condition')
    require(set(by_name) == set(local.CONDITIONS[1:]), 'Missing/extra source-build contexts')
    measured, sensitivities = [], []
    for name in local.CONDITIONS[1:]:
        row = by_name[name]
        require(row.get('source_admission_events') == 32, 'Source admission count differs')
        cost = row.get('storage_encode_ms'); finite(cost); require(cost >= 0, 'Negative source preparation cost')
        measured.append(dict(condition=name, measured_storage_encode_total_ms=cost,
            measured_storage_encode_mean_ms_per_admission=cost/32))
    for policy in local.POLICIES:
        require(by_name[capacity.condition(2, policy)]['storage_encode_ms'] == by_name[capacity.condition(8, policy)]['storage_encode_ms'],
            'Shared source encoding differs across budgets')
    for budget in local.BUDGETS:
        for before, after in (('RAW_QKV', 'KV_COMP'), ('RAW_QKV', 'Q24_KV_COMP'), ('KV_COMP', 'Q24_KV_COMP')):
            a, b = capacity.condition(budget, after), capacity.condition(budget, before)
            incremental = (by_name[a]['storage_encode_ms']-by_name[b]['storage_encode_ms'])/32
            sensitivities.append(dict(before=b, after=a, incremental_mean_storage_encode_ms=incremental,
                amortized_per_request_ms={str(n): incremental/n for n in REUSE_COUNTS}))
    return dict(label='SECONDARY AMORTIZATION SENSITIVITY', source_cost_in_primary_steady_state=False,
        excluded='source forward/capture/admission and disk/model load; sensitivity uses incremental measured storage preparation/encode only',
        reuse_counts=list(REUSE_COUNTS), shared_budget_encoding_not_summed=True, measured_source_encode=measured, comparisons=sensitivities)


def classify(pairs, codecs, tolerance):
    contexts = []
    for budget in local.BUDGETS:
        for policy in ('KV_COMP', 'Q24_KV_COMP'):
            name = capacity.condition(budget, policy)
            competitors = {local.REFERENCE, capacity.condition(budget, 'RAW_QKV')}
            if policy == 'Q24_KV_COMP': competitors.add(capacity.condition(budget, 'KV_COMP'))
            competitive = []
            for bandwidth in BANDWIDTHS:
                values = [r for r in pairs if r['after'] == name and r['before'] in competitors and r['bandwidth_mbps'] == bandwidth]
                require(len(values) == len(competitors), 'Incomplete classification comparisons')
                if all(r['mean_delta_e2e_ms'] <= tolerance for r in values): competitive.append(bandwidth)
            suffix = 'SYSTEM_COMPETITIVE' if len(competitive) == len(BANDWIDTHS) else 'NETWORK_CONDITIONAL' if competitive else 'NOT_LATENCY_COMPETITIVE'
            contexts.append(dict(condition=name, recommendation='CURRENT_PY_CODEC_'+suffix, competitive_bandwidths_mbps=competitive))
    suffix = 'SYSTEM_COMPETITIVE' if all(r['recommendation'].endswith('SYSTEM_COMPETITIVE') for r in contexts) else 'NETWORK_CONDITIONAL' if any(r['competitive_bandwidths_mbps'] for r in contexts) else 'NOT_LATENCY_COMPETITIVE'
    primary = [r for r in codecs['comparator_unchanged'] if r['evidence_role'] == 'PRIMARY_VS_UNCOMPRESSED']+codecs['same_speedup_q24_vs_kv']
    native = any(r['aggregate_mean']['required_speedup'] is not None and r['aggregate_mean']['required_speedup'] > 1 for r in primary)
    return dict(recommendation='CURRENT_PY_CODEC_'+suffix, native_backend_required=native,
        rule=CLASSIFICATION_RULE, native_backend_rule=NATIVE_RULE, contexts=contexts,
        no_finite_speedup_primary_contexts=sum(r['aggregate_mean']['required_speedup'] is None for r in primary))


def summary_md(prepared, bandwidth, roots, codecs, amortization, classification):
    lines = ['# C9-B network / storage-codec break-even', '',
        'MEASURED: C9-A prompt local latency. ANALYTICAL: payload transfer and summed end-to-end estimates. HYPOTHETICAL_ACCELERATION: decode time divided by S.',
        'No distributed hardware, RTT, quality evaluation, codec execution, or final policy freeze. Frozen32 remains a selection-based controlled trace.',
        'Storage/capacity benefit (C8), current Python codec latency (C9-A), and potential network/native-backend benefit are separate evidence.', '',
        'Transport is raw user LoRA Q/K/V delta, not resident compressed TOTAL-QKV frames. Frozen adapters have FP32 delta weights; cached TOTAL tensors are FP16.',
        'HIT skips 3 rows per role/layer. Full and MISS account all prompt rows; prompt lengths vary. C6 FULL native zero counters are an instrumentation gap, not zero EdgeLoRA payload.',
        'Network transactions/control RTT are unproven; only logical projection payload counts are reported.', '',
        '| Condition | HITs | Total raw delta bytes /32 targets | Mean raw bytes | Saved vs FULL | Mean MEASURED local ms |', '|---|---:|---:|---:|---:|---:|']
    by_transport = {r['condition']: r for r in prepared.transport_summary['conditions']}
    for name in local.CONDITIONS:
        r = by_transport[name]
        measured_local = mean(row['target_total_ms'] for row in prepared.local_cases if row['condition'] == name)
        lines.append(f"| {name} | {r['target_hits']} | {r['total_transport_bytes']} | {r['mean_transport_bytes']:.3f} | {r['bytes_saved_vs_FULL']} | {measured_local:.6f} |")
    for budget in local.BUDGETS:
        q, kv = [by_transport[capacity.condition(budget, p)] for p in ('Q24_KV_COMP', 'KV_COMP')]
        lines.append(f"B{budget}: additional Q24 coverage saves {kv['total_transport_bytes']-q['total_transport_bytes']} bytes versus KV over 32 prompts.")
    lines += ['', 'Exact bandwidth break-even from mean local and payload deltas (not the coarse sweep):',
        '| A | Comparator B | Break-even Mbps | A winning region /reason |', '|---|---|---:|---|']
    for r in roots:
        a = r['aggregate_mean']; lines.append(f"| {r['after']} | {r['before']} | {a['break_even_bandwidth_mbps']} | {a['winning_bandwidth_region']} /{a['reason']} |")
    lines += ['', 'Lowest mean ANALYTICAL_END_TO_END_ESTIMATE at each tested bandwidth (includes FULL):']
    for budget in local.BUDGETS:
        for bw in BANDWIDTHS:
            rows = [r for r in bandwidth if r['bandwidth_mbps'] == bw and (r['condition'] == local.REFERENCE or r['condition'].startswith('B'+str(budget)+'_'))]
            winner = min(rows, key=lambda r: r['analytical_e2e_ms']['mean'])
            lines.append(f"B{budget} at {bw}Mbps: {winner['condition']}, mean {winner['analytical_e2e_ms']['mean']:.6f}ms.")
    lines += ['', 'Exact HYPOTHETICAL_ACCELERATION factors; null means NO_FINITE_CODEC_SPEEDUP_CAN_BREAK_EVEN:',
        '| A | Comparator B | Mbps | Minimum S | Model |', '|---|---|---:|---:|---|']
    for r in codecs['comparator_unchanged']+codecs['same_speedup_q24_vs_kv']:
        a = r['aggregate_mean']
        lines.append(f"| {r['after']} | {r['before']} | {r['bandwidth_mbps']} | {a['required_speedup']} | {r['evidence_role']} /{a['status']} |")
    lines += ['', 'The primary Q24-vs-KV native-backend model accelerates BOTH decode paths equally. One-sided Q24 acceleration is only secondary sensitivity. A non-decode/network floor cannot be repaired by codec speed alone.',
        '', 'SECONDARY AMORTIZATION SENSITIVITY (excluded from primary results):']
    for r in amortization['measured_source_encode']:
        lines.append(f"{r['condition']}: measured preparation/encode total {r['measured_storage_encode_total_ms']:.6f}ms; mean/admission {r['measured_storage_encode_mean_ms_per_admission']:.6f}ms.")
    for r in amortization['comparisons']:
        lines.append(f"{r['after']} minus {r['before']}: incremental encode/admission {r['incremental_mean_storage_encode_ms']:.6f}ms; amortized N=1/8/32: {r['amortized_per_request_ms']}.")
    lines += ['', 'Classification rule: '+CLASSIFICATION_RULE, 'Native backend flag rule: '+NATIVE_RULE,
        'Optimized/native backend required under this descriptive flag: '+str(classification['native_backend_required'])+'.',
        'C6 artifact transport cross-check: '+prepared.transport_summary['cross_checks']['status']+'.',
        'Payload-only bandwidth regions and hypothetical speedups do not establish measured distributed performance. No policy is frozen.',
        '', classification['recommendation'], '']
    return '\n'.join(lines)


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        writer.writerows(capacity.csv_form(row) for row in rows)


def run(args, prepared):
    root = args.output_root
    provenance.output_location(root)
    require(not root.exists() or not any(root.iterdir()), 'Refusing nonempty C9-B output root')
    cases = analyzed_cases(prepared)
    sweep, bandwidth, pairs, roots = bandwidth_analysis(cases, prepared.tie_tolerance_ms)
    codec_sweep, codecs = codec_analysis(cases, prepared.tie_tolerance_ms)
    amortization = source_amortization(prepared.source)
    classification = classify(pairs, codecs, prepared.tie_tolerance_ms)
    provenance.verify_files(prepared.input_hashes)
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root/'transport_bytes_per_case.csv', prepared.transport)
    write_csv(root/'bandwidth_sweep.csv', sweep)
    write_csv(root/'codec_speedup_sweep.csv', codec_sweep)
    for name, value in (('transport_byte_summary.json', prepared.transport_summary),
        ('bandwidth_summary.json', dict(estimate='ANALYTICAL_END_TO_END_ESTIMATE', network_model='PAYLOAD_TRANSFER_ONLY', results=bandwidth)),
        ('network_break_even.json', dict(model='delta_local_ms + .008*delta_bytes/B_Mbps', comparisons=roots)),
        ('codec_break_even.json', codecs), ('source_amortization.json', amortization),
        ('pairwise_system_comparisons.json', dict(estimate='ANALYTICAL_END_TO_END_ESTIMATE', comparisons=pairs, classification=classification))):
        provenance.write(root/name, value)
    (root/'summary.md').write_text(summary_md(prepared, bandwidth, roots, codecs, amortization, classification))
    manifest = dict(stage='C9-B', status='COMPLETE',
        research_scope='analytical network-bandwidth and resident-storage-codec break-even analysis',
        network_model='PAYLOAD_TRANSFER_ONLY', distributed_network_measured=False, analytical_end_to_end=True,
        bandwidth_mbps=list(BANDWIDTHS), codec_speedup_factors=list(SPEEDUPS), source_amortization_reuse_counts=list(REUSE_COUNTS),
        policies=list(local.POLICIES), budgets=list(local.BUDGETS), conditions=list(local.CONDITIONS),
        physical_safety_contract=provenance.CONTRACT, cached_payload='TOTAL_QKV', transport_compression_enabled=False,
        c7b3_freeze_sha256=provenance.sha(args.freeze_decision),
        c8a_manifest_sha256=provenance.sha(args.c8a_root/'manifest.json'), c8a_residency_trace_sha256=provenance.sha(args.c8a_root/'residency_trace.json'),
        c8b_manifest_sha256=provenance.sha(args.c8b_root/'manifest.json'),
        c9a_artifact_hashes={name: provenance.sha(args.c9a_root/name) for name in ('manifest.json', *REQUIRED_INPUTS)},
        q24_profile_sha256=provenance.Q_SHA, kv_profile_sha256=provenance.KV_SHA,
        model_revision=prepared.c9_manifest['model_revision'], adapter_hashes=prepared.c9_manifest['adapter_hashes'],
        transport_accounting_source=prepared.transport_summary['transport_accounting_source'], transport_accounting_contract=transport.CONTRACT,
        local_latency_source='MEASURED_C9_A', network_latency_source='ANALYTICAL_PAYLOAD_ONLY',
        codec_acceleration_source='HYPOTHETICAL_ANALYTICAL', runtime_profile_fit_count=0,
        **{k: 0 for k in capacity.SAFETY},
        model_inference_performed=False, quality_evaluated=False, training_performed=False, system_policy_frozen=False,
        source_build_in_primary_steady_state=False, frozen32_used_for_q_selection=True, unbiased_final_test=False,
        tie_tolerance_ms=prepared.tie_tolerance_ms, classification=classification,
        recommendation=classification['recommendation'], native_backend_required=classification['native_backend_required'],
        input_hashes=prepared.input_hashes, git=provenance.git(),
        output_hashes={p.name: provenance.sha(p) for p in root.iterdir() if p.is_file()})
    provenance.write(root/'manifest.json', manifest)
    print('C9-B COMPLETE: '+classification['recommendation'])
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in dict(plan_dir='results/cachegen/c6b3/multiwoz_plan_v2',
        freeze_decision='results/cachegen/c7/q24_freeze/freeze_decision.json', c8a_root='results/cachegen/c8/a_capacity_replay',
        c8b_root='results/cachegen/c8/b_quality_b2_b8', c9a_root='results/cachegen/c9/a_local_latency_retry1',
        output_root='results/cachegen/c9/b_network_break_even').items():
        parser.add_argument('--'+name.replace('_', '-'), type=Path, default=Path(default))
    args = parser.parse_args(argv)
    require(not args.output_root.exists() or not any(args.output_root.iterdir()), 'Refusing nonempty C9-B output root')
    provenance.output_location(args.output_root)
    run(args, prepare(args))
