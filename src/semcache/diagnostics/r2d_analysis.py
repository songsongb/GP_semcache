"""Separate-cohort analytical results; C4 observations never enter M9-A totals."""
from collections import defaultdict
from statistics import mean

from semcache.system_cost.network import transfer_ms
from .r2d_evidence import MISSING_E2E, compatibility
from .r2d_network import break_even, modeled_point, payload_ledger, rtt_sensitivity

R3_NEXT_STEPS = [
    'Implement and identify the actual ES base / UD adapter partition; verify adapter weights and output parity.',
    'Trace every application message: type, direction, layer, token rows, payload and framed bytes, including h0/hL.',
    'Measure message counts, dependency round trips, RTT, achieved throughput and the actual serialization/overlap schedule.',
    'Record CPU/GPU placement, wire dtype conversions, synchronization, queueing and exclusive host/device timing boundaries.',
    'Measure cold build/fill and warm request walls, residency/refresh traffic and reuse frequency under normal Admission.',
    'Pair matching no-reuse/reuse requests at the same length, endpoint, hardware and adapter; validate strict token/logit quality.',
    'Measure distributed wall clock, TTFT, full generation/decode and token quality; retain failed and negative results.',
]


def diagnostic_components(rows):
    """Summarize each diagnostic clock individually, preserving nesting labels."""
    groups = defaultdict(list)
    for row in rows:
        for c in row['components']:
            key = (row['case_id'], row['arm'], c['component'], c.get('location'),
                   c.get('parent'), c.get('status'), c.get('inclusive'))
            groups[key].append(c)
    output = []
    for key, block in sorted(groups.items(), key=lambda p: repr(p[0])):
        case, arm, component, location, parent, status, inclusive = key
        clocks = {}
        for field in ('host_ms', 'cuda_ms'):
            values = [c[field] for c in block if c.get(field) is not None]
            clocks[field] = dict(status='AVAILABLE' if values else 'NOT_ESTIMABLE',
                count=len(values), mean_ms=mean(values) if values else None)
        output.append(dict(case_id=case, arm=arm, component=component, location=location,
            parent=parent, status=status, inclusive=inclusive, clocks=clocks,
            provenance='MEASURED_SEPARATE_NONMINIMAL_DIAGNOSTIC', do_not_sum_nested=True,
            pooled_with_minimal_latency=False,
            gpu_interval_is_kernel_self_time=False))
    return output


def _root_record(offset, delta_bytes, rtt_delta, bandwidths):
    root = break_even(offset, delta_bytes, rtt_delta)
    value = root['break_even_bandwidth_mbps']
    return dict(root, tested_bandwidth_domain_mbps=[min(bandwidths), max(bandwidths)],
        root_outside_tested_domain=value is not None and not min(bandwidths) <= value <= max(bandwidths),
        out_of_domain_warning='A root is algebraic extrapolation beyond tested bandwidths when root_outside_tested_domain=true; historical compute is never extrapolated to other lengths.')


def analyze(m9a, c4, bandwidths, rtts):
    sweep, historical = [], []
    fixture = m9a['summary'].get('TEST_FIXTURE_ONLY') is True or c4['manifest'].get('TEST_FIXTURE_ONLY') is True
    dims = m9a['dimensions']
    for comparison in m9a['records']:
        r = comparison['row']
        identifier = f"{r['experiment_id']}:{r['query_id']}:{r['user_id']}:rep{r['repeat_index']}"
        ledger = payload_ledger(dims, r['prompt_tokens'], r['reused_tokens'],
            hidden_element_bytes=r['hidden_element_bytes'], delta_element_bytes=r['delta_element_bytes'],
            boundary_transfers=m9a['boundary_transfers'])
        full_compute = r['es_compute_ms']+r['ud_lora_ms']
        reuse_compute = r['semcache_es_ms']+r['semcache_ud_ms']
        control = r['semcache_control_ms']
        offset = reuse_compute+control-full_compute
        delta_bytes = ledger['reuse']['total_network_bytes']-ledger['no_reuse']['total_network_bytes']
        roots = []
        for rtt_ms in rtts:
            rtt = rtt_sensitivity(dims.layers, r['fresh_tokens'], rtt_ms,
                                  boundary_transfers=m9a['boundary_transfers'])
            root = _root_record(offset, delta_bytes, rtt['delta_rtt_ms'], bandwidths)
            winners = defaultdict(list)
            for bandwidth in bandwidths:
                payload = modeled_point(full_compute, reuse_compute+control, ledger, bandwidth, rtt,
                                        include_hypothetical_rtt=False)
                hypothetical = modeled_point(full_compute, reuse_compute+control, ledger, bandwidth, rtt,
                                             include_hypothetical_rtt=True)
                winners[hypothetical['winner']].append(bandwidth)
                row = dict(evidence_class='HISTORICAL_M9A', TEST_FIXTURE_ONLY=fixture, condition_id=identifier, model=dims.model_id,
                    prompt_tokens=r['prompt_tokens'], reused_tokens=r['reused_tokens'], reuse_ratio=r['reuse_ratio'],
                    repeat_index=r['repeat_index'], bandwidth_mbps=bandwidth, rtt_ms=rtt_ms,
                    source_scope=comparison['scope'], validity_status='SUPPORTED_HISTORICAL_MODEL' if r['primary_comparison_eligible'] else 'EXPLORATORY_PROXY_NOT_PRIMARY',
                    transfer_policy='SERIALIZED_NO_OVERLAP', payload_provenance='SIMULATED',
                    hidden_element_bytes=r['hidden_element_bytes'], delta_element_bytes=r['delta_element_bytes'],
                    boundary_policy=ledger['boundary_policy'], actual_message_counts='NOT_ESTIMABLE',
                    actual_rtt_status=rtt['actual_rtt_status'], hypothetical_rtt_provenance='HYPOTHETICAL_RTT',
                    hypothetical_scenario=rtt['scenario'],
                    hypothetical_no_reuse_exchanges=rtt['no_reuse_exchanges'], hypothetical_reuse_exchanges=rtt['reuse_exchanges'],
                    hypothetical_no_reuse_rtt_ms=rtt['no_reuse_rtt_ms'], hypothetical_reuse_rtt_ms=rtt['reuse_rtt_ms'],
                    no_reuse_es_ms=r['es_compute_ms'], reuse_es_ms=r['semcache_es_ms'], es_provenance=r['es_compute_ms_provenance'],
                    no_reuse_ud_ms=r['ud_lora_ms'], reuse_ud_ms=r['semcache_ud_ms'], ud_provenance=r['ud_lora_ms_provenance'],
                    reuse_ud_provenance=r['semcache_ud_ms_provenance'],
                    no_reuse_compute_ms=full_compute, reuse_compute_ms=reuse_compute,
                    reuse_control_ms=control, control_provenance='MEASURED',
                    delta_local_ms=offset, payload_only_reuse_minus_no_reuse_ms=payload['reuse_minus_no_reuse_ms'],
                    payload_only_no_reuse_total_ms=payload['no_reuse_total_ms'], payload_only_reuse_total_ms=payload['reuse_total_ms'],
                    payload_only_winner=payload['winner'], payload_only_speedup_ratio=payload['speedup_ratio'],
                    payload_only_speedup_percent=payload['speedup_percent'],
                    hypothetical_reuse_minus_no_reuse_ms=hypothetical['reuse_minus_no_reuse_ms'],
                    hypothetical_no_reuse_total_ms=hypothetical['no_reuse_total_ms'], hypothetical_reuse_total_ms=hypothetical['reuse_total_ms'],
                    hypothetical_speedup_ratio=hypothetical['speedup_ratio'], hypothetical_speedup_percent=hypothetical['speedup_percent'],
                    hypothetical_winner=hypothetical['winner'], total_provenance='SIMULATED',
                    actual_rtt_adjusted_delta_ms=payload['reuse_minus_no_reuse_ms'] if rtt_ms == 0 else None,
                    bandwidth_root_status=root['status'], break_even_bandwidth_mbps=root['break_even_bandwidth_mbps'],
                    break_even_qualification='HISTORICAL_PAYLOAD_MODEL_ONLY' if rtt_ms == 0 else 'HYPOTHETICAL_RTT_ONLY',
                    historical_primary_eligible=r['primary_comparison_eligible'], historical_correctness_gate=r['correctness_gate_passed'],
                    output_correctness_qualification='historical controlled fixture only; not distributed output validation',
                    correctness_qualified_speedup=False, real_network_measured=False)
                for path in ('no_reuse', 'reuse'):
                    for field in ('ud_to_es_bytes', 'es_to_ud_bytes', 'total_network_bytes'):
                        row[path+'_'+field] = ledger[path][field]
                    for direction in ('ud_to_es', 'es_to_ud'):
                        row[path+'_'+direction+'_transfer_ms'] = transfer_ms(ledger[path][direction+'_bytes'], bandwidth)
                    row[path+'_transfer_ms'] = payload[path+'_transfer_ms']
                row['saved_payload_bytes'] = -delta_bytes
                sweep.append(row)
            roots.append(dict(rtt_ms=rtt_ms, qualification='PAYLOAD_ONLY' if rtt_ms == 0 else 'HYPOTHETICAL_RTT',
                actual_rtt_adjusted_root_status='PAYLOAD_ONLY_ZERO_RTT' if rtt_ms == 0 else 'NOT_ESTIMABLE',
                rtt_assumptions=rtt, root=root, tested_bandwidths_by_winner=dict(winners)))
        historical.append(dict(condition_id=identifier, prompt_tokens=r['prompt_tokens'], reused_tokens=r['reused_tokens'],
            reuse_ratio=r['reuse_ratio'], ledger=ledger, delta_local_ms=offset, delta_payload_bytes=delta_bytes,
            components=comparison['components'], original_accounting_bandwidth_mbps=r['bandwidth_mbps'],
            historical_components_scope='original validated M9-A accounting; new sweep recomputes payload at each bandwidth without readding these network totals',
            roots=roots,
            historical_primary_eligible=r['primary_comparison_eligible'],
            zero_reuse_network_control=dict(status='ANALYTICAL_IDENTITY_NOT_A_MEASUREMENT',
                saved_projection_payload_bytes=0, boundary_savings_bytes=0,
                zero_reuse_compute_delta_ms=None, compute_status='NOT_ESTIMABLE_WITHOUT_MATCHING_PROFILE')))

    local = []
    for c in c4['conditions']:
        penalty = c['comparisons']['baseline_minus_no_reuse_ms']['observed_equal_session_mean_ms']
        positive = c['actual_reused_tokens'] > 0
        thresholds = []
        for bandwidth in bandwidths:
            # This solves for REQUIRED saved bytes, not observed C4 traffic.
            required = penalty*bandwidth/.008 if positive else None
            thresholds.append(dict(bandwidth_mbps=bandwidth, required_avoided_payload_bytes=required,
                required_saved_round_trips_by_rtt=[dict(rtt_ms=t, signed_required_saved_round_trips=penalty/t if positive and t else None,
                    status='HYPOTHETICAL_RTT_ONLY' if positive and t else 'NOT_APPLICABLE') for t in rtts],
                assumption='equal RTT charges for payload threshold; zero payload difference for RTT threshold; each is separate, fractional thresholds are not application message counts'))
            for rtt_ms in rtts:
                sweep.append(dict(evidence_class='R2C4_CONDITIONAL_ILLUSTRATIVE', TEST_FIXTURE_ONLY=fixture, condition_id=c['case_id'], model=c4['cohort']['model'],
                    prompt_tokens=c['length'], reused_tokens=c['actual_reused_tokens'], reuse_ratio=c['actual_coverage'],
                    bandwidth_mbps=bandwidth, rtt_ms=rtt_ms, source_scope='separate local conditional request observation',
                    validity_status='NOT_ESTIMABLE' if positive else 'NOT_APPLICABLE',
                    actual_message_counts='NOT_ESTIMABLE', actual_rtt_status='NOT_ESTIMABLE',
                    payload_provenance='NOT_ESTIMABLE', transfer_policy='HYPOTHETICAL_SERIALIZED_THRESHOLD_ONLY',
                    observed_local_raw_minus_no_reuse_ms=penalty, local_measurement_provenance='MEASURED',
                    required_avoided_payload_bytes=required, threshold_provenance='HYPOTHETICAL_NETWORK_ASSUMPTION',
                    bandwidth_root_status='NOT_ESTIMABLE', break_even_qualification='UNQUALIFIED_ILLUSTRATIVE_ONLY',
                    output_correctness_qualification='PASS_LOCAL_ONLY' if c['full_relative_strict']['RAW_REPLAY_BASELINE'] else 'STRICT_FULL_RELATIVE_FAIL',
                    correctness_qualified_speedup=False, real_network_measured=False,
                    hypothetical_scenario='required byte savings; C4 distributed bytes and M9-A compute are not inserted'))
        local.append(dict(case_id=c['case_id'], length=c['length'], actual_reused_tokens=c['actual_reused_tokens'],
            actual_coverage=c['actual_coverage'], comparisons=c['comparisons'],
            full_relative_strict=c['full_relative_strict'],
            status='ILLUSTRATIVE_COUNTERFACTUAL_ONLY' if positive else 'NOT_APPLICABLE',
            bandwidth_root_status='NOT_ESTIMABLE', thresholds=thresholds,
            oracle_diagnostics_deployable=False,
            oracle_benefits_subtracted_from_m9a=False, local_wall_added_to_m9a=False,
            zero_reuse_interpretation=None if positive else 'negative-control noise; no mapping/snapshot work removed'))
    compat = compatibility(m9a, c4)
    result = dict(schema='r2d_break_even_v1', TEST_FIXTURE_ONLY=fixture, sign_convention='reuse minus no-reuse; negative favors reuse',
        units=dict(time='ms', bytes='tensor payload bytes', bandwidth='decimal Mbps (1e6 bits/s)'),
        transfer_policy='SERIALIZED_NO_OVERLAP', historical_m9a=historical,
        r2c4_conditional_illustrative=local, r2c4_separate_diagnostic_components=diagnostic_components(c4['breakdown']),
        r2c4_logical_copy_counters=[dict(case_id=r['case_id'], arm=r['arm'], repeat=r['repeat'],
            counters=r.get('path', {}).get('qkv_memory_counters'), network_bytes=False,
            interpretation='GPU-side logical memory copy counters, not ES/UD traffic') for r in c4['breakdown']],
        actual_rtt_cost_status='NOT_ESTIMABLE', overlap_policy_status='NOT_ESTIMABLE',
        historical_bandwidth_duplicates_collapsed=True, historical_compute_length_extrapolated=False,
        source_build_amortization_status='NOT_ESTIMABLE', natural_semantic_hit_rate_status='NOT_ESTIMABLE',
        assumptions=[
            'M9-A equations only skip per-layer fresh-row hidden and three LoRA delta payloads; retained h0/hL are not skipped.',
            'Tensor occurrences do not establish message counts, one-way sends or round trips.',
            'A hypothetical serial RTT/layer plus boundary pair bundles delta replies; partial reuse changes bytes, not exchange count.',
            'Different framing, separate delta messages, residency/fill traffic or serialized validation can change or erase the gain.',
            'Compute/network overlap requires a measured dependency schedule; no max() model or speedup is substituted.',
            'Cold build/refresh cost and natural reuse frequency are unknown; the model is conditional on resident reuse.',
            'C4 oracle contrasts and nested CPU/CUDA intervals are diagnostics; they are never additive optimizations.',
        ])
    decision = dict(execution_status='COMPLETE', classification='ANALYTICAL_DIAGNOSTIC', TEST_FIXTURE_ONLY=fixture,
        validity_status='VALID_SEPARATE_COHORTS_NOT_COMPARABLE',
        primary_claim_eligible=False, real_network_measured=False, full_generation_e2e_measured=False,
        correctness_qualified_speedup=False, safe_reuse_claimed=False,
        gpu_inference_performed=False, network_executed=False,
        analytical_competitiveness=dict(status='TEST_FIXTURE_ONLY' if fixture else 'HISTORICAL_MODEL_ONLY',
            supported_within_m9a=bool(m9a['cohort']['strict_primary_eligible']) and not fixture,
            tested_payload_winners=sorted({r['payload_only_winner'] for r in sweep if r['evidence_class'] == 'HISTORICAL_M9A'}),
            measured_distributed_speedup_status='NOT_ESTIMABLE'),
        counterfactual_vs_supported=dict(historical_m9a='validated original compute/payload accounting; SIMULATED totals',
            r2c4='validated MEASURED local conditional contrasts; strict failures retained',
            rtt='HYPOTHETICAL_RTT; real exchange count NOT_ESTIMABLE',
            c4_network_thresholds='ILLUSTRATIVE_COUNTERFACTUAL_ONLY; traffic savings unknown'),
        compatibility_status=compat['status'], missing_e2e_components=MISSING_E2E, r3_next_steps=R3_NEXT_STEPS,
        conclusion='R2-D reconciles historical M9-A analytical payload/compute accounting and separate R2-C4 local diagnostics. Bandwidth competitiveness is conditional on the historical resident-reuse model; RTT sensitivity and C4 traffic thresholds are hypothetical. The cohorts are NOT_COMPARABLE for pooled latency. This establishes neither correctness-qualified performance nor measured distributed E2E speedup; R3 remains required.')
    return sweep, result, decision, compat
