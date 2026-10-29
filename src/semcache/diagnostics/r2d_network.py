"""Pure EdgeLoRA payload arithmetic and explicitly hypothetical RTT sensitivity.

All deltas are reuse minus no-reuse. Mbps is decimal. No message schedule is
inferred from a tensor count, and no socket, model or device is constructed.
"""
import math

from semcache.system_cost.common import integer, nonnegative
from semcache.system_cost.network import communication, transfer_ms


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return value


def payload_ledger(dims, prompt_tokens, reused_tokens, *, hidden_element_bytes,
                   delta_element_bytes, boundary_transfers):
    """Retain h0/hL once each, even at 100% reuse; skip only projection exchange."""
    if type(boundary_transfers) is not bool:
        raise ValueError('boundary_transfers must be explicit boolean')
    kwargs = dict(hidden_element_bytes=hidden_element_bytes,
                  delta_element_bytes=delta_element_bytes,
                  boundary_transfers=boundary_transfers)
    full = communication(dims, prompt_tokens, **kwargs)
    reuse = communication(dims, prompt_tokens, reused_tokens, **kwargs)
    entries = []
    for name, direction, width, branches, boundary in (
        ('hidden_activation', 'ES_TO_UD', hidden_element_bytes, 1, False),
        ('three_lora_deltas', 'UD_TO_ES', delta_element_bytes, 3, False),
        ('full_h0', 'UD_TO_ES', hidden_element_bytes, 1, True),
        ('full_hL', 'ES_TO_UD', hidden_element_bytes, 1, True),
    ):
        multiplier = int(boundary_transfers) if boundary else dims.layers
        before = prompt_tokens * dims.hidden_size * width * branches
        after = (prompt_tokens if boundary else prompt_tokens-reused_tokens) * dims.hidden_size * width * branches
        entries.append(dict(message_type=name, direction=direction,
            tensor_occurrences=multiplier, application_message_count=None,
            occurrence_unit='boundary tensor' if boundary else 'layer tensor group',
            hidden_size=dims.hidden_size, element_bytes=width, projection_branches=branches,
            no_reuse_tokens=prompt_tokens, reuse_tokens=prompt_tokens if boundary else prompt_tokens-reused_tokens,
            no_reuse_bytes_per_occurrence=before, reuse_bytes_per_occurrence=after,
            no_reuse_bytes=before*multiplier, reuse_bytes=after*multiplier,
            saved_bytes=(before-after)*multiplier, boundary_tensor=boundary,
            boundary_retained=boundary and boundary_transfers,
            provenance='SIMULATED', source='M9-A communication() tensor_payload_only_REPRODUCTION_CHOICE'))
    fields = ('ud_to_es_bytes', 'es_to_ud_bytes', 'total_network_bytes')
    return dict(entries=entries, no_reuse={k: full['request'][k] for k in fields},
        reuse={k: reuse['request'][k] for k in fields},
        saved={k: full['request'][k]-reuse['request'][k] for k in fields},
        hidden_element_bytes=hidden_element_bytes, delta_element_bytes=delta_element_bytes,
        boundary_policy=full['boundary_policy'], provenance='SIMULATED',
        residency_assumption='required reusable projections already resident; no cache fill/refresh traffic modeled',
        transfer_policy='SERIALIZED_NO_OVERLAP', actual_message_counts='NOT_ESTIMABLE',
        excluded='headers, conversion, cache fill, protocols, contention, overlap, actual RTT scheduling')


def rtt_sensitivity(layers, fresh_tokens, rtt_ms, *, boundary_transfers,
                    exchanges_per_layer=1, boundary_exchanges=1):
    """Editable ASSUMPTION: serialized one RTT/layer, plus one boundary pair.

    Bundling all three deltas into the reply is hypothetical. Nonzero fresh
    tokens shrink bytes but do not remove any layer exchange in this scenario.
    A completely reused layer omits its hypothetical exchange. No inference
    about the real implementation's messages or overlap follows from this.
    """
    integer(layers, 'layers', 1)
    integer(fresh_tokens, 'fresh_tokens')
    integer(exchanges_per_layer, 'exchanges_per_layer')
    integer(boundary_exchanges, 'boundary_exchanges')
    nonnegative(rtt_ms, 'rtt_ms')
    if type(boundary_transfers) is not bool:
        raise ValueError('boundary_transfers must be boolean')
    boundary = boundary_exchanges if boundary_transfers else 0
    full = layers*exchanges_per_layer + boundary
    reuse = (layers*exchanges_per_layer if fresh_tokens else 0) + boundary
    return dict(scenario='ONE_SERIAL_RTT_PER_ACTIVE_LAYER_PLUS_BOUNDARY_PAIR',
        provenance='HYPOTHETICAL_RTT', policy='SERIALIZED_NO_OVERLAP',
        exchanges_per_layer=exchanges_per_layer, retained_boundary_exchanges=boundary,
        no_reuse_exchanges=full, reuse_exchanges=reuse,
        no_reuse_rtt_ms=full*rtt_ms, reuse_rtt_ms=reuse*rtt_ms,
        delta_rtt_ms=(reuse-full)*rtt_ms,
        actual_rtt_status='PAYLOAD_ONLY_ZERO_RTT' if rtt_ms == 0 else 'NOT_ESTIMABLE',
        actual_message_counts=None, assumption_supported_by_measurement=False)


def break_even(delta_local_ms, delta_bytes, delta_rtt_ms=0.):
    """Solve delta_local + delta_rtt + .008*delta_bytes/B = 0, B > 0.

    Accept signed byte differences to cover either traffic/compute tradeoff.
    Roots at infinite or zero bandwidth are limits, never finite roots.
    """
    finite(delta_local_ms, 'delta_local_ms')
    finite(delta_bytes, 'delta_bytes')
    finite(delta_rtt_ms, 'delta_rtt_ms')
    offset = finite(delta_local_ms+delta_rtt_ms, 'offset_ms')
    coefficient = finite(.008*delta_bytes, 'inverse_bandwidth_coefficient')
    result = dict(equation='reuse_minus_no_reuse_ms = offset_ms + .008 * delta_bytes / bandwidth_mbps',
        offset_ms=offset, delta_bytes=delta_bytes, break_even_bandwidth_mbps=None,
        domain='0 < bandwidth_mbps < infinity', strict_tie_at_root=True,
        monotonic_regime='INCREASING' if delta_bytes < 0 else 'DECREASING' if delta_bytes > 0 else 'CONSTANT')
    if offset == 0 and coefficient == 0:
        return dict(result, status='TIED', winning_region='NONE', losing_region='NONE', ties='ALL_POSITIVE_BANDWIDTHS')
    if (offset > 0 and coefficient < 0) or (offset < 0 and coefficient > 0):
        root = -coefficient/offset
        if not math.isfinite(root) or root <= 0:
            return dict(result, status='NOT_ESTIMABLE', winning_region=None,
                        losing_region=None, reason='root outside finite floating point range')
        return dict(result, status='FINITE_ROOT', break_even_bandwidth_mbps=root,
            winning_region='BELOW_ROOT' if offset > 0 else 'ABOVE_ROOT',
            losing_region='ABOVE_ROOT' if offset > 0 else 'BELOW_ROOT')
    wins = offset <= 0 and coefficient <= 0
    return dict(result, status='ALWAYS_WINNING' if wins else 'ALWAYS_LOSING',
        winning_region='ALL_POSITIVE_BANDWIDTHS' if wins else 'NONE',
        losing_region='NONE' if wins else 'ALL_POSITIVE_BANDWIDTHS',
        asymptotic_tie_at_infinite_bandwidth=offset == 0)


def modeled_point(no_reuse_compute_ms, reuse_compute_ms, ledger, bandwidth_mbps,
                  rtt, *, include_hypothetical_rtt):
    for name, value in (('no_reuse_compute_ms', no_reuse_compute_ms), ('reuse_compute_ms', reuse_compute_ms)):
        nonnegative(value, name)
    before = ledger['no_reuse']['total_network_bytes']
    after = ledger['reuse']['total_network_bytes']
    full_transfer, reuse_transfer = transfer_ms(before, bandwidth_mbps), transfer_ms(after, bandwidth_mbps)
    full_rtt = rtt['no_reuse_rtt_ms'] if include_hypothetical_rtt else 0.
    reuse_rtt = rtt['reuse_rtt_ms'] if include_hypothetical_rtt else 0.
    full = finite(no_reuse_compute_ms+full_transfer+full_rtt, 'no_reuse_total_ms')
    reuse = finite(reuse_compute_ms+reuse_transfer+reuse_rtt, 'reuse_total_ms')
    delta = reuse-full
    return dict(no_reuse_transfer_ms=full_transfer, reuse_transfer_ms=reuse_transfer,
        no_reuse_total_ms=full, reuse_total_ms=reuse, reuse_minus_no_reuse_ms=delta,
        time_saved_ms=-delta, speedup_ratio=full/reuse if reuse else None,
        speedup_percent=100*(full-reuse)/full if full else None,
        winner='REUSE' if delta < 0 else 'NO_REUSE' if delta > 0 else 'TIED')
