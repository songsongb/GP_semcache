"""Pure diagnostic recomposition. Original strict inputs/results are read-only."""
import math
from semcache.system_cost.common import nonnegative
from .impact import MODES


def break_even(saved_bytes, compute_delta_ms):
    """Delta(R_Mbps) = compute_delta_ms + saved_bytes * .008 / R_Mbps."""
    nonnegative(saved_bytes, 'saved network bytes')
    if compute_delta_ms < 0 and saved_bytes > 0:
        return dict(analytical_break_even_bandwidth_mbps=saved_bytes*.008/(-compute_delta_ms),
                    break_even_regime='REUSE_below_threshold_RECOMPUTE_above')
    if compute_delta_ms > 0 or (compute_delta_ms == 0 and saved_bytes > 0):
        regime = 'REUSE_at_all_finite_positive_bandwidths'
    elif compute_delta_ms == 0:
        regime = 'TIE_at_all_bandwidths'
    else:
        regime = 'RECOMPUTE_at_all_bandwidths'
    return dict(analytical_break_even_bandwidth_mbps=None, break_even_regime=regime)


def recompose(strict_comparison, replacement_ms, implementation, *, bandwidth_mbps=None):
    if implementation not in MODES:
        raise ValueError('Unknown impact implementation')
    nonnegative(replacement_ms, 'replacement semantic impact')
    source = strict_comparison['row']
    if source['result_label'] != 'STRICT_BASE_ONLY_SYSTEM_MODEL' or not source['correctness_gate_passed']:
        raise ValueError('Diagnostic requires correctness-valid strict base-only composition')
    for key in ('es_compute_ms', 'ud_lora_ms', 'semcache_es_ms', 'semcache_ud_ms',
                'semcache_control_ms', 'attention_impact_ms', 'network_ms', 'semcache_network_ms',
                'edge_total_network_bytes', 'semcache_total_network_bytes', 'communication_saved_bytes'):
        nonnegative(source[key], key)
    if source['bandwidth_mbps'] <= 0:
        raise ValueError('Original bandwidth must be positive')
    for latency, count in (('network_ms', 'edge_total_network_bytes'),
                           ('semcache_network_ms', 'semcache_total_network_bytes')):
        if not math.isclose(source[latency], source[count]*.008/source['bandwidth_mbps'], abs_tol=1e-9):
            raise ValueError('Original network model is not the serialized tensor byte model')
    expected_edge = source['es_compute_ms'] + source['ud_lora_ms'] + source['network_ms']
    expected_reuse = source['semcache_control_ms'] + source['semcache_es_ms'] + source['semcache_ud_ms'] + source['semcache_network_ms']
    if (not math.isclose(expected_edge, source['edge_lora_total_ms'], abs_tol=1e-9)
            or not math.isclose(expected_reuse, source['semcache_total_ms'], abs_tol=1e-9)
            or source['edge_total_network_bytes']-source['semcache_total_network_bytes'] != source['communication_saved_bytes']):
        raise ValueError('Original system decomposition/byte accounting is inconsistent')
    old = source['attention_impact_ms']
    control = source['semcache_control_ms'] - old + replacement_ms
    if control < 0:
        raise ValueError('Impact exceeds original control scope')
    bandwidth = source['bandwidth_mbps'] if bandwidth_mbps is None else bandwidth_mbps
    nonnegative(bandwidth, 'bandwidth_mbps')
    if bandwidth == 0:
        raise ValueError('Bandwidth must be positive')
    # Preserve the original composition's explicit tensor/wire choices.
    edge_network = source['edge_total_network_bytes']*.008/bandwidth
    reuse_network = source['semcache_total_network_bytes']*.008/bandwidth
    edge = source['es_compute_ms'] + source['ud_lora_ms'] + edge_network
    semcache = control + source['semcache_es_ms'] + source['semcache_ud_ms'] + reuse_network
    compute_delta = source['es_compute_ms'] + source['ud_lora_ms'] - control - source['semcache_es_ms'] - source['semcache_ud_ms']
    return dict(implementation=implementation, bandwidth_mbps=bandwidth,
        source_repeat_index=source['repeat_index'], original_impact_ms=old, replacement_impact_ms=replacement_ms,
        original_semcache_total_ms=source['semcache_total_ms'], original_bandwidth_mbps=source['bandwidth_mbps'],
        edge_lora_total_ms=edge, edge_network_ms=edge_network, semcache_network_ms=reuse_network,
        semcache_total_ms=semcache, system_delta_ms=edge-semcache,
        semcache_control_ms=control, compute_only_delta_ms=compute_delta,
        saved_network_bytes=source['communication_saved_bytes'],
        **break_even(source['communication_saved_bytes'], compute_delta),
        selected_action='REUSE' if semcache < edge else 'RECOMPUTE',
        result_label='RESEARCH_EXTENSION_DIAGNOSTIC', provenance='SIMULATED_RESEARCH_EXTENSION',
        replacement_provenance='MEASURED' if implementation == 'CURRENT_BLOCKWISE' else 'MEASURED_RESEARCH_EXTENSION',
        es_provenance='MEASURED', ud_provenance='CALIBRATED', network_provenance='SIMULATED',
        original_control_provenance='MEASURED', diagnostic_control_provenance='SIMULATED_RESEARCH_EXTENSION',
        original_result_modified=False, decision_applied_to_inference=False,
        replacement_timing_source='mean uninstrumented_impact_total_ms; attribution pass excluded',
        scope='only semantic-impact component replaced; other measured/calibrated/simulated terms unchanged')
