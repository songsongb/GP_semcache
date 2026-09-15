"""Qualitative sensitivity analysis of the existing logical SemCache runner."""
from copy import deepcopy
from math import isfinite

from .baseline_runner import run_baseline, SUMMARY_FIELDS
from .manifest import canonical, sha256
from .provenance import configuration_provenance
from semcache.simulation.cost_model import communication_seconds

TARGETS = {'cache_size': ('logical_cache_capacity_gb',),
           'admission_threshold': ('admission', 'threshold'),
           'bandwidth': ('system', 'bandwidth_mbps')}


def sweep_override(config, sweep, value):
    path = TARGETS[sweep]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError('Sweep values must be finite numbers')
    if (sweep == 'admission_threshold' and not 0 <= value <= 1) or (sweep != 'admission_threshold' and value <= 0):
        raise ValueError('Invalid sweep value')
    c = deepcopy(config)
    target = c
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return c


def communication_summary(result, system):
    elements = result['communication_elements_total'] - result['communication_elements_saved']
    size = system['communication_element_bytes']
    return dict(communication_elements=elements,
                communication_bytes=None if size is None else elements * size,
                communication_time_s=None if size is None else communication_seconds(elements, size, system['bandwidth_mbps']),
                communication_time_available=size is not None,
                communication_time_unavailable_reason='communication_element_bytes not supplied' if size is None else None,
                timing_provenance='REPRODUCTION_CHOICE', timing_method='ANALYTICAL',
                total_latency_s=None, latency_available=False,
                latency_unavailable_reason='Sweep reports communication only; total Eq.19 latency is not evaluated. Missing inputs: ' + ', '.join(k for k in ('es_tflops', 'ud_tflops', 'communication_element_bytes') if system[k] is None))


def run_sweep(rows, config, model_spec, *, workload_manifest, sweep, values,
              max_queries, seed=None, compact=False):
    values = list(values)
    if not values:
        raise ValueError('At least one sweep value required')
    base = deepcopy(config)
    if seed is not None:
        base['seed'] = seed
    # Match the existing baseline CLI's explicit logical QKV storage choice.
    base['system']['qkv_precision_bits'] = base['system']['qkv_precision_bits'] or 16
    if sweep == 'admission_threshold' and any(base['admission'][k] != v for k, v in dict(alpha=.5, beta=.3, delta=.2).items()):
        raise ValueError('Threshold sweeps require alpha=.5, beta=.3, delta=.2')
    configs = [sweep_override(base, sweep, value) for value in values]
    canonical_result = None
    points = []
    for value, c in zip(values, configs):
        if canonical_result is None or sweep != 'bandwidth':
            result = run_baseline(rows, c, model_spec, workload_manifest=workload_manifest,
                                  baseline='SEMCACHE', max_queries=max_queries,
                                  seed=base['seed'], run_id='qualitative_sweep', compact=compact)
            canonical_result = result
        else:
            result = canonical_result
        provenance = configuration_provenance(c)
        point = {k: result[k] for k in SUMMARY_FIELDS}
        point.update(value=value, dataset=result['dataset'], cache_capacity_gb=c['logical_cache_capacity_gb'],
                     admission_threshold=c['admission']['threshold'], bandwidth_mbps=c['system']['bandwidth_mbps'],
                     query_token_count=result['query_token_count'], final_cache_bytes=result['logical_cache_bytes'],
                     peak_cache_bytes=result['peak_logical_cache_bytes'],
                     communication_elements_total=result['communication_elements_total'],
                     executed_workload_sha256=result['fairness']['executed_workload_sha256'],
                     seed=c['seed'], subsequence_window=c['subsequence_window'], cluster_count=c['cluster_count'],
                     config_sha256=sha256(canonical(c).encode()),
                     provenance=dict(metric_source='SIMULATED', execution_mode='ANALYTICAL_SIMULATION',
                                     sweep_design='REPRODUCTION_CHOICE', parameter_source=provenance['.'.join(TARGETS[sweep])]),
                     safe_reuse_claimed=False, **communication_summary(result, c['system']))
        if not compact:
            point['query_results'] = result['query_results']
        points.append(point)
    hashes = {p['executed_workload_sha256'] for p in points}
    if len(hashes) != 1:
        raise AssertionError('Sweep workload fairness mismatch')
    fairness = {k: v for k, v in canonical_result['fairness'].items()
                if k not in ('query_ids', 'user_ids', 'cache_capacity_bytes', 'bandwidth_mbps', 'system_spec')}
    return dict(schema_version='semcache.qualitative_sweep.v1', experiment='qualitative sensitivity analysis',
                sweep=sweep, compact=compact, cache_simulation_count=1 if sweep == 'bandwidth' else len(points),
                config=base, configuration_provenance=configuration_provenance(base),
                sweep_parameter='.'.join(TARGETS[sweep]), fairness=fairness,
                reproduction_choices=canonical_result['reproduction_choices'],
                semantic_encoder_actual_tinybert=False, attention_impact_available=False,
                chu_pbr_active=False, safe_reuse_claimed=False, metric_source='SIMULATED',
                execution_mode='ANALYTICAL_SIMULATION', points=points)
