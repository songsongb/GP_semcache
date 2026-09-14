"""Aggregate typed metrics only; never turn unavailable values into zero."""
from .provenance import metric
from .audit import percentile


def aggregate(rows):
    result = {}
    names = set().union(*(r.keys() for r in rows)) if rows else set()
    for name in sorted(names):
        items = [r[name] for r in rows if name in r]
        signatures = {(v['metric_source'], v['metric_scope'], v['unit']) for v in items}
        if len(signatures) != 1:
            raise ValueError(f'Cannot mix metric provenance/scope/units: {name}')
        source, scope, unit = signatures.pop()
        values = [v['value'] for v in items if v['value'] is not None]
        # Counters are additive; occupancy, latency and quality are observations.
        operation = 'sum' if name.endswith(('_count', '_saved')) else 'avg'
        value = (sum(values) if operation == 'sum' else sum(values)/len(values)) if values else None
        result[f'{operation}_{name}'] = metric(value, source, scope, unit)
        if 'latency' in name:
            for p in (.5, .95):
                result[f'p{int(p*100)}_{name}'] = metric(percentile(values,p), source, scope, unit)
    # Ratios use pooled raw numerators/denominators, not mean-of-query ratios.
    for name, num, den in [('block_hit_ratio','block_hit_count','block_lookup_count'),
                            ('token_reuse_ratio','reused_token_count','query_token_count'),
                            ('admission_rate','admission_count','admission_candidate_count')]:
        if all(f'sum_{key}' in result for key in (num, den)):
            a,b = result[f'sum_{num}'], result[f'sum_{den}']
            if a['metric_source'] != b['metric_source']:
                raise ValueError('Mixed ratio provenance')
            result[name] = metric(a['value']/b['value'] if b['value'] else 0, a['metric_source'], 'pooled workload counts', 'ratio')
    return result


def compare(ours, reference, *, our_context, paper_context, approximate_reason=None):
    required = ('model', 'precision', 'dataset', 'transformation', 'hardware', 'execution_mode', 'execution_scope', 'metric_scope', 'unit')
    missing = any(our_context.get(k) is None or paper_context.get(k) is None for k in required)
    equal = all(our_context.get(k) == paper_context.get(k) for k in required)
    hard = ('model', 'precision', 'execution_scope', 'metric_scope', 'unit')
    incompatible = any(our_context.get(k) != paper_context.get(k) for k in hard)
    status = 'DIRECT' if equal and not missing else ('APPROXIMATE' if approximate_reason and not incompatible and not missing else 'NOT_COMPARABLE')
    delta = ours-reference if status != 'NOT_COMPARABLE' and ours is not None and reference is not None else None
    return dict(our_result=ours, paper_reference=reference, comparability=status,
                absolute_difference=abs(delta) if delta is not None else None,
                relative_difference=delta/reference if delta is not None and reference else None,
                reason=approximate_reason)
