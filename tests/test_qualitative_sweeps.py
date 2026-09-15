from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
from test_m6_baselines import setup
from semcache.experiments.baseline_runner import run_baseline, SUMMARY_FIELDS
from semcache.experiments.sweeps import run_sweep, sweep_override, TARGETS


def run(sweep='cache_size', values=(5, 20), compact=True, config=None):
    rows, manifest, c, model = setup()
    return run_sweep(rows, config or c, model, workload_manifest=manifest,
                     sweep=sweep, values=values, max_queries=8, seed=42, compact=compact)


@pytest.mark.parametrize('sweep,value', [('cache_size', 5), ('admission_threshold', .1), ('bandwidth', 1000)])
def test_only_target_changes(sweep, value):
    _, _, c, _ = setup()
    original = deepcopy(c)
    updated = sweep_override(c, sweep, value)
    before, after = original, updated
    for key in TARGETS[sweep][:-1]:
        before, after = before[key], after[key]
    key = TARGETS[sweep][-1]
    assert after[key] == value
    after[key] = before[key]
    assert updated == c == original


@pytest.mark.parametrize('sweep,values', [('cache_size', [5, 10, 15, 20]), ('admission_threshold', [.1, .2, .3, .4, .5]), ('bandwidth', [200, 500, 1000])])
def test_determinism_fairness_compact_provenance(sweep, values):
    a = run(sweep, values)
    assert a == run(sweep, values)
    assert len({p['executed_workload_sha256'] for p in a['points']}) == 1
    assert all(k not in json.dumps(a) for k in ('query_results', 'query_ids', 'user_ids'))
    assert a['safe_reuse_claimed'] is False
    assert a['semantic_encoder_actual_tinybert'] is a['attention_impact_available'] is a['chu_pbr_active'] is False
    assert all(p['provenance']['metric_source'] == 'SIMULATED' and not p['safe_reuse_claimed'] for p in a['points'])
    assert all(p['total_latency_s'] is None and not p['latency_available'] for p in a['points'])
    debug = run(sweep, values, compact=False)
    for compact_point, debug_point in zip(a['points'], debug['points']):
        assert len(debug_point.pop('query_results')) == 8
        assert compact_point == debug_point


@pytest.mark.parametrize('sweep,value', [('cache_size', 20), ('admission_threshold', .3)])
def test_canonical_equivalence(sweep, value):
    rows, manifest, c, model = setup()
    baseline = run_baseline(rows, c, model, workload_manifest=manifest, baseline='SEMCACHE', max_queries=8, seed=42)
    point = run(sweep, [value])['points'][0]
    assert all(point[k] == baseline[k] for k in SUMMARY_FIELDS)
    assert point['peak_cache_bytes'] == baseline['peak_logical_cache_bytes'] >= point['final_cache_bytes']
    assert point['provenance']['parameter_source'] == 'PAPER_DEFINED'


def test_bandwidth_single_simulation_and_conversion(monkeypatch):
    import semcache.experiments.sweeps as sweeps
    original = sweeps.run_baseline
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(sweeps, 'run_baseline', counted)
    _, _, c, _ = setup()
    c['system']['communication_element_bytes'] = 2
    result = run('bandwidth', [200, 500, 1000], config=c)
    assert len(calls) == result['cache_simulation_count'] == 1
    points = result['points']
    for key in ('block_hit_count', 'reused_token_count', 'admission_count', 'eviction_count', 'communication_elements'):
        assert len({p[key] for p in points}) == 1
    for p in points:
        assert p['communication_time_s'] == pytest.approx(p['communication_elements'] * 2 * 8 / (p['bandwidth_mbps'] * 1e6))
        assert p['total_latency_s'] is None
    assert points[0]['communication_time_s'] == pytest.approx(5 * points[-1]['communication_time_s'])
    assert run('bandwidth', [200])['points'][0]['communication_time_s'] is None


@pytest.mark.parametrize('sweep,value', [('cache_size', 0), ('bandwidth', -1), ('admission_threshold', 1.1), ('cache_size', float('nan'))])
def test_invalid_values(sweep, value):
    with pytest.raises(ValueError):
        run(sweep, [value])


def test_plots(tmp_path):
    pytest.importorskip('matplotlib')
    path = Path(__file__).resolve().parents[1] / 'scripts/26_plot_semcache_sweeps.py'
    spec = importlib.util.spec_from_file_location('sweep_plot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for sweep, values in [('cache_size', [5, 20]), ('admission_threshold', [.1, .3]), ('bandwidth', [200, 1000])]:
        output = tmp_path / f'{sweep}.png'
        module.plot_sweep(run(sweep, values), output)
        assert output.read_bytes().startswith(b'\x89PNG')


def test_peak_with_eviction():
    from semcache.cache.global_cache import GlobalCache
    from semcache.cache.cache_entry import CacheEntry
    cache = GlobalCache(10)
    for i in range(4):
        cache.insert(CacheEntry(0, (i,), (0, 1), 6))
        assert cache.peak_logical_cache_bytes == 6
        assert cache.logical_cache_bytes == 6
