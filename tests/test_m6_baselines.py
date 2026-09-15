from copy import deepcopy
from pathlib import Path
import ast
import pytest
from semcache.experiments.baselines import BaselineKind, FrequencyLRUCache
from semcache.experiments.baseline_runner import run_baseline, run_comparison
from semcache.experiments.config import load_paper_config, load_model_spec
from semcache.experiments.workload import build_workload
from semcache.experiments.runner import make_logical_engine, run_workload
from semcache.experiments.paper_reference import TABLE_II

ROOT = Path(__file__).resolve().parents[1]


def setup(texts=None):
    c = load_paper_config(ROOT/'configs/paper/snips.yaml')
    c['system']['qkv_precision_bits'] = 16
    rows,m = build_workload('snips',[dict(id=str(i),utterance=t,intent='PlayMusic') for i,t in enumerate(texts or ['play some music please']*8)],{'source_split':'train'})
    return rows,m,c,load_model_spec(c,ROOT/'configs/paper')


def test_interface_fairness_and_provenance(monkeypatch):
    rows,m,c,model = setup()
    results = run_comparison(rows,c,model,workload_manifest=m,max_queries=5)
    assert all(set(r) == set(results[0]) for r in results)
    assert all(r['fairness'] == results[0]['fairness'] for r in results)
    assert all(r['query_count'] == 5 and r['analytical_latency_s'] is None and r['system_memory_bytes'] is None for r in results)
    assert results[2]['fbc_metadata']['provenance'] == 'REPRODUCTION_CHOICE'
    assert all(r['baseline_semantics_provenance'] == 'PAPER_DEFINED' for r in results)
    assert all(v['metric_source'] == 'SIMULATED' for r in results for v in r['metrics'].values())
    def forbidden(*a,**kw):
        raise AssertionError('Semantic engine constructed')
    monkeypatch.setattr('semcache.experiments.baseline_runner.make_logical_engine',forbidden)
    for b in (BaselineKind.UD_ONLY,BaselineKind.ES_ONLY):
        out = run_baseline(rows,c,model,workload_manifest=m,baseline=b,max_queries=8)
        assert out['reused_token_count'] == out['base_flops_saved'] == out['communication_elements_total'] == out['communication_elements_saved'] == 0
        assert out['admission_count'] is out['eviction_count'] is out['block_lookup_count'] is None
        assert out['base_flops_total'] > 0 and out['lora_flops_total'] > 0
        assert ('UD:' if b == BaselineKind.UD_ONLY else 'ES:') in out['compute_placement']


def test_fbc_frequency_lru_and_overflow():
    cache = FrequencyLRUCache(2,1,1)
    cache.query('a b')
    cache.query('a')
    cache.query('c')
    assert list(cache.entries) == [('a',),('c',)]
    assert cache.frequencies[('a',)] == 2
    assert cache.query('b')['eviction_count'] == 1
    assert cache.frequencies[('b',)] == 2
    assert FrequencyLRUCache(1,3,2).query('a b c')['admission_count'] == 0
    assert FrequencyLRUCache(2,3,1).query('a b')['block_lookup_count'] == 0


def test_fbc_ignores_semantics_and_users_and_is_deterministic():
    rows,m,c,model = setup()
    a = run_baseline(rows,c,model,workload_manifest=m,baseline='FBC',max_queries=8)
    assert a['query_results'][0]['user_id'] != a['query_results'][1]['user_id']
    assert a['query_results'][1]['summary']['block_hit_count'] > 0
    assert a == run_baseline(rows,c,model,workload_manifest=m,baseline='FBC',max_queries=8)
    c['cluster_count'] = 1
    c['semantic_impact']['rho'] = 0
    c['admission']['threshold'] = 1
    b = run_baseline(rows,c,model,workload_manifest=m,baseline='FBC',max_queries=8)
    assert a['metrics'] == b['metrics']
    # Labels in workload metadata never enter FBC lookup.
    from semcache.experiments.workload import serialize
    from semcache.experiments.manifest import sha256
    for i,row in enumerate(rows):
        row['cluster_id'] = i
    m['sha256'] = sha256(serialize(rows))
    assert a['metrics'] == run_baseline(rows,c,model,workload_manifest=m,baseline='FBC',max_queries=8)['metrics']


def test_semcache_m6a_regression(tmp_path):
    rows,m,c,model = setup()
    old,_ = run_workload(rows,c,model,engine=make_logical_engine(rows,c),workload_manifest=m,
        run_id='regression',output_root=tmp_path,max_queries=8)
    new = run_baseline(rows,c,model,workload_manifest=m,baseline='SEMCACHE',max_queries=8)
    for k in ('block_lookup_count','block_hit_count','reused_token_count','admission_candidate_count','admission_count','eviction_count','base_flops_saved','lora_flops_saved'):
        assert new[k] == old['sum_'+k]['value']
    assert new['block_hit_ratio'] == old['block_hit_ratio']['value']


def test_edge_cases_and_cost_scope():
    rows,m,c,model = setup(['a b','c d e','f g h'])
    out = run_baseline(rows,c,model,workload_manifest=m,baseline='FBC',max_queries=3)
    assert out['block_hit_count'] == out['reused_token_count'] == 0
    assert out['query_results'][0]['summary']['block_lookup_count'] == 0
    c['system'].update(es_tflops=1,ud_tflops=.1,communication_element_bytes=2)
    rows,m,_,_ = setup()
    out = run_baseline(rows,c,model,workload_manifest=m,baseline='FBC',max_queries=8)
    assert out['analytical_latency_s'] > 0
    assert out['base_flops_saved'] == 6*out['reused_token_count']*model['hidden_size']**2*model['layers']
    for bad in (0,-1,True):
        with pytest.raises(ValueError):
            run_baseline(rows,c,model,workload_manifest=m,baseline='FBC',max_queries=bad)
    with pytest.raises(ValueError):
        run_comparison(rows[::-1],c,model,workload_manifest=m,max_queries=8)


def test_reference_isolation():
    assert TABLE_II['GPT2']['FBC']['metric_source'] == 'PAPER_REFERENCE'
    for name in ('baselines','baseline_runner'):
        tree = ast.parse((ROOT/f'src/semcache/experiments/{name}.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert 'paper_reference' not in (node.module or '')
