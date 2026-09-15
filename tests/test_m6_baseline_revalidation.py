"""Frequency sensitivity and immutable workload provenance revalidation."""
from copy import deepcopy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from semcache.experiments.baselines import FrequencyLRUCache, ALL_BASELINES
from semcache.experiments.baseline_runner import run_baseline, run_comparison, comparison_summary, SUMMARY_FIELDS
from semcache.experiments.config import load_paper_config, load_model_spec
from semcache.experiments.provenance import configuration_provenance
from semcache.experiments.workload import build_workload, save_workload, read_workload, serialize
from semcache.experiments.manifest import sha256

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    c = load_paper_config(ROOT/'configs/paper/snips.yaml')
    c['system']['qkv_precision_bits'] = 16
    rows,m = build_workload('snips',[
        dict(id=str(i),utterance='one two three',intent='PlayMusic') for i in range(6)
    ], {'source_split':'train'})
    return rows,m,c,load_model_spec(c,ROOT/'configs/paper')


def test_v2_first_second_third_and_recency():
    cache = FrequencyLRUCache(2,3,1,frequency_threshold=2)
    key = ('one','two','three')
    first = cache.query('one two three')
    assert cache.frequencies[key] == 1 and key not in cache.entries
    assert first['block_hit_count'] == first['admission_count'] == 0
    assert first['admission_candidate_count'] == 1
    second = cache.query('one two three')
    assert cache.frequencies[key] == 2 and key in cache.entries
    assert second['block_hit_count'] == 0 and second['admission_count'] == 1
    cache.query('four five six')
    cache.query('four five six')
    third = cache.query('one two three')
    assert cache.frequencies[key] == 3
    assert third['block_hit_count'] == 1 and third['reused_token_count'] == 3
    assert third['admission_candidate_count'] == 0
    assert list(cache.entries)[-1] == key


def test_v2_lru_readmission_and_capacity():
    cache = FrequencyLRUCache(2,1,1,frequency_threshold=2)
    cache.query('a b')
    cache.query('a b')
    cache.query('a')
    cache.query('c')
    assert cache.query('c')['eviction_count'] == 1
    assert list(cache.entries) == [('a',),('c',)]
    out = cache.query('b')
    assert cache.frequencies[('b',)] == 3
    assert out['block_hit_count'] == 0 and out['admission_count'] == out['eviction_count'] == 1
    assert list(cache.entries) == [('c',),('b',)]
    assert cache.query('b')['block_hit_count'] == 1
    oversized = FrequencyLRUCache(1,3,2,frequency_threshold=2)
    oversized.query('a b c')
    assert oversized.query('a b c')['admission_count'] == 0
    assert not oversized.entries


def test_v2_repeated_windows_observed_before_batch_admission():
    cache = FrequencyLRUCache(10,3,1,frequency_threshold=2)
    out = cache.query('a a a a a')
    assert cache.frequencies[('a','a','a')] == 3
    assert out['block_hit_count'] == 0  # Even third occurrence in same query misses.
    assert out['admission_candidate_count'] == out['admission_count'] == 1
    assert cache.query('a a a')['block_hit_count'] == 1


def test_v2_cross_user_semantic_independence_and_determinism(monkeypatch):
    rows,m,c,model = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('FBC must not construct semantic engine')
    monkeypatch.setattr('semcache.experiments.baseline_runner.make_logical_engine', forbidden)
    args = dict(workload_manifest=m,baseline='FBC_V2',max_queries=6)
    out = run_baseline(rows,c,model,**args)
    assert len(set(out['fairness']['user_ids'][:3])) == 3
    assert [q['summary']['block_hit_count'] for q in out['query_results'][:3]] == [0,0,1]
    assert out == run_baseline(rows,c,model,**args)
    altered = deepcopy(c)
    altered['cluster_count'] = 1
    altered['semantic_impact']['rho'] = 0
    altered['admission']['threshold'] = 1
    altered['eviction'].update(alpha=0,beta=0,gamma=0,delta=1)
    for i,row in enumerate(rows):
        row['cluster_id'] = i
    m['sha256'] = sha256(serialize(rows))
    changed = run_baseline(rows,altered,model,**args)
    assert out['metrics'] == changed['metrics']
    assert out['query_results'] == changed['query_results']
    meta = out['fbc_metadata']
    assert meta['fbc_variant'] == 'exact_block_frequency2_lru_v2'
    assert meta['frequency_threshold'] == 2 and meta['semantic_awareness'] is False
    assert meta['provenance'] == 'REPRODUCTION_CHOICE'


def test_five_way_fairness_summary_and_v1_compatibility():
    rows,m,c,model = fixture()
    before = serialize(rows)
    results = run_comparison(rows,c,model,workload_manifest=m,max_queries=5)
    assert [r['baseline'] for r in results] == ['UD_ONLY','ES_ONLY','FBC_V1','FBC_V2','SEMCACHE']
    assert all(set(r) == set(results[0]) for r in results)
    assert all(r['fairness'] == results[0]['fairness'] for r in results)
    assert results[0]['fairness']['executed_workload_sha256'] == sha256(serialize(rows[:5]))
    assert serialize(rows) == before
    old = run_baseline(rows,c,model,workload_manifest=m,baseline='FBC',max_queries=5)
    assert old['metrics'] == results[2]['metrics']
    assert old['query_results'] == results[2]['query_results']
    assert results[2]['fbc_metadata']['frequency_tracked_but_not_policy_driving'] is True
    assert results[3]['admission_rate'] < results[2]['admission_rate']
    summary = comparison_summary(results)
    assert summary['metric_source'] == 'SIMULATED'
    assert all(set(r) == set(SUMMARY_FIELDS) for r in summary['rows'])
    assert summary['rows'][0]['admission_count'] is None
    assert summary['rows'][0]['token_reuse_ratio'] == 0


@pytest.mark.parametrize('dataset,rule',[
    ('multiwoz','seeded_group_balanced'),('coqa','seeded_group_balanced'),('snips','seeded_round_robin')])
def test_assignment_contract(dataset,rule):
    c = load_paper_config(ROOT/f'configs/paper/{dataset}.yaml')
    assert c['user_assignment']['mode'] == rule
    assert configuration_provenance(c)['user_assignment.mode'] == 'REPRODUCTION_CHOICE'
    # A minimal prepared manifest carries the fixed-workload assignment contract.
    if dataset == 'coqa':
        examples = [dict(id='conversation',story='Story',questions=[dict(turn_id=i,input_text='Where is it?') for i in (1,2)],answers=[])]
        rows,m = build_workload(dataset,examples,{'source_split':'train'},assignment=rule)
        assert m['user_assignment_rule'] == c['user_assignment']['mode']
        assert len({r['user_id'] for r in rows}) == 1
        assert m['provenance']['user_assignment_rule'] == 'REPRODUCTION_CHOICE'
    else:
        rows,m,_,_ = fixture()
        for row in rows:
            row['dataset'] = dataset
        m.update(user_assignment_rule=rule,sha256=sha256(serialize(rows)))
    model = load_model_spec(c,ROOT/'configs/paper')
    before = deepcopy((rows,m))
    run_baseline(rows,c,model,workload_manifest=m,baseline='UD_ONLY',max_queries=2)
    assert (rows,m) == before
    c['user_assignment']['mode'] = 'deterministic_hash'
    with pytest.raises(ValueError,match='user assignment rule mismatch'):
        run_baseline(rows,c,model,workload_manifest=m,baseline='UD_ONLY',max_queries=2)


@pytest.mark.parametrize('dataset',['multiwoz','coqa','snips'])
def test_local_fixed_manifest_assignment_when_available(dataset):
    path = ROOT/f'results/workloads/{dataset}.jsonl'
    if not path.exists():
        pytest.skip('Fixed workload artifact is available on SERAPH, not this checkout')
    # Read-only hash validation; never reconstruct or run the full workload.
    _,manifest = read_workload(path)
    c = load_paper_config(ROOT/f'configs/paper/{dataset}.yaml')
    assert manifest['user_assignment_rule'] == c['user_assignment']['mode']
    assert manifest['provenance']['user_assignment_rule'] == 'REPRODUCTION_CHOICE'


def test_cli_all_variants_and_legacy(tmp_path):
    rows,m,_,_ = fixture()
    workload = tmp_path/'workload.jsonl'
    save_workload(workload,rows,m)
    for selection,expected in [(['--all-baselines'],[b.value for b in ALL_BASELINES]),
                                (['--baseline','FBC'],['FBC']),(['--baseline','FBC_V2'],['FBC_V2'])]:
        output = tmp_path/'comparison.json'
        subprocess.run([sys.executable,str(ROOT/'scripts/24_run_baseline_comparison.py'),
            '--workload',str(workload),'--config',str(ROOT/'configs/paper/snips.yaml'),
            '--max-queries','5','--seed','42','--output',str(output),*selection],check=True,capture_output=True,text=True)
        data = json.loads(output.read_text())
        assert [r['baseline'] for r in data['results']] == expected
        assert [r['baseline'] for r in data['summary']['rows']] == expected
