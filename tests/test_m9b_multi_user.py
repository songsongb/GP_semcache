import json
import unittest
import tempfile
from pathlib import Path
from semcache.simulation.multi_user import (
    CAPACITY, PROVENANCE, aggregate, digest, execute_reuse, read_workload,
    run_matrix, safety_eligible, simulate, fixture_costs,
)


def workload():
    return [dict(source_id=str(i), token_ids=[1,2,3,4,i%3], cluster_id=0) for i in range(120)]


def test_assignment_order_corpus():
    rows = workload()
    before = digest(rows)
    a, b = simulate(rows, 10), simulate(rows, 10)
    assert a == b
    c = simulate(rows, 50)
    assert a['query_order_hash'] == c['query_order_hash'] == before
    assert a['user_assignment_hash'] != c['user_assignment_hash']
    assert [r['source_id'] for r in a['trace']] == [str(i) for i in range(120)]
    assert digest(rows) == before


def test_distinct_counters_and_communication():
    result = simulate(workload(), 10)
    totals = aggregate(result['trace'])
    assert totals['candidate_hit_count'] > totals['safe_candidate_count'] == totals['cost_effective_reuse_count'] == 0
    assert totals['candidate_potential_saved_bytes'] > 0
    assert totals['reused_token_count'] + totals['fresh_token_count'] == 600
    assert totals['communication_baseline_bytes'] - totals['communication_reuse_bytes'] == totals['communication_saved_bytes'] == 0
    assert aggregate(result['trace']) == totals


def test_gates():
    target = dict(user_id='a', adapter_id='adapter', prompt_hash='prompt')
    source = dict(target, start=0)
    evidence = {('a','adapter','prompt',0)}
    assert safety_eligible(source, target, 0, evidence)
    assert not safety_eligible(source, dict(target, user_id='b'), 0, evidence)
    assert not safety_eligible(source, target, 1, evidence)
    assert not safety_eligible(source, target, 0, set())
    assert not execute_reuse(False, 1, 10)
    assert not execute_reuse(True, 10, 1)
    assert not execute_reuse(True, 1, 1)
    assert not execute_reuse(True, None, 1)
    assert execute_reuse(True, 1, 10)


def test_capacity_and_accounting():
    entry_size = 3*3*2560*32*2
    result = simulate(workload(), 10, capacity=entry_size)
    assert result['cache_peak_bytes'] <= entry_size
    totals = aggregate(result['trace'])
    assert totals['eviction_count'] > 0
    assert (totals['admission_count']-totals['eviction_count'])*entry_size == result['cache_final_bytes']
    tiny = simulate(workload(), 10, capacity=1)
    assert aggregate(tiny['trace'])['rejection_count'] > 0
    assert tiny['cache_final_bytes'] == 0
    assert CAPACITY == 21474836480
    assert simulate(workload(),10)['cache_capacity_bytes'] == CAPACITY


def test_matrix_pairing_per_user_provenance_reproducibility(tmp_path):
    a = run_matrix({'snips':workload(),'multiwoz':workload()}, tmp_path/'a', smoke=True)
    b = run_matrix({'snips':workload(),'multiwoz':workload()}, tmp_path/'b', smoke=True)
    assert len(a) == 36 and a == b
    for dataset in ('snips','multiwoz'):
        for users in (10,25,50):
            group = [r for r in a if r['dataset']==dataset and r['user_count']==users]
            assert len({r['cache_trace_hash'] for r in group}) == 1
            assert len({r['candidate_hit_count'] for r in group}) == 1
            sim = simulate(workload(), users)
            sums = [aggregate([r for r in sim['trace'] if r['user_id']==f'user_{u:03d}']) for u in range(users)]
            for key in ('query_count','lookup_count','candidate_hit_count','safe_candidate_count','communication_saved_bytes'):
                assert sum(x[key] for x in sums) == group[0][key]
    assert all(not r['safe_reuse_claimed'] for r in a)
    assert PROVENANCE['capacity']=='PAPER_REFERENCE'
    assert PROVENANCE['decision']=='RESEARCH_EXTENSION'
    assert PROVENANCE['ud_measurement_label']=='CALIBRATED_ON_SERAPH_CPU'
    assert (tmp_path/'a/summary.csv').read_bytes()==(tmp_path/'b/summary.csv').read_bytes()
    assert len(list((tmp_path/'a').glob('*.csv'))) == 10


def test_workload_refuses_text_and_missing_semantics(tmp_path):
    path = tmp_path/'input.jsonl'
    path.write_text(json.dumps({'query_text':'hello'})+'\n')
    with unittest.TestCase().assertRaises(ValueError):
        read_workload(path, 'snips')
    row = dict(token_ids=[1,2,3],cluster_id=1,model_id='facebook/opt-2.7b',
               tokenizer_id='local-opt',semantic_assignment_source='existing TinyBERT artifact')
    path.write_text(json.dumps(row)+'\n')
    assert read_workload(path,'snips')[0]['token_ids']==[1,2,3]


def test_fixture_recomposition(tmp_path):
    # Synthetic arithmetic test only; never emitted as measured experiment data.
    row = dict(rank=8, es_compute_policy='strict-base-only', es_compute_ms_provenance='MEASURED',
        ud_lora_ms_provenance='CALIBRATED', model='facebook/opt-2.7b', source_prompt_hash='fixture',
        reused_tokens=3, prompt_tokens=32, repeat_index=0,
        result_label='STRICT_BASE_ONLY_SYSTEM_MODEL', correctness_gate_passed=True,
        es_compute_ms=10., ud_lora_ms=5., semcache_es_ms=9., semcache_ud_ms=4.,
        semcache_control_ms=20., attention_impact_ms=19., network_ms=40.,
        semcache_network_ms=20., edge_total_network_bytes=1000000,
        semcache_total_network_bytes=500000, communication_saved_bytes=500000,
        bandwidth_mbps=200, edge_lora_total_ms=55., semcache_total_ms=53.)
    system, impact = tmp_path/'system.json', tmp_path/'impact.json'
    system.write_text(json.dumps({'comparisons':[{'row':row}]}))
    impact.write_text(json.dumps([{'implementation':'TOKEN_PRECOMPUTE',
        'uninstrumented_impact_total_ms':{'mean':2.}}]))
    before = system.read_bytes()
    costs = fixture_costs(system, impact)
    assert len(costs)==6
    assert costs['CURRENT_BASELINE',200]['normalized_fixture_mean_modeled_latency_ms']==53.
    assert costs['TOKEN_PRECOMPUTE_DIAGNOSTIC',200]['normalized_fixture_mean_modeled_latency_ms']==36.
    assert costs['CURRENT_BASELINE',1000]['normalized_fixture_reuse_decision_fraction']==0.
    assert costs['CURRENT_BASELINE',200]['normalized_fixture_reuse_decision_fraction']==1.
    assert system.read_bytes()==before


class ModelFreeTests(unittest.TestCase):
    def test_assignment(self):
        test_assignment_order_corpus()

    def test_counters(self):
        test_distinct_counters_and_communication()

    def test_safety_and_cost(self):
        test_gates()

    def test_capacity(self):
        test_capacity_and_accounting()

    def test_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            test_matrix_pairing_per_user_provenance_reproducibility(Path(directory))

    def test_cost_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            test_fixture_recomposition(Path(directory))

    def test_input(self):
        with tempfile.TemporaryDirectory() as directory:
            test_workload_refuses_text_and_missing_semantics(Path(directory))


if __name__ == '__main__':
    unittest.main()
