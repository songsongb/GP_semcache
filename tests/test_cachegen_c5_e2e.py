"""CPU-only C5 planning and paired logical invariants; never loads OPT."""
import argparse
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

from semcache.experiments.cachegen import c5_e2e as c5
from semcache.experiments.m9b_semantic_workload import assignment_source
from semcache.semantic.intent_clusterer import IntentClusterer
from semcache.simulation.multi_user import digest


def _rows(dataset, *, count=2):
    c = 30 if dataset == 'snips' else 20
    rows = []
    for i in range(c+2*count):
        if i < c:
            ids, cluster = [2, 1000+i, 2000+i, 3000+i, 4000+i, 5000+i, 6000+i], 0
        else:
            pair = (i-c)//2
            ids, cluster = ([2, 100+pair, 200+pair, 11+pair, 21+pair, 31+pair, 41+pair]
                            if (i-c)%2 == 0 else
                            [2, 300+pair, 400+pair, 11+pair, 21+pair, 31+pair, 51+pair]), 1
        source_id = (f'AddToPlaylist:{i}' if dataset == 'snips' else f'dialogue{i//2}:{i}')
        rows.append(dict(dataset=dataset, source_id=source_id,
            conversation_id=None if dataset == 'snips' else f'dialogue{i//2}',
            query_text=f'prepared query {i}', token_ids=ids, cluster_id=cluster,
            model_id=c5.MODEL_ID, model_revision=c5.MODEL_REVISION,
            tokenizer_id=f'{c5.MODEL_ID}@{c5.MODEL_REVISION}',
            window_size=3, semantic_assignment_source=assignment_source(dataset),
            semantic_embedding_sha256=f'hash{i}'))
    return rows


def test_natural_discovery_is_deterministic_and_exact_w3():
    rows = _rows('snips')
    users = ['user_000' if i%2 == 0 else 'user_001' for i in range(len(rows))]
    a, b = c5.discover(rows, 'snips', users), c5.discover(rows, 'snips', users)
    assert a == b and len(a) >= 2
    for episode in a:
        assert episode['source_index'] < episode['target_index']
        assert episode['source_id'] != episode['target_id']
        assert episode['source_query_ne_target_query']
        assert rows[episode['source_index']]['token_ids'] != rows[episode['target_index']]['token_ids']
        assert episode['semantic_match_evidence']['prepared_cluster_id'] == 1
        assert episode['semantic_candidate_match']
        assert episode['source_cluster_id'] == episode['target_cluster_id'] == 1
        assert episode['physical_safety_contract'] == c5.SAFETY_CONTRACT
        assert episode['m9b_strict_safety']['decision'] == 'REJECTED'
        assert not episode['m9b_strict_safety']['eligible']
        assert episode['nonprefix_reuse']
        assert episode['initial_special_token_id_avoided']
        assert episode['exact_w3_safe'] and episode['expected_natural_hits'] >= 1
        for span in episode['spans']:
            source = rows[episode['source_index']]['token_ids']
            target = rows[episode['target_index']]['token_ids']
            assert source[span['source_start']:span['source_end']] == span['token_ids']
            assert target[span['target_start']:span['target_end']] == span['token_ids']
            assert span['source_start'] >= 3 and span['target_start'] >= 3


def test_prefix_only_and_mixed_prefix_hits_are_excluded():
    rows = _rows('snips', count=1)
    users = ['user_000']*len(rows)
    rows[-2]['token_ids'] = [2, 7, 8, 30, 31, 32, 40]
    rows[-1]['token_ids'] = [2, 7, 8, 50, 51, 52, 60]
    assert c5.discover(rows, 'snips', users) == []
    # A valid later shared span does not permit the engine's simultaneous
    # prefix hit to slip into the selected C5 episode.
    rows[-2]['token_ids'] = [2, 7, 8, 30, 55, 66, 77]
    rows[-1]['token_ids'] = [2, 7, 8, 50, 55, 66, 77]
    assert c5.discover(rows, 'snips', users) == []


def test_nonprefix_selection_prefers_no_initial_token_id():
    candidates = [dict(group='a', cross_user=True, target_index=i,
                       expected_compressed_admissions=3,
                       initial_special_token_id_avoided=avoid)
                  for i, avoid in enumerate((False, True))]
    assert c5.select_diverse(candidates, 1)[0]['initial_special_token_id_avoided']


def test_selection_diagnostics_preserve_choice_and_separate_guards():
    candidates = [dict(episode_id=str(i), group=group, cross_user=False,
        target_index=target, expected_compressed_admissions=cost,
        source_id=f's{i}', target_id=f't{i}',
        spans=[dict(source_start=3, source_end=6, target_start=4,
                    target_end=7, token_ids=[11, 12, 13])])
        for i, (group, cost, target) in enumerate((
            ('a', 4, 10), ('b', 7, 11), ('c', 11, 12),
            ('d', 3, 2048), ('a', 5, 13), ('a', 4, 14)))]
    baseline = c5.select_diverse(candidates, 2, max_encodes=10,
                                 max_semantic_prefix_rows=2048)
    diagnostics = {}
    selected = c5.select_diverse(candidates, 2, max_encodes=10,
                                 max_semantic_prefix_rows=2048,
                                 diagnostics=diagnostics)
    assert selected == baseline
    assert [episode['episode_id'] for episode in selected] == ['0', '5']
    assert diagnostics['natural_episodes_after_nonprefix_filter'] == 6
    assert diagnostics['episodes_passing_prefix_guard'] == 5
    assert diagnostics['episodes_passing_encode_guard'] == 4
    assert diagnostics['final_selectable_episodes'] == 2
    assert diagnostics['rejection_counts'] == dict(
        rejected_by_max_encodes=2,
        rejected_by_max_semantic_prefix_rows=1,
        rejected_by_other_guard=1)
    assert {x['episode']['episode_id']: x['reason'] for x in diagnostics['rejected']} == {
        '1': 'rejected_by_max_encodes', '2': 'rejected_by_max_encodes',
        '3': 'rejected_by_max_semantic_prefix_rows', '4': 'rejected_by_other_guard'}


def test_m9b_strict_gate_is_diagnostic_only():
    source = dict(token_ids=[1, 2, 3, 4], adapter_id='same')
    target = dict(token_ids=[1, 2, 3, 5], adapter_id='same')
    spans = [dict(source_start=0, target_start=0)]
    diagnostic = c5.m9b_strict_diagnostic(source, target, 'user_000', 'user_000', spans)
    assert diagnostic['decision'] == 'REJECTED' and diagnostic['diagnostic_only']
    assert diagnostic['per_span'][0]['same_user']
    assert diagnostic['per_span'][0]['adapter_present_and_equal']
    assert not diagnostic['per_span'][0]['full_prompt_identity']
    assert not diagnostic['per_span'][0]['external_fixture_evidence_present']
    # Even an exact prompt is not M9-B eligible without external fixture evidence.
    same_prompt = c5.m9b_strict_diagnostic(source, source, 'user_000', 'user_000', spans)
    assert same_prompt['decision'] == 'REJECTED'
    assert same_prompt['per_span'][0]['full_prompt_identity']


def test_discovery_rejects_semantic_or_token_mismatch():
    rows = _rows('multiwoz', count=1)
    users = ['user_000']*len(rows)
    assert c5.discover(rows, 'multiwoz', users)
    rows[-1]['cluster_id'] = 2
    assert c5.discover(rows, 'multiwoz', users) == []
    rows[-1]['cluster_id'] = 1
    rows[-1]['token_ids'] = [9, 8, 7, 6]
    assert c5.discover(rows, 'multiwoz', users) == []


def test_selection_guard_and_diversity():
    episodes = [dict(group=group, cross_user=cross, expected_compressed_admissions=size,
                     target_index=i)
                for i, (group, cross, size) in enumerate(
                    [('a',False,5), ('a',True,5), ('b',True,6), ('c',False,20)])]
    selected = c5.select_diverse(episodes, 2, max_encodes=12)
    assert [x['group'] for x in selected] == ['a', 'b']
    assert sum(x['expected_compressed_admissions'] for x in selected) <= 12


def _budget_episode(name, group, cost, index):
    return dict(episode_id=f'{name}:{index}', group=group, cross_user=True,
                target_index=index, expected_compressed_admissions=cost,
                initial_special_token_id_avoided=True)


def test_global_encode_budget_accepts_31_without_dataset_split():
    discovered = dict(
        snips=[_budget_episode('snips', 'a', 9, 0),
               _budget_episode('snips', 'b', 8, 1)],
        multiwoz=[_budget_episode('multiwoz', 'x', 7, 0),
                  _budget_episode('multiwoz', 'y', 7, 1)])
    first, diagnostics = c5.select_balanced_global(discovered, 2, max_encodes=32)
    second, _ = c5.select_balanced_global(discovered, 2, max_encodes=32)
    assert first == second
    assert [e['expected_compressed_admissions'] for e in first['snips']] == [8, 9]
    assert [e['expected_compressed_admissions'] for e in first['multiwoz']] == [7, 7]
    assert sum(e['expected_compressed_admissions'] for episodes in first.values()
               for e in episodes) == 31
    assert sum(e['expected_compressed_admissions'] for e in first['snips']) == 17
    assert diagnostics['snips']['candidates_considered'] == 2


def test_cli_max_encodes_default_remains_32(monkeypatch):
    script = Path(__file__).resolve().parents[1]/'scripts/47_cachegen_c5_e2e.py'
    spec = importlib.util.spec_from_file_location('c5_cli_default_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = []
    monkeypatch.setattr(module, 'run', lambda args: observed.append(args.max_encodes))
    monkeypatch.setattr(sys, 'argv', [str(script)])
    module.main()
    assert observed == [32]
    assert c5.MAX_ENCODES == 32
    assert c5.MAX_ENCODES_LIMIT == 96


def test_known_34_encode_balanced_plan_needs_more_than_32():
    discovered = dict(
        snips=[_budget_episode('snips', 'a', 9, 0),
               _budget_episode('snips', 'b', 11, 1)],
        multiwoz=[_budget_episode('multiwoz', 'x', 7, 0),
                  _budget_episode('multiwoz', 'y', 7, 1)])
    assert c5.minimum_balanced_plan(discovered, 2, max_encodes=36)[
        'minimum_balanced_encode_upper_bound'] == 34
    chosen_32, _ = c5.select_balanced_global(discovered, 2, max_encodes=32)
    chosen_36, _ = c5.select_balanced_global(discovered, 2, max_encodes=36)
    assert chosen_32 is None
    assert [e['episode_id'] for episodes in chosen_36.values() for e in episodes] == [
        'snips:0', 'snips:1', 'multiwoz:0', 'multiwoz:1']
    assert sum(e['expected_compressed_admissions'] for episodes in chosen_36.values()
               for e in episodes) == 34
    with pytest.raises(ValueError, match=r'1\.\.96'):
        c5.select_balanced_global(discovered, 2, max_encodes=97)


def test_known_72_encode_four_plus_four_plan_fits_80_but_not_64():
    # Synthetic episodes reproduce the measured dataset totals, 42 + 30.
    discovered = {name: [_budget_episode(name, str(i), cost, i)
                         for i, cost in enumerate(costs)]
                  for name, costs in (('snips', [9, 11, 11, 11]),
                                      ('multiwoz', [7, 7, 8, 8]))}
    minimum = c5.minimum_balanced_plan(discovered, 4, max_encodes=80)
    assert minimum['minimum_balanced_encode_upper_bound'] == 72
    assert minimum['dataset_minima']['snips']['minimum_valid_n_episode_combined_encode_cost'] == 42
    assert minimum['dataset_minima']['multiwoz']['minimum_valid_n_episode_combined_encode_cost'] == 30
    assert c5.select_balanced_global(discovered, 4, max_encodes=64)[0] is None
    for cap in (80, 96):
        selected, _ = c5.select_balanced_global(discovered, 4, max_encodes=cap)
        assert all(len(episodes) == 4 for episodes in selected.values())
        assert sum(e['expected_compressed_admissions'] for episodes in selected.values()
                   for e in episodes) == 72


@pytest.mark.parametrize('count', [1, 2, 4])
def test_generic_planner_matches_exhaustive_small_reference(count):
    rng = random.Random(42)
    for _ in range(20):
        discovered = {name: [_budget_episode(name, rng.randrange(4), rng.randrange(3, 13), i)
                             for i in range(6)] for name in ('snips', 'multiwoz')}
        for episodes in discovered.values():
            for episode in episodes:
                episode['cross_user'] = bool(rng.randrange(2))
                episode['initial_special_token_id_avoided'] = bool(rng.randrange(2))
        cap = rng.randrange(16, 65)
        ordered = {name: sorted(episodes, key=lambda e: (
            not e['initial_special_token_id_avoided'], e['expected_compressed_admissions'],
            not e['cross_user'], e['target_index'])) for name, episodes in discovered.items()}
        combos = {name: list(combinations(episodes, count)) for name, episodes in ordered.items()}
        cost = lambda episodes: sum(e['expected_compressed_admissions'] for e in episodes)
        expected = None
        for distinct_s, distinct_m in ((True, True), (True, False), (False, True), (False, False)):
            for snips in combos['snips']:
                if distinct_s and len({e['group'] for e in snips}) != count:
                    continue
                for multiwoz in combos['multiwoz']:
                    if distinct_m and len({e['group'] for e in multiwoz}) != count:
                        continue
                    if cost(snips)+cost(multiwoz) <= cap:
                        expected = dict(snips=list(snips), multiwoz=list(multiwoz))
                        break
                if expected is not None:
                    break
            if expected is not None:
                break
        selected, diagnostics = c5.select_balanced_global(discovered, count, max_encodes=cap)
        assert selected == expected
        minimum = c5.minimum_balanced_plan(discovered, count, max_encodes=cap)
        assert minimum['minimum_balanced_encode_upper_bound'] == sum(
            min(cost(combo) for combo in combos[name]) for name in combos)
        for name, episodes in ordered.items():
            other = 'multiwoz' if name == 'snips' else 'snips'
            for episode in episodes:
                can_participate = any(
                    any(e is episode for e in combo) and cost(combo)+cost(peer) <= cap
                    for combo in combos[name] for peer in combos[other])
                assert minimum['participation'][id(episode)] == can_participate
            assert 'rejected_by_other_guard' not in diagnostics[name]['rejection_counts']


@pytest.mark.parametrize('cost,expected_total', [(8, 64), (9, 72)])
def test_four_per_dataset_large_candidate_pools(cost, expected_total):
    discovered = {name: [_budget_episode(name, i % 7, cost, i) for i in range(size)]
                  for name, size in (('snips', 561), ('multiwoz', 721))}
    minimum = c5.minimum_balanced_plan(discovered, 4, max_encodes=64)
    assert minimum['balanced_per_dataset'] == 4
    assert minimum['minimum_balanced_encode_upper_bound'] == expected_total
    assert minimum['minimum_encode_cap_excess'] == max(0, expected_total-64)
    assert len(minimum['minimum_balanced_episode_ids']) == 8
    selected, diagnostics = c5.select_balanced_global(discovered, 4, max_encodes=64)
    if expected_total <= 64:
        assert all(len(episodes) == 4 for episodes in selected.values())
        assert len({e['episode_id'] for episodes in selected.values() for e in episodes}) == 8
    else:
        assert selected is None
        assert all(d['candidates_participating_in_feasible_balanced_selection'] == 0
                   for d in diagnostics.values())


@pytest.mark.parametrize('count', [1, 2, 4])
def test_balanced_shortfall_preserves_prefix_guard_and_unique_episode_count(count):
    discovered = {name: [_budget_episode(name, i, 1, i) for i in range(count)]
                  for name in ('snips', 'multiwoz')}
    discovered['snips'][-1]['target_index'] = 2048
    minimum = c5.minimum_balanced_plan(discovered, count, max_encodes=64)
    selected, diagnostic = c5.select_balanced_global(discovered, count, max_encodes=64)
    assert selected is None
    assert minimum['minimum_balanced_encode_upper_bound'] is None
    assert minimum['minimum_encode_cap_excess'] is None
    assert minimum['minimum_balanced_episode_ids'] == []
    assert minimum['dataset_minima']['snips']['minimum_valid_n_episode_combined_encode_cost'] is None
    assert minimum['dataset_minima']['multiwoz']['minimum_valid_n_episode_combined_encode_cost'] == count
    assert diagnostic['snips']['rejection_counts']['rejected_by_max_semantic_prefix_rows'] == 1
    assert not any(minimum['participation'].values())


def test_global_encode_budget_rejects_33_and_finds_first_feasible_alternative():
    discovered = dict(
        snips=[_budget_episode('snips', 'a', 9, 0),
               _budget_episode('snips', 'b', 9, 1)],
        multiwoz=[_budget_episode('multiwoz', 'x', 8, 0),
                  _budget_episode('multiwoz', 'y', 7, 1)])
    chosen, diagnostics = c5.select_balanced_global(discovered, 2, max_encodes=32)
    assert chosen is None  # 9 + 9 + 8 + 7 = 33
    assert diagnostics['snips']['episodes_passing_encode_guard'] == 2
    assert diagnostics['multiwoz']['episodes_passing_encode_guard'] == 2
    assert diagnostics['snips']['candidates_participating_in_feasible_balanced_selection'] == 0
    assert diagnostics['multiwoz']['candidates_participating_in_feasible_balanced_selection'] == 0
    assert diagnostics['snips']['rejection_counts']['rejected_by_max_encodes'] == 0
    assert diagnostics['multiwoz']['rejection_counts'][
        'cannot_participate_in_balanced_selection_within_encode_cap'] == 2
    discovered['snips'].append(_budget_episode('snips', 'c', 8, 2))
    chosen, _ = c5.select_balanced_global(discovered, 2, max_encodes=32)
    assert [e['episode_id'] for e in chosen['snips']] == ['snips:2', 'snips:0']
    assert sum(e['expected_compressed_admissions'] for episodes in chosen.values()
               for e in episodes) == 32


def test_global_selection_skips_early_over_budget_combinations_deterministically():
    discovered = dict(
        snips=[_budget_episode('snips', 'a', 15, 0),
               _budget_episode('snips', 'b', 9, 1),
               _budget_episode('snips', 'c', 8, 2)],
        multiwoz=[_budget_episode('multiwoz', 'x', 7, 0),
                  _budget_episode('multiwoz', 'y', 7, 1)])
    discovered['snips'][2]['initial_special_token_id_avoided'] = False
    first, _ = c5.select_balanced_global(discovered, 2, max_encodes=32)
    second, _ = c5.select_balanced_global(discovered, 2, max_encodes=32)
    assert first == second
    assert [e['episode_id'] for e in first['snips']] == ['snips:1', 'snips:2']
    assert sum(e['expected_compressed_admissions'] for episodes in first.values()
               for e in episodes) == 31


def test_minimum_balanced_is_exact_deterministic_and_prefix_guarded():
    discovered = dict(
        snips=[_budget_episode('snips', 'a', 9, 0),
               _budget_episode('snips', 'b', 8, 1),
               _budget_episode('snips', 'c', 1, 2048)],
        multiwoz=[_budget_episode('multiwoz', 'x', 7, 0),
                  _budget_episode('multiwoz', 'y', 7, 1)])
    first = c5.minimum_balanced_plan(discovered, 2, max_encodes=32)
    second = c5.minimum_balanced_plan(discovered, 2, max_encodes=32)
    assert first == second
    assert first['minimum_balanced_encode_upper_bound'] == 31
    assert first['minimum_balanced_episode_ids'] == [
        'snips:1', 'snips:0', 'multiwoz:0', 'multiwoz:1']
    assert first['dataset_minima']['snips']['minimum_single_episode_encode_cost'] == 8
    assert first['dataset_minima']['snips']['minimum_valid_n_episode_combined_encode_cost'] == 17
    assert first['dataset_minima']['multiwoz']['minimum_valid_n_episode_ids'] == [
        'multiwoz:0', 'multiwoz:1']
    assert not first['participation'].get(id(discovered['snips'][2]), False)


def test_logical_pair_requires_same_hit_admission_and_key():
    summary = dict(cluster_id=1, block_hit_count=1, executed_nonoverlap_hits=1,
        admitted_block_count=1, rejected_block_count=0, reused_unique_token_count=3)
    events = [dict(event_type='CLUSTER_ASSIGN'),
              dict(event_type='HIT', cache_key=(1,(1,2,3)), target_start=0, target_end=3),
              dict(event_type='INSERT', cache_key=(1,(4,5,6)))]
    raw = dict(summary=summary.copy(), events=events)
    comp = dict(summary=summary.copy(), events=[e.copy() for e in events])
    c5.assert_logical_pair([raw], [comp])
    comp['events'][1]['cache_key'] = (2,(1,2,3))
    with pytest.raises(ValueError, match='semantic/key'):
        c5.assert_logical_pair([raw], [comp])


def test_compressed_resident_cannot_retain_raw_kv():
    cache = SimpleNamespace(entries={(1,(1,2,3)): SimpleNamespace(tensors=None,
                                                      compressed_kv=object())})
    c5.assert_compressed_residency(cache)
    cache.entries[(1,(1,2,3))].tensors = {'raw': object()}
    with pytest.raises(ValueError, match='raw K/V'):
        c5.assert_compressed_residency(cache)


def test_metric_aggregation():
    pair = dict(dataset='snips', natural_hit_count=2, cross_user=True,
        exact_sequence_match=False, prefix_agreement_length=3,
        token_position_agreement=.75, normalized_token_edit_distance=.25)
    physical = dict(dataset='snips', storage_mode='COMPRESSED_SEMCACHE',
                    source_raw_qkv_resident_bytes=100, source_physical_resident_bytes=40)
    result = c5.summarize([pair], [physical])
    assert result['snips'] == result['overall']
    assert result['overall']['natural_hit_count'] == 2
    assert result['overall']['compressed_qkv_bytes'] == 40


def test_episode_reports_physical_and_strict_safety_separately():
    signatures = [dict(cache_key=[1, [1, 2, 3]], token_ids=[1, 2, 3],
        source_start=2, source_end=5, target_start=4, target_end=7)]
    episode = dict(episode_id='snips:1->2', dataset='snips', source_id='s', target_id='t',
        source_user='user_000', target_user='user_001', cross_user=True, cluster_id=1,
        m9b_strict_safety=dict(eligible=False, decision='REJECTED', diagnostic_only=True))
    def observation(mode, size):
        account = dict(raw_qkv_bytes=100, actual_qkv_entry_bytes=size)
        return dict(mode=mode, source=dict(summary=dict(admitted_block_count=1)),
            target=dict(summary=dict(admitted_block_count=0, block_hit_count=1,
                block_lookup_count=2, executed_nonoverlap_hits=1)),
            source_accounting=[account], cache=SimpleNamespace(entries={
                (1,(1,2,3)): SimpleNamespace(physical_tensor_bytes=size)}),
            raw_kv_resident=mode == 'RAW_SEMCACHE', generated_token_ids=[7, 8],
            timings=dict(target_request_ms=1.))
    observations = dict(RAW_SEMCACHE=observation('RAW_SEMCACHE', 100),
                        COMPRESSED_SEMCACHE=observation('COMPRESSED_SEMCACHE', 40))
    rows, pair = c5._episode_rows(episode, observations, signatures)
    assert len(rows) == 2
    assert all(r['physical_safety_contract'] == c5.SAFETY_CONTRACT for r in rows)
    assert pair['source_query_ne_target_query'] and pair['semantic_candidate_match']
    assert json.loads(pair['exact_reused_w3_token_ids']) == [[1, 2, 3]]
    assert json.loads(pair['source_positions']) == [[2, 5]]
    assert json.loads(pair['target_positions']) == [[4, 7]]
    assert pair['m9b_strict_decision'] == 'REJECTED'
    assert pair['m9b_strict_diagnostic_only']


def test_live_semantic_state_replay_checks_prepared_assignment():
    rows = _rows('snips', count=1)
    vectors = {row['query_text']: [float(i), 0.] for i, row in enumerate(rows)}
    clusterer = IntentClusterer(30, initialization='first_k', update_mode='buffered',
                                update_interval=100)
    clusterer.initialize([vectors[row['query_text']] for row in rows[:30]])
    for row in rows:
        vector = vectors[row['query_text']]
        row['semantic_embedding_sha256'] = digest(vector)
        row['cluster_id'] = clusterer.observe(vector)
    class Encoder:
        def encode(self, texts):
            return [vectors[text] for text in texts]
    episode = dict(dataset='snips', source_index=30, target_index=31)
    snapshots = c5.semantic_snapshots(rows, [episode], Encoder(), batch_size=8)
    assert set(snapshots) == {30, 31}
    assert snapshots[30].queries == 30 and snapshots[31].queries == 31
    rows[31]['cluster_id'] = 0
    with pytest.raises(ValueError, match='semantic cluster'):
        c5.semantic_snapshots(rows, [episode], Encoder(), batch_size=8)


def test_dry_run_never_loads_model_or_codec(tmp_path, monkeypatch, capsys):
    paths = {}
    for dataset in ('snips', 'multiwoz'):
        paths[dataset] = tmp_path/f'{dataset}.jsonl'
        paths[dataset].write_text(''.join(json.dumps(row)+'\n' for row in _rows(dataset, count=1)))
    monkeypatch.setattr(c5, 'verify_profile', lambda _: dict(sha256=c5.PROFILE_SHA256))
    monkeypatch.setattr(c5, 'FrozenK20V16Codec', lambda *a, **k: pytest.fail('codec loaded'))
    monkeypatch.setattr(c5.fmt, 'fit', lambda *a, **k: pytest.fail('CDF fit called'))
    args = argparse.Namespace(snips=paths['snips'], multiwoz=paths['multiwoz'],
        profile_path=tmp_path/'unused.bin', coder_backend=c5.fmt.FAST_CODER,
        per_dataset=1, seed=42, max_prompt_tokens=8, max_encodes=32,
        max_semantic_prefix_rows=2048, dry_run=True)
    result = c5.run(args)
    out = capsys.readouterr().out
    assert len(result['selected']) == 2
    assert 'natural_episodes_after_nonprefix_filter=1' in out
    assert 'episodes_passing_prefix_guard=1' in out
    assert 'episodes_passing_encode_guard=1' in out
    assert 'final_selectable_episodes=1' in out
    assert 'source_start' in out and 'target_start' in out and 'token_ids' in out
    assert 'cross_user=' in out and 'expected_compressed_admissions<=' in out
    assert out.count('cumulative_compressed_encode_upper_bound=') == 2
    assert 'global_encode_limit=32' in out
    assert 'feasible_balanced_selection=true' in out
    assert 'per_dataset_encode_limit' not in out
    assert 'candidates_considered=1' in out
    assert 'expected_model_requests=8' in out
    assert 'expected_compressed_entry_encodes_upper_bound=' in out
    args.per_dataset = 2
    with pytest.raises(ValueError, match='prefix reuse is not a fallback'):
        c5.run(args)
    shortfall_output = capsys.readouterr().out
    assert 'feasible_balanced_selection=false' in shortfall_output
    assert 'snips: natural_episodes_after_nonprefix_filter=1 candidates_considered=1 selected=0/2' in shortfall_output
    assert 'multiwoz: natural_episodes_after_nonprefix_filter=1 candidates_considered=1 selected=0/2' in shortfall_output


def test_dry_run_reports_exact_minimum_when_global_cap_cannot_fit(
        tmp_path, monkeypatch, capsys):
    paths = {}
    for dataset in ('snips', 'multiwoz'):
        paths[dataset] = tmp_path/f'{dataset}.jsonl'
        paths[dataset].write_text(''.join(json.dumps(row)+'\n'
                                              for row in _rows(dataset, count=2)))
    monkeypatch.setattr(c5, 'verify_profile', lambda _: dict(sha256=c5.PROFILE_SHA256))
    monkeypatch.setattr(c5, 'FrozenK20V16Codec', lambda *a, **k: pytest.fail('codec loaded'))
    monkeypatch.setattr(c5.fmt, 'fit', lambda *a, **k: pytest.fail('CDF fit called'))
    args = argparse.Namespace(snips=paths['snips'], multiwoz=paths['multiwoz'],
        profile_path=tmp_path/'unused.bin', coder_backend=c5.fmt.FAST_CODER,
        per_dataset=2, seed=42, max_prompt_tokens=8, max_encodes=32,
        max_semantic_prefix_rows=2048, dry_run=True)
    with pytest.raises(ValueError, match='global encode limit 32'):
        c5.run(args)
    out = capsys.readouterr().out
    assert 'minimum_balanced_encode_upper_bound=36' in out
    assert "minimum_balanced_episode_ids=['snips:30->31', 'snips:32->33', 'multiwoz:20->21', 'multiwoz:22->23']" in out
    assert out.count('minimum_cost_episode=') == 4
    assert 'cumulative_minimum_encode_upper_bound=36' in out
    assert 'source_positions=' in out and 'target_positions=' in out
    assert 'cross_user=' in out and 'semantic_cluster=' in out
    assert 'exact_w3_token_ids=' in out
    assert 'minimum_single_episode_encode_cost=9' in out
    assert 'minimum_valid_n_episode_combined_encode_cost=18' in out
    assert 'episodes_passing_encode_guard=2' in out
    assert 'individual_candidates_passing_max_encodes=2' in out
    assert 'candidates_participating_in_feasible_balanced_selection=0' in out
    assert 'planner_reason=no_global_balanced_combination_within_encode_cap' in out
    args.max_encodes = 36
    result = c5.run(args)
    accepted_output = capsys.readouterr().out
    assert len(result['selected']) == 4
    assert result['expected_encodes_upper_bound'] == 36
    assert 'global_encode_limit=36 feasible_balanced_selection=true' in accepted_output
    args.max_encodes = 97
    with pytest.raises(ValueError, match=r'1\.\.96'):
        c5.run(args)


@pytest.mark.parametrize('cost,cap,feasible', [(8, 64, True), (9, 64, False), (9, 80, True)])
def test_four_per_dataset_dry_run_reports_actual_n_without_model_or_codec(
        tmp_path, monkeypatch, capsys, cost, cap, feasible):
    paths = {}
    for dataset in ('snips', 'multiwoz'):
        paths[dataset] = tmp_path/f'{dataset}.jsonl'
        paths[dataset].write_text(''.join(json.dumps(row)+'\n'
                                        for row in _rows(dataset, count=4)))
    discover = c5.discover
    def synthetic_costs(*args, **kwargs):
        episodes = discover(*args, **kwargs)
        for episode in episodes:
            episode['expected_compressed_admissions'] = cost
        return episodes
    monkeypatch.setattr(c5, 'discover', synthetic_costs)
    monkeypatch.setattr(c5, 'verify_profile', lambda _: dict(sha256=c5.PROFILE_SHA256))
    monkeypatch.setattr(c5, 'FrozenK20V16Codec', lambda *a, **k: pytest.fail('codec loaded'))
    monkeypatch.setattr(c5.fmt, 'fit', lambda *a, **k: pytest.fail('CDF fit called'))
    monkeypatch.setitem(sys.modules, 'semcache.models.loader', SimpleNamespace(
        load_model=lambda *a, **k: pytest.fail('model loaded')))
    args = argparse.Namespace(snips=paths['snips'], multiwoz=paths['multiwoz'],
        profile_path=tmp_path/'unused.bin', coder_backend=c5.fmt.FAST_CODER,
        per_dataset=4, seed=42, max_prompt_tokens=8, max_encodes=cap,
        max_semantic_prefix_rows=2048, dry_run=True)
    if feasible:
        result = c5.run(args)
        assert len(result['selected']) == 8
        assert result['expected_encodes_upper_bound'] == cost*8
    else:
        with pytest.raises(ValueError, match='No balanced 4\\+4'):
            c5.run(args)
    out = capsys.readouterr().out
    assert 'balanced_per_dataset=4' in out
    assert f'minimum_balanced_encode_upper_bound={cost*8}' in out
    assert f'minimum_encode_cap_excess={max(0, cost*8-cap)}' in out
    assert f'minimum_valid_n_episode_combined_encode_cost={cost*4}' in out
    assert out.count('minimum_cost_episode=') == 8
    assert f'dataset_cumulative_encode_upper_bound={cost*4}' in out
    assert f'cumulative_minimum_encode_upper_bound={cost*8}' in out
    assert 'source_positions=' in out and 'target_positions=' in out
    assert 'exact_w3_token_ids=' in out and 'semantic_cluster=' in out
    assert 'cross_user=' in out
    assert '2plus2' not in out and 'rejected_by_other_guard' not in out
    assert out.count('selected=4/4' if feasible else 'selected=0/4') == 2
