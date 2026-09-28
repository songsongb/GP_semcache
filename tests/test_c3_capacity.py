"""C3 synthetic logical replay checks; no OPT fixture, model, or codec."""
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.cache.metric_manager import CacheMetricManager
from semcache.experiments.cachegen.c3_capacity import (
    COMPRESSED, MAPPING_MODE, PROFILE_BYTES, RAW, RAW_ENTRY_BYTES, RAW_Q_BYTES,
    EmpiricalSizeAssigner, paired_delta, replay_cell, run, validate_budgets,
    validate_size_index, workload_facts,
)
from semcache.experiments.cachegen.c15c.holdout import UNIFORM
from semcache.experiments.cachegen.common import digest
from semcache.simulation.multi_user import aggregate, simulate


def holdout_fixtures():
    blocks = [dict(block_id='b', partition='evaluation', dataset='snips', token_group_size=3,
                   query_id='snips_eval', source_group_id='snips_eval', query_token_ids=[1, 2, 3],
                   token_ids=[1, 2, 3], start_position=0, absolute_positions=[0, 1, 2],
                   model_revision='m', tokenizer_revision='t', file='b.pt', sha256='x',
                   layers=32, hidden_dim=2560, heads=32, head_dim=80, dtype='float16'),
              dict(block_id='a', partition='evaluation', dataset='multiwoz', token_group_size=3,
                   query_id='multiwoz_eval', source_group_id='multiwoz_eval', query_token_ids=[1, 2, 3],
                   token_ids=[1, 2, 3], start_position=0, absolute_positions=[0, 1, 2],
                   model_revision='m', tokenizer_revision='t', file='a.pt', sha256='y',
                   layers=32, hidden_dim=2560, heads=32, head_dim=80, dtype='float16'),
              dict(block_id='c', partition='evaluation', dataset='snips', token_group_size=10)]
    sampling = {dataset: dict(calibration=[], evaluation=[dict(source_id=f'{dataset}_eval')],
                              hashes=dict(calibration=digest([]),
                                          evaluation=digest([dict(source_id=f'{dataset}_eval')])))
                for dataset in ('snips', 'multiwoz')}
    capture = dict(status='CAPTURED', model_config=dict(name='facebook/opt-2.7b', dtype='float16'),
                   model_metadata=dict(resolved_model_revision='m', resolved_tokenizer_revision='t'),
                   scope='base_raw_unscaled_linear_projection; no LoRA adapter',
                   blocks=blocks, sampling=sampling, sampling_sha256=digest(sampling))
    manifest = dict(status='COMPLETED', selected_policy=UNIFORM, evaluation_block_ids=['a', 'b'],
                    selected_holdout_count=2, required_window_size=3)
    def group(sizes):
        return dict(block_count=len(sizes), physical_storage=dict(per_block_bitstream_bytes=sizes,
            bitstream_pool_bytes=sum(sizes), global_profile_bytes=PROFILE_BYTES,
            total_physical_bytes=sum(sizes)+PROFILE_BYTES))
    summary = dict(evaluation_block_count=2, required_window_size=3,
                   overall={UNIFORM: group([100, 200])},
                   datasets={'multiwoz': {UNIFORM: group([100])}, 'snips': {UNIFORM: group([200])}})
    return capture, manifest, summary


def tiny_index():
    return dict(count=2, entries=[dict(dataset='snips', compressed_kv_frame_bytes=100),
                                  dict(dataset='multiwoz', compressed_kv_frame_bytes=200)])


def tiny_rows(dataset='snips'):
    return [dict(dataset=dataset, source_id=str(i), token_ids=[n, n+1, n+2], cluster_id=0)
            for i, n in enumerate((1, 4, 1, 4, 1, 4))]


def test_raw_entry_bytes_and_profile_once():
    assert RAW_ENTRY_BYTES == 1_474_560
    assert RAW_Q_BYTES == 491_520
    cache = GlobalCache(RAW_ENTRY_BYTES, capacity_charge=lambda e: 491_620,
                        shared_overhead_bytes=PROFILE_BYTES)
    assert cache.charged_cache_bytes == PROFILE_BYTES
    first = CacheEntry(0, (1, 2, 3), (0, 3), RAW_ENTRY_BYTES)
    second = CacheEntry(0, (4, 5, 6), (0, 3), RAW_ENTRY_BYTES)
    assert cache.insert(first, 1)
    assert cache.insert(second, 1)
    assert cache.charged_cache_bytes == PROFILE_BYTES+2*491_620
    assert cache.logical_cache_bytes == 2*RAW_ENTRY_BYTES


def test_byte_budget_eviction_order_same_as_reference():
    original = GlobalCache(2*RAW_ENTRY_BYTES)
    charged = GlobalCache(2*RAW_ENTRY_BYTES, capacity_charge=lambda e: e.size_bytes)
    fast = GlobalCache(2*RAW_ENTRY_BYTES, capacity_charge=lambda e: e.size_bytes,
                       homogeneous_logical_fastpath=True)
    for i in range(5):
        for cache in (original, charged, fast):
            cache.advance(i)
        entry = CacheEntry(0, (i, i+1, i+2), (0, 3), RAW_ENTRY_BYTES)
        a, b, c = [], [], []
        original.insert(entry, i+1, on_event=lambda kind, e, score: a.append((kind, e.key)))
        charged.insert(CacheEntry(0, entry.token_ids, (0, 3), RAW_ENTRY_BYTES), i+1,
                       on_event=lambda kind, e, score: b.append((kind, e.key)))
        fast.insert(CacheEntry(0, entry.token_ids, (0, 3), RAW_ENTRY_BYTES), i+1,
                    on_event=lambda kind, e, score: c.append((kind, e.key)))
        assert a == b == c
        assert set(original.entries) == set(charged.entries) == set(fast.entries)
        assert original.logical_cache_bytes == charged.charged_cache_bytes == fast.charged_cache_bytes
    fast.assert_homogeneous_replay()
    with pytest.raises(ValueError, match='no-reuse'): fast.record_reuse(next(iter(fast.entries.values())))


def test_incremental_appearance_counts_equal_original_window_sum():
    manager = CacheMetricManager(GlobalCache(10*RAW_ENTRY_BYTES), frequency_window=3)
    sequence = [[('a',), ('a',)], [('b',)], [('a',), ('c',)], [('c',)], [('a',)]]
    for keys in sequence:
        manager.arrive(keys)
        assert manager.frequencies == sum(manager.appearances, Counter())


def test_size_index_pairing_w3_and_dataset_subsequences():
    capture, manifest, summary = holdout_fixtures()
    index = validate_size_index(capture, manifest, summary, expected_count=2)
    assert index['mapping_mode'] == MAPPING_MODE
    assert [(e['block_id'], e['compressed_kv_frame_bytes']) for e in index['entries']] == [('a', 100), ('b', 200)]
    assert all(e['token_group_size'] == 3 for e in index['entries'])
    manifest['evaluation_block_ids'] = ['b', 'a']
    with pytest.raises(ValueError, match='ordering'): validate_size_index(capture, manifest, summary, expected_count=2)
    manifest['evaluation_block_ids'] = ['a', 'b']
    summary['datasets']['snips'][UNIFORM]['physical_storage']['per_block_bitstream_bytes'] = [100]
    with pytest.raises(ValueError, match='Per-dataset'): validate_size_index(capture, manifest, summary, expected_count=2)
    summary['datasets']['snips'][UNIFORM]['physical_storage']['per_block_bitstream_bytes'] = [200]
    summary['overall'][UNIFORM]['physical_storage']['per_block_bitstream_bytes'] = [100, 0]
    with pytest.raises(ValueError, match='frame lengths'): validate_size_index(capture, manifest, summary, expected_count=2)
    summary['overall'][UNIFORM]['physical_storage']['per_block_bitstream_bytes'] = [100, 200]
    capture['blocks'][1]['block_id'] = 'b'
    with pytest.raises(ValueError): validate_size_index(capture, manifest, summary, expected_count=2)


def test_stable_dataset_stratified_assignment():
    index = tiny_index()
    a = EmpiricalSizeAssigner('snips', index)
    b = EmpiricalSizeAssigner('snips', index)
    key = (2, (1, 2, 3))
    assert a.frame_bytes(key) == b.frame_bytes(key) == 100
    assert EmpiricalSizeAssigner('multiwoz', index).frame_bytes(key) == 200


def test_reuse_diagnostics_distinguish_same_query_duplicates_from_revisits():
    one_query = [dict(dataset='snips', token_ids=[1, 2, 3, 1, 2, 3], cluster_id=0)]
    first = workload_facts(one_query)
    assert first['access_count'] == 4
    assert first['unique_cache_keys'] == 3
    assert first['repeated_accesses'] == 1
    assert first['keys_accessed_more_than_once'] == 1
    assert first['total_revisit_events'] == 0
    assert first['max_accesses_for_one_key'] == 2
    assert first['reuse_opportunity'] is False
    later_query = one_query + [dict(dataset='snips', token_ids=[1, 2, 3], cluster_id=0)]
    second = workload_facts(later_query)
    assert second['access_count'] == 5
    assert second['unique_cache_keys'] == 3
    assert second['repeated_accesses'] == 2
    assert second['keys_accessed_more_than_once'] == 1
    assert second['total_revisit_events'] == 1
    assert second['max_accesses_for_one_key'] == 3
    assert second['reuse_opportunity'] is True


def test_same_access_trace_and_capacity_only_divergence():
    rows, index = tiny_rows(), tiny_index()
    raw = replay_cell(rows, 'snips', RAW_ENTRY_BYTES, RAW, index)
    compressed = replay_cell(rows, 'snips', RAW_ENTRY_BYTES, COMPRESSED, index)
    assert raw['access_sequence_sha256'] == compressed['access_sequence_sha256'] == workload_facts(rows)['access_sequence_sha256']
    assert raw['total_accesses'] == compressed['total_accesses'] == 6
    assert compressed['cache_hits'] > raw['cache_hits']
    assert compressed['peak_resident_entries'] > raw['peak_resident_entries']
    assert compressed['final_physical_resident_bytes'] <= RAW_ENTRY_BYTES
    assert compressed['mean_compressed_entry_bytes'] == RAW_Q_BYTES+100
    delta = paired_delta(raw, compressed)
    assert delta['compressed_minus_raw_hit_rate_percentage_points'] > 0
    assert delta['compressed_minus_raw_miss_count'] < 0
    roomy_raw = replay_cell(rows, 'snips', 10*RAW_ENTRY_BYTES, RAW, index)
    roomy_compressed = replay_cell(rows, 'snips', 10*RAW_ENTRY_BYTES, COMPRESSED, index)
    assert roomy_raw['cache_hits'] == roomy_compressed['cache_hits']
    assert roomy_raw['admissions'] == roomy_compressed['admissions']
    assert roomy_raw['evictions'] == roomy_compressed['evictions'] == 0
    reference = aggregate(simulate(rows, 25, capacity=RAW_ENTRY_BYTES, dataset='snips')['trace'])
    assert raw['cache_hits'] == reference['candidate_hit_count']
    assert raw['admissions'] == reference['admission_count']
    assert raw['evictions'] == reference['eviction_count']
    assert raw['rejected_admissions'] == reference['rejection_count']


def test_no_codec_calls_and_dry_run(monkeypatch, tmp_path, capsys):
    from semcache.experiments.cachegen import c3_capacity as c3
    from semcache.experiments.cachegen.shared import core
    from semcache.experiments.cachegen.c15c.policy import UniformKVPolicy
    monkeypatch.setattr(core, 'arithmetic_encode', lambda *a, **k: pytest.fail('codec used'))
    monkeypatch.setattr(core, 'arithmetic_decode', lambda *a, **k: pytest.fail('codec used'))
    monkeypatch.setattr(UniformKVPolicy, 'quantize', lambda *a, **k: pytest.fail('quantizer used'))
    index = tiny_index()
    assert replay_cell(tiny_rows(), 'snips', RAW_ENTRY_BYTES, COMPRESSED, index)['total_requests'] == 6
    monkeypatch.setattr(c3, 'load_measured_sizes', lambda *a, **k: index)
    monkeypatch.setattr(c3, 'replay_cell', lambda *a, **k: pytest.fail('dry-run replayed'))
    paths = {}
    for dataset in ('snips', 'multiwoz'):
        path = tmp_path/f'{dataset}.jsonl'
        row = dict(dataset=dataset, token_ids=[1, 2, 3], cluster_id=0, model_id='facebook/opt-2.7b',
                   tokenizer_id='t', semantic_assignment_source='fixture')
        rows = [row, dict(row)] if dataset == 'snips' else [row]
        path.write_text(''.join(json.dumps(item)+'\n' for item in rows))
        paths[dataset] = path
    output = tmp_path/'output'
    args = SimpleNamespace(capture_manifest=tmp_path/'capture.json', holdout_dir=tmp_path/'holdout',
        rate_calibration_dir=tmp_path/'calibration', snips=paths['snips'], multiwoz=paths['multiwoz'],
        output_dir=output, budgets_mib=(64, 128, 256, 512), users=25, seed=42, dry_run=True)
    result = run(args)
    assert result['dry_run'] and not output.exists()
    report = capsys.readouterr().out
    assert 'expected experiment cells=16' in report
    assert ('snips: total_accesses=2 unique_cache_keys=1 repeated_accesses=1 '
            'keys_accessed_more_than_once=1 total_revisit_events=1 '
            'max_accesses_for_one_key=2 reuse_opportunity=true') in report
    assert ('multiwoz: total_accesses=1 unique_cache_keys=1 repeated_accesses=0 '
            'keys_accessed_more_than_once=0 total_revisit_events=0 '
            'max_accesses_for_one_key=1 reuse_opportunity=false') in report
    with pytest.raises(ValueError): validate_budgets((64, 64))
