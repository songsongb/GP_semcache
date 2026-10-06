"""Deterministic CPU byte replay; no tensor, model, quality, or codec execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from semcache.experiments.cachegen import c8a_capacity as capacity
from semcache.experiments.cachegen import c7b3_q_freeze as freeze
from test_cachegen_c7b3_q_freeze import evidence, put, csv_rows


def episode(i, cluster=0):
    return dict(episode_id=str(i), cluster=cluster, token_ids=[i+10, i+11, i+12])


def test_exact_policies_budgets_and_raw_bytes():
    assert capacity.POLICIES == ('RAW_QKV', 'KV_COMP', 'Q24_KV_COMP')
    assert capacity.BUDGETS == (2, 4, 8, 16)
    assert capacity.RAW_ROLE_BYTES == 491520 and capacity.RAW_ENTRY_BYTES == 1474560
    assert [b*capacity.RAW_ENTRY_BYTES for b in capacity.BUDGETS] == [2949120, 5898240, 11796480, 23592960]
    with pytest.raises(ValueError): capacity.replay([episode(0)], {'0': 1}, 'Q32_KV_COMP', 10)


def test_variable_byte_lru_and_lookup_refresh():
    cache = capacity.ByteLRU(10)
    a, b, c = [(0, (i, i+1, i+2)) for i in range(3)]
    cache.admit(a, 4, 'a'); cache.admit(b, 5, 'b')
    assert cache.lookup(a)['hit']
    result = cache.admit(c, 6, 'c')
    assert result['evicted_keys'] == [capacity.serialized_key(b)]
    assert result['evicted_bytes'] == 5 and cache.resident_bytes == 10
    assert list(cache.entries) == [a, c] and cache.peak_bytes == 10
    assert result['evictions'][0]['source_episode_id'] == 'b'


def test_duplicate_matches_global_cache_no_update_no_refresh():
    from semcache.cache.global_cache import GlobalCache
    from semcache.cache.cache_entry import CacheEntry
    native = GlobalCache(100)
    first = CacheEntry(0, (1, 2, 3), (3, 6), 4)
    assert native.insert(first)
    assert not native.insert(CacheEntry(0, (1, 2, 3), (7, 10), 7))
    assert native.entries[first.key] is first
    cache = capacity.ByteLRU(10)
    cache.admit(first.key, 4, 'first'); other = (0, (4, 5, 6)); cache.admit(other, 4, 'other')
    result = cache.admit(first.key, 7, 'replacement')
    assert result['event_type'] == 'DUPLICATE_KEY_REJECTED' and not result['admitted']
    assert cache.entries[first.key]['entry_bytes'] == 4 and cache.entries[first.key]['source_episode_id'] == 'first'
    assert list(cache.entries) == [first.key, other]
    result = cache.admit((1, (1, 2, 3)), 4, 'new')
    assert result['evicted_keys'] == [capacity.serialized_key(first.key)]
    assert cache.admit(first.key, 2, 'readmitted')['admitted']


def test_oversize_reject_preserves_residents():
    cache = capacity.ByteLRU(10); key = (0, (1, 2, 3))
    cache.admit(key, 8, 'keep')
    result = cache.admit((0, (4, 5, 6)), 11, 'oversize')
    assert result['admission_rejected_oversize'] and not result['admitted'] and not result['evicted_keys']
    assert cache.resident_bytes == 8 and cache.lookup(key)['hit']


def test_accounting_does_not_assume_every_compressed_entry_saves_bytes(evidence):
    account = dict(evidence.rows[32]['storage_accounting'])
    account.update(compressed_q_frame_bytes=capacity.RAW_ROLE_BYTES+1000,
        compressed_q_bitstream_bytes=capacity.RAW_ROLE_BYTES+1000-account['local_q_metadata_bytes'],
        total_resident_qkv_bytes=capacity.RAW_ROLE_BYTES+1000+account['compressed_kv_frame_bytes'],
        incremental_resident_byte_reduction_vs_kv_baseline=-1000)
    capacity.validate_account(account, True)


def test_two_phases_key_only_hits_and_determinism():
    cases = [episode(i) for i in range(4)]
    # Different source episode, same logical key: resident duplicate is rejected,
    # but either request hits if that key survives. Cluster remains part of key.
    cases += [dict(cases[-1], episode_id='duplicate'), episode(3, cluster=1)]
    sizes = {e['episode_id']: 4 for e in cases}
    first = capacity.replay(cases, sizes, 'RAW_QKV', 8)
    assert first == capacity.replay(cases, sizes, 'RAW_QKV', 8)
    summary, events, trace = first
    assert [r['phase'] for r in events] == ['ADMISSION']*6+['LOOKUP']*6
    assert summary['duplicate_key_updates'] == 0 and summary['duplicate_key_rejections'] == 1
    assert summary['final_resident_entries'] == 2 and summary['target_hits'] == 3
    assert summary['retained_selected_source_count'] == 3 and summary['retained_unique_key_count'] == 2
    assert events[-2]['resident_source_episode_id'] == '3' and events[-2]['hit']
    assert len(trace['after_admission']) == len(trace['after_lookup']) == 2


def test_causal_deltas_and_at_most_two_capacity_only_budget_recommendations():
    summaries = []
    for b in capacity.BUDGETS:
        for policy, retained in zip(capacity.POLICIES, (b, min(32, b*3), min(32, b*5))):
            summaries.append(dict(budget_raw_entry_equivalent=b, policy=policy,
                final_resident_entries=retained, eviction_count=32-retained,
                target_hits=retained, hit_rate=retained/32))
    deltas = capacity.comparisons(summaries)
    extra = next(r for r in deltas if r['budget_raw_entry_equivalent'] == 8 and r['comparison'] == 'Q24_KV_COMP - KV_COMP')
    assert extra['final_resident_entries'] == extra['target_hits'] == 8
    assert extra['eviction_count'] == -8 and extra['hit_rate'] == 0.25
    assert capacity.representative_budgets(summaries, deltas) == [2, 8]
    for r in summaries:
        r.update(final_resident_entries=32, eviction_count=0, target_hits=32, hit_rate=1.)
    assert capacity.representative_budgets(summaries, capacity.comparisons(summaries)) == []


def test_full_bound_replay_outputs_and_no_input_changes(evidence):
    e = evidence; freeze.freeze(e.args)
    args = SimpleNamespace(freeze_decision=e.args.output_root/'freeze_decision.json', plan_dir=e.plan,
                           c7b2_root=e.b2, output_root=e.tmp/'replay')
    before = {str(p): freeze.sha(p) for p in e.tmp.rglob('*') if p.is_file()}
    episodes, sizes, shared, _ = capacity.prepare(args)
    assert len(set(sizes['KV_COMP'].values())) == len(set(sizes['Q24_KV_COMP'].values())) == 32
    assert shared['Q24_KV_COMP'] == shared['KV_COMP']+(e.q/'profiles/q24.bin').stat().st_size
    summary = capacity.run(args)
    assert len(summary['results']) == 12 and summary['shared_profile_bytes_charged_per_entry'] == 0
    for row in summary['results']:
        assert row['budget_bytes'] == row['budget_raw_entry_equivalent']*1474560
        assert row['target_lookups'] == 32 and row['target_hits']+row['target_misses'] == 32
        assert row['peak_resident_bytes'] <= row['budget_bytes']
    raw = next(r for r in summary['results'] if r['policy'] == 'RAW_QKV' and r['budget_raw_entry_equivalent'] == 2)
    assert raw['target_hits'] == raw['final_resident_entries'] == 2 and raw['eviction_count'] == 30
    trace = freeze.read(args.output_root/'residency_trace.json')
    for run in trace['runs']:
        assert sum(r['entry_bytes'] for r in run['after_admission']) <= run['budget_bytes']
    events = freeze.read_csv(args.output_root/'per_event.csv')
    assert len(events) == 12*64
    for budget in (2, 4, 8, 16):
        sequences = [[(r['phase'], r['episode_id'], r['logical_key']) for r in events if
            r['policy'] == policy and int(r['budget_raw_entry_equivalent']) == budget] for policy in capacity.POLICIES]
        assert sequences[0] == sequences[1] == sequences[2]
    manifest = freeze.read(args.output_root/'manifest.json')
    for field in ('model_inference_performed', 'quality_evaluated', 'latency_evaluated', 'transport_compression_enabled'):
        assert manifest[field] is False
    assert manifest['status'] == 'COMPLETE' and len(manifest['output_hashes']) == 6
    assert before == {p: freeze.sha(p) for p in before}
    assert len(summary['recommended_c8b_budgets_raw_entry_equivalent']) <= 2
    assert 'bleu' not in json.dumps(summary).lower()
    with pytest.raises(ValueError, match='overwrite'): capacity.run(args)


@pytest.mark.parametrize('damage', ('totals', 'entry', 'shared', 'selection', 'freeze'))
def test_replay_provenance_and_bytes_fail_closed(evidence, damage):
    e = evidence
    if damage == 'totals':
        path = e.b2/'storage_accounting.json'; data = freeze.read(path)
        data['modes'][freeze.BASELINE]['totals']['total_resident_qkv_bytes'] += 1; put(path, data)
    elif damage == 'entry':
        e.rows[0]['storage_accounting']['total_resident_qkv_bytes'] += 10; csv_rows(e.b2/'per_case.csv', e.rows)
    elif damage == 'shared':
        e.rows[0]['storage_accounting']['shared_profile_bytes_charged_per_entry'] = 100; csv_rows(e.b2/'per_case.csv', e.rows)
    e.refresh(); freeze.freeze(e.args)
    args = SimpleNamespace(freeze_decision=e.args.output_root/'freeze_decision.json', plan_dir=e.plan,
                           c7b2_root=e.b2, output_root=e.tmp/'replay')
    if damage == 'selection': (e.plan/'evaluation_selection.json').write_text('{}')
    if damage == 'freeze':
        saved = freeze.read(args.freeze_decision); saved['q_bins'] = 32; put(args.freeze_decision, saved)
    with pytest.raises(ValueError): capacity.run(args)
    assert not args.output_root.exists()


def test_imports_and_cli_help_have_no_model_codec_or_quality_dependencies():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root/'src'), CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
    code = ('import sys; from semcache.experiments.cachegen import c7b3_q_freeze, c8a_capacity; '
            'assert not any(n in sys.modules for n in ("torch", "transformers", "peft", "sacrebleu")); '
            'assert not any("c7b_q_profiles" in n or "physical_storage" in n for n in sys.modules)')
    subprocess.run([sys.executable, '-c', code], env=env, check=True)
    for script in ('64_freeze_cachegen_c7_q24.py', '65_run_cachegen_c8a_capacity_replay.py'):
        result = subprocess.run([sys.executable, str(root/'scripts'/script), '--help'], env=env, capture_output=True, text=True)
        assert result.returncode == 0 and '--output-root' in result.stdout
