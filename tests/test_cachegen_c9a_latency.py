"""CPU/synthetic C9-A tests; no CUDA, OPT, codecs, quality, or network measurement."""
from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semcache.experiments.cachegen import c9a_latency as latency
from semcache.experiments.cachegen import c9a_runtime as runtime
from semcache.experiments.cachegen import c8b_quality as c8
from semcache.experiments.cachegen import c7b3_q_freeze as provenance
from test_cachegen_c7b3_q_freeze import evidence, put
from test_cachegen_c8b_quality import replay


@pytest.fixture
def upstream(replay):
    e, args = replay
    prepared = c8.verify_capacity(args)
    root = e.tmp/'quality'; root.mkdir(); args.c8b_root = root
    rows = []
    for name, saved in prepared.expected.items():
        for episode, event in zip(prepared.episodes, saved['lookups']):
            layers = [dict(layer=l, **{role+'_'+suffix: 3 for role in 'qkv'
                for suffix in ('reused_projection_rows', 'native_projection_rows_skipped')}) for l in range(32)]
            audit = dict(same_reuse_mask_qkv=True, hit_positions=list(range(episode['target_start'], episode['target_start']+3)), per_layer=layers)
            # Match C8-B's producer: ByteLRU.lookup fields, not A's replay wrapper.
            lookup = {k: event[k] for k in ('hit', 'event_type', 'resident_bytes',
                'resident_entry_count', 'resident_source_episode_id')}
            rows.append(dict(lookup, condition=name, episode_id=episode['episode_id'],
                c8a_hit_exact_match=True, same_reuse_mask_qkv=True, target_admission_performed=False,
                reused_projection_rows_per_role_per_layer=3 if event['hit'] else 0,
                native_rows_skipped_per_role_per_layer=3 if event['hit'] else 0,
                reuse_audits=[audit, audit] if event['hit'] else []))
    put(root/'hit_audit.json', dict(cases=rows))
    m = dict(stage='C8-B', status='COMPLETE', recommendation='C8_B_SUPPORTS_Q24_KV_FOR_C9',
        c8a_residency_and_hit_vectors_reproduced=True, miss_paths_match_full=True,
        budgets_raw_entry_equivalent=[2, 8], policies=list(c8.POLICIES), reference_mode=c8.REFERENCE,
        physical_safety_contract=provenance.CONTRACT, cached_payload='TOTAL_QKV',
        c7b3_freeze_decision_sha256=provenance.sha(args.freeze_decision),
        frozen32_selection_sha256=provenance.SELECTION_SHA, q24_profile_sha256=provenance.Q_SHA,
        kv_profile_sha256=provenance.KV_SHA, **prepared.frozen['model_namespace'],
        **{k: 0 for k in c8.SAFETY}, transport_compression_enabled=False,
        input_hashes=dict(prepared.input_hashes), output_hashes={'hit_audit.json': provenance.sha(root/'hit_audit.json')})
    for name in ('manifest.json', 'summary.json', 'residency_trace.json', 'per_event.csv'):
        m['c8a_'+name.rsplit('.', 1)[0]+'_sha256'] = provenance.sha(args.c8a_root/name)
    put(root/'manifest.json', m)
    return e, args, prepared, m


def test_exact_conditions_and_bounded_options():
    assert latency.CONDITIONS == ('FULL_RECOMPUTE', 'B2_RAW_QKV', 'B2_KV_COMP', 'B2_Q24_KV_COMP',
        'B8_RAW_QKV', 'B8_KV_COMP', 'B8_Q24_KV_COMP')
    assert latency.BUDGETS == (2, 8) and len(latency.POLICIES) == 3
    latency.validate_options(5, 3)
    for warm, repeats in ((4, 3), (5, 4), (5, 30), (True, 3)):
        with pytest.raises(ValueError): latency.validate_options(warm, repeats)
    assert latency.tie_tolerance_ms() > 0


def test_bounded_execution_and_hit_only_codec_counts(upstream):
    _, _, prepared, _ = upstream
    counts = latency.expected_execution_counts(prepared.expected)
    assert counts['source_forward_count'] == 32
    assert counts['target_prompt_forward_count'] == 677
    assert counts['q_encode_count'] == 33  #32 sources + untimed real-block warmup
    assert counts['q_decode_count'] == 136  #(13+32) hits x3 + warmup
    assert counts['storage_decode_count'] == 203  #(4+13+18+32) hits x3 +2 warmups
    assert counts['target_greedy_forward_count'] == counts['teacher_forced_forward_count'] == 0
    assert all(counts[k] == 0 for k in c8.SAFETY)
    with pytest.raises(ValueError): latency.expected_execution_counts({})


def test_c7_c8_provenance_and_vectors_bound_without_quality_rerun(upstream, monkeypatch):
    e, args, prepared, _ = upstream
    monkeypatch.setattr(c8, 'prepare', lambda *a: pytest.fail('C8 quality prepare/BLEU must not be called'))
    latency.verify_c8b(args, prepared)
    assert str((args.c8b_root/'manifest.json').resolve()) in prepared.input_hashes
    assert str((args.c8b_root/'hit_audit.json').resolve()) in prepared.input_hashes
    assert prepared.c8b_manifest['recommendation'] == 'C8_B_SUPPORTS_Q24_KV_FOR_C9'


def test_actual_a_wrapper_b_condition_only_schema_regression(upstream):
    _, args, prepared, _ = upstream
    a = prepared.expected['B2_RAW_QKV']['lookups'][0]
    b = provenance.read(args.c8b_root/'hit_audit.json')['cases'][0]
    assert a['budget'] == 'B2' and a['policy'] == 'RAW_QKV'
    assert b['condition'] == 'B2_RAW_QKV' and 'budget' not in b and 'policy' not in b
    # This is exactly the positional raw-schema assumption that broke SERAPH.
    with pytest.raises(KeyError, match='policy'):
        all(b[k] == a[k] for k in a)
    assert latency.normalize_target_record(a, 'C8-A') == latency.normalize_target_record(b, 'C8-B')
    latency.verify_c8b(args, prepared)


@pytest.mark.parametrize('budget', ('B2', 'B8'))
@pytest.mark.parametrize('policy', ('RAW_QKV', 'KV_COMP', 'Q24_KV_COMP'))
def test_exact_condition_parsing_preserves_full_policy_suffix(budget, policy):
    assert latency.parse_target_condition(budget+'_'+policy) == (budget, policy)


@pytest.mark.parametrize('condition', ('B4_RAW_QKV', 'B16_KV_COMP', 'B8_Q32_KV_COMP',
    'B2_QKV', 'B2_Q24_KV_COMP_extra', 'FULL_RECOMPUTE', 'B2', None, ['B2_RAW_QKV']))
def test_unknown_malformed_condition_fails_closed(condition):
    with pytest.raises(ValueError, match='Unknown/malformed C8 target condition'):
        latency.parse_target_condition(condition)


def save_hit_audit(args, manifest, data):
    put(args.c8b_root/'hit_audit.json', data)
    manifest['output_hashes']['hit_audit.json'] = provenance.sha(args.c8b_root/'hit_audit.json')
    put(args.c8b_root/'manifest.json', manifest)


@pytest.mark.parametrize('damage', ('missing', 'duplicate', 'extra', 'unknown_condition', 'hit',
    'source', 'bytes', 'entry_count', 'target_admission', 'redundant_identity'))
def test_normalized_target_mismatch_fails_with_explicit_validation(upstream, damage):
    _, args, prepared, manifest = upstream
    data = provenance.read(args.c8b_root/'hit_audit.json')
    row = next(r for r in data['cases'] if r['hit'])
    if damage == 'missing': data['cases'].remove(row)
    elif damage == 'duplicate': data['cases'].append(dict(row))
    elif damage == 'extra': row['episode_id'] = 'unexpected-episode'
    elif damage == 'unknown_condition': row['condition'] = 'B4_RAW_QKV'
    elif damage == 'hit': row.update(hit=False, event_type='MISS', resident_source_episode_id=None)
    elif damage == 'source': row['resident_source_episode_id'] = 'wrong-source'
    elif damage == 'bytes': row['resident_bytes'] += 1
    elif damage == 'entry_count': row['resident_entry_count'] += 1
    elif damage == 'target_admission': row['target_admission_performed'] = True
    elif damage == 'redundant_identity': row['policy'] = 'Q24_KV_COMP'
    save_hit_audit(args, manifest, data)
    with pytest.raises(ValueError) as exc:
        latency.verify_c8b(args, prepared)
    message = str(exc.value)
    if damage in ('hit', 'source', 'bytes', 'entry_count', 'target_admission'):
        field = dict(hit='hit', source='resident_source_episode_id', bytes='resident_bytes',
            entry_count='resident_entry_count', target_admission='target_admission_performed')[damage]
        assert f'field={field}' in message and 'key=' in message
        assert 'C8-A=' in message and 'C8-B=' in message
    elif damage == 'missing': assert 'missing=' in message
    elif damage == 'duplicate': assert 'Duplicate C8-B canonical target key' in message
    elif damage == 'extra': assert 'extra=' in message


@pytest.mark.parametrize('field', ('condition', 'episode_id', 'hit', 'resident_bytes',
    'resident_entry_count', 'target_admission_performed', 'reuse_audits'))
def test_missing_b_target_fields_raise_value_error_not_raw_key_error(upstream, field):
    _, args, prepared, manifest = upstream
    data = provenance.read(args.c8b_root/'hit_audit.json')
    del data['cases'][0][field]
    save_hit_audit(args, manifest, data)
    with pytest.raises(ValueError): latency.verify_c8b(args, prepared)


def test_miss_source_absence_normalized_but_nonnull_source_rejected(upstream):
    _, args, prepared, manifest = upstream
    data = provenance.read(args.c8b_root/'hit_audit.json')
    row = next(r for r in data['cases'] if not r['hit'])
    del row['resident_source_episode_id']
    save_hit_audit(args, manifest, data)
    latency.verify_c8b(args, prepared)
    row['resident_source_episode_id'] = 'unexpected-source-on-miss'
    save_hit_audit(args, manifest, data)
    with pytest.raises(ValueError, match='field=resident_source_episode_id'):
        latency.verify_c8b(args, c8.verify_capacity(args))


@pytest.mark.parametrize('damage', ('missing_cases', 'nondict_row', 'missing_layers', 'missing_layer_id', 'missing_bindings'))
def test_malformed_audit_or_proof_fails_explicitly_without_key_error(upstream, damage):
    _, args, prepared, manifest = upstream
    data = provenance.read(args.c8b_root/'hit_audit.json')
    row = next(r for r in data['cases'] if r['hit'])
    if damage == 'missing_cases': del data['cases']
    elif damage == 'nondict_row': data['cases'][0] = None
    elif damage == 'missing_layers': del row['reuse_audits'][0]['per_layer']
    elif damage == 'missing_layer_id': del row['reuse_audits'][0]['per_layer'][0]['layer']
    elif damage == 'missing_bindings': del manifest['input_hashes']
    save_hit_audit(args, manifest, data)
    with pytest.raises(ValueError): latency.verify_c8b(args, prepared)


def test_incidental_b_fields_and_row_order_do_not_change_semantic_verification(upstream):
    _, args, prepared, manifest = upstream
    data = provenance.read(args.c8b_root/'hit_audit.json')
    data['cases'].reverse()
    for row in data['cases']: row.update(phase='audit-only', event_index=-1)
    save_hit_audit(args, manifest, data)
    latency.verify_c8b(args, prepared)


def test_bound_a_aggregate_hit_assertion_preserved(upstream):
    _, args, prepared, _ = upstream
    prepared.expected['B2_RAW_QKV']['summary']['target_hits'] = 3
    with pytest.raises(ValueError, match='aggregate HIT count differs: B2_RAW_QKV'):
        latency.verify_c8b(args, prepared)


@pytest.mark.parametrize('damage', ('recommendation', 'fitting', 'transport', 'hash', 'hit', 'source', 'mask', 'profile'))
def test_c8_provenance_or_hit_mismatch_fails_closed(upstream, damage):
    e, args, prepared, m = upstream
    root = args.c8b_root
    if damage == 'recommendation': m['recommendation'] = 'C8_B_REQUIRES_FURTHER_QUALITY_REVIEW'
    elif damage == 'fitting': m['runtime_q_profile_fit_count'] = 1
    elif damage == 'transport': m['transport_encode_calls'] = 1
    elif damage == 'profile': m['q24_profile_sha256'] = 'wrong'
    else:
        data = provenance.read(root/'hit_audit.json'); row = next(r for r in data['cases'] if r['hit'])
        if damage == 'hash':
            (root/'hit_audit.json').write_text('tampered')
        else:
            if damage == 'hit': row['hit'] = False
            if damage == 'source': row['resident_source_episode_id'] = 'wrong-source'
            if damage == 'mask': row['reuse_audits'][0]['hit_positions'] = [0, 1, 2]
            put(root/'hit_audit.json', data)
            m['output_hashes']['hit_audit.json'] = provenance.sha(root/'hit_audit.json')
    put(root/'manifest.json', m)
    with pytest.raises(ValueError): latency.verify_c8b(args, prepared)


class FakeClock:
    def __init__(self): self.ticks, self.sync_count = 0, 0
    def now(self): self.ticks += 1_000_000; return self.ticks
    def sync(self): self.sync_count += 1
    def measure(self, fn, gpu=True):
        return runtime.Clock(self.sync, self.now).measure(fn, gpu)


def test_synchronized_clock_and_cpu_only_region():
    fake = FakeClock(); timer = runtime.Clock(fake.sync, fake.now)
    result, ms = timer.measure(lambda: 42)
    assert result == 42 and ms == 1 and fake.sync_count == 2
    timer.measure(lambda: 43, gpu=False)
    assert fake.sync_count == 2


def test_native_projection_hooks_passive_and_restored():
    modules = {role: torch.nn.Linear(2, 2) for role in 'qkv'}
    adapter = SimpleNamespace(layers=[0], projection_modules=lambda _: modules)
    recorded = []
    class Event:
        def record(self): recorded.append(self)
    x = torch.randn(1, 4, 2)
    native = {role: module(x).detach().clone() for role, module in modules.items()}
    with runtime.native_projection_events(adapter, Event) as events:
        measured = {role: module(x) for role, module in modules.items()}
    assert len(events.pairs) == 3 and len(recorded) == 6
    assert all(torch.equal(native[role], measured[role]) for role in modules)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in modules.values())
    with pytest.raises(RuntimeError):
        with runtime.native_projection_events(adapter, Event): raise RuntimeError('fixture failure')
    assert all(not module._forward_hooks for module in modules.values())


def test_timed_request_schema_decode_only_on_compressed_hit():
    episode = dict(target_index=0, target_user='fixture', target_start=1, token_ids=[3, 4, 5])
    calls = []
    backend = SimpleNamespace(rows=[dict(token_ids=[2, 3, 4, 5, 6])],
        prefill=lambda ids, user, hits: (SimpleNamespace(logits=torch.ones(1, 5, 8)), hits, []))
    fake = FakeClock()
    values, evidence = runtime.timed_request(backend, None, episode, fake)
    assert values['lookup_ms'] == values['storage_decode_ms'] == 0 and evidence.hits is None
    assert fake.sync_count == 2
    def cache(hit, compressed):
        resident = object()
        return SimpleNamespace(logical_lookup=lambda e: (resident if hit else None,
            dict(hit=hit, resident_source_episode_id='source' if hit else None)),
            storage=SimpleNamespace(decode_entry=lambda r: calls.append(r) or r) if compressed else None)
    for hit, compressed in ((False, True), (True, False), (True, True)):
        values, evidence = runtime.timed_request(backend, cache(hit, compressed), episode, FakeClock())
        assert (values['storage_decode_ms'] > 0) == (hit and compressed)
        assert (evidence.hits is not None) == hit
        assert abs(values['target_total_ms']-sum(values[k] for k in ('lookup_ms', 'storage_decode_ms', 'model_forward_ms', 'control_overhead_ms'))) < 1e-10
    assert len(calls) == 1


def synthetic_raw(episodes, expected):
    rows = []
    for name in latency.CONDITIONS:
        for index, e in enumerate(episodes):
            hit = False if name == latency.REFERENCE else expected[name]['lookups'][index]['hit']
            source = None if name == latency.REFERENCE else expected[name]['lookups'][index]['resident_source_episode_id']
            for repeat in range(3):
                lookup = 0 if name == latency.REFERENCE else .1
                decode = .2 if hit and not name.endswith('_RAW_QKV') else 0
                model = (9 if hit else 10)+repeat
                rows.append(dict(condition=name, episode_id=e['episode_id'], repeat=repeat, hit=hit,
                    retained_source_episode_id=source, lookup_ms=lookup, storage_decode_ms=decode,
                    model_forward_ms=model, projection_or_mixed_projection_ms=1., control_overhead_ms=.01,
                    target_total_ms=lookup+decode+model+.01))
    return rows


def test_three_repeat_medians_stats_hit_miss_full_speedup_and_pairs(upstream):
    _, _, prepared, _ = upstream
    raw = synthetic_raw(prepared.episodes, prepared.expected)
    rows = latency.per_case(raw, prepared.episodes, prepared.expected, 1e-8)
    assert len(rows) == 224 and len(raw) == 672
    full = next(r for r in rows if r['condition'] == latency.REFERENCE)
    assert full['model_forward_ms'] == 11 and full['speedup_vs_full'] == 1
    summary = latency.summaries(rows)
    assert next(r for r in summary if r['condition'] == 'B8_Q24_KV_COMP')['MISS']['cases'] == 0
    assert next(r for r in summary if r['condition'] == 'B2_Q24_KV_COMP')['HIT']['cases'] == 13
    assert latency.stats([1, 2, 3, 4])['p50'] == 2.5
    assert latency.stats([1, 2, 3, 4])['p95'] == pytest.approx(3.85)
    assert latency.stats([])['mean'] is None
    pairs = latency.paired(rows, .00001)
    assert len(pairs) == 12
    pair = next(r for r in pairs if r['before'] == 'B8_KV_COMP' and r['after'] == 'B8_Q24_KV_COMP')
    assert len(pair['per_target']) == 32
    assert pair['components']['target_total_ms']['faster'] == 14
    assert pair['components']['lookup_ms']['effectively_tied'] == 32
    hit = next(r for r in rows if r['condition'] == 'B8_Q24_KV_COMP')
    assert hit['speedup_vs_full'] == pytest.approx(11.01/10.31)


@pytest.mark.parametrize('damage', ('missing', 'duplicate', 'hit', 'nan', 'negative', 'nested', 'sum', 'miss_decode'))
def test_bad_timing_or_repeated_decision_invalid(upstream, damage):
    _, _, prepared, _ = upstream
    rows = synthetic_raw(prepared.episodes, prepared.expected)
    if damage == 'missing': rows.pop()
    elif damage == 'duplicate': rows[1]['repeat'] = 0
    elif damage == 'hit': rows[0]['hit'] = True
    elif damage == 'nan': rows[0]['target_total_ms'] = float('nan')
    elif damage == 'negative': rows[0]['lookup_ms'] = -1
    elif damage == 'nested': rows[0]['projection_or_mixed_projection_ms'] = 1000
    elif damage == 'sum': rows[0]['control_overhead_ms'] = 1000
    elif damage == 'miss_decode': rows[0]['storage_decode_ms'] = .1
    with pytest.raises(ValueError): latency.per_case(rows, prepared.episodes, prepared.expected, 1e-8)


def test_evaluate_repeats_restore_cache_state_and_reuse_full_once(upstream, monkeypatch):
    _, _, prepared, _ = upstream
    prepared.episodes = [dict(e, target_index=i) for i, e in enumerate(prepared.episodes)]
    prepared.tie_tolerance_ms = 1e-6
    calls = []
    class Backend:
        adapter = SimpleNamespace(layers=[0])
        rows = [dict(token_ids=[0]*12) for _ in range(32)]
        counts = {k: 0 for k in c8.SAFETY}
        def prefill(self, ids, user, hits):
            calls.append(hits is not None)
            return SimpleNamespace(logits=torch.ones(1, 12, 8)), None, [0, 1, 2]
        def validate_logits(self, logits): assert torch.isfinite(logits).all()
    monkeypatch.setattr(runtime.c8run, 'representation_check', lambda *a: None)
    monkeypatch.setattr(runtime.c8run, 'validate_retained_hit', lambda *a: None)
    monkeypatch.setattr(runtime.c8run.previous, 'projection_audit', lambda audit, layers, n, hits:
        dict(hit_positions=list(range(hits[0].window.start, hits[0].window.end))))
    caches = {}
    fake = FakeClock()
    for name, saved in prepared.expected.items():
        policy = saved['summary']['policy']; budget = saved['summary']['budget_raw_entry_equivalent']
        cache = runtime.TimedCache(budget, policy, None, fake)
        for episode in prepared.episodes:
            entry = SimpleNamespace(key=runtime.capacity.logical_key(episode),
                physical_tensor_bytes=prepared.sizes[policy][episode['episode_id']])
            cache.admit(episode, entry)
        caches[name] = cache
    raw = []
    runtime.evaluate(Backend(), prepared, caches, fake, raw, lambda pairs: [.01]*len(pairs))
    assert len(calls) == len(raw) == 672 and sum(calls) == 77*3
    assert len([r for r in raw if r['condition'] == latency.REFERENCE]) == 32*3
    assert all(cache.lru.snapshot() == prepared.expected[name]['residency']['after_lookup'] for name, cache in caches.items())


def test_reports_exclude_source_quality_network_and_no_policy_freeze(upstream, tmp_path):
    _, _, prepared, _ = upstream
    source = [dict(condition=name, source_forward_ms=1., source_capture_ms=.1, storage_encode_ms=2., cache_admission_ms=.01)
        for name in latency.CONDITIONS[1:] for _ in range(32)]
    summary = latency.source_summary(source)
    assert summary['source_build_cost_in_primary_target_latency'] is False
    assert summary['conditions'][0]['mean_source_build_ms_per_admission'] == pytest.approx(3.11)
    raw = synthetic_raw(prepared.episodes, prepared.expected)
    root = tmp_path/'output'; root.mkdir()
    manifest = dict(status='COMPLETE', recommendation='C9_A_READY_FOR_NETWORK_ACCOUNTING', tie_tolerance_ms=1e-8,
        quality_evaluated_in_c9=False, network_latency_evaluated=False, system_policy_frozen=False)
    latency.write_reports(root, raw, source, prepared, manifest)
    assert len(manifest['output_hashes']) == 8
    saved = provenance.read(root/'target_latency_summary.json')
    assert saved['conditions'][0]['ALL']['target_total_ms']['mean'] < 12
    assert not (root/'freeze_decision.json').exists() and not (root/'transport_byte_accounting.json').exists()
    assert 'nested' in (root/'summary.md').read_text()
    # Own partial reports may be updated safely, preserving original failure.
    manifest.update(status='INVALID', recommendation='C9_A_LOCAL_LATENCY_NEEDS_REVIEW', failure_message='primary failure')
    latency.write_reports(root, raw[:1], [], prepared, manifest)
    assert provenance.read(root/'manifest.json')['failure_message'] == 'primary failure'
    assert 'NOT_COMPLETED' in (root/'summary.md').read_text()


@pytest.mark.parametrize('role', ('q', 'kv'))
def test_runtime_fitting_forbidden(role):
    from semcache.experiments.cachegen.c7b2_runtime import forbid_fitting, runtime_counts
    counts = runtime_counts()
    def fit(): pass
    with pytest.raises(RuntimeError):
        with forbid_fitting(counts, role): fit()


def test_cli_help_does_not_load_models_or_quality_dependencies():
    import os
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root/'src'), CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
    code = 'import sys; from semcache.experiments.cachegen import c9a_latency; assert not any(n in sys.modules for n in ("torch", "peft", "sacrebleu", "transformers"))'
    subprocess.run([sys.executable, '-c', code], env=env, check=True)
    result = subprocess.run([sys.executable, str(root/'scripts/67_run_cachegen_c9a_local_latency.py'), '--help'], env=env, text=True, capture_output=True)
    assert result.returncode == 0 and '--warmup' in result.stdout and '--repeats' in result.stdout
