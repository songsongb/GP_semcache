"""CPU-only C8-B provenance, physical reuse, and quality orchestration checks."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semcache.experiments.cachegen import c8b_quality as quality
from semcache.experiments.cachegen import c8b_runtime as runtime
from semcache.experiments.cachegen import c8a_capacity as capacity
from semcache.experiments.cachegen import c7b3_q_freeze as freeze
from test_cachegen_c7b3_q_freeze import evidence, put, csv_rows
from test_cachegen_c7b_q_calibration import backend
from test_cachegen_c7b2_q_quality import physical


@pytest.fixture
def replay(evidence):
    e = evidence
    # Variable measured sizes yield the six documented A hit counts, rather than
    # monkeypatching those assertions or substituting a global average.
    for r in e.rows:
        a = r['storage_accounting']; i = r['episode_index']
        a['compressed_kv_frame_bytes'] = 150000+193*i
        a['total_resident_qkv_bytes'] = a['resident_raw_q_bytes']+a['compressed_q_frame_bytes']+a['compressed_kv_frame_bytes']
    csv_rows(e.b2/'per_case.csv', e.rows)
    accounting = freeze.read(e.b2/'storage_accounting.json')
    for mode, saved in accounting['modes'].items():
        accounts = [r['storage_accounting'] for r in e.rows if r['mode'] == mode]
        saved['totals'] = {k: sum(a[k] for a in accounts) for k in saved['totals']}
        total = saved['totals']['total_resident_qkv_bytes']
        saved.update(total_resident_bytes=total, mean_resident_bytes=total/32,
            whole_qkv_compression_ratio=32*capacity.RAW_ENTRY_BYTES/total)
    put(e.b2/'storage_accounting.json', accounting)
    e.refresh(); freeze.freeze(e.args)
    args = SimpleNamespace(freeze_decision=e.args.output_root/'freeze_decision.json', plan_dir=e.plan,
        c7b2_root=e.b2, c8a_root=e.tmp/'capacity', output_root=e.tmp/'capacity')
    capacity.run(args)
    return e, args


def test_exact_conditions_and_selection_based_label():
    assert quality.BUDGETS == (2, 8)
    assert quality.POLICIES == ('RAW_QKV', 'KV_COMP', 'Q24_KV_COMP')
    assert len([quality.condition(b, p) for b in quality.BUDGETS for p in quality.POLICIES]) == 6
    assert quality.LABEL == 'SELECTION_BASED_CONTROLLED_QUALITY_REPLAY'
    for b, p in ((4, 'RAW_QKV'), (16, 'KV_COMP'), (2, 'Q32_KV_COMP'), (8, 'Q20')):
        with pytest.raises(ValueError): quality.condition(b, p)


def test_hash_bound_capacity_and_hit_vectors(replay):
    e, args = replay
    prepared = quality.verify_capacity(args)
    assert len(prepared.expected) == 6
    for b in quality.BUDGETS:
        for p, n in zip(quality.POLICIES, quality.EXPECTED_HITS[b]):
            saved = prepared.expected[quality.condition(b, p)]
            assert sum(r['hit'] for r in saved['lookups']) == saved['summary']['target_hits'] == n
            assert len(saved['residency']['after_admission']) == saved['summary']['final_resident_entries']
    assert str((args.c8a_root/'manifest.json').resolve()) in prepared.input_hashes
    assert str(args.freeze_decision.resolve()) in prepared.input_hashes


@pytest.mark.parametrize('damage', ('freeze', 'hash', 'vector', 'resident', 'recommendation'))
def test_capacity_provenance_mismatch_fails_closed(replay, damage):
    e, args = replay
    if damage == 'freeze':
        value = freeze.read(args.freeze_decision); value['selected_q_candidate'] = 'Q32'; put(args.freeze_decision, value)
    elif damage == 'hash':
        with (args.c8a_root/'per_event.csv').open('a') as stream: stream.write('altered\n')
    else:
        m = freeze.read(args.c8a_root/'manifest.json')
        if damage == 'recommendation': m['recommended_c8b_budgets_raw_entry_equivalent'] = [2, 4]
        if damage == 'resident':
            data = freeze.read(args.c8a_root/'residency_trace.json'); data['runs'][0]['after_admission'][0]['source_episode_id'] = 'wrong'
            put(args.c8a_root/'residency_trace.json', data)
        if damage == 'vector':
            rows = freeze.read_csv(args.c8a_root/'per_event.csv'); row = next(r for r in rows if r['phase'] == 'LOOKUP')
            row['hit'] = 'True' if row['hit'] == 'False' else 'False'; csv_rows(args.c8a_root/'per_event.csv', rows)
        m['output_hashes'] = {p.name: freeze.sha(p) for p in args.c8a_root.iterdir() if p.name != 'manifest.json'}
        put(args.c8a_root/'manifest.json', m)
    with pytest.raises(ValueError): quality.verify_capacity(args)


def test_actual_resident_wrapper_lru_duplicate_and_no_target_admission():
    cache = runtime.ResidentCache(2, 'RAW_QKV', None)
    # LRU itself accounts the real physical entry property; no logical count cap.
    def e(i): return dict(episode_id=str(i), cluster=1, token_ids=[i, i+1, i+2])
    def entry(i): return SimpleNamespace(key=capacity.logical_key(e(i)), physical_tensor_bytes=capacity.RAW_ENTRY_BYTES)
    cache.admit(e(0), entry(0)); cache.admit(e(1), entry(1))
    assert not cache.admit(dict(e(0), episode_id='duplicate'), entry(0))['admitted']
    assert cache.lru.entries[capacity.logical_key(e(0))]['source_episode_id'] == '0'
    cache.admit(e(2), entry(2))
    before = cache.lru.snapshot()
    view, event = cache.lookup(e(0))
    assert view is None and not event['hit'] and cache.lru.snapshot() == before
    assert len(cache.payloads) == 2 and cache.lru.resident_bytes == 2*capacity.RAW_ENTRY_BYTES


def test_physical_representation_contracts_and_temporary_decoded_view(monkeypatch):
    x = torch.ones(1, 3, 2560, dtype=torch.float16)
    tensors = {l: (x.clone(), x.clone(), x.clone()) for l in range(32)}
    key = (1, (5, 6, 7))
    raw = SimpleNamespace(key=key, tensors=tensors, physical_tensor_bytes=capacity.RAW_ENTRY_BYTES)
    kv = SimpleNamespace(profile_sha256=freeze.KV_SHA)
    baseline = SimpleNamespace(key=key, tensors=None, q_tensors={l: x.clone() for l in range(32)}, compressed_kv=kv,
        physical_tensor_bytes=600000)
    q = SimpleNamespace(key=key, tensors=None, q_tensors=None, compressed_kv=kv, compressed_q_frame=b'frame',
        compressed_q_profile_sha256=freeze.Q_SHA, physical_tensor_bytes=210000)
    decoded = []
    def decode(resident):
        decoded.append(resident)
        return SimpleNamespace(resident=resident, tensors={l: ((x/2).clone(), x.clone(), x.clone()) for l in range(32)})
    storage = SimpleNamespace(q=SimpleNamespace(profile=SimpleNamespace(metadata={'layers': 32})),
        decode_entry=decode, validate_decoded=lambda r, v: None)
    assert runtime.representation_check('RAW_QKV', raw)['raw_kv_resident_after_insert']
    assert runtime.representation_check('KV_COMP', baseline, storage)['raw_q_resident_after_insert']
    assert not runtime.representation_check('Q24_KV_COMP', q, storage)['raw_q_resident_after_insert']
    cache = runtime.ResidentCache(2, 'Q24_KV_COMP', storage)
    episode = dict(episode_id='a', cluster=1, token_ids=[5, 6, 7])
    cache.admit(episode, q); view, event = cache.lookup(episode)
    assert event['hit'] and decoded == [q] and view.resident is q and q.q_tensors is None
    assert torch.equal(view.tensors[0][0], x/2)
    q.q_tensors = {0: x}
    with pytest.raises(ValueError): runtime.representation_check('Q24_KV_COMP', q, storage)
    baseline.tensors = tensors
    with pytest.raises(ValueError): runtime.representation_check('KV_COMP', baseline, storage)


def test_retained_duplicate_source_payload_provenance():
    from semcache.cache.cache_entry import CacheEntry
    from semcache.semantic.subsequence import Subsequence
    from semcache.semantic.hit_selection import CacheHit
    target = dict(episode_id='second', cluster=1, token_ids=[5, 6, 7], source_start=9, target_start=2,
        source_user='user_b', source_id='second-source')
    retained = dict(target, episode_id='first', source_start=4, source_user='user_a', source_id='first-source')
    entry = CacheEntry(1, (5, 6, 7), (4, 7), 100,
        qkv_metadata=dict(component_scope='total_qkv', source_user='user_a', source_id='first-source'))
    hit = CacheHit(Subsequence((5, 6, 7), 2, 5), entry)
    runtime.validate_retained_hit(target, retained, hit)
    with pytest.raises(ValueError): runtime.validate_retained_hit(target, dict(retained, cluster=2), hit)


def test_actual_c2_frozen_lookup_via_byte_resident_wrapper(physical, monkeypatch):
    from semcache.cache.cache_entry import CacheEntry
    p = physical
    monkey_entries = {'RAW_QKV': CacheEntry.from_tensors(p.episode['cluster'], p.episode['token_ids'],
        (p.episode['source_start'], p.episode['source_start']+3), p.tensors)}
    storages = dict(RAW_QKV=None, KV_COMP=p.storages[runtime.previous.MODES[0]],
        Q24_KV_COMP=p.storages[runtime.previous.MODES[1]])
    # Synthetic fitted-profile bytes have their own literal hash; production
    # always checks the pinned authoritative Q24 hash, with no CLI override.
    monkeypatch.setattr(freeze, 'Q_SHA', storages['Q24_KV_COMP'].q.sha256)
    for policy in ('KV_COMP', 'Q24_KV_COMP'):
        monkey_entries[policy], _ = storages[policy].encode(p.episode, p.tensors)
    e = dict(p.episode, episode_id='source')
    views = {}
    for policy, entry in monkey_entries.items():
        cache = runtime.ResidentCache(2, policy, storages[policy])
        assert cache.admit(e, entry)['admitted']
        views[policy], event = cache.lookup(e)
        assert event['hit'] and cache.payloads[entry.key] is entry
        assert cache.lru.resident_bytes == entry.physical_tensor_bytes
    assert monkey_entries['Q24_KV_COMP'].q_tensors is None
    expected_q = storages['Q24_KV_COMP'].q.decode(monkey_entries['Q24_KV_COMP'].compressed_q_frame)
    for l in range(32):
        assert torch.equal(views['RAW_QKV'].tensors[l][0], views['KV_COMP'].tensors[l][0])
        assert torch.equal(views['Q24_KV_COMP'].tensors[l][0], expected_q[l:l+1])
        for role in (1, 2):
            assert torch.equal(views['KV_COMP'].tensors[l][role], views['Q24_KV_COMP'].tensors[l][role])
    assert all(p.counts[k] == 0 for k in quality.SAFETY)


def test_full_two_phase_quality_orchestration_no_repeated_source_captures(replay, monkeypatch):
    e, args = replay
    prepared = quality.verify_capacity(args)
    prepared.episodes = [dict(r, source_index=i, target_index=i) for i, r in enumerate(prepared.episodes)]
    prepared.rows = [dict(token_ids=[5, 6, 7, 8, 9], reference_text='same') for _ in range(32)]
    prepared.official = [dict(generated_token_ids=[2], generated_text='same') for _ in range(32)]
    # This fixture checks routing/state/counters. The separate real mixed and C2
    # tests above check tensor injection and physical encoding/decoding.
    monkeypatch.setattr(runtime, 'representation_check', lambda policy, entry, storage=None:
        dict(policy=policy, resident_bytes=entry.physical_tensor_bytes))
    class ToyBackend:
        def __init__(self):
            self.counts = runtime.previous.runtime_counts()
            self.tokenizer = SimpleNamespace(decode=lambda tokens, skip_special_tokens: 'same')
            self.rows = prepared.rows; self.reuse_audits = []; self.greedy_calls = []; self.teacher_calls = []
            self.storages = dict(RAW_QKV=None, KV_COMP=self, Q24_KV_COMP=self)
        def build_source(self, episode):
            self.counts['source_forward_count'] += 1
            return {p: SimpleNamespace(key=capacity.logical_key(episode),
                physical_tensor_bytes=prepared.sizes[p][episode['episode_id']], positions=(episode['source_start'], episode['source_start']+3),
                qkv_metadata=dict(source_user=episode['source_user'], source_id=episode['source_id'])) for p in quality.POLICIES}
        def decode_entry(self, resident): return SimpleNamespace(**resident.__dict__, resident=resident)
        def validate_decoded(self, resident, view): assert view.resident is resident
        def greedy(self, context, episode):
            self.greedy_calls.append((episode['episode_id'], context.hits is not None))
            if context.hits:
                self.reuse_audits.append(dict(hit=True))
                return [7, 8], 'different greedy continuation'
            return [2], 'same'
        def teacher_logits(self, context, episode, canonical):
            assert canonical == [2]
            self.teacher_calls.append((episode['episode_id'], context.hits is not None))
            if context.hits: self.reuse_audits.append(dict(hit=True))
            self.counts['teacher_forced_forward_count'] += 1
            return torch.ones(1, 8)
    model = ToyBackend(); builds = []; cases = []; hits = []
    caches = runtime.build_caches(model, prepared, builds)
    assert model.counts['source_forward_count'] == 32 and len(builds) == 6
    runtime.evaluate_targets(model, prepared, caches, cases, hits)
    assert len(cases) == 32*7 and len(hits) == len(model.greedy_calls) == 32*6
    assert len(model.teacher_calls) == 32*7 and model.counts['source_forward_count'] == 32
    assert sum(hit for _, hit in model.greedy_calls) == 77
    assert all(r['teacher_forced_canonical_token_ids'] == [2] for r in cases)
    assert all(not r['generation_fidelity']['exact_generation'] for r in cases if r['hit'])
    for name in caches:
        assert sum(r['hit'] for r in hits if r['condition'] == name) == prepared.expected[name]['summary']['target_hits']
    assert all(not r['target_admission_performed'] for r in hits)
    assert all(model.counts[k] == 0 for k in quality.SAFETY)


def test_real_mixed_projection_hit_skips_native_rows_qkv_and_miss_is_native(monkeypatch):
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear
    from semcache.cache.cache_entry import CacheEntry
    from semcache.semantic.subsequence import Subsequence
    from semcache.semantic.hit_selection import CacheHit
    from semcache.edgelora.mixed_projection import mixed_projection_path
    modules = {}
    for role in 'qkv':
        kwargs = dict(config=LoraConfig(r=2, lora_alpha=2)) if 'config' in inspect.signature(Linear).parameters else {}
        modules[role] = Linear(torch.nn.Linear(4, 4), 'audit', r=2, lora_alpha=2, **kwargs).eval()
    adapter = SimpleNamespace(layers=[0], projection_modules=lambda l: modules)
    source = tuple(torch.full((1, 3, 4), 20+i) for i in range(3))
    entry = CacheEntry.from_tensors(1, (5, 6, 7), (0, 3), {0: source})
    entry.qkv_metadata = dict(component_scope='total_qkv')
    hit = CacheHit(Subsequence((5, 6, 7), 1, 4), entry)
    inputs = torch.randn(1, 6, 4)
    with mixed_projection_path(adapter, 'audit', [hit], 6) as audit:
        outputs = {r: m(inputs) for r, m in modules.items()}
    proof = runtime.previous.projection_audit(audit, 1, 6, [hit])
    assert proof['hit_positions'] == [1, 2, 3]
    for i, role in enumerate('qkv'):
        assert torch.equal(outputs[role][:, 1:4], source[i])
        assert audit.records[0, role]['native_positions'] == [0, 4, 5]
    # C8-B MISS dispatch calls the existing native FULL forward, not mixed reuse.
    from semcache.experiments.cachegen.c6b2_runtime import Backend as Native
    observed = []
    monkeypatch.setattr(Native, 'forward', lambda self, ids, user, hits, transport, **kw: observed.append(hits) or ('full', None))
    backend = runtime.Backend.__new__(runtime.Backend)
    assert backend.forward([1], 'audit', None, False) == ('full', None)
    assert observed == [None]
    with pytest.raises(ValueError): backend.forward([1], 'audit', None, True)


def test_miss_generation_and_teacher_checks_remain_strict():
    official = dict(generated_token_ids=[1, 2], generated_text='same')
    assert quality.generation_check([1, 2], 'same', official, False)['exact_generation']
    for tokens, text in (([1, 3], 'same'), ([1, 2], 'changed')):
        with pytest.raises(ValueError, match='MISS'): quality.generation_check(tokens, text, official, False)
    logits = torch.randn(3, 8)
    assert quality.teacher_metrics(logits, logits, False)['max_abs_logit_difference'] == 0
    with pytest.raises(ValueError, match='MISS'): quality.teacher_metrics(logits, logits+0.1, False)
    assert quality.teacher_metrics(logits, logits+0.1, True)['max_abs_logit_difference'] > 0


def synthetic_cases():
    official = [dict(episode_id=str(i), generated_text='response '+str(i), generated_token_ids=[i, 2], reference_text='response '+str(i)) for i in range(32)]
    cases = [dict(full, condition=quality.REFERENCE, hit=False,
        generation_fidelity=quality.generation_check(full['generated_token_ids'], full['generated_text'], full, False),
        teacher_forced=quality.teacher_metrics(torch.ones(1, 8), torch.ones(1, 8), False)) for full in official]
    for b in quality.BUDGETS:
        for p, hits in zip(quality.POLICIES, quality.EXPECTED_HITS[b]):
            for i, full in enumerate(official):
                cases.append(dict(full, condition=quality.condition(b, p), hit=i >= 32-hits,
                    generation_fidelity=quality.generation_check(full['generated_token_ids'], full['generated_text'], full, i >= 32-hits),
                    teacher_forced=quality.teacher_metrics(torch.ones(1, 8), torch.ones(1, 8), False)))
    return cases, official


def test_split_aggregation_pairwise_and_review_rule():
    cases, official = synthetic_cases()
    summary, comparisons = quality.results(cases, official)
    assert len(summary['conditions']) == len(comparisons) == 6
    assert summary['reference']['cases'] == 32 and summary['reference']['target_hits'] == 0
    full_q = next(r for r in summary['conditions'] if r['condition'] == 'B8_Q24_KV_COMP')
    assert full_q['subsets']['HIT']['cases'] == 32 and full_q['subsets']['MISS']['cases'] == 0
    assert full_q['subsets']['MISS']['corpus_bleu'] is None
    assert summary['recommendation'] == 'C8_B_SUPPORTS_Q24_KV_FOR_C9'
    delta = next(r for r in comparisons if r['comparison'] == 'Q24_KV_COMP - KV_COMP' and r['budget_raw_entry_equivalent'] == 8)
    assert delta['target_hits'] == 14 and delta['corpus_bleu'] == 0
    delta.update(corpus_bleu=-1, exact_generation_match_rate=-.1, teacher_forced_mean_kl=.01, teacher_forced_top1_agreement=-.1)
    assert quality.review_quality(comparisons)[0] == 'C8_B_REQUIRES_FURTHER_QUALITY_REVIEW'


@pytest.mark.parametrize('role', ('q', 'kv'))
def test_runtime_fitting_is_forbidden(role):
    counts = runtime.previous.runtime_counts()
    def fit(): pass
    with pytest.raises(RuntimeError):
        with runtime.previous.forbid_fitting(counts, role): fit()
    assert counts['runtime_q_profile_fit_count' if role == 'q' else 'runtime_storage_kv_cdf_fit_count'] == 1


def test_partial_invalid_reports_and_complete_reports_no_automatic_freeze(tmp_path):
    failed = tmp_path/'failed'; failed.mkdir()
    manifest = dict(status='INVALID', failure_stage='target_quality_replay', failure_type='ValueError', failure_message='root cause')
    quality.write_reports(failed, [], [], [], manifest)
    assert freeze.read(failed/'manifest.json')['failure_message'] == 'root cause'
    assert 'NOT_COMPLETED' in (failed/'summary.md').read_text()
    assert len(manifest['output_hashes']) == 8
    complete = tmp_path/'complete'; complete.mkdir()
    cases, official = synthetic_cases(); summary, deltas = quality.results(cases, official)
    quality.write_reports(complete, cases, [], [], dict(status='COMPLETE', system_policy_frozen=False), summary, deltas)
    assert freeze.read(complete/'manifest.json')['system_policy_frozen'] is False
    assert not (complete/'freeze_decision.json').exists()
    assert quality.LABEL in (complete/'summary.md').read_text()
