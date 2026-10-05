"""Actual tiny CPU OPT attention/mixed path; literal test CDFs, no model download."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semcache.experiments.cachegen import c7b2a_q_causal_audit as audit
from semcache.experiments.cachegen import c7b2_runtime as runtime
from semcache.experiments.cachegen import c7b_q_profiles as qcodec
from semcache.edgelora.mixed_projection import mixed_projection_path
from semcache.semantic.hit_selection import CacheHit
from semcache.semantic.subsequence import Subsequence
from semcache.models.model_adapter import OPTModelAdapter
from semcache.models.lora_fixtures import create_controlled_users
from test_lora import tiny_base
from test_cachegen_c7b_q_calibration import backend


def episodes():
    return [dict(episode_id=f'frozen-{i}', history_depth=(i+2)%4+1) for i in range(32)]


@pytest.fixture(scope='module')
def decoded(backend):
    q = torch.linspace(-2.13, 1.87, 32*3*16).reshape(32, 3, 16).half()
    cdf = tuple(round(65536*i/255) for i in range(256))
    profiles = {label: qcodec.QProfile(dict(bins=bins, layers=32, tokens=3, hidden=16,
        transform=qcodec.TRANSFORM), (cdf, cdf), backend) for label, bins in (('Q24', 24), ('Q32', 32))}
    counts = runtime.runtime_counts()
    decoded = {}
    for label, profile in profiles.items():
        with runtime.forbid_fitting(counts, 'q'):
            frame, _ = qcodec.encode(q, profile, backend)
            decoded[label] = qcodec.decode(frame, profile, backend)
    audit.safety(counts)
    return q, decoded


@pytest.fixture(scope='module')
def traced(decoded):
    model, _ = create_controlled_users(tiny_base())
    model.requires_grad_(False)
    adapter = OPTModelAdapter(model)
    raw, reconstructed = decoded
    k = torch.linspace(-1, 3, 3*16).reshape(1, 3, 16)
    v = torch.linspace(3, -2, 3*16).reshape(1, 3, 16)
    native = {l: (raw[l:l+1], k, v) for l in range(2)}
    entry = SimpleNamespace(tensors=native, token_ids=(4, 5, 6), qkv_metadata=dict(component_scope='total_qkv'))
    context = SimpleNamespace(hits=[CacheHit(Subsequence(entry.token_ids, 2, 5), entry)], entry=entry)
    contexts = {'KV_BASELINE': context}
    for label in ('Q24', 'Q32'):
        view = SimpleNamespace(**vars(entry))
        view.tensors = {l: (reconstructed[label][l:l+1], k, v) for l in range(2)}
        contexts[label] = SimpleNamespace(hits=[CacheHit(context.hits[0].window, view)])
    contexts['Q_ZERO_COUNTERFACTUAL'] = audit.zero_context(context)
    result = {}
    for label, c in contexts.items():
        with torch.inference_mode(), mixed_projection_path(adapter, 'user_a', c.hits, 8) as reuse:
            with audit.capture_mixed(adapter, 8) as trace:
                output = model(input_ids=torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]]), use_cache=False)
        result[label] = SimpleNamespace(trace=trace, logits=output.logits[0, 5:].detach(),
            reuse=runtime.projection_audit(reuse, 2, 8, c.hits), context=c)
    return result, adapter, native


def test_quality_blind_first_per_depth():
    e = episodes()
    assert audit.select_cases(e) == [2, 3, 0, 1]
    assert [e[i]['history_depth'] for i in audit.select_cases(e)] == [1, 2, 3, 4]
    for row in e:
        row.update(quality_score=999, cross_user=True, q_distortion=-1)
    assert audit.select_cases(e) == [2, 3, 0, 1]
    with pytest.raises(ValueError): audit.select_cases(e[:4])
    e[2]['history_depth'] = e[6]['history_depth'] = e[10]['history_depth'] = e[14]['history_depth'] = 3
    e[18]['history_depth'] = e[22]['history_depth'] = e[26]['history_depth'] = e[30]['history_depth'] = 3
    with pytest.raises(ValueError): audit.select_cases(e)


@pytest.mark.parametrize('label', ('Q24', 'Q32'))
def test_actual_q_codec_nonzero_distortion(decoded, label):
    raw, values = decoded
    differences = [audit.difference(raw[l:l+1], values[label][l:l+1], fingerprints=True) for l in range(32)]
    assert any(not d['exact_equal'] and d['mse'] > 0 for d in differences)
    assert all(d['raw_sha256'] == audit.fingerprint(raw[l:l+1]) for l, d in enumerate(differences))
    audit.validate_distortion({'Q24': differences, 'Q32': differences})
    with pytest.raises(ValueError, match='INVALID'):
        audit.validate_distortion({'Q24': [dict(exact_equal=True)], 'Q32': differences})


@pytest.mark.parametrize('label', audit.MODES)
def test_actual_injection_fresh_q_kv_unchanged_and_native_skips(traced, label):
    modes, _, _ = traced
    current, base = modes[label], modes['KV_BASELINE']
    result = audit.injection_check(current.trace, base.trace, current.context.hits, current.reuse, base.reuse)
    assert result['q_injection_verified'] and result['fresh_q_rows_equal']
    assert result['k_rows_equal'] and result['v_rows_equal'] and result['hit_mask_equal']
    for record in current.reuse['per_layer']:
        for role in 'qkv':
            assert record[role+'_reused_projection_rows'] == record[role+'_native_projection_rows_skipped'] == 3
    if label == audit.MODES[-1]:
        assert all(torch.count_nonzero(t['q'][:, 2:5]) == 0 for t in current.trace.values())


def test_zero_changes_actual_hit_attention_but_not_fresh_attention(traced):
    modes, _, original = traced
    base, zero = modes['KV_BASELINE'], modes['Q_ZERO_COUNTERFACTUAL']
    result = audit.internal_effect(base.trace, zero.trace, base.reuse['hit_positions'])
    assert result['hit_row_causal_effect_verified']
    assert any(r['hit_attention']['max_absolute_error'] > 0 for r in result['per_layer'])
    assert result['fresh_attention_unchanged'] and result['fresh_attention_numerically_negligible']
    for l in original:
        assert torch.count_nonzero(original[l][0]) > 0
        for i in (1, 2):
            assert zero.context.hits[0].entry.tensors[l][i] is original[l][i]
    assert zero.context.entry is base.context.entry
    assert zero.context.hits[0].window is base.context.hits[0].window
    with pytest.raises(ValueError, match='no hit-row'):
        audit.internal_effect(base.trace, base.trace, base.reuse['hit_positions'])


def test_actual_continuation_measured_not_assumed(traced):
    modes, _, _ = traced
    base, zero = modes['KV_BASELINE'], modes['Q_ZERO_COUNTERFACTUAL']
    result = audit.continuation_effect(base.logits, zero.logits)
    assert result['exact_equal'] and result['numerically_negligible']
    assert result['max_abs_logit_difference'] == result['mean_abs_logit_difference'] == 0
    assert result['mean_kl'] == 0 and result['top1_agreement'] == result['top5_agreement'] == 1
    changed = zero.logits.clone(); changed[0, 0] += 0.1
    assert not audit.continuation_effect(base.logits, changed)['numerically_negligible']


@pytest.mark.parametrize('role', ('q', 'k', 'v', 'fresh_q', 'mask'))
def test_contradictory_injection_fails_closed(traced, role):
    modes, _, _ = traced
    base = modes['KV_BASELINE']; trace = copy.deepcopy(base.trace); reuse = copy.deepcopy(base.reuse)
    if role == 'mask':
        reuse['hit_positions'] = [1, 2, 3]
    elif role == 'fresh_q':
        trace[0]['q'][:, 0] += 1
    else:
        trace[0][role][:, 2:5] += 1
    with pytest.raises(ValueError, match='INVALID'):
        audit.injection_check(trace, base.trace, base.context.hits, reuse, base.reuse)


def test_capture_restores_wrappers_and_hooks_on_error(traced):
    _, adapter, _ = traced
    with pytest.raises(RuntimeError, match='intentional'):
        with mixed_projection_path(adapter, 'user_a', [], 8):
            with audit.capture_mixed(adapter, 8):
                raise RuntimeError('intentional')
    for l, block in enumerate(adapter.layers):
        assert not block._forward_hooks and not block.self_attn.out_proj._forward_hooks
        assert all('forward' not in m.__dict__ and not m._forward_hooks for m in adapter.projection_modules(l).values())


def test_zero_decoded_delegating_view_does_not_copy_or_mutate_resident():
    # C2 views delegate metadata via __getattr__; copy.copy on those objects can
    # recurse during reconstruction. The audit instead wraps the actual view.
    class View:
        def __init__(self):
            self.resident = SimpleNamespace(token_ids=(1, 2, 3))
            self.tensors = {0: (torch.ones(1, 3, 4),)*3}
        def __getattr__(self, name): return getattr(self.resident, name)
    original = View(); context = SimpleNamespace(hits=[CacheHit(Subsequence((1, 2, 3), 1, 4), original)])
    zero = audit.zero_context(context)
    assert zero.hits[0].entry.resident is original.resident
    assert zero.hits[0].entry.token_ids == (1, 2, 3)
    assert torch.count_nonzero(original.tensors[0][0]) == 12


@pytest.mark.parametrize('counter', audit.SAFETY_COUNTERS)
def test_no_fitting_transport_or_freeze(counter, tmp_path):
    counts = runtime.runtime_counts(); audit.safety(counts)
    counts[counter] = 1
    with pytest.raises(ValueError, match='forbidden'): audit.safety(counts)
    for role in ('q', 'kv'):
        def fit_profile(): return None
        with pytest.raises(RuntimeError, match='forbidden'):
            with runtime.forbid_fitting(runtime.runtime_counts(), role): fit_profile()
    reports = {name: dict(cases=[]) for name in ('selected_cases', 'q_distortion', 'injection_audit',
        'internal_causal_effect', 'continuation_causal_effect')}
    manifest = dict(status='INVALID', greedy_executed=False, q_profile_frozen=False, selected_q_candidate=None)
    audit.write_reports(tmp_path, reports, manifest)
    assert len(list(tmp_path.iterdir())) == 7 and not (tmp_path/'freeze_decision.json').exists()
    assert json.loads((tmp_path/'manifest.json').read_text())['selected_q_candidate'] is None
    assert 'Not established' in (tmp_path/'summary.md').read_text()


def test_fingerprint_bit_identity_and_finite_guards():
    assert audit.fingerprint(torch.tensor([0.])) != audit.fingerprint(torch.tensor([-0.]))
    for invalid in (float('nan'), float('inf')):
        with pytest.raises(ValueError): audit.fingerprint(torch.tensor([invalid]))
    with pytest.raises(ValueError): audit.continuation_effect(torch.ones(1, 6), torch.ones(2, 6))


def test_traced_teacher_uses_identical_established_continuation(traced, monkeypatch):
    modes, adapter, _ = traced
    backend = audit.AuditBackend.__new__(audit.AuditBackend)
    backend.model = adapter.model; backend.adapter = adapter; backend.args = SimpleNamespace(device='cpu')
    backend.rows = [dict(token_ids=[2, 3, 4, 5, 6, 7, 8, 9])]
    backend.counts = runtime.runtime_counts(); backend.reuse_audits = []
    def forbidden_cuda(*args, **kwargs): raise AssertionError('CPU fixture attempted CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden_cuda)
    canonical = [10, 11, 12]
    for mode in audit.MODES:
        context = modes[mode].context; context.transport = False
        logits, trace, reuse = backend.traced_teacher(context, dict(target_index=0, target_user='user_a'), canonical)
        assert logits.shape == (3, 256) and trace[0]['q'].shape == (1, 10, 16)
        assert reuse['hit_positions'] == [2, 3, 4]
        assert not backend.tracing and not hasattr(backend, 'last_trace')
    assert canonical == [10, 11, 12] and backend.counts['teacher_forced_forward_count'] == 4
    audit.safety(backend.counts)
    # The source capture is also the actual native TOTAL Q before encoding.
    backend.source_start = 2
    _, projected = backend.forward([2, 3, 4, 5, 6, 7, 8, 9], 'user_a', [], False, capture=True)
    for layer in range(2):
        assert backend.raw_source_q[layer].shape == (1, 3, 16)
        assert audit.fingerprint(backend.raw_source_q[layer]) == audit.fingerprint(projected[layer]['q'][:, 2:5])


def b2_evidence(tmp_path):
    root = tmp_path/'b2'; root.mkdir()
    calibration = tmp_path/'calibration'; calibration.mkdir()
    (calibration/'manifest.json').write_text('{}')
    eps = episodes(); canonical = [10, 11]
    rows = [dict(e, mode=mode, logical_event_hash=audit.b0.digest(e),
        teacher_forced_canonical_token_ids=json.dumps(canonical),
        teacher_forced_canonical_sha256=audit.b0.digest(canonical)) for e in eps for mode in runtime.MODES]
    import csv
    with (root/'per_case.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    gate = audit.gate
    m = dict(stage='C7-B2', status='COMPLETE', baseline_matches_c6b3_2=True,
        runtime_q_profile_fit_count=0, runtime_storage_kv_cdf_fit_count=0,
        transport_compression_enabled=False, executed_modes=list(runtime.MODES),
        model=audit.b0.MODEL, model_revision=audit.b0.REVISION, tokenizer_revision=audit.b0.REVISION,
        prompt_version=audit.b0.VERSION, adapter_hashes=audit.b0.WEIGHTS, q_profile_hashes=gate.Q_SHAS,
        kv_profile_sha256=gate.PROFILE_SHA, frozen32_selection_sha256=gate.b3.SELECTION_SHA,
        adapter_freeze_decision_sha256=gate.FREEZE_SHA, c7b1_manifest_sha256=audit.b0.sha(calibration/'manifest.json'),
        source_sha256=gate.SOURCE_SHA, semantic_sha256=gate.SEMANTIC_SHA,
        physical_safety_contract=gate.CONTRACT, cached_payload='TOTAL_QKV',
        q_profile_frozen=False, selected_q_candidate=None, training_performed=False,
        counters=runtime.runtime_counts(), input_hashes={}, output_hashes={'per_case.csv': audit.b0.sha(root/'per_case.csv')})
    (root/'manifest.json').write_text(json.dumps(m))
    return root, SimpleNamespace(episodes=eps, official=[dict(generated_token_ids=canonical) for _ in eps], input_hashes={}), \
        SimpleNamespace(q_calibration_root=calibration), m


@pytest.mark.parametrize('field,wrong', [('stage', 'C7-B1'), ('status', 'INVALID'),
    ('baseline_matches_c6b3_2', False), ('runtime_q_profile_fit_count', 1),
    ('runtime_storage_kv_cdf_fit_count', 1), ('q_profile_hashes', {}), ('kv_profile_sha256', 'wrong')])
def test_b2_prerequisite_fails_closed(tmp_path, field, wrong):
    root, prepared, args, manifest = b2_evidence(tmp_path)
    assert audit.verify_b2(root, prepared, args)
    manifest[field] = wrong; (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): audit.verify_b2(root, prepared, args)


def test_b2_hash_and_canonical_continuation_binding(tmp_path):
    root, prepared, args, manifest = b2_evidence(tmp_path)
    path = root/'per_case.csv'; original = path.read_bytes()
    path.write_bytes(original+b'\n')
    with pytest.raises(ValueError, match='hash mismatch'): audit.verify_b2(root, prepared, args)
    path.write_bytes(original)
    prepared.official[0]['generated_token_ids'] = [99]
    with pytest.raises(ValueError, match='continuation changed'): audit.verify_b2(root, prepared, args)
