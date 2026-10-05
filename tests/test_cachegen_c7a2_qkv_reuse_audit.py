"""Corrected C7-A2 contract tests. No network, models or CUDA execution."""
import copy
import json
from pathlib import Path

import pytest

from semcache.experiments.cachegen import c7a2_qkv_reuse_audit as a2


@pytest.fixture(scope='module')
def probes():
    raw, trace = a2.fixture()
    callback, callback_trace = a2.fixture(callback=True)
    view, view_trace = a2.fixture(view_probe=True)
    counter = a2.counterfactual(raw, 3)
    return raw, callback, view, counter, trace+callback_trace+view_trace


def test_subsequence_hit_unit(probes):
    raw = probes[0]
    assert raw['hit_unit'] == 'w3_token_subsequence'
    assert raw['cached_payload_contract'] == 'TOTAL_QKV'
    assert raw['physical_safety_contract'] == 'PHYSICAL_EXACT_W3_WITHIN_SEMANTIC_CLUSTER'
    assert raw['selected_hit_count'] == 1
    assert raw['cluster_and_exact_token_isolation']
    assert raw['target_hit_positions'] == [2, 3, 4]


@pytest.mark.parametrize('role', 'qkv')
def test_all_layer_payload_values_and_fresh_computation(probes, role):
    raw = probes[0]
    records = [r for r in raw['records'] if r['role']==role]
    assert [r['layer'] for r in records] == [0, 1, 2]
    for r in records:
        assert all(r[k] for k in a2.ROLE_FIELDS)
        assert r['reused_projection_rows'] == r['native_projection_rows_skipped'] == r['hit_rows_from_cache'] == 3
        assert r['native_projection_rows'] == r['fresh_projection_rows_observed'] == 4
        assert r['fresh_rows_match'] and r['cached_and_fresh_values_distinct']
        assert r['reuse_mask'] == raw['selected_mask'] == raw['mixed_mask']


def test_q_objects_are_distinct(probes):
    rows = probes[4]
    classifications = {r['object_role'] for r in rows if r['role']=='q'}
    assert classifications == {'RESIDENT_CACHED_TOTAL_Q', 'FRESH_TARGET_Q', 'TRANSPORT_LORA_Q_DELTA'}
    delta = [r for r in rows if r['object_role']=='TRANSPORT_LORA_Q_DELTA']
    assert all('NO transport encode/decode' in r['notes'] for r in delta)


def test_transport_callback_excludes_hit_rows(probes):
    callback = probes[1]
    assert not callback['transport_codec_executed']
    for row in callback['records']:
        assert row['callback_received_only_fresh_rows']
        assert row['fresh_rows_callback_reconstructed'] == 4
        assert row['fresh_rows_transport_reconstructed'] is None
        assert row['hit_rows_from_cache'] == row['native_projection_rows_skipped'] == 3
        assert row['native_projection_rows'] == 0


def test_decoded_view_contract_preserves_q_and_reuses_qkv(probes):
    view = probes[2]
    assert view['decoded_view_q_preserved_contract_probe']
    assert all(r['injected_into_target_hit_rows'] for r in view['records'])
    # Never turn a constructed view into evidence that real compressed lookup ran.
    assert not view['real_compressed_lookup_executed']
    assert view['fixture_kind'] == 'DECODED_VIEW_CONTRACT_ONLY'


def test_real_c6_validator_rejects_changed_decoded_q():
    import torch
    from semcache.cache.cache_entry import CacheEntry
    from semcache.experiments.cachegen.c6_runtime import Storage
    entry = CacheEntry.from_tensors(1, (1, 2, 3), (0, 3),
        {0: tuple(torch.ones(1, 3, 4, dtype=torch.float16) for _ in 'qkv')})
    resident, view = a2.decoded_view_contract(entry)
    view.tensors[0] = (view.tensors[0][0]+1, *view.tensors[0][1:])
    with pytest.raises(ValueError, match='changed resident Q'):
        Storage.validate_decoded(resident, view)


def faithful_evidence():
    return dict(**a2.CONTRACT, contradictory_evidence=False,
        roles={r: dict.fromkeys(a2.ROLE_FIELDS, True) for r in 'qkv'},
        same_reuse_mask_qkv=True, raw_semcache_qkv_reuse_confirmed=True,
        storage_kv_comp_qkv_reuse_confirmed=True, transport_hit_qkv_bypass_confirmed=True,
        full_pipeline_hit_qkv_reuse_confirmed=True)


def test_faithful_requires_actual_storage_evidence(probes):
    evidence = faithful_evidence()
    assert a2.decision(evidence) == 'SEMCACHE_FAITHFUL_QKV_REUSE_CONFIRMED'
    for key in ('storage_kv_comp_qkv_reuse_confirmed', 'full_pipeline_hit_qkv_reuse_confirmed'):
        unknown = dict(evidence, **{key:None})
        assert a2.decision(unknown) == 'INCONCLUSIVE'
    raw, callback, view, counter, _ = probes
    audit = a2.aggregate(raw, callback, view, counter, dict(actual_compressed_lookup_confirmed=None, reason='missing'))
    assert audit['recommendation'] == 'INCONCLUSIVE'
    assert audit['next_stage'] == 'FURTHER_AUDIT_ONLY'
    assert audit['raw_semcache_qkv_reuse_confirmed']
    assert audit['transport_hit_qkv_bypass_confirmed']


def test_missing_layer_role_is_contradictory(probes):
    raw, callback, view, counter, _ = probes
    broken = copy.deepcopy(raw)
    broken['records'].pop()
    audit = a2.aggregate(broken, callback, view, counter, dict(actual_compressed_lookup_confirmed=None, reason='missing'))
    assert not audit['all_synthetic_projection_layers_covered']
    assert audit['contradictory_evidence']
    assert audit['recommendation'] == 'INCONCLUSIVE'


@pytest.mark.parametrize('field', ['injected_into_target_hit_rows', 'fresh_compute_skipped_on_hit_rows'])
def test_kv_only_or_recomputed_q_is_gap(field):
    evidence = faithful_evidence()
    evidence['roles']['q'][field] = False
    assert a2.decision(evidence) == 'Q_REUSE_IMPLEMENTATION_GAP'
    assert a2.NEXT[a2.decision(evidence)] == 'REPAIR_Q_REUSE_BEFORE_C7_B'


def test_contradictory_or_unknown_evidence_is_inconclusive():
    evidence = faithful_evidence()
    evidence['contradictory_evidence'] = True
    assert a2.decision(evidence) == 'INCONCLUSIVE'
    evidence = faithful_evidence()
    evidence['roles']['q']['consumed_on_hit'] = None
    assert a2.decision(evidence) == 'INCONCLUSIVE'
    evidence['hit_unit'] = 'Q hit'
    assert a2.decision(evidence) == 'INCONCLUSIVE'


def test_independent_counterfactuals(probes):
    counter = probes[3]
    assert counter['status'] == 'COMPLETE'
    for role, case in counter['cases'].items():
        assert case['changed_roles'] == {r:r==role for r in 'qkv'}
        assert case['only_corrupted_payload_role_changed']
        assert case['fresh_rows_unchanged'] and case['logical_hit_unchanged']


def test_no_gpu_or_download(monkeypatch):
    import torch
    import transformers
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('GPU/model/network access forbidden')
    monkeypatch.setattr(torch.cuda, 'init', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(torch.Tensor, 'cuda', forbidden)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    result, _ = a2.fixture(layers=1)
    assert all(r['injected_into_target_hit_rows'] for r in result['records'])


def test_outputs_provenance_and_refusal(tmp_path, probes, monkeypatch):
    raw, callback, view, counter, trace = probes
    def fixture(**kwargs):
        return (view if kwargs.get('view_probe') else callback if kwargs.get('callback') else raw), trace
    monkeypatch.setattr(a2, 'fixture', fixture)
    monkeypatch.setattr(a2, 'counterfactual', lambda *args: counter)
    output = tmp_path/'audit'
    root = Path(a2.__file__).resolve().parents[4]
    before = a2.protected_hashes(root)
    a2.run(output)
    assert before == a2.protected_hashes(root)
    assert {p.name for p in output.iterdir()} == {'qkv_reuse_audit.json', 'qkv_reuse_trace.csv',
        'qkv_counterfactual.json', 'manifest.json', 'summary.md'}
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['previous_c7a_superseded'] and manifest['protected_files_unchanged']
    assert manifest['stage'] == 'C7-A2' and not manifest['c6_modified']
    assert all(a2.sha(output/p)==value for p, value in manifest['output_hashes'].items())
    with pytest.raises(FileExistsError):
        a2.run(output)
    with pytest.raises(ValueError, match='protected'):
        a2.run(root/'results/cachegen/c7/a_q_residency_audit/new_child')


def test_unavailable_runtime_writes_unknown(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise ImportError('No CPU torch')
    monkeypatch.setattr(a2, 'fixture', fail)
    audit = a2.run(tmp_path/'audit')
    assert audit['recommendation']=='INCONCLUSIVE'
    assert audit['roles']['q']['consumed_on_hit'] is None
    assert json.loads((tmp_path/'audit/qkv_counterfactual.json').read_text())['status']=='NOT_RUN'
