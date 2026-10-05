"""CPU-only lifecycle audit tests; never import the CUDA codec or load a model."""
import json
import sys

import pytest

from semcache.experiments.cachegen import c7_q_residency as c7


def drop_evidence():
    return dict(q_written_to_resident_cache=True, q_read_on_hit=False,
                q_value_used_after_insert=False, **dict.fromkeys(c7.NEGATIVES, False))


def test_writes_reads_transport_and_fresh_separate():
    trace = c7.Trace()
    trace.resident('q', 'write', 'insert')
    trace.post_insert = trace.hit_path = True
    trace.resident('q', 'read', 'reuse')
    before = dict(trace.counts)
    trace.transport('encode')
    trace.transport('decode')
    trace.event('FRESH_TARGET_Q', 'project', 'fresh', 'native', consumed=True)
    assert trace.counts['resident_q_write_count'] == 1
    assert trace.counts['resident_q_read_count'] == 1
    assert trace.counts['resident_q_hit_read_count'] == 1
    assert trace.counts['resident_q_value_consumer_count'] == 0
    assert trace.counts['transport_q_delta_encode_count'] == 1
    assert trace.counts['transport_q_delta_decode_count'] == 1
    assert all(trace.counts[k] == v for k, v in before.items() if k.startswith('resident'))
    assert trace.rows[-1]['q_role'] == 'FRESH_TARGET_Q'


def test_accounting_is_not_semantic_use():
    trace = c7.Trace()
    trace.post_insert = True
    trace.phase = 'accounting'
    observed = c7.ObservedQKV((object(), object(), object()), trace)
    list(observed)
    assert trace.counts['resident_q_read_count'] == 1
    assert trace.counts['resident_q_value_consumer_count'] == 0
    assert trace.rows[0]['q_role'] == 'ACCOUNTING_ONLY_Q'


def test_transport_wrappers_count_only_successful_q_calls():
    trace = c7.Trace()
    # Identity callables test instrumentation only, not codec behavior.
    for role in 'qkv':
        encode, decode = trace.wrap_transport(role, lambda x: x, lambda x: x)
        assert decode(encode('delta')) == 'delta'
    assert trace.counts['transport_q_delta_encode_count'] == 1
    assert trace.counts['transport_q_delta_decode_count'] == 1
    assert trace.counts['resident_q_read_count'] == 0
    def fail(x):
        raise ValueError('failed encode')
    encode, _ = trace.wrap_transport('q', fail, lambda x: x)
    with pytest.raises(ValueError):
        encode('delta')
    assert trace.counts['transport_q_delta_encode_count'] == 1


@pytest.mark.parametrize('key', c7.NEGATIVES)
@pytest.mark.parametrize('value', [None, True])
def test_unknown_or_positive_dependency_prevents_drop(key, value):
    evidence = drop_evidence()
    evidence[key] = value
    assert c7.recommend(evidence) == 'INCONCLUSIVE'


def test_decision_schema_not_value_requirement():
    evidence = drop_evidence()
    evidence['q_schema_field_required'] = True
    evidence['q_value_required'] = False
    assert c7.recommend(evidence) == 'DROP_RESIDENT_Q'
    evidence['q_read_on_hit'] = True  # Accounting-only access is permitted.
    assert c7.recommend(evidence) == 'DROP_RESIDENT_Q'
    evidence['q_value_used_after_insert'] = True
    assert c7.recommend(evidence) == 'INCONCLUSIVE'
    assert c7.recommend(evidence, runtime_dependency=True, counterfactual_changed=True) == 'COMPRESS_RESIDENT_Q'


def test_contradictions_fail_closed():
    evidence = drop_evidence()
    assert c7.recommend(evidence, runtime_dependency=True, counterfactual_changed=True) == 'INCONCLUSIVE'
    assert c7.recommend(evidence, contradictory=True) == 'INCONCLUSIVE'
    assert c7.recommend(evidence, counterfactual_changed=True) == 'INCONCLUSIVE'
    evidence['q_written_to_resident_cache'] = None
    assert c7.recommend(evidence) == 'INCONCLUSIVE'


@pytest.mark.parametrize('key', ['q_required_for_admission', 'q_required_for_eviction', 'q_required_for_lora_recombination'])
def test_other_known_value_dependencies_prevent_drop(key):
    evidence = drop_evidence()
    evidence[key] = True
    assert c7.recommend(evidence) == 'INCONCLUSIVE'


@pytest.fixture(scope='module')
def numerical():
    return c7.counterfactuals()


def test_real_hit_and_counterfactuals(numerical):
    counter, traces = numerical
    normal, zero, none = (counter['variants'][v] for v in ('normal', 'zero', 'none'))
    assert normal['reuse_success'] and zero['reuse_success']
    assert all(row['lookup_success'] and row['logical_safety_assertion_passed'] for row in (normal, zero, none))
    assert not zero['changes_vs_normal']['selected_cache_entry']
    assert not zero['changes_vs_normal']['reconstructed_kv']
    assert zero['changes_vs_normal']['target_reuse_output']
    assert not none['reuse_success'] and none['schema_field_required']
    assert none['changes_vs_normal']['target_reuse_output'] is None
    assert not none['changes_vs_normal']['reconstructed_kv']
    audit = c7.evidence_from(counter, traces)
    assert audit['recommended_c7_path'] == 'COMPRESS_RESIDENT_Q'
    assert audit['q_value_required'] and audit['q_read_on_hit']
    counts = audit['counters']
    assert counts['resident_q_write_count'] == 1
    assert counts['resident_q_read_count'] >= 2
    assert counts['resident_q_hit_read_count'] == 1
    assert counts['resident_q_value_consumer_count'] > 0
    assert all(counts[f'resident_{r}_read_count'] >= 2 for r in 'kv')
    assert counts['transport_q_delta_encode_count'] == counts['transport_q_delta_decode_count'] == 0
    assert any(r['q_role'] == 'FRESH_TARGET_Q' for r in traces['normal'].rows)
    assert all(r['file'].endswith('edgelora/mixed_projection.py')
               for r in traces['normal'].rows if r['operation'] in ('materialize', 'consume'))


def test_fixture_requires_no_cuda_or_model(monkeypatch):
    import torch
    import transformers
    def forbidden(*args, **kwargs):
        raise AssertionError('GPU/model access forbidden')
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(torch.cuda, 'init', forbidden)
    monkeypatch.setattr(torch.Tensor, 'cuda', forbidden)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', forbidden)
    result, _ = c7.fixture()
    assert result['reuse_success']
    assert 'semcache.edgelora.cachegen_codec' not in sys.modules


def test_outputs_and_refusal(tmp_path, numerical, monkeypatch):
    monkeypatch.setattr(c7, 'counterfactuals', lambda: numerical)
    output = tmp_path/'audit'
    c7.run(output)
    assert {p.name for p in output.iterdir()} == {
        'q_residency_audit.json', 'q_residency_trace.csv', 'q_counterfactual.json', 'manifest.json', 'summary.md'}
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['stage'] == 'C7-A'
    assert not manifest['gpu_required'] and not manifest['resident_q_removed']
    assert all(c7.sha(output/name) == value for name, value in manifest['output_hashes'].items())
    with pytest.raises(FileExistsError):
        c7.run(output)


def test_runtime_unavailable_writes_inconclusive(tmp_path, monkeypatch):
    def unavailable():
        raise ImportError('CPU torch unavailable')
    monkeypatch.setattr(c7, 'counterfactuals', unavailable)
    audit = c7.run(tmp_path/'audit')
    assert audit['recommended_c7_path'] == 'INCONCLUSIVE'
    assert audit['q_value_required'] is None
    assert json.loads((tmp_path/'audit/q_counterfactual.json').read_text())['status'] == 'NOT_RUN'


def test_optional_accounting_metadata_not_double_counted(tmp_path):
    path = tmp_path/'summary.json'
    row = dict(resident_q_bytes=20, resident_kv_frame_bytes=10,
               resident_payload_bytes=30, resident_local_metadata_bytes=2,
               raw_qkv_bytes=60, raw_kv_bytes=40)
    path.write_text(json.dumps(dict(storage_accounting={m:row for m in ('STORAGE_KV_COMP', 'FULL_PIPELINE')})))
    context = c7.storage_context(path, 'DROP_RESIDENT_Q')
    what = context['modes']['FULL_PIPELINE']['what_if']
    assert what['kv_only_payload'] == 10
    assert what['bytes_removed_if_q_dropped'] == 20
    assert what['ratio_vs_original_raw_qkv'] == 10/60
    assert what['ratio_vs_original_raw_kv'] == 10/40
    assert 'what_if' not in c7.storage_context(path, 'COMPRESS_RESIDENT_Q')['modes']['FULL_PIPELINE']
