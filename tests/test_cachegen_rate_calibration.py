"""Tiny CPU synthetic tests only; the real calibration entry point is never run."""
from dataclasses import replace
import importlib.util
from unittest.mock import patch

import pytest

from semcache.experiments.cachegen.b2 import format as fmt
from semcache.experiments.cachegen.common import DIMENSIONS, digest
from semcache.experiments.cachegen.c15c import rate_calibration as rc, rate_storage as rs
from semcache.experiments.cachegen.c15c.policy import UniformKVPolicy, validate_candidates, DEFAULT_CANDIDATE_BINS
from semcache.experiments.cachegen.shared import core

torch_available = pytest.mark.skipif(not importlib.util.find_spec('torch'), reason='Existing torch needed for tiny CPU tensors')
SCOPE = digest({'tiny_calibration': True})


def test_uniform_K_and_V_constant_all_layers_and_different_role_bins():
    p = UniformKVPolicy(24, 18)
    assert p.name == 'UNIFORM_K24_V18'
    assert p.profile() == dict(K=[24]*32, V=[18]*32)
    for layer in range(32):
        assert p.bins_for(layer, 'K') == 24
        assert p.bins_for(layer, 'V') == 18


@pytest.mark.parametrize('bins', [0, 2, 7, 9, 129, 130, 256, True, 16.0, '16'])
def test_candidate_bin_validation(bins):
    with pytest.raises(ValueError): validate_candidates([bins])
    with pytest.raises(ValueError): UniformKVPolicy(bins, 16)
    with pytest.raises(ValueError): UniformKVPolicy(16, bins)


def test_candidate_list_and_safe_shifted_int8_bounds():
    assert DEFAULT_CANDIDATE_BINS == tuple(range(8, 33, 2))
    assert validate_candidates([32, 8, 16]) == (8, 16, 32)
    assert validate_candidates([4, 128]) == (4, 128)
    for values in ([], [8, 8]):
        with pytest.raises(ValueError): validate_candidates(values)


@torch_available
@pytest.mark.parametrize('bins', [8, 18, 32, 128])
def test_uniform_same_released_formula_axis_dtype_rounding(bins):
    import torch
    from semcache.experiments.cachegen.c15c.reference import reference_functions
    quant, dequant = reference_functions()
    rng = torch.Generator().manual_seed(9)
    x = torch.randn(32, 3, 4, generator=rng).half()
    x[:, 0] = torch.tensor([-1, 1, 0, .5]).half()
    p = UniformKVPolicy(bins, bins)
    for role in ('K', 'V'):
        actual = p.quantize(x, role)
        q, maximum = quant(torch.full((32,), float(bins)), x)
        assert torch.equal(actual.symbols, q)
        assert torch.equal(actual.maxabs, maximum)
        assert torch.equal(actual.dequantize(), dequant(q.float(), torch.full((32,), float(bins)), maximum))
        assert actual.maxabs.dtype == torch.float16 and actual.bins.dtype == torch.float32


def role_profile(role, seed):
    return rs.fit_role(role, [[seed+i for i in range(255)], [seed+2*i for i in range(255)]], SCOPE)


def test_independent_profile_composition_roles_scope_order_and_serialization():
    k, v = role_profile('K', 1), role_profile('V', 17)
    p = rs.compose_profile(k, v)
    assert p.mode == fmt.MODES[1] and p.cdfs == k.cdfs+v.cdfs
    assert fmt.Profile.from_bytes(p.to_bytes(), p.sha256) == p
    assert len(p.to_bytes()) == 4108
    assert rs.profile_attribution() == dict(role_cdf_bytes=2048, shared_profile_header_bytes=12, full_profile_bytes=4108)
    for a, b in ((v, k), (k, replace(v, scope_hash='0'*64))):
        with pytest.raises(ValueError): rs.compose_profile(a, b)


def test_actual_frame_matches_normal_B2_encode_without_additional_coder_pass():
    p = rs.compose_profile(role_profile('K', 1), role_profile('V', 17))
    shape = (32, 3, 2)
    domains = (bytes(i % 255 for i in range(192)), bytes((13*i) % 255 for i in range(192)))
    streams = fmt.streams_from_domains(p.mode, domains, shape, expected_tokens=3)
    metadata = bytes(768)
    payloads = tuple(core.arithmetic_encode(s, cdf) for s, cdf in zip(streams, p.cdfs))
    with patch.object(core, 'arithmetic_encode', side_effect=AssertionError('Framing must not re-encode')):
        framed = fmt.frame_payloads(p, payloads, metadata, shape, expected_tokens=3)
    assert framed == fmt.encode(p, streams, metadata, shape, expected_tokens=3)
    assert fmt.decode(framed, p, expected_tokens=3) == (streams, metadata, shape)
    for malformed in ((b'',)*4, payloads[:3]):
        with pytest.raises(ValueError): fmt.frame_payloads(p, malformed, metadata, shape, expected_tokens=3)


def test_physical_accounting_once_per_workload_and_role_profile_attribution():
    a = rs.physical_accounting([100, 120], [200, 220], [(32, 3, 2)]*2, profile_bytes=4108)
    assert a['k_payload_bytes'] == 220 and a['v_payload_bytes'] == 420
    assert a['scale_maxabs_metadata_bytes'] == 2*768
    assert a['local_transform_metadata_bytes'] == 2*99
    assert a['local_metadata_bytes'] == 2*867
    assert a['per_block_bitstream_bytes'] == [1167, 1207]
    assert a['total_physical_bytes'] == 220+420+2*867+4108
    assert a['original_fp16_kv_bytes'] == 2*(32*3*2*4)
    assert a['compression_ratio'] == a['original_fp16_kv_bytes']/a['total_physical_bytes']
    with pytest.raises(ValueError): rs.physical_accounting([100], [], [(32, 3, 2)], profile_bytes=4108)


def selection_inputs(k_values, v_values, k_target=100, v_target=100):
    q = dict(k_payload_bytes=k_target, v_payload_bytes=v_target, local_metadata_bytes=867,
        global_profile_bytes=4108, total_physical_bytes=k_target+v_target+867+4108, original_fp16_kv_bytes=983040)
    rows = [dict(role=role, bins=8+2*i, payload_bytes=n, reconstruction_metrics={'never_read': float('nan')})
            for role, values in (('K', k_values), ('V', v_values)) for i, n in enumerate(values)]
    return q, rows


def test_role_selection_never_compensates_with_other_role_or_uses_quality():
    q, rows = selection_inputs([99, 80, 130], [102, 70, 120])
    result = rc.select_policy(q, rows)
    assert result['primary_policy'] == 'UNIFORM_K8_V8'
    assert result['primary']['K_rate_gap_percent'] == -1
    assert result['primary']['V_rate_gap_percent'] == 2
    assert not result['both_primary_role_gaps_within_one_percent']
    # 80+120 exactly matches total but changes role allocation; secondary only.
    secondary = result['secondary_best_total_rate_pair']
    assert (secondary['K_bins'], secondary['V_bins']) == (10, 12)
    assert secondary['overall_physical_storage_gap_percent'] == 0
    assert result['secondary_pair_count'] == 9 and result['cartesian_pair_encodes'] == 0
    assert result['primary']['total_calibration_bytes'] == 99+102+867+4108


def test_brackets_use_storage_order_not_bin_order_and_exact_matches_separate():
    q, rows = selection_inputs([105, 96, 102, 99], [100, 90, 110, 100])
    r = rc.select_policy(q, list(reversed(rows)))
    assert r['brackets']['K']['lower']['bins'] == 14
    assert r['brackets']['K']['upper']['bins'] == 12
    assert r['brackets']['K']['nearest_absolute']['bins'] == 14
    assert r['brackets']['V']['exact_match_bins'] == [8, 14]
    assert r['primary']['V_bins'] == 8
    assert r['both_primary_role_gaps_within_one_percent']
    q, rows = selection_inputs([98, 102], [110, 120])
    r = rc.select_policy(q, rows)
    assert r['primary']['K_bins'] == 8  # absolute tie chooses smaller bin count
    assert r['brackets']['V']['lower'] is None


def test_selection_invalid_targets_candidates_and_accounting():
    q, rows = selection_inputs([90], [110])
    with pytest.raises(ValueError): rc.select_policy(dict(q, k_payload_bytes=0), rows)
    with pytest.raises(ValueError): rc.select_policy(q, rows[:1])
    with pytest.raises(ValueError): rc.select_policy(dict(q, total_physical_bytes=1), rows)
    with pytest.raises(ValueError): rc.select_policy(q, rows+rows[:1])


def test_calibration_partition_filtered_before_evaluation_details_are_read():
    class PoisonEvaluation(dict):
        def __getitem__(self, key):
            if key != 'partition': raise AssertionError('Evaluation details accessed')
            return 'evaluation'
    sample = dict(source_id='cal_query')
    block = dict(block_id='cal', partition='calibration', dataset='snips', token_group_size=3,
        start_position=0, query_token_ids=[2, 3, 4], query_id='cal_query', source_group_id='cal_query',
        absolute_positions=[0, 1, 2], token_ids=[2, 3, 4], model_revision='model', tokenizer_revision='tokenizer', **DIMENSIONS)
    capture = dict(status='CAPTURED', model_config=dict(name='facebook/opt-2.7b', dtype='float16'),
        model_metadata=dict(resolved_model_revision='model', resolved_tokenizer_revision='tokenizer'),
        scope='base_raw_unscaled_linear_projection; no LoRA adapter', blocks=[PoisonEvaluation(), block],
        sampling=dict(snips=dict(calibration=[sample], hashes=dict(calibration=digest([sample])))))
    assert rc.calibration_blocks(capture) == [block]
    with pytest.raises(ValueError): rc.calibration_blocks(dict(capture, blocks=[PoisonEvaluation()]))


@torch_available
def test_tiny_engine_fresh_CDFs_no_cartesian_encodes_same_blocks_and_exact_metrics():
    import torch
    rng = torch.Generator().manual_seed(12)
    fixtures = [{r: torch.randn(32, 3, 2, generator=rng).half() for r in ('k', 'v')} for _ in range(2)]
    blocks = [dict(block_id=str(i), partition='calibration', token_group_size=3) for i in range(2)]
    loaded = []
    def loader(block):
        assert block['partition'] == 'calibration'
        loaded.append(block['block_id'])
        return fixtures[int(block['block_id'])]
    with patch.object(core, 'arithmetic_encode', wraps=core.arithmetic_encode) as enc, \
         patch.object(core, 'arithmetic_decode', wraps=core.arithmetic_decode) as dec:
        result = rc.calibrate(blocks, loader, device='cpu', candidates=[8, 16], scope_hash=SCOPE)
        assert enc.call_count == dec.call_count == 2*2*3*2  # blocks * roles * (QL2+2 bins) * 2 streams
    assert loaded == ['0', '1', '0', '1']
    assert len(result['candidates']) == 4
    assert result['workload']['arithmetic_encode_calls'] == 24
    assert result['ql2']['calibration_block_ids'] == ['0', '1']
    assert result['ql2_profile'].cdfs != result['matched_profile'].cdfs
    for row in (*result['candidates'], *result['ql2']['roles'].values()):
        assert row['storage_symbol_mismatch'] == 0
        assert sum(row['calibration_counts'][0]) == 2*32*2
        assert sum(row['calibration_counts'][1]) == 2*32*2*2
        assert all(v is None or __import__('math').isfinite(v) for v in row['reconstruction_metrics'].values())
    selected = result['selection']['primary']['physical_accounting']
    assert selected['total_physical_bytes'] == result['selection']['primary']['total_calibration_bytes']
    # Fit is candidate-specific, not the old calibration-CDF compatibility path.
    own = [r for r in result['candidates'] if r['role'] == 'K' and r['bins'] == result['selection']['primary']['K_bins']][0]
    assert result['matched_profile'].cdfs[:2] == tuple(core.cdf_from_counts(h) for h in own['calibration_counts'])
    with pytest.raises(ValueError, match='calibration'):
        rc.calibrate([dict(blocks[0], partition='evaluation')], lambda b: pytest.fail('Evaluation loaded'),
                     device='cpu', candidates=[8], scope_hash=SCOPE)


@torch_available
def test_zero_observation_report_and_fail_closed_no_repair():
    import torch
    e = UniformKVPolicy(8, 8).quantize(torch.zeros(32, 3, 2).half(), 'K')
    report = rc.observe(e, 'tiny-zero', 'fit')
    assert report['zero_maxabs_vector_count'] == 96
    assert sum(report['zero_vector_symbol_histogram'].values()) == 192
    broken = replace(e, symbols=torch.full_like(e.symbols, -128))
    with pytest.raises(rc.ObservationError) as error: rc.observe(broken, 'tiny-zero', 'fit')
    assert error.value.report['invalid_symbol_count'] == 192
    assert error.value.report['zero_maxabs_vector_count'] == 96
    with pytest.raises(ValueError): rc._finite_metrics({k: float('nan') for k in (*rc.SUM_FIELDS, 'max_abs_error')})


def test_workload_default_no_169_pair_encodes():
    w = rc.workload_shape(5, DEFAULT_CANDIDATE_BINS)
    assert w['K_candidates'] == w['V_candidates'] == 13
    assert w['role_distribution_encode_passes'] == 28
    assert w['arithmetic_encode_calls'] == 56*5
    assert w['fixture_loads'] == 10 and w['cartesian_pair_encodes'] == 0
