"""C1.5C CPU-only tiny synthetic tests; never invokes the real smoke command."""
from dataclasses import replace
from hashlib import sha256
import importlib.util
import os
import struct
from unittest.mock import patch

import pytest

from semcache.experiments.cachegen.b2 import format as fmt
from semcache.experiments.cachegen.c15c.policy import CACHEGEN_RELEASED_QL2 as policy
from semcache.experiments.cachegen.c15c.parity import run_parity
from semcache.experiments.cachegen.c15c import harness, reference, storage
from semcache.experiments.cachegen.shared import core


@pytest.mark.parametrize('role,layer,bins', [
    ('K', 0, 32), ('K', 9, 32), ('K', 10, 16), ('K', 19, 16), ('K', 20, 16), ('K', 31, 16),
    ('V', 0, 32), ('V', 1, 32), ('V', 2, 16), ('V', 31, 16)])
def test_literal_boundaries(role, layer, bins):
    assert policy.bins_for(layer, role) == bins


@pytest.mark.parametrize('layer,role', [(-1, 'K'), (32, 'V'), (True, 'K'), (0, 'Q')])
def test_invalid_layer_role(layer, role):
    with pytest.raises(ValueError):
        policy.bins_for(layer, role)


def uniform_profile(mode=fmt.MODES[1]):
    return fmt.Profile(mode, (core.cdf_from_counts([0]*255),)*4)


def test_existing_frozen_profile_loaded_without_refitting_and_corruption_rejected(tmp_path):
    from semcache.experiments.cachegen.common import write_json, file_hash, digest
    p = uniform_profile()
    (tmp_path/'profile.bin').write_bytes(p.to_bytes())
    counts = [[0]*255 for _ in range(4)]
    manifest = dict(status='FROZEN_BEFORE_EVALUATION', fitting_partition='calibration',
        base_contract={}, base_contract_sha256=digest({}), profiles={p.mode: dict(
            file='profile.bin', sha256=p.sha256, calibration_counts=counts,
            counts_sha256=digest(counts), serialized_bytes=len(p.to_bytes()))})
    write_json(tmp_path/'b2_profile_manifest.json', manifest)
    write_json(tmp_path/'manifest.json', dict(profile_status='FROZEN',
        profile_manifest_sha256=file_hash(tmp_path/'b2_profile_manifest.json')))
    original = (tmp_path/'profile.bin').read_bytes()
    actual, _ = storage.load_profile(tmp_path)
    assert actual == p and (tmp_path/'profile.bin').read_bytes() == original
    (tmp_path/'profile.bin').write_bytes(original+b'x')
    with pytest.raises(ValueError, match='SHA256'): storage.load_profile(tmp_path)


# Generated using the original format.py at GP_semcache dcb593abb1694958f87d704776614e92cbb14b82.
@pytest.mark.parametrize('mode,expected_sha', list(zip(fmt.MODES, (
    '09b551b8d64a6bd27ce8091d7216d95abeaed712fecfe51a39b0eeda91144bfd',
    '9db810273732ddbbc483054ec2b56566167c4a23d17b6e1ea756d59d03a10e26',
    'a02a124de95458ce418544ce80baf50b8fbc67cd3c9d6fef8a551370c293183b'))))
def test_original_B2_T10_bitstreams_byte_identical(mode, expected_sha):
    shape = (32, 10, 2)
    domains = (bytes(i % 255 for i in range(640)), bytes((i*13+29) % 255 for i in range(640)))
    maxima = struct.pack('<640f', *[i/16 for i in range(640)])
    profile = uniform_profile(mode)
    streams = fmt.streams_from_domains(mode, domains, shape)
    blob = fmt.encode(profile, streams, maxima, shape)
    assert len(blob) == 3941
    assert sha256(blob).hexdigest() == expected_sha
    decoded, actual, decoded_shape = fmt.decode(blob, profile)
    assert actual == maxima and decoded_shape == shape
    assert fmt.domains_from_streams(mode, decoded, shape) == domains


@pytest.mark.parametrize('mode', fmt.MODES)
def test_T3_geometry_explicit_opt_in_same_mod255(mode):
    shape = (32, 3, 2)
    domains = (bytes(i % 255 for i in range(192)), bytes((i*13+29) % 255 for i in range(192)))
    with pytest.raises(ValueError):
        fmt.streams_from_domains(mode, domains, shape)
    streams = fmt.streams_from_domains(mode, domains, shape, expected_tokens=3)
    assert list(map(len, streams)) == [64, 128, 64, 128]
    assert fmt.domains_from_streams(mode, streams, shape, expected_tokens=3) == domains
    if mode == fmt.MODES[1]:
        # Every nonanchor references token 0; modulus remains 255, not QL2 bins.
        assert streams[0][:2] == domains[0][:2]
        assert streams[1][:4] == bytes((domains[0][i]-domains[0][i % 2]+127) % 255 for i in range(2, 6))
    maxima = bytes(192*4)
    p = uniform_profile(mode)
    blob = fmt.encode(p, streams, maxima, shape, expected_tokens=3)
    with pytest.raises(ValueError):
        fmt.decode(blob, p)
    assert fmt.decode(blob, p, expected_tokens=3) == (streams, maxima, shape)
    with pytest.raises(ValueError):
        fmt.validate_shape((32, 4, 2), expected_tokens=4)


torch_available = pytest.mark.skipif(not importlib.util.find_spec('torch'), reason='Existing torch required; no model/GPU/download')


@torch_available
def test_reference_formula_parity_and_machine_readable_metrics():
    report = run_parity()
    assert report['status'] == 'PASS'
    assert report['source']['commit'] == reference.REVISION
    assert report['device'] == 'cpu' and report['synthetic_only']
    assert len(report['tests']) == 20
    for row in report['tests']:
        assert row['symbol_equality'] and row['passed']
        assert row['scale_max_abs_difference'] == row['reconstruction_max_abs_difference'] == 0


@torch_available
def test_pinned_AST_reference_when_checkout_is_supplied():
    path = os.environ.get('CACHEGEN_REFERENCE_REPO')
    if not path:
        pytest.skip('Optional pinned checkout not supplied; direct reference formula always tested')
    report = run_parity(path)
    assert report['status'] == 'PASS'
    assert report['source']['checkout_status'] == 'PINNED_REVISION_AND_SOURCE_HASHES_VERIFIED'
    assert report['tests'] == run_parity()['tests']
    with patch.object(reference, 'file_hash', return_value='changed'):
        with pytest.raises(ValueError, match='source differs'):
            reference.reference_functions(path)


@torch_available
@pytest.mark.parametrize('role,layer', [('K', 9), ('K', 10), ('V', 1), ('V', 2)])
def test_parity_detects_each_off_by_one_boundary(role, layer):
    original = type(policy).bins_for
    def wrong(self, index, component):
        value = original(self, index, component)
        return (16 if value == 32 else 32) if (index, component) == (layer, role) else value
    with patch.object(type(policy), 'bins_for', wrong):
        report = run_parity()
    assert report['status'] == 'FAIL'
    assert all(not r['passed'] for r in report['tests'] if r['role'] == role and r['layer'] == layer)


@torch_available
def test_shift_before_rounding_dtype_scale_axis_and_layout():
    import torch
    # C=15: shift changes ties-to-even. Signed round(.5) is 0, released q-15 is 1.
    x = torch.tensor([[[-15, 15, 0, .5], [-30, 30, 0, -1]]], dtype=torch.float16)
    e = policy.quantize(x, 'K', layer_indices=[9])
    assert e.symbols.dtype == torch.int8 and e.maxabs.dtype == torch.float16
    assert e.scale.dtype == torch.float32
    assert e.scale.tolist() == [[[1], [2]]]
    assert e.symbols.tolist() == [[[0, 30, 15, 16], [0, 30, 15, 14]]]
    assert e.signed_symbols.tolist()[0][0][-1] == 1
    assert e.reconstructed.dtype == x.dtype
    assert torch.equal(e.symbols, policy.quantize(x, 'K', layer_indices=[9]).symbols)
    heads = x.reshape(1, 2, 2, 2).permute(0, 2, 1, 3)
    assert torch.equal(e.symbols, policy.quantize(heads, 'K', layer_indices=[9]).symbols)
    assert e.maxabs.shape == (1, 2, 1)
    with pytest.raises(ValueError): policy.quantize(x, 'K')
    with pytest.raises(ValueError): policy.quantize(x, 'K', layer_indices=[9, 9])
    with pytest.raises(ValueError): policy.quantize(x.double(), 'K', layer_indices=[9])


@torch_available
def test_zero_and_large_extrema_follow_released_no_guard_formula():
    import torch
    quant, dequant = reference.reference_functions()
    x = torch.tensor([[[0, 0, 0, 0], [-65504, 65504, .01, -.01], [0, .0001, -.0001, 0]]], dtype=torch.float16)
    e = policy.quantize(x, 'V', layer_indices=[2])
    q, m = quant(torch.tensor([16.]), x)
    assert torch.equal(e.symbols, q) and torch.equal(e.maxabs, m)
    assert torch.equal(e.dequantize(), dequant(q.float(), torch.tensor([16.]), m))
    assert e.scale[0, 0].item() == 0  # No substituted unit step or epsilon.


@torch_available
def test_QL2_storage_zero_symbol_corruption_metadata_and_reconstruction():
    import torch
    rng = torch.Generator().manual_seed(42)
    x, y = (torch.randn(32, 3, 4, generator=rng).half() for _ in range(2))
    encoded = (policy.quantize(x, 'K'), policy.quantize(y, 'V'))
    p = uniform_profile()
    with patch.object(core, 'arithmetic_encode', wraps=core.arithmetic_encode) as enc, \
         patch.object(core, 'arithmetic_decode', wraps=core.arithmetic_decode) as dec:
        blob, reconstructed, proof = storage.roundtrip(encoded, p)
        assert enc.call_count == dec.call_count == 4
    assert proof['storage_roundtrip_symbol_mismatch'] == {'K': 0, 'V': 0}
    assert proof['maxima_metadata_exact'] and proof['reconstruction_equality']
    assert proof['shape'] == [32, 3, 4]
    assert proof['scale_metadata_bytes'] == 768
    assert len(blob) == proof['total_payload_bytes']+proof['local_metadata_bytes']
    assert all(torch.equal(a, e.reconstructed) for a, e in zip(reconstructed, encoded))
    broken = replace(encoded[0], symbols=torch.full_like(encoded[0].symbols, -128))
    with pytest.raises(ValueError): storage.roundtrip((broken, encoded[1]), p)
    with pytest.raises(ValueError): fmt.decode(blob[:-1], p, expected_tokens=3)


@torch_available
def test_original_uniform_INT8_formula_and_B2_preserved():
    import torch
    from semcache.experiments.cachegen.codecs import Baseline
    from semcache.experiments.cachegen.b2.harness import pack_encoded, restored_encoded
    x = torch.arange(640).reshape(32, 10, 2).remainder(43).sub(21).half()
    x[0, 0] = 0
    encoded = Baseline('UNIFORM_INT8').encode(x, -x)
    for original, (symbols, step) in zip((x, -x), encoded):
        expected = original.float().abs().amax(-1, keepdim=True)/127
        expected[expected == 0] = 1
        assert torch.equal(step, expected)
        assert torch.equal(symbols, torch.round(original.float()/expected).clamp(-127, 127).to(torch.int8))
    domains, scales, shape = pack_encoded(encoded)
    p = uniform_profile()
    streams = fmt.streams_from_domains(p.mode, domains, shape)
    blob = fmt.encode(p, streams, scales, shape)
    decoded, stored_scales, stored_shape = fmt.decode(blob, p)
    restored_domains = fmt.domains_from_streams(p.mode, decoded, stored_shape)
    recovered = restored_encoded(restored_domains, stored_scales, stored_shape)
    for (a, sa), (b, sb) in zip(encoded, recovered):
        assert torch.equal(a, b) and torch.equal(sa, sb)
    assert all(torch.equal(a, b) for a, b in zip(Baseline('UNIFORM_INT8').decode(encoded), Baseline('UNIFORM_INT8').decode(recovered)))


def test_smoke_selection_bounded_deterministic_dataset_balance_without_loading():
    capture = dict(model_config=dict(name='facebook/opt-2.7b', dtype='float16'),
        scope='base_raw_unscaled_linear_projection; no LoRA adapter', blocks=[
            dict(block_id=f'{d}_{i}_{t}', dataset=d, token_group_size=t, partition='evaluation')
            for d in ('snips', 'multiwoz') for i in range(12) for t in (3, 10)])
    with patch.object(harness, 'verify_capture'):
        a = harness.select_blocks(capture, 12, 42)
        b = harness.select_blocks(dict(capture, blocks=list(reversed(capture['blocks']))), 12, 42)
        assert a == b and len(a) == 12
        assert all(x['token_group_size'] == 3 for x in a)
        assert sum(x['dataset'] == 'snips' for x in a) == 6
        for count in (0, 17, 1000):
            with pytest.raises(ValueError): harness.select_blocks(capture, count, 42)


@torch_available
def test_pooled_metrics_and_zero_norm_handling():
    import torch
    x, y = torch.tensor([1., 2., 3., 4.]), torch.tensor([1., 1., 2., 4.])
    rows = []
    for i in range(2):
        sums = harness.metric_sums(x[2*i:2*i+2], y[2*i:2*i+2])
        rows.append(dict(block_id=str(i), role='K', layer=i, bins=32, **sums))
    summary = next(r for r in harness.aggregate(rows) if r['group'] == 'overall')
    assert summary['MSE'] == .5 and summary['max_abs_error'] == 1
    assert summary['relative_l2'] == pytest.approx((2/30)**.5)
    assert summary['cosine_similarity'] == pytest.approx(25/(30*22)**.5)
    zeros = harness.metrics(harness.metric_sums(torch.zeros(4), torch.zeros(4)))
    assert zeros['relative_l2'] is zeros['cosine_similarity'] is None


def test_cli_never_offers_full_workload_and_protects_existing_artifacts():
    with pytest.raises(SystemExit): harness.main(['storage-full'])
    with pytest.raises(SystemExit): harness.main(['parity', '--output-dir', 'results/cachegen/c1'])
    with pytest.raises(SystemExit): harness.main(['smoke', '--allow-download'])
