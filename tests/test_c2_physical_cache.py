"""C2 physical cache tests use tiny CPU tensors and a synthetic profile only."""
import pytest

torch = pytest.importorskip('torch')

from semcache.cache.admission import AdmissionPolicy
from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.experiments.cachegen.c2 import physical_storage as ps
from semcache.experiments.cachegen.b2 import format as fmt
from semcache.experiments.cachegen.c15c.rate_storage import MODE
from semcache.experiments.cachegen import c2_storage_smoke as c2
from semcache.experiments.cachegen.shared import core


@pytest.fixture
def synthetic_profile(tmp_path, monkeypatch):
    profile = fmt.Profile(MODE, (core.cdf_from_counts([0]*255),)*4)
    path = tmp_path/'profile.bin'
    path.write_bytes(profile.to_bytes())
    monkeypatch.setattr(ps, 'PROFILE_SHA256', profile.sha256)
    return path, profile


def blocks(seed=7):
    generator = torch.Generator().manual_seed(seed)
    return {i: tuple(torch.randn(1, 3, 2, generator=generator).half() for _ in range(3))
            for i in range(32)}


def compressed_cache(profile, *, capacity=10000, instrument=True, coder_backend=fmt.REFERENCE_CODER):
    codec = ps.FrozenK20V16Codec(profile, quantization_device='cpu', decode_device='cpu',
                                 expected_hidden=2, instrument=instrument, coder_backend=coder_backend)
    return GlobalCache(capacity, admission=AdmissionPolicy(threshold=0),
        physical_storage_mode=ps.MODE_COMPRESSED, physical_codec=codec,
        instrument_storage=instrument)


def test_raw_insert_lookup_remains_same_object_and_bytes():
    cache = GlobalCache(10000, admission=AdmissionPolicy(threshold=0))
    source = blocks()
    entry = cache.make_entry(3, (10, 11, 12), (0, 3), source)
    assert cache.insert(entry)
    assert cache.lookup(entry.key) is entry
    assert cache.lookup((3, (99,))) is None
    assert entry.tensors and entry.compressed_kv is None and entry.q_tensors is None
    assert entry.storage_accounting['raw_qkv_bytes'] == 32*3*3*2*2
    assert cache.physical_tensor_bytes == entry.physical_tensor_bytes == entry.size_bytes


@pytest.mark.parametrize('backend', [fmt.REFERENCE_CODER, fmt.FAST_CODER])
def test_compressed_entry_roundtrip_q_accounting_key_and_no_raw_kv(synthetic_profile, monkeypatch, backend):
    path, profile = synthetic_profile
    cache = compressed_cache(path, coder_backend=backend)
    source = blocks()
    with monkeypatch.context() as scoped:
        scoped.setattr(core, 'cdf_from_counts', lambda *a: pytest.fail('Runtime CDF fit'))
        entry = cache.make_entry(3, (10, 11, 12), (0, 3), source)
        assert cache.insert(entry)
        view = cache.lookup(entry.key)
    assert cache.entries[entry.key] is entry and view.resident is entry
    assert entry.key == view.key and entry.tensors is None and entry.q_tensors
    assert all(t.dtype == torch.float16 and t.device.type == 'cpu' for t in entry.q_tensors.values())
    assert isinstance(entry.compressed_kv.bitstream, bytes)
    assert entry.compressed_kv.profile_sha256 == profile.sha256
    assert cache.lookup((3, (99,))) is None
    k = torch.cat([source[i][1] for i in range(32)])
    v = torch.cat([source[i][2] for i in range(32)])
    for role, original, index in [('K', k, 1), ('V', v, 2)]:
        direct = ps.POLICY.quantize(original, role)
        assert torch.equal(view.kv_symbols[index-1], direct.symbols)
        assert torch.equal(torch.cat([view.tensors[i][index] for i in range(32)]), direct.reconstructed)
    for i in range(32):
        assert torch.equal(view.tensors[i][0], source[i][0])
    account = entry.storage_accounting
    assert account['raw_q_bytes'] == 32*3*2*2
    assert account['raw_kv_bytes'] == 2*32*3*2*2
    assert account['raw_qkv_bytes'] == entry.size_bytes
    assert account['stored_q_bytes'] == account['raw_q_bytes']
    assert account['compressed_kv_entry_bytes'] == len(entry.compressed_kv.bitstream)
    assert account['encoded_kv_payload_bytes']+account['kv_metadata_bytes'] == account['compressed_kv_entry_bytes']
    assert sum(account['role_payload_bytes'].values()) == account['encoded_kv_payload_bytes']
    assert account['actual_qkv_entry_bytes'] == cache.physical_tensor_bytes
    assert account['global_profile_bytes_charged_per_entry'] == 0
    assert cache.physical_codec.profile_bytes == len(profile.to_bytes())
    assert {'quantize_ms', 'encode_ms'} <= entry.storage_timings_ms.keys()
    assert {'decode_ms', 'dequantize_ms'} <= view.storage_timings_ms.keys()
    assert {'insert', 'lookup'} <= {r['operation'] for r in cache.storage_timing_records}
    cache.record_reuse(view)
    assert entry.frequency == 2  # One default lookup plus explicit reuse.


def test_profile_missing_and_hash_mismatch(tmp_path, synthetic_profile, monkeypatch):
    path, _ = synthetic_profile
    with pytest.raises(FileNotFoundError, match='profile missing'):
        compressed_cache(tmp_path/'missing.bin')
    monkeypatch.setattr(ps, 'PROFILE_SHA256', '0'*64)
    with pytest.raises(ValueError, match='SHA256'):
        compressed_cache(path)


def test_cache_mode_rejects_wrong_physical_entry(synthetic_profile):
    path, _ = synthetic_profile
    source = blocks()
    raw = GlobalCache(10000, admission=AdmissionPolicy(threshold=0))
    compressed = compressed_cache(path)
    raw_entry = raw.make_entry(1, (1, 2, 3), (0, 3), source)
    compressed_entry = compressed.make_entry(1, (1, 2, 3), (0, 3), source)
    with pytest.raises(ValueError, match='Compressed cache cannot retain raw'):
        compressed.insert(raw_entry)
    with pytest.raises(ValueError, match='requires a compressed cache'):
        raw.insert(compressed_entry)


def test_c2_entry_frame_identical_across_coders(synthetic_profile):
    path, _ = synthetic_profile
    source = blocks()
    reference = compressed_cache(path, instrument=False)
    fast = compressed_cache(path, instrument=False, coder_backend=fmt.FAST_CODER)
    a = reference.make_entry(0, (1, 2, 3), (0, 3), source)
    b = fast.make_entry(0, (1, 2, 3), (0, 3), source)
    assert a.compressed_kv.bitstream == b.compressed_kv.bitstream
    assert a.storage_accounting == b.storage_accounting


def test_pinned_profile_and_bounded_smoke_selection(monkeypatch):
    assert ps.PROFILE_SHA256 == '8defa27a83e8ad16e9c7efa0f40e68c302aef90363e14a1451288a3bf433cb2c'
    selected = [dict(block_id=f'{dataset}_{i}', dataset=dataset, token_group_size=3)
                for dataset in ('multiwoz', 'snips') for i in range(6)]
    monkeypatch.setattr(c2, 'evaluation_blocks', lambda capture: (selected, {'required_window_size': 3}))
    chosen, counts = c2.select_smoke_blocks({}, 4)
    assert len(chosen) == 8 and counts['required_window_size'] == 3
    assert [b['block_id'] for b in chosen] == [f'{dataset}_{i}' for dataset in ('multiwoz', 'snips') for i in range(4)]
    with pytest.raises(ValueError, match='capped'):
        c2.select_smoke_blocks({}, 5)


def test_logical_eviction_and_hit_miss_match_raw(synthetic_profile):
    path, _ = synthetic_profile
    first, second = blocks(1), blocks(2)
    raw = GlobalCache(32*3*3*2*2, admission=AdmissionPolicy(threshold=0))
    compressed = compressed_cache(path, capacity=raw.capacity_bytes, instrument=False)
    for cache in (raw, compressed):
        for cluster, source in [(1, first), (2, second)]:
            entry = cache.make_entry(cluster, (1, 2, 3), (0, 3), source)
            assert cache.insert(entry)
        assert len(cache.entries) == 1
    assert set(raw.entries) == set(compressed.entries)
    for key in [(1, (1, 2, 3)), (2, (1, 2, 3)), (9, (1, 2, 3))]:
        assert (raw.lookup(key) is None) == (compressed.lookup(key) is None)
    assert raw.logical_cache_bytes == compressed.logical_cache_bytes
    assert raw.hits == compressed.hits and raw.misses == compressed.misses
