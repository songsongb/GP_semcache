"""C2.5 CPU-only bitstream parity against the original B2 arithmetic coder."""
import random

import pytest

from semcache.experiments.cachegen.shared import core
from semcache.experiments.cachegen.b2 import format as fmt
from semcache.experiments.cachegen.c25_coder_bench import benchmark_streams


@pytest.mark.parametrize('length', [1, 2, 3, 7, 31, 256, 8192])
@pytest.mark.parametrize('distribution', ['uniform', 'skewed', 'extremes'])
def test_fast_coder_exact_reference_bits_and_symbols(length, distribution):
    rng = random.Random(1000+length)
    cdf = core.cdf_from_counts([rng.randrange(30) for _ in range(255)])
    if distribution == 'uniform':
        data = bytes(rng.randrange(255) for _ in range(length))
    elif distribution == 'skewed':
        data = bytes(rng.choice((0, 1, 1, 1, 127, 254)) for _ in range(length))
    else:
        data = bytes((0, 254)[i % 2] for i in range(length))
    reference = core.arithmetic_encode(data, cdf)
    fast = core.arithmetic_encode_fast(data, cdf)
    assert fast == reference
    assert core.arithmetic_decode(reference, length, cdf) == data
    assert core.arithmetic_decode_fast(fast, length, cdf) == data


def test_fast_coder_rejects_invalid_streams():
    cdf = core.cdf_from_counts([0]*255)
    with pytest.raises(ValueError, match='outside support'):
        core.arithmetic_encode_fast(bytes([255]), cdf)
    with pytest.raises(ValueError, match='Nonempty'):
        core.arithmetic_decode_fast(b'', 1, cdf)


def test_b2_four_role_frame_is_byte_identical_and_cross_decodable():
    cdf = core.cdf_from_counts([i % 7 for i in range(255)])
    profile = fmt.Profile(fmt.MODES[1], (cdf,)*4)
    shape = (32, 3, 2)
    counts = fmt.stream_counts(shape, expected_tokens=3)
    streams = tuple(bytes((i*17+j) % 255 for i in range(n)) for j, n in enumerate(counts))
    scales = bytes(2*32*3*4)
    reference = fmt.encode(profile, streams, scales, shape, expected_tokens=3)
    fast = fmt.encode(profile, streams, scales, shape, expected_tokens=3, coder_backend=fmt.FAST_CODER)
    assert fast == reference
    for coder in (fmt.REFERENCE_CODER, fmt.FAST_CODER):
        decoded, metadata, actual_shape = fmt.decode(fast, profile, expected_tokens=3, coder_backend=coder)
        assert decoded == streams and metadata == scales and actual_shape == shape


def test_cpu_microbenchmark_reports_four_roles_and_equal_bytes():
    cdf = core.cdf_from_counts([0]*255)
    profile = fmt.Profile(fmt.MODES[1], (cdf,)*4)
    counts = fmt.stream_counts((32, 3, 2), expected_tokens=3)
    streams = tuple(bytes((i*3+j) % 255 for i in range(n)) for j, n in enumerate(counts))
    report, samples = benchmark_streams(streams, profile, repeats=2)
    assert set(report) == {fmt.REFERENCE_CODER, fmt.FAST_CODER}
    assert samples[fmt.REFERENCE_CODER][0] == samples[fmt.FAST_CODER][0]
    for backend in report:
        assert report[backend]['symbol_mismatches'] == 0
        assert report[backend]['encoded_payload_bytes'] == sum(map(len, samples[backend][0]))
        assert len(report[backend]['roles']) == 4
        assert all(v['encode_MB_per_s'] > 0 and v['decode_symbols_per_s'] > 0
                   for v in report[backend]['roles'].values())
