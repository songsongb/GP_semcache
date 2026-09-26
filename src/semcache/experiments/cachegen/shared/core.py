"""Deterministic CPU arithmetic coding; RESEARCH_EXTENSION, not official CacheGen.

No torch dependency. Fixed integer CDFs, no adaptation during encode/decode.
All byte formats are little endian except arithmetic bits (MSB first).
"""
from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256
import struct

MODES = ('SHARED_CDF_GLOBAL', 'SHARED_CDF_LAYERGROUP')
TOTAL = 65536
SUPPORT = 255
GROUPS = ((0, 11), (11, 22), (22, 32))
PROFILE_HEADER = struct.Struct('<8sBBH')
BLOCK_HEADER = struct.Struct('<8sBHHIH32s')
CONFIG = dict(schema_version=1, alphabet='int8 [-127,127] -> uint8 [0,254] by +127',
              smoothing='Laplace +1; REPRODUCTION_CHOICE', total=TOTAL,
              normalization='reserve 1 per symbol; apportion remaining mass from smoothed counts; '
                            'largest remainder, ties by ascending symbol',
              layer_groups=[list(g) for g in GROUPS], layer_group_provenance='REPRODUCTION_CHOICE',
              grouping='model-level; both datasets and T values pooled; separate K/V',
              quantization='existing cachegen.codecs.Baseline(UNIFORM_INT8), unchanged',
              coder='CPU integer arithmetic coder, 32-bit state; RESEARCH_EXTENSION')


def map_symbols(values):
    values = list(values)
    if any(type(v) is not int or not -127 <= v <= 127 for v in values):
        raise ValueError('int8 symbols must be in [-127,127]')
    return bytes(v+127 for v in values)


def unmap_symbols(data):
    if any(v >= SUPPORT for v in data):
        raise ValueError('Mapped symbols must be in [0,254]')
    return [v-127 for v in data]


def cdf_from_counts(counts):
    if len(counts) != SUPPORT or any(type(c) is not int or c < 0 for c in counts):
        raise ValueError('Expected 255 nonnegative integer counts')
    smooth = [c+1 for c in counts]
    denominator = sum(smooth)
    mass = TOTAL-SUPPORT
    freqs = [1 + c*mass//denominator for c in smooth]
    order = sorted(range(SUPPORT), key=lambda i: (-(smooth[i]*mass % denominator), i))
    for i in order[:TOTAL-sum(freqs)]:
        freqs[i] += 1
    cdf = [0]
    for f in freqs:
        cdf.append(cdf[-1]+f)
    return tuple(cdf)


def validate_cdf(cdf):
    if (len(cdf) != SUPPORT+1 or cdf[0] != 0 or cdf[-1] != TOTAL
            or any(type(v) is not int for v in cdf)
            or any(a >= b for a, b in zip(cdf, cdf[1:]))):
        raise ValueError('CDF requires strictly increasing 256 integers from 0 to 65536')


@dataclass(frozen=True)
class Profile:
    mode: str
    cdfs: tuple

    def __post_init__(self):
        if self.mode not in MODES or len(self.cdfs) != (2 if self.mode == MODES[0] else 6):
            raise ValueError('Invalid profile mode/CDF count')
        if not isinstance(self.cdfs, tuple) or any(not isinstance(c, tuple) for c in self.cdfs):
            raise ValueError('CDFs must be immutable tuples')
        for cdf in self.cdfs:
            validate_cdf(cdf)

    @property
    def groups(self):
        return ((0, 32),) if self.mode == MODES[0] else GROUPS

    def to_bytes(self):
        return (PROFILE_HEADER.pack(b'SCCDF001', MODES.index(self.mode), 32, len(self.cdfs))
                + b''.join(struct.pack('<256I', *c) for c in self.cdfs))

    @property
    def sha256(self):
        return sha256(self.to_bytes()).hexdigest()

    @property
    def logical_bytes(self):
        return len(self.cdfs)*256*4

    @classmethod
    def from_bytes(cls, data, expected_sha):
        if sha256(data).hexdigest() != expected_sha:
            raise ValueError('Profile SHA256 mismatch')
        if len(data) < PROFILE_HEADER.size:
            raise ValueError('Truncated profile')
        magic, mode, layers, count = PROFILE_HEADER.unpack_from(data)
        if magic != b'SCCDF001' or mode >= len(MODES) or layers != 32 or len(data) != PROFILE_HEADER.size+count*1024:
            raise ValueError('Invalid profile format')
        return cls(MODES[mode], tuple(struct.unpack_from('<256I', data, PROFILE_HEADER.size+i*1024)
                                     for i in range(count)))


def partition_ids(blocks):
    ids = {p: [] for p in ('calibration', 'evaluation')}
    seen = set()
    for block in blocks:
        if block['partition'] not in ids or block['block_id'] in seen:
            raise ValueError('Invalid partition or duplicate/overlapping block ID')
        seen.add(block['block_id'])
        ids[block['partition']].append(block['block_id'])
    if not all(ids.values()) or set(ids['calibration']) & set(ids['evaluation']):
        raise ValueError('Nonempty disjoint calibration/evaluation sets required')
    return {p: sorted(v) for p, v in ids.items()}


def fit_profiles(blocks, load_histograms):
    """Loader is invoked ONLY on calibration blocks; fit both fixed modes once.

    Loader yields K then V, each as 32 histograms of 255 integer counts.
    No evaluation labels, tensors, symbols, or statistics enter fitting.
    """
    partition_ids(blocks)
    counts = {mode: [[0]*SUPPORT for _ in range(2 if mode == MODES[0] else 6)] for mode in MODES}
    fitted = []
    for block in sorted(blocks, key=lambda b: b['block_id']):
        if block['partition'] != 'calibration':
            continue
        assert block['partition'] == 'calibration'
        histograms = load_histograms(block)
        if len(histograms) != 2 or any(len(h) != 32 for h in histograms):
            raise ValueError('Expected K/V histograms for all 32 layers')
        for component, layers in enumerate(histograms):
            for layer, hist in enumerate(layers):
                if len(hist) != SUPPORT or any(type(c) is not int or c < 0 for c in hist):
                    raise ValueError('Invalid calibration histogram')
                group = 0 if layer < 11 else 1 if layer < 22 else 2
                for mode, index in ((MODES[0], component), (MODES[1], group*2+component)):
                    counts[mode][index] = [a+b for a, b in zip(counts[mode][index], hist)]
        fitted.append(block['block_id'])
    return ({mode: Profile(mode, tuple(cdf_from_counts(c) for c in hist)) for mode, hist in counts.items()},
            fitted, counts)


# Classic E1/E2/E3 arithmetic coding. Decoder reads implicit zero bits after
# termination; the enclosing length and SHA256 reject truncation/corruption.
HALF, QUARTER, THREE_QUARTERS = 1 << 31, 1 << 30, 3 << 30
MASK = (1 << 32)-1


def arithmetic_encode(symbols, cdf):
    validate_cdf(cdf)
    low, high, pending = 0, MASK, 0
    out = bytearray()
    byte, bits = 0, 0
    def emit(bit):
        nonlocal byte, bits
        byte = (byte << 1) | bit
        bits += 1
        if bits == 8:
            out.append(byte)
            byte, bits = 0, 0
    for symbol in symbols:
        if not 0 <= symbol < SUPPORT:
            raise ValueError('Arithmetic symbol outside support')
        span = high-low+1
        high = low + span*cdf[symbol+1]//TOTAL-1
        low += span*cdf[symbol]//TOTAL
        while True:
            if high < HALF:
                bit = 0
            elif low >= HALF:
                bit = 1
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                pending += 1
                low -= QUARTER
                high -= QUARTER
                low *= 2
                high = high*2+1
                continue
            else:
                break
            emit(bit)
            for _ in range(pending):
                emit(1-bit)
            pending = 0
            low *= 2
            high = high*2+1
    pending += 1
    bit = int(low >= QUARTER)
    emit(bit)
    for _ in range(pending):
        emit(1-bit)
    if bits:
        out.append(byte << (8-bits))
    return bytes(out)


def arithmetic_decode(data, count, cdf):
    validate_cdf(cdf)
    if count < 1 or not data:
        raise ValueError('Nonempty arithmetic stream required')
    position = 0
    def read():
        nonlocal position
        i, offset = divmod(position, 8)
        position += 1
        return (data[i] >> (7-offset)) & 1 if i < len(data) else 0
    low, high, value = 0, MASK, 0
    for _ in range(32):
        value = (value << 1) | read()
    out = bytearray()
    for _ in range(count):
        span = high-low+1
        target = ((value-low+1)*TOTAL-1)//span
        symbol = bisect_right(cdf, target)-1
        if not 0 <= symbol < SUPPORT:
            raise ValueError('Invalid arithmetic stream')
        out.append(symbol)
        high = low + span*cdf[symbol+1]//TOTAL-1
        low += span*cdf[symbol]//TOTAL
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                value -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                low -= QUARTER
                high -= QUARTER
                value -= QUARTER
            else:
                break
            low *= 2
            high = high*2+1
            value = value*2+read()
    return bytes(out)


def stream_counts(profile, shape):
    layers, tokens, hidden = shape
    if layers != 32 or tokens not in (3, 10) or not 1 <= hidden <= 65536:
        raise ValueError('Expected [32,T,hidden] with T=3 or T=10')
    return [(end-start)*tokens*hidden for start, end in profile.groups for _ in 'kv']


def encode(profile, streams, scales, shape):
    counts = stream_counts(profile, shape)
    if len(streams) != len(counts) or any(len(s) != n for s, n in zip(streams, counts)):
        raise ValueError('Symbol stream shapes differ from profile')
    if len(scales) != 2*shape[0]*shape[1]*4:
        raise ValueError('Expected unchanged FP32 K/V per-vector scales')
    payloads = [arithmetic_encode(s, cdf) for s, cdf in zip(streams, profile.cdfs)]
    header = BLOCK_HEADER.pack(b'SCKV0001', MODES.index(profile.mode), *shape,
                               len(streams), bytes.fromhex(profile.sha256))
    lengths = struct.pack('<'+'I'*len(streams), *(len(p) for p in payloads))
    body = header + lengths + scales + b''.join(payloads)
    return body + sha256(body).digest()


def inspect_block(data, profile):
    if len(data) < BLOCK_HEADER.size+32 or sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError('Corrupt/truncated bitstream: SHA256 mismatch')
    magic, mode, layers, tokens, hidden, nstreams, fingerprint = BLOCK_HEADER.unpack_from(data)
    if magic != b'SCKV0001' or mode != MODES.index(profile.mode) or fingerprint.hex() != profile.sha256:
        raise ValueError('Block profile SHA256/mode mismatch')
    shape = layers, tokens, hidden
    counts = stream_counts(profile, shape)
    if nstreams != len(counts):
        raise ValueError('Invalid stream count')
    offset = BLOCK_HEADER.size
    scales_length = 2*layers*tokens*4
    metadata = offset+4*nstreams+scales_length+32
    if len(data) < metadata:
        raise ValueError('Truncated local metadata')
    lengths = struct.unpack_from('<'+'I'*nstreams, data, offset)
    if any(n < 1 for n in lengths) or metadata+sum(lengths) != len(data):
        raise ValueError('Invalid arithmetic stream lengths')
    offset += 4*nstreams
    scales = data[offset:offset+scales_length]
    offset += scales_length
    payloads = []
    for length in lengths:
        payloads.append(data[offset:offset+length])
        offset += length
    return shape, scales, payloads, dict(encoded_payload_bytes=sum(lengths), local_metadata_bytes=metadata,
                                        scale_metadata_bytes=scales_length)


def decode(data, profile):
    shape, scales, payloads, _ = inspect_block(data, profile)
    streams = tuple(arithmetic_decode(p, n, cdf) for p, n, cdf in
                    zip(payloads, stream_counts(profile, shape), profile.cdfs))
    return streams, scales, shape


def pool_accounting(rows, shared_profile_bytes):
    if not rows:
        raise ValueError('Empty evaluation pool')
    raw_kv = sum(r['raw_kv_bytes'] for r in rows)
    q = sum(r['raw_q_bytes'] for r in rows)
    payload = sum(r['encoded_payload_bytes'] for r in rows)
    local = sum(r['local_metadata_bytes'] for r in rows)
    encoded = shared_profile_bytes+payload+local
    return dict(block_count=len(rows), raw_kv_pool_bytes=raw_kv, raw_semcache_pool_bytes=q+raw_kv,
                encoded_payload_pool_bytes=payload, local_metadata_pool_bytes=local,
                shared_profile_bytes=shared_profile_bytes, encoded_kv_pool_bytes=encoded,
                encoded_semcache_pool_bytes=q+encoded, pool_kv_compression_ratio=raw_kv/encoded,
                pool_semcache_compression_ratio=(q+raw_kv)/(q+encoded),
                payload_only_kv_compression_ratio=raw_kv/payload,
                payload_only_semcache_compression_ratio=(q+raw_kv)/(q+payload))
