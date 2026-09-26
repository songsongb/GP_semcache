"""Four-role framing adapter around the unchanged C1.5-A arithmetic primitives."""
from dataclasses import dataclass
from hashlib import sha256
import struct

from .. import b1
from ..shared import core

MODES = ('B2_RAW_ROLE_SPLIT', 'B2_ANCHOR_MOD_RESIDUAL_KV', 'B2_K_RESIDUAL_V_RAW')
FILES = dict(zip(MODES, ('raw_role_split.bin', 'anchor_mod_residual_kv.bin', 'k_residual_v_raw.bin')))
CLASSIFICATIONS = dict(zip(MODES, ('FAIR_CONTROL', 'PRIMARY', 'POST_HOC_EXPLORATORY')))
STREAM_NAMES = ('k_anchor_payload_bytes', 'k_nonanchor_or_residual_payload_bytes',
                'v_anchor_payload_bytes', 'v_nonanchor_or_residual_payload_bytes')


def labels(mode):
    return dict(result_classification=CLASSIFICATIONS[mode],
        predeclaration='NOT_PREDECLARED_PRIMARY' if mode == MODES[2] else
                      'B1_PREDECLARED_TRANSFORM' if mode == MODES[1] else 'FAIR_CONTROL',
        experiment_role='PRIMARY_B2' if mode == MODES[1] else CLASSIFICATIONS[mode],
        provenance='MEASURED_RESEARCH_EXTENSION', inspiration='CACHEGEN_INSPIRED',
        transform_classification='LOSSLESS_SYMBOL_TRANSFORM')


def validate_shape(shape):
    if (len(shape) != 3 or shape[0] != 32 or shape[1] != 10 or
            type(shape[2]) is not int or not 1 <= shape[2] <= 65536):
        raise ValueError('B2 requires [32,10,hidden]; T=3 and padding forbidden')


def stream_counts(shape):
    validate_shape(shape)
    n = shape[0]*shape[2]
    return n, 9*n, n, 9*n


def residual_components(mode):
    if mode not in MODES:
        raise ValueError('Unknown B2 mode')
    return (mode != MODES[0], mode == MODES[1])


def streams_from_domains(mode, domains, shape):
    validate_shape(shape)
    if len(domains) != 2:
        raise ValueError('Separate K and V domains required')
    streams = []
    for data, residual in zip(domains, residual_components(mode)):
        b1.validate_domain(data, shape)
        representation = b1.transform(data, shape) if residual else data
        streams.extend(b1.role_domains(representation, shape))
    return tuple(streams)


def domains_from_streams(mode, streams, shape):
    expected = stream_counts(shape)
    if len(streams) != 4 or any(len(s) != n or 255 in s for s, n in zip(streams, expected)):
        raise ValueError('Invalid B2 symbol streams')
    layers, _, hidden = shape
    domains = []
    for component, residual in enumerate(residual_components(mode)):
        anchor, remainder = streams[2*component:2*component+2]
        packed = b''.join(anchor[l*hidden:(l+1)*hidden]+remainder[l*9*hidden:(l+1)*9*hidden]
                          for l in range(layers))
        domains.append(b1.transform(packed, shape, inverse=True) if residual else packed)
    return tuple(domains)


@dataclass(frozen=True)
class Profile:
    mode: str
    cdfs: tuple

    def __post_init__(self):
        if self.mode not in MODES or not isinstance(self.cdfs, tuple) or len(self.cdfs) != 4:
            raise ValueError('Exactly four CDFs per B2 mode required')
        for cdf in self.cdfs:
            core.validate_cdf(cdf)

    def to_bytes(self):
        # Same structs, endian, uint32 CDF entries and framing widths as C1.5-A.
        # Separate magic/mode namespace prevents an A or B1 artifact being misread.
        return (core.PROFILE_HEADER.pack(b'SCCDFB02', MODES.index(self.mode), 32, 4)+
                b''.join(struct.pack('<256I', *cdf) for cdf in self.cdfs))

    @property
    def sha256(self):
        return sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data, expected_hash):
        if sha256(data).hexdigest() != expected_hash:
            raise ValueError('B2 profile SHA256 mismatch')
        if len(data) != core.PROFILE_HEADER.size+4*256*4:
            raise ValueError('Invalid B2 profile size')
        magic, mode, layers, n = core.PROFILE_HEADER.unpack_from(data)
        if magic != b'SCCDFB02' or mode >= len(MODES) or layers != 32 or n != 4:
            raise ValueError('Invalid B2 profile framing')
        return cls(MODES[mode], tuple(struct.unpack_from('<256I', data, core.PROFILE_HEADER.size+i*1024)
                                     for i in range(4)))


def fit(blocks, loader):
    """Only calibration T=10 blocks can be passed to the tensor loader."""
    core.partition_ids(blocks)
    counts = {m: [[0]*255 for _ in range(4)] for m in MODES}
    fitted = []
    for block in blocks:
        if block['partition'] != 'calibration':
            continue
        if block['token_group_size'] != 10:
            raise ValueError('B2 calibration must be T=10')
        domains, _, shape = loader(block)
        for mode in MODES:
            for target, stream in zip(counts[mode], streams_from_domains(mode, domains, shape)):
                for i, n in enumerate(b1.counts(stream)):
                    target[i] += n
        fitted.append(block['block_id'])
    return {m: Profile(m, tuple(core.cdf_from_counts(h) for h in counts[m])) for m in MODES}, counts, fitted


def encode(profile, streams, scales, shape):
    expected = stream_counts(shape)
    if len(streams) != 4 or any(len(s) != n for s, n in zip(streams, expected)):
        raise ValueError('B2 stream size mismatch')
    if len(scales) != 2*shape[0]*shape[1]*4:
        raise ValueError('Unchanged C1 FP32 scales required')
    payloads = [core.arithmetic_encode(s, cdf) for s, cdf in zip(streams, profile.cdfs)]
    header = core.BLOCK_HEADER.pack(b'SCKVB002', MODES.index(profile.mode), *shape, 4,
                                    bytes.fromhex(profile.sha256))
    lengths = struct.pack('<4I', *(len(p) for p in payloads))
    body = header+lengths+scales+b''.join(payloads)
    return body+sha256(body).digest()


def inspect(data, profile):
    overhead = core.BLOCK_HEADER.size+16+32
    if len(data) < overhead or sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError('Corrupt B2 bitstream SHA256')
    magic, mode, layers, tokens, hidden, n, fingerprint = core.BLOCK_HEADER.unpack_from(data)
    if magic != b'SCKVB002' or mode != MODES.index(profile.mode) or n != 4 or fingerprint.hex() != profile.sha256:
        raise ValueError('B2 bitstream mode/profile mismatch')
    shape = layers, tokens, hidden
    stream_counts(shape)
    scale_size = 2*layers*tokens*4
    lengths = struct.unpack_from('<4I', data, core.BLOCK_HEADER.size)
    if any(n < 1 for n in lengths) or len(data) != overhead+scale_size+sum(lengths):
        raise ValueError('B2 bitstream lengths/accounting mismatch')
    offset = core.BLOCK_HEADER.size+16
    scales = data[offset:offset+scale_size]
    offset += scale_size
    payloads = []
    for length in lengths:
        payloads.append(data[offset:offset+length])
        offset += length
    sizes = dict(zip(STREAM_NAMES, lengths))
    sizes.update(total_payload_bytes=sum(lengths), scale_metadata_bytes=scale_size,
        local_transform_metadata_bytes=overhead, local_metadata_bytes=overhead+scale_size)
    return shape, scales, payloads, sizes


def decode(data, profile):
    shape, scales, payloads, _ = inspect(data, profile)
    return tuple(core.arithmetic_decode(p, n, cdf) for p, n, cdf in
                 zip(payloads, stream_counts(shape), profile.cdfs)), scales, shape
