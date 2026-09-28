"""Independent K/V components using the existing B2 representation and coder."""
from dataclasses import dataclass
from hashlib import sha256
import struct

from .. import b1
from ..b2 import format as fmt
from ..shared import core

MODE = fmt.MODES[1]


@dataclass(frozen=True)
class RoleProfile:
    role: str
    cdfs: tuple
    scope_hash: str

    def __post_init__(self):
        if self.role not in ('K', 'V') or not isinstance(self.cdfs, tuple) or len(self.cdfs) != 2:
            raise ValueError('A K/V profile requires exactly two immutable role CDFs')
        if len(self.scope_hash) != 64 or any(c not in '0123456789abcdef' for c in self.scope_hash):
            raise ValueError('Calibration scope SHA256 required')
        for cdf in self.cdfs:
            core.validate_cdf(cdf)


def fit_role(role, counts, scope_hash):
    if len(counts) != 2:
        raise ValueError('Anchor and residual counts required')
    return RoleProfile(role, tuple(core.cdf_from_counts(c) for c in counts), scope_hash)


def compose_profile(k, v):
    if k.role != 'K' or v.role != 'V' or k.scope_hash != v.scope_hash:
        raise ValueError('Compose ordered K/V components from the same calibration scope')
    return fmt.Profile(MODE, k.cdfs+v.cdfs)


def role_streams(encoded):
    """Use B2's established single-domain transform and anchor/residual splitter."""
    import torch
    shape = tuple(encoded.symbols.shape)
    fmt.validate_shape(shape, expected_tokens=shape[1])
    domain = bytes((encoded.signed_symbols.cpu().to(torch.int16)+127).to(torch.uint8).flatten().tolist())
    transformed = fmt._representation(domain, shape, residual=True)
    return fmt._roles(transformed, shape)


def stream_counts(streams):
    return [b1.counts(s) for s in streams]


def encode_role(encoded, profile):
    """Two coder calls, two inverse checks; no profile/candidate counterpart needed."""
    import torch
    if encoded.role != profile.role:
        raise ValueError('Role/profile mismatch')
    shape = tuple(encoded.symbols.shape)
    streams = role_streams(encoded)
    payloads = tuple(core.arithmetic_encode(s, cdf) for s, cdf in zip(streams, profile.cdfs))
    decoded = tuple(core.arithmetic_decode(p, len(s), cdf) for p, s, cdf in zip(payloads, streams, profile.cdfs))
    if decoded != streams:
        raise ValueError('Calibration storage symbol mismatch')
    layers, tokens, hidden = shape
    anchor, residual = decoded
    packed = b''.join(anchor[l*hidden:(l+1)*hidden]+residual[l*(tokens-1)*hidden:(l+1)*(tokens-1)*hidden]
                      for l in range(layers))
    domain = fmt._representation(packed, shape, residual=True, inverse=True)
    signed = (torch.frombuffer(bytearray(domain), dtype=torch.uint8).to(torch.int16)-127).reshape(shape)
    metadata = struct.pack('<96f', *encoded.storage_metadata.cpu().flatten().tolist())
    recovered_metadata = torch.tensor(struct.unpack('<96f', metadata), dtype=torch.float32).reshape(32, 3, 1)
    recovered = encoded.restore_storage(signed, recovered_metadata)
    if not torch.equal(recovered.symbols, encoded.symbols):
        raise ValueError('Calibration storage inverse symbol mismatch')
    if not torch.equal(recovered.storage_metadata, encoded.storage_metadata):
        raise ValueError('Calibration storage metadata mismatch')
    if not torch.equal(recovered.dequantize(), encoded.dequantize()) or not torch.equal(recovered.reconstructed, encoded.reconstructed):
        raise ValueError('Storage reconstruction differs from quantization-only reconstruction')
    return payloads, metadata


def physical_accounting(k_payload, v_payload, shapes, *, profile_bytes, expected_tokens=3):
    """Fixed B2 framing + measured role lengths; one full profile per workload."""
    if len(k_payload) != len(v_payload) or len(shapes) != len(k_payload) or not shapes:
        raise ValueError('Matching nonempty block populations required')
    if type(profile_bytes) is not int or profile_bytes < 1:
        raise ValueError('Actual serialized full-profile bytes required')
    transform_per_block = core.BLOCK_HEADER.size+16+sha256().digest_size
    local, scales, raw = [], [], 0
    for shape in shapes:
        fmt.validate_shape(shape, expected_tokens=shape[1] if expected_tokens is None else expected_tokens)
        scale_size = 2*shape[0]*shape[1]*4
        scales.append(scale_size)
        local.append(transform_per_block+scale_size)
        raw += 2*shape[0]*shape[1]*shape[2]*2
    if any(type(v) is not int or v < 2 for v in (*k_payload, *v_payload)):
        raise ValueError('Measured positive two-stream role payload lengths required')
    bitstreams = [k+v+m for k, v, m in zip(k_payload, v_payload, local)]
    total = sum(bitstreams)+profile_bytes
    return dict(k_payload_bytes=sum(k_payload), v_payload_bytes=sum(v_payload),
        total_payload_bytes=sum(k_payload)+sum(v_payload), per_block_bitstream_bytes=bitstreams,
        bitstream_pool_bytes=sum(bitstreams), local_metadata_bytes=sum(local),
        local_transform_metadata_bytes=transform_per_block*len(shapes), scale_maxabs_metadata_bytes=sum(scales),
        global_profile_bytes=profile_bytes, total_physical_bytes=total,
        original_fp16_kv_bytes=raw, compression_ratio=raw/total)


def profile_attribution():
    # Full B2 profile has a shared header plus four serialized 256-entry uint32 CDFs.
    cdf_bytes = struct.calcsize('<256I')
    return dict(role_cdf_bytes=2*cdf_bytes, shared_profile_header_bytes=core.PROFILE_HEADER.size,
                full_profile_bytes=core.PROFILE_HEADER.size+4*cdf_bytes)
