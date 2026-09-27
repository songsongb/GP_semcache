"""QL2 adapter to existing B2 framing/transform/coder and frozen profiles.

QL2 shifted codes are losslessly centered before the SAME +127 mapping and
modulus-255 B2 transform. No bin-dependent modulus, new CDF or profile fit.
FP16 source maxabs is exactly widened into existing FP32 metadata slots.
The policy object owns centering/dequantization; the codec sees bytes only.
"""
import struct
from pathlib import Path

from ..b2 import format as fmt
from ..common import read_json, file_hash, digest
from ..shared import core


def load_profile(directory):
    directory = Path(directory).resolve()
    manifest_path = directory/'b2_profile_manifest.json'
    state = read_json(directory/'manifest.json')
    if state.get('profile_status') != 'FROZEN' or state.get('profile_manifest_sha256') != file_hash(manifest_path):
        raise ValueError('B2 profile manifest not frozen/hash mismatch')
    manifest = read_json(manifest_path)
    if (manifest.get('status') != 'FROZEN_BEFORE_EVALUATION' or
            manifest.get('fitting_partition') != 'calibration' or
            manifest.get('base_contract_sha256') != digest(manifest['base_contract'])):
        raise ValueError('An existing frozen B2 profile is required; smoke never fits CDFs')
    entry = manifest['profiles'][fmt.MODES[1]]
    path = (directory/entry['file']).resolve()
    if not path.is_relative_to(directory):
        raise ValueError('B2 profile path escapes profile directory')
    profile = fmt.Profile.from_bytes(path.read_bytes(), entry['sha256'])
    expected = fmt.Profile(fmt.MODES[1], tuple(core.cdf_from_counts(c) for c in entry['calibration_counts']))
    if (profile != expected or digest(entry['calibration_counts']) != entry['counts_sha256'] or
            len(profile.to_bytes()) != entry['serialized_bytes']):
        raise ValueError('Profile differs from existing frozen calibration counts/bytes')
    if profile.mode != fmt.MODES[1]:
        raise ValueError('Expected existing B2 anchor/mod-residual K+V profile')
    return profile, dict(profile_manifest_path=str(manifest_path), profile_manifest_sha256=file_hash(manifest_path),
        b2_manifest_sha256=file_hash(directory/'manifest.json'),
        profile_path=str(path), profile_sha256=profile.sha256, profile_bytes=len(profile.to_bytes()),
        profile_policy='Existing Uniform INT8 B2 calibration CDF reused unchanged; compatibility only')


def roundtrip(encoded, profile):
    """One encode/decode, no timing repeats; returns blob, reconstruction, proof."""
    import torch
    if len(encoded) != 2 or tuple(e.role for e in encoded) != ('K', 'V'):
        raise ValueError('Separate K then V policies required')
    shape = tuple(encoded[0].symbols.shape)
    fmt.validate_shape(shape, expected_tokens=3)
    if any(tuple(e.symbols.shape) != shape or tuple(e.storage_metadata.shape) != (32, 3, 1) or
           e.layer_indices != tuple(range(32)) for e in encoded):
        raise ValueError('Storage smoke requires all 32 w=3 layers with matching metadata')
    signed = [e.signed_symbols.cpu() for e in encoded]
    if any(s.min().item() < -127 or s.max().item() > 127 for s in signed):
        raise ValueError('Policy symbols outside existing B2 finite signed alphabet')
    domains = tuple(bytes((s.to(torch.int16)+127).to(torch.uint8).flatten().tolist()) for s in signed)
    maxima = struct.pack('<192f', *[v for e in encoded for v in e.storage_metadata.cpu().flatten().tolist()])
    streams = fmt.streams_from_domains(profile.mode, domains, shape, expected_tokens=3)
    blob = fmt.encode(profile, streams, maxima, shape, expected_tokens=3)
    decoded, actual_maxima, actual_shape = fmt.decode(blob, profile, expected_tokens=3)
    recovered = fmt.domains_from_streams(profile.mode, decoded, actual_shape, expected_tokens=3)
    if actual_shape != shape or actual_maxima != maxima:
        raise ValueError('B2 storage changed shape/maxabs metadata')
    values = torch.tensor(struct.unpack('<192f', actual_maxima)).reshape(2, 32, 3, 1)
    mismatches, restored, reconstructed = [], [], []
    for i, (e, data) in enumerate(zip(encoded, recovered)):
        signed_symbols = (torch.frombuffer(bytearray(data), dtype=torch.uint8).to(torch.int16)-127)
        actual = e.restore_storage(signed_symbols.reshape(shape), values[i])
        mismatch = int((actual.symbols != e.symbols).sum().item())
        mismatches.append(mismatch)
        if mismatch or not torch.equal(actual.storage_metadata, e.storage_metadata):
            raise ValueError('Storage symbol/maxabs corruption')
        restored.append(actual)
        reconstructed.append(restored[-1].reconstructed)
        if not torch.equal(reconstructed[-1], e.reconstructed):
            raise ValueError('Storage introduced reconstruction error')
    return blob, tuple(reconstructed), dict(
        storage_roundtrip_symbol_mismatch=dict(zip(('K', 'V'), mismatches)),
        symbol_equality=True, maxima_metadata_exact=True, shape=list(shape),
        symbols_dtype=str(encoded[0].symbols.dtype), source_maxabs_dtype=str(encoded[0].storage_metadata.dtype),
        stored_metadata_dtype='FP32 (losslessly widened maxabs, not Uniform INT8 step)',
        reconstruction_equality=True, **fmt.inspect(blob, profile, expected_tokens=3)[3])
