"""Recorded C1 quantization device is part of the experiment, not a fallback hint."""
from ..common import file_hash, read_json


def canonical_device(device):
    if device == 'cpu':
        return 'cpu'
    if device == 'cuda':
        return 'cuda:0'
    if isinstance(device, str) and device.startswith('cuda:') and device[5:].isdigit():
        return f'cuda:{int(device[5:])}'
    raise ValueError(f'Unsupported/missing recorded C1 quantization device: {device!r}')


def make_contract(reference, requested=None, *, cuda_available=False, cuda_device_count=0):
    requested = reference if requested is None else requested
    resolved = canonical_device(requested)
    if resolved != canonical_device(reference):
        raise ValueError('Quantization device must equal recorded C1 reference device; no CPU fallback')
    if resolved.startswith('cuda:') and (not cuda_available or int(resolved[5:]) >= cuda_device_count):
        raise ValueError('Recorded C1 CUDA quantization/reconstruction device unavailable; no CPU fallback')
    return dict(reference_c1_device=reference, quantization_device_requested=requested,
                quantization_device_resolved=resolved, reconstruction_device=resolved,
                entropy_coder_device='cpu', quantization_device_provenance='MEASURED_C1_REFERENCE_DEVICE',
                profile_symbol_generation_device=resolved)


def resolve_contract(root):
    import torch
    path = root/'environment.json'
    recorded = (read_json(path).get('benchmark') or {}).get('device')
    contract = make_contract(recorded, cuda_available=torch.cuda.is_available(),
                             cuda_device_count=torch.cuda.device_count())
    contract['c1_environment_sha256'] = file_hash(path)
    return contract


def verify_contract(profile_contract, active_contract):
    if not profile_contract or profile_contract != active_contract:
        raise ValueError('Obsolete/incompatible profile quantization device contract; '
                         'regenerate C1.5 profiles from calibration on the recorded C1 device')
    resolved = canonical_device(active_contract['reference_c1_device'])
    if (any(active_contract[k] != resolved for k in ('quantization_device_resolved',
            'reconstruction_device', 'profile_symbol_generation_device'))
            or canonical_device(active_contract['quantization_device_requested']) != resolved
            or active_contract['entropy_coder_device'] != 'cpu'):
        raise ValueError('Profile/evaluation/reference C1 device mismatch')
