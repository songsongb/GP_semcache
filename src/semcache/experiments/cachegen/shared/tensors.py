"""C1 device quantization/reconstruction; CPU symbols at the entropy boundary."""
import struct
from ..codecs import Baseline
from .core import decode


def quantize(k, v, *, device):
    import torch
    if k.device.type != 'cpu' or v.device.type != 'cpu' or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise ValueError('C1.5 requires captured FP16 K/V loaded on CPU')
    # Quantize BEFORE transferring integer symbols/scales to the CPU coder.
    encoded = Baseline('UNIFORM_INT8').encode(k.to(device), v.to(device))
    return tuple((symbols.cpu(), scales.cpu()) for symbols, scales in encoded)


def reconstruct(encoded, *, device):
    """Reconstruct on C1's device, then copy final FP16 values to storage CPU."""
    on_device = tuple((symbols.to(device), scales.to(device)) for symbols, scales in encoded)
    return tuple(x.cpu() for x in Baseline('UNIFORM_INT8').decode(on_device))


def histograms(encoded):
    import torch
    return [[torch.bincount((layer.to(torch.int64)+127).flatten(), minlength=255).tolist()
             for layer in symbols] for symbols, _ in encoded]


def prepare(profile, encoded):
    """No Q parameter exists at the entropy boundary. Dataset labels are unused."""
    import torch
    shape = tuple(encoded[0][0].shape)
    streams = tuple(bytes((symbols[start:end].to(torch.int16)+127).to(torch.uint8).flatten().tolist())
                    for start, end in profile.groups for symbols, _ in encoded)
    values = [value for _, scales in encoded for value in scales.flatten().tolist()]
    scales = struct.pack('<'+'f'*len(values), *values)
    return streams, scales, shape


def restore(streams, scales, shape, profile):
    import torch
    layers, tokens, hidden = shape
    values = struct.unpack('<'+'f'*(2*layers*tokens), scales)
    scale_tensors = torch.tensor(values, dtype=torch.float32).reshape(2, layers, tokens, 1)
    parts = [[], []]
    for i, (start, end) in enumerate(profile.groups):
        for component in range(2):
            symbols = (torch.tensor(list(streams[2*i+component]), dtype=torch.int16)-127).to(torch.int8)
            parts[component].append(symbols.reshape(end-start, tokens, hidden))
    return tuple((torch.cat(parts[c]), scale_tensors[c]) for c in range(2))


def check_roundtrip(profile, encoded, blob, *, reconstruction_device, reference=None):
    import torch
    recovered = restore(*decode(blob, profile), profile)
    for (symbols, scales), (rs, rc) in zip(encoded, recovered):
        if not torch.equal(symbols, rs) or not torch.equal(scales, rc):
            raise ValueError('Entropy codec changed UNIFORM_INT8 symbols/scales')
    expected = reconstruct(encoded, device=reconstruction_device)
    actual = reconstruct(recovered, device=reconstruction_device)
    if not all(torch.equal(a, b) for a, b in zip(expected, actual)):
        raise ValueError('Entropy codec introduced numerical loss')
    if reference is not None and not all(torch.equal(reference[n], x) for n, x in zip('kv', actual)):
        raise ValueError('Shared-CDF reconstruction differs from saved C1 UNIFORM_INT8')
    return actual
