"""Frozen C1.5C K20/V16 physical K/V representation for SemCache entries."""
from dataclasses import dataclass
from pathlib import Path
import struct
import time

from ..b2 import format as fmt
from ..c15c.policy import QuantizedTensor, UniformKVPolicy
from ..c15c.rate_calibration import observe
from ..shared import core

MODE_RAW = 'RAW_FP16'
MODE_COMPRESSED = 'COMPRESSED_KV_K20_V16'
PROFILE_SHA256 = '8defa27a83e8ad16e9c7efa0f40e68c302aef90363e14a1451288a3bf433cb2c'
DEFAULT_PROFILE = (Path(__file__).resolve().parents[5] /
    'results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin')
POLICY = UniformKVPolicy(20, 16)


@dataclass(frozen=True)
class CompressedKVPayload:
    bitstream: bytes  # In-memory CPU bytes; B2 frame includes shape and FP32 maxabs metadata.
    shape: tuple[int, int, int]
    dtype: str
    profile_sha256: str
    role_payload_bytes: tuple[int, int, int, int]
    local_metadata_bytes: int

    @property
    def payload_bytes(self):
        return sum(self.role_payload_bytes)


class DecodedCacheEntryView:
    """Temporary retrieval view; the resident CacheEntry keeps only Q and bytes."""
    def __init__(self, resident, tensors, symbols, timings_ms):
        self.resident = resident
        self.tensors = tensors
        self.kv_symbols = symbols
        self.storage_timings_ms = timings_ms

    def __getattr__(self, name):
        return getattr(self.resident, name)


class FrozenK20V16Codec:
    def __init__(self, profile_path=DEFAULT_PROFILE, *, quantization_device='cuda:0',
                 decode_device=None, expected_hidden=2560, instrument=False,
                 coder_backend=fmt.REFERENCE_CODER):
        self.profile_path = Path(profile_path).resolve()
        if not self.profile_path.is_file():
            raise FileNotFoundError(f'Frozen C1.5C profile missing: {self.profile_path}')
        data = self.profile_path.read_bytes()
        self.profile = fmt.Profile.from_bytes(data, PROFILE_SHA256)
        if self.profile.mode != fmt.MODES[1]:
            raise ValueError('Frozen profile must use B2_ANCHOR_MOD_RESIDUAL_KV')
        self.profile_bytes = len(data)  # Shared cache-level cost, never charged per entry.
        self.quantization_device = str(quantization_device)
        self.decode_device = str(decode_device or quantization_device)
        self.expected_hidden = expected_hidden
        self.instrument = instrument
        fmt.coder_functions(coder_backend)  # Explicit opt-in; defaults to frozen reference bits.
        self.coder_backend = coder_backend

    @staticmethod
    def _clock(device):
        import torch
        if str(device).startswith('cuda:'):
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _elapsed(self, start, device):
        return 1000*(self._clock(device)-start) if self.instrument else None

    def make_entry(self, entry_type, cluster_id, token_ids, positions, tensors, storage_device):
        import torch
        if len(token_ids) != 3 or set(tensors) != set(range(32)):
            raise ValueError('Frozen K20/V16 requires all 32 OPT layers and w=3')
        for qkv in tensors.values():
            if (len(qkv) != 3 or any(t.shape != (1, 3, self.expected_hidden) or t.dtype != torch.float16
                                       for t in qkv)):
                raise ValueError('Frozen K20/V16 requires FP16 Q/K/V [1,3,OPT hidden]')
        raw_size = sum(t.numel()*t.element_size() for qkv in tensors.values() for t in qkv)
        start = self._clock(self.quantization_device) if self.instrument else None
        k, v = (torch.cat([tensors[layer][index].detach() for layer in range(32)], dim=0)
                .to(self.quantization_device) for index in (1, 2))
        quantized = (POLICY.quantize(k, 'K'), POLICY.quantize(v, 'V'))
        quantize_ms = self._elapsed(start, self.quantization_device)
        for role, result in zip(('K', 'V'), quantized):
            observe(result, str((cluster_id, tuple(token_ids))), 'C2_insert_'+role)
        start = self._clock(self.quantization_device) if self.instrument else None
        domains = tuple(bytes((result.signed_symbols.cpu().to(torch.int16)+127)
                              .to(torch.uint8).flatten().tolist()) for result in quantized)
        shape = tuple(quantized[0].symbols.shape)
        maxima = struct.pack('<192f', *[value for result in quantized
            for value in result.storage_metadata.cpu().flatten().tolist()])
        streams = fmt.streams_from_domains(self.profile.mode, domains, shape, expected_tokens=3)
        bitstream = fmt.encode(self.profile, streams, maxima, shape, expected_tokens=3,
                               coder_backend=self.coder_backend)
        _, _, _, sizes = fmt.inspect(bitstream, self.profile, expected_tokens=3)
        encode_ms = self._elapsed(start, self.quantization_device)
        q_tensors = {layer: tensors[layer][0].detach().to(storage_device).clone().contiguous()
                     for layer in range(32)}
        payload = CompressedKVPayload(bitstream, shape, 'torch.float16', PROFILE_SHA256,
            tuple(sizes[name] for name in fmt.STREAM_NAMES), sizes['local_metadata_bytes'])
        if payload.payload_bytes+payload.local_metadata_bytes != len(bitstream):
            raise ValueError('B2 payload/local metadata accounting mismatch')
        return entry_type(cluster_id, tuple(token_ids), positions, raw_size, q_tensors=q_tensors,
                          compressed_kv=payload, storage_timings_ms=dict(
                              quantize_ms=quantize_ms, encode_ms=encode_ms))

    def decode_entry(self, resident):
        import torch
        payload = resident.compressed_kv
        if payload is None or payload.profile_sha256 != PROFILE_SHA256 or payload.dtype != 'torch.float16':
            raise ValueError('Compressed entry/profile/dtype mismatch')
        start = self._clock(self.decode_device) if self.instrument else None
        streams, maxima, shape = fmt.decode(payload.bitstream, self.profile, expected_tokens=3,
                                           coder_backend=self.coder_backend)
        if shape != payload.shape or shape != (32, 3, self.expected_hidden):
            raise ValueError('Compressed entry shape mismatch')
        domains = fmt.domains_from_streams(self.profile.mode, streams, shape, expected_tokens=3)
        values = torch.tensor(struct.unpack('<192f', maxima), dtype=torch.float16).reshape(2, 32, 3, 1)
        signed = tuple((torch.frombuffer(bytearray(data), dtype=torch.uint8).to(torch.int16)-127)
                       .reshape(shape) for data in domains)
        decode_ms = self._elapsed(start, self.decode_device)
        start = self._clock(self.decode_device) if self.instrument else None
        reconstructed, symbols = [], []
        for index, bins in enumerate((20, 16)):
            limit = bins//2-1
            code = (signed[index]+limit).to(torch.int8).to(self.decode_device)
            if (code < 0).any() or (code > 2*limit).any():
                raise ValueError('Decoded K/V symbol outside frozen quantizer range')
            symbols.append(code)
            quantized = QuantizedTensor(POLICY.name, 'KV'[index], tuple(range(32)),
                torch.full((32,), bins, dtype=torch.float32, device=self.decode_device),
                torch.full((32,), limit, dtype=torch.float32, device=self.decode_device),
                values[index].to(self.decode_device), code, torch.float16)
            reconstructed.append(quantized.reconstructed)
        dequantize_ms = self._elapsed(start, self.decode_device)
        tensors = {layer: (resident.q_tensors[layer], reconstructed[0][layer:layer+1],
                           reconstructed[1][layer:layer+1]) for layer in range(32)}
        return DecodedCacheEntryView(resident, tensors, tuple(symbols),
            dict(decode_ms=decode_ms, dequantize_ms=dequantize_ms))
