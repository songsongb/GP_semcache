"""Pinned CacheGen active vectorized formula, preserving operation order/dtypes.

No anchor/delta transform or entropy coding belongs in this module. In
particular, shifted rounding is NOT round(x/step) followed by an integer shift.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class QuantizedTensor:
    policy: str
    role: str
    layer_indices: tuple
    bins: object
    limits: object
    maxabs: object
    symbols: object  # Released shifted int8 codes, not signed centered codes.
    input_dtype: object

    @property
    def scale(self):
        """Reconstruction step; stored source metadata is maxabs, not this step."""
        return self.maxabs / self.limits[:, None, None]

    @property
    def signed_symbols(self):
        """Lossless storage view in B2's existing signed int8 alphabet."""
        import torch
        if (self.symbols < 0).any() or (self.symbols > 2*self.limits[:, None, None]).any():
            raise ValueError('Released symbols outside nominal range; no epsilon/repair is applied')
        return (self.symbols.to(torch.int16)-self.limits.to(torch.int16)[:, None, None]).to(torch.int8)

    @property
    def storage_metadata(self):
        return self.maxabs

    def restore_storage(self, signed_symbols, metadata):
        """Policy-owned inverse mapping; storage never needs to know its bins."""
        import torch
        symbols = (signed_symbols.to(device=self.symbols.device, dtype=torch.int16)+
                   self.limits.to(torch.int16)[:, None, None]).to(self.symbols.dtype)
        maxabs = metadata.to(device=self.maxabs.device, dtype=self.maxabs.dtype)
        return replace(self, symbols=symbols, maxabs=maxabs)

    def dequantize(self, *, symbols=None, maxabs=None):
        # cachegen_decoder.do_dequantize, with the decoder's out.float() input.
        c = self.limits[:, None, None]
        t = (self.symbols if symbols is None else symbols).float()
        t = t - c
        t = t / c
        t = t * (self.maxabs if maxabs is None else maxabs)
        return t

    @property
    def reconstructed(self):
        # CacheGenDeserializer.from_bytes(fmt='huggingface') finishes in FP16.
        return self.dequantize().to(self.input_dtype)


class ReleasedFormulaPolicy:
    """One released formula shared by QL2 and constant-bin uniform policies."""
    def profile(self):
        return {role: [self.bins_for(l, role) for l in range(32)] for role in ('K', 'V')}

    def quantize(self, x, role, *, layer_indices=None):
        """[L,T,hidden], or OPT [L,heads,T,head_dim]; reduce merged hidden.

        Partial synthetic layer sets require explicit indices. All-zero vectors
        deliberately retain released 0*inf -> NaN -> int8 behavior: there is no
        epsilon/zero guard in the source. Do not silently repair it here.
        """
        import torch
        from ..common import from_heads
        if x.ndim == 4:
            x = from_heads(x)
        if x.ndim != 3 or any(n < 1 for n in x.shape) or x.dtype not in (torch.float16, torch.float32):
            raise ValueError('Expected nonempty FP16/FP32 [L,T,hidden] tensor')
        if not torch.isfinite(x).all():
            raise ValueError('Finite input required')
        if layer_indices is None:
            if x.shape[0] != 32:
                raise ValueError('Partial tensors require explicit layer_indices')
            layer_indices = tuple(range(32))
        layer_indices = tuple(layer_indices)
        if len(layer_indices) != x.shape[0] or len(set(layer_indices)) != len(layer_indices):
            raise ValueError('Distinct layer indices must match tensor layers')
        # Released make_key_bins/make_value_bins use FP32 torch.zeros.
        bins = torch.tensor([self.bins_for(l, role) for l in layer_indices], dtype=torch.float32, device=x.device)
        limits = bins // 2 - 1
        maxabs = torch.amax(torch.abs(x), dim=-1, keepdim=True)
        factor = limits[:, None, None] / maxabs
        symbols = torch.round(x * factor + limits[:, None, None]).to(torch.int8)
        return QuantizedTensor(self.name, role, layer_indices, bins, limits, maxabs, symbols, x.dtype)


def validate_layer_role(layer, role):
    if type(layer) is not int or not 0 <= layer < 32 or role not in ('K', 'V'):
        raise ValueError('Expected layer 0..31 and role K or V')


def validate_bins(bins):
    # Released shifted int8 codes must fit 0..bins-2 without wrapping.
    if type(bins) is not int or bins < 4 or bins > 128 or bins % 2:
        raise ValueError('Bins must be even integers in [4,128] (positive C; shifted int8 range)')
    return bins


DEFAULT_CANDIDATE_BINS = tuple(range(8, 33, 2))


def validate_candidates(candidates):
    candidates = tuple(candidates)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError('Candidate bins must be nonempty and distinct')
    return tuple(sorted(validate_bins(b) for b in candidates))


@dataclass(frozen=True)
class ReleasedQL2Policy(ReleasedFormulaPolicy):
    name: str = 'CACHEGEN_RELEASED_QL2'

    def bins_for(self, layer, role):
        validate_layer_role(layer, role)
        return 32 if layer < (10 if role == 'K' else 2) else 16


@dataclass(frozen=True)
class UniformKVPolicy(ReleasedFormulaPolicy):
    k_bins: int
    v_bins: int

    def __post_init__(self):
        validate_bins(self.k_bins)
        validate_bins(self.v_bins)

    @property
    def name(self):
        return f'UNIFORM_K{self.k_bins}_V{self.v_bins}'

    def bins_for(self, layer, role):
        validate_layer_role(layer, role)
        return self.k_bins if role == 'K' else self.v_bins


CACHEGEN_RELEASED_QL2 = ReleasedQL2Policy()
