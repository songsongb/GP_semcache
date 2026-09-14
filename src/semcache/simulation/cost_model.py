"""Paper Eqs. 15–20, per layer. SI units throughout.

Eq. 19 retains attention/FFN costs for ALL n tokens. Communication counts
activation elements, converted using explicit wire precision, NOT weight bits.
No encoder, cache lookup, embedding, logits, queueing or decode costs included.
"""
from abc import ABC, abstractmethod
import math


def positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')
    return value


def mbps_to_bits_per_second(mbps):
    return positive(mbps, 'Mbps') * 1_000_000


def bytes_to_bits(byte_count):
    return positive(byte_count, 'bytes', True) * 8


def tflops_to_flops_per_second(tflops):
    return positive(tflops, 'TFLOP/s') * 1_000_000_000_000


def communication_seconds(elements, element_size_bytes, bandwidth_mbps):
    return bytes_to_bits(positive(elements, 'elements', True) * positive(element_size_bytes, 'element size')) / mbps_to_bits_per_second(bandwidth_mbps)


def projection_savings(n_reused, d, r, layers=1):
    for name, value in [('n_reused', n_reused), ('d', d), ('r', r), ('layers', layers)]:
        positive(value, name, name == 'n_reused')
        if not isinstance(value, int):
            raise ValueError(f'{name} must be integer')
    return dict(base_flops_saved=6*n_reused*d*d*layers,
                lora_flops_saved=6*n_reused*d*r*layers,
                comm_elements_saved=4*n_reused*d*layers)


def paper_latency(*, n, n_reused, d, r, f_ES, f_UD, B, element_size_bytes):
    """Eq. 19 seconds/layer. f_ES/f_UD FLOP/s; B bit/s; precision bytes/element."""
    projection_savings(n_reused, d, r)
    if not isinstance(n, int) or isinstance(n, bool) or n < 1 or n_reused > n:
        raise ValueError('Require integer 0 <= n_reused <= n and n > 0')
    for name, value in [('f_ES', f_ES), ('f_UD', f_UD), ('B', B), ('element_size_bytes', element_size_bytes)]:
        positive(value, name)
    fresh = n - n_reused
    base = 6*fresh*d*d/f_ES
    lora = 6*fresh*d*r/f_UD
    comm = bytes_to_bits(4*fresh*d*element_size_bytes)/B
    rest = (18*n*d*d + 4*n*n*d + 16*n*d)/f_ES
    return dict(latency_s=max(base, lora+comm)+rest, base_projection_s=base,
                lora_projection_s=lora, communication_s=comm, remaining_es_s=rest)


class CostModel(ABC):
    @abstractmethod
    def estimate(self, *, tokens, reused_tokens, model_config, system_config):
        pass


class PaperCostModel(CostModel):
    def estimate(self, *, tokens, reused_tokens, model_config, system_config):
        # Explicit scalar whitelist: paper-reference objects cannot be inputs.
        d, r, layers = (model_config[k] for k in ('hidden_size', 'lora_rank', 'layers'))
        savings = projection_savings(reused_tokens, d, r, layers)
        timing = paper_latency(n=tokens, n_reused=reused_tokens, d=d, r=r,
            f_ES=tflops_to_flops_per_second(system_config['es_tflops']),
            f_UD=tflops_to_flops_per_second(system_config['ud_tflops']),
            B=mbps_to_bits_per_second(system_config['bandwidth_mbps']),
            element_size_bytes=system_config['communication_element_bytes'])
        return dict(**{k: v*layers for k, v in timing.items()}, **savings,
                    comm_bytes_saved=savings['comm_elements_saved']*system_config['communication_element_bytes'],
                    metric_source='SIMULATED', metric_scope='all-layer prefill Eq.19; excluded overheads documented')
