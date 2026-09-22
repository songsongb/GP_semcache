"""Serialized analytical tensor transfer costs; no sleeps or sockets."""
from .common import integer, nonnegative


def transfer_ms(byte_count, bandwidth_mbps=200):
    integer(byte_count, 'byte_count')
    nonnegative(bandwidth_mbps, 'bandwidth_mbps')
    if not bandwidth_mbps:
        raise ValueError('Bandwidth must be positive')
    return byte_count * 8 / (bandwidth_mbps * 1_000_000) * 1000


def directional_cost(ud_to_es_bytes, es_to_ud_bytes, bandwidth_mbps=200):
    up = transfer_ms(ud_to_es_bytes, bandwidth_mbps)
    down = transfer_ms(es_to_ud_bytes, bandwidth_mbps)
    return dict(ud_to_es_bytes=ud_to_es_bytes, es_to_ud_bytes=es_to_ud_bytes,
        total_network_bytes=ud_to_es_bytes+es_to_ud_bytes,
        ud_to_es_ms=up, es_to_ud_ms=down, total_network_ms=up+down,
        provenance='SIMULATED', bandwidth_mbps=bandwidth_mbps,
        transfer_schedule='serialized_REPRODUCTION_CHOICE')


def communication(dims, prompt_tokens, reused_tokens=0, *, hidden_element_bytes=2,
                  delta_element_bytes=2, bandwidth_mbps=200, boundary_transfers=True):
    integer(prompt_tokens, 'prompt_tokens', 1)
    integer(reused_tokens, 'reused_tokens')
    if reused_tokens > prompt_tokens:
        raise ValueError('Reuse exceeds prompt length')
    integer(hidden_element_bytes, 'hidden_element_bytes', 1)
    integer(delta_element_bytes, 'delta_element_bytes', 1)
    fresh = prompt_tokens - reused_tokens
    # Eq.17 projection exchange: ES->UD hidden; UD->ES three LoRA deltas.
    layer = directional_cost(3*fresh*dims.hidden_size*delta_element_bytes,
                             fresh*dims.hidden_size*hidden_element_bytes, bandwidth_mbps)
    # Explicit boundary choice: full input h0 up and full final hL down, once.
    # These are not eliminated by projection reuse.
    boundary_bytes = prompt_tokens*dims.hidden_size*hidden_element_bytes if boundary_transfers else 0
    boundary = directional_cost(boundary_bytes, boundary_bytes, bandwidth_mbps)
    request = directional_cost(layer['ud_to_es_bytes']*dims.layers + boundary_bytes,
                               layer['es_to_ud_bytes']*dims.layers + boundary_bytes, bandwidth_mbps)
    return dict(per_layer=layer, boundaries=boundary, request=request,
                layers=dims.layers, fresh_tokens=fresh,
                hidden_element_bytes=hidden_element_bytes, delta_element_bytes=delta_element_bytes,
                byte_accounting='tensor_payload_only_REPRODUCTION_CHOICE',
                boundary_policy=('full_h0_and_full_hL_REPRODUCTION_CHOICE' if boundary_transfers
                                 else 'projection_exchange_only_REPRODUCTION_CHOICE'),
                excluded='headers, RTT, serialization, contention, protocol overhead, overlap')
