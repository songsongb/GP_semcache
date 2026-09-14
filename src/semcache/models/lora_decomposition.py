"""Component-explicit, inference-only decomposition of vanilla PEFT Linear LoRA."""
from dataclasses import dataclass
from typing import Any


@dataclass
class ProjectionComponents:
    base_output: Any
    lora_delta: Any
    combined_output: Any
    native_peft_output: Any


def validate_projection(module, adapter_name):
    import torch
    from peft.tuners.lora.layer import Linear
    if type(module) is not Linear or type(module.get_base_layer()) is not torch.nn.Linear:
        raise ValueError('Only unquantized PEFT Linear wrapping torch.nn.Linear is supported')
    if module.training or any(m.training for m in module.modules()):
        raise ValueError('Decomposition requires eval mode (deterministic dropout)')
    if module.merged or module.disable_adapters:
        raise ValueError('Merged or disabled adapters are unsupported')
    if list(module.active_adapters) != [adapter_name]:
        raise ValueError('Exactly one active adapter matching adapter_name is required')
    if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
        raise ValueError('Adapter missing from projection')
    if module.use_dora.get(adapter_name, False) or getattr(module, 'lora_variant', {}).get(adapter_name) is not None:
        raise ValueError('DoRA and other LoRA variants are unsupported')
    if module.fan_in_fan_out:
        raise ValueError('fan_in_fan_out is unsupported for OPT Linear')


def projection_parts(module, hidden_states, adapter_name):
    """Logical ES base and UD delta, using PEFT's actual cast/path/scaling order."""
    import torch
    validate_projection(module, adapter_name)
    with torch.inference_mode():
        base = module.get_base_layer()(hidden_states)
        a, b = module.lora_A[adapter_name], module.lora_B[adapter_name]
        x = module._cast_input_dtype(hidden_states, a.weight.dtype)
        delta = b(a(module.lora_dropout[adapter_name](x))) * module.scaling[adapter_name]
        combined = (base + delta).to(base.dtype)
    return base, delta, combined


def decompose_projection(peft_projection_module, hidden_states, adapter_name):
    import torch
    base, delta, combined = projection_parts(peft_projection_module, hidden_states, adapter_name)
    with torch.inference_mode():
        # Direct forward bypasses local reconstruction hooks, preventing recursion.
        native = peft_projection_module.forward(hidden_states)
    return ProjectionComponents(base, delta, combined, native)


def assert_decomposition(parts, tolerance=1e-6, require_nonzero=True):
    import math
    from semcache.evaluation.qkv_metrics import similarity
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    metrics = similarity(parts.native_peft_output, parts.combined_output)
    if metrics['max_abs_error'] > tolerance or metrics['relative_l2'] is None or metrics['relative_l2'] > tolerance:
        raise AssertionError(f'PEFT decomposition mismatch; inspect scaling/dtype/state: {metrics}')
    if require_nonzero and parts.lora_delta.count_nonzero().item() == 0:
        raise AssertionError('LoRA delta is zero; this cannot validate a nonzero adapter fixture')
    return metrics
