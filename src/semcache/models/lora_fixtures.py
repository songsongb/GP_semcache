"""Synthetic correctness fixtures, never trained personalization adapters."""
import hashlib
import math
from pathlib import Path

DEFAULT_LORA = dict(rank=8, alpha=8, dropout=0.0,
                    target_modules=['q_proj', 'k_proj', 'v_proj'])


def base_weight_fingerprint(adapter, layers=None):
    import torch
    result = {}
    for layer in range(len(adapter.layers)) if layers is None else layers:
        for name, module in adapter.projection_modules(layer).items():
            base = adapter.base_projection(module)
            for kind in ('weight', 'bias'):
                value = getattr(base, kind, None)
                if value is None:
                    continue
                value = value.detach().cpu().contiguous()
                result[f'{layer}.{name}.{kind}'] = dict(shape=list(value.shape), dtype=str(value.dtype),
                    norm=value.double().norm().item(), sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())
    return result


def assert_frozen_base(model):
    for name, parameter in model.named_parameters():
        if '.lora_A.' not in name and '.lora_B.' not in name and parameter.requires_grad:
            raise AssertionError(f'Non-adapter parameter is trainable: {name}')


def initialize_fixture(adapter, name, seed, scale=0.01):
    """CPU float32 N(0, scale²), local generator, layer order then q/k/v then A/B."""
    import torch
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Fixture scale must be finite and positive')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    with torch.no_grad():
        for layer in range(len(adapter.layers)):
            for module in adapter.projection_modules(layer).values():
                for matrix in (module.lora_A[name], module.lora_B[name]):
                    value = torch.randn(matrix.weight.shape, generator=generator, dtype=torch.float32) * scale
                    matrix.weight.copy_(value.to(matrix.weight))
                    if matrix.bias is not None:
                        raise ValueError('Fixture LoRA bias is unsupported')


def create_controlled_users(base_model, config=None):
    import torch
    from peft import LoraConfig, get_peft_model
    from .model_adapter import OPTModelAdapter
    c = {**DEFAULT_LORA, **(config or {})}
    if c.get('adapter_source', 'controlled_fixture') not in ('controlled_fixture', 'controlled_test_adapter'):
        raise ValueError('Use load_external_adapter for externally trained adapters')
    if set(c['target_modules']) != set(DEFAULT_LORA['target_modules']):
        raise ValueError('Milestone 4 requires exactly q_proj/k_proj/v_proj targets')
    if type(c['rank']) is not int or c['rank'] <= 0 or not math.isfinite(c['alpha']) or c['alpha'] <= 0:
        raise ValueError('Positive rank and finite positive alpha required')
    if not 0 <= c['dropout'] < 1:
        raise ValueError('Dropout must be in [0, 1)')
    fixture = c.get('fixture', {})
    seeds = [fixture.get('user_a_seed', 101), fixture.get('user_b_seed', 202)]
    if seeds[0] == seeds[1]:
        raise ValueError('Two users require distinct fixture seeds')
    scale = fixture.get('initialization_scale', 0.01)
    before = base_weight_fingerprint(OPTModelAdapter(base_model))
    lora = LoraConfig(r=c['rank'], lora_alpha=c['alpha'], lora_dropout=c['dropout'],
                      target_modules=c['target_modules'], bias='none', task_type='CAUSAL_LM')
    # PEFT constructors consume random numbers, but their initialization is replaced.
    # Preserve caller RNG state, including already-initialized CUDA generators.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        model = get_peft_model(base_model, lora, adapter_name='user_a')
        model.add_adapter('user_b', lora)
    adapter = OPTModelAdapter(model)
    metadata = {}
    for name, seed in zip(('user_a', 'user_b'), seeds):
        initialize_fixture(adapter, name, seed, scale)
        metadata[name] = dict(adapter_name=name, seed=seed, rank=c['rank'], alpha=c['alpha'], dropout=c['dropout'],
            target_modules=c['target_modules'], trained_adapter=False, adapter_source='controlled_fixture',
            initialization_strategy='CPU float32 normal(0, scale^2); layer/qkv/A-B order; local per-user generator',
            initialization_scale=scale, paper_setting_rank=c['rank'] == 8)
    model.set_adapter('user_a')
    model.eval()
    assert_frozen_base(model)
    if before != base_weight_fingerprint(adapter):
        raise AssertionError('Base QKV changed during adapter creation')
    return model, metadata


def load_external_adapter(base_model, path, adapter_name='external'):
    """Local PEFT adapter interface; caller supplies training provenance separately."""
    from peft import PeftConfig, PeftModel
    from .model_adapter import OPTModelAdapter
    from .lora_decomposition import validate_projection
    path = Path(path)
    if not path.is_dir():
        raise ValueError('External adapter must be an existing local directory')
    c = PeftConfig.from_pretrained(str(path), local_files_only=True)
    if (str(c.peft_type) not in ('PeftType.LORA', 'LORA') or c.bias != 'none' or c.modules_to_save
            or c.use_dora or getattr(c, 'use_rslora', False)
            or getattr(c, 'lora_bias', False) or getattr(c, 'alora_invocation_tokens', None)
            or set(c.target_modules) != set(DEFAULT_LORA['target_modules'])):
        raise ValueError('External adapter must be vanilla QKV-only LoRA with no base changes')
    before = base_weight_fingerprint(OPTModelAdapter(base_model))
    model = PeftModel.from_pretrained(base_model, str(path), adapter_name=adapter_name,
                                     local_files_only=True, is_trainable=False).eval()
    adapter = OPTModelAdapter(model)
    for layer in range(len(adapter.layers)):
        for module in adapter.projection_modules(layer).values():
            validate_projection(module, adapter_name)
    assert_frozen_base(model)
    if before != base_weight_fingerprint(adapter):
        raise AssertionError('External adapter changed base QKV')
    return model
