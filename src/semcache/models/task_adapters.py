"""Task-trained QKV LoRA helpers; controlled C6-A fixtures are untouched."""
from pathlib import Path
from semcache.experiments.m9b_semantic_workload import MODEL_ID, MODEL_REVISION
from .lora_fixtures import DEFAULT_LORA, base_weight_fingerprint, assert_frozen_base

USERS = ('user_a', 'user_b')
ADAPTER_CONFIG = dict(r=8, lora_alpha=8, lora_dropout=0.0,
    target_modules=list(DEFAULT_LORA['target_modules']), bias='none', task_type='CAUSAL_LM')


def validate_task_config(config):
    """Reject variants which invalidate the existing decomposition path."""
    if (str(config.peft_type) not in ('PeftType.LORA', 'LORA')
            or str(config.task_type) not in ('TaskType.CAUSAL_LM', 'CAUSAL_LM')
            or config.r != 8 or config.lora_alpha != 8 or config.lora_dropout != 0
            or config.bias != 'none' or set(config.target_modules) != set(DEFAULT_LORA['target_modules'])
            or config.base_model_name_or_path != MODEL_ID or config.revision != MODEL_REVISION):
        raise ValueError('Task adapters require pinned OPT and rank8/alpha8/dropout0 QKV-only causal LoRA')
    for field in ('modules_to_save', 'use_dora', 'use_rslora', 'lora_bias', 'alora_invocation_tokens',
                  'rank_pattern', 'alpha_pattern', 'layers_to_transform', 'layers_pattern',
                  'layer_replication', 'target_parameters', 'trainable_token_indices', 'fan_in_fan_out',
                  'use_qalora', 'megatron_config', 'arrow_config', 'ensure_weight_tying', 'exclude_modules'):
        if getattr(config, field, None):
            raise ValueError(f'Unsupported task adapter variant: {field}')


def activate_task_user(model, name):
    if name not in USERS or name not in model.peft_config:
        raise ValueError('Expected user_a or user_b task adapter')
    model.set_adapter(name)
    # PEFT set_adapter may re-enable gradients even for evaluation-only adapters.
    model.requires_grad_(False)
    model.eval()
    assert_frozen_base(model)


def select_training_user(model, name):
    """Only one user's A/B matrices may receive gradients in an optimizer."""
    if name not in USERS or name not in model.peft_config:
        raise ValueError('Expected user_a or user_b task adapter')
    model.set_adapter(name)
    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad_(f'.lora_A.{name}.' in parameter_name
                                 or f'.lora_B.{name}.' in parameter_name)
    assert_frozen_base(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('No trainable task-adapter parameters')
    return parameters


def load_two_task_users(base_model, user_a, user_b):
    from peft import PeftConfig, PeftModel
    from .model_adapter import OPTModelAdapter
    from .lora_decomposition import validate_projection
    paths = [Path(user_a), Path(user_b)]
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError('Two distinct local adapter directories required')
    for path in paths:
        if not (path/'adapter_config.json').is_file():
            raise ValueError(f'Missing standard local PEFT adapter: {path}')
        validate_task_config(PeftConfig.from_pretrained(str(path), local_files_only=True))
    before = base_weight_fingerprint(OPTModelAdapter(base_model))
    model = PeftModel.from_pretrained(base_model, str(paths[0]), adapter_name='user_a',
                                     local_files_only=True, is_trainable=False)
    model.load_adapter(str(paths[1]), adapter_name='user_b', local_files_only=True, is_trainable=False)
    adapter = OPTModelAdapter(model)
    for user in USERS:
        activate_task_user(model, user)
        for layer in range(len(adapter.layers)):
            for projection in adapter.projection_modules(layer).values():
                validate_projection(projection, user)
                if projection.r[user] != 8 or projection.lora_alpha[user] != 8:
                    raise ValueError('Loaded projection config differs from C6 task architecture')
    if before != base_weight_fingerprint(adapter):
        raise ValueError('Loading task adapters changed base QKV weights')
    activate_task_user(model, 'user_a')
    return model
