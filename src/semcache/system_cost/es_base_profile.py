"""Strict ES profile contracts and control-plane extraction; no model imports."""
import copy
from semcache.models.tokenizer_provenance import validate_tokenizer_snapshot, is_snapshot_commit
from .common import integer, nonnegative
from .profiles import pair_key, reuse_gate

MEASUREMENT_LABEL = 'MEASURED_ES_BASE_ONLY'
PROXY_LABEL = 'SIMULATED_PROXY_DOUBLE_COUNTS_LORA'
BASE_MODES = ('ES_BASE_NATIVE', 'ES_BASE_SEMCACHE_REUSE')
CONTROL_FIELDS = {
    'semantic_encode_ms': ('semantic_encode_ms',),
    'clustering_ms': ('cluster_assign_update_ms',),
    'subsequence_ms': ('subsequence_extract_ms',),
    'lookup_ms': ('cache_lookup_ms',),
    'hit_selection_ms': ('hit_selection_ms',),
    'attention_impact_ms': ('attention_impact_ms',),
    'policy_ms': ('cache_policy_ms', 'chu_update_ms', 'pbr_update_ms'),
}
PAPER_SETTINGS = dict(impact_reducer_type='paper_row_l2_sum', cluster_schedule_mode='buffered',
    cluster_update_interval_queries=100, rho=.8, history_lambda=100, pbr_interval_queries=100)


def assert_no_lora(model, adapter):
    """Reject even disabled/merged PEFT wrappers, rather than trust an enable flag."""
    if hasattr(model, 'peft_config') or getattr(model, '_hf_peft_config_loaded', False):
        raise ValueError('Base-only profiler requires a bare OPT checkpoint, not a PEFT wrapper')
    for name, module in model.named_modules():
        if any(hasattr(module, attr) for attr in ('lora_A', 'lora_B', 'lora_embedding_A')):
            raise ValueError(f'LoRA-capable module in base-only model: {name}')
    for layer in range(len(adapter.layers)):
        for module in adapter.projection_modules(layer).values():
            if not all(hasattr(module, attr) for attr in ('weight', 'in_features', 'out_features')):
                raise ValueError('Expected plain base projection')
    return dict(lora_absent=True, active_lora_projection_count=0,
                validation='bare_OPT_module_tree_and_plain_linear_QKV')


def validate_fresh_m85(row):
    for field, value in PAPER_SETTINGS.items():
        if row.get(field) != value:
            raise ValueError(f'Fresh M8.5 paper-aligned setting required: {field}={value}')
    if row.get('requested_prompt_tokens') != 32 or row.get('prompt_tokens') != 32:
        raise ValueError('M9-A.1 primary profile is exactly 32 prompt tokens')
    if row.get('query_id') != 'same_user_exact':
        raise ValueError('M9-A.1 starts with same_user_exact only')
    if row.get('attention_implementation') != 'eager':
        raise ValueError('Fresh M8.5 eager attention required')
    if row.get('model_id') == 'facebook/opt-2.7b' and row.get('dtype') != 'torch.float16':
        raise ValueError('Primary OPT-2.7B profile requires float16')
    if not is_snapshot_commit(row.get('model_revision')):
        raise ValueError('Resolved model snapshot commit is required')
    validate_tokenizer_snapshot(row)
    if row.get('warmup_state_semantics') != 'fresh_discarded_trace':
        raise ValueError('Fresh/discarded warmup state required')
    integer(row.get('warmup_runs'), 'warmup_runs', 1)
    integer(row.get('measured_runs'), 'measured_runs', 1)
    if row['mode'] == 'SEMCACHE_PHYSICAL_REUSE':
        reuse_gate(row)


def control_plane(row):
    """Disjoint M8 CPU scopes; materialization is already inside cache_policy."""
    values = {name: sum(nonnegative(row.get(field), field) for field in fields)
              for name, fields in CONTROL_FIELDS.items()}
    total = row['request_wall_ms'] - row['prefill_wall_ms'] - row['tokenization_ms']
    residual = total - sum(values.values())
    if residual < -1e-6:
        raise ValueError('Overlapping/inconsistent M8 control timers')
    return dict(**values, control_unaccounted_ms=max(0., residual),
                total_control_ms=total, provenance='MEASURED',
                source='fresh M8.5 PEFT control only; original prefill excluded',
                policy_scope='admission/eviction/materialization + CHU + PBR',
                tokenization_excluded=True)


def validate_base_profile(row, expected_mode):
    profile = row.get('es_base_profile') or {}
    if (profile.get('mode') != expected_mode or profile.get('measurement_label') != MEASUREMENT_LABEL
            or profile.get('provenance') != 'MEASURED' or profile.get('lora_absent') is not True
            or profile.get('active_lora_projection_count') != 0
            or profile.get('personalized_output_quality_claimed') is not False):
        raise ValueError('strict-base-only requires a measured LoRA-absent ES profile')
    if (row.get('es_base_compute_provenance') != 'MEASURED'
            or row.get('es_base_compute_measurement_label') != MEASUREMENT_LABEL
            or row.get('es_base_compute_scope') != 'prefill_base_only_excluding_lora_control'):
        raise ValueError('Conflicting base-only timing provenance or scope')
    validate_fresh_m85(row)
    if tuple(profile.get('source_pair_key', ())) != pair_key(row):
        raise ValueError('ES base profile identity differs from M8 source')
    if not row.get('fresh_m85_profile_id') or profile.get('fresh_m85_profile_id') != row['fresh_m85_profile_id']:
        raise ValueError('Missing fresh M8.5 profile run identity')
    if not profile.get('source_m8_sha256') or not profile.get('source_environment_sha256'):
        raise ValueError('Missing fresh source artifact hashes')
    for field in ('warmup_runs', 'measured_runs', 'dtype', 'attention_implementation'):
        if profile.get(field) != row.get(field):
            raise ValueError(f'Base profile mismatched {field}')
    validate_tokenizer_snapshot(profile)
    for field in ('model_revision', 'tokenizer_revision', 'tokenizer_source_id'):
        if profile.get(field) != row.get(field):
            raise ValueError(f'Base profile mismatched {field}')
    expected_rows = row['reused_tokens'] if expected_mode == 'ES_BASE_SEMCACHE_REUSE' else 0
    if profile.get('executed_reused_tokens') != expected_rows:
        raise ValueError('Base profile must skip exactly the M8 physically reused positions')
    if profile.get('executed_fresh_tokens') != row['prompt_tokens'] - expected_rows:
        raise ValueError('Incorrect fresh-row accounting')
    if (profile.get('cache_storage_device') != row.get('physical_cache_storage_device')
            or profile.get('cache_transfer_path') != row.get('cache_transfer_path')):
        raise ValueError('Base profile cache placement differs from M8')
    if profile.get('cache_dtype') != row['dtype']:
        raise ValueError('Base cache dtype differs from M8 projection dtype')
    if profile.get('control_plane_inside_es_compute') is not False:
        raise ValueError('Control-plane work cannot be inside base ES timing')
    nonnegative(profile.get('prefill_wall_ms'), 'base prefill_wall_ms')
    if row.get('es_base_compute_ms') != profile['prefill_wall_ms']:
        raise ValueError('Base ES compute value differs from profile')
    return profile


def attach_profile(row, profile, fresh_id):
    result = copy.deepcopy(row)
    result.update(fresh_m85_profile_id=fresh_id, es_base_profile=profile,
        es_base_compute_ms=profile['prefill_wall_ms'], es_base_compute_provenance='MEASURED',
        es_base_compute_measurement_label=MEASUREMENT_LABEL,
        es_base_compute_scope='prefill_base_only_excluding_lora_control')
    if row['mode'] == 'SEMCACHE_PHYSICAL_REUSE':
        result['semcache_control_profile'] = control_plane(row)
    validate_base_profile(result, profile['mode'])
    return result
