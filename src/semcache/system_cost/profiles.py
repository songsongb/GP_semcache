"""Read-only M8 artifact boundary, correctness gate and explicit ES interpretation."""
import json
from pathlib import Path
from .common import MODELS, integer, nonnegative, tagged_ms

REUSE_MODES = ('SEMCACHE_PHYSICAL_REUSE', 'SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC')
MODES = ('NATIVE_NO_CACHE', 'SEMCACHE_LOOKUP_NO_REUSE', *REUSE_MODES)


def load_rows(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and 'rows' in data:
            rows = data['rows']
        else:
            rows = [data]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Expected M8 raw row(s), not aggregate CSV/summary statistics')
    return rows


def validate_row(row):
    if row.get('model_id') not in MODELS or row.get('mode') not in MODES:
        raise ValueError('Unsupported model/mode for M9-A')
    if row.get('measured_or_analytical') != 'MEASURED':
        raise ValueError('ES input must be MEASURED; reference/simulated timing is forbidden')
    if row.get('execution_scope') not in (None, 'prefill_only'):
        raise ValueError('M9-A accepts prefill-only inputs')
    n = integer(row.get('prompt_tokens'), 'prompt_tokens', 1)
    reused = integer(row.get('reused_tokens'), 'reused_tokens')
    if reused > n:
        raise ValueError('Reused tokens exceed input tokens')
    if row['mode'] not in REUSE_MODES and reused:
        raise ValueError('Native/lookup-only input cannot claim physical reuse')
    if row.get('recomputed_tokens') is not None and row['recomputed_tokens'] != n - reused:
        raise ValueError('Inconsistent recomputed token count')
    if row.get('token_reuse_ratio') is not None and abs(row['token_reuse_ratio'] - reused/n) > 1e-9:
        raise ValueError('Inconsistent reuse ratio')
    if reused and (row.get('physical_reuse_used') is not True or row.get('projection_skip_used') is not True):
        raise ValueError('Reused tokens need physical reuse and projection skipping evidence')
    for key in ('prefill_wall_ms', 'request_wall_ms', 'tokenization_ms'):
        nonnegative(row.get(key), key)
    residual = row['request_wall_ms'] - row['prefill_wall_ms'] - row['tokenization_ms']
    if residual < -1e-6:
        raise ValueError('Request time is smaller than prefill plus tokenization')
    for key in ('mixed_qkv_execution_ms', 'qkv_execution_ms', 'saved_communication_bytes'):
        if row.get(key) is not None:
            nonnegative(row[key], key)
    return row


def reuse_gate(row, allow_invalid_reuse=False):
    status = str(row.get('correctness_validation_status') or '').lower()
    parity = row.get('controlled_exact_parity_passed')
    reason = None
    if parity is False:
        reason = 'controlled_exact_parity_passed=false'
    elif 'not_validated' in status or 'failed' in status or 'invalid' in status:
        reason = f'correctness status: {status}'
    elif parity is not True and status not in ('controlled_exact_parity_passed', 'position_aligned_parity_passed'):
        reason = 'missing affirmative correctness validation'
    if reason and not allow_invalid_reuse:
        raise ValueError('Reuse input rejected: ' + reason + '; explicit --allow-invalid-reuse required')
    return dict(correctness_gate_passed=reason is None,
        invalid_reuse_override=reason is not None, correctness_gate_reason=reason,
        comparison_scope='correctness_gated_fixture_only' if reason is None else 'UNSAFE_REUSE_EXPLORATION',
        safe_reuse_claimed=False)


def pair_key(row):
    fields = ('experiment_id', 'model_id', 'query_id', 'user_id', 'adapter_name',
              'prompt_token_ids_sha256', 'prompt_tokens', 'repeat_index', 'dtype',
              'model_revision', 'tokenizer_revision', 'seed', 'attention_implementation',
              'hostname', 'gpu_name', 'requested_prompt_tokens', 'actual_prompt_tokens')
    for key in ('experiment_id', 'query_id', 'user_id', 'adapter_name', 'prompt_token_ids_sha256', 'dtype'):
        if not row.get(key):
            raise ValueError(f'M8 row lacks pairing identity: {key}')
    integer(row.get('repeat_index'), 'repeat_index')
    return tuple(row.get(key) for key in fields)


def validate_policy_match(lookup, physical):
    for key in ('impact_reducer_type', 'impact_reducer_metadata', 'cluster_update_interval_queries',
                'cluster_schedule_mode', 'rho', 'history_lambda', 'pbr_interval_queries',
                'physical_cache_storage_device', 'cache_addressing_mode'):
        if lookup.get(key) != physical.get(key):
            raise ValueError(f'Mismatched M8 policy profile: {key}')


def pair_rows(rows, model_id, query_id='same_user_exact', reuse_mode='SEMCACHE_PHYSICAL_REUSE'):
    if reuse_mode not in REUSE_MODES:
        raise ValueError('Unsupported physical reuse mode')
    groups = {}
    for row in rows:
        if row.get('model_id') != model_id or row.get('query_id') != query_id:
            continue
        validate_row(row)
        if row['mode'] not in ('NATIVE_NO_CACHE', 'SEMCACHE_LOOKUP_NO_REUSE', reuse_mode):
            continue
        group = groups.setdefault(pair_key(row), {})
        if row['mode'] in group:
            raise ValueError('Ambiguous duplicate M8 row; select a single experiment file')
        group[row['mode']] = row
    if not groups:
        raise ValueError('No matching M8 rows')
    pairs = []
    for group in groups.values():
        if not all(mode in group for mode in ('NATIVE_NO_CACHE', 'SEMCACHE_LOOKUP_NO_REUSE', reuse_mode)):
            raise ValueError('Need paired native, lookup-no-reuse and physical rows with identical identity')
        lookup, physical = group['SEMCACHE_LOOKUP_NO_REUSE'], group[reuse_mode]
        validate_policy_match(lookup, physical)
        pairs.append((group['NATIVE_NO_CACHE'], lookup, physical))
    return pairs


def es_components(native, physical, policy='require-base-only'):
    """No FLOP-ratio subtraction or invented GPU LoRA timing.

    Base-only profile extension fields must be separately supplied measurements
    with explicit scope/provenance. Existing M8 cannot populate those fields.
    """
    if policy == 'require-base-only':
        terms = []
        for row in (native, physical):
            if (row.get('es_base_compute_provenance') != 'MEASURED'
                    or row.get('es_base_compute_scope') != 'prefill_base_only_excluding_lora_control'):
                raise ValueError('M8 PEFT timing is not base-only: provide explicit base-only measured fields '
                                 'or opt into --es-compute-policy peft-prefill-proxy')
            terms.append(tagged_ms(row.get('es_base_compute_ms'), 'MEASURED', 'explicit_base_only_profile'))
        return terms
    if policy != 'peft-prefill-proxy':
        raise ValueError('Unknown ES compute policy')
    return [tagged_ms(row['prefill_wall_ms'], 'SIMULATED',
            'REPRODUCTION_CHOICE: measured PEFT prefill proxy; includes GPU LoRA and device-local work',
            dependencies=('MEASURED',)) for row in (native, physical)]
