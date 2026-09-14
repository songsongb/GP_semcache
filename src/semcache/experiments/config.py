"""Paper fragments have explicit inheritance; M1–M5 config loader is unchanged."""
from pathlib import Path
from copy import deepcopy
from semcache.utils.io import load_config
from semcache.cache.admission import AdmissionPolicy
from semcache.cache.eviction import EvictionPolicy


def merge(left, right):
    result = deepcopy(left)
    for key, value in right.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_paper_config(path, _seen=()):
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError('Cyclic config inheritance')
    config = load_config(path)
    parent = config.pop('extends', None)
    if parent:
        config = merge(load_paper_config(path.parent/parent, (*_seen,path)), config)
    validate_config(config)
    return config


def validate_config(c):
    for key, values in [('execution_mode', ('MEASURED_MODEL','ANALYTICAL_SIMULATION','HYBRID')),
                        ('execution_scope', ('prefill_only','full_generation')),
                        ('cache_storage_mode', ('logical_only','physical_cpu','physical_model')),
                        ('baseline', ('UD_ONLY','ES_ONLY','FBC','SEMCACHE'))]:
        if c.get(key) not in values:
            raise ValueError(f'Explicit valid {key} required')
    for key in ('num_users','lora_rank','subsequence_window','cluster_update_interval_queries'):
        if not isinstance(c[key], int) or isinstance(c[key],bool) or c[key] < 1:
            raise ValueError(f'Invalid {key}')
    if c.get('dataset') and (c['dataset'] not in ('multiwoz','coqa','snips') or not isinstance(c.get('cluster_count'),int) or c['cluster_count'] < 1):
        raise ValueError('Dataset-specific cluster_count required')
    if c['workload']['order'] not in ('source_order','seeded_shuffle'):
        raise ValueError('Invalid ordering')
    if c['clustering'] != dict(initialization='first_k', update_rule='batched_incremental_mean'):
        raise ValueError('Unsupported clustering policy')
    if c['match_rule'] != 'exact_token_ids_within_cluster':
        raise ValueError('Unsupported matching rule')
    AdmissionPolicy(**c['admission'])
    EvictionPolicy(**c['eviction'])
    from semcache.simulation.cost_model import positive
    positive(c['logical_cache_capacity_gb'], 'cache GB')
    positive(c['system']['bandwidth_mbps'], 'Mbps')
    if not 0 <= c['semantic_impact']['rho'] <= 1 or c['semantic_impact']['history_lambda'] < 1:
        raise ValueError('Invalid impact settings')
    if c['workload']['transformation'] not in ('raw_query','paper_reproduction_v1'):
        raise ValueError('Invalid transformation')
    if c['user_assignment']['mode'] not in ('seeded_round_robin','deterministic_hash'):
        raise ValueError('Invalid user assignment')
    limit = c['workload']['max_queries']
    if limit is not None and (not isinstance(limit,int) or isinstance(limit,bool) or limit < 1):
        raise ValueError('Invalid workload limit')
    for key in ('es_tflops','ud_tflops','communication_element_bytes','qkv_precision_bits'):
        if c['system'][key] is not None:
            positive(c['system'][key], key)
    return c


def load_model_spec(config, config_dir):
    spec = load_config(Path(config_dir)/f'model_{config["model_spec"]}.yaml')
    spec['lora_rank'] = config['lora_rank']
    return spec
