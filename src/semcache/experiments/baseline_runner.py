"""Common CPU logical/analytical baseline runner. No paper reference inputs."""
from copy import deepcopy
from .baselines import BaselineKind, FrequencyLRUCache, FBC_METADATA, FBC_V2_METADATA, FBC_KINDS, ALL_BASELINES
from .runner import make_logical_engine, WhitespaceTokenizer
from .config import validate_config
from .manifest import canonical, sha256
from .workload import serialize
from .provenance import metric, configuration_provenance
from semcache.simulation.memory_model import qkv_block_bytes
from semcache.simulation.cost_model import projection_savings, PaperCostModel


def run_baseline(rows, config, model_spec, *, workload_manifest, baseline, max_queries,
                 seed=None, run_id='m6b1', encoder_kind='fixture', tokenizer=None):
    kind = BaselineKind(baseline)
    c = deepcopy(config)
    c['baseline'] = kind.value
    if seed is not None:
        c['seed'] = seed
    validate_config(c)
    if (c['execution_mode'], c['cache_storage_mode'], c['execution_scope']) != ('ANALYTICAL_SIMULATION', 'logical_only', 'prefill_only'):
        raise ValueError('M6B-1 requires logical analytical prefill')
    if isinstance(max_queries, bool) or not isinstance(max_queries, int) or max_queries < 1:
        raise ValueError('max_queries must be positive')
    if sha256(serialize(rows)) != workload_manifest['sha256']:
        raise ValueError('Workload hash mismatch')
    if workload_manifest['user_count'] != c['num_users'] or any(r['dataset'] != c['dataset'] or r['global_query_index'] != i for i,r in enumerate(rows)):
        raise ValueError('Workload/config identity, ordering or user count mismatch')
    if workload_manifest.get('user_assignment_rule') != c['user_assignment']['mode']:
        raise ValueError('Workload/config user assignment rule mismatch; correct config without changing the fixed workload')
    if model_spec['lora_rank'] != c['lora_rank']:
        raise ValueError('Model/config rank mismatch')
    chosen = rows[:max_queries]
    if not chosen:
        raise ValueError('Empty workload')
    tok = tokenizer or WhitespaceTokenizer()
    capacity = int(c['logical_cache_capacity_gb']*1e9)
    system = c['system']
    cached = kind in (*FBC_KINDS, BaselineKind.SEMCACHE)
    block_bytes = qkv_block_bytes(c['subsequence_window'], model_spec['layers'], model_spec['hidden_size'], model_spec['kv_dimension'], system['qkv_precision_bits']) if cached else None
    engine = make_logical_engine(chosen,c,encoder_kind,tokenizer=tok) if kind == BaselineKind.SEMCACHE else None
    fbc = FrequencyLRUCache(capacity,c['subsequence_window'],block_bytes,tok,
        frequency_threshold=2 if kind == BaselineKind.FBC_V2 else 1) if kind in FBC_KINDS else None
    summaries = []
    for row in chosen:
        if engine:
            summary = engine.logical_query(row['query_text'],row['user_id'],row['global_query_index'],logical_block_bytes=block_bytes)['summary']
        elif fbc:
            summary = fbc.query(row['query_text'])
        else:
            n = len(tok(row['query_text'])['input_ids'])
            if not n:
                raise ValueError('Empty tokenized query')
            summary = dict(query_token_count=n, reused_token_count=0, logical_global_cache_bytes=0,
                **{k:None for k in ('block_lookup_count','block_hit_count','admission_candidate_count','admission_count','eviction_count')})
        summaries.append(summary)
    values = {k:sum(s[k] for s in summaries) if summaries[0][k] is not None else None for k in
        ('query_token_count','reused_token_count','block_lookup_count','block_hit_count','admission_candidate_count','admission_count','eviction_count')}
    for key,num,den in [('block_hit_ratio','block_hit_count','block_lookup_count'),('token_reuse_ratio','reused_token_count','query_token_count'),('admission_rate','admission_count','admission_candidate_count')]:
        values[key] = values[num]/values[den] if values[den] else None
    d,r,layers = (model_spec[k] for k in ('hidden_size','lora_rank','layers'))
    totals = projection_savings(values['query_token_count'],d,r,layers)
    savings = projection_savings(values['reused_token_count'],d,r,layers)
    # Same Eq.19 base accounting as M6A: remaining attention/FFN never saved.
    values.update(base_flops_total=totals['base_flops_saved']+sum((18*s['query_token_count']*d*d+4*s['query_token_count']**2*d+16*s['query_token_count']*d)*layers for s in summaries),
        lora_flops_total=totals['lora_flops_saved'], base_flops_saved=savings['base_flops_saved'],
        lora_flops_saved=savings['lora_flops_saved'], communication_elements_total=totals['comm_elements_saved'] if cached else 0,
        communication_elements_saved=savings['comm_elements_saved'] if cached else 0,
        logical_cache_bytes=summaries[-1]['logical_global_cache_bytes'],
        peak_logical_cache_bytes=fbc.peak_bytes if fbc else (None if cached else 0))
    missing = [k for k in ('es_tflops','ud_tflops','communication_element_bytes') if system[k] is None] if cached else ['single-device full-inference latency model unavailable']
    latency = None if missing else sum(PaperCostModel().estimate(tokens=s['query_token_count'],reused_tokens=s['reused_token_count'],model_config=model_spec,system_config=system)['latency_s'] for s in summaries)
    values.update(analytical_latency_s=latency, system_memory_bytes=None, base_model_bytes=None,
        adapter_bytes=None, cache_bytes=values['logical_cache_bytes'], activation_bytes=None,total_system_memory_bytes=None)
    latency_scope = 'workload total all-layer Eq.19 prefill; excludes encoder/cache/embedding/logits/decode/queueing overheads'
    metrics = {}
    for k,v in values.items():
        unit = 'bytes' if k.endswith('_bytes') else 'FLOP' if 'flops' in k else 'elements' if 'elements' in k else 'seconds' if k.endswith('_s') else 'ratio' if k.endswith(('_ratio','_rate')) else 'count'
        scope = latency_scope if k == 'analytical_latency_s' else 'all-layer Eq.19 accounting; QKV-only savings' if 'flops' in k else 'EdgeLoRA projection exchange' if 'communication' in k else 'logical fixed-workload cache simulation'
        metrics[k] = dict(metric(v,'SIMULATED',scope,unit), comparability='NOT_COMPARABLE' if unit in ('seconds','bytes','FLOP','elements') else 'APPROXIMATE')
    fairness = dict(workload_sha256=workload_manifest['sha256'], executed_workload_sha256=sha256(serialize(chosen)),
        query_ids=[r['source_id'] for r in chosen], user_ids=[r['user_id'] for r in chosen],
        model_spec=deepcopy(model_spec), seed=c['seed'], cache_capacity_bytes=capacity,
        window_size=c['subsequence_window'],user_count=c['num_users'],bandwidth_mbps=system['bandwidth_mbps'],
        system_spec=deepcopy(system),tokenizer=getattr(tok,'name_or_path',type(tok).__name__))
    return dict(schema_version='semcache.baseline.v1',run_id=run_id,baseline=kind.value,dataset=c['dataset'],
        model_name=model_spec['name'],query_count=len(chosen),user_count=c['num_users'],execution_mode=c['execution_mode'],
        **values,analytical_latency_scope=latency_scope,unavailable_components=missing,
        memory_scope='logical QKV cache only; ES + all UD model/adapters/activations unavailable',
        compute_placement='UD: full base + user LoRA' if kind == BaselineKind.UD_ONLY else 'ES: full base + user LoRA' if kind == BaselineKind.ES_ONLY else 'ES base + UD LoRA; collaborative EdgeLoRA',
        metric_source='SIMULATED',comparability='APPROXIMATE',metrics=metrics,fairness=fairness,
        config=c,config_sha256=sha256(canonical(c).encode()),configuration_provenance=configuration_provenance(c),
        workload_provenance=deepcopy(workload_manifest),baseline_semantics_provenance='PAPER_DEFINED',
        fbc_metadata=deepcopy(FBC_V2_METADATA if kind == BaselineKind.FBC_V2 else FBC_METADATA) if fbc else None,
        reproduction_choices=dict(provenance='REPRODUCTION_CHOICE',tokenization=fairness['tokenizer'],
            overlap_selection='existing earliest-start nonoverlapping selection; lookups precede admissions',
            semantic_encoder=encoder_kind if engine else None,
            logical_semantics='existing M6A fixture hash encoder by default; no attention impact or CHU/PBR; no physical reuse claim',
            cost_scope='Eq.19 analytical FLOPs, excludes embeddings/logits; not full model execution'),
        query_results=[dict(source_id=row['source_id'],user_id=row['user_id'],summary=s) for row,s in zip(chosen,summaries)])


def run_comparison(rows, config, model_spec, *, baselines=ALL_BASELINES, **kwargs):
    results = [run_baseline(rows,config,model_spec,baseline=b,**kwargs) for b in baselines]
    if results and any(r['fairness'] != results[0]['fairness'] for r in results):
        raise AssertionError('Baseline fairness mismatch')
    return results


SUMMARY_FIELDS = (
    'baseline', 'query_count', 'block_lookup_count', 'block_hit_count', 'block_hit_ratio',
    'reused_token_count', 'token_reuse_ratio', 'admission_candidate_count',
    'admission_count', 'admission_rate', 'eviction_count', 'logical_cache_bytes',
    'base_flops_saved', 'lora_flops_saved', 'communication_elements_saved',
)


def comparison_summary(results):
    """Compact simulated result table; full per-metric provenance stays in results."""
    return dict(metric_source='SIMULATED',
        metric_metadata={r['baseline']: {k:r['metrics'][k] for k in SUMMARY_FIELDS if k in r['metrics']} for r in results},
        rows=[{k:r[k] for k in SUMMARY_FIELDS} for r in results])
