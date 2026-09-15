"""Bounded runner interfaces. Full generation and other baselines fail explicitly."""
from enum import Enum
import re
from semcache.utils.timing import timer
from pathlib import Path
from semcache.semcache_engine import SemCacheEngine
from semcache.semantic.encoder import ControlledEncoder, HuggingFaceTextEncoder
from semcache.semantic.intent_clusterer import IntentClusterer
from semcache.cache.global_cache import GlobalCache
from semcache.cache.admission import AdmissionPolicy
from semcache.cache.eviction import EvictionPolicy
from semcache.simulation.memory_model import qkv_block_bytes
from semcache.simulation.cost_model import projection_savings, PaperCostModel
from .user_assignment import stable_digest
from .provenance import metric
from .manifest import run_manifest, write_manifest, canonical, sha256
from .workload import serialize
from .aggregation import aggregate


class Baseline(str, Enum):
    UD_ONLY = 'UD_ONLY'  # full inference on user device
    ES_ONLY = 'ES_ONLY'  # base + user LoRA on edge server
    FBC = 'FBC'  # legacy identifier for the admit-first reproduction variant
    FBC_V1 = 'FBC_V1'
    FBC_V2 = 'FBC_V2'
    SEMCACHE = 'SEMCACHE'  # EdgeLoRA + semantic-aware global QKV cache


class ExecutionMode(str, Enum):
    MEASURED_MODEL = 'MEASURED_MODEL'
    ANALYTICAL_SIMULATION = 'ANALYTICAL_SIMULATION'
    HYBRID = 'HYBRID'


class WhitespaceTokenizer:
    name_or_path = 'whitespace_proxy_v1'
    def __call__(self, text):
        # Stable collision-free string IDs, valid for the logical exact-key path.
        return {'input_ids': text.split()}


def make_logical_engine(rows, config, encoder_kind='fixture', allow_download=False, tokenizer=None):
    c = config['cluster_count']
    texts = [r['query_text'] for r in rows]
    if not texts:
        raise ValueError('Empty workload')
    if encoder_kind == 'fixture':
        vectors = {t:[int(stable_digest('fixture', t)[:8],16)/0xffffffff] for t in texts}
        encoder = ControlledEncoder(vectors)
        centroids = [[i/max(1,c-1)] for i in range(c)]
    elif encoder_kind == 'real':
        enc = config['semantic_encoder']
        if not enc['model_id'] or not enc['revision']:
            raise ValueError('Real encoder requires explicit model_id and revision')
        encoder = HuggingFaceTextEncoder(enc['model_id'], enc['revision'], pooling=enc['pooling'],
            max_length=enc['max_length'], local_files_only=not allow_download)
        if len(texts) < c:
            raise ValueError('Need at least C warmup queries for real encoder; C is never silently reduced')
        centroids = encoder.encode(texts[:c])
    else:
        raise ValueError('Unknown encoder kind')
    clusterer = IntentClusterer(c,config['cluster_update_interval_queries'])
    clusterer.initialize(centroids, counts=[0]*c if encoder_kind == 'fixture' else None)
    cache = GlobalCache(int(config['logical_cache_capacity_gb']*1e9),
        AdmissionPolicy(**config['admission']), EvictionPolicy(**config['eviction']))
    return SemCacheEngine(None, tokenizer or WhitespaceTokenizer(), None, encoder, clusterer, cache,
        window_size=config['subsequence_window'], rho=config['semantic_impact']['rho'],
        history_lambda=config['semantic_impact']['history_lambda'],
        frequency_window=config['cache']['admission_frequency_window_queries'], seed=config['seed'])


def run_workload(rows, config, model_spec, *, engine, workload_manifest, run_id, output_root,
                 max_queries, scope='prefill_only'):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('run_id must be a safe filename identifier')
    if sha256(serialize(rows)) != workload_manifest['sha256']:
        raise ValueError('Input rows do not match prepared workload SHA256')
    if any((Path(output_root)/folder/f'{run_id}.json').exists() for folder in ('manifests','raw','aggregate')):
        raise FileExistsError('Run ID already exists; choose a new run_id')
    mode = ExecutionMode(config['execution_mode'])
    if Baseline(config['baseline']) != Baseline.SEMCACHE:
        raise NotImplementedError('Baseline interface only; FBC policy details require documented choices')
    if scope != 'prefill_only' or config['execution_scope'] != 'prefill_only':
        raise NotImplementedError('Full-generation SemCache loop is a Table II/BLEU blocker')
    if not isinstance(max_queries,int) or isinstance(max_queries,bool) or max_queries < 1:
        raise ValueError('Smoke requires an explicit positive max_queries')
    logical = config['cache_storage_mode'] == 'logical_only'
    if logical != (mode == ExecutionMode.ANALYTICAL_SIMULATION):
        raise ValueError('Logical-only events require ANALYTICAL_SIMULATION; measured/hybrid require physical model engine')
    if not logical and engine.model is None:
        raise ValueError('Physical run requires an actual loaded M5 model/adapter engine')
    if (engine.cache.capacity_bytes != int(config['logical_cache_capacity_gb']*1e9)
            or engine.clusterer.num_clusters != config['cluster_count']
            or engine.extractor.window_size != config['subsequence_window']
            or engine.clusterer.update_interval != config['cluster_update_interval_queries']):
        raise ValueError('Provisioned engine cache/cluster/window settings differ from run config')
    if not logical:
        module = engine.adapter.projection_module(0, 'q')
        if model_spec['hidden_size'] != module.in_features or model_spec['layers'] != len(engine.adapter.layers):
            raise ValueError('Model specification does not match measured engine dimensions')
        if model_spec.get('weight_precision_bits') is not None and model_spec['weight_precision_bits'] != module.get_base_layer().weight.element_size()*8:
            raise ValueError('Model weight precision does not match measured engine')
        if config['cache_storage_mode'] == 'physical_cpu' and engine.storage_device != 'cpu':
            raise ValueError('physical_cpu requires CPU cache storage')
    if rows and any(r['dataset'] != config['dataset'] for r in rows):
        raise ValueError('Workload dataset/config mismatch')
    chosen = rows[:max_queries]
    if not chosen:
        raise ValueError('Empty smoke workload')
    system = config['system']
    bits = system['qkv_precision_bits']
    if logical and bits is None:
        raise ValueError('Explicit QKV activation precision required; weight precision does not define cache precision')
    block_bytes = qkv_block_bytes(config['subsequence_window'],model_spec['layers'],model_spec['hidden_size'],model_spec['kv_dimension'],bits) if logical else None
    timing_ready = all(system[k] is not None for k in ('es_tflops','ud_tflops','communication_element_bytes'))
    results, events, metrics = [], [], []
    for row in chosen:
        if logical:
            out = engine.logical_query(row['query_text'],row['user_id'],row['global_query_index'],logical_block_bytes=block_bytes)
            summary = out['summary']
        else:
            if engine.adapter.projection_module(0, 'q').r[row['user_id']] != model_spec['lora_rank']:
                raise ValueError('Measured adapter rank differs from model specification')
            device = next(engine.model.parameters()).device
            with timer(device) as measured_time:
                out = engine.query(row['query_text'],row['user_id'],row['global_query_index'])
            s = out['summary']
            summary = dict(s, reused_token_count=s['reused_unique_token_count'],query_token_count=s['query_token_count'],
                logical_global_cache_bytes=s['logical_cache_occupancy'],physical_cache_tensor_bytes=s['physical_cache_bytes'],
                admission_count=sum(e['event_type']=='INSERT' for e in out['events']),
                admission_candidate_count=sum(e['event_type'] in ('ADMIT','DENY') for e in out['events']),
                eviction_count=sum(e['event_type']=='EVICT' for e in out['events']))
        if not logical:
            import torch
            parameters = list(engine.model.named_parameters())
            summary.update(model_parameter_bytes=sum(p.numel()*p.element_size() for name,p in parameters if 'lora_' not in name),
                adapter_bytes=sum(p.numel()*p.element_size() for name,p in parameters if 'lora_' in name),
                gpu_allocated_bytes=torch.cuda.memory_allocated(device) if device.type == 'cuda' else None,
                gpu_reserved_bytes=torch.cuda.memory_reserved(device) if device.type == 'cuda' else None)
        source = 'SIMULATED' if logical else 'MEASURED'
        if not logical:
            summary.pop('metric_source', None)
            for legacy_key in list(summary):
                if legacy_key.startswith('paper_estimated_'):
                    summary.pop(legacy_key)
        summary['metric_provenance_contract'] = 'See per-metric metrics object; summary is event/debug metadata'
        typed = {k:metric(summary[k],source,'logical cache simulation' if logical else 'model prefill cache events','count') for k in
            ('block_lookup_count','block_hit_count','reused_token_count','query_token_count','admission_count','admission_candidate_count','eviction_count')}
        for key in ('logical_global_cache_bytes','physical_cache_tensor_bytes','gpu_allocated_bytes','gpu_reserved_bytes','model_parameter_bytes','adapter_bytes','analytical_system_memory_bytes'):
            metric_source = 'SIMULATED' if key in ('logical_global_cache_bytes','analytical_system_memory_bytes') else 'MEASURED'
            typed[key] = metric(summary.get(key),metric_source,key,'bytes')
        savings = projection_savings(summary['reused_token_count'],model_spec['hidden_size'],model_spec['lora_rank'],model_spec['layers'])
        for key, value in savings.items():
            typed[key] = metric(value,'SIMULATED','all-layer analytical projection savings','elements' if 'elements' in key else 'FLOP')
        typed['measured_prefill_latency_s'] = metric(measured_time['seconds'] if not logical else None,'MEASURED','instrumented local M5 prefill including cache/encoder/copies; not distributed system','seconds')
        latency = None
        if timing_ready and mode != ExecutionMode.MEASURED_MODEL:
            estimate = PaperCostModel().estimate(tokens=summary['query_token_count'],reused_tokens=summary['reused_token_count'],model_config=model_spec,system_config=system)
            latency = estimate['latency_s']
        typed['analytical_latency_s'] = metric(latency,'SIMULATED','Eq.19 all-layer prefill, excludes overheads','seconds')
        typed['comm_bytes_saved'] = metric(savings['comm_elements_saved']*system['communication_element_bytes'] if system['communication_element_bytes'] else None,'SIMULATED','all-layer projection exchange','bytes')
        results.append(dict(source_id=row['source_id'],summary=summary,metrics=typed))
        events.extend(dict(e, metric_source=source, execution_mode=mode.value) for e in out['events'])
        metrics.append(typed)
    root = Path(output_root)
    manifest = run_manifest(run_id,config,workload_manifest,model_spec=model_spec,
        model_metadata=engine.metadata, semantic_encoder=engine.encoder.metadata,
        actual_engine=dict(logical_cache_capacity_bytes=engine.cache.capacity_bytes,
            cluster_count=engine.clusterer.num_clusters,window_size=engine.extractor.window_size,
            physical_storage_device=engine.storage_device if not logical else None,
            attention_impact_available=not logical),
        tokenizer=dict(model_id=getattr(engine.tokenizer,'name_or_path',None),
            requested_revision=getattr(engine.tokenizer,'init_kwargs',{}).get('revision'),
            resolved_revision=getattr(engine.tokenizer,'init_kwargs',{}).get('_commit_hash')),
        execution_mode=mode.value,execution_scope=scope,cache_storage_mode=config['cache_storage_mode'],
        query_count=len(chosen),executed_workload_sha256=sha256(serialize(chosen)),
        seed=config['seed'], workload_sha256=workload_manifest['sha256'],
        cluster_count=config['cluster_count'], w=config['subsequence_window'],
        logical_cache_capacity_bytes=int(config['logical_cache_capacity_gb']*1e9),
        configured_user_count=config['num_users'],active_user_count=len({r['user_id'] for r in chosen}),
        timing_available=timing_ready and mode != ExecutionMode.MEASURED_MODEL,
        generation=config['generation'],reported_hit_rate_definition=None,
        reproduction_choices=dict(fixture_encoder='SHA256 scalar features and evenly spaced C centroids; no learned semantics',
            logical_impact='unavailable; null mapped to zero only for admission/eviction scoring; no CHU/PBR',
            logical_tokenization='whitespace proxy unless explicit tokenizer supplied',
            latency_scope='paper per-layer Eq.19 multiplied by layers; no full generation'))
    write_manifest(root/'manifests'/f'{run_id}.json',manifest)
    write_manifest(root/'raw'/f'{run_id}.json',results)
    (root/'raw'/f'{run_id}.events.jsonl').write_text(''.join(canonical(e)+'\n' for e in events),encoding='utf-8')
    output = aggregate(metrics)
    write_manifest(root/'aggregate'/f'{run_id}.json',output)
    return output, manifest
