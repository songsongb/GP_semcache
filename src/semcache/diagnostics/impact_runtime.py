"""Explicitly invoked local-model capture for the M9-A.2 audit only."""
import copy
import platform
from statistics import mean, median

from semcache.cache.attention_impact import PaperRowL2SumReducer
from semcache.cache.metric_manager import QueryImpactHistory
from semcache.metrics.m8 import token_ids_sha256, exact_parity_passed, percentile
from semcache.models.tokenizer_provenance import compare_snapshot_provenance
from .impact import MODES, PARTS, measure_impact, parity
from .impact_policy import replay_policy, compare_policy


class CaptureReducer(PaperRowL2SumReducer):
    """Records references/values; always delegates to the production reducer."""
    def begin(self):
        self.attentions, self.values = None, []

    def reduce(self, attentions, start, end, valid_attention_mask=None):
        self.attentions = attentions
        value = super().reduce(attentions, start, end, valid_attention_mask)
        self.values.append(value)
        return value


def capture_fixture(source, env, device):
    import torch
    from semcache.models.loader import load_model
    from semcache.models.lora_fixtures import create_controlled_users
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.semantic.encoder import TinyBERTSemanticEncoder
    from semcache.semantic.intent_clusterer import IntentClusterer
    from semcache.cache.global_cache import GlobalCache
    from semcache.semcache_engine import SemCacheEngine
    from semcache.metrics.scaling_workload import controlled_length_trace
    from semcache.metrics.inference import native_request
    from semcache.utils.seed import seed_everything
    seed_everything(42)
    if (source.get('hostname') != platform.node()
            or source.get('gpu_name') != (torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None)):
        raise ValueError('Audit must run on the same ES hardware as strict input')
    dtype = source['dtype'].removeprefix('torch.')
    model, tokenizer, metadata = load_model(dict(name=source['model_id'],
        tokenizer=source['tokenizer_source_id'], revision=source['model_revision'],
        tokenizer_revision=source['tokenizer_revision'], dtype=dtype, device=device,
        attention_implementation='eager', local_files_only=True))
    snapshot_comparison = compare_snapshot_provenance(source, metadata)
    model, lora = create_controlled_users(model)
    if lora != env['lora']:
        raise ValueError('Controlled adapter fixture differs from original M8.5 environment')
    encoder_meta = env['semantic_encoder']
    encoder = TinyBERTSemanticEncoder(model_id=encoder_meta['checkpoint'],
        revision=encoder_meta.get('resolved_revision') or encoder_meta['revision'],
        pooling=encoder_meta['pooling'], max_length=encoder_meta['max_length'],
        device=device, dtype=dtype, local_files_only=True)
    trace = controlled_length_trace(tokenizer, 32)
    ids = trace[1]['token_ids']
    if len(ids) != 32 or token_ids_sha256(ids) != source['prompt_token_ids_sha256']:
        raise ValueError('Controlled length-32 token IDs differ from strict source')
    clusterer = IntentClusterer(2, initialization='first_k', update_mode='buffered', update_interval=100)
    clusterer.initialize(encoder.encode([trace[0]['text'], trace[-1]['text']]))
    reducer = CaptureReducer()
    cache = GlobalCache(256*1024*1024)
    engine = SemCacheEngine(model, tokenizer, OPTModelAdapter(model), encoder, clusterer, cache,
        metadata=metadata, storage_device=source['physical_cache_storage_device'],
        impact_reducer=reducer, rho=.8, history_lambda=100, pbr_interval_queries=100)
    captures, queries, results = [], [], []
    for item in trace[:2]:
        reference = native_request(model, tokenizer, item['text'], item['user_id'], collect_quality=False)
        reducer.begin()
        history_before = copy.deepcopy(engine.metrics.history)
        result = engine.query(item['text'], item['user_id'], item['condition'],
            execution_mode='SEMCACHE_PHYSICAL_REUSE', collect_timing=True, baseline_logits=reference['logits'])
        summary = result['summary']
        windows = engine.extractor.extract(ids)
        # Sizes are actual materialized QKV entry sizes from the admission events.
        # Same input span width/model/dtype => one logical block size in this fixture.
        sizes = {e['S'] for e in result['events'] if e['event_type'] == 'ADMISSION_SCORE'}
        if sizes:
            if len(sizes) != 1:
                raise ValueError('Expected fixed w=3 QKV block size')
            block_size = sizes.pop()
        elif not queries:
            raise ValueError('Cold fixture did not record admission sizes')
        captures.append(dict(attentions=reducer.attentions, windows=windows,
            values=list(reducer.values), history_before=history_before, cluster=summary['cluster_id']))
        queries.append(dict(windows=windows, values=list(reducer.values), cluster=summary['cluster_id'],
            query_id=item['condition'], token_count=len(ids), sizes=[block_size]*len(windows)))
        results.append(result)
    target = results[-1]['summary']
    quality_row = dict(max_abs_logit_diff=target['max_abs_logit_diff'],
        relative_l2_logit_diff=target['relative_l2_logit_diff'],
        last_position_kl=target['last_position_kl_baseline_to_injected'],
        argmax_agreement=target['last_argmax_agreement'])
    if exact_parity_passed(quality_row) is not True:
        raise ValueError('Recreated same-user exact fixture failed existing correctness tolerances')
    if target['reused_unique_token_count'] != source['reused_tokens']:
        raise ValueError('Recreated fixture reuses a different token count')
    fields = ('destination_start', 'destination_end', 'source_start', 'source_end', 'token_ids')
    def spans(items):
        return [{key: item[key] for key in fields} for item in items]
    if spans(target['reuse_block_provenance']) != spans(source['reuse_block_provenance']):
        raise ValueError('Recreated physical reuse spans differ from strict source')
    replay = replay_policy(queries, capacity_bytes=cache.capacity_bytes)
    actual_admissions = [dict(query_id=r['summary']['query_id'], key=e['cache_key'],
        start=e['target_start'], end=e['target_end'], decision=e['admission_decision'], score=e['admission_score'])
        for r in results for e in r['events'] if e['event_type'] == 'ADMISSION_SCORE']
    actual_evictions = [dict(query_id=r['summary']['query_id'], key=e['cache_key'], score=e['eviction_score'])
        for r in results for e in r['events'] if e['event_type'] == 'EVICT']
    actual_selections = [[(item['destination_start'], item['destination_end'], item['cache_key'])
        for item in r['summary']['reuse_block_provenance']] for r in results]
    if (replay['admissions'] != actual_admissions or replay['evictions'] != actual_evictions
            or replay['selections'] != actual_selections):
        raise ValueError('Policy replay differs from actual production engine events')
    capture_record = dict(model=metadata, snapshot_comparison=snapshot_comparison,
        lora=lora, encoder=encoder.metadata, quality=quality_row, recreated_exact_parity_passed=True,
        capacity_bytes=cache.capacity_bytes, prompt_token_ids=ids,
        prompt_token_ids_sha256=token_ids_sha256(ids), reuse_block_provenance=target['reuse_block_provenance'],
        production_policy_replay_validated=True, production_policy=replay,
        captured_production_impact_values=[c['values'] for c in captures],
        captured_production_attention_impact_ms=target['timing']['attention_impact_ms'],
        capture_timing_scope='production path with reference-capture wrapper; not used as benchmark total',
        hardware=dict(hostname=platform.node(), gpu=source.get('gpu_name'), torch_version=torch.__version__),
        fixture_scope='existing controlled untrained PEFT cold -> same_user_exact; actual physical reuse attention')
    # Retain model/cache/encoder to preserve resident-memory conditions during audit.
    return captures, queries, capture_record, (model, tokenizer, engine, encoder)


def benchmark(captures, queries, capacity_bytes, warmups, measured):
    import torch
    rows = []
    reference_policy = replay_policy(queries, capacity_bytes=capacity_bytes)
    def run(capture, mode, instrument, production=False):
        if capture['attentions'][0].is_cuda:
            torch.cuda.synchronize(capture['attentions'][0].device)  # Outside impact_total_ms.
        return measure_impact(capture['attentions'], capture['windows'], mode,
            cluster=capture['cluster'], history=copy.deepcopy(capture['history_before']),
            instrument=instrument, production_reference=production)
    for phase, count in (('warmup', warmups), ('measured', measured)):
        for repeat in range(count):
            # Rotate mode order to expose/reduce systematic thermal/order bias.
            order = MODES[repeat % len(MODES):] + MODES[:repeat % len(MODES)]
            for order_index, mode in enumerate(order):
                diagnostic = run(captures[1], mode, True)
                # Production code is the uninstrumented CURRENT baseline. Optimized
                # code uses the same implementation without events/stage clocks.
                plain = run(captures[1], mode, False, production=mode == MODES[0])
                cold = run(captures[0], mode, False, production=mode == MODES[0])
                checks = [parity(captures[1]['values'], diagnostic['impact_values']),
                          parity(captures[1]['values'], plain['impact_values']),
                          parity(captures[0]['values'], cold['impact_values'])]
                variant = [dict(queries[0], values=cold['impact_values']),
                           dict(queries[1], values=plain['impact_values'])]
                policy = compare_policy(reference_policy, replay_policy(variant, capacity_bytes=capacity_bytes))
                eligible = all(c['impact_values_within_tolerance'] for c in checks) and all(policy[k] for k in (
                    'admission_decisions_identical', 'admission_scores_within_tolerance',
                    'eviction_decisions_identical', 'eviction_scores_within_tolerance',
                    'physical_hit_selection_identical', 'resident_keys_identical'))
                row = dict(diagnostic, phase=phase, repeat_index=repeat, order_index=order_index,
                    uninstrumented_impact_total_ms=plain['impact_total_ms'],
                    instrumentation_delta_ms=diagnostic['impact_total_ms']-plain['impact_total_ms'],
                    parity_checks=checks, **checks[0], policy_parity=policy,
                    diagnostic_composition_eligible=eligible,
                    benchmark_scope='fixed actual physical-reuse attention microbenchmark; no forward in timed region',
                    warmup_count=warmups, measured_count=measured,
                    synchronization_before_measurement='one cuda.synchronize outside total per timed call')
                rows.append(row)
    return rows


def summarize(rows):
    measured = [r for r in rows if r['phase'] == 'measured']
    baseline = [r for r in measured if r['implementation'] == MODES[0]]
    summaries = []
    for mode in MODES:
        group = [r for r in measured if r['implementation'] == mode]
        summary = dict(implementation=mode, count=len(group), provenance=group[0]['provenance'],
            formula_provenance='PAPER_DEFINED', interpretation_provenance='REPRODUCTION_CHOICE')
        fields = sorted({k for r in group for k, v in r.items() if k.endswith('_ms') and isinstance(v, (int, float))})
        for field in fields:
            values = [r[field] for r in group if field in r]
            summary[field] = dict(mean=mean(values), p50=median(values), p95=percentile(values, .95))
        for field in ('impact_total_ms', 'uninstrumented_impact_total_ms'):
            summary[field+'_speedup_vs_current'] = mean(r[field] for r in baseline)/mean(r[field] for r in group)
        summary.update(impact_value_max_absolute_error=max(r['impact_value_max_absolute_error'] for r in group),
            impact_value_relative_l2_error=max(r['impact_value_relative_l2_error'] or 0. for r in group),
            diagnostic_composition_eligible=all(r['diagnostic_composition_eligible'] for r in group),
            admission_decisions_identical=all(r['policy_parity']['admission_decisions_identical'] for r in group),
            eviction_decisions_identical=all(r['policy_parity']['eviction_decisions_identical'] for r in group),
            eviction_scores_within_tolerance=all(r['policy_parity']['eviction_scores_within_tolerance'] for r in group),
            eviction_occurred=any(r['policy_parity']['eviction_occurred'] for r in group))
        summaries.append(summary)
    return summaries
