"""Controlled prefill orchestration. Architecture details belong in ModelAdapter."""
import math
from collections import defaultdict
from .semantic.subsequence import SubsequenceExtractor
from .semantic.matcher import ExactTokenMatcher
from .semantic.hit_selection import CacheHit, select_nonoverlapping
from .cache.cache_entry import CacheEntry
from .cache.metric_manager import CacheMetricManager, actual_attention_impact
from .cache.attention_impact import MeanLayerHeadFrobeniusReducer
from .cache.cache_metrics import normalize
from .edgelora.mixed_projection import mixed_projection_path
from .evaluation.logit_metrics import compare_logits
from .simulation.cost_model import projection_savings
from .metrics.timing import CPUWallTimer, TimingRegistry, close_request_timing
from .edgelora.mixed_projection import mixed_qkv_elapsed_ms


class SemCacheEngine:
    def __init__(self, model, tokenizer, adapter, encoder, clusterer, cache, *, window_size=3,
                 storage_device='cpu', rho=0.8, history_lambda=100, frequency_window=100,
                 metadata=None, seed=42, impact_reducer=None, pbr_interval_queries=None,
                 verbose_impact_events=False):
        self.model, self.tokenizer, self.adapter = model, tokenizer, adapter
        self.encoder, self.clusterer, self.cache = encoder, clusterer, cache
        self.extractor, self.matcher = SubsequenceExtractor(window_size), ExactTokenMatcher()
        self.storage_device, self.metadata, self.seed = storage_device, metadata or {}, seed
        self.metrics = CacheMetricManager(cache, rho, history_lambda, frequency_window)
        self.impact_reducer = impact_reducer or MeanLayerHeadFrobeniusReducer()
        if pbr_interval_queries is not None and pbr_interval_queries < 1:
            raise ValueError('PBR interval must be positive or None')
        self.pbr_interval_queries = pbr_interval_queries
        self.verbose_impact_events = verbose_impact_events
        self.events = []
        self.query_id = None

    def emit(self, event_type, **details):
        event = dict(query_id=self.query_id, step=len(self.events)+1, event_type=event_type,
                     safe_reuse_claimed=False, **details)
        self.events.append(event)

    def entry_fields(self, entry):
        return dict(cache_key=entry.key, cluster_id=entry.cluster_id,
            source_start=entry.positions[0], source_end=entry.positions[1],
            logical_block_bytes=entry.size_bytes, physical_block_bytes=entry.physical_tensor_bytes,
            logical_cache_occupancy=self.cache.logical_cache_bytes,
            F=entry.frequency, A=entry.age(self.cache.now), I=entry.impact, S=entry.size_bytes,
            component_scope=entry.qkv_metadata.get('component_scope'))

    def query(self, query_text, user_id, query_id, compare_baseline=False,
              execution_mode='SEMCACHE_PHYSICAL_REUSE', collect_timing=False,
              baseline_logits=None):
        import torch
        if collect_timing and compare_baseline and baseline_logits is None:
            raise ValueError('Timed queries require baseline_logits computed outside request timing')
        external_baseline = baseline_logits.detach().cpu() if baseline_logits is not None else None
        if execution_mode not in ('SEMCACHE_LOOKUP_NO_REUSE', 'SEMCACHE_PHYSICAL_REUSE'):
            raise ValueError('Engine supports SemCache modes; native mode bypasses the engine')
        request_timer = CPUWallTimer()
        request_timer.__enter__()
        timings = TimingRegistry(next(self.model.parameters()).device)
        self.query_id = query_id
        event_start = len(self.events)
        with timings.cpu('tokenization_ms', parent='request_wall_ms'):
            ids = list(self.tokenizer(query_text)['input_ids'])
        if not ids:
            raise ValueError('Empty tokenized query')
        self.model.set_adapter(user_id)
        self.model.eval()
        self.emit('QUERY', query_text=query_text, user_id=user_id, adapter_name=user_id, token_ids=ids)
        with timings.cpu('semantic_encode_ms', parent='request_wall_ms'):
            vector = self.encoder.encode([query_text])[0]
        with timings.cpu('cluster_assign_update_ms', parent='request_wall_ms'):
            cluster_update = self.clusterer.observe_with_diagnostics(vector)
        cluster = cluster_update['cluster_id']
        distance = cluster_update['nearest_centroid_distance_pre_update']
        updated = cluster_update['centroid_update_applied']
        self.emit('CLUSTER_ASSIGN', cluster_distance=distance, cluster_updated=updated,
                  **cluster_update)
        with timings.cpu('subsequence_extract_ms', parent='request_wall_ms'):
            windows = self.extractor.extract(ids)
        self.metrics.arrive([self.matcher.key(cluster, w) for w in windows])
        self.emit('SUBSEQUENCE_EXTRACT', cluster_id=cluster, window_size=self.extractor.window_size,
                  windows=[dict(token_ids=w.token_ids, target_start=w.start, target_end=w.end) for w in windows])
        hits, misses = [], []
        with timings.cpu('cache_lookup_ms', parent='request_wall_ms'):
            for window in windows:
                key = self.matcher.key(cluster, window)
                entry = self.cache.lookup(key, record_reuse=False)
                details = dict(cache_key=key, target_start=window.start, target_end=window.end, cluster_id=cluster)
                self.emit('CACHE_LOOKUP', **details)
                self.emit('HIT' if entry is not None else 'MISS', **details)
                if entry is None:
                    misses.append(window)
                else:
                    hits.append(CacheHit(window, entry, entry.impact or 0.0))
        with timings.cpu('hit_selection_ms', parent='request_wall_ms'):
            selected, mask = select_nonoverlapping(hits, len(ids))
        execution_hits = selected if execution_mode == 'SEMCACHE_PHYSICAL_REUSE' else []
        self.emit('NON_OVERLAP_RESOLVE', hit_count=len(hits), accepted_nonoverlap_hits=len(selected), reused_mask=mask)
        for hit in selected:
            self.emit('FETCH', **self.entry_fields(hit.entry), target_start=hit.window.start, target_end=hit.window.end)
        inputs = dict(input_ids=torch.tensor([ids], device=next(self.model.parameters()).device), use_cache=False)
        with torch.inference_mode():
            baseline = (external_baseline if external_baseline is not None else
                        (self.model(**inputs).logits.detach().cpu() if compare_baseline else None))
            cuda_measure = collect_timing and inputs['input_ids'].is_cuda
            if cuda_measure:
                with timings.cuda('prefill_gpu_ms', parent=None, inclusive='inclusive'):
                    with mixed_projection_path(self.adapter, user_id, execution_hits, len(ids), measure_cuda=True) as audit:
                        output = self.model(**inputs, output_attentions=True)
            else:
                with mixed_projection_path(self.adapter, user_id, execution_hits, len(ids)) as audit:
                    output = self.model(**inputs, output_attentions=True)
        if cuda_measure:
            # Resolve the enclosing event once before any host materialization;
            # all per-projection event pairs are now complete as well.
            timings.values['prefill_gpu_ms'].resolve(synchronize=True)
        for record in audit.records.values():
            self.emit('MIXED_PROJECT', **record)
        impacts = defaultdict(list)
        span_impacts = {}
        for w in windows:
            impact = actual_attention_impact(output.attentions, w.start, w.end,
                                             torch.ones(len(ids), dtype=torch.bool),
                                             self.impact_reducer)
            span_impacts[w.start] = impact
            impacts[self.matcher.key(cluster, w)].append(impact)
        # A query contributes one observation per key; repeated occurrences mean.
        per_query = {key: sum(values)/len(values) for key, values in impacts.items()}
        self.metrics.history.append(cluster, query_id, self.cache.now, per_query)
        chu_timer = CPUWallTimer()
        chu_timer.__enter__()
        for hit in selected:
            old = hit.entry.impact
            self.metrics.reused(hit.entry, span_impacts[hit.window.start])
            self.emit('CHU', **self.entry_fields(hit.entry), old_I=old,
                      current_attention_impact=span_impacts[hit.window.start], new_I=hit.entry.impact,
                      old_impact=old, current_impact=span_impacts[hit.window.start],
                      block_key=hit.entry.key, impact_update_kind='chu')
        chu_timer.__exit__(None, None, None)
        timings.values['chu_update_ms'] = chu_timer.elapsed_ms
        timings.scopes['chu_update_ms'] = dict(timing_scope='chu_update_ms', timing_parent='request_wall_ms',
            inclusive_or_exclusive='exclusive', clock='cpu_perf_counter_ns')
        considered = set()
        policy_timer = CPUWallTimer()
        policy_timer.__enter__()
        materialization_ms = 0.0
        for window in misses:
            key = self.matcher.key(cluster, window)
            if key in self.cache.entries or key in considered:
                continue
            considered.add(key)
            blocks = {layer: tuple(record[n][:, window.start:window.end] for n in 'qkv')
                      for layer, record in audit.projections.items()}
            size = sum(t.numel()*t.element_size() for qkv in blocks.values() for t in qkv)
            entry = CacheEntry(cluster, window.token_ids, (window.start, window.end), size,
                impact=span_impacts[window.start], qkv_metadata=dict(component_scope='total_qkv', source_user=user_id))
            frequencies = self.metrics.frequencies
            normalized = self.cache.admission_metrics(entry, frequencies[key], frequencies)
            score = self.cache.admission.score(normalized)
            allowed = self.cache.admission.admit(normalized) and size <= self.cache.capacity_bytes
            self.emit('ADMISSION_SCORE', **self.entry_fields(entry), admission_F=frequencies[key],
                      F_bar=normalized.frequency, A_bar=normalized.age, I_bar=normalized.impact,
                      S_bar=normalized.size, admission_score=score, admission_decision=allowed,
                      target_start=window.start, target_end=window.end)
            self.emit('ADMIT' if allowed else 'DENY', **self.entry_fields(entry), admission_score=score,
                      admission_decision=allowed)
            before = self.cache.physical_tensor_bytes
            def materialize():
                nonlocal materialization_ms
                materialize_timer = CPUWallTimer()
                materialize_timer.__enter__()
                physical = CacheEntry.from_tensors(cluster, window.token_ids, (window.start, window.end),
                                                  blocks, self.storage_device)
                physical.impact, physical.qkv_metadata = entry.impact, entry.qkv_metadata
                materialize_timer.__exit__(None, None, None)
                materialization_ms += materialize_timer.elapsed_ms
                return physical
            def cache_event(kind, block, eviction_score):
                normalized_fields = {}
                if kind == 'EVICT':
                    population = [self.cache._metrics(e) for e in self.cache.entries.values()]
                    normalized_eviction = normalize(self.cache._metrics(block), population)
                    normalized_fields = dict(F_bar=normalized_eviction.frequency, A_bar=normalized_eviction.age,
                                             I_bar=normalized_eviction.impact, S_bar=normalized_eviction.size)
                self.emit(kind, **self.entry_fields(block), eviction_score=eviction_score,
                          eviction_decision=kind == 'EVICT', **normalized_fields)
            self.cache.insert(entry, frequencies[key], frequencies, materialize=materialize, on_event=cache_event)
            if not allowed and self.cache.physical_tensor_bytes != before:
                raise AssertionError('Denied candidate allocated cache storage')
        policy_timer.__exit__(None, None, None)
        timings.values['cache_policy_ms'] = policy_timer.elapsed_ms
        timings.scopes['cache_policy_ms'] = dict(timing_scope='cache_policy_ms', timing_parent='request_wall_ms',
            inclusive_or_exclusive='inclusive', clock='cpu_perf_counter_ns')
        timings.values['cache_materialization_ms'] = materialization_ms
        timings.scopes['cache_materialization_ms'] = dict(timing_scope='cache_materialization_ms', timing_parent='cache_policy_ms',
            inclusive_or_exclusive='exclusive', clock='cpu_perf_counter_ns')
        pbr_updates = []
        pbr_timer = CPUWallTimer()
        pbr_timer.__enter__()
        if self.pbr_interval_queries and self.clusterer.queries % self.pbr_interval_queries == 0:
            pbr_updates = self.recalculate_cluster(cluster, trigger='fixed_query_interval')
        pbr_timer.__exit__(None, None, None)
        timings.values['pbr_update_ms'] = pbr_timer.elapsed_ms
        timings.scopes['pbr_update_ms'] = dict(timing_scope='pbr_update_ms', timing_parent='request_wall_ms',
            inclusive_or_exclusive='exclusive', clock='cpu_perf_counter_ns')
        logical_reused = sum(mask)
        reused = logical_reused if execution_mode == 'SEMCACHE_PHYSICAL_REUSE' else 0
        recomputed = len(ids)-reused
        assert reused + recomputed == len(ids)
        # The measured request ends after execution and cache-policy work.
        # Logit transfer/comparison below exists only for benchmark diagnostics.
        timing_values, timing_scopes = close_request_timing(
            request_timer, timings, collect_timing)
        quality_timer = CPUWallTimer()
        quality_timer.__enter__()
        logits = output.logits.detach().cpu()
        quality = compare_logits(baseline, logits, selected[0].window.start if selected else 0) if baseline is not None else dict.fromkeys([
            'max_abs_logit_diff', 'mean_abs_logit_diff', 'relative_l2_logit_diff', 'logit_cosine_similarity',
            'last_position_kl_baseline_to_injected', 'affected_suffix_mean_kl', 'baseline_last_argmax_token_id',
            'injected_last_argmax_token_id', 'last_argmax_agreement', 'prefix_max_abs_logit_diff'])
        quality_timer.__exit__(None, None, None)
        if collect_timing:
            timing_values['quality_diagnostics_ms'] = quality_timer.elapsed_ms
            timing_scopes['quality_diagnostics_ms'] = dict(timing_scope='logit materialization and quality comparison',
                timing_parent=None, inclusive_or_exclusive='exclusive_outside_request', clock='cpu_perf_counter_ns')
        module = self.adapter.projection_module(0, 'q')
        d, r, layers = module.in_features, module.r[user_id], len(self.adapter.layers)
        # Analytical equations per layer, then sum over the all-layer scope.
        analytical = projection_savings(reused, d, r, layers)
        savings = dict(paper_estimated_base_flops_saved=analytical['base_flops_saved'],
            paper_estimated_lora_flops_saved=analytical['lora_flops_saved'],
            paper_estimated_comm_elements_saved=analytical['comm_elements_saved'])
        # PEFT casts hidden rows to adapter dtype for the delta calculation.
        comm_bytes = reused*d*layers*(module.get_base_layer().weight.element_size()
                                      + 3*module.lora_B[user_id].weight.element_size())
        if collect_timing:
            timing_values['mixed_qkv_execution_ms'] = mixed_qkv_elapsed_ms(
                audit, enclosing_region_synchronized='prefill_gpu_ms' in timing_values)
            timing_values['qkv_execution_ms'] = timing_values['mixed_qkv_execution_ms']
            timing_scopes['mixed_qkv_execution_ms'] = dict(timing_scope='mixed_qkv_execution_ms',
                timing_parent='prefill_gpu_ms', inclusive_or_exclusive='exclusive', clock='cuda_event')
            timing_scopes['qkv_execution_ms'] = dict(timing_scope='alias of mixed_qkv_execution_ms',
                timing_parent='prefill_gpu_ms', inclusive_or_exclusive='exclusive', clock='cuda_event')
        projection_skip_used = any(record['reused_projection_rows'] > 0 for record in audit.records.values())
        row = dict(query_id=query_id, user_id=user_id, adapter_name=user_id, query_text=query_text,
            token_ids=ids,
            model_id=self.metadata.get('model'), resolved_model_revision=self.metadata.get('resolved_model_revision'),
            dtype=str(next(self.model.parameters()).dtype), seed=self.seed,
            encoder_kind='fixture' if self.encoder.metadata.get('checkpoint') == 'controlled_vectors_v1' else 'huggingface_text',
            encoder_model_id=None if self.encoder.metadata.get('checkpoint') == 'controlled_vectors_v1' else self.encoder.metadata.get('checkpoint'),
            cluster_id=cluster, cluster_distance=distance,
            nearest_centroid_distance_pre_update=distance,
            cluster_updated=updated, centroid_update_applied=updated,
            cluster_count_before=cluster_update['cluster_count_before'],
            cluster_count_after=cluster_update['cluster_count_after'],
            centroid_shift_l2=cluster_update['centroid_shift_l2'],
            cluster_update_mode=self.clusterer.update_mode,
            window_size=self.extractor.window_size, match_policy=self.matcher.match_rule,
            candidate_windows=len(windows), block_lookup_count=len(windows), block_hit_count=len(hits),
            accepted_nonoverlap_hits=len(selected), reused_unique_token_count=reused,
            logical_reusable_token_count=logical_reused, query_token_count=len(ids),
            recomputed_tokens=recomputed, token_reuse_ratio=reused/len(ids),
            block_hit_ratio=len(hits)/len(windows) if windows else 0,
            logical_cache_occupancy=self.cache.logical_cache_bytes, physical_cache_bytes=self.cache.physical_tensor_bytes,
            physical_storage_device=self.storage_device, component_scope='total_qkv',
            native_projection_rows=recomputed, reused_projection_rows=reused, projection_integrity_passed=True,
            projection_accounting_scope='rows per Q/K/V per layer; all layers', projection_layers=layers,
            **savings, paper_estimated_comm_bytes_saved=comm_bytes,
            savings_scope='analytical sum over all layers; no wall-clock speedup claim',
            semantic_impact_provider='actual_attention_probabilities',
            attention_impact_reducer=self.impact_reducer.metadata,
            chu_rho=self.metrics.updater.rho, pbr_history_lambda=self.metrics.history.history_lambda,
            pbr_trigger_policy='manual' if self.pbr_interval_queries is None else 'fixed_query_interval_REPRODUCTION_CHOICE',
            pbr_updates_this_query=len(pbr_updates),
            baseline_comparison_available=baseline is not None, **quality,
            mode=execution_mode, physical_reuse_used=projection_skip_used,
            projection_skip_used=projection_skip_used,
            timing=timing_values, timing_scopes=timing_scopes,
            metric_source='measured except logical capacity and paper_estimated analytical savings', safe_reuse_claimed=False)
        return dict(summary=row, events=self.events[event_start:], logits=logits, projection_audit=audit.records)

    def logical_query(self, query_text, user_id, query_id, *, logical_block_bytes):
        """Workload/cache-event simulation using the M5 policies, without tensors.

        No attention provider: impact stays None, CHU/PBR unavailable. Selected
        positions represent predicted reuse, not executed projection reuse.
        """
        self.query_id = query_id
        start = len(self.events)
        ids = list(self.tokenizer(query_text)['input_ids'])
        if not ids:
            raise ValueError('Empty tokenized query')
        vector = self.encoder.encode([query_text])[0]
        cluster_update = self.clusterer.observe_with_diagnostics(vector)
        cluster = cluster_update['cluster_id']
        windows = self.extractor.extract(ids)
        self.metrics.arrive([self.matcher.key(cluster, w) for w in windows])
        self.emit('CLUSTER_ASSIGN', execution_mode='ANALYTICAL_SIMULATION', **cluster_update)
        self.emit('SUBSEQUENCE_EXTRACT', candidate_windows=len(windows))
        hits, misses = [], []
        for w in windows:
            entry = self.cache.lookup(self.matcher.key(cluster, w), record_reuse=False)
            self.emit('HIT' if entry else 'MISS', cluster_id=cluster, target_start=w.start, token_ids=w.token_ids)
            if entry:
                hits.append(CacheHit(w, entry, entry.impact or 0.0))
            else:
                misses.append(w)
        selected, mask = select_nonoverlapping(hits, len(ids))
        for hit in selected:
            self.cache.record_reuse(hit.entry)
        considered = set()
        admissions = candidates = evictions = 0
        def cache_event(kind, entry, score):
            nonlocal admissions, evictions
            admissions += int(kind == 'INSERT')
            evictions += int(kind == 'EVICT')
            self.emit(kind, **self.entry_fields(entry), eviction_score=score)
        for w in misses:
            key = self.matcher.key(cluster, w)
            if key in considered or key in self.cache.entries:
                continue
            considered.add(key)
            candidates += 1
            entry = CacheEntry(cluster, w.token_ids, (w.start,w.end), logical_block_bytes,
                qkv_metadata=dict(component_scope='logical_total_qkv', source_user=user_id))
            normalized = self.cache.admission_metrics(entry, self.metrics.frequencies[key], self.metrics.frequencies)
            allowed = self.cache.admission.admit(normalized) and logical_block_bytes <= self.cache.capacity_bytes
            self.emit('ADMIT' if allowed else 'DENY', admission_score=self.cache.admission.score(normalized), **self.entry_fields(entry))
            self.cache.insert(entry, self.metrics.frequencies[key], self.metrics.frequencies, on_event=cache_event)
        return dict(summary=dict(query_id=query_id, user_id=user_id, cluster_id=cluster,
            nearest_centroid_distance_pre_update=cluster_update['nearest_centroid_distance_pre_update'],
            centroid_update_applied=cluster_update['centroid_update_applied'],
            cluster_count_before=cluster_update['cluster_count_before'],
            cluster_count_after=cluster_update['cluster_count_after'],
            centroid_shift_l2=cluster_update['centroid_shift_l2'],
            candidate_windows=len(windows), block_lookup_count=len(windows), block_hit_count=len(hits),
            reused_token_count=sum(mask), query_token_count=len(ids),
            admission_count=admissions, admission_candidate_count=candidates, eviction_count=evictions,
            logical_global_cache_bytes=self.cache.logical_cache_bytes, physical_cache_tensor_bytes=0,
            impact_available=False, chu_pbr_status='unavailable_without_attention_provider',
            execution_scope='prefill_only', metric_source='SIMULATED', safe_reuse_claimed=False),
            events=self.events[start:])

    def pbr(self, cluster):
        return self.recalculate_cluster(cluster)

    def recalculate_cluster(self, cluster, trigger='manual'):
        """Force paper Eq.13 over bounded scalar history for one cluster."""
        updates = self.metrics.pbr(cluster)
        for update in updates:
            self.emit('PBR', cluster_id=cluster, impact_update_kind='pbr',
                      trigger_policy=trigger, **update)
        return updates

    def lookup_latest_token(self, cluster_id, token_id, target_position):
        """Decode interface only: singleton exact-key lookup, no prefill rescan.

        w=3 prefill entries cannot satisfy this singleton lookup. Decode population
        and past-key-values execution are explicitly deferred to M6.
        """
        if target_position < 0:
            raise ValueError('Invalid decode position')
        entry = self.cache.lookup((cluster_id, (token_id,)), record_reuse=False)
        return dict(cluster_id=cluster_id, token_id=token_id, target_start=target_position,
                    target_end=target_position+1, hit=entry is not None,
                    execution_scope='lookup_interface_only', safe_reuse_claimed=False)
