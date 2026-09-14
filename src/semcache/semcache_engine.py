"""Controlled prefill orchestration. Architecture details belong in ModelAdapter."""
import math
from collections import defaultdict
from .semantic.subsequence import SubsequenceExtractor
from .semantic.matcher import ExactTokenMatcher
from .semantic.hit_selection import CacheHit, select_nonoverlapping
from .cache.cache_entry import CacheEntry
from .cache.metric_manager import CacheMetricManager, attention_impact
from .cache.cache_metrics import normalize
from .edgelora.mixed_projection import mixed_projection_path
from .evaluation.logit_metrics import compare_logits


class SemCacheEngine:
    def __init__(self, model, tokenizer, adapter, encoder, clusterer, cache, *, window_size=3,
                 storage_device='cpu', rho=0.8, history_lambda=100, frequency_window=100,
                 metadata=None, seed=42):
        self.model, self.tokenizer, self.adapter = model, tokenizer, adapter
        self.encoder, self.clusterer, self.cache = encoder, clusterer, cache
        self.extractor, self.matcher = SubsequenceExtractor(window_size), ExactTokenMatcher()
        self.storage_device, self.metadata, self.seed = storage_device, metadata or {}, seed
        self.metrics = CacheMetricManager(cache, rho, history_lambda, frequency_window)
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

    def query(self, query_text, user_id, query_id, compare_baseline=False):
        import torch
        self.query_id = query_id
        event_start = len(self.events)
        ids = list(self.tokenizer(query_text)['input_ids'])
        if not ids:
            raise ValueError('Empty tokenized query')
        self.model.set_adapter(user_id)
        self.model.eval()
        self.emit('QUERY', query_text=query_text, user_id=user_id, adapter_name=user_id, token_ids=ids)
        vector = self.encoder.encode([query_text])[0]
        cluster = self.clusterer.assign(vector)
        distance = math.dist(vector, self.clusterer.centroids[cluster])
        self.clusterer.observe(vector)
        updated = self.clusterer.queries % self.clusterer.update_interval == 0
        self.emit('CLUSTER_ASSIGN', cluster_id=cluster, cluster_distance=distance, cluster_updated=updated)
        windows = self.extractor.extract(ids)
        self.metrics.arrive([self.matcher.key(cluster, w) for w in windows])
        self.emit('SUBSEQUENCE_EXTRACT', cluster_id=cluster, window_size=self.extractor.window_size,
                  windows=[dict(token_ids=w.token_ids, target_start=w.start, target_end=w.end) for w in windows])
        hits, misses = [], []
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
        selected, mask = select_nonoverlapping(hits, len(ids))
        self.emit('NON_OVERLAP_RESOLVE', hit_count=len(hits), accepted_nonoverlap_hits=len(selected), reused_mask=mask)
        for hit in selected:
            self.emit('FETCH', **self.entry_fields(hit.entry), target_start=hit.window.start, target_end=hit.window.end)
        inputs = dict(input_ids=torch.tensor([ids], device=next(self.model.parameters()).device), use_cache=False)
        with torch.inference_mode():
            baseline = self.model(**inputs).logits.detach().cpu() if compare_baseline else None
            with mixed_projection_path(self.adapter, user_id, selected, len(ids)) as audit:
                output = self.model(**inputs, output_attentions=True)
        for record in audit.records.values():
            self.emit('MIXED_PROJECT', **record)
        impacts = defaultdict(list)
        span_impacts = {}
        for w in windows:
            impact = attention_impact(output.attentions, w.start, w.end)
            span_impacts[w.start] = impact
            impacts[self.matcher.key(cluster, w)].append(impact)
        # A query contributes one observation per key; repeated occurrences mean.
        per_query = {key: sum(values)/len(values) for key, values in impacts.items()}
        self.metrics.history.append(cluster, query_id, self.cache.now, per_query)
        for hit in selected:
            old = hit.entry.impact
            self.metrics.reused(hit.entry, span_impacts[hit.window.start])
            self.emit('CHU', **self.entry_fields(hit.entry), old_impact=old,
                      current_impact=span_impacts[hit.window.start], impact_update_kind='chu')
        considered = set()
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
                physical = CacheEntry.from_tensors(cluster, window.token_ids, (window.start, window.end),
                                                  blocks, self.storage_device)
                physical.impact, physical.qkv_metadata = entry.impact, entry.qkv_metadata
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
        reused = sum(mask)
        recomputed = len(ids)-reused
        assert reused + recomputed == len(ids)
        logits = output.logits.detach().cpu()
        quality = compare_logits(baseline, logits, selected[0].window.start if selected else 0) if baseline is not None else dict.fromkeys([
            'max_abs_logit_diff', 'mean_abs_logit_diff', 'relative_l2_logit_diff', 'logit_cosine_similarity',
            'last_position_kl_baseline_to_injected', 'affected_suffix_mean_kl', 'baseline_last_argmax_token_id',
            'injected_last_argmax_token_id', 'last_argmax_agreement', 'prefix_max_abs_logit_diff'])
        module = self.adapter.projection_module(0, 'q')
        d, r, layers = module.in_features, module.r[user_id], len(self.adapter.layers)
        # Analytical equations per layer, then sum over the all-layer scope.
        savings = dict(paper_estimated_base_flops_saved=6*reused*d*d*layers,
            paper_estimated_lora_flops_saved=6*reused*d*r*layers,
            paper_estimated_comm_elements_saved=4*reused*d*layers)
        # PEFT casts hidden rows to adapter dtype for the delta calculation.
        comm_bytes = reused*d*layers*(module.get_base_layer().weight.element_size()
                                      + 3*module.lora_B[user_id].weight.element_size())
        row = dict(query_id=query_id, user_id=user_id, adapter_name=user_id, query_text=query_text,
            model_id=self.metadata.get('model'), resolved_model_revision=self.metadata.get('resolved_model_revision'),
            dtype=str(next(self.model.parameters()).dtype), seed=self.seed,
            encoder_kind='fixture' if self.encoder.metadata.get('checkpoint') == 'controlled_vectors_v1' else 'huggingface_text',
            encoder_model_id=None if self.encoder.metadata.get('checkpoint') == 'controlled_vectors_v1' else self.encoder.metadata.get('checkpoint'),
            cluster_id=cluster, cluster_distance=distance,
            cluster_updated=updated, window_size=self.extractor.window_size, match_policy=self.matcher.match_rule,
            candidate_windows=len(windows), block_lookup_count=len(windows), block_hit_count=len(hits),
            accepted_nonoverlap_hits=len(selected), reused_unique_token_count=reused, query_token_count=len(ids),
            recomputed_tokens=recomputed, token_reuse_ratio=reused/len(ids),
            block_hit_ratio=len(hits)/len(windows) if windows else 0,
            logical_cache_occupancy=self.cache.logical_cache_bytes, physical_cache_bytes=self.cache.physical_tensor_bytes,
            physical_storage_device=self.storage_device, component_scope='total_qkv',
            native_projection_rows=recomputed, reused_projection_rows=reused, projection_integrity_passed=True,
            projection_accounting_scope='rows per Q/K/V per layer; all layers', projection_layers=layers,
            **savings, paper_estimated_comm_bytes_saved=comm_bytes,
            savings_scope='analytical sum over all layers; no wall-clock speedup claim',
            baseline_comparison_available=baseline is not None, **quality,
            metric_source='measured except logical capacity and paper_estimated analytical savings', safe_reuse_claimed=False)
        return dict(summary=row, events=self.events[event_start:], logits=logits, projection_audit=audit.records)

    def pbr(self, cluster):
        updates = self.metrics.pbr(cluster)
        for update in updates:
            self.emit('OPTIONAL_PBR', cluster_id=cluster, impact_update_kind='manual_pbr', **update)
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
