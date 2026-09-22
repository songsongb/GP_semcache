"""Tensor-free replay through existing policies; never mutates an inference cache."""
from semcache.cache.global_cache import GlobalCache
from semcache.cache.cache_entry import CacheEntry
from semcache.cache.cache_metrics import normalize
from semcache.cache.metric_manager import CacheMetricManager
from semcache.semantic.hit_selection import CacheHit, select_nonoverlapping
from .impact import finalize, parity


def replay_policy(queries, *, capacity_bytes, rho=.8):
    """Cold -> same-user exact only, using actual sizes/cluster IDs from capture.

    A query supplies windows, impact values, cluster, query_id, token_count, sizes.
    Return decisions plus scores; tests exercise eviction with a smaller capacity.
    """
    cache = GlobalCache(capacity_bytes)
    metrics = CacheMetricManager(cache, rho=rho, history_lambda=100)
    admissions, evictions, selections, resident_scores = [], [], [], []
    for q in queries:
        windows, values, cluster = q['windows'], q['values'], q['cluster']
        if len(windows) != len(values) or len(windows) != len(q['sizes']):
            raise ValueError('Policy replay needs one impact and size per window')
        keys = [(cluster, w.token_ids) for w in windows]
        metrics.arrive(keys)
        hits, misses = [], []
        for w, key, size in zip(windows, keys, q['sizes']):
            entry = cache.lookup(key, record_reuse=False)
            if entry is None:
                misses.append((w, key, size))
            else:
                hits.append(CacheHit(w, entry, entry.impact or 0.))
        selected, _ = select_nonoverlapping(hits, q['token_count'])
        selections.append([(h.window.start, h.window.end, h.entry.key) for h in selected])
        spans, _ = finalize(windows, values, cluster, metrics.history)
        for h in selected:
            metrics.reused(h.entry, spans[h.window.start])
        considered = set()
        for w, key, size in misses:
            if key in cache.entries or key in considered:
                continue
            considered.add(key)
            entry = CacheEntry(cluster, w.token_ids, (w.start, w.end), size, impact=spans[w.start])
            normalized = cache.admission_metrics(entry, metrics.frequencies[key], metrics.frequencies)
            allowed = cache.admission.admit(normalized) and size <= capacity_bytes
            admissions.append(dict(query_id=q['query_id'], key=key, start=w.start, end=w.end,
                decision=allowed, score=cache.admission.score(normalized)))
            def on_event(kind, victim, score):
                if kind == 'EVICT':
                    evictions.append(dict(query_id=q['query_id'], key=victim.key, score=score))
            cache.insert(entry, metrics.frequencies[key], metrics.frequencies, on_event=on_event)
        population = [cache._metrics(e) for e in cache.entries.values()]
        resident_scores.append([(k, cache.eviction.score(normalize(cache._metrics(e), population)))
                                for k, e in sorted(cache.entries.items())])
    return dict(admissions=admissions, evictions=evictions, selections=selections,
        resident_scores=resident_scores, resident_keys=list(sorted(cache.entries)))


def compare_policy(reference, candidate):
    def decisions(records):
        return [{k: v for k, v in x.items() if k != 'score'} for x in records]
    admissions = decisions(reference['admissions']) == decisions(candidate['admissions'])
    evictions = decisions(reference['evictions']) == decisions(candidate['evictions'])
    resident_keys = [[k for k, _ in stage] for stage in reference['resident_scores']] == [
        [k for k, _ in stage] for stage in candidate['resident_scores']]
    a = [x['score'] for x in reference['admissions']]
    b = [x['score'] for x in candidate['admissions']]
    e = [x['score'] for x in reference['evictions']] + [s for stage in reference['resident_scores'] for _, s in stage]
    f = [x['score'] for x in candidate['evictions']] + [s for stage in candidate['resident_scores'] for _, s in stage]
    a_error = parity(a, b) if admissions else None
    e_error = parity(e, f) if evictions and resident_keys else None
    return dict(admission_decisions_identical=admissions,
        admission_scores_within_tolerance=bool(a_error and a_error['impact_values_within_tolerance']),
        eviction_decisions_identical=evictions, eviction_occurred=bool(reference['evictions']),
        eviction_scores_within_tolerance=bool(e_error and e_error['impact_values_within_tolerance']),
        admission_score_errors=a_error, eviction_score_errors=e_error,
        physical_hit_selection_identical=reference['selections'] == candidate['selections'],
        resident_keys_identical=reference['resident_keys'] == candidate['resident_keys'],
        scope='existing cold/exact admission, CHU, eviction policies; no tensor materialization; no PBR at two queries')
