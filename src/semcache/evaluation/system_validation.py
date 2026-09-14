"""Download-free numerical validators shared by CLIs and unit tests."""
import math
from semcache.semantic.encoder import ControlledEncoder
from semcache.semantic.intent_clusterer import IntentClusterer
from semcache.semantic.subsequence import SubsequenceExtractor, Subsequence
from semcache.semantic.matcher import ExactTokenMatcher
from semcache.semantic.hit_selection import CacheHit, select_nonoverlapping
from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.cache.cache_metrics import CacheMetrics, normalize
from semcache.cache.metric_manager import CacheMetricManager


def validate_frontend():
    encoder = ControlledEncoder({'health-like': [0., 0.], 'weather-like': [10., 10.]})
    assert encoder.encode(['health-like']) == encoder.encode(['health-like'])
    clusterer = IntentClusterer(2, update_interval=2)
    clusterer.initialize([[0, 0], [10, 10]], counts=[1, 1])
    assert clusterer.observe([2, 0]) == 0 and clusterer.centroids[0] == [0, 0]
    assert clusterer.observe([4, 0]) == 0 and clusterer.centroids[0] == [2, 0]
    assert clusterer.assign(encoder.encode(['weather-like'])[0]) == 1
    windows = SubsequenceExtractor(3).extract([1, 2, 3, 4, 5, 6, 7, 8])
    assert len(windows) == 6 and windows[1].token_ids == (2, 3, 4)
    matcher = ExactTokenMatcher()
    entries = [CacheEntry(0, w.token_ids, (w.start, w.end), 10) for w in windows]
    shifted = Subsequence((1, 2, 3), 4, 7)
    assert matcher.key(0, shifted) == entries[0].key
    assert matcher.key(1, shifted) != entries[0].key
    assert entries[0].key != entries[1].key
    selected, mask = select_nonoverlapping([CacheHit(w, e) for w, e in reversed(list(zip(windows, entries)))], 8)
    assert [h.window.start for h in selected] == [0, 3] and sum(mask) == 6
    return [dict(check='semantic_frontend', passed=True, clusters=2, window_size=3,
        source_start=0, shifted_target_start=4, selected_starts=[h.window.start for h in selected],
        reused_unique_tokens=sum(mask), match_policy=matcher.match_rule,
        metric_source='fixture correctness', safe_reuse_claimed=False)]


def validate_policy():
    cache = GlobalCache(20)  # DEVELOPMENT-only two 10-byte logical blocks.
    manager = CacheMetricManager(cache, history_lambda=100)
    zero = CacheMetrics(0, 0, 0, 0)
    assert normalize(zero, [zero]) == zero
    m = normalize(CacheMetrics(3, 2, 1, 4), [CacheMetrics(5, 5, 5, 5)])
    assert m == CacheMetrics(.6, .4, .2, .8)
    assert math.isclose(cache.admission.score(m), .46)
    assert math.isclose(cache.eviction.score(m), .46)
    assert not cache.admission.admit(CacheMetrics(0, 1, 0, 1))  # exactly 0.3
    entries = [CacheEntry(0, (i,), (0, 1), 10, impact=1.) for i in (1, 2, 3)]
    first, second, third = entries
    manager.arrive([first.key, second.key])
    assert cache.insert(first) and cache.insert(second)
    manager.arrive([first.key])
    assert manager.frequencies[first.key] == 2 and second.age(cache.now) == 1
    manager.reused(first, 6.)
    assert math.isclose(first.impact, 2.) and first.frequency == 1 and first.age(cache.now) == 0
    assert first.size_bytes == 10
    for i in range(102):
        manager.history.append(0, f'q{i}', i, {first.key: 4. if i % 2 else 2.})
    assert len(manager.history.history[0]) == 100
    manager.pbr(0)
    assert first.impact == 3. and second.impact == 1.  # denominator zero retains
    events = []
    cache.insert(third, on_event=lambda kind, e, score: events.append((kind, e.key, score)))
    assert second.key not in cache.entries and first.key in cache.entries and third.key in cache.entries
    assert events[-1][0:2] == ('EVICT', second.key)
    # A low-frequency, zero-impact candidate fails before any factory invocation.
    denied = CacheEntry(0, (4,), (0, 1), 10, impact=0.)
    def forbidden():
        raise AssertionError('Denied candidate must not materialize')
    assert not cache.insert(denied, 1, {first.key: 10}, materialize=forbidden)
    return [dict(check='cache_policy', passed=True, normalized_F=m.frequency,
        admission_score=.46, eviction_score=.46, strict_threshold_passed=True,
        chu_impact=2., pbr_impact=first.impact, history_length=100, zero_denominator_retained=True,
        evicted_key=second.key, victim_score=events[-1][2], logical_capacity_bytes=20,
        logical_cache_occupancy=cache.logical_cache_bytes, physical_cache_bytes=cache.physical_tensor_bytes,
        denied_factory_calls=0, metric_source='synthetic numerical correctness', safe_reuse_claimed=False)]
