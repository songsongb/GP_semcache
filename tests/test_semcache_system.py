"""M5 offline semantic/policy tests: no model dependency or download."""
import pytest
from semcache.evaluation.system_validation import validate_frontend, validate_policy
from semcache.semantic.encoder import HuggingFaceTextEncoder
from semcache.semantic.hit_selection import CacheHit, select_nonoverlapping
from semcache.semantic.subsequence import Subsequence
from semcache.cache.cache_entry import CacheEntry
from semcache.cache.metric_manager import QueryImpactHistory, CacheMetricManager
from semcache.cache.global_cache import GlobalCache


def test_frontend_and_policy_numerical_validators():
    assert validate_frontend()[0]['passed']
    assert validate_policy()[0]['passed']


def test_encoder_requires_explicit_checkpoint():
    with pytest.raises(ValueError, match='explicit'):
        HuggingFaceTextEncoder(None, 'main')


def test_overlap_ties_and_source_target_positions():
    a = CacheEntry(0, (1, 2, 3), (9, 12), 10)
    b = CacheEntry(0, (2, 3, 4), (8, 11), 10)
    wa, wb = Subsequence(a.token_ids, 0, 3), Subsequence(b.token_ids, 0, 3)
    hits, mask = select_nonoverlapping([CacheHit(wa, a, 1), CacheHit(wb, b, 2)], 4)
    assert hits[0].entry is b and mask == [True, True, True, False]
    hits, _ = select_nonoverlapping([CacheHit(wb, b), CacheHit(wa, a)], 4)
    assert hits[0].entry is a and a.positions == (9, 12) and hits[0].window.start == 0


def test_history_cluster_isolation_horizon_and_nonfinite():
    history = QueryImpactHistory(2)
    entry = CacheEntry(0, (1,), (0, 1), 10, impact=5)
    for order, value in enumerate((10., 2., 4.)):
        history.append(0, str(order), order, {entry.key: value})
    history.append(1, 'other', 3, {(1, (1,)): 100.})
    assert history.recent(entry, 0, 2) == 3.
    assert [q['query_id'] for q in history.history[0]] == ['1', '2']
    for value in (float('nan'), float('inf'), -1.):
        with pytest.raises(ValueError):
            history.append(0, 'bad', 4, {entry.key: value})


def test_passive_lookup_only_actual_reuse_changes_metrics():
    cache = GlobalCache(20)
    manager = CacheMetricManager(cache, frequency_window=2)
    entry = CacheEntry(0, (1,), (0, 1), 10, impact=1)
    cache.insert(entry)
    manager.arrive([entry.key, entry.key])
    assert cache.lookup(entry.key, record_reuse=False) is entry
    assert entry.frequency == 0 and entry.age(cache.now) == 1
    manager.reused(entry, 6.)
    assert entry.frequency == 1 and entry.impact == pytest.approx(2.) and entry.age(cache.now) == 0
    manager.arrive([])
    manager.arrive([])
    assert manager.frequencies[entry.key] == 0 and entry.size_bytes == 10
