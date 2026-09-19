from pathlib import Path
import random
import pytest
from semcache.semantic.intent_clusterer import IntentClusterer
from semcache.semantic.subsequence import SubsequenceExtractor, Subsequence
from semcache.semantic.matcher import ExactTokenMatcher
from semcache.cache.cache_entry import CacheEntry
from semcache.cache.cache_metrics import CacheMetrics, normalize
from semcache.cache.admission import AdmissionPolicy
from semcache.cache.eviction import EvictionPolicy
from semcache.cache.global_cache import GlobalCache
from semcache.cache.impact_updater import SemanticImpactUpdater
from semcache.demo import run_demo
from semcache.utils.io import load_config
from semcache.utils.seed import seed_everything


def test_cluster_assignment_and_mean():
    c = IntentClusterer(2)
    c.initialize([[0, 0], [2, 0]])
    assert c.assign([1, 0]) == 0  # tie -> lowest ID
    assert c.assign([1.9, 0]) == 1
    c.update(0, [2, 4])
    assert c.centroids[0] == [1, 2]
    assert c.counts == [2, 1]


def test_batched_update():
    # Optional legacy batching is a REPRODUCTION_CHOICE, not paper Eq.9 timing.
    c = IntentClusterer(1, update_interval=2, update_mode='buffered')
    c.initialize([[0]])
    c.observe([3])
    assert c.centroids == [[0]]
    c.observe([6])
    assert c.centroids == [[3]]
    assert c.counts == [3]
    with pytest.raises(ValueError):
        c.assign([1, 2])


@pytest.mark.parametrize('w', [1, 2, 3, 4, 5])
def test_windows(w):
    tokens = [1, 2, 3, 4, 5]
    windows = SubsequenceExtractor(w).extract(tokens)
    assert len(windows) == 6-w
    assert windows[-1].token_ids == tuple(tokens[-w:])
    assert windows[-1].end == len(tokens)
    assert SubsequenceExtractor(w).extract([]) == []


def test_exact_matching():
    m = ExactTokenMatcher()
    a = Subsequence((1, 2, 3), 0, 3)
    b = Subsequence((1, 2, 3), 4, 7)
    assert m.key(0, a) == m.key(0, b)
    assert m.key(0, a) != m.key(1, b)
    assert m.key(0, a) != m.key(0, Subsequence((3, 2, 1), 0, 3))


def entry(token=1, size=10):
    return CacheEntry(0, (token,), (0, 1), size)


def test_cache_bytes_hits_misses_duplicate_oversize():
    c = GlobalCache(20)
    e = entry()
    assert c.lookup(e.key) is None
    assert c.insert(e)
    assert not c.insert(entry())
    assert c.logical_cache_bytes == 10
    assert c.physical_tensor_bytes == 0
    c.advance(3)
    assert c.lookup(e.key) is e
    assert e.frequency == 1 and e.age(5) == 2
    assert (c.hits, c.misses) == (1, 1)
    assert not c.insert(entry(2, 30))
    assert c.logical_cache_bytes == 10


def test_eviction_chooses_maximum_and_ties_are_stable():
    c = GlobalCache(20)
    assert c.insert(entry(1))
    assert c.insert(entry(2))
    c.advance(1)
    c.lookup((0, (1,)))
    assert c.insert(entry(3))
    assert (0, (2,)) not in c.entries  # cold and older than candidate
    assert c.logical_cache_bytes == 20
    tied = GlobalCache(10)
    tied.insert(entry(1))
    tied.insert(entry(2))
    assert list(tied.entries) == [(0, (2,))]


def test_equations_and_normalization():
    m = CacheMetrics(.6, .4, .2, .8)
    assert AdmissionPolicy().score(m) == pytest.approx(.46)
    assert EvictionPolicy().score(m) == pytest.approx(.46)
    assert not AdmissionPolicy(threshold=.5).admit(CacheMetrics(1, 0, 0, 1))
    raw = CacheMetrics(3, 2, 1, 4)
    assert normalize(raw, [CacheMetrics(5, 5, 5, 5)]) == m
    zero = CacheMetrics(0, 0, 0, 0)
    assert normalize(zero, [zero]) == zero
    with pytest.raises(ValueError):
        normalize(CacheMetrics(-1, 0, 0, 0), [zero])
    with pytest.raises(ValueError):
        AdmissionPolicy(alpha=2)


def test_impact_requires_observations_and_modes():
    e = entry()
    with pytest.raises(NotImplementedError):
        SemanticImpactUpdater().on_hit(e, None)
    class Provider:
        def current(self, entry, context):
            return context
        def recent(self, entry, cluster_id, query_window):
            return 3
    e.impact = 1
    updater = SemanticImpactUpdater(Provider())
    updater.on_hit(e, 2)
    assert e.impact == pytest.approx(1.2)
    updater.periodic([e], 0, 99)
    assert e.impact == pytest.approx(1.2)
    updater.periodic([e], 0, 100)
    assert e.impact == 3
    SemanticImpactUpdater(mode='pbr').on_hit(e, None)
    SemanticImpactUpdater(mode='chu').periodic([e], 0, 100)


def test_seed_and_end_to_end():
    seed_everything(42)
    a = [random.random() for _ in range(5)]
    seed_everything(42)
    assert a == [random.random() for _ in range(5)]
    config = load_config(Path(__file__).parents[1] / 'configs/base.yaml')
    first = run_demo(config)
    assert first == run_demo(config)
    assert [e['status'] for e in first['events']] == ['MISS', 'MISS', 'MISS', 'HIT', 'HIT', 'MISS']
    assert first['summary']['cache_hits'] == 2
    assert first['logical_cache_bytes'] == 3*3*3*32*4096*2
    assert first['insertions'][-1]['admitted'] is False
    assert first['physical_tensor_bytes'] == 0
    assert first['summary']['reused_tokens'] == 0
    assert first['matched_token_positions'] == 4  # overlap counted once
    assert first['summary']['system_latency_s'] is None


def test_physical_storage_accounting_without_torch():
    class Storage:
        def data_ptr(self): return 123
        def nbytes(self): return 24
    class Tensor:
        device = 'cpu'
        def untyped_storage(self): return Storage()
    t = Tensor()
    e = entry()
    e.tensors = {0: (t, t, t)}
    c = GlobalCache(20)
    c.insert(e)
    second = entry(2)
    second.tensors = {0: (t, t, t)}
    c.insert(second)
    assert e.physical_tensor_bytes == 24
    assert c.physical_tensor_bytes == 24  # aliased storage is counted once
