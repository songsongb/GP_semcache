"""M7 offline tests. No Hugging Face download is performed."""
import pytest

torch = pytest.importorskip("torch")

from semcache.semantic.encoder import TinyBERTSemanticEncoder, masked_mean_pool
from semcache.semantic.intent_clusterer import IntentClusterer
from semcache.cache.attention_impact import MeanLayerHeadFrobeniusReducer
from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.cache.metric_manager import CacheMetricManager, QueryImpactHistory


class FakeTokenizer:
    name_or_path = "offline-tinybert-tokenizer"

    def __call__(self, texts, **kwargs):
        rows = [[(ord(char) % 17) + 1 for char in text] for text in texts]
        width = max(map(len, rows))
        ids = [row + [0] * (width-len(row)) for row in rows]
        mask = [[1] * len(row) + [0] * (width-len(row)) for row in rows]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(18, 4)
        self.config = type("Config", (), {"hidden_size": 4, "_commit_hash": "offline"})()

    def forward(self, input_ids, attention_mask):
        return type("Output", (), {"last_hidden_state": self.embedding(input_ids)})()


def make_encoder():
    torch.manual_seed(4)
    return TinyBERTSemanticEncoder("offline-tinybert", "test", tokenizer=FakeTokenizer(),
                                   model=FakeModel(), device="cpu")


def test_tinybert_deterministic_batch_single_and_no_grad():
    encoder = make_encoder()
    first = torch.tensor(encoder.encode(["same"])[0])
    second = torch.tensor(encoder.encode(["same"])[0])
    batch = torch.tensor(encoder.encode(["same", "longer"])[0])
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, batch)
    assert not encoder.model.training
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())


def test_masked_mean_ignores_padding():
    hidden = torch.tensor([[[1., 2.], [3., 4.], [100., 200.]]])
    actual = masked_mean_pool(hidden, torch.tensor([[1, 1, 0]]))
    torch.testing.assert_close(actual, torch.tensor([[2., 3.]]))


def test_immediate_eq9_uses_preupdate_centroid_and_updates_only_assignment():
    clusterer = IntentClusterer(2)
    clusterer.initialize([[1., 3.], [20., 20.]], counts=[3, 2])
    untouched = list(clusterer.centroids[1])
    result = clusterer.observe_with_diagnostics([5., 7.])
    assert result['cluster_id'] == 0
    assert result['nearest_centroid_distance_pre_update'] == pytest.approx(32 ** .5)
    assert result['centroid_update_applied'] is True
    assert result['cluster_count_before'] == 3 and result['cluster_count_after'] == 4
    assert result['centroid_shift_l2'] == pytest.approx(2 ** .5)
    assert clusterer.centroids[0] == pytest.approx([2., 4.])
    assert clusterer.centroids[1] == untouched
    assert clusterer.counts == [4, 2]


def test_second_query_sees_updated_centroid_and_each_observation_counts_once():
    clusterer = IntentClusterer(2)
    clusterer.initialize([[0.], [10.]])  # first_k samples imply counts [1, 1].
    first = clusterer.observe_with_diagnostics([4.])
    assert first['cluster_id'] == 0 and clusterer.centroids[0] == [2.]
    second = clusterer.observe_with_diagnostics([6.])
    # With the updated centroid, distances tie at four and stable tie-break selects 0.
    assert second['cluster_id'] == 0
    assert second['nearest_centroid_distance_pre_update'] == pytest.approx(4.)
    assert clusterer.centroids[0] == pytest.approx([10./3])
    assert clusterer.counts == [3, 1]


def test_identical_embedding_applies_eq9_and_increments_count_with_zero_shift():
    clusterer = IntentClusterer(1)
    clusterer.initialize([[2., 4.]])
    result = clusterer.observe_with_diagnostics([2., 4.])
    assert result['centroid_update_applied']
    assert result['cluster_count_before'] == 1 and result['cluster_count_after'] == 2
    assert result['centroid_shift_l2'] == 0
    assert clusterer.counts == [2]


def test_first_k_counts_and_optional_buffered_mode_are_explicit():
    clusterer = IntentClusterer(2)
    clusterer.initialize([[1.], [9.], [100.]])
    assert clusterer.centroids == [[1.], [9.]] and clusterer.counts == [1, 1]
    buffered = IntentClusterer(1, update_interval=2, update_mode='buffered')
    buffered.initialize([[0.]])
    assert not buffered.observe_with_diagnostics([2.])['centroid_update_applied']
    buffered.observe([4.])
    assert buffered.centroids == [[2.]] and buffered.counts == [3]


def test_attention_reducer_known_value_and_invalid_cells_excluded():
    reducer = MeanLayerHeadFrobeniusReducer()
    # Block keys [0:2), all ones. Causal valid cells are (q0,k0),
    # (q1,k0/k1), (q2,k0/k1): sqrt(5) for one layer/head.
    attention = torch.ones(1, 1, 3, 3)
    assert reducer.reduce([attention], 0, 2) == pytest.approx(5 ** .5)
    # Padding removes q2 and key1, leaving two valid cells.
    mask = torch.tensor([1, 0, 1])
    assert reducer.reduce([attention], 0, 2, mask) == pytest.approx(2 ** .5)


def test_chu_hit_only_exact_arithmetic():
    cache = GlobalCache(100)
    manager = CacheMetricManager(cache, rho=.8)
    entry = CacheEntry(0, (1,), (0, 1), 10, impact=.5)
    assert cache.insert(entry)
    cache.lookup(entry.key, record_reuse=False)  # lookup alone is not CHU
    assert entry.impact == .5
    manager.reused(entry, 1.)
    assert entry.impact == pytest.approx(.6)


def test_pbr_equation_bounded_history_and_zero_occurrence():
    cache = GlobalCache(100)
    manager = CacheMetricManager(cache, history_lambda=2)
    present = CacheEntry(0, (1,), (0, 1), 10, impact=9.)
    absent = CacheEntry(0, (2,), (0, 1), 10, impact=7.)
    assert cache.insert(present) and cache.insert(absent)
    manager.history.append(0, "old", 0, {present.key: 100.})
    manager.history.append(0, "a", 1, {present.key: 2.})
    manager.history.append(0, "b", 2, {present.key: 4.})
    assert [q["query_id"] for q in manager.history.history[0]] == ["a", "b"]
    updates = manager.pbr(0)
    assert present.impact == pytest.approx(3.)
    assert absent.impact == 7.
    assert next(row for row in updates if row["cache_key"] == absent.key)["denominator_zero"]


def test_optional_real_tinybert(monkeypatch):
    import os
    if os.environ.get("SEMCACHE_TINYBERT_INTEGRATION") != "1":
        pytest.skip("Set SEMCACHE_TINYBERT_INTEGRATION=1; local model files required")
    encoder = TinyBERTSemanticEncoder(local_files_only=True)
    assert encoder.encode(["hello"])[0] == pytest.approx(encoder.encode(["hello"])[0])
