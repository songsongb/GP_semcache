"""Query clock, appearance history, actual-reuse updates and manual PBR."""
from collections import Counter, defaultdict, deque
from .impact_updater import AttentionImpactProvider, SemanticImpactUpdater
from .attention_impact import MeanLayerHeadFrobeniusReducer


def attention_impact(attentions, start, end):
    """Choice: row L2 over keys, mean heads, sum target tokens and layers.

    Batch one, actual eager attention probabilities [1, heads, query, key].
    """
    import torch
    if not attentions or not 0 <= start < end:
        raise ValueError('Need attention tensors and a nonempty span')
    value = 0.0
    for attention in attentions:
        if attention is None or attention.ndim != 4 or attention.shape[0] != 1 or end > attention.shape[2]:
            raise ValueError('Need batch-one actual attention weights covering span')
        rows = attention.detach()[:, :, start:end, :].double()
        if not torch.isfinite(rows).all():
            raise ValueError('Nonfinite attention')
        value += rows.norm(dim=-1).mean(dim=1).sum().item()
    SemanticImpactUpdater._validate(value)
    return value


def actual_attention_impact(attentions, start, end, valid_attention_mask=None, reducer=None):
    """M7 reducer entry point; old ``attention_impact`` remains M5-compatible."""
    return (reducer or MeanLayerHeadFrobeniusReducer()).reduce(
        attentions, start, end, valid_attention_mask)


class QueryImpactHistory(AttentionImpactProvider):
    def __init__(self, history_lambda=100):
        if history_lambda < 1:
            raise ValueError('History must be positive')
        self.history_lambda = history_lambda
        self.history = defaultdict(lambda: deque(maxlen=history_lambda))

    def append(self, cluster, query_id, order, impacts):
        for value in impacts.values():
            SemanticImpactUpdater._validate(value)
        self.history[cluster].append(dict(query_id=query_id, order=order, impacts=dict(impacts)))

    def current(self, entry, attention_context):
        return attention_context

    def recent(self, entry, cluster_id, query_window):
        values = [q['impacts'][entry.key] for q in list(self.history[cluster_id])[-query_window:]
                  if entry.key in q['impacts']]
        return sum(values)/len(values) if values else None


class CacheMetricManager:
    def __init__(self, cache, rho=0.8, history_lambda=100, frequency_window=100):
        if frequency_window < 1:
            raise ValueError('Frequency window must be positive')
        self.cache = cache
        self.appearances = deque(maxlen=frequency_window)
        self.history = QueryImpactHistory(history_lambda)
        self.updater = SemanticImpactUpdater(self.history, rho=rho)
        self.frequencies = Counter()

    def arrive(self, keys):
        self.cache.advance(self.cache.now + 1)
        self.appearances.append(Counter(keys))
        self.frequencies = sum(self.appearances, Counter())

    def reused(self, entry, impact):
        self.cache.record_reuse(entry)
        self.updater.on_hit(entry, impact)

    def pbr(self, cluster):
        """Explicit caller-controlled operation; no invented low-load scheduler."""
        updates = []
        for entry in self.cache.entries.values():
            if entry.cluster_id != cluster:
                continue
            value = self.history.recent(entry, cluster, self.history.history_lambda)
            old = entry.impact
            if value is not None:
                self.updater._validate(value)
                entry.impact = value
                entry.updated_at = self.cache.now
            updates.append(dict(cache_key=entry.key, old_I=old, new_I=entry.impact,
                                old_impact=old, I=entry.impact,
                                occurrence_count=sum(entry.key in q['impacts'] for q in self.history.history[cluster]),
                                denominator_zero=value is None))
        return updates
