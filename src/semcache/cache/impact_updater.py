from abc import ABC, abstractmethod
import math


class AttentionImpactProvider(ABC):
    @abstractmethod
    def current(self, entry, attention_context):
        """Return attention-norm impact; never substitute fabricated observations."""

    @abstractmethod
    def recent(self, entry, cluster_id, query_window):
        """Eq. 13 conditional mean, or None when no matching observations exist."""


class SemanticImpactUpdater:
    def __init__(self, provider=None, rho=0.8, pbr_interval_queries=100, mode="chu+pbr"):
        if not 0 <= rho < 1 or pbr_interval_queries < 1 or mode not in {"chu", "pbr", "chu+pbr"}:
            raise ValueError("Invalid impact policy")
        self.provider, self.rho, self.interval, self.mode = provider, rho, pbr_interval_queries, mode

    def _require_provider(self):
        if self.provider is None:
            raise NotImplementedError("Attention-norm reconstruction is not implemented")

    @staticmethod
    def _validate(value):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid impact observation")

    def on_hit(self, entry, context):
        if self.mode == "pbr":
            return
        self._require_provider()
        new = self.provider.current(entry, context)
        self._validate(new)
        entry.impact = new if entry.impact is None else self.rho*entry.impact + (1-self.rho)*new

    def periodic(self, entries, cluster_id, cluster_query_count):
        if self.mode == "chu" or cluster_query_count <= 0 or cluster_query_count % self.interval:
            return
        self._require_provider()
        for e in entries:
            if e.cluster_id == cluster_id:
                new = self.provider.recent(e, cluster_id, self.interval)
                if new is not None:
                    self._validate(new)
                    e.impact = new
