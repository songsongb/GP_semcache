from .cache_metrics import CacheMetrics, normalize
from .admission import AdmissionPolicy
from .eviction import EvictionPolicy


class GlobalCache:
    """One model/tokenizer namespace per pool. Single-threaded logical cache.

    Entry size/tensor storage must not be mutated after admission. Impact None is
    excluded evidence, mapped to zero solely for the index-demo policy score.
    """
    def __init__(self, capacity_bytes, admission=None, eviction=None, zero_max=0.0):
        if capacity_bytes < 0 or zero_max != 0:
            raise ValueError("Invalid cache configuration")
        self.capacity_bytes = capacity_bytes
        self.admission = admission or AdmissionPolicy()
        self.eviction = eviction or EvictionPolicy()
        self.entries = {}
        self.hits = self.misses = 0
        self.now = 0
        self.zero_max = zero_max

    def advance(self, query_index):
        if query_index < self.now:
            raise ValueError("Query clock cannot go backwards")
        self.now = query_index

    @property
    def logical_cache_bytes(self):
        return sum(e.size_bytes for e in self.entries.values())

    @property
    def physical_tensor_bytes(self):
        storages = {}
        for e in self.entries.values():
            for qkv in (e.tensors or {}).values():
                for t in qkv:
                    s = t.untyped_storage()
                    storages[(str(t.device), s.data_ptr())] = s.nbytes()
        return sum(storages.values())

    def lookup(self, key):
        e = self.entries.get(key)
        if e is None:
            self.misses += 1
        else:
            self.hits += 1
            e.frequency += 1
            e.last_access = e.updated_at = self.now
        return e

    def _metrics(self, e, frequency=None):
        return CacheMetrics(e.frequency if frequency is None else frequency,
                            e.impact if e.impact is not None else 0.0,
                            e.age(self.now), e.size_bytes)

    def insert(self, entry, observed_frequency=1, admission_frequencies=None):
        if entry.key in self.entries or entry.size_bytes > self.capacity_bytes:
            return False
        frequencies = admission_frequencies or {}
        candidate = self._metrics(entry, observed_frequency)
        population = [self._metrics(e, frequencies.get(k, 0)) for k, e in self.entries.items()] + [candidate]
        if not self.admission.admit(normalize(candidate, population, self.zero_max)):
            return False
        entry.created_at = entry.updated_at = entry.last_access = self.now
        self.entries[entry.key] = entry
        while self.logical_cache_bytes > self.capacity_bytes:
            metrics = {k: self._metrics(e) for k, e in self.entries.items()}
            # Largest score; stable lexicographic key breaks ties.
            victim = min(metrics, key=lambda k: (-self.eviction.score(normalize(metrics[k], metrics.values(), self.zero_max)), k))
            del self.entries[victim]
        return entry.key in self.entries
