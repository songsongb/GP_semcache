from .cache_metrics import CacheMetrics, normalize
from .admission import AdmissionPolicy
from .eviction import EvictionPolicy
import time


class GlobalCache:
    """One model/tokenizer namespace per pool. Single-threaded logical cache.

    Entry size/tensor storage must not be mutated after admission. Impact None is
    excluded evidence, mapped to zero solely for the index-demo policy score.
    """
    def __init__(self, capacity_bytes, admission=None, eviction=None, zero_max=0.0, *,
                 physical_storage_mode='RAW_FP16', physical_codec=None,
                 instrument_storage=False):
        if capacity_bytes < 0 or zero_max != 0:
            raise ValueError("Invalid cache configuration")
        self.capacity_bytes = capacity_bytes
        self.admission = admission or AdmissionPolicy()
        self.eviction = eviction or EvictionPolicy()
        self.entries = {}
        self.hits = self.misses = 0
        self.now = 0
        self.peak_logical_cache_bytes = 0
        self.zero_max = zero_max
        if physical_storage_mode not in ('RAW_FP16', 'COMPRESSED_KV_K20_V16'):
            raise ValueError('Unknown physical cache storage mode')
        if (physical_storage_mode == 'RAW_FP16') != (physical_codec is None):
            raise ValueError('Compressed mode requires an injected physical codec; RAW mode takes none')
        self.physical_storage_mode = physical_storage_mode
        self.physical_codec = physical_codec
        self.instrument_storage = instrument_storage
        self.storage_timing_records = [] if instrument_storage else None

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
        compressed_bytes = 0
        for e in self.entries.values():
            if e.compressed_kv is not None:
                compressed_bytes += e.physical_tensor_bytes
                continue
            for qkv in (e.tensors or {}).values():
                for t in qkv:
                    s = t.untyped_storage()
                    storages[(str(t.device), s.data_ptr())] = s.nbytes()
        return sum(storages.values()) + compressed_bytes

    def make_entry(self, cluster_id, token_ids, positions, tensors, storage_device='cpu'):
        """Build the configured physical representation after logical admission."""
        from .cache_entry import CacheEntry
        return CacheEntry.from_tensors(cluster_id, token_ids, positions, tensors,
            storage_device, codec=self.physical_codec)

    def lookup(self, key, record_reuse=True):
        """hits/misses count index lookups. Physical engine always passes False.

        Legacy logical callers may simulate reuse with record_reuse=True; these
        counters do not establish any physically executed projection saving.
        """
        start = time.perf_counter() if self.instrument_storage else None
        e = self.entries.get(key)
        if e is None:
            self.misses += 1
        else:
            self.hits += 1
            if record_reuse:
                self.record_reuse(e)
            if e.compressed_kv is not None:
                e = self.physical_codec.decode_entry(e)
        if self.instrument_storage:
            self.storage_timing_records.append(dict(operation='lookup', key=key,
                lookup_ms=1000*(time.perf_counter()-start),
                decode_ms=e.storage_timings_ms['decode_ms'] if e is not None and e.compressed_kv else None,
                dequantize_ms=e.storage_timings_ms['dequantize_ms'] if e is not None and e.compressed_kv else None))
        return e

    def record_reuse(self, entry):
        entry = getattr(entry, 'resident', entry)
        entry.frequency += 1
        entry.last_access = entry.updated_at = self.now

    def admission_metrics(self, entry, observed_frequency=1, admission_frequencies=None):
        frequencies = admission_frequencies or {}
        candidate = self._metrics(entry, observed_frequency)
        population = [self._metrics(e, frequencies.get(k, 0)) for k, e in self.entries.items()] + [candidate]
        return normalize(candidate, population, self.zero_max)

    def _metrics(self, e, frequency=None):
        return CacheMetrics(e.frequency if frequency is None else frequency,
                            e.impact if e.impact is not None else 0.0,
                            e.age(self.now), e.size_bytes)

    def insert(self, entry, observed_frequency=1, admission_frequencies=None, materialize=None, on_event=None):
        start = time.perf_counter() if self.instrument_storage else None
        inserted = self._insert(entry, observed_frequency, admission_frequencies, materialize, on_event)
        if self.instrument_storage:
            resident = self.entries.get(entry.key)
            timing = resident.storage_timings_ms if resident is not None else {}
            self.storage_timing_records.append(dict(operation='insert', key=entry.key,
                admitted=inserted, insert_ms=1000*(time.perf_counter()-start),
                quantize_ms=timing.get('quantize_ms'), encode_ms=timing.get('encode_ms')))
        return inserted

    def _insert(self, entry, observed_frequency, admission_frequencies, materialize, on_event):
        if entry.key in self.entries or entry.size_bytes > self.capacity_bytes:
            return False
        if not self.admission.admit(self.admission_metrics(entry, observed_frequency, admission_frequencies)):
            return False
        if materialize is not None:
            physical = materialize()
            if physical.key != entry.key or physical.size_bytes != entry.size_bytes:
                raise ValueError('Materialization changed candidate identity or size')
            entry = physical
        if entry.compressed_kv is not None and self.physical_codec is None:
            raise ValueError('Compressed entry requires a compressed cache codec')
        if entry.tensors is not None and self.physical_codec is not None:
            raise ValueError('Compressed cache cannot retain raw Q/K/V tensors')
        entry.created_at = entry.updated_at = entry.last_access = self.now
        self.entries[entry.key] = entry
        if on_event:
            on_event('INSERT', entry, None)
        while self.logical_cache_bytes > self.capacity_bytes:
            metrics = {k: self._metrics(e) for k, e in self.entries.items()}
            # Largest score; stable lexicographic key breaks ties.
            victim = min(metrics, key=lambda k: (-self.eviction.score(normalize(metrics[k], metrics.values(), self.zero_max)), k))
            if on_event:
                on_event('EVICT', self.entries[victim], self.eviction.score(normalize(metrics[victim], metrics.values(), self.zero_max)))
            del self.entries[victim]
        # Resident occupancy after eviction; excludes temporary insertion overflow.
        self.peak_logical_cache_bytes = max(self.peak_logical_cache_bytes, self.logical_cache_bytes)
        return entry.key in self.entries
