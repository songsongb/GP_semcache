from .cache_metrics import CacheMetrics, normalize
from .admission import AdmissionPolicy
from .eviction import EvictionPolicy
import math
import heapq
import time


class GlobalCache:
    """One model/tokenizer namespace per pool. Single-threaded logical cache.

    Entry size/tensor storage must not be mutated after admission. Impact None is
    excluded evidence, mapped to zero solely for the index-demo policy score.
    """
    def __init__(self, capacity_bytes, admission=None, eviction=None, zero_max=0.0, *,
                 physical_storage_mode='RAW_FP16', physical_codec=None,
                 instrument_storage=False, capacity_charge=None, shared_overhead_bytes=0,
                 homogeneous_logical_fastpath=False):
        if capacity_bytes < 0 or zero_max != 0:
            raise ValueError("Invalid cache configuration")
        self.capacity_bytes = capacity_bytes
        self.admission = admission or AdmissionPolicy()
        self.eviction = eviction or EvictionPolicy()
        self.entries = {}
        self._logical_entry_sum = 0
        self._charged_entry_sum = 0
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
        # C3-only accounting hook. Logical size continues to drive admission and
        # eviction scores; the default keeps the original byte-capacity behavior.
        if shared_overhead_bytes < 0 or shared_overhead_bytes > capacity_bytes:
            raise ValueError('Invalid shared cache overhead')
        self.capacity_charge = capacity_charge
        self.shared_overhead_bytes = shared_overhead_bytes
        if homogeneous_logical_fastpath and (capacity_charge is None or self.eviction.gamma <= 0):
            raise ValueError('Homogeneous fast path requires byte charges and age-sensitive eviction')
        self.homogeneous_logical_fastpath = homogeneous_logical_fastpath
        self._uniform_entry_size = None
        self._eviction_heap = []
        self._entry_serials = {}
        self._next_serial = 0

    def advance(self, query_index):
        if query_index < self.now:
            raise ValueError("Query clock cannot go backwards")
        self.now = query_index

    @property
    def logical_cache_bytes(self):
        return self._logical_entry_sum

    def charged_entry_bytes(self, entry):
        value = entry.size_bytes if self.capacity_charge is None else self.capacity_charge(entry)
        if type(value) is not int or value <= 0:
            raise ValueError('Capacity charge must be a positive integer')
        return value

    @property
    def charged_cache_bytes(self):
        return self.shared_overhead_bytes + self._charged_entry_sum

    def _eviction_victim(self, metrics):
        """C3 hook: reuse the exact paper score with one shared normalization pass."""
        values = tuple(metrics.values())
        names = ('frequency', 'impact', 'age', 'size')
        if any(not math.isfinite(getattr(m, name)) or getattr(m, name) < 0
               for m in values for name in names):
            raise ValueError('Metrics must be finite and nonnegative')
        maxima = {name: max(getattr(m, name) for m in values) for name in names}
        def score(key):
            m = metrics[key]
            scaled = CacheMetrics(*(getattr(m, name)/maxima[name] if maxima[name] else self.zero_max
                                    for name in names))
            return self.eviction.score(scaled)
        victim = min(metrics, key=lambda key: (-score(key), key))
        return victim, score(victim)

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
        if self.homogeneous_logical_fastpath:
            raise ValueError('Homogeneous no-reuse replay cannot record physical reuse')
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
        if entry.key in self.entries or self.charged_entry_bytes(entry) + self.shared_overhead_bytes > self.capacity_bytes:
            return False
        if self.homogeneous_logical_fastpath:
            # M9-B has one raw logical entry size, no measured impact, and no
            # executed reuse. Candidate frequency is positive while every
            # resident F remains zero, so normalize(candidate)=(1,0,0,1).
            if (observed_frequency <= 0 or entry.frequency != 0 or entry.impact not in (None, 0)
                    or (self._uniform_entry_size is not None and entry.size_bytes != self._uniform_entry_size)):
                raise ValueError('Homogeneous replay invariant violated')
            admitted = self.admission.admit(CacheMetrics(1, 0, 0, 1))
        else:
            admitted = self.admission.admit(self.admission_metrics(entry, observed_frequency, admission_frequencies))
        if not admitted:
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
        self._logical_entry_sum += entry.size_bytes
        self._charged_entry_sum += self.charged_entry_bytes(entry)
        if self.homogeneous_logical_fastpath:
            if self._uniform_entry_size is None:
                self._uniform_entry_size = entry.size_bytes
            self._next_serial += 1
            self._entry_serials[entry.key] = self._next_serial
            heapq.heappush(self._eviction_heap, (entry.last_access, entry.key, self._next_serial))
        if on_event:
            on_event('INSERT', entry, None)
        while self.charged_cache_bytes > self.capacity_bytes:
            if self.homogeneous_logical_fastpath:
                while True:
                    last_access, candidate_key, serial = heapq.heappop(self._eviction_heap)
                    if self._entry_serials.get(candidate_key) == serial:
                        break
                victim = candidate_key
                # Oldest age has normalized age 1 (or 0 if all inserted now).
                score = self.eviction.score(CacheMetrics(0, 0,
                    1 if self.now > last_access else 0, 1))
            else:
                metrics = {k: self._metrics(e) for k, e in self.entries.items()}
                # Largest score; stable lexicographic key breaks ties.
                if self.capacity_charge is None:
                    victim = min(metrics, key=lambda k: (-self.eviction.score(normalize(metrics[k], metrics.values(), self.zero_max)), k))
                    score = self.eviction.score(normalize(metrics[victim], metrics.values(), self.zero_max))
                else:
                    victim, score = self._eviction_victim(metrics)
            if on_event:
                on_event('EVICT', self.entries[victim], score)
            self._logical_entry_sum -= self.entries[victim].size_bytes
            self._charged_entry_sum -= self.charged_entry_bytes(self.entries[victim])
            del self.entries[victim]
            if self.homogeneous_logical_fastpath:
                del self._entry_serials[victim]
        # Resident occupancy after eviction; excludes temporary insertion overflow.
        self.peak_logical_cache_bytes = max(self.peak_logical_cache_bytes, self.logical_cache_bytes)
        return entry.key in self.entries

    def assert_homogeneous_replay(self):
        """Audit the algebraic C3 shortcut after M9-B's periodic PBR update."""
        if not self.homogeneous_logical_fastpath:
            raise ValueError('Homogeneous replay is not enabled')
        if any(e.size_bytes != self._uniform_entry_size or e.frequency != 0 or e.impact not in (None, 0)
               for e in self.entries.values()):
            raise ValueError('M9-B logical state invalidates homogeneous fast path')
