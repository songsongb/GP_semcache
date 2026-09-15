"""M6B-1 logical policies; unspecified FBC details are reproduction choices."""
from collections import Counter, OrderedDict
from types import SimpleNamespace
from .runner import Baseline as BaselineKind, WhitespaceTokenizer
from semcache.semantic.subsequence import SubsequenceExtractor
from semcache.semantic.hit_selection import CacheHit, select_nonoverlapping

ALL_BASELINES = (BaselineKind.UD_ONLY, BaselineKind.ES_ONLY, BaselineKind.FBC_V1,
                 BaselineKind.FBC_V2, BaselineKind.SEMCACHE)
FBC_KINDS = (BaselineKind.FBC, BaselineKind.FBC_V1, BaselineKind.FBC_V2)

FBC_METADATA = dict(fbc_variant='exact_block_frequency_lru_v1', semantic_awareness=False,
    admission_rule='first eligible observation; frequency >= 1; unique misses admitted after all query lookups',
    eviction_rule='lru', key_definition='exact token tuple; no user, position, or semantic cluster; cache isolated per run/model',
    frequency_rule='lifetime count of every window observation, including misses and overlapping windows; retained after eviction',
    recency_rule='every resident lookup in window order, then insertion order',
    frequency_threshold=1, frequency_tracked_but_not_policy_driving=True,
    provenance='REPRODUCTION_CHOICE')

FBC_V2_METADATA = dict(FBC_METADATA,
    fbc_variant='exact_block_frequency2_lru_v2',
    admission_rule='global exact-block observation count >= 2', eviction_rule='LRU',
    frequency_threshold=2, frequency_tracked_but_not_policy_driving=False,
    readmission_rule='on a later miss, count >= 2 permits immediate re-admission after query lookups',
    lookup_admission_order='all query lookups before admissions; second observation is a miss; first hit in a later query',
    repeated_windows='every occurrence counts; repeated blocks in one query can reach threshold; one candidate per unique miss')


class FrequencyLRUCache:
    def __init__(self, capacity_bytes, window_size, block_bytes, tokenizer=None, *, frequency_threshold=1):
        if frequency_threshold not in (1, 2) or isinstance(frequency_threshold, bool):
            raise ValueError("Supported frequency thresholds are 1 and 2")
        self.frequency_threshold = frequency_threshold
        self.capacity_bytes = capacity_bytes
        self.block_bytes = block_bytes
        self.extractor = SubsequenceExtractor(window_size)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.entries = OrderedDict()
        self.frequencies = Counter()
        self.peak_bytes = 0

    def query(self, text):
        ids = list(self.tokenizer(text)['input_ids'])
        if not ids:
            raise ValueError('Empty tokenized query')
        windows = self.extractor.extract(ids)
        hits, misses = [], []
        for w in windows:
            key = w.token_ids
            self.frequencies[key] += 1
            if key in self.entries:
                self.entries.move_to_end(key)
                hits.append(CacheHit(w, self.entries[key]))
            else:
                misses.append(w)
        _, mask = select_nonoverlapping(hits, len(ids))
        candidates = admissions = evictions = 0
        seen = set()
        for w in misses:
            key = w.token_ids
            if key in seen or key in self.entries:
                continue
            seen.add(key)
            candidates += 1
            if self.frequencies[key] < self.frequency_threshold or self.block_bytes > self.capacity_bytes:
                continue
            while (len(self.entries)+1)*self.block_bytes > self.capacity_bytes:
                self.entries.popitem(last=False)
                evictions += 1
            self.entries[key] = SimpleNamespace(key=key, token_ids=key)
            admissions += 1
            self.peak_bytes = max(self.peak_bytes, len(self.entries)*self.block_bytes)
        return dict(query_token_count=len(ids), block_lookup_count=len(windows), block_hit_count=len(hits),
            reused_token_count=sum(mask), admission_candidate_count=candidates, admission_count=admissions,
            eviction_count=evictions, logical_global_cache_bytes=len(self.entries)*self.block_bytes)
