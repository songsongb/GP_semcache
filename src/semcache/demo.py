"""Controlled index demonstration; never performs or claims QKV reuse."""
from collections import Counter, deque
from .semantic.encoder import ControlledEncoder
from .semantic.intent_clusterer import IntentClusterer
from .semantic.subsequence import SubsequenceExtractor
from .semantic.matcher import make_matcher
from .cache.cache_entry import CacheEntry
from .cache.global_cache import GlobalCache
from .cache.admission import AdmissionPolicy
from .cache.eviction import EvictionPolicy
from .simulation.memory_model import qkv_logical_bytes
from .evaluation.metrics import empty_row
from .utils.seed import seed_everything


def run_demo(config):
    seed_everything(config["seed"])
    d = config["demo"]
    if d["tokenizer"] != "controlled_whitespace_v1" or d["encoder"] != "controlled_vectors_v1":
        raise ValueError("Demo requires explicitly selected controlled fixtures")
    if config["cache"]["normalization"] != "current_pool_plus_candidate" or config["cache"]["age_unit"] != "query":
        raise ValueError("Unsupported cache policy")
    queries = d["queries"]
    if len(queries) != len(d["embeddings"]):
        raise ValueError("Need one controlled embedding per query")
    vocab = {word: i for i, word in enumerate(sorted({w for q in queries for w in q.split()}))}
    encoder = ControlledEncoder(dict(zip(queries, d["embeddings"])))
    clusterer = IntentClusterer(config["intent_clusters"][config["dataset"]], config["cluster_update_interval_queries"], config["clustering"]["initialization"])
    if config["clustering"]["update_rule"] != "batched_incremental_mean":
        raise ValueError("Unsupported centroid update policy")
    clusterer.initialize(d["centroids"], counts=[0]*clusterer.num_clusters)
    extractor = SubsequenceExtractor(config["subsequence_window"])
    matcher = make_matcher(config["match_rule"])
    cache = GlobalCache(int(config["logical_cache_capacity_gb"]*10**9), AdmissionPolicy(**config["admission"]), EvictionPolicy(**config["eviction"]), config["cache"]["zero_max"])
    history = deque(maxlen=config["cache"]["admission_frequency_window_queries"])
    events = []
    insertions = []
    total_tokens = matched_tokens = 0
    for index, (query, vector) in enumerate(zip(queries, encoder.encode(queries)), 1):
        cache.advance(index)
        c = clusterer.observe(vector)
        ids = [vocab[w] for w in query.split()]
        windows = extractor.extract(ids)
        total_tokens += len(ids)
        history.append(Counter(matcher.key(c, s) for s in windows))
        frequencies = sum(history, Counter())
        pending = []
        covered = set()
        # All lookups happen before insertion: no artificial same-request hits.
        for s in windows:
            key = matcher.key(c, s)
            hit = cache.lookup(key)
            if hit is not None:
                covered.update(range(s.start, s.end))
            else:
                pending.append((s, key))
            events.append({"request": index, "query": query, "cluster_id": c,
                           "token_ids": list(s.token_ids), "positions": [s.start, s.end],
                           "status": "HIT" if hit is not None else "MISS", "metric_source": "measured"})
        matched_tokens += len(covered)
        for s, key in pending:
            size = qkv_logical_bytes(len(s.token_ids), **d["projection_metadata"])
            admitted = cache.insert(CacheEntry(c, s.token_ids, (s.start, s.end), size,
                         qkv_metadata={**d["projection_metadata"], "scope": "unmaterialized_qkv", "numerically_verified": False}),
                         frequencies[key], frequencies)
            insertions.append({"request": index, "token_ids": list(s.token_ids), "admitted": admitted})
    row = empty_row()
    row.update(run_id=f"controlled-seed-{config['seed']}", seed=config["seed"],
        model=config["model"]["name"], model_revision=config["model"]["revision"],
        tokenizer=d["tokenizer"], dtype=config["model"]["dtype"], dataset="controlled_manual_queries",
        method="index_only", num_users=config["num_users"], lora_rank=config["lora_rank"],
        cluster_count=clusterer.num_clusters, window_size=extractor.window_size,
        semantic_encoder=encoder.metadata["checkpoint"], match_rule=matcher.match_rule,
        cache_size_gb=config["logical_cache_capacity_gb"], theta_admit=config["admission"]["threshold"],
        rho=config["semantic_impact"]["rho"], pbr_interval=config["semantic_impact"]["pbr_interval_queries"],
        bandwidth_mbps=config["default_simulated_bandwidth_mbps"], num_requests=len(queries),
        total_tokens=total_tokens, cache_hits=cache.hits, cache_misses=cache.misses,
        hit_rate=cache.hits/(cache.hits+cache.misses) if cache.hits+cache.misses else 0,
        reused_tokens=0, reuse_ratio=0, logical_memory_gb=cache.logical_cache_bytes/10**9)
    row["metric_source"] = {name: "measured" for name in ("num_requests", "total_tokens", "cache_hits", "cache_misses", "hit_rate", "reused_tokens", "reuse_ratio")}
    row["metric_source"]["logical_memory_gb"] = "simulated"
    return {"summary": row, "events": events, "insertions": insertions, "logical_cache_bytes": cache.logical_cache_bytes,
            "physical_tensor_bytes": cache.physical_tensor_bytes, "matched_token_positions": matched_tokens,
            "metric_source": {"logical_cache_bytes": "simulated", "physical_tensor_bytes": "measured", "matched_token_positions": "measured"},
            "metadata": {"model_loaded": False, "configured_target_dataset": config["dataset"],
                         "semantic_encoder": encoder.metadata, "qkv_reuse_verified": False,
                         "impact_available": False, "config": config}}
