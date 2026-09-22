# M8.5 prefill paper-alignment cleanup

**Closed — user-reported SERAPH regression:** alignment 11/11 passed; focused
M8/M8.5 36 passed; full suite 225 passed, 9 skipped, 1 warning. These supersede
local skipped-test limitations below without claiming a new local execution.

This document supersedes M7/M8 descriptions of reducer defaults, CHU gating and
CLI schedules. Scope remains **prefill only**. No latency optimization, new model
support, model execution or downloads are part of this cleanup. Existing guarded
OPT-6.7B code predates M8.5 and is neither extended nor exercised here.

## A. Paper-defined behavior

Primary source: the repository-root INFOCOM 2026 SemCache PDF, §IV-A/B/C,
Algorithm 1 and Table I.

- Eq. 8 nearest-L2 cluster assignment and Eq. 9 incremental centroid mean.
- Table I cluster update interval: 100 queries; CHU rho: 0.8; PBR Lambda: 100.
- Semantic impact: `I(c,p) = sum_l sum_{t in S_c} ||A_l[t]||_2`.
- CHU on reused blocks; PBR conditional mean over recent queries containing the
  subsequence. Paper PBR is scheduled during low query load.
- Admission uses query appearance frequency; eviction uses block reuse frequency.
- Cluster-local Q/K/V blocks, matched non-overlapping spans and fresh computation
  for unmatched tokens. Neither attention nor FFN computation is eliminated.

## LOOKUP_NO_REUSE bug and fix

Before M8.5, physical execution received `execution_hits=[]`, but CHU, FETCH and
reuse provenance iterated over `selected`. `metrics.reused()` consequently
incremented frequency, reset last access and changed impact even without reuse.

The physical engine now derives CHU, fetch/provenance events, executed span count,
reused token count, analytical saved work/communication and the quality diagnostic
suffix from **execution_hits**. `projection_skip_used` remains an executor audit
observation. Candidate selection, lookup HIT counts and logical reusable coverage
remain explicitly separate diagnostics, not physical reuse measurements.

| Quantity | Source / allowed changes in lookup-only mode |
|---|---|
| Admission F | Every window occurrence in the latest `frequency_window` queries, including misses and repeated occurrences in one query; advances normally |
| Eviction F | Executed physical reuse occurrences only; unchanged by candidate lookup |
| Last access | Initialized on insertion, subsequently reset only by physical reuse; existing entries unchanged by candidate lookup |
| Age | Query clock minus last access; advances even without reuse |
| CHU impact update | Executed physical reuse only; absent in lookup-only mode |
| Lookup hits/misses | Dictionary lookup observations; still counted |
| Admission / eviction | May run on misses; lookup-only mode does not freeze the cache |
| Query history / PBR | Uses query attention observations, not reuse; may independently change impact and `updated_at`, but never reuse F or last access |

PBR can legitimately change impact in lookup-only mode. The no-contamination
invariant applies to **reuse-derived** changes, not to all policy state. Admission
initialization and PBR must not be misreported as CHU. Logical simulation retains
simulated reuse counters and is labelled separately from physical execution.

## Semantic impact modes

`paper_row_l2_sum` is the engine and M7/M8 CLI default. Given eager prefill
attention `[1, H, N, N]`, its exact implementation is:

```
I = sum_l sum_{t=start}^{end-1} (1/H_l) sum_h
        sqrt(sum_{k valid, k<=t} A[l,h,t,k]^2)
```

Padded query rows contribute zero; padded/future key cells are excluded. It sums
rows and layers, without averaging either. With one head this is the paper's
row-sum formula. The paper omits a head aggregation rule: **mean of per-head row
norms** is our explicit REPRODUCTION_CHOICE, not a paper-specified operation.

The paper also writes attention using block Q/K. We interpret CHU's current-query
attention as the actual full-prefill attention rows, retaining all valid causal
keys, rather than recomputing a block-local softmax. This interpretation is
explicit in metadata; the implementation does not claim to resolve that ambiguity.
Indices are zero-based, end-exclusive, and decode-shaped nonsquare inputs fail.

`reproduction_frobenius_mean` preserves the prior implementation:

```
I = mean_{l,h} sqrt(sum_{valid q,k; k in [start,end)} A[l,h,q,k]^2)
```

It selects block **key columns**, rather than block query rows. The prior class
`MeanLayerHeadFrobeniusReducer` remains available; only its public mode name is
standardized. Historical artifacts named `mean_layer_head_frobenius_v1` describe
that older mode; their provenance is not rewritten. Configs use the new names.

Reducer type, full interpretation metadata and schedule settings accompany engine
summaries/events, M7 JSON, M8 raw JSONL, summary CSV, reuse-delta CSV, environment
JSON and prompt-audit JSON. Native rows and prompt audits record the configured
reducer but mark it unapplied. Aggregation separates profiles instead of pooling
different reducers/schedules; old records without provenance remain unknown.
Training-only artifacts do not execute a reducer and are outside this inference
profile.

## Update schedules

M7/M8 CLI defaults are now:

```
--impact-reducer paper_row_l2_sum
--cluster-update-mode buffered --cluster-update-interval 100
--rho 0.8 --history-lambda 100
--pbr-mode interval --pbr-interval 100
```

Assignment observes current centroids. The buffered mode retains assignments and
embeddings, and applies Eq. 9 sequentially every 100 global queries. This honors
the Table I interval; retaining assignments until flush is a REPRODUCTION_CHOICE
because the paper does not detail batching. Eq. 9 alone does not specify the
Table I scheduling mechanism. Buffered boundary diagnostics now correctly report
that a centroid update was applied, even when its numerical shift is zero.

Automatic PBR runs after admission every configured number of global requests,
visiting all resident clusters with each cluster's most recent Lambda observations.
The interval value, global clock, all-cluster sweep and synchronous placement are
**REPRODUCTION_CHOICE**, not a low-load detector. A query with no matching history
leaves an entry's impact unchanged. `--pbr-mode manual` disables the automatic
trigger; `--cluster-update-mode immediate_eq9` preserves the earlier online mode.

The short M8 trace reconstructs state every repetition and has only four requests.
Consequently the default interval of 100 will not fire in that trace; warmup or
repetition counts do not accumulate toward it. To exercise boundaries in a small
regression, override both intervals to 2 and label that run a reproduction choice.

The low-level `IntentClusterer` retains its historical immediate default for
existing M1–M6 callers; the physical engine retains manual PBR unless explicitly
configured. M7/M8 entry points select the new profile explicitly. Legacy base and
development YAML schedules remain explicit immediate/manual choices; their
reducer default is updated. `_semcache_common` now honors the configured reducer
and accepts buffered/interval choices. These distinctions appear in runtime
metadata rather than silently changing historical analytical experiments.

## Cache addressing audit and design

**Paper specification:** §IV-B indexes a block as `(c,p)`, where `p` is its token
range **in cluster c's cache pool**. The paper describes a contiguous `[Q K V]`
payload and matches input subsequences to resident ranges. This does not establish
that `p` must equal an absolute token offset in every source/destination query.

**Underspecified:** how content maps to pool ranges, range allocation/reclamation,
multiple occurrences of identical content, matching equality, selection between
several source contexts/adapters, and numerical eligibility of cross-user reuse.

**Current choice, preserved:** dictionary key `(cluster_id, exact token tuple)`;
source `[start,end)` and user/adapter/query provenance are metadata. The first
admitted occurrence owns the payload. Repeated keys within a query are deduplicated;
a resident key is not overwritten. Matching does not require equal absolute
position or equal user. Payloads remain per-layer Q/K/V tensors, not a new packed
pool allocation.

**Occurrence ambiguity:** identical triples at different positions map to one key,
but positional encodings and preceding context can produce different Q/K/V. A
same-user repeat of an identical long prompt can reuse the first occurrence's
payload at later occurrences. Content equality is not numerical equivalence.

`semantic/addressing.py` introduces design-only value objects:

- `PoolBlockAddress(cluster_id, start, end)`: cluster-pool token range.
- `SourceOccurrence(query_id, user_id, adapter_id, start, end)`: source identity.
- `BlockDescriptor(address, token_ids, source)`: separates match key from storage
  and occurrence identity. Multiple descriptors may share one match key.

Future integration would use a content-to-descriptor catalog and an explicit
correctness policy to choose a resident address. No allocator, catalog lookup or
physical reuse migration is implemented. Unit tests establish type/range invariants
and demonstrate repeated-key ambiguity; they do **not** validate a new physical
reuse policy. Migration additionally requires position/context/adapter tests,
logit comparisons, eviction/reclamation tests and an explicit approximation policy.

## B. Remaining reproduction choices

Head aggregation and full-row attention interpretation; TinyBERT checkpoint,
text retokenization instead of the Eq. 7 embedding bridge, pooling and truncation;
first-k centroid initialization; buffered assignment handling; automatic PBR
schedule; trailing admission frequency horizon; per-query averaging of repeated
key impacts; exact token matching, greedy overlap selection and normalization
population; CPU cache placement; controlled untrained adapters and small traces.

## C. Measured implementation behavior and validation

The regression suite uses the actual engine control flow with a stubbed forward
and projection executor. It checks lookup-only isolation, a positive physical-hit
control, query-based PBR, multi-cluster automatic scheduling, 100-query cluster
boundaries, address identity and artifact profile separation. These are software
control-flow observations, not measured LLM latency or output quality.

Optional tensor-only tests use literal attention arrays to verify the exact
formula, head/layer aggregation, query-row versus key-column selection, masking
and invalid input rejection. No test in `test_m85_alignment.py` instantiates or
executes a model. Local validation: **9 passed, 2 tensor-only tests skipped** because PyTorch is
absent. Syntax compilation and `git diff --check` passed. No models were run or
downloaded; no tensor-formula execution is claimed for this environment.

Recommended **small SERAPH regression only**, using the existing Python environment:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_m85_alignment.py -v
```

This runs no models, makes no downloads and needs no GPU. On SERAPH with PyTorch,
the literal-tensor checks should run rather than skip. No latency benchmark or
large-model run is recommended as part of this cleanup.

## D. Still unimplemented / unvalidated full-system scope

- Full autoregressive decode to EOS, decode cache population and past-KV integration.
- Actual UD–ES distributed execution, network transfers and execution overlap.
- Trained user-specific LoRA inference in the prefill evaluation pipeline.
- Paper-scale dataset/user/cache experiments and end-to-end answer-quality reproduction.

`safe_reuse_claimed=false` remains. Projection skipping is distinct from numerical
reuse safety, BLEU preservation and total-system speedup.

## Files changed

| Area | Files |
|---|---|
| Prefill orchestration / policy | `src/semcache/semcache_engine.py`, `src/semcache/cache/global_cache.py`, `src/semcache/cache/metric_manager.py` |
| Reducers / clustering / address design | `src/semcache/cache/attention_impact.py`, `src/semcache/semantic/intent_clusterer.py`, `src/semcache/semantic/addressing.py` (new) |
| Configuration / artifacts | `src/semcache/metrics/alignment.py` (new), `src/semcache/metrics/inference.py`, `src/semcache/metrics/m8.py`, `configs/base.yaml`, `configs/development.yaml` |
| Entry points | `scripts/29_run_m7_semcache_integration.py`, `scripts/30_run_m8_inference_timing.py`, `scripts/_semcache_common.py` |
| Tests | `tests/test_m85_alignment.py` (new) |
| Documentation | `README.md`, `docs/M7_FULL_SEMANTIC_SEMCACHE.md`, `docs/M8_LATENCY_MEASUREMENT.md`, `docs/M85_PAPER_ALIGNMENT.md` (new) |
