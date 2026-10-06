# C7-B3 manual Q24 freeze and C8-A capacity replay

Both entry points use only the Python standard library. They read authoritative
artifacts, validate hashes, and write new outputs. They do not load models, execute
codecs, fit profiles, evaluate new quality, or benchmark latency.

C7-B3 records the explicit manual decision **SELECT_Q24**, freezing Q24/K20/V16.
It validates the pinned B1/B2 manifest SHA256 values, the exact Q24 and K/V profile
hashes, the Q transform, zero fitting counters, and the shared model/adapter
namespace. It checks saved B1 rate evidence, B2 corpus-score deltas and all 32
paired generations, and B2A's four-case selection, per-layer distortion/injection,
zero-Q internal effect, fresh-attention isolation, and continuation isolation.
The B2A manifest hash is computed from the actual file. Its bound human summary
must agree with the structured evidence. A contradictory or missing artifact is
a hard failure. No existing freeze decision is overwritten.

`freeze_decision.json` includes the manual rationale, Q/K/V bins, transform,
profile paths/hashes, all three manifest paths/hashes, quality and causal evidence,
input hashes, Git provenance, and no-further-fitting flags. It explicitly records
`frozen32_used_for_selection=true`; no claim of an untouched final evaluation set
is made. Cached TOTAL Q/K/V and the exact-w3-within-cluster contract remain intact.

C8-A requires and revalidates that freeze. Its only policies are:

| Policy | Physical resident representation |
|---|---|
| RAW_QKV | FP16 Q/K/V |
| KV_COMP | FP16 Q + frozen K20/V16 frame |
| Q24_KV_COMP | frozen Q24 frame + the same frozen K20/V16 frame |

Each raw role is 491,520 bytes; a raw entry is 1,474,560 bytes. RAW sizes are checked
against the B2 per-case raw counts. Compressed sizes come from each canonical B2
`per_case.csv` row's measured storage accounting, with complete episode identity
checks and aggregate validation against `storage_accounting.json`. Sizes are never
replaced with an average. Local metadata is already included in the frame lengths;
it is not added twice. Shared profile bytes are reported separately and do not
consume the resident-entry budget.
This is a metadata replay of measured sizes: it does not allocate Q/K/V tensors
or decode frames. The budget follows C6 tensor/frame accounting and excludes
logical-index, Python-object, and allocator overhead.

| Budget | Raw-entry equivalents | Identical bytes for every policy |
|---|---:|---:|
| B2 | 2 | 2,949,120 |
| B4 | 4 | 5,898,240 |
| B8 | 8 | 11,796,480 |
| B16 | 16 | 23,592,960 |

The protocol is **CONTROLLED_TWO_PHASE_CAPACITY_REPLAY**. All 32 source admission
attempts run in frozen32 order, followed by all 32 target lookups in that same
order. No target admission occurs. This is a controlled capacity-pressure trace,
not a natural arrival trace.

An experiment-only BYTE_AWARE_LRU wrapper evicts oldest entries until the incoming
entry fits. A single oversized entry is rejected without disturbing residents.
A successful lookup refreshes recency. Consistent with `GlobalCache.insert`, a
duplicate **currently resident** key is rejected: the existing payload, size, and
recency stay unchanged. If that key has already been evicted, admission can succeed
again using the incoming source. Thus `duplicate_key_updates=0`; duplicate
rejections have an additional explicit counter.

The key is `(semantic cluster, exact three token IDs)`. Hits depend only on key
residency. A surviving representative may originate from another selected source
with the same key; every lookup records its resident source episode ID. The
`retained_selected_source_count` counts available selected logical source keys per
episode (including repeats); `retained_unique_key_count` counts physical resident
entries. This distinction prevents duplicate requests from inflating entry counts.

Reports contain admissions, duplicates, oversize rejections, eviction counts and
bytes, peak/final resident bytes and entries, hits/misses and rates, exact eviction
victims, and LRU-order snapshots. Causal comparisons report KV minus RAW, Q24 minus
KV, and Q24 minus RAW at every budget. Storage savings alone do not establish a
cache-behavior benefit; the summary identifies budgets with measured additional
retention, fewer evictions, or more exact-w3 hits.

At most two budgets are suggested for a later C8-B review: the smallest budget at
which policies differ, and the largest larger budget with additional Q24-versus-KV
capacity/hit benefit, when one exists. This deterministic rule reads no model
quality data. C8-B is not launched; C9 is not implemented.

Run from `/data/khuss/repos/GP_semcache` on SERAPH, without a GPU allocation:

```bash
CUDA_VISIBLE_DEVICES="" python -u scripts/64_freeze_cachegen_c7_q24.py \
  --q-calibration-root results/cachegen/c7/b_q_calibration_retry2 \
  --c7b2-root results/cachegen/c7/b2_q_quality_gate_frozen32 \
  --c7b2a-root results/cachegen/c7/b2a_q_causal_audit_retry1 \
  --kv-profile results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin \
  --output-root results/cachegen/c7/q24_freeze

CUDA_VISIBLE_DEVICES="" python -u scripts/65_run_cachegen_c8a_capacity_replay.py \
  --freeze-decision results/cachegen/c7/q24_freeze/freeze_decision.json \
  --plan-dir results/cachegen/c6b3/multiwoz_plan_v2 \
  --c7b2-root results/cachegen/c7/b2_q_quality_gate_frozen32 \
  --output-root results/cachegen/c8/a_capacity_replay
```

The freeze writes `freeze_decision.json`. C8-A requires a new output root and writes
`summary.csv`, `summary.json`, `causal_comparisons.json`, `per_event.csv`,
`residency_trace.json`, `manifest.json`, and `summary.md`. Its input/output hashes
and Git provenance allow the byte replay to be reproduced without model inference.
