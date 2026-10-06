# C8-B bounded fixed-byte quality replay

C8-B executes only B2 (2,949,120 bytes) and B8 (11,796,480 bytes), with
RAW_QKV, KV_COMP, and Q24_KV_COMP. The reference is canonical FULL_RECOMPUTE.
The label is **SELECTION_BASED_CONTROLLED_QUALITY_REPLAY**: frozen32 already
informed Q-profile selection. These results provide bounded mechanistic evidence,
not an unbiased final generalization test. An untouched cohort would be needed
for that claim.

Before loading a model, the entry point revalidates the C7-B3 SELECT_Q24 freeze,
its pinned Q24/K20/V16 profiles and upstream evidence, the canonical B3 plan and
adapter decision, and every required C8-A output hash. It reconstructs A's
admission/lookup transitions from the measured variable per-entry bytes and
requires identical summaries, eviction events, retained source identities,
LRU-order snapshots, and HIT vectors. The saved recommendations must be [2,8].
Expected counts are checked against the actual artifacts: at B2, 2/4/13 hits;
at B8, 8/18/32 hits. They do not substitute for the recorded per-target vectors.
No logical selection or semantic encoder is rerun.

The canonical FULL reference is imported from its hash-bound capability artifacts
through C6-B3-2's existing validation. Its manifest, summary, and generation CSV
must agree with the continuation provenance saved in C7-B2. Installed SacreBLEU
must reproduce the canonical score/signature. The reference's greedy generation
is not rerun. One native FULL teacher forward per case supplies reference logits
on the same canonical generated continuation used in C6-B3/C7-B2.

There are 32 source forwards, one per admission event. Each captures source TOTAL
Q/K/V once; selected [1,3,2560] rows for all 32 layers are copied to CPU. The three
representations are built from those same rows. Frozen K/V encoding is unchanged;
its frames must match between compressed policies. Encoded entry objects are
shared between the two budget constructions. Raw working copies are temporary,
not counted as compressed resident cache storage, and discarded after construction
unless a RAW cache retains them. Resident Q24 has no raw Q/K/V tensors; KV_COMP
has raw Q only. Decoding constructs a temporary lookup view, never replaces or
expands the resident representation. Shared profiles are not charged per entry.

An experiment-only cache owns the physical payloads while using the exact C8-A
BYTE_AWARE_LRU transition code. All source admissions precede target evaluation;
targets are never admitted. Resident duplicates are rejected without changing
the existing payload/size/recency; an evicted key can be readmitted. Successful
lookups touch LRU. Every HIT uses the actual retained source payload, including
an earlier source representative of a duplicate key. Target cluster, exact token
IDs, and target hit positions remain unchanged. The existing mixed projection
path must reuse three rows and skip native computation for Q, K, and V at every
layer, in both greedy prefill and teacher forwarding. MISS uses the normal native
FULL path. No transport or runtime profile fitting is permitted.

All six conditions independently generate all 32 targets greedily with the
unchanged 160-token/EOS protocol. Task quality uses corpus SacreBLEU 13a/exp,
case-sensitive, one reference, effective_order=false, range0–100, explicitly a
REPRODUCTION_CHOICE. Per-case output text/tokens, exact FULL match, normalized edit
distance, and token-position agreement are saved. Teacher metrics include max/mean
absolute logit difference, KL, cosine, top1/top5 agreement, and output-margin delta.
All/HIT/MISS subsets have separate aggregation and subset corpus BLEU; an empty
subset is represented by null metrics, not zero quality. MISS text/token equality
is strict; teacher numerical equivalence requires max_abs<=1e-6, mean_abs<=1e-7,
and identical top1/top5. These are predeclared numerical checks, not task-quality
acceptance thresholds.

Pairwise comparisons at each budget report KV minus RAW, Q24 minus KV, and Q24
minus RAW for coverage and quality. Codec perturbation and capacity-induced HIT
coverage are distinct mechanisms. A positive BLEU change is not evidence that
compression improves model quality.

The descriptive review rule is declared in code and the manifest before execution.
Exact A reproduction and faithful MISS paths are mandatory. At either budget,
Q24 requires further review when lower BLEU **and** (lower FULL-generation match
rate **or** higher edit distance) coincide with higher teacher KL **and** lower top1
agreement versus KV_COMP. Otherwise the stage records descriptive support for
carrying the resident representation into C9 review. Every raw metric remains
available; no weighted score or paper-quality threshold is claimed. No C9 system
policy is frozen, no latency is measured, and no later stage is launched.

Run from `/data/khuss/repos/GP_semcache` with one GPU allocated. Models and all
artifacts must already be present; loading uses local_files_only and cache paths
under `/data/khuss`. This script creates no shell or Slurm files.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u scripts/66_run_cachegen_c8b_quality_b2_b8.py \
  --adapter-root results/cachegen/c6b3/b1_full \
  --adapter-freeze-decision results/cachegen/c6b3/b1_full_freeze_decision.json \
  --plan-dir results/cachegen/c6b3/multiwoz_plan_v2 \
  --source results/workloads/multiwoz.jsonl \
  --semantic results/workloads/c6b3_multiwoz_history_semantic.jsonl \
  --freeze-decision results/cachegen/c7/q24_freeze/freeze_decision.json \
  --c8a-root results/cachegen/c8/a_capacity_replay \
  --kv-profile results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin \
  --canonical-full-root results/cachegen/c6b3/b1_full_frozen32 \
  --storage-src /data/khuss/repos/GP_semcache/.c6_storage_src/src \
  --device cuda:0 --seed 42 \
  --output-root results/cachegen/c8/b_quality_b2_b8
```

A new output root is required. Outputs: summary.csv/json, per_case.csv,
cache_build_audit.json, hit_audit.json, paired_quality.json, causal_comparisons.json,
manifest.json, summary.md. Partial execution failures remain INVALID and retain
the root failure and available evidence. There are 192 independent target greedy
generations, 224 teacher forwards, and 32 source captures. Wall time depends on
the allocated GPU and generated lengths; no measured runtime estimate exists.
