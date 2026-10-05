# C7-B2 frozen32 resident-Q quality gate

This stage independently executes exactly three storage modes:
`STORAGE_KV_COMP_BASELINE`, `STORAGE_Q24_KV_COMP`, and `STORAGE_Q32_KV_COMP`.
The exact w=3 token subsequence within its semantic cluster remains the HIT
unit. Cached TOTAL Q/K/V remain the shared payload and use the same reuse mask.

The baseline already contains the frozen K20/V16 compression. Candidate-minus-
baseline comparisons therefore isolate resident-Q compression. No transport
compression, profile fitting, adapter training, latency benchmark, capacity
experiment, reselection, or automatic profile freeze occurs.

## Physical path

The experiment calls the unchanged C6 `Storage.encode` and C2 K/V decoder in
all modes. The baseline retains raw FP16 Q. Q24/Q32 use the already-fitted B1
Q codec, storing its frame and clearing raw Q **before cache admission**. An
experiment-local entry subclass accounts for Q frame bytes without changing
the raw logical size used by admission/eviction.

The existing `GlobalCache.lookup` calls the injected experiment codec. It
reconstructs Q into a temporary input for the unchanged C2 decoder, then returns
a temporary TOTAL-Q/K/V view to the existing mixed projection path. No decoded
Q is stored on the resident entry. Runtime assertions verify every layer's
three reused rows separately for Q/K/V and native computation on fresh rows
only. K/V local metadata is already inside the K/V frame; it is not charged
twice. Shared Q and K/V profile sizes are reported separately from entries.

## Provenance and reproduction

All user-specified model, adapter, freeze-decision, workload, selection,
calibration-capture, cohort and Q/K/V profile hashes are pinned in the new
quality module. The B1 manifest must be complete with the 96/32 calibration
split, zero overlap/fitting counts, role-generic Q transform and no frozen
candidate. Profile metadata must name the original fit block IDs and digest.
Existing C6-B3-2 provenance helpers verify the frozen selection, adapter freeze
chain and canonical FULL_RECOMPUTE continuation. Input hashes are checked again
after execution. Existing artifacts are read-only.

Every mode generates greedily and independently using the C6-B3-2 protocol
(maximum 160 tokens, EOS respected, no sampling). Baseline text **and** token IDs
must match all 32 canonical C6 STORAGE_KV_COMP cases. Installed SacreBLEU must
reproduce the canonical signature/protocol and baseline score
`3.1478254770301533`. A disagreement aborts the run; it cannot yield COMPLETE.

Teacher forcing reuses the C6-B3-2 canonical FULL_RECOMPUTE continuation,
imported through the adapter freeze decision. Every mode receives target prompt
plus the same canonical tokens except its final token. Logits predict that
same continuation. No mode teacher-forces its own greedy output. FULL_RECOMPUTE
and other imported context modes are not newly executed C7-B2 modes.

Metrics include corpus/user/depth SacreBLEU, incremental BLEU deltas, exact
generation matches, normalized edit distance, token-position agreement, KL,
logit cosine, top-1 agreement and top-5 set overlap (the established C6
definition). Generation fidelity and teacher-forced fidelity remain separate.
Per-case outputs preserve identities, references, generations/tokens, canonical
continuation tokens/hash, storage bytes, reuse evidence and runtime counters.

This is a bounded quality gate without an invented pass threshold or weighted
winner score. A positive BLEU perturbation is not evidence that compression
improves task quality. Manual review and an explicit later freeze are required.

## SERAPH execution

Use the existing local model cache and Python environment under `/data/khuss`
with one allocated GPU. The loader is offline and rejects runtime caches outside
`/data/khuss` on SERAPH. Do not execute this command on the local workstation.

```bash
cd /data/khuss/repos/GP_semcache
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
python -u scripts/62_run_cachegen_c7b2_q_quality_gate.py \
  --adapter-root results/cachegen/c6b3/b1_full \
  --freeze-decision results/cachegen/c6b3/b1_full_freeze_decision.json \
  --plan-dir results/cachegen/c6b3/multiwoz_plan_v2 \
  --source results/workloads/multiwoz.jsonl \
  --semantic results/workloads/c6b3_multiwoz_history_semantic.jsonl \
  --q-calibration-root results/cachegen/c7/b_q_calibration_retry2 \
  --kv-profile results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin \
  --storage-src /data/khuss/repos/GP_semcache/.c6_storage_src/src \
  --c6b3-baseline-root results/cachegen/c6b3/b2_frozen32 \
  --device cuda:0 --seed 42 \
  --output-root results/cachegen/c7/b2_q_quality_gate_frozen32
```

The output root must be new. Outputs are `summary.json`, `summary.csv`,
`per_case.csv`, `paired_quality.json`, `causal_comparisons.json`,
`storage_accounting.json`, `manifest.json`, and `summary.md`.
`q_profile_frozen=false`, `selected_q_candidate=null`, and
`manual_freeze_required=true`; no freeze decision file is produced.

CPU tests use the real archived C2 cache/codec implementations with literal
synthetic CDFs. They do not use the absent SERAPH profile artifacts or claim an
OPT quality result.
