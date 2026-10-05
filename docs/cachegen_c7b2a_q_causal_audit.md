# C7-B2A cached-Q causal sanity audit

This read-only audit uses exactly four frozen32 cases: the first canonical episode
in stable selection order at each history depth 1–4. `selected_cases.json` records
the exact IDs and original selection indices after canonical provenance validation.
No cases are chosen using quality, distortion, storage savings, or cross-user status.

It reuses C7-B2's model, adapters, Q24/Q32 profiles, unchanged C6 K20/V16 storage,
actual GlobalCache lookup, and canonical teacher-forced helper. It requires the B2
manifest to be COMPLETE, match C6-B3-2, have zero fitting/transport counters, and
bind the current input/output hashes. All C7-B2 provenance checks also run before
model loading. Existing artifacts are read only; a new output directory is required.

The four modes are KV_BASELINE, Q24, Q32, and Q_ZERO_COUNTERFACTUAL. The zero mode
wraps the baseline's temporary decoded HIT view, replacing only its cached Q with
zeros immediately before mixed reuse. Resident entries and K/V are untouched.
Q24/Q32 continue to retain compressed Q frames only. Raw source Q copies made for
measurement are CPU audit working data, not resident cache representation.

Passive wrappers installed *after* the real mixed path copy the actual Q/K/V return
tensors before attention consumes them. Hooks on OPT `self_attn.out_proj` capture
attention output before decoder residual/dropout; decoder layer hooks additionally
capture post-layer hidden states. All wrappers/hooks are restored even on failure.
Nothing reprojects the HIT rows or substitutes a synthetic attention calculation.

For all 4 × 32 layer blocks, the audit measures raw/decoded Q distortion and bit
fingerprints, verifies exact injection, checks fresh Q and HIT K/V identity, and
checks the same mask and three skipped native rows for each projection role.
Nonidentical decoded Q and a positive zero-Q internal effect are mandatory for
each case. Contradictory evidence produces INVALID with partial reports.

Baseline and zero-Q internal traces compare HIT and fresh positions independently.
Fresh attention differences are measured, including exact equality, rather than
assumed; tiny differences alone do not invalidate the audit. Continuation metrics
use the identical imported FULL_RECOMPUTE continuation previously used by C7-B2,
with `prompt + canonical[:-1]`. Reports include maximum/mean logit difference,
KL, cosine, and top-1/top-5 agreement. Q24/Q32 continuation comparisons are also
retained as diagnostics. Material continuation changes are valid observations,
and make causal isolation false.

The predeclared numerical tolerances are max logit difference ≤ 1e-6, mean logit
difference ≤ 1e-7, and fresh attention max difference ≤ 1e-6. Isolation additionally
requires unchanged continuation top-1/top-5 and, when requested, greedy token IDs.
These are causal measurement criteria, not compression-quality acceptance gates.

Run from `/data/khuss/repos/GP_semcache` on SERAPH with one allocated GPU:

```bash
python -u scripts/63_run_cachegen_c7b2a_q_causal_audit.py \
  --adapter-root results/cachegen/c6b3/b1_full \
  --plan-dir results/cachegen/c6b3/multiwoz_plan_v2 \
  --source results/workloads/multiwoz.jsonl \
  --semantic results/workloads/c6b3_multiwoz_history_semantic.jsonl \
  --q-calibration-root results/cachegen/c7/b_q_calibration_retry2 \
  --kv-profile results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin \
  --storage-src /data/khuss/repos/GP_semcache/.c6_storage_src/src \
  --c7b2-root results/cachegen/c7/b2_q_quality_gate_frozen32 \
  --device cuda:0 --seed 42 \
  --output-root results/cachegen/c7/b2a_q_causal_audit
```

Default freeze-decision and canonical C6 baseline arguments match C7-B2. Runtime
artifacts/caches are restricted to `/data/khuss`; Hugging Face loading is offline.
The default performs 12 source and 16 teacher-forced full-sequence forwards plus
12 storage encodes/lookups, of which eight use Q compression. Optional `--greedy`
adds the established bounded greedy helper (up to 160 tokens) for four cases/four
modes, and checks the baseline against the canonical C6 generations. Wall time
depends on SERAPH hardware, prompt lengths, and passive host copies; no timing
benchmark or unsupported minute estimate is supplied.

Outputs: `selected_cases.json`, `q_distortion.json`, `injection_audit.json`,
`internal_causal_effect.json`, `continuation_causal_effect.json`, `manifest.json`,
`summary.md`. Profiles are never fitted or frozen. An unchanged continuation does
not imply Q is unnecessary: interpretation is limited to whether reused cached Q
changes HIT-row computation and propagates to fresh continuation under fixed K/V.
