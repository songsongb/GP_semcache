# C7-B resident TOTAL-Q calibration

C7-A2 established that the exact w=3 subsequence HIT reuses cached TOTAL Q/K/V.
C7-B calibrates storage compression of that TOTAL Q. It does not run generation
quality evaluation, change C6, refit K/V, or freeze a Q candidate.

## B0 cohort and capture

Script 60 reuses the B3 input parser and replay checks. It requires the pinned
frozen32 selection and both frozen adapter weight hashes. It replays capability64
and the saved full two-epoch training pool. It also scans the B3 results and plan
directories for existing evaluation selections and capability cohorts, accepting
only their known schemas and manifest-bound hashes. Missing, changed, unbound,
or inconsistent provenance fails closed. Training holdouts are excluded by the
saved/replayed training-pool membership.

Selection uses only eligible training rows, current-user token spans, and stable
seed-42 hashes. A deterministic conversation matching assigns one block per
distinct conversation: 16 blocks for each user/depth combination. The first 12
slots per combination are `profile_fit`, and the remaining four are
`candidate_select`. This fixes 96/32 disjoint subsets before loading the model
or observing Q values. No selection uses tensor statistics or quality results.

A passive native `q_proj` output hook captures the active user's base-plus-LoRA
TOTAL Q. It copies only the selected `[1,3,2560]` FP16 rows from each of 32 layers
to CPU. Each saved block is `[32,3,2560]`; no full-prompt Q, K/V, hidden states,
or transport-only delta is retained. B0 uses the pinned OPT revision, B3 prompt
semantics and tokenization, and one allocated GPU. All inputs are local-only.

From the SERAPH checkout under `/data/khuss/repos/GP_semcache`, using its existing
Python environment and one allocated GPU:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u scripts/60_capture_cachegen_c7b_q_calibration.py \
  --adapter-root results/cachegen/c6b3/b1_full \
  --plan-dir results/cachegen/c6b3/multiwoz_plan_v2 \
  --source results/workloads/multiwoz.jsonl \
  --semantic results/workloads/c6b3_multiwoz_history_semantic.jsonl \
  --device cuda:0 --seed 42 \
  --output-root results/cachegen/c7/b_q_calibration
```

Output: `cohort.json`, `capture/q_blocks.pt`, and
`capture/capture_manifest.json`. A non-empty B0 root is refused. SERAPH output
and runtime caches must resolve under `/data/khuss`; an existing environment
variable pointing elsewhere is rejected. Offline loading is mandatory.

## B1 transform, fitting, and accounting

The existing B2 mathematical transform is projection-role generic: it leaves
each layer's first-token anchor unchanged and represents later tokens as
`(symbol - anchor + 127) % 255`. Its public four-stream K/V frame is specific to
K/V pairing, so C7-B uses the same transform through the existing implementation
with an independent `B2_ANCHOR_MOD_RESIDUAL_Q` two-stream Q frame/profile namespace.
The actual loaded transform and stream ordering are checked against this algebra.

Q16/Q20/Q24/Q32 are **bin counts**, uniform across all layers. Quantization
reproduces the C1.5C shifted-rounding formula with a per-layer/token maximum and
`limit = bins // 2 - 1`. Q-only all-zero rows use centered symbols and reconstruct
to zero. The frozen K/V codec/profile is never changed.

Each candidate fits two shared anchor/residual CDFs on the 96 fit blocks only,
using the existing CPU `FAST_PY_BITEXACT` arithmetic backend. The saved profile is
reloaded before held-out evaluation. Only the 32 selection blocks enter
encode/decode measurement, under a runtime guard that aborts any profile fitting.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
python -u scripts/61_calibrate_cachegen_c7b_q_profiles.py \
  --capture-root results/cachegen/c7/b_q_calibration \
  --output-root results/cachegen/c7/b_q_calibration \
  --storage-src /data/khuss/repos/GP_semcache/.c6_storage_src/src \
  --kv-profile results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin
```

The external source is loaded by extending the cachegen package path, as in C6.
If that CPU backend is absent, B1 fails clearly; it neither downloads nor
substitutes a codec. It verifies the frozen K20/V16 artifact hash read-only.
B1 reuses hash-matching B0 files and never regenerates capture. A completed or
partial B1 root is refused to prevent overwrites.

Output: `profiles/q16.bin`, `q20.bin`, `q24.bin`, `q32.bin`,
`candidate_summary.csv/json`, `per_block_metrics.csv`, `per_layer_metrics.csv`,
`rate_distortion.json`, `summary.md`, and `manifest.json`.

Resident bytes count entropy payload plus local frame headers, FP32 maxima,
profile fingerprint and checksum. Shared profile bytes are reported separately.
The distortion reports finite MSE, relative L2, cosine and maximum absolute error
per block and per block/layer, plus mean/median/p95/max aggregates and cosine
minimum. Percentiles use linear interpolation at `(n-1)*p`. The Pareto frontier
compares resident compression ratio against mean relative L2; it does not choose
a winner. These tensor measurements do not establish BLEU or generation-quality
preservation. `selected_q_candidate` remains null and no C7-B2 freeze is performed.
