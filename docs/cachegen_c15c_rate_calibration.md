# C1.5C-2: role-wise matched-rate calibration

**Real calibration: NOT RUN — requires SERAPH.** Implementation and tiny CPU
synthetic tests only. C1.5C-1 storage correctness is already established; this
stage fits fair policy-specific entropy models and selects a storage-rate
comparator. No model inference, download, Slurm submission or C1.5C-3 evaluation
is implemented.

## Policies and data

`CACHEGEN_RELEASED_QL2` retains K layers 0–9 at 32 bins, K10–31 at 16,
V0–1 at 32 and V2–31 at 16. `UniformKVPolicy(K_bins,V_bins)` uses one constant
K count and one constant V count across all 32 layers. K/V counts may differ.
Both use the same inherited released vectorized formula, maxabs axis, FP16
input/maxabs, FP32 bin/factor behavior, shifted rounding and dequantization.

Defaults: `8 10 12 14 16 18 20 22 24 26 28 30 32`. Candidate counts must be
distinct even integers. Configurable counts are bounded to 4…128: C must be
positive and the released shifted int8 codes `0…bins-2` must not wrap. No odd,
fractional or interpolated settings are used.

Only complete `partition=calibration`, `token_group_size=3` blocks are selected,
sorted by block ID. Selection tests the partition before reading block details.
The existing C1 provenance verifier receives a calibration-only view with empty
evaluation blocks/samples. No evaluation tensor or distribution is accessed.
The original complete manifest is hashed for provenance; its calibration sample
hashes, exact selected IDs and capture paths are recorded. Seed is recorded but
does not subsample the complete partition. The existing hash-checking tensor
loader and recorded C1 CUDA device contract are reused, without CPU fallback.

## Fresh CDFs and independent roles

No existing Uniform INT8 CDF is reused. A fresh QL2 profile and fresh uniform
candidate profiles are fitted from each policy's own calibration symbols.
Fitting/rate estimation deliberately share the complete calibration partition.
Each distribution uses the existing B2 token-zero anchor/modulus-255 residual
transform, +127 signed byte mapping, and two independent CDFs. Smoothing and
integer normalization are exactly `shared.core.cdf_from_counts`.

The full existing B2 profile is four CDFs, ordered K-anchor, K-residual,
V-anchor, V-residual. `RoleProfile`/`compose_profile` safely compose two K and
two V CDFs, checking their role order and shared calibration scope hash.
`format.frame_payloads` is a small extraction of the existing B2 framing code;
normal `encode` calls it with identical behavior. Independent payloads can be
framed without additional arithmetic coding. The original B2 transform,
alphabet, scale slots, checksum and bitstream format are unchanged.

Two capture scans are used:

1. Fit counts for QL2 plus every candidate independently for K and V.
2. Requantize the same blocks, encode/decode each role's two streams using its
   own fitted CDFs, and collect payload lengths and reconstruction statistics.

Actual candidate frames use that role's already encoded payloads plus the QL2
counterpart. Their byte lengths are checked against fixed-format accounting.
Primary/secondary pair sizes are exact compositions of measured role lengths
and fixed overhead; no Cartesian pair is re-encoded. Candidate full-physical
CSV/JSON columns explicitly identify the QL2 counterpart. No candidate
bitstreams are retained, and only two selected full profiles are persisted.

## Accounting and selection

For each role, payload is anchor bytes plus residual bytes, including each
arithmetic stream's termination. This is the primary matching quantity.

For every w=3 block, unchanged B2 metadata is:

* 99 bytes: block header, four payload lengths and SHA256 checksum.
* 768 bytes: `2*32*3` FP32 maxabs slots. FP16 maxabs is widened exactly, with
  identical metadata bytes for every policy. These slots hold maxabs, not a
  policy-dependent scale step.
* 867 bytes total local metadata.

The full four-CDF profile is 4,108 bytes: a 12-byte shared header and four
1,024-byte CDF tables. Each role accounts for 2,048 CDF bytes; the shared header
is accounted once. Full physical bytes are:

```text
K payload pool + V payload pool + 867*N + 4108
```

The actual serializer lengths and metadata breakdown are checked per block.
Global profile bytes are counted once per policy/workload. Compression ratio
is original FP16 K+V bytes divided by full physical bytes; Q is excluded.

Primary selection independently minimizes the absolute relative K payload gap
and V payload gap against fresh QL2 targets. Ties choose smaller bins. Quality
metrics are never consulted. The result is `UNIFORM_KXX_VYY`, with no automatic
adjustment when a nearest gap exceeds 1%.

Secondary selection minimizes the full physical byte gap over all candidate
pairs using addition only. Ties use ascending `(K_bins,V_bins)`. It is labelled
diagnostic and cannot replace the primary policy. Signed K/V/overall gaps and
compression ratios are reported for both choices. Brackets are the closest
strictly lower and strictly upper payload rates, with missing sides set to
null and exact matches listed separately. Nonmonotonic measured rates are
handled by comparing bytes, not bin order.

## Quality and failure behavior

MSE, RMSE, relative L2, cosine and max absolute error use C1.5C-1's float64
pooled sufficient statistics, original FP16 versus final FP16 reconstruction.
The calculation pools complete tensors over calibration blocks. Zero-norm
ratios are null, and no NaN/Inf is allowed into summaries. These are diagnostics.

Each measured role encode is decoded, inverted through the same transform,
and checked for exact quantized symbols and metadata. FP32 and final FP16
storage reconstruction must equal quantization-only reconstruction. No storage
correctness smoke is relaunched.

There is no zero guard. Zero-maxabs vector counts and their resulting integer
code histograms are recorded per policy/role. Invalid released cast results
or nonfinite reconstruction fail explicitly, preserving the observed counts
and context in `calibration_manifest.json`; values are never repaired.

## Manual SERAPH command

Use the existing C1 Python environment and a GPU allocation exposing the
recorded C1 device. This command has **NOT been run** during implementation:

```bash
cd /data/khuss/repos/GP_semcache
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/43_cachegen_released_ql2.py rate-calibrate \
  --capture-manifest /data/khuss/repos/GP_semcache/results/cachegen/c1/capture_manifest.json \
  --cachegen-repo /data/khuss/repos/CacheGen \
  --output-dir /data/khuss/repos/GP_semcache/results/cachegen/c1_5c/rate_calibration \
  --candidate-bins 8 10 12 14 16 18 20 22 24 26 28 30 32 \
  --seed 42
```

Optional `--device` asserts the recorded device; it cannot change execution
device. CacheGen HEAD/source hashes must match the already pinned released
commit `6bed34ca9d495289955ed754cf2ad0c43346ee48`; no official CUDA package import
or reference smoke is needed. The output directory must be fresh and remain
inside `results/cachegen/c1_5c/rate_calibration/`. For a repeat, use a fresh
subdirectory such as `rate_calibration/repeat_01`; no result is overwritten.

Expected outputs:

* `calibration_manifest.json`: input/code/reference hashes, device, exact IDs,
  fitting/selection/accounting rules, progress/status, output/profile hashes.
* `ql2_calibration_summary.json`: fresh QL2 rate, physical accounting and K/V
  reconstruction diagnostics.
* `uniform_rate_search.csv`: one row per role/bin, with payloads, attributed CDF
  bytes, gaps, quality and physical bytes with QL2 counterpart.
* `uniform_rate_search.json`: candidate details, counts/hashes, zero statistics,
  per-block actual frame/payload sizes and metrics.
* `selection.json`: primary policy, secondary diagnostic, role/overall gaps,
  strict brackets, exact matches, <=1% flag, exact composed block sizes and
  selected profile references.
* `profiles/cachegen_released_ql2.bin`
* `profiles/matched_uniform_kXX_vYY.bin`

Only a COMPLETED manifest with `profiles_status=FROZEN_FOR_C15C3` is eligible
for later reuse. C1.5C-3 is intentionally absent.

## Workload and local tests

The real calibration block count N is not available in the local workspace.
The CLI records N from the frozen calibration w=3 manifest, without guessing.
For B=13 candidates per role:

* 14 K distributions + 14 V distributions, including QL2.
* 28 role fitting passes and 28 role encoding passes over N blocks.
* Two fixture scans, 2N fixture loads, 56N quantization calls.
* 56N arithmetic encode calls and 56N decode calls (two logical streams/role).
* 169 secondary pairs compared arithmetically; zero Cartesian pair encodes.

Tests use tiny CPU tensors, synthetic byte counts and synthetic profiles.
They cover constant layer allocation, candidate validation, independent CDF
composition, exact frame equality, single-profile accounting, quality-independent
primary selection, nonmonotonic brackets/ties/missing sides, secondary matching,
partition isolation, finite statistics, zero behavior, and the two-scan encode
call count. Existing QL2 parity and B2 golden bitstream tests are retained.

Local validation: the focused C1.5C tests passed (60 tests), and the complete
lightweight pytest suite passed with **393 passed, 22 skipped, 38 subtests
passed**. GPUs were hidden and optional real-model/dataset integration was
disabled. Syntax and whitespace checks passed. No real calibration artifact
was generated and no previous result artifact was modified.

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_cachegen_rate_calibration.py tests/test_cachegen_c15c.py
```
