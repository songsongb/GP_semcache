# C1.5-B1: lossless symbol representation and locality probe

Implementation status: implemented, not run on real C1 fixtures locally.
This is `MEASURED_RESEARCH_EXTENSION` / `CACHEGEN_INSPIRED` /
`B1_LOSSLESS_SYMBOL_TRANSFORM` when measured. It does not reproduce official
CacheGen floating-point anchor/delta encoding. B0 evidence remains in
[the audit](cachegen_c15b_audit.md); the B1 protocol here supersedes its provisional
B1 design according to the subsequent user specification.

From the repository root on SERAPH, in the existing CUDA-enabled environment:

```bash
conda activate semcache
python scripts/41_cachegen_anchor_probe.py \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --c15a-dir results/cachegen/c1_5 \
  --output-dir results/cachegen/c1_5b
```

One CUDA GPU is required by the frozen C1 quantization/reconstruction contract.
The command reads every frozen calibration/evaluation T=10 fixture, calibration
first, preserving manifest order within each split. Evaluation must contain
236 T=10 blocks. Both datasets must appear in each split. Exact per-split
SNIPS/MultiWOZ/ALL block counts and IDs are recorded from the capture manifest;
the calibration count is not assumed. T=3 blocks are excluded and never loaded.
There is no padding, regrouping, sampling, inference, download or timing loop.

The operation is a linear tensor scan: one unchanged C1 CUDA quantization per
block, CPU integer transforms/counts, and descriptive tensor reductions. It
should be a minutes-scale diagnostic on ordinary storage, rather than the
hours-scale Python arithmetic-coding job; runtime depends on calibration count,
I/O and host CPU. This is an estimate, not a measured timing result. No official
checkout or CacheGen extension is imported or required.

## Exact representation

K and V each have `[32,10,2560]` captured FP16 values. The existing C1
`Baseline(UNIFORM_INT8).encode` runs on the recorded CUDA device. It yields int8
symbols `s` in `[-127,127]` and FP32 `[layer,token,1]` scales, unchanged.

For each layer/channel, `u=s+127` maps bijectively to `[0,254]`.
Keep token 0's `u` as anchor `a`. For each token `i=1..9`:

```text
r_i = (u_i - a) mod 255
 d_i = r_i           if 0 <= r_i <= 127
       r_i - 255    if 128 <= r_i <= 254

r_i = d_i mod 255
u_i = (a + r_i) mod 255
s_i = u_i - 127
```

Subtraction/modulo use wide integers. For packed histograms, residual slots hold
`d_i+127`, so zero residual is histogram bin 127. Anchors retain their original
`s+127` indices. All nine tokens reference token 0, never their predecessor.
This is a cyclic integer representation, not a floating-point KV delta.

The inverse must recover every original K/V symbol exactly. Original scales
are retained, never recomputed. Reconstruction uses the C1 CUDA semantics and
must equal direct C1 reconstruction. Evaluation tensors must additionally pass
`torch.equal` against the saved C1 UNIFORM_INT8 reconstruction, with its file hash
validated. Calibration has no saved C1 reconstruction and records that check as
not applicable, not as a fabricated success. Q remains untouched FP16. This
exact inverse introduces no quality change relative to C1 UNIFORM_INT8.

## Fair entropy controls

| Family | Probability models per K/V component |
|---|---|
| RAW_GLOBAL | Original symbols, all ten token positions pooled |
| RAW_ROLE_SPLIT | Original token-0 symbols; original tokens 1–9 separately |
| ANCHOR_MOD_RESIDUAL | Original token-0 symbols; centered residuals separately |

All layers are pooled; no new layer/channel model is introduced. Histograms
are empirical integer counts, with no smoothing, CDF normalization, profile
fitting or arithmetic coding. Calibration and evaluation are completely separate.
Dataset-specific distributions and an ALL-dataset distribution are reported.
ALL entropy is calculated from ALL counts, not averaged dataset entropies.

For each role, `H=-sum(p*log2(p))`. Family cost is
`sum(role_symbol_count * H_role) / sum(role_symbol_count)`, using actual counts.
K+V is likewise the symbol-count-weighted cost of **separate K and V models**;
it is not entropy of a pooled K/V histogram. Unique-symbol counts are reported
for individual distributions; a mixture of models has no single unique-count
value, so that field is blank. Component/role histograms remain available in
JSON fields in the CSVs. Per-block entropies are diagnostic; primary population
entropy is computed after aggregating symbol counts across the population.

The primary causal comparison is `ANCHOR_MOD_RESIDUAL` against `RAW_ROLE_SPLIT`.
Comparison against `RAW_GLOBAL` also includes the effect of role modeling.
Percent reduction is `100*(H_control-H_variant)/H_control`. With a zero baseline,
zero-to-zero is recorded as 0%; otherwise relative reduction is undefined.
These are empirical entropy diagnostics, not encoded bytes or storage ratios.

## Frozen practical decision

Before any fixture reads, the run writes its transform/configuration and this
`REPRODUCTION_CHOICE` rule into both its manifest and the B1 design entry:

- Overall **evaluation K+V** relative entropy reduction versus RAW_ROLE_SPLIT
  must be **at least 3%**.
- Both SNIPS and MultiWOZ must individually show a **strictly positive**
  reduction for the same evaluation comparison.
- All correctness, input-identity and coverage checks must pass.

If all conditions hold, report `GO_TO_B2`; otherwise `STOP`. This is only a
recommendation for a future bitstream experiment. It neither executes B2 nor
asserts statistical significance or guaranteed byte savings. There are no CLI
options for changing the threshold, transform, population or alphabet after
seeing evaluation results.

Raw FP16 locality is descriptive only. Analysis promotes values to float64 and
computes `x_i-x_0` for distances 1–9, separately for K/V. Mean absolute delta and
RMS use all included scalar elements. Relative L2 divides delta norm by the
norm of the corresponding original non-anchor values. Cosine uses concatenated
target and repeated-anchor vectors, not an average of per-vector cosines. Zero
norm denominators produce blank values. Aggregate distances 1–9 are also
reported. No differential entropy or floating-point quantization is estimated.

## Artifacts and failure behavior

Outputs are exclusively under `results/cachegen/c1_5b/b1/`:

- `manifest.json`: frozen rule/configuration, source/input hashes, device contract,
  populations, coverage, status and recommendation.
- `b1_entropy_raw.csv`: ten rows per block (K/V × five family/role distributions),
  including all 255 symbol counts per row.
- `b1_entropy_summary.csv`: 54 rows (two splits × three datasets/ALL × K/V/K+V ×
  three families), including component/role population counts and entropies.
- `b1_locality_summary.csv`: 120 rows (two splits × three datasets/ALL × K/V ×
  nine distances plus their aggregate).
- `b1_roundtrip.json`: exactness results by block, with saved-reference scope explicit.
- `environment.json`: operational environment.

The parent `design_manifest.json` is updated only for B1 execution state and
rule; B2/B3 definitions/statuses are preserved. The runner validates completed
C1.5-A manifests, original Baseline source, capture/reconstruction identity,
CUDA contract and frozen profile hashes by reading them. It does not load CDFs
for modeling or refit them. Metadata hashes are checked again at completion.
Original fixture/reconstruction files are checked when loaded.

A new `b1/` directory is created exclusively; existing results are never
silently overwritten. This short diagnostic job does not implement resume.
Preserve/archive an interrupted or failed B1 directory before a deliberate
restart. SIGINT/SIGTERM record INCOMPLETE; correctness and other failures record
FAILED. Only a complete run has `diagnostic_result_eligible=true`. Any CSVs from
an interrupted finalization must be interpreted using the manifest status.
`primary_storage_result` is always false: B1 performs no storage experiment.

No B2/B3 codec, new lossy quantizer, C2, physical-page redesign or production
GlobalQKVCache integration is implemented by this command.
