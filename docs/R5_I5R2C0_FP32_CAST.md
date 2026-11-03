# R5-I5R-2C-0: offline cast feasibility

CPU-only, read-only Layer 0 analysis. No production change or GPU execution.
Job 152859 data are on SERAPH, not assumed available locally. No local GPU
findings are claimed.

Script 136 requires explicit sealed S1 experiment, raw run and independent audit
paths. It verifies their complete inventories, matching configuration/source/run
identity, recorded historical Git blobs, archived independent audit PASS, tensor
hashes/shapes/dtypes and every role/layout/repetition. Original S1 source checks
are untouched: historical-source verification uses recorded Git blobs rather than
requiring old sources to equal this new checkout. This consumes a verified audit
export; it is not a fresh execution of the original S1 auditor.

For each role, layout and repetition, all first 64 positions are aligned:
A=native FP16 short, B=native FP16 long prefix, C=FP32 short rounded to FP16,
D=FP32 long prefix rounded to FP16. A/B, C/D, C/A, D/B and C/B have separate
metrics. Reference is the first named operand for relative L2. Numeric differing
values and bitwise differing elements/percentages are both recorded (signed zeros
can differ only in bits). Existing frozen thresholds and finite FP16 ULP metrics
are reused unchanged.

The cast uses Python struct's IEEE binary16 round-to-nearest/ties-to-even
conversion of exact recorded FP32 values. CPU tests cover even/odd midpoint ties,
negative values, adjacent FP32 values across a midpoint, signed zero and
subnormals. Nonfinite inputs and finite-FP16 overflow fail closed. C/D remaining
numeric mismatches report adjacent-bin midpoint straddles, exact ties, multiple-bin
cases and at most eight examples per record. This does not infer the rounding
history of native FP16 B. C==D never implies C==B.

`CAST_PREFIX_NUMERICALLY_CLOSER` requires both maximum absolute and relative L2
errors to be non-increasing versus A/B, with at least one strictly smaller.
`SOURCE_ONLY_BASELINE_INCOMPATIBLE` means C/B is not bitwise equal, independently
of whether a frozen tolerance passes. Counts retain the repetition denominator
and are grouped by role/layout. Missing/inconsistent required evidence exits
nonzero before output creation (`INCOMPLETE_EVIDENCE`); no partial PASS is emitted.

Fresh outputs only under `results/r5/i5r/numerical_fidelity/fp32_cast_feasibility/`
(or an external temporary directory for tests). No tensors are exported. At most
three roles × two layouts × eight repetitions generate small records; all reports
receive the standard SHA-256 `i5_output_manifest.json` seal.

Local tests:

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r2c0 -v
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r2bs1 test_r5_i5r2b -v
```

After user review, commit/push and manual SERAPH pull, run without a GPU allocation:

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
read -r -e -p 'Sealed S1 experiment directory: ' CAST_PLAN
read -r -e -p 'Original S1 job 152859 raw run directory: ' CAST_RUN
read -r -e -p 'S1 independent audit directory: ' CAST_AUDIT
CAST_OUT="$PWD/results/r5/i5r/numerical_fidelity/fp32_cast_feasibility/review_$(date -u +%Y%m%dT%H%M%SZ)"
CUDA_VISIBLE_DEVICES='' python scripts/136_analyze_r5_i5r2c0.py \
  --experiment "$CAST_PLAN" --run-dir "$CAST_RUN" --audit-dir "$CAST_AUDIT" \
  --output-dir "$CAST_OUT"
```

No historical path is guessed. Historical Git objects must be available; missing
objects fail strict provenance verification. No Torch/NumPy dependency is needed
for this analyzer; it uses stdlib and existing project modules. Source identity
requires the existing integration branch. Inputs and their manifests remain
unchanged. Reports: `i5r2c0_comparisons.json`, `i5r2c0_summary.json`,
`i5r2c0_provenance.json`, `i5r2c0_report.md`, `i5_output_manifest.json`.

No root cause confirmation, performance benefit, whole-model fidelity or restored
numerical fidelity follows from these Layer 0 casts. Integration corrections
require a separate reviewed implementation and unchanged-threshold validation.
