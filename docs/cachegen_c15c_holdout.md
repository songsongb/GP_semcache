# C1.5C-3: frozen-policy evaluation holdout

This command compares the two policies frozen by C1.5C-2:
`CACHEGEN_RELEASED_QL2` and `UNIFORM_K20_V16`. It reads only blocks marked
`partition=evaluation` from the existing C1 capture manifest. It rejects an
empty evaluation partition and records every selected block ID. The local
checkout has only a `NOT_RUN` capture manifest, so real evaluation must be
run on SERAPH.

The command loads each evaluation fixture once, quantizes K and V under both
policies, and arithmetic-encodes four B2 streams per policy using the two
frozen C1.5C-2 profiles. It does not fit a CDF, run inference, or decode
evaluation streams. The expected encode count is `8 × evaluation blocks`.
Both T=3 and T=10 evaluation blocks are included if present in the manifest;
the command prints their counts before starting. Quantization, the B2
anchor/mod-255 residual organization, framing, and metadata accounting retain
their established definitions.

From `/data/khuss/repos/GP_semcache` on SERAPH, first check the frozen inputs
and workload size without reading fixtures or creating an output directory:

```bash
python scripts/43_cachegen_released_ql2.py holdout-compare \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --rate-calibration-dir results/cachegen/c1_5c/rate_calibration \
  --output-dir results/cachegen/c1_5c/holdout \
  --seed 42 --dry-run
```

Then run the holdout comparison once, using a fresh output directory:

```bash
python scripts/43_cachegen_released_ql2.py holdout-compare \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --rate-calibration-dir results/cachegen/c1_5c/rate_calibration \
  --output-dir results/cachegen/c1_5c/holdout \
  --seed 42
```

The command writes only these compact files under
`results/cachegen/c1_5c/holdout/`:

- `manifest.json`: source hashes, block IDs/counts, frozen policy provenance,
  device contract, progress, and encode count.
- `evaluation_summary.json`: element-pooled K/V reconstruction metrics and
  exact role payload, local metadata, global profile, total physical byte,
  and FP16 compression accounting for the overall partition and each dataset.
- `evaluation_comparison.json`: Uniform-minus-QL2 storage gaps, signed
  reconstruction differences, and decision facts. Error deltas below zero
  favor Uniform; the cosine delta is QL2 minus Uniform so its negative values
  also favor Uniform. The selected policy remains `UNIFORM_K20_V16`.
- `per_dataset_summary.csv`: dataset/policy/role metrics and byte accounting.

The profile bytes are charged once per reported workload, not once per block.
No bitstreams, quantized symbols, or reconstructed tensors are persisted.
Arithmetic decoding is skipped entirely; storage symbol equality was already
verified in C1.5C-1. C1.5C-3 does not decide which policy wins the broader
research comparison.
