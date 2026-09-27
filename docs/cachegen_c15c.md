# C1.5C-0 / C1.5C-1: released CacheGen QL2

Implemented scope: deterministic CPU quantizer parity and a bounded capture-only
OPT-2.7B w=3 smoke. **Real OPT-2.7B smoke: NOT RUN — requires SERAPH.**
No model loading, capture generation, full-workload command, matched-rate search,
new dependency, or Slurm submission is included.

## Released source

The requested SERAPH reference path was not mounted in the implementation
workspace. The official repository was fetched into
`/tmp/semcache-c15c-cachegen-reference` and checked out at
`6bed34ca9d495289955ed754cf2ad0c43346ee48`. All three source files were inspected;
their SHA256 values and exact functions are recorded in
`results/cachegen/c1_5c/design_manifest.json` and every parity/smoke manifest.

Sources, all under `LMCache/lmcache/storage_backend/serde/`:

* [cachegen_basics.py](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/lmcache/storage_backend/serde/cachegen_basics.py):
  `CacheGenConfig.from_model_name`, Mistral-7B `QUANT_LEVEL == "2"` branch.
* [cachegen_encoder.py](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/lmcache/storage_backend/serde/cachegen_encoder.py):
  `CacheGenSerializer.make_key_bins`, `make_value_bins`, `_split_kv`,
  `torch_quant_vectorized`, and active `encode_function`.
* [cachegen_decoder.py](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/lmcache/storage_backend/serde/cachegen_decoder.py):
  `do_dequantize` and `CacheGenDeserializer.from_bytes`.

OPT is absent from the released model whitelist. This experiment copies the
32-layer schedule literally to OPT, as requested. It is quantizer parity plus
an OPT adaptation, not a claim that the official serializer supports OPT.

## Quantizer

| Role | Layers | Bin count | C = bins//2 - 1 | Released shifted codes | Signed storage view |
|---|---|---:|---:|---|---|
| K | 0–9 | 32 | 15 | 0…30 | −15…15 |
| K | 10–19 | 16 | 7 | 0…14 | −7…7 |
| K | 20–31 | 16 | 7 | 0…14 | −7…7 |
| V | 0–1 | 32 | 15 | 0…30 | −15…15 |
| V | 2–31 | 16 | 7 | 0…14 | −7…7 |

For `[layers,tokens,hidden]`, where `hidden=heads*head_dim`:

```text
C = (bins // 2 - 1)[:,None,None]       # FP32 bins from released torch.zeros
m = amax(abs(x), dim=-1, keepdim=True) # retains input FP16 dtype
factor = C / m                       # FP32, vectorized promotion
q = round(x * factor + C).to(int8)    # shift BEFORE ties-to-even rounding
xhat = (q.float() - C) / C * m        # FP32, exact released operation order
final = xhat.to(float16)             # huggingface deserializer output
scale = m / C                        # exposed step; m is the stored source metadata
```

No clamp, epsilon, anchor/delta, or entropy coding is added to the quantizer.
The older signed `torch_quant` helper is not the reference. All-zero channel
vectors have the released `0*inf -> NaN -> int8` behavior. That cast is device
dependent; CPU parity measures it without inventing a zero guard. The storage
adapter rejects codes outside their nominal range rather than silently fixing
them. Smoke records zero-maxabs vector counts.

The reusable policy accepts OPT head layout `[L,H,T,Dh]` as well as merged
`[L,T,hidden]`. Partial layer sets require explicit original layer indices;
boundaries are never rescaled. Quantized results expose bins, limits, maxabs,
step, shifted codes, signed storage view, FP32 dequantization and final dtype
reconstruction.

## Parity and local tests

`parity` always uses tiny fixed CPU FP16/FP32 tensors with signs, zero,
extrema/near-extrema, shifted rounding ties, independent token scales and
all-zero vectors. It covers K0/K9/K10/K19/K20/K31 and V0/V1/V2/V31.
The JSON records per-case bins/range, maxabs/scale differences, symbol equality,
reconstruction differences and pass/fail, and the CLI prints a compact table.

With `--cachegen-repo`, source hashes and HEAD must match before isolated
released functions are executed via AST. CacheGen imports/decorators are
excluded; only the bin builders' final `.cuda()` is replaced by identity.
Without that option, independent direct transcriptions of the released
vectorized formulas and literal bin slices are used. The official package is
never a runtime dependency.

Focused tests check every boundary, deliberately mutate each adjacent boundary
to verify parity failure, test scale axes/dtypes/shifted ties/zero behavior,
and verify exact storage metadata, symbols and reconstruction. Three fixed
SHA256 golden bitstreams from the original B2 source at SemCache commit
`dcb593abb1694958f87d704776614e92cbb14b82` prove unchanged T=10 bytes for all modes.
Uniform INT8's quantizer, zero guard, scales and decoder remain unchanged.

Local validation on the implementation workspace: 367 pytest tests passed,
22 skipped, and 38 subtests passed. GPUs were hidden and all optional real
model/dataset integration switches disabled. Existing PyTorch 2.12.1+cu130 was
used only on CPU; pytest was installed into a temporary `/tmp` test directory.
The hash-verified released AST parity passed all 20 FP16/FP32 layer cases with
zero maxabs, scale, symbol, FP32 reconstruction and final-cast differences.
Tiny synthetic QL2 storage tests reported zero K/V symbol mismatches. Syntax
and Git whitespace checks passed. No real capture/storage artifact was used.

Full local unit-suite invocation (paths are specific to this workspace):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  SEMCACHE_OPT_INTEGRATION=0 SEMCACHE_TINYBERT_INTEGRATION=0 SEMCACHE_DATASET_INTEGRATION=0 \
  CACHEGEN_REFERENCE_REPO=/tmp/semcache-c15c-cachegen-reference \
  PYTHONPATH=/tmp/semcache-c15c-test-deps:src \
  /home/songsong/projects/fedavg-study/.venv/bin/python -m pytest -q
```

Optional exact-checkout test command on SERAPH (CPU only):

```bash
cd /data/khuss/repos/GP_semcache
CUDA_VISIBLE_DEVICES='' CACHEGEN_REFERENCE_REPO=/data/khuss/repos/CacheGen \
  python -m pytest -q tests/test_cachegen_c15c.py
```

## Real smoke and existing storage

The smoke reads the existing C1 `capture_manifest.json` and uses its existing
verification, tensor loader/hash validation, dimensions, and recorded device
contract. It selects 12 evaluation w=3 blocks by default, with seed 42 and
deterministic round-robin dataset sampling; the hard maximum is 16. Both K and
V, all 32 layers, are evaluated. Quantization/reconstruction use the recorded
C1 CUDA device, with no CPU fallback. The entropy coder remains CPU-side.

Per-layer/block rows include elements, FP16 raw bytes, nominal symbol count,
MSE, RMSE, relative L2, cosine and max absolute error. Summaries pool
float64 sufficient statistics over concatenated elements for K/V overall,
each layer, and each 32-bin/16-bin group. Metrics compare original FP16 against
final FP16 reconstruction. Zero-norm relative L2/cosine are null.

Optional storage uses the frozen B2 `B2_ANCHOR_MOD_RESIDUAL_KV` profile from its
existing directory. Manifest/profile hashes, calibration-only fitting and
CDF/count parity are verified. No profiles are fitted, transformed or replaced.
QL2 codes are losslessly centered (`q-C`), then mapped with B2's existing +127
byte mapping. Anchor/residual roles, modulus 255, framing, checksums and coder
are reused unchanged. FP16 maxabs is exactly widened into existing FP32 metadata
slots, then restored before the policy's dequantization.

B2 formerly forbade T=3. Its format now takes an explicit `expected_tokens=3`
option; defaults still require T=10 and retain the original rejection message.
The original B1 transform is untouched. T=3 uses the same token-zero anchor
and modulus-255 inverse, without padding or regrouping. No alternate-alphabet
change was needed because QL2's signed symbols are subsets of B2's alphabet.
The policy owns storage conversion/metadata restoration, so future quantizers
can implement that interface without changing entropy code.

Frozen original B2 run contracts include implementation hashes. The updated
format source therefore intentionally cannot resume an older B2 run contract;
do not edit its manifests or hashes. Its existing profiles and bitstreams stay
readable, and C1.5C reads the frozen profile through its own diagnostic adapter.

Each selected block gets one encode/decode with symbol mismatch counts for K/V
and exact metadata/reconstruction checks. Quantization reconstruction error and
storage corruption are separate reports. Any nonzero storage mismatch fails
the run. Storage counts include the existing framing/metadata overhead; this
smoke makes no compression-performance or downstream-quality claims.

## Exact SERAPH commands

Use the already configured C1 Python environment. These are manual commands;
**real smoke: NOT RUN — requires SERAPH**.

```bash
cd /data/khuss/repos/GP_semcache

# CPU parity against the live hash-verified pinned reference.
CUDA_VISIBLE_DEVICES='' python scripts/43_cachegen_released_ql2.py parity \
  --cachegen-repo /data/khuss/repos/CacheGen

# Small real smoke, including the existing frozen B2 storage profile.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/43_cachegen_released_ql2.py smoke \
  --capture-manifest /data/khuss/repos/GP_semcache/results/cachegen/c1/capture_manifest.json \
  --cachegen-repo /data/khuss/repos/CacheGen \
  --num-blocks 12 --seed 42 \
  --b2-profile-dir /data/khuss/repos/GP_semcache/results/cachegen/c1_5b/b2
```

Omit `--b2-profile-dir` for quantization-only smoke. A GPU allocation with the
recorded C1 device is needed for smoke; the CLI submits no jobs. Reference HEAD
and source files must match the pinned commit. No model is loaded or downloaded.

Expected outputs under `results/cachegen/c1_5c/`:

* `c15c_release_parity.json`: locally runnable synthetic parity.
* `smoke/manifest.json`: reproducibility, source/input/code hashes and status.
* `smoke/c15c_release_parity.json`: smoke preflight CPU parity.
* `smoke/c15c_smoke_raw.csv`: block/role/layer reconstruction metrics.
* `smoke/c15c_smoke_summary.json`: pooled K/V, layers and bin-group metrics.
* `smoke/c15c_storage_roundtrip.json`: separate symbol corruption checks.
* `smoke/storage/b2_profile.bin` and `smoke/storage/block_000.bin` …
  `block_011.bin`, only when storage is requested.

The smoke refuses to overwrite an existing output directory. For a repeat, add
`--output-dir results/cachegen/c1_5c/repeat_01`; outputs go in its `smoke/`
subdirectory. All CLI output directories must stay inside the C1.5C namespace.
