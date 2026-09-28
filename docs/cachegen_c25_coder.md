# C2.5: B2 entropy-coder hotspot and one-block benchmark

The C2 path calls `c2/physical_storage.py` → `b2/format.py:encode/decode` →
`shared/core.py:arithmetic_encode/arithmetic_decode`. The K anchor, K
mod-255 residual, V anchor, and V mod-255 residual are four separate B2
streams with one fixed CDF each. At OPT-2.7B `w=3`, their symbol counts are
81,920, 163,840, 81,920, and 163,840, respectively: 491,520 sequential
symbol iterations on each encode and decode. The coder is pure Python. Each
symbol updates 32-bit `low/high` arithmetic state and may enter a Python
renormalization loop; decode also performs one scalar `bisect_right` CDF
lookup and Python bit reads per renormalized bit. There are no tensor
`.item()` calls or CPU/GPU transfers *inside* the arithmetic coder. The
tensor-to-CPU/byte conversion and B2 transform happen before it. Four CDF
validations (255 intervals each) occur, but no CDF fitting or adaptation.

The pinned released CacheGen commit is
`6bed34ca9d495289955ed754cf2ad0c43346ee48`. Its
`LMCache/lmcache/storage_backend/serde/cachegen_encoder.py:encode_function`
calls `torchac_cuda.calculate_cdf` and `encode_fast_new`; its
`cachegen_decoder.py:decode_function_gpu` calls
`torchac_cuda.decode_fast_prefsum`. The native extension is bound in
`LMCache/third_party/torchac_cuda/main.cpp` and built from CUDA/C++ sources
listed in `setup.py`. The released path is GPU-native and uses a different
CDF layout and bitstream, so it cannot decode the frozen B2 profile/frames
without a separately validated format change. No `torchac_cuda`, `torchac`,
`constriction`, `arithmeticcoding`, or `numba` module is installed in this
local test environment. No dependency was installed.

An optional `FAST_PY_BITEXACT` coder in `shared/core.py` inlines bit
emission/reads, replaces division by 65536 with identical right shifts, and
reuses a 64 KiB inverse-CDF symbol table for each immutable frozen role CDF.
`REFERENCE_PY` remains the default. Unit tests verify identical stream and
B2 frame bytes, cross-decoding, C2 reconstructed K/V equality, Q unchanged,
frozen-profile SHA checking, and cache hit/miss behavior. The B2 transform,
four roles, CDF, profile, and physical byte accounting do not change.

One local synthetic 102,000-symbol stream measured approximately 0.442 s
versus 0.379 s encode, and 0.543 s versus 0.482 s decode (reference versus
fast; three runs each). This is only a local CPU diagnostic and does not
establish SERAPH latency. It suggests a modest reduction, not an online-latency
solution.

Run the one-block benchmark manually on SERAPH from
`/data/khuss/repos/GP_semcache`:

```bash
python scripts/45_cachegen_c25_coder_bench.py \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --profile-path results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin \
  --cachegen-repo /data/khuss/repos/CacheGen \
  --output-dir results/cachegen/c2_5/coder_benchmark \
  --repeats 3
```

It reads one sorted evaluation `w=3` C1 fixture, quantizes on the recorded
C1 CUDA device, then benchmarks both coders on the same four CPU streams.
It writes `manifest.json` and `coder_benchmark.json`; no model inference,
new CDF, or payload files are produced. The report includes transform time,
per-role and total encode/decode mean/p50, symbol counts, encoded bytes,
throughput, physical ratios, symbol/reconstruction checks, and bitstream
identity. The fast backend is **not** selected automatically.

If the one-block result merits a second check, the existing eight-block C2
smoke accepts `--coder-backend FAST_PY_BITEXACT` with a *new* output
subdirectory under `results/cachegen/c2/storage_smoke/`. Do not overwrite the
reference smoke artifacts.
