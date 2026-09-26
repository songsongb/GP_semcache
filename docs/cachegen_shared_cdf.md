# C1.5-A: fixed/shared entropy models for SemCache KV storage

This is a **CacheGen-inspired fixed/shared entropy-model adaptation for SemCache
KV storage**, labelled **RESEARCH_EXTENSION**, not official CacheGen. It changes
only storage of the existing C1 UNIFORM_INT8 symbols. Q stays captured FP16.
Production SemCache and the C1 capture, benchmark, and quality files are untouched.

The implementation is on `exp/cachegen-c1-5-shared-cdf`. No profile fitting,
real-data benchmark, or model quality run has been performed in this checkout:
the completed C1 fixtures and `/data/khuss/repos/CacheGen` live on SERAPH. Local
C1 files are placeholders, so executing against them fails closed.

## Pinned official source audit

Revision: `6bed34ca9d495289955ed754cf2ad0c43346ee48`,
[official tree](https://github.com/UChi-JCL/CacheGen/tree/6bed34ca9d495289955ed754cf2ad0c43346ee48).
The SERAPH checkout was not mounted during development. The following files were
read from their **commit-pinned upstream URLs**, with SHA256 hashes embedded in
`shared/source_audit.py`. `profile` verifies the local checkout's HEAD and these
source hashes when it is present; a mismatching revision/file fails closed.
No import, build, or modification of the official checkout occurs.

* [Binding](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/third_party/torchac_cuda/main.cpp):
  `torchac_cuda.encode_fast_new(cdf, symbols, output_buffer, output_lengths)`;
  `decode_fast_new(cdf, bytestreams, lengths, output)` and
  `decode_fast_prefsum(cdf, packed_stream, cumulative_lengths, output)` accept
  externally supplied CDFs and do not fit them.
* [Encoder kernel](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/third_party/torchac_cuda/torchac_kernel_enc_new.cu):
  CDF is CUDA `int16 [layers, channels, Lp]`, interpreted as unsigned 16-bit
  cumulative counts; final upper bound is implicitly 65536. Symbols are CUDA
  `int8 [layers, tokens, channels]`, read as unsigned bytes. Output buffers are
  `uint8`, lengths `int32`. Requires CUDA and a prebuilt extension; channel/block
  divisibility and token/buffer limits also apply. **`MAX_LP=48`**.
* [Decoder kernel](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/third_party/torchac_cuda/torchac_kernel_dec_new.cu):
  same CDF representation, unsigned output, **`MAX_LP=64`**.
* [High-level encoder](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/lmcache/storage_backend/serde/cachegen_encoder.py):
  `encode_function` calls `calculate_cdf(new_key, ...)` and
  `calculate_cdf(new_value, ...)` on the object being encoded. This path is
  forbidden for C1.5 evaluation and is never called.

The unchanged C1 alphabet has 255 symbols and requires **256 CDF entries**. The
pinned GPU boundary cannot directly encode it. Changing quantization, factoring
symbols into a different alphabet, or patching the CUDA limits would change the
experiment/integration; none is done. The imported official arithmetic-coder path
is therefore **null**, and **no official code is reused**. The selected boundary
is `semcache.experiments.cachegen.shared.core`, a deterministic Python CPU
32-bit integer arithmetic coder with fixed 16-bit cumulative frequencies.
This reference adapter prioritizes correctness over throughput; large-pool
25-pass CPU timing can be slow. Its timing is not an official CacheGen latency.

## Profiles and separation proof

Both profiles aggregate **calibration blocks only**, pooling SNIPS/MultiWOZ and
T=3/T=10. Every captured calibration block contributes once; overlapping tokens
across the two T granularities are counted per block. This fixed weighting is
recorded, not tuned on evaluation. Evaluation encoding does not take a dataset
label. K and V distributions are separate.

| Mode | CDF order | Logical bytes | Serialized bytes |
|---|---|---:|---:|
| SHARED_CDF_GLOBAL | K, V, all 32 layers | 2048 | 2060 |
| SHARED_CDF_LAYERGROUP | early-K/V, middle-K/V, late-K/V | 6144 | 6156 |

Layer groups are `[0,11)`, `[11,22)`, `[22,32)`, explicitly
**REPRODUCTION_CHOICE**. Each CDF contains 256 little-endian uint32 entries.
The 12-byte profile header carries magic/version, mode, layer count, CDF count.
There is no pickle or `.pt` envelope. The logical byte count uses the actual
uint32 serialized representation, including the terminal 65536.

Quantization delegates directly to **C1 `Baseline('UNIFORM_INT8')`**: per
[layer,token] FP32 max-absolute/127 scale (zero -> 1), ties-to-even rounding,
clamp [-127,127], int8, separate K/V scales, FP16 reconstruction. Mapping adds
127, producing [0,254]. Laplace +1 smoothing is **REPRODUCTION_CHOICE**.
Normalization reserves one count for every symbol, apportions the remaining
65536−255 counts in proportion to the smoothed histogram, then distributes
remainders in descending fractional-remainder order (ties by symbol index).
CDFs are strictly increasing; even unseen symbols have nonzero probability.
No smoothing or grouping parameter is exposed for evaluation-time tuning.

Quantization **and reconstruction** now use the device recorded in immutable
`results/cachegen/c1/environment.json` under `benchmark.device`. For the SERAPH
C1 artifacts this is CUDA (`cuda` resolves explicitly to `cuda:0`). FP16 K/V
move to CUDA before the unchanged `Baseline.encode`; only the resulting int8
symbols and FP32 scales move to CPU. Histograms and arithmetic coding stay on
CPU. Decoded symbols and unchanged scales move back to CUDA for `Baseline.decode`,
then final FP16 K/V move to CPU for exact saved-C1 comparison. Device transfers
and quantization/reconstruction remain outside entropy latency timing.

There is no primary CPU fallback when the recorded device is CUDA. Fitting,
benchmark, and quality all use the same resolver and fail if that device is
unavailable. The standalone historical CPU diagnostic is non-primary only.

`fit_profiles` checks nonempty disjoint partitions and invokes the histogram
loader only for `partition == 'calibration'`. The profile stage loads no evaluation
fixture tensors. The model-free loader-spy test proves this control flow.
`profile_manifest.json` records:

* Exact calibration/evaluation ID lists and canonical hashes of the sorted lists;
  exact fitted IDs must equal calibration IDs, with empty intersection with evaluation.
* Actual SHA256 of the five immutable C1 data/quality inputs **and C1 environment**,
  including the literal capture
  manifest file; a separately named canonical JSON hash binds the existing C1
  reconstruction convention.
* Original model/tokenizer metadata, full fixed configuration, calibration
  histogram counts and hashes, profile filenames/binary SHA256, logical/actual
  bytes, and the four C1 quality fixture IDs.
* Source audit, quantizer/codec implementation hashes, frozen-before-evaluation status.
* Schema version 2 `device_contract` and its SHA256: `reference_c1_device`,
  `quantization_device_requested`, `quantization_device_resolved`,
  `reconstruction_device`, `entropy_coder_device`,
  `quantization_device_provenance=MEASURED_C1_REFERENCE_DEVICE`,
  `profile_symbol_generation_device`, and the C1 environment SHA256.
  The same contract is recorded in benchmark/quality stage manifests.
* The proven SERAPH diagnostic summary and diagnostic JSON hash (when present),
  plus the obsolete-profile archive location when replacing old CPU profiles.

The stage manifest binds the profile manifest SHA256. Loading revalidates all
bindings and regenerates CDFs from the recorded calibration counts. CDF tuples
are immutable. Encode/decode have no histogram update API. Benchmark reloads
profile hashes after evaluation and checks each in-memory profile remains byte
identical. Profiles without the device contract are rejected, as are mismatches
between the profile, evaluation, and recorded C1 device. The profile manifest
binds Baseline source hash, capture SHA256, calibration ID hash and device
contract. Refitting a compatible existing profile is refused. Runtime C1
input snapshots are checked before and after each stage.

## Per-block representation and storage accounting

A block contains one stream for each component/group (2 or 6 streams), flattening
[layer,token,hidden] in row-major order. Block local metadata includes:

* 51-byte versioned header: codec mode, dimensions, stream count, profile SHA256;
* one uint32 stream length per arithmetic stream;
* **unchanged FP32 scales**, K then V, 2×32×T×4 bytes;
* 32-byte SHA256 over the complete header, scales, and payload for corruption detection.

Only arithmetic stream bytes count as `encoded_payload_bytes`; all other block
bytes count as `local_metadata_bytes`. `scale_metadata_bytes` is also reported.
Q has no parameter at this codec boundary and is never entropy coded. The profile
is separate, stored once per mode/model configuration. No per-block filename or
experiment JSON audit envelope is charged, consistent with C1's logical storage
scope; the binary codec framing and checksums **are** charged. The profile
manifest contains fitting provenance, not additional decoder state: decoding
requires only the profile `.bin` and block `.bin`.

For the complete evaluation pool, separately for each codec:

```
raw_kv_pool_bytes = sum(raw K + raw V)
encoded_kv_pool_bytes = serialized_shared_profile_bytes + sum(payload + local_metadata)
pool_kv_compression_ratio = raw_kv_pool_bytes / encoded_kv_pool_bytes

raw_semcache_pool_bytes = sum(Q + raw K + raw V)
encoded_semcache_pool_bytes = serialized_shared_profile_bytes + sum(Q + payload + local_metadata)
pool_semcache_compression_ratio = raw_semcache_pool_bytes / encoded_semcache_pool_bytes
```

These are primary ratios. Payload-only KV and total-SemCache ratios are diagnostic
and exclude local/shared metadata. The profile is not charged per dataset, T, or
block. Baseline FP16_RAW/UNIFORM_INT8 rows use C1's logical representation (zero
shared profile bytes) over the **same evaluation pool**; existing artifacts are
not rewritten. Baseline timings are blank, labelled `storage baseline only`.
Provenance labels are `MEASURED`, `MEASURED; REPRODUCTION_CHOICE`, and
`MEASURED_RESEARCH_EXTENSION` for the shared modes respectively.

## Correctness, quality, timing

Every evaluation block is quantized by C1's implementation. Each shared mode
encodes and decodes it, requiring exact symbol and FP32-scale equality and exact
FP16 reconstruction equality with UNIFORM_INT8. The latter also must equal the
**existing C1 reconstruction file**, after checking its SHA256. Any difference
fails closed. Arithmetic corruption, truncation, wrong profile, and extra bytes
are rejected by framing/integrity validation.

Quality reads the same four fixture IDs from C1's completed `c1_quality.csv`;
it does not pick fixtures based on C1.5 results. All three C1 control modes must
be present for SNIPS/MultiWOZ × T=3/10. Quality executes native, raw reuse,
uniform reuse, and both decoded shared reuse paths with original FP16 Q.
It uses `base_projection_path` and the fixed C1 `quality_hits` helper, checking
exact reused/fresh counts. FP16_RAW must remain exactly native; both shared
logits must be `torch.equal` to UNIFORM_INT8. It writes native-relative max-abs,
relative L2, native||candidate affected-suffix mean KL, all-position argmax
agreement, and max-abs vs UNIFORM_INT8 (required zero for shared modes).

Encode and decode use **separate 5-warmup/20-measured passes** with synchronous
CPU wall clocks (`perf_counter_ns`). No CUDA work occurs inside those calls.
Quantization, tensor/byte preparation, reconstruction, file I/O, and additional
correctness comparisons are outside timing. Framing/checksum encode and integrity
validation decode are inside. Each block reports mean, median, p95, device,
method, and repetition counts; all 20 observations are persisted. These are
entropy-stage timings, not an end-to-end cache-service benchmark. No timing
number mixes CUDA events with CPU wall time.

## Diagnostic runtime smoke

Use the corrected CUDA-symbol-derived frozen profiles to estimate CPU runtime before launching the
full benchmark. Do **not** rerun `profile` for this diagnostic:

```bash
conda activate semcache
cd "$SEMCACHE_REPO"
python scripts/40_cachegen_shared_cdf.py benchmark \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --output-dir results/cachegen/c1_5 \
  --smoke --warmup-runs 1 --measured-runs 2
```

`--smoke` chooses the **first evaluation block in capture-manifest order** for
each of SNIPS T=3, SNIPS T=10, MultiWOZ T=3, and MultiWOZ T=10. All four groups
must exist. Calibration blocks are filtered out before selection; compression
outcomes never affect selection. Both GLOBAL and LAYERGROUP run on every selected
block. No fitting, model inference, or model download occurs.

`--output-dir` still identifies the directory containing the frozen profiles.
Smoke results go automatically to its **`smoke/` subdirectory**, even if the
parent already contains a completed primary benchmark. Parent manifests,
profiles, environment, and final CSVs remain unchanged. Existing smoke results
are refused rather than overwritten. Smoke outputs are:

```
results/cachegen/c1_5/smoke/
  manifest.json
  benchmark_manifest.json
  c15_block_raw.csv
  c15_summary.csv
  timing_repeats.json
  environment.json
  run_diagnostics.json
  run_status.json
  bitstreams/SHARED_CDF_GLOBAL/<id-hash>.bin
  bitstreams/SHARED_CDF_LAYERGROUP/<id-hash>.bin
```

All JSON outputs and CSV rows carry `DIAGNOSTIC_ONLY` provenance and
`primary_result_eligible=false` (CSV serializes the boolean as `False`). Binary
streams retain the unchanged codec format; their manifest entries carry the
diagnostic labels. The CSV includes dataset, `token_group_size` (T), block ID,
K/V `symbol_count`, payload/local bytes, separate encode/decode mean/median/p95,
`symbol_roundtrip_exact`, and `uniform_int8_tensor_exact`. The existing exact
checks remain mandatory: decoded symbols/scales equal inputs, reconstruction
equals UNIFORM_INT8, and that baseline equals the existing C1 reconstruction.

`run_diagnostics.json` records `total_command_wall_ms`, spanning CLI dispatch,
input validation, loading, correctness checks, timing passes, output writes,
and final C1 integrity verification. It excludes its own report write and Python
startup/imports. This is scheduling information, **not codec latency**.

Full benchmark defaults remain **5 warmups / 20 measured repetitions**. `--smoke`
does not change them implicitly; the command above explicitly requests 1/2.
`--warmup-runs` accepts zero or more; `--measured-runs` requires at least one.

`--max-blocks-per-group N` is another diagnostic control, selecting at most the
first N evaluation blocks per dataset/T using the same manifest ordering. N must
be positive. When combined with `--smoke`, only N=1 is accepted so smoke always
has exactly four blocks. A limiter or non-default timing counts without `--smoke`
routes results to **`diagnostic/`** and also sets primary eligibility to false.
Do not use these overrides for the final primary benchmark.

## Investigating a C1 reconstruction compatibility failure

The exact C1 reconstruction check is mandatory. A mismatch occurs **before
entropy coding for that block** and is not evidence about the arithmetic coder.
Do not refit profiles, regenerate C1, relax equality, or change quantization to
make this check pass.

Run the diagnostic independently of a failed smoke directory:

```bash
python scripts/40_cachegen_shared_cdf.py diagnose-uniform \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --output-dir results/cachegen/c1_5
```

This historical CPU-path diagnostic scans only the four deterministic smoke candidates and stops at the first
incompatible block. It writes `c1_5/uniform_compatibility_diagnostic.json` before
raising the mismatch. No arithmetic coding, profile fitting, or model inference
runs. The report identifies block/dataset/T, the exact selected reconstruction
entry and path, fixture path, hashes, loaded dictionary keys, and per-component
shape/dtype/device/stride/element count. Saved C1 (A) versus fresh C1.5 (B) includes
exact equality, max/mean absolute difference, unequal count, first unequal index,
and both values at that index.

C1's reconstruction files persist only reconstructed K/V, **not symbols/scales**.
The diagnostic therefore replays the actual C1 `measured_baseline` helper on this
one block (quantization/dequantization only). It uses CPU and, if different, the
C1 benchmark device recorded in `c1/environment.json`; an unavailable recorded
device is reported rather than substituted. It compares replayed symbols/scales
to C1.5, replayed K/V to both saved C1 and C1.5, and explicitly checks swapped
components. The current baseline source path, file/AST hashes, git history,
PyTorch versions, and static layout/axis/casting/rounding paths are included.
Replayed intermediates are clearly labelled as reconstructed evidence, not
historically saved tensors. The diagnostic refuses to overwrite an existing
report; preserve that file before a repeat.

Local source audit: `codecs.py` is unchanged from its original C1 commit
`6dc63ff`. Both paths directly call the same `Baseline` with K then V in
`[L,T,D]`; reduction is over hidden dimension, scales are FP32, and reconstruction
is FP16. The SERAPH diagnostic proved the cause on block
`19874b2e7181f079f989acbf` (SNIPS T=3): old C1.5 CPU execution differed from C1
CUDA near FP32 quantization boundaries. CUDA replay exactly reproduced saved
C1 K/V. Seven K scales and three K symbols differed; the first symbol was at
`[4,2,1626]` (CUDA −64, CPU −63), causing maximum reconstructed difference
0.1015625. Two V scales differed, but V symbols/reconstruction were exact.
This evidence is supplied by the SERAPH diagnostic, not measured locally.
The corrected primary path uses CUDA for both encode and reconstruct. The
real-artifact T=3/T=10 regressions now compare CUDA replay and the corrected
C1.5 path to saved C1 exactly; they skip when artifacts, PyTorch, or required
CUDA are unavailable. No tolerance is introduced.

Smoke now writes `run_status.json` as `INCOMPLETE` before work, `FAILED` on an
exception (also marking `manifest.json` and any benchmark manifest `FAILED`),
and `COMPLETED` only after final input verification and output persistence.
Partial CSVs must not be consumed unless this status is `COMPLETED`. An abrupt
termination leaves `INCOMPLETE`. Existing partial directories are refused.
On a mismatch, smoke also saves `smoke/uniform_compatibility_failure.json`.

## Migrate obsolete CPU profiles and rerun only smoke

Old profiles must be rebuilt from calibration symbols generated on C1's device.
`--replace-incompatible` first resolves/validates that device, then archives old
C1.5 profile binaries, manifests, and any dependent primary result files under
`obsolete_profiles/<unique-id>/`. An `obsolete_manifest.json` marks them
`OBSOLETE_INCOMPATIBLE_DEVICE_CONTRACT`, `DIAGNOSTIC_ONLY`, and
`primary_result_eligible=false`, preserving original file hashes and bytes.
The original `uniform_compatibility_diagnostic.json` stays in place and is also
copied into the archive. Old smoke directories are preserved until the explicit
cleanup below. Compatible new profiles cannot be replaced by this flag.

Run these commands on SERAPH. Only C1.5 calibration profiles are regenerated;
no C0, C1 capture/benchmark/reconstruction, full benchmark, or inference runs:

```bash
conda activate semcache
cd "$SEMCACHE_REPO"
cp -pn -- results/cachegen/c1_5/uniform_compatibility_diagnostic.json \
  results/cachegen/c1_5/uniform_compatibility_diagnostic.before_cuda_profiles.json

python scripts/40_cachegen_shared_cdf.py profile \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --output-dir results/cachegen/c1_5 \
  --cachegen-repo /data/khuss/repos/CacheGen \
  --replace-incompatible

rm -rf -- results/cachegen/c1_5/smoke
python scripts/40_cachegen_shared_cdf.py benchmark \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --output-dir results/cachegen/c1_5 \
  --smoke --warmup-runs 1 --measured-runs 2
```

The rebuilt profiles require CUDA-generated calibration symbols and no
evaluation histogram updates. The smoke still fails closed on any exact-symbol,
scale, or saved-C1 reconstruction mismatch. C1 artifacts are read-only throughout.
The six shared CDF definitions, smoothing, arithmetic coder, and byte accounting
are unchanged; only device-correct calibration symbols change their fitted values.

## SERAPH execution

Use the existing `semcache` environment with its installed torch/models. No
CacheGen environment, build, new dependencies, model download, capture, or C0
rerun is required. Set `SEMCACHE_REPO` to the existing GP_semcache checkout.
The paths below assume the completed C1 reconstruction manifest and fixtures
are under `results/cachegen/c1` as in C1.

```bash
conda activate semcache
cd "$SEMCACHE_REPO"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/40_cachegen_shared_cdf.py profile \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --cachegen-repo /data/khuss/repos/CacheGen \
  --output-dir results/cachegen/c1_5

python scripts/40_cachegen_shared_cdf.py benchmark \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --output-dir results/cachegen/c1_5

python scripts/40_cachegen_shared_cdf.py quality \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --output-dir results/cachegen/c1_5 \
  --device cuda
```

Each command refuses completed stage outputs. If a stage fails after writing
partial files, preserve the failure directory and use a new isolated output
directory for the three-stage run; no implicit overwrite/resume is attempted.
Never use `results/cachegen/c1` as output (the CLI rejects it and aliases).
Only quality loads the already-cached OPT model, with `local_files_only=True`.

Expected outputs:

```
results/cachegen/c1_5/
  manifest.json
  profile_manifest.json
  shared_cdf_global.bin
  shared_cdf_layergroup.bin
  c15_block_raw.csv
  c15_summary.csv
  c15_quality.csv
  environment.json
  benchmark_manifest.json
  timing_repeats.json
  bitstreams/SHARED_CDF_GLOBAL/<id-hash>.bin
  bitstreams/SHARED_CDF_LAYERGROUP/<id-hash>.bin
```

Targeted tests (no model downloads):

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_cachegen_shared_cdf.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_cachegen_storage.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_m9a1_base_profile.py' -v
```

## C1.5-B audit and blockers

The pinned **active** `encode_function` does not choose an anchor, subtract an
anchor, apply a delta transform, or form groups of ten. Its steps are direct
per-vector quantization, self-fitted per-layer/channel CDF construction, and
arithmetic chunks of **up to 256 tokens**. Thus anchor choice, exact delta
semantics, and group-of-ten transform cannot be recovered from this active
boundary. They remain **UNVERIFIED / NOT IMPLEMENTED**, including T=10; no T=3
adaptation is inferred from paper descriptions.

The exact active quantizer is `MAX = bins//2−1`,
`max1 = max(abs(input), hidden_axis)`,
`xq = round(input*(MAX/max1) + MAX).to(int8)` with stored `max1`.
It differs from C1 UNIFORM_INT8 and is not reused in C1.5-A. CDF inputs are these
shifted quantized keys/values directly, not verified anchor/delta residuals.

[Model configuration](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/lmcache/storage_backend/serde/cachegen_basics.py)
chooses model-specific layer cutoffs and bin counts. The LongChat/7B default
uses K cutoffs 10/20/32 with bins 32/16/16, V cutoff 2 with bins 32/16.
70B uses K cutoffs 20/40/80 with bins 32/32/16, V cutoff 20 with bins 32/16.
The Mistral special branch also depends on `QUANT_LEVEL` (1/2/3); its bin settings
vary. OPT is absent. Those integer bins are not proof of the paper's anchor or
delta quantizer settings. C1.5-B needs a separately verified transform source
and reconstruction contract before implementing anything on T=10.

There is no algorithmic blocker to leakage-free C1.5-A with the CPU adapter.
Local execution is blocked by absent completed C1 data and torch/model runtime;
SERAPH can run the commands above. Direct reuse of the pinned CUDA coder is
blocked by alphabet capacity. GPU speedups, scale compression, anchor/delta,
dataset-specific profiles, and evaluation-fitted probabilities are deferred.
