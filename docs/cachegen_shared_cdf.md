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

`fit_profiles` checks nonempty disjoint partitions and invokes the histogram
loader only for `partition == 'calibration'`. The profile stage loads no evaluation
fixture tensors. The model-free loader-spy test proves this control flow.
`profile_manifest.json` records:

* Exact calibration/evaluation ID lists and canonical hashes of the sorted lists;
  exact fitted IDs must equal calibration IDs, with empty intersection with evaluation.
* Actual SHA256 of all five immutable C1 inputs, including the literal capture
  manifest file; a separately named canonical JSON hash binds the existing C1
  reconstruction convention.
* Original model/tokenizer metadata, full fixed configuration, calibration
  histogram counts and hashes, profile filenames/binary SHA256, logical/actual
  bytes, and the four C1 quality fixture IDs.
* Source audit, quantizer/codec implementation hashes, frozen-before-evaluation status.

The stage manifest binds the profile manifest SHA256. Loading revalidates all
bindings and regenerates CDFs from the recorded calibration counts. CDF tuples
are immutable. Encode/decode have no histogram update API. Benchmark reloads
profile hashes after evaluation and checks each in-memory profile remains byte
identical. Refitting/overwriting an existing profile is refused. Runtime C1
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
