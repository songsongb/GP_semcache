# C1.5-B: CacheGen anchor/delta audit and proposed ablations

Audit date: 2026-09-27. **Design only: no B experiment implemented or run.**
C1.5-A is complete according to the supplied primary result and remains frozen.
This audit does not relabel its smoke results or rerun its evaluation.

The main finding is that the pinned public artifact has **no active anchor/delta
transform**. Its implemented pipeline quantizes raw K/V, derives CDFs from the
object being encoded, and arithmetic-codes those symbols. An OPT anchor/delta
codec based on this audit would be a **CacheGen-inspired RESEARCH_EXTENSION**.

## Evidence and verification boundary

The requested `/data/khuss/repos/CacheGen` does not exist in this environment.
Read-only attempts to run `remote -v`, `rev-parse HEAD`, and
`status --porcelain=v1` all failed with exit 128. Therefore the actual SERAPH
remote, revision, and clean/dirty state are **unknown**, not verified clean.

Instead, this audit downloaded the official upstream archive at
[`6bed34ca9d495289955ed754cf2ad0c43346ee48`](https://github.com/UChi-JCL/CacheGen/commit/6bed34ca9d495289955ed754cf2ad0c43346ee48)
and its commit/recursive-tree metadata. The tree response was not truncated.
All **76 archived file blobs** match their pinned Git blob SHA-1 values; SHA-256
hashes are recorded in [source_audit.json](../results/cachegen/c1_5b/source_audit.json).
The five hashes from the earlier C1.5-A source audit also match. No official
module was imported, extension built, or checkout changed. These checks prove
which upstream bytes were read; they do not prove which extension is installed
or imported on SERAPH.

Before an experiment, record on SERAPH:

```bash
git -C /data/khuss/repos/CacheGen remote -v
git -C /data/khuss/repos/CacheGen rev-parse HEAD
git -C /data/khuss/repos/CacheGen status --porcelain=v1
```

Compare the HEAD and relevant file hashes with the audit JSON. Any local
changes must be identified explicitly rather than attributed to the pinned
upstream implementation.

## Paper contract

The [published paper, §5.2 and Figure 6](https://cs.stanford.edu/~keithw/sigcomm2024/sigcomm24-final1571-acmpaginated.pdf#page=6)
defines ten-token contiguous groups: the first token is independently encoded;
all others reference that same anchor. Transformation precedes quantization.
Anchors use 8-bit quantization; deltas use vectorwise quantization with coarser
bins through early/middle/late layer thirds. Probability models are offline,
layer × channel, and separate anchors from deltas. Appendix C lists 0.5/1/1.5.

The reviewed text does not settle subtraction sign, original versus reconstructed
anchor prediction, arithmetic dtype, tail padding, or an executable normalization
formula for those fractional bins. These remain reproduction decisions.

## Paper/source behavior matrix

P1–P5 refer to the paper claims in `source_audit.json`; source references below
are pinned to the audited revision. `PAPER_DEFINED` classifies those paper
claims; the table classifies their relationship to this artifact.

| Behavior | Classification | Audited implementation |
|---|---|---|
| P1: ten-token transform groups | PAPER_ONLY_NOT_FOUND_IN_SOURCE | Actual input T; arithmetic substreams up to 256 tokens. |
| P1: independently encoded first-token anchor | PAPER_ONLY_NOT_FOUND_IN_SOURCE | No token-zero special case. |
| P2: same-anchor residual transform | PAPER_ONLY_NOT_FOUND_IN_SOURCE | No inter-token subtraction or reconstruction addition. |
| P3: special anchor precision | OFFICIAL_SOURCE_DIFFERENT | Layer-selected quantization applies to every token. |
| P3: delta-layer quantization rules | OFFICIAL_SOURCE_DIFFERENT | Raw K has three configured ranges; raw V has two. |
| P4: fractional bin defaults | PAPER_ONLY_NOT_FOUND_IN_SOURCE | Integer `bins` with no established fractional mapping. |
| Integer quantizer/symbol generation | OFFICIAL_SOURCE_IMPLEMENTED | Per-token/channel-vector maxabs normalization and shifted integer codes. |
| P5: layer × channel distributions | OFFICIAL_SOURCE_IMPLEMENTED | Separate K/V histograms; no channel pooling. |
| P5: anchor/delta probability separation | PAPER_ONLY_NOT_FOUND_IN_SOURCE | No token-role split. |
| P5: reusable offline distributions | OFFICIAL_SOURCE_DIFFERENT | Fit on each serialized object's tokens; stored inside its output. |
| Layer boundaries and bin schedules | MODEL_SPECIFIC | Named model table; no OPT-2.7B entry. |
| Delta sign, predictor precision, short groups, OPT split | REPRODUCTION_CHOICE_REQUIRED | No executable official anchor/delta contract to reuse. |

This negative finding uses caller-to-decoder inspection, not only keyword
search. A recursive source search found no `anchor` and only an unrelated
streaming API `delta.content`. The inactive `CacheGenEncoderImpl` also directly
quantizes raw K/V. Old CUDA files are not the selected build sources and do not
supply an alternative anchor transform.

## Active source path and exact tensor semantics

The official drivers call `CacheGenSerializer.to_bytes` directly:
[`run_cachegen.py:43–56`](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/run_cachegen.py#L43).
The serialization and inverse paths are:

```text
CacheGenSerializer.to_bytes
  -> HuggingFace layout permutation
  -> encode_function -> _split_kv
  -> torch_quant_vectorized, independently K and V
  -> calculate_cdf, independently K and V
  -> concatenate K/V layers
  -> encode_ntokens -> encode_fast_new -> encode_cuda_new
  -> CacheGenGPUEncoderOutput.to_bytes

CacheGenDeserializer.from_bytes
  -> decode_function_gpu -> decode_chunk -> decode_fast_prefsum
  -> do_dequantize, independently K and V
  -> reshape / format permutation / output dtype conversion
```

Evidence: [encoder](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/lmcache/storage_backend/serde/cachegen_encoder.py#L247),
[decoder](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/lmcache/storage_backend/serde/cachegen_decoder.py#L144),
[bindings](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/third_party/torchac_cuda/main.cpp#L11),
and [selected CUDA sources](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/third_party/torchac_cuda/setup.py#L8).

| Boundary | Shape and axes |
|---|---|
| HuggingFace input | `[L,2,H,T,Dh]`; no batch dimension here |
| Normalized / vLLM input | `[L,2,T,H,Dh]` |
| Separate K and V | `[L,T,C]`, `C=H*Dh`, flattened channel `c=h*Dh+d` |
| Arithmetic symbols | `[2L,T,C]`; all K layers, then all V layers |
| CDF | `[2L,C,Bmax+1]`; component maximum bin parameter |
| Maxabs metadata | Separate K/V `[L,T,1]` tensors |

Thus token/layer/channel axes at the arithmetic boundary are 1/0/2.
There is **no source delta sign**, no before/after-quantization delta operation,
and no separate anchor quantizer for either K or V.

The serializer handles a last arithmetic segment shorter than 256 using actual
`ntokens`, with no token padding. Zero-filled output-byte buffers are scratch
space, not padded K/V. Caller behavior is not uniform:
[`CacheGenEngine.chunk_kv`](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/src/cachegen_interface.py#L10)
uses floor division at its default 1024-token chunk size and omits a remainder;
[`LMCacheEngine._chunk_kv`](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/lmcache/cache_engine.py#L146)
and `run_adaptation.py` iterate slices that can include a short tail. None of
these behaviors establishes a final short **anchor group** policy.

## What the implemented bins and CDFs mean

For each K/V layer `l`, token `t`, and channel vector `x`, the active
[`torch_quant_vectorized`](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/lmcache/storage_backend/serde/cachegen_encoder.py#L42)
implements:

```text
C = floor(b_l / 2) - 1
m = max_channel(abs(x))
q = int8(round(x * (C / m) + C))
x_hat = (float(q) - C) / C * m
```

Here `b_l` is a nominal integer bin-count/range parameter, `m/C` is the
reconstruction step, and `C` is the code for zero. For even `b`, the nominal
produced range is `0..b-2`: 11, 15 and 31 values for b=12,16,32. Int8 is the
container; it does not imply 8-bit anchor precision. There is no explicit clamp
or zero-max guard. `m` inherits input dtype; the bin tensors normally use FP32.
The exact shifted-rounding expression matters: the older signed helper is not
a substitute for the active numerical path at rounding boundaries.

The model table in
[`CacheGenConfig.from_model_name`](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/lmcache/storage_backend/serde/cachegen_basics.py#L27)
uses:

| Branch | K boundaries / bins | V boundaries / bins |
|---|---|---|
| Mistral-7B, QUANT_LEVEL=1 | 0/10/20/32; 16/16/12 | 0/2/32; 16/12 |
| Mistral-7B, QUANT_LEVEL=2 | 0/10/20/32; 32/16/16 | 0/2/32; 32/16 |
| Mistral-7B, QUANT_LEVEL=3 | 0/10/20/32; 32/32/32 | 0/2/32; 32/32 |
| Other listed 7B fallback, including LongChat | 0/10/20/32; 32/16/16 | 0/2/32; 32/16 |
| LongAlpaca-70B | 0/20/40/80; 32/32/16 | 0/20/80; 32/16 |

`QUANT_LEVEL` selects a schedule; it is not the bin value. There is no supported
OPT-2.7B branch. In particular, the V rules and 70B K boundaries should not be
silently described as equal layer thirds.

**No source evidence establishes 0.5/1.0/1.5 as bits, absolute steps,
multiplicative width factors, or equivalents of these integer schedules.**
Using fractional values in this implementation's `bins` formula would not
reproduce the proposed paper quantizer. B3 needs further authoritative evidence
or an explicitly chosen OPT quantizer, with separate quality evaluation.

[`calculate_cdf`](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/LMCache/third_party/torchac_cuda/cal_cdf.cu#L11)
counts symbols across all object tokens for each layer/channel before arithmetic
stream subdivision. K/V are fit separately, then concatenated, not pooled.
For `B=max(layer bins)` in that component and cumulative count `H_i`:

```text
cdf[i] = floor((65535-B) * H_i / T) + i
```

The kernel stores unsigned 16-bit values in an int16 tensor. Its terminal
entry is 65535; arithmetic coding uses an implicit upper bound of 65536 for
the last symbol. Histogram counters are also uint16; long-object limits were
not runtime-tested here. The CDF is serialized in each object, along with
K/V maxabs tensors, head shape, byte lengths and actual subchunk token counts.
The decoder additionally relies on matching model/QUANT_LEVEL configuration
and format; the output object does not serialize a complete bin schedule.

The pinned CUDA encoder permits `MAX_LP=48`, the decoder 64, and CDF generation
requires `max_bins<64`. Therefore C1.5-A's 255-symbol/256-entry CDF is not directly
compatible with those kernels. Reusing our deterministic CPU coder is an
explicit research adaptation. No external offline probability table or
anchor/delta probability selector appears in the active source.

## Option 1, Option 2, and a strict lossless alternative

**Option 1 is technically coherent as a new transform-domain lossy codec.**
For a chosen sign, write `a=x_0`, `d_i=x_i-a`. Applying the existing C1 quantizer
`Q` gives reconstruction approximately `Q(a)+Q(d_i)`, generally different from
`Q(x_i)`. Anchor error is shared across nine reconstructions. The quantizer
algorithm and per-vector axis can remain the same while the scales, symbols
and overall error distribution change. This is not C1 reconstruction parity.

There is also a concrete input-contract issue: C1.5-A's `quantize` helper
requires CPU FP16 inputs before transfer to CUDA. A FP32 residual must either
be rounded to FP16, introducing a transform-boundary rounding step, or use a
new declared input contract. FP16 subtraction can overflow even when both
inputs are finite. Sign, subtraction/addition dtype, predictor choice, overflow
handling and final rounding must all be frozen. A full-precision delta should
not be assumed to remain exactly invertible after finite-precision subtraction
and addition. This is design reasoning from the
[local C1 implementation](../src/semcache/experiments/cachegen/codecs.py), not a
claim of measured OPT behavior.

**Option 2 is also coherent after those choices are specified**, but the pinned
source supplies no complete official anchor/delta implementation to invoke.
Changing transform, quantizer and channel-layer probability model together
would obscure attribution. Any quality-equivalence claim needs a separate
quality study; this requirement applies to Option 1 as well as B3.

For initial isolation, recommend a **lossless symbol-domain arm**:

1. Quantize original K/V on CUDA exactly as C1; keep every FP32 token scale.
2. Treat token 0's int8 symbols as anchors. Ordinary signed residuals require
   a wide integer type and lie in `[-254,254]`: 509 symbols, incompatible with
   the unchanged 255-symbol format.
3. As an explicit alternative, use centered modulo-255 residuals:
   `d_i=((s_i-s_0+127) mod 255)-127`. Recover
   `s_i=((s_0+d_i+127) mod 255)-127`. Both stay in `[-127,127]`.
4. Restore original symbols before the existing CUDA reconstruction; require
   exact equality to saved C1 and preserve Q as FP16.

The modulo representation is a **proposed RESEARCH_EXTENSION**, not an official
CacheGen definition or an implemented experiment. It can reuse the arithmetic
primitive/alphabet but needs its own transform format identity and B profile
contract. Keeping scale metadata for all ten tokens is mandatory. Because
C1 scales vary per token, symbol differences are not floating KV differences;
both must be measured separately. No compression benefit follows merely from
smaller raw delta magnitudes, especially after independent maxabs normalization.

## Proposed staged study

The machine-readable plan is
[design_manifest.json](../results/cachegen/c1_5b/design_manifest.json).
All B1/B2/B3 statuses are NOT_IMPLEMENTED/NOT_RUN.

| Stage | Scope | Stop/go rule |
|---|---|---|
| B0 | This paper/source audit; bind SERAPH checkout and completed A artifacts before execution. | Do not claim local verification or fabricate missing hashes. |
| B1 | T=10 only. Raw-value locality plus existing-C1-symbol entropy diagnostics; no new lossy quantization. | Positive calibration-held-out whole-group savings estimate after anchors and overhead, with exact inverse checks. Small raw residuals alone are insufficient. |
| B2 | If justified, lossless anchor-relative symbol transform and fixed shared coding. | Actual complete-pool bytes improve versus the matched raw control; all exactness checks pass. Report each frozen A T=10 comparison separately. |
| B3 | Optional transform-domain/layer-wise lossy quantization with registered quality thresholds. | Only if justified and semantics resolved. No assumed fractional-bin equivalence; no automatic need for this stage. |

B1 should compute raw same-anchor differences in float64 analysis arithmetic
from captured finite FP16, separately for K/V, layer and distance 1–9. Compare
RMS/variance, absolute quantiles and zero fractions with the **matching
non-anchor original positions**, not all ten originals. Such statistics are
locality diagnostics, not measured storage or continuous-variable Shannon
entropy. If binned histograms are used, fix boundaries/overflow handling on
calibration and report their diagnostic role.

In parallel, replay unchanged C1 CUDA quantization and examine signed and
modulo residual histograms, zero/tail/wrap fractions, and empirical entropy.
Use calibration-only, source-query/dialogue-disjoint held-out estimates to
choose the eventual representation. Include anchors with weight 1/10 and
residuals with weight 9/10, plus unchanged scale and profile/header costs.
Marginal entropy is not joint entropy: a reversible transform preserves joint
information but can improve a restricted entropy model. Freeze choices before
evaluation; do not repeatedly select variants on the 236 evaluation blocks.

B2 should initially hold probability-model capacity at GLOBAL: separate K/V,
all layers and token roles pooled. Add a new **raw T=10-calibration-fit control**
alongside the transformed T=10-calibration-fit arm. This controls for A having
used pooled T=3+T=10 calibration. New B profiles may eventually be fit only in
the B output area; A profiles must never be refit. Actual serialized profile
bytes are charged once in each independent T=10 pool.

Separate anchor/delta CDFs would add another modeling change. If investigated
later, pair a raw anchor/non-anchor role-split control with the transformed
role-split arm (four CDFs total, K/V separate). Do not attribute gains from extra
CDFs to the transform alone. Channel-layer CDFs and physical-page changes are
outside initial B1/B2. B3 should likewise separate the transform-only lossy
arm from additional layer-wise quantization, with probability pooling held
fixed. Its future quality study should include tensor distortion, logit/task
metrics and preregistered acceptance limits; none is run here.

## Population, frozen controls, and missing local evidence

Each T=10 fixture supplies one complete contiguous group with anchor index 0.
Verify absolute positions are contiguous and never join different fixtures,
queries or partitions. This avoids an unproven short-group rule. Initial B
excludes T=3. Later logical reuse width 3 with physical pages of at least ten
is a separate research extension, not part of this plan.

The expected evaluation population is **236 T=10 blocks**, both datasets,
in frozen manifest order. Verify that count and hash the actual IDs on SERAPH;
calibration count/IDs must also come from the frozen manifest. No inference,
recapture or 1,308-block benchmark is needed for B1/B2. Captured base-projection
OPT tensors and short queries are a different workload from a generic model's
runtime KV cache, so transfer of the paper's locality benefit is a hypothesis.

Required comparison: C1 UNIFORM_INT8, A SHARED_CDF_GLOBAL,
A SHARED_CDF_LAYERGROUP, and the B variant, **all on the same T=10 IDs**.
Read A's `full_storage/c15_full_summary.csv` rows with `stratum=T=10`, and
reconcile against its per-block CSV. Charge one complete profile per T=10 pool,
as A's summary does; do not compare different amortization populations. B2's
raw control additionally separates transform gains from changed calibration
weighting. Report SNIPS/T10 and MultiWOZ/T10 as well as ALL_T10.

The user-supplied completion counts, exactness and overall ratios are recorded
as USER_REPORTED_COMPLETED_PRIMARY_RESULT, not locally verified measurements.
This checkout has only placeholder C1 files and a C1.5-A README; the completed
A profiles/manifests/CSVs are not mounted. Consequently **completed-reference
hashes and exact T=10 ratios are null**, explicitly gated before B execution.
Available placeholder hashes are stored separately and are not accepted as
completed-baseline evidence. No T=10 ratio is inferred from the mixed-T ratios.

B2 primary metrics are actual KV/SemCache pool bytes and ratios, payload-only
ratio, additional savings against each T=10 control, exact symbols/scales and
saved-C1 reconstruction, coverage and zero failures. Profile/header/scale
costs must all be included. Operational timing, if recorded later, is not a
primary latency result. Negative or null compression results must be reported.

Only audit/design artifacts and their Git ignore exceptions were created.
No model inference/download, CDF fitting, tensor capture, experiment codec,
C1/C1.5-A benchmark, GlobalQKVCache change, C2, or B experiment was performed.
