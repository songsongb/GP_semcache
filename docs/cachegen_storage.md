# C0/C1 offline CacheGen storage feasibility

Implementation is in `src/semcache/experiments/cachegen/{common,codecs,harness}.py`,
with entry point `scripts/39_cachegen_storage.py` and model-free tests in
`tests/test_cachegen_storage.py`. No production cache files or dependencies change.
No measurements, models, downloads, package installations, or CUDA builds were
performed while implementing this harness.

## Compatibility finding and limits

The reference inspected is the official
[UChi-JCL/CacheGen artifact](https://github.com/UChi-JCL/CacheGen), `main`, inspected
2026-09-25. This is a source-level finding, not a pinned local runtime validation.
Every runtime audit records the actual checkout HEAD, dirty state, source hashes,
import paths, Python, torch, CUDA, GPU, and prebuilt `torchac_cuda` availability.
A missing checkout/runtime is **UNKNOWN**, not proof of codec compatibility.

* [`cachegen_basics.py`](https://github.com/UChi-JCL/CacheGen/blob/main/LMCache/lmcache/storage_backend/serde/cachegen_basics.py)
  uses a model-name whitelist in `CacheGenConfig.from_model_name`; OPT-2.7B is absent.
  Direct OPT support is **UNSUPPORTED in this reviewed artifact**. Supplying another
  model's name for OPT would be an undocumented adaptation, which this harness
  does not do. Integer layer bin profiles are model-specific.
* [`cachegen_encoder.py`](https://github.com/UChi-JCL/CacheGen/blob/main/LMCache/lmcache/storage_backend/serde/cachegen_encoder.py)
  accepts Hugging Face `[L,2,H,T,Dh]`, converts to `[L,2,T,H,Dh]`, and derives head
  dimensions from the tensor. The active path directly quantizes vectors and
  calls `calculate_cdf` on the current K and V symbols. It does not consume a
  calibration-only CDF. Consequently **C1 CACHEGEN_FULL is unavailable** under
  the required no-evaluation-fitting policy, independently of the OPT lookup.
  Python-level dimensional generality does not prove CUDA kernel support for
  OPT's 80-wide heads or three-token blocks.
* [`cachegen_decoder.py`](https://github.com/UChi-JCL/CacheGen/blob/main/LMCache/lmcache/storage_backend/serde/cachegen_decoder.py)
  reconstructs with stored head dimensions and its model-selected bins; it uses
  CUDA and returns FP16 in Hugging Face format. Its exposed symbol decode helper
  allows an exact arithmetic-symbol check separately from reconstructed FP16 error.

No verified anchor/delta quantize-only boundary is present in that active path:
**CACHEGEN_QUANT is unavailable**. The paper concepts (10-token groups, 8-bit
anchor, layer groups, 0.5/1.0/1.5 settings) are recorded as PAPER_DEFINED requested
parameters, **not asserted to be applied**. Integer artifact bins must not be
relabeled as those paper settings. A newer LMCache API or calibration-profile
interface needs a reviewed adapter; changing a report flag cannot enable it.
There is no codec reimplementation, vendored source, model-name substitution,
CDF monkey patch, padding, or invented CacheGen quantization stage.

C0 defaults to analytic synthetic K/V with *supported LongChat model metadata*
(32 layers, 4096 channels, 32 heads). It loads no model. This validates the codec
independently; these synthetic measurements are never C1 OPT results. Its T=3 and
T=10 cases do not claim anchor/delta behavior. C0 may self-fit its synthetic CDF;
C1 evaluation may not. `--codec-model facebook/opt-2.7b --hidden-dim 2560` is an
explicit negative compatibility probe and is expected to fail on this artifact.

## Exact SERAPH commands (future operator execution only)

Set these site-specific values before running the commands. `OPT_REVISION` must
be a commit already cached on SERAPH; no command below downloads a model. Source
paths are local raw SNIPS and MultiWOZ files/directories accepted by the existing
`dataset_adapters` loader. The entire corpus can be read for sampling, but only
128 calibration and 128 evaluation queries per dataset are executed.

```bash
export SEMCACHE_REPO=/path/on/SERAPH/SemCache
export CACHEGEN_REPO=/path/on/SERAPH/CacheGen
export SNIPS_SOURCE=/path/on/SERAPH/2017-06-custom-intent-engines
export MULTIWOZ_SOURCE=/path/on/SERAPH/multiwoz/data.json
export OPT_REVISION=REPLACE_WITH_LOCALLY_CACHED_OPT_COMMIT
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Bootstrap the **separate future cachegen environment** using the official
[installation procedure](https://github.com/UChi-JCL/CacheGen#installation).
These commands install packages and compile extensions; they were **not run by
Codex**. The historical environment may require SERAPH CUDA/compiler adjustments;
its resolution/build is an outstanding runtime prerequisite, not a tested claim.
Clone at a selected commit for reproducibility (replace the placeholder), then
record that commit. Do not install this codec into `semcache`.

```bash
git clone https://github.com/UChi-JCL/CacheGen.git "$CACHEGEN_REPO"
git -C "$CACHEGEN_REPO" checkout REPLACE_WITH_SELECTED_CACHEGEN_COMMIT
git -C "$CACHEGEN_REPO" rev-parse HEAD
conda env create -f "$CACHEGEN_REPO/env.yaml"
conda activate cachegen
python -m pip install -e "$CACHEGEN_REPO/LMCache"
cd "$CACHEGEN_REPO/LMCache/third_party/torchac_cuda"
python setup.py install
python -c 'import torchac; import torchac_cuda; import torch; print(torch.__version__, torch.version.cuda)'
cd "$SEMCACHE_REPO"
```

The official encoder also imports `torchac`, which may JIT-build its CPU extension
on first import; the bootstrap import above prepares it. The harness issues no
install/build commands and requires an already-built `torchac_cuda` binary.

C0 audit and smoke (five warmups, twenty measured repeats per case, each timing
protocol; artifacts in `results/cachegen/c0`):

```bash
conda activate cachegen
cd "$SEMCACHE_REPO"
python scripts/39_cachegen_storage.py c0 --cachegen-repo "$CACHEGEN_REPO" --audit-only
python scripts/39_cachegen_storage.py c0 --cachegen-repo "$CACHEGEN_REPO" --tokens 3 10
```

C1 capture, using the **existing semcache environment** and existing
`qkv_capture` hooks. No new projection instrumentation is introduced. Local-only
model loading is hardwired; there is no `--allow-download` option.

```bash
conda activate semcache
cd "$SEMCACHE_REPO"
python scripts/39_cachegen_storage.py capture \
  --snips "$SNIPS_SOURCE" --snips-split train_full \
  --multiwoz "$MULTIWOZ_SOURCE" --multiwoz-split train \
  --revision "$OPT_REVISION" --tokenizer-revision "$OPT_REVISION" \
  --queries-per-split 128 --seed 42 --tokens 3 10 --device cuda
```

Append `--plan-only` to validate sources and save the sampling plan without
loading a model. Default capture: all 32 layers, FP16, at most 32 input tokens,
nonoverlapping complete groups at starts 0,T,2T,... for each T. Short queries are
recorded as skipped for that T; incomplete tails are excluded, never padded.
The same query can contribute to both granularities. MultiWOZ dialogue groups
are shuffled with seed 42 and consumed by one partition only. Partial dialogue
groups at each 128-query boundary are not reassigned to the other partition.
Partition hashes cover identities and query content; fixture SHA256 covers each
`.pt`. Tensor files contain only named Q/K/V tensors, read with
`torch.load(weights_only=True)`; all experiment metadata is JSON.

C1 benchmark in **cachegen**, after copying the capture directory if necessary:

```bash
conda activate cachegen
cd "$SEMCACHE_REPO"
python scripts/39_cachegen_storage.py benchmark \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --cachegen-repo "$CACHEGEN_REPO" --device cuda \
  --modes FP16_RAW UNIFORM_INT8 CACHEGEN_QUANT CACHEGEN_FULL
```

This computes FP16/INT8 results and writes explicit UNAVAILABLE rows for both
CacheGen modes with this artifact. It does not try to encode evaluation data
with a self-fitted CDF. Unknown external versions also fail closed. Calibration
fixtures are retained for a future *official* calibration/profile interface.
`--device cpu` permits the baselines in any already-provisioned PyTorch runtime.
No CacheGen package is needed for either baseline.

C1 quality back in **semcache**, using the reconstructed K/V `.pt` files plus
JSON produced by the benchmark (no codec import needed):

```bash
conda activate semcache
cd "$SEMCACHE_REPO"
python scripts/39_cachegen_storage.py quality \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --reconstruction-manifest results/cachegen/c1/reconstruction_manifest.json \
  --device cuda \
  --output-dir results/cachegen/c1
```

Quality uses the existing M9-A `base_projection_path` and `materialize_hits` to
physically reuse the same query's captured projections at the same absolute
positions. It compares native, raw QKV reuse, and original Q plus reconstructed
K/V reuse. Default selection is the first evaluation block for each dataset/T;
`--all-blocks` expands it. No trained/personalized LoRA quality claim is made.
This command reruns only quality against the existing capture and reconstruction
artifacts; do not rerun C0, capture, or benchmark for a quality harness fix.
The C1 helper normalizes JSON token-ID lists to tuples before materialization
and requires `component_scope='base_qkv_latency_only'`, which the base runtime
materializer already supplies. Previously, the list in `Subsequence` compared
unequal to the tuple in `CacheEntry`, triggering the cached-span validation error.
A single partial span is valid: every projection must reuse exactly T rows and
compute the remaining query_length−T rows natively. FP16_RAW−NATIVE measures
physical reuse error; UNIFORM_INT8−FP16_RAW isolates compression error, with
captured FP16 Q unchanged and only K/V reconstructed.
The uncompressed control must pass existing M8 `exact_parity_passed` thresholds
(max abs 1e-5, relative L2 1e-6, last-position KL 1e-8, last argmax equality).
Failure is persisted as REFERENCE_FAILED; compressed quality is then unavailable
rather than attributed to compression. Physical subset GEMMs can themselves
change numerics, which this control explicitly detects. It does not relax the
existing tolerances. This remains an offline fixture experiment, with no global
cache admission or production behavior change.

## Accounting, timing, and outputs

Q always stays FP16. For T=3: Q=491,520, KV=983,040, QKV=1,474,560 bytes.
KV ratio is raw KV / encoded KV including codec metadata. Whole-block ratio is
raw QKV / (raw Q + encoded KV including metadata). They are distinct.

UNIFORM_INT8 is REPRODUCTION_BASELINE: independently for each `[layer,token]`
vector and each K/V, compute FP32 `scale=max(abs(x))/127`, replacing zero scale
with 1; ties-to-even round `x/scale`, clamp [-127,127], store int8 symbols and one
FP32 scale. Reconstruct and cast to FP16. For T=3, symbols occupy 491,520 bytes
and scales 768 bytes. No probability fitting occurs. Baseline sizes describe
the logical codec representation; `.pt` interchange framing and shared JSON
experiment metadata are not per-block codec overhead. Shape and dtype are fixed
by that shared schema. Official bytes include all actual serialization overhead:
arithmetic payload is summed stream bytes; everything else in `len(encoded)` is
metadata, including CDFs, scales, lengths, shape, and serialization framing.
The official codec's internally pickled bytes are treated as its opaque storage
payload, never as our experiment metadata or a cross-environment fixture format.

Timing uses `perf_counter_ns` with CUDA synchronization for wall measurements.
CUDA Events are collected in a **separate** pass; event-instrumented wall values
are discarded. Five warmups precede twenty measured repeats in each pass.
Both clocks include the harness's codec layout conversion/copy costs; neither
includes errors, symbol validation, disk writes, or model execution. CUDA event
elapsed time spans host-launch gaps and serialization in the encode/decode region;
it is not a sum of isolated kernel times. C1 block timing is the mean of repeats;
summary percentiles are across block means. Individual repeats are retained in
`timing_repeats.json`. Baseline timing covers tensor encode/decode, not `.pt` I/O;
C0 includes official byte serialization/deserialization.

`symbol_roundtrip_exact` is independently tested when the exposed official symbol
helpers are available. Null means unobservable/not applicable, never success.
K/V maximum absolute error, reference-relative L2, and cosine are separate from
symbol exactness. FP16 reconstruction is potentially **lossy**.

Generated outputs (all commands support `--output-dir`):

* C0: `c0_codec_smoke.csv`, `compatibility.json`, `environment.json`, `manifest.json`,
  plus the final encoded `.bin` per case.
* C1: `capture_manifest.json`, `c1_block_raw.csv`, `c1_summary.csv`, `c1_quality.csv`,
  `environment.json`, `manifest.json`; additionally `compatibility.json`,
  `reconstruction_manifest.json`, `timing_repeats.json`, fixture/reconstruction `.pt`.

Unavailable modes retain one row per block with empty measurements and a reason;
summary aggregates only MEASURED rows. No invented zero-byte success. Quality KL
is native || candidate averaged over the affected suffix; argmax agreement is the
fraction across all input positions. `compression_only_logit_delta` is maximum
absolute difference from raw reuse logits, while raw reference error is vs native.
`anchor_count` and `delta_token_count` are null for baselines/unavailable stages.
T=3 is RESEARCH_EXTENSION, T=10 PAPER_REFERENCE_GRANULARITY (not an exact paper
reproduction claim), OPT adaptation REPRODUCTION_CHOICE, observed results MEASURED.
The initialized result CSVs are header-only and marked NOT_RUN/AUDIT_ONLY until
real execution; no compression benefit or quality result is presently measured.

## Tests and remaining integration gates

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_cachegen_storage.py -v
PYTHONPATH=src python3 -m unittest discover -s tests -p test_m85_alignment.py -v
PYTHONPATH=src python3 -m unittest discover -s tests -p test_m9a1_base_profile.py -v
```

Tests are model-free. Optional PyTorch tests use only literal tensors; absence of
PyTorch is an explicit skip, never an installation trigger. Physical-cache
integration remains blocked on a working official runtime, verified OPT metadata
support, a compliant calibration-only probability interface, actual C0/C1
measurements, strict uncompressed parity, compressed logit quality, and a positive
whole-block storage/latency result. Cache hit-rate changes are outside this work.
