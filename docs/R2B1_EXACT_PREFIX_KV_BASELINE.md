# R2-B1 — Exact Prefix K/V Cache baseline

Status at local implementation: **IMPLEMENTED_LOCAL_ONLY**. No Torch,
Transformers or PEFT is installed locally. No GPU correctness, latency or speedup
claim is made. Classification is `EXACT_PREFIX_BASELINE`,
`semantic_reuse_safety_claimed=false`, `cross_user_reuse=false`.

Repository inspection: branch `exp/raw-semcache-r2`, HEAD
`90304ac2c380ed0bca7c94cfbd431b2158a6c8b2`. Both user-supplied documents
`R2-B1_실험명세_2026-10-27.md` and `R2-B1_Codex_구현지시_2026-10-27.md` were read;
the detailed specification governs this implementation. Their files are preserved.

The existing OPT loader, controlled rank-8/alpha-8/dropout-0 PEFT fixture,
`OPTModelAdapter`, tokenizer provenance, deterministic seed utility, memory screen,
raw-only import guard and scalar/report helpers are reused. The SemCache Engine,
semantic encoder, clustering, w3 matching, R1 projection injection/executors and
R2-A profiling are not executed by this baseline. Existing files/results and
Slurm scripts are unchanged. No git push.

## Native execution contracts

Let X contain L integer IDs, P be the exact prefix length and S=L−P.
Batch size is 1 with no padding, `0<=P<L`, and at least one suffix token.

| Path | Actual native call | Input/mask/position/output |
|---|---|---|
| FULL_USE_CACHE | Native PEFT/OPT Forward, no incoming cache | X with L tokens; mask ones(1,L); positions 0…L−1; logits and KV length L |
| PREFIX_SOURCE_PREFILL | Independent native Forward, no incoming cache | X[:P] with P tokens; mask ones(1,P); positions 0…P−1; actual KV length P |
| EXACT_PREFIX_WARM | Native Forward with fresh request-local past_key_values | X[P:] with S tokens; mask ones(1,L); absolute positions P…L−1; logits length S and output KV length L |

All paths set the same synthetic adapter, `eval()`, `torch.inference_mode()`,
`use_cache=True`, `output_attentions=True`, `return_dict=True`, and eager attention.
No generation occurs. The endpoint is pretokenized input to logits, not generated
TTFT. IDs come from the existing deterministic natural-text fixture and must
roundtrip through the same tokenizer to exactly the requested length. BOS/special
IDs are retained inside X and the prefix; prefix/suffix text is never re-tokenized.

Source preparation finishes before any target Full or warm call. The canonical
prefix snapshot is built only from this independent P-token Forward, never from
Full(X) output or a slice of its cache. A second independent prefix Forward with
hooks verifies source processing and cache repeatability. Its diagnostic cost is
reported separately; it is not included in `prefix_prefill_ms`. Both actual source
calls are counted. P=0 creates no prefix source or snapshot, reports no cache hit,
and calls the identical native Full computation for the warm-labelled control.
P=L and S=0 are rejected.

The correct logit alignment is **Full[:,P:L,:] versus Warm[:,0:S,:]**. The last
row of each predicts the next token after X. Source's last logit versus Full at
P−1 is a separate boundary sanity check; it must not be aligned with Warm row 0.

## Installed API compatibility

Local installed OPT/Transformers/PEFT signatures cannot be inspected because the
packages are absent. On SERAPH the runner records the actual package versions,
PEFT wrapper/native OPT/decoder/position-embedding forward signatures, and SHA256
of their installed Python source files. Explicit optional kwargs are sent only
when both the native OPT and decoder signatures support them; generic `**kwargs`
is not treated as proof of native support.

Required native keywords: input_ids, attention_mask, past_key_values, use_cache,
return_dict, output_attentions. The PEFT wrapper must expose those or accept
forwarded kwargs. Supported `position_ids`/`cache_position` get the actual absolute
range. Supported logits_to_keep/num_logits_to_keep options are set to 0 to retain
all needed rows. Actual output shapes are checked even when no such option exists.

Supported cache transports:

1. Native legacy tuple of exactly (K,V) per decoder layer.
2. **Exact DynamicCache class** with working `to_legacy_cache()` and
   `from_legacy_cache()` conversions. Each request uses a new container made from
   independent cloned tensors. Unknown subclasses, static, quantized, offloaded,
   sliding/truncated caches, unavailable converters or unexpected shapes fail
   explicitly as UNSUPPORTED. There is no hidden Full or source-prefill fallback.

Official code references reviewed for the two contracts:
[OPT 4.40](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/opt/modeling_opt.py),
[OPT 4.57](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/opt/modeling_opt.py),
[DynamicCache 4.57](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/cache_utils.py).
These references guide detection; installed signatures and actual Forward audits
remain authoritative. No dependency upgrade or new loader was introduced.

## Namespace, source integrity and mutation

Prefix identity binds model/source ID and resolved commit, full base-parameter
hash, tokenizer ID/resolved revision and BOS/tokenization policy, actual LoRA
parameter hashes and rank/alpha/scaling/dropout configuration, selected adapter,
dtype/device, eager backend, eval/cache/attention flags, position/mask conventions
and exact prefix IDs. Resolved tokenizer provenance is checked through the
existing snapshot-evidence validator. CLI downloads default to disabled.

All model parameters are SHA256-hashed once outside timing and rechecked after
cases; parameter identity/version/shape/stride/device checks precede requests.
Canonical source tensors have independent owned storage and version counters.
Their namespace/layout, pointers, versions and content hashes are checked outside
minimal request timing. Persistent `.data` writes that bypass version counters
are caught by hashes. This is a serial experimental baseline, not a concurrency
or adversarial-memory safety system.

Every warm request clones all canonical K/V tensors and constructs its required
container **inside its timer**, including when the installed legacy tuple API
might permit immutable sharing. The canonical tensors are never passed directly
to the model. Container conversion may allocate additional memory or copy again;
these costs stay inside the timer. `request_clone_bytes` denotes explicit K/V
clone bytes, not a profiler-derived count of internal DynamicCache allocations.
There is no prebuilt cache pool, request-cache reset trick or CPU transfer path.

Each request starts with actual per-layer cache length P and ends with output
length L. An incoming DynamicCache can itself grow to L; that is acceptable for
the isolated request container. The canonical source remains P. Legacy tuple
input can stay P while its returned output cache has L. These states are logged
separately. The returned output cache's first P K/V values must equal the canonical
source bit-for-bit. Repeated warm calls cannot accumulate prior suffixes.

The default cache residency is the model device. Explicit different cache-device
arguments are rejected; CPU→GPU storage/transmission belongs to R2-B4.

## Physical execution audit

Untimed hook passes capture every decoder layer, Q/K/V projection, self-attention,
and both FFN linear inputs. Full must execute each once at L tokens; source at P;
warm at S. FFN implementations may flatten (batch,tokens,hidden) to (tokens,hidden),
so the audit handles both shapes for batch one. Missing/duplicate calls or wrong
lengths are failures, not inferred skips.

Root Forward and token-embedding input counts also verify one S-token call,
without an internal prefix prefill. Three further warm audit calls check repeated
cache lengths, suffix-only execution and exact target output stability.

The actual learned position embedding output is compared **exactly** against
embedding weights indexed at absolute positions P…L−1 plus the installed OPT
embedding offset. This verifies position semantics without assuming RoPE.
Every attention module's actual additive mask must expose all P prior columns,
allow current/earlier suffix columns, and mask future suffix columns. Actual
returned attention tensors must have query extent S and KV extent L at every
layer. Unsupported/missing position or mask evidence fails as UNVERIFIED or
UNSUPPORTED; kwargs alone are insufficient proof.

Counts in `forward_skip_audit.json` are decoder-layer token rows, not GPU FLOPs
or a claimed percentage of total attention operations. Suffix attention still
uses all L keys/values. This baseline skips prefix Layer/FFN execution, in contrast
to R1 QKV-only substitution.

## Correctness and rejection evidence

Fixed strict tolerance: `atol=rtol=1e-3`. Record torch.equal, allclose, max/mean
absolute error, relative L2, violation count/fraction, last-position argmax and KL.
Full repeatability, suffix alignment, final row and prefix-boundary comparisons
remain separate. Strict FAIL and argmax PASS are different outcomes.

A structurally verified FP16 run may finish COMPLETE with STRICT_FAIL; no
roundoff root cause is asserted without further evidence. A FP32 smoke that
strictly fails does not satisfy the B1 correctness gate. Exact P=0 parity and
repeat-target equality are required. Missing cache/position/Forward evidence,
cache mutation or runtime failure cannot become VERIFIED.

Negative artifacts distinguish:

- Actual prefix-token mutation rejected before Forward.
- Actual switch to the other initialized adapter, whose real weights differ,
  rejected before cache use; no cross-user cached Forward is executed.
- Model/revision/base-weight/tokenizer/dtype/device/backend/adapter-configuration
  mismatch tests using metadata substitutes, clearly labelled METADATA_ONLY.
- Incorrect suffix position, suffix-only mask, and non-prefix source-offset
  argument rejection tests. They do not claim another model/device was executed.

## Exploratory timing and accounting

Minimal request wall uses synchronization before/after native Forward and includes
adapter/eval selection, input/mask/position tensor creation, namespace validation,
per-request KV clones/container preparation and complete output cache production.
Input text tokenization, model loading/fixture construction, source prefill,
canonical snapshot creation, expensive integrity/quality checks and CPU logit
transfer are outside the timer. KV preparation host time measures enqueue/container
work; GPU copy completion is charged to the synchronized request wall. It must not
be added to that parent time.

Source cost fields separate native `prefix_prefill_ms`, canonical `snapshot_ms`
(including source clone/metadata/hash construction), and a separately hooked
`diagnostic_prefix_prefill_ms`. The latter is evidence acquisition, not service
source cost. Source model/adapter hashing/initialization is another setup cost.
Prevalidated immutable-source checks outside timing make this a conditional warm
experiment, not a complete cache-serving service benchmark. They can change GPU
clocks/cache state; B2 must study measurement reproducibility independently.

Warmup defaults to 3 and measured to 3, with 3–5 exploratory repeats supported.
Seeded balanced Full/Warm order is saved; an odd count necessarily has one extra
first position. No profiler is enabled and no detailed hook timing enters minimal
samples. `D(F,H)=Full−Warm` is positive when warm is faster. Means/p50/p95/std,
paired deltas/wins and an observed ratio are descriptive only; B1 always records
`speedup_claim_qualified=false`. No R1 `use_cache=False` latency is pooled with
this Full `use_cache=True` baseline. B2/B3 own independent-run inference and
break-even; B4 owns source/E2E/residency costs.

The reused CUDA memory screen budgets attention/activations/logits plus multiple
KV copies and reserve, and the loader has a pre-load weight screen. Estimates
are not guarantees. A 2.7B FP16 model has roughly 5.4 GB of parameter values;
canonical prefix K/V needs `2 * layers * heads * P * head_dim * element_size`
bytes, plus request clones/container temporaries and the output cache. Report
actual allocations/cache bytes from each run; no local RTX 3090 preflight occurred.

## CLI / SERAPH

New directories are mandatory; existing output directories are refused. There is
no resume, broad sweep, profiler trace or Slurm generation. Separate tags preserve
all failures and partial outputs. Explicit small case lists support up to six
cases at lengths 4…256; S=0 and long-context sweeps are rejected in B1.

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/85_run_r2b1_exact_prefix.py --help

# Smoke A/B/C: half prefix, one-token suffix, zero-prefix native control.
python scripts/85_run_r2b1_exact_prefix.py --phase smoke \
  --model facebook/opt-125m --dtype float32 --cases 32:16,32:31,32:0 \
  --adapter user_a --device cuda:0 --cache-device cuda:0 \
  --warmup-runs 3 --measured-runs 3 \
  --output-dir results/r2/b1/smoke125m_01

# Only after GPU Smoke's API/position/cache/Forward/correctness gates pass.
python scripts/85_run_r2b1_exact_prefix.py --phase correctness \
  --model facebook/opt-2.7b --dtype float16 --cases 128:64,128:127 \
  --adapter user_a --device cuda:0 --cache-device cuda:0 \
  --warmup-runs 3 --measured-runs 3 --order-seed 601 \
  --output-dir results/r2/b1/pilot27b_01

# Optional context extension after the preceding cases and memory screen pass.
python scripts/85_run_r2b1_exact_prefix.py --phase correctness \
  --model facebook/opt-2.7b --dtype float16 --total-length 256 --prefix-length 192 \
  --device cuda:0 --cache-device cuda:0 --warmup-runs 3 --measured-runs 3 \
  --output-dir results/r2/b1/optional256_01
```

Pin `--revision` and `--tokenizer-revision` when matching a specific snapshot.
Source/Target in a run always share the one resolved loader instance. Download
requires explicit `--allow-download` (and appropriate HF offline settings changed
by the user). CPU testing can use `--device cpu --cache-device cpu` only with
suitable local model packages/weights.

Artifacts under `results/r2/b1/<tag>/`:

- `manifest.json`: code/environment/actual installed API and source hashes,
  model/tokenizer/adapter/base-weight identity, settings, workload and status.
- `case_summary.csv`, `paired_samples.csv`: only minimal exploratory Full/Warm
  timings; source and clone costs remain explicitly identified.
- `source_costs.json`: independent source, snapshot and diagnostic source costs.
- `correctness.json`: repeat Full, aligned suffix/last row/prefix boundary metrics.
- `cache_integrity.json`: canonical metadata/hashes, before/after cache lengths,
  repeat checks, output-prefix equality and final model-weight integrity.
- `forward_skip_audit.json`: all observed layer/QKV/FFN/mask/position evidence.
- `negative_tests.json`: actual versus metadata-only rejection evidence.
- `decision_summary.json`: execution, structural and strict gates; no B1 speedup
  claim or semantic/cross-user safety claim.
- `failure.json`: failure code/type/context and traceback, when applicable.

Cache shape/value/API failure preserves partial artifacts. Unsupported APIs are
UNSUPPORTED; unavailable local dependencies or runtime exceptions remain FAILED
with UNVERIFIED correctness. A local dependency failure retains
IMPLEMENTED_LOCAL_ONLY. No tensor values, full KV or attention dumps are written.

## Local validation

The real package-dependent API/position/cache and CUDA paths are **not locally
verified**. CPU utility tests cover namespace rejection, fake mutable containers,
snapshot pollution, independent request copies, missing physical evidence, logit
offset, pairing sign, failure/collision handling and strict-failure preservation.
Actual Torch cloning tests and offline tiny random OPT/PEFT integration tests are
provided but SKIP without dependencies. Those tiny model tests construct a random
OPT configuration locally and use the same controlled LoRA fixture; they do not
download a checkpoint or claim equivalence to OPT-125M/2.7B GPU evidence.

```bash
PYTHONPATH=src:tests python3 -m unittest test_r2b1_prefix_cache \
  test_r2b1_prefix_runtime -v

PYTHONPATH=src:tests python3 -m unittest test_r2b1_prefix_cache \
  test_r2b1_prefix_runtime test_r2a_overhead_audit test_r2a_runtime test_r2a_profiler \
  test_r1b3_gemm_shape test_r1b3_b3_correctness test_r1b3_b3_break_even \
  test_r1b3_b21_mapping test_r1b3_batched_executor test_r1b3_fast_selection \
  test_r1b3_occurrences test_r1b2 test_r1b1_d1 test_r1b1_impact \
  test_m9a2_impact_audit test_m85_alignment test_r1_raw_semcache -q
```

Recorded local verification: **17 new tests collected, 13 passed, 4 skipped**;
combined R1/R2-A/A5/B1 regression: **191 collected, 135 passed, 56 skipped,
zero failures**. `py_compile` and CLI help passed. Pytest itself is not installed,
so those regressions ran with `unittest`; no pytest run is claimed.

The actual local Smoke CLI failed before model execution at `import torch`.
`results/r2/b1/local_validation_20261027_final/` contains its required schemas,
`failure.json`, `IMPLEMENTED_LOCAL_ONLY / FAILED / UNVERIFIED` manifest and the
seven implementation-file SHA256 values. These are local failure/format-validation
artifacts, not successful model/GPU measurements. Existing output directories
were preserved.
