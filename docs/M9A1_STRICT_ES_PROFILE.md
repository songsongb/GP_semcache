# M9-A.1 strict ES base-only profiling

The primary M9-A comparison now uses **strict-base-only** ES profiles, not PEFT
prefill proxies. No normal M7/M8 inference or timing implementation is changed.
This is a profiling-only addition: no optimization, decode, networking, Cloud,
OPT-6.7B or concurrent-user execution.

## How the profiler works

`scripts/35_profile_m9a1_es_base.py` first invokes the existing M8 runner in a
separate process with explicit M8.5 settings: paper-row reducer, buffered cluster
updates every 100 queries, rho=.8, Lambda=100 and interval PBR=100. It always uses
requested length 32 and verifies **actual token count 32**. The primary model is
OPT-2.7B FP16; OPT-125M is allowed for development smoke. The normal M8 run still
uses its original PEFT model and correctness checks. The driver selects only the
`same_user_exact` native/lookup/physical triplets for system comparison.

The output directory must be new. No old sweep input option exists. A manifest
records the fresh run ID, exact command, artifact hashes and completion state.
If the fresh same-user run fails its existing correctness gate, profiling stops;
old evidence is not substituted and no correctness tolerance is relaxed.

After the PEFT process exits, a bare OPT checkpoint is loaded **local-only** at the
fresh run's resolved model and tokenizer revisions, same dtype/device/eager
attention. No adapter is loaded. The module-tree guard rejects PEFT wrappers,
LoRA A/B modules (even disabled/merged ones), and loaded adapter flags. Plain
Transformer models may have an inactive PEFT mixin method; this alone is not an
active LoRA module. The mixed profiler requires plain `torch.nn.Linear` Q/K/V.

Two measured modes are produced:

- `ES_BASE_NATIVE`: all-token base Q/K/V plus attention, FFN, normalization and
  other ordinary bare-OPT prefill work. No LoRA projection.
- `ES_BASE_SEMCACHE_REUSE`: fresh-row base Q/K/V plus cached rows at exactly the
  physical positions selected by the fresh M8.5 trace. Attention/FFN remain full.

All base timing records carry `provenance=MEASURED` and the specific label
`measurement_label=MEASURED_ES_BASE_ONLY`. No base logits are serialized or compared
as personalized-output evidence; `personalized_output_quality_claimed=false`.

## Cache layout and profiling scope

The dedicated `base_projection_path` mirrors the M8 executor's gather, native
subset projection, output allocation, `index_copy_`, per-hit `.to(output)` and
slice merge. It is a separate profiling context; the PEFT executor is untouched.

The cache is rebuilt from a cold **bare-base** capture each repetition. Compact
per-layer Q/K/V spans are materialized using the existing `CacheEntry.from_tensors`
path onto the M8 storage device. Span lengths, hidden width, dtype, batch-one
layout, storage and transfer path are checked. A single source block remains
shared by repeated references to the same source key. These are latency-only base
payloads rather than personalized total-QKV values. The values need not match the
PEFT cache, but the tensor dimensions and CPU/device movement pattern do.

The original physical span provenance must identify the cold request of the same
user/adapter. Source/destination token tuples must agree; overlaps, stale source
identities, token-hash mismatch or aggregate reused-count mismatch fail. Saved UD
work and network bytes use actual physical spans, never index-HIT counts.

Host wall and CUDA-event timing use the existing M8 timer utilities, eager
attention outputs, inference mode and a completion barrier. The LoRA-absence
checks and profiling-context setup are not optimized away. The base profiler
records the same warmup/repetition counts and discarded-warmup/reconstructed-state
methodology. Each repetition reconstructs the cold → same-user exact precondition
and performs an untimed native reference before the timed target, as M8 does.
Only that two-request precondition is replayed for base timing; later cross-user
and unrelated conditions from the M8 trace are intentionally excluded and this
trace scope is recorded. Cache capture/materialization occurs outside hit-prefill
timing, as cold priming is outside M8's same-user hit interval.

The base computation scope remains a **full OPT prefill**: local embedding/logit
work is retained. This fixes GPU LoRA double-counting but does not implement the
paper's physical UD placement of input/output layers. Real ES/UD execution overlap,
network protocol overhead and deployed device behavior remain outside scope.

## Control-plane separation

Fresh M8.5 supplies these separate CPU wall fields:

| M9-A.1 field | Existing M8 field(s) |
|---|---|
| `semantic_encode_ms` | `semantic_encode_ms` |
| `clustering_ms` | `cluster_assign_update_ms` |
| `subsequence_ms` | `subsequence_extract_ms` |
| `lookup_ms` | `cache_lookup_ms` |
| `hit_selection_ms` | `hit_selection_ms` |
| `attention_impact_ms` | `attention_impact_ms` |
| `policy_ms` | `cache_policy_ms + chu_update_ms + pbr_update_ms` |

Cache materialization is already inside cache-policy timing and is not added
again. The remaining measured host residual is `control_unaccounted_ms`.
`total_control_ms = request_wall_ms - prefill_wall_ms - tokenization_ms`.
Inconsistent overlapping scopes fail. Tokenization is excluded from both modeled
paths. These are control timings from the correctness-validated PEFT run, not from
the latency-only base surrogate, and remain labelled MEASURED with source metadata.

## Strict versus proxy composition

The cost consumer defaults to `--es-compute-policy strict-base-only`.
`require-base-only` remains a backwards-compatible spelling of the same strict
validation, not a weaker legacy escape hatch. It requires the new profile contract,
fresh M8.5 settings/run identity, matching revisions and prompt/repetition identity,
cache placement/dtype, no active LoRA, exact executed fresh/reused counts and
separate control-plane data. Merely adding `es_base_compute_ms` to an old M8 row
no longer qualifies.

```
EdgeLoRA = measured ES_BASE_NATIVE
         + calibrated UD LoRA at full token count
         + simulated full network

SemCache = measured fresh M8.5 control
         + measured ES_BASE_SEMCACHE_REUSE
         + calibrated UD LoRA at fresh token count
         + simulated remaining network
```

UD calibration is included once in each total. Fresh-token calibration is already
for the smaller workload; no second reuse-ratio reduction is applied. Network
savings are derived once from the same physical reused-token count. Base-only
measurement is not additional to PEFT prefill: it **replaces** that ES cost input.
PEFT timings remain source diagnostics, never additive compute components.

Totals remain `SIMULATED` under the four-class provenance scheme, because they
compose measured, calibrated and simulated terms. Strict results additionally
carry `result_label=STRICT_BASE_ONLY_SYSTEM_MODEL`; component ES timing preserves
`MEASURED_ES_BASE_ONLY`. Primary eligibility requires OPT-2.7B FP16 and the fresh correctness gate;
OPT-125M is labelled DEVELOPMENT_SMOKE. Neither claims general safe cross-user
reuse or paper hardware reproduction.

`--es-compute-policy peft-prefill-proxy` remains debugging-only. Both totals and
JSON aggregate metadata explicitly carry
`SIMULATED_PROXY_DOUBLE_COUNTS_LORA`, and `primary_comparison_eligible=false`.
No proxy run can silently become the primary comparison.

## Outputs

Inside the new profile directory:

- `fresh_m85/`: unmodified fresh M8 raw JSONL, timing summaries and environment.
- `es_base_profiles.json`: native/reuse base timings and absence/layout evidence.
- `strict_es_input.jsonl`: paired original M8 records plus validated base-profile
  attachments and separate control fields, consumable by scripts 33 and 34.
- `profile_manifest.json`: fresh invocation, identity/hashes and COMPLETE status.

Attachments preserve `es_base_compute_ms`, `es_base_compute_provenance`,
`es_base_compute_measurement_label`, `es_base_profile`, and the fresh run ID.
The profile records wall/GPU/QKV timings, executed row counts, revisions, dtype,
device, attention implementation, cache storage/transfer/dtype, layout and scope.
No timing values or personalized correctness results are fabricated.

## SERAPH commands (not executed by Codex)

All models, TinyBERT, tokenizers and dependencies must already be provisioned.
There is no download flag. Pick a new output directory for each run.

Small OPT-125M strict-profile smoke (one command; includes fresh M8.5 measurement):

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 scripts/35_profile_m9a1_es_base.py \
  --model facebook/opt-125m --dtype float32 --device cuda \
  --prompt-length 32 --warmup-runs 1 --measured-runs 2 \
  --output-dir results/m9a1/opt125m_strict_smoke
```

Eventual primary OPT-2.7B FP16 length-32 profile:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 scripts/35_profile_m9a1_es_base.py \
  --model facebook/opt-2.7b --dtype float16 --device cuda \
  --prompt-length 32 --warmup-runs 3 --measured-runs 10 \
  --output-dir results/m9a1/opt27b_len32_strict
```

After the strict profile and correctness checks succeed, use its fresh artifact
for exact-length UD calibration and primary composition:

```bash
python3 scripts/33_calibrate_m9a_ud_lora.py \
  --model facebook/opt-2.7b \
  --es-input results/m9a1/opt27b_len32_strict/strict_es_input.jsonl \
  --output results/m9a1/opt27b_len32_strict/ud_lora_calibration.json

python3 scripts/34_run_m9a_single_user_cost.py \
  --model facebook/opt-2.7b \
  --es-input results/m9a1/opt27b_len32_strict/strict_es_input.jsonl \
  --ud-calibration results/m9a1/opt27b_len32_strict/ud_lora_calibration.json \
  --es-compute-policy strict-base-only --bandwidth-mbps 200 500 1000 \
  --output-dir results/m9a1/opt27b_len32_strict/costs
```

## Model-free validation and changes

Tests cover LoRA rejection (including disabled/merged wrappers), fresh-row-only
plain projection dispatch with shape stubs, forward restoration, strict rejection
of PEFT and legacy labels, proxy labelling, measured provenance, exact-once UD/
network accounting, separate control timers, profile identity/layout mismatches,
fresh driver configuration, and unchanged default M7/M8 integration.

Added: `src/semcache/system_cost/es_base_profile.py`, `base_projection.py`,
`base_runtime.py`; `scripts/35_profile_m9a1_es_base.py`;
`tests/test_m9a1_base_profile.py`; this document.

Changed: M9-A `profiles.py`, `model.py`, script 34 and its previous tests;
README and M9-A documentation links. Normal M7/M8 implementation files are unchanged.

Local validation: **25 M9-A/M9-A.1 model-free tests passed**. The M8.5 regression
reported **9 passed, 2 tensor-only skips** because PyTorch is unavailable locally.
Syntax compilation, CLI help and diff checks passed. Neither profiler nor any
model was run, and nothing was downloaded. Actual CUDA/base-profile measurements
remain pending the SERAPH smoke command above.
