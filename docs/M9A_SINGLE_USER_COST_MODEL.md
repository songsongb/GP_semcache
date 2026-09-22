# M9-A single-user UD–ES cost model

M9-A composes existing M8 prefill measurements, a CPU-only synthetic LoRA
calibration, and analytical communication. It is an **additive, single-request
system model**, not measured end-to-end distributed latency. M7/M8 inference,
cache policies, reuse decisions and timing instrumentation are unchanged.

M8.5 closure is user-reported SERAPH evidence: 11/11 alignment tests, 36 focused
M8/M8.5 tests, and 225 passed / 9 skipped / 1 warning in the full suite. Those
counts are not new local execution evidence for M9-A.

## Equations and accounting

```
T_edge_lora = T_ES_base + T_UD_lora + T_network
T_semcache  = T_semcache_control + T_ES_remaining + T_UD_remaining + T_network_remaining
system_delta_ms = T_edge_lora - T_semcache
compute_saved_ms = (T_ES_base + T_UD_lora) - (T_ES_remaining + T_UD_remaining)
communication_saved_ms = T_network - T_network_remaining
reuse_overhead_ms = T_semcache_control
system_delta_ms = compute_saved_ms + communication_saved_ms - reuse_overhead_ms
```

Positive delta favors reuse; negative favors recomputation. `REUSE` is selected
only when `T_semcache < T_edge_lora`; ties select `RECOMPUTE`. This diagnostic is
labelled **RESEARCH_EXTENSION**, never applied to M7/M8 behavior.

The serialized sum is the M9-A requested abstraction. It does not implement the
paper's per-layer `max(ES, UD+network)` overlap equation. Only skipped projection
rows reduce modeled UD work and projection traffic. Attention/FFN savings are
not inferred from hit counts. ES costs come from whole-prefill profiles, including
any observed slowdown. Cache merge/device transfers remain inside the physical
ES prefill term and are not added a second time as overhead.

Control is the measured residual:

`physical.request_wall_ms - physical.prefill_wall_ms - physical.tokenization_ms`.

This captures control/cache-policy work and unaccounted host overhead without
summing inclusive nested M8 timers. Tokenization is excluded from both paths.
TinyBERT and attention-impact processing remain in this residual. Native
non-tokenization host overhead outside prefill is excluded from the baseline;
this and UD input/output compute omissions make the comparison a scoped model,
not a prediction of complete deployed query latency.

## Provenance and the ES measurement boundary

Each latency component uses exactly one class: `MEASURED`, `CALIBRATED`,
`SIMULATED`, or `PAPER_REFERENCE`. Every CSV latency has a companion
`*_provenance`; JSON components include source and dependency provenance. Derived
system totals and deltas are **SIMULATED**, even when some terms are measured.

| Quantity | Classification |
|---|---|
| Imported M8 wall/GPU/QKV timings | MEASURED, source M8 row |
| SemCache control residual | MEASURED, arithmetic on exclusive measured intervals |
| Exact-length CPU LoRA timing × layer count | CALIBRATED, explicit layer extrapolation |
| Zero fresh-token LoRA cost | SIMULATED zero, no calibration sample fabricated |
| Network byte-derived latency | SIMULATED |
| Additive totals, savings and system deltas | SIMULATED with dependency labels |
| A100 80GB / Xeon Gold 6338; four-core 2.3 GHz UD; 8 GiB; 200 Mbps | PAPER_REFERENCE configuration only |

**M8 does not measure ES base-only compute.** Its PEFT prefill includes both base
and GPU LoRA execution, and physical reuse timing additionally includes fetch/
merge overhead. QKV timing likewise combines base, LoRA and mixed-path work.
There is no justified subtraction to isolate base latency from these artifacts.

The default `--es-compute-policy require-base-only` therefore requires explicit
profile-extension fields on native and physical rows:

```
es_base_compute_ms: <independently measured value>
es_base_compute_provenance: MEASURED
es_base_compute_scope: prefill_base_only_excluding_lora_control
```

M9-A supplies the consumer interface, not a new profiler or invented measurements.
An existing M8-only artifact will fail this default rather than pretend to contain
base-only timing.

To explore existing M8 results, opt into **`--es-compute-policy peft-prefill-proxy`**.
This uses M8 `prefill_wall_ms` unchanged as each modeled ES term. Original timing
is retained as MEASURED in `es_source_timings`, but the interpreted base/remaining
term is labelled **SIMULATED / REPRODUCTION_CHOICE**, with MEASURED dependency.
**This proxy retains GPU LoRA cost while adding UD LoRA cost**, and retains local
input/output-layer work. It is not a faithful placement measurement or an unbiased
EdgeLoRA latency estimate. No GPU LoRA fraction is guessed from FLOPs.

Actual GPU/hostname come from the M8 artifact. RTX 3090 / SERAPH host CPU is recorded
separately as the user-reported current setup; missing hardware information is not
filled with A100/Xeon reference data. The composition host is not the measured ES.
No frequency scaling from SERAPH to a hypothetical 2.3 GHz device is performed.

## UD CPU calibration

`scripts/33_calibrate_m9a_ud_lora.py` reads an explicit local OPT `config.json`, or
an already-cached Hugging Face config. It does not invoke Transformers, load model
weights, or access the network. Only OPT-2.7B (primary) and OPT-125M (development)
are allowed. Config hidden width/layer count must agree with the selected model.

The benchmark uses CPU FP32 synthetic tensors and a seeded CPU generator. Each
sample executes Q, K and V sequentially, each as `(x @ A) @ B`, rank 8: six matrix
multiplications for one layer. Input and parameters are created outside timing;
matmul output allocation is included. There is no training, dropout, activation
transfer, or base projection. `torch.set_num_threads(4)` is applied and affinity
is restricted to at most four available logical CPUs before resizing the worker
pool when the OS supports it. Original thread/affinity settings are restored even
on failure. Affinity success, selected IDs and errors are recorded.

Metadata includes model ID, dimensions, rank, sequence lengths, actual hostname,
CPU model, `/proc/cpuinfo` frequency observations where available, warmup/measured
counts, all raw milliseconds, mean/p50/p95, seed, dtype and torch version.
`provenance=CALIBRATED` is the required class; the more specific
`calibration_provenance=CALIBRATED_ON_SERAPH_CPU` records the operator-declared
host label (override `--host-label` on another machine). Actual hostname and CPU
remain recorded independently. This is not the paper's UD hardware.

Per-request UD projection cost is the exact token-count mean × transformer layer
count, labelled CALIBRATED with REPRODUCTION_CHOICE extrapolation. Remaining LoRA
uses a separate calibration at `prompt_tokens - reused_tokens`. No linear
interpolation or ratio scaling is performed. Missing lengths fail with a clear
error. Zero fresh rows yield SIMULATED zero without timing an empty matmul.

`--es-input` derives all required positive full/fresh lengths from M8 rows before
calibration, including rows that may later fail correctness gating. This length
collection does not authorize reuse or consume their timing as trusted data.

## Network accounting

`network_ms = transferred_bytes * 8 / (bandwidth_mbps * 1_000_000) * 1000`.

Default is 200 Mbps; 500 and 1000 Mbps and other positive values are supported.
There are no sockets, sleeps or communication emulation.

For batch one, fresh rows `m = n - reused`, hidden width `d`, and explicit wire
widths `b_h` and `b_delta`, each layer transfers:

- ES → UD hidden states: `m*d*b_h` bytes.
- UD → ES Q/K/V LoRA deltas: `3*m*d*b_delta` bytes.

Multiply both by the actual config layer count for per-request projection traffic.
Default boundary policy additionally transfers `n*d*b_h` bytes once in each
direction: initial h0 to ES and full final hL to UD. Boundaries do not shrink with
projection reuse. Full hL versus last-token-only output, redundant first-layer
exchange, and wire precision are **REPRODUCTION_CHOICE** interpretations, not
claimed paper implementation details. `--projection-exchange-only` explicitly
omits both boundaries for an Eq.17-style projection-only comparison.

The default hidden wire width follows M8 activation dtype; delta width defaults
to that same width. Override `--delta-element-bytes 4` if the artifact used FP32
LoRA deltas with FP16 hidden states. CPU calibration remains FP32 regardless of
wire precision; cast costs are not included. M8 `saved_communication_bytes`, when
present, must agree exactly with the dimension-derived projection savings; a
mismatch fails rather than changing assumptions silently. Input M8 estimated
network milliseconds are not used.

JSON reports layer, boundary and request `ud_to_es_bytes`, `es_to_ud_bytes`,
`total_network_bytes`, and the corresponding directional/total milliseconds.
CSV prefixes request fields with `edge_` and `semcache_`. Headers, RTT, protocol,
serialization, contention, duplex overlap and transport buffers are excluded.

## Logical UD memory

Capacity is exactly `8 * 1024**3` bytes. No capacity-sized allocation is performed.
The estimated FP32 payload contains:

- All-layer Q/K/V adapter A/B parameters: `6*L*d*r*parameter_bytes`.
- One-layer input tensor `n*d`, three output tensors `3*n*d`, and three low-rank
  intermediates `3*n*r`, each multiplied by temporary element bytes.
- Optional future user-local cache payload (default zero; no cache implementation).

Report usage and `fits`; do not automatically change the reuse decision. This is
not whole-device peak RSS: embedding/output weight tables, logits, runtime/OS,
tokenizer, allocator overhead and wire-cast buffers are excluded explicitly.

## Profile pairing and correctness gate

Accept raw JSONL, JSON row arrays, a single raw row, or a JSON object with `rows`.
Aggregated summary statistics are not a substitute for per-request timings.

Pair native, lookup-no-reuse and physical (or position-aligned diagnostic) rows by
experiment, model, query, user/adapter, prompt hash/count, repetition, dtype,
revisions, seed, attention implementation and recorded hardware. Policy profiles
must agree between lookup and reuse rows. Missing/duplicate pairs, inconsistent
token counts and non-MEASURED ES rows fail. Lookup QKV/prefill timing is retained
for diagnostics, but is not substituted silently for the native EdgeLoRA baseline.

Reuse requires affirmative correctness evidence. Reject `not_validated`/failed/
invalid statuses, `controlled_exact_parity_passed=false`, and missing affirmative
validation. Even a true parity flag cannot overrule a `not_validated` status.
`--allow-invalid-reuse` is an explicit override, labelled
`UNSAFE_REUSE_EXPLORATION`, `invalid_reuse_override=true` and
`correctness_gate_passed=false`. Rejected rows are not silently discarded.
A passed gate establishes only that fixture's recorded validation;
`safe_reuse_claimed=false` remains in all results.

## Commands for SERAPH (not executed by Codex)

Use the existing SERAPH environment with PyTorch and provisioned model config.
Point `--es-input` to one existing OPT-2.7B M8/M8.5 **raw** result file. The short
same-user exact fixture is the first comparison; correctness-invalid long rows
will be rejected. No new M8 run is required by these commands.

```bash
python3 scripts/33_calibrate_m9a_ud_lora.py \
  --model facebook/opt-2.7b \
  --es-input results/m8/inference_raw.jsonl \
  --warmup-count 5 --measured-count 20 \
  --output results/m9a/ud_lora_calibration.json

python3 scripts/34_run_m9a_single_user_cost.py \
  --model facebook/opt-2.7b \
  --es-input results/m8/inference_raw.jsonl \
  --ud-calibration results/m9a/ud_lora_calibration.json \
  --query-id same_user_exact \
  --es-compute-policy peft-prefill-proxy \
  --bandwidth-mbps 200 500 1000 \
  --output-dir results/m9a
```

Add `--model-config /path/to/local/config.json` to both commands if cache resolution
is unavailable or ambiguous. There is intentionally no download flag. For a
125M sanity run, use `--model facebook/opt-125m` and matching 125M artifacts/config;
do not pair different model calibrations/profiles.

Outputs are `ud_lora_calibration.json`, `single_user_cost_breakdown.csv`,
`m9a_environment.json`, and `m9a_summary.json`. One cost row represents one measured
request repetition at one bandwidth; no aggregate replaces raw request evidence.
Input hashes, selected settings, provenance and all reproduction choices are kept.
The CLI validates every comparison before writing output files.

## Tests and scope limits

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_m9a_cost_model.py -v
```

Tests use hand-checkable numbers, temporary local fixture JSON, and stub resource
controls. They do not import/run torch or any LLM. They cover units/directions,
capacity, provenance, calibration matching, strict versus proxy ES interpretation,
correctness rejection/override, identities, signs/ties, output serialization and
absence of M7/M8 mutation. No production result or CPU calibration is fabricated.

Still excluded: 50-user concurrency, full autoregressive decode, real distributed
networking, OPT-6.7B, Cloud placement and performance optimization. Trained
user-specific LoRA inference and paper-scale reproduction also remain unimplemented.

## Change inventory and local verification

Added:

- `src/semcache/system_cost/__init__.py`
- `src/semcache/system_cost/common.py`
- `src/semcache/system_cost/network.py`
- `src/semcache/system_cost/memory.py`
- `src/semcache/system_cost/calibration.py`
- `src/semcache/system_cost/profiles.py`
- `src/semcache/system_cost/model.py`
- `scripts/33_calibrate_m9a_ud_lora.py`
- `scripts/34_run_m9a_single_user_cost.py`
- `tests/test_m9a_cost_model.py`
- `docs/M9A_SINGLE_USER_COST_MODEL.md`

Changed: `README.md` links this milestone; `docs/M85_PAPER_ALIGNMENT.md` records
the user-reported SERAPH closure. No M7/M8 implementation file changed.

Local verification: **14 M9-A tests passed**. The M8.5 model-free regression had
**9 passed, 2 tensor-only skips** because local PyTorch is absent. CLI help,
syntax compilation and `git diff --check` passed. No full-suite run, CPU timing
calibration, model execution or download occurred locally. Temporary fixture
outputs used in CLI tests were not written as production results.
