# R2-B3.5 Output-matched native Exact Prefix KV validation

Status: **IMPLEMENTED_LOCAL_ONLY; GPU_VALIDATION=NOT_RUN**. Torch, Transformers
and PEFT are unavailable locally. No real matched GPU speedup, native endpoint
equivalence or FP16 correctness result is claimed.

Initial checkout: `exp/raw-semcache-r2`, HEAD
`e3445ebd3a2346d1099537d38258befaa4885240`, initially clean. Frozen B1/B1.5/B2/B3
code, tests, docs and artifacts are preserved. Historical B3 1024-token speedups
remain output-confounded reference evidence, not samples in this experiment.

## Real matched implementation

The existing B2 runtime loads native OPT with synthetic rank-8 QKV LoRA and exact
tokenizer/weight provenance. Capability discovery resolves **actual**
`PEFT.get_base_model()`, `native.get_decoder()` and
`native.get_output_embeddings()` and checks their identity against the runtime
and native aliases. No unadapted model, new decoder, or copied head is created.
The current fixture's head must be native Linear without an adapter; unknown
head adapters, prompt adapters and non-QKV fixture modifications fail closed.

Installed native source is inspected and hashed. This implementation supports
the explicit OPT contract `outputs = self.model.decoder(...)`, followed by
`logits = self.lm_head(outputs[0]).contiguous()` and direct
`CausalLMOutputWithPast` fields. Unknown output casts, hidden transformations or
return contracts are OUTPUT_MATCHED_UNAVAILABLE. The decoder includes native
position embeddings, final LayerNorm, project_out and all adapted projections.
Actual installed compatibility awaits SERAPH Smoke; it is not inferred merely
from the reported version number.

The source contract was also checked against the
[official Transformers v4.57.6 OPT implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/opt/modeling_opt.py).
That upstream source check is separate from installed runtime/model validation.

| Path | Decoder tokens | Actual head tokens | KV length |
|---|---|---|---|
| LEGACY_FULL | L | L | L |
| LEGACY_WARM_NORMAL | S | S | L |
| MATCHED_FULL | L | 1 | L |
| MATCHED_WARM_NORMAL | S | 1 | L |

Matched requests still call the **actual PEFT model wrapper**, preserving its
forward-hook context, active adapter and invocation arguments. For this explicit
opt-in call only, a context replaces that native OPT instance's Forward body
with `actual_decoder(**same_arguments)` → `hidden[:,-1:,:]` →
`actual_head(last_hidden).contiguous()` → native output container. Labels and
unsupported arguments are rejected. The original bound Forward and endpoint
metadata are restored on success or exception. This is runtime routing confined
to the new executor; no frozen source file or Legacy execution is changed.

The inherited B2 Forward is reused intact, including Normal's per-request Cache
materialize and synchronized request timer. A temporary one-row shape-validation
setting reuses its validation only; saved endpoint metadata is explicitly
`matched_last_hidden_head`, not a claim that native logits_to_keep exists.
Scope installation/restoration is outside the B2 request boundary; the matched
decoder/head routing and argument filtering execute inside. Consequently the
ablation includes wrapper-body/GEMM-shape effects, not only head arithmetic.

No all-token Logits are computed then sliced. Untimed hooks prove exactly one
decoder call and one actual head call: matched input [1,1,head_input_dim], output
[1,1,V]. Extra all-token or double head invocations invalidate the path. OPT's
head input dimension may be word_embed_proj_dim; it is read from actual weight
shape rather than assumed from model hidden_size.

## Adapter, Cache and physical audit

Each actual Q/K/V LoRA B module must execute once per layer with the selected
adapter enabled, unmerged and active. The audit records module identities,
scaling, query shape and observed nonzero B contribution. Global base/adapter
weight hashes/configuration are inherited. These are live source/target module
references, not an unadapted decoder shortcut. Hooks are removed even on failure.

B1's actual root IDs, token embedding inputs, absolute learned position indices,
causal masks, layer/QKV/FFN token counts and cache geometry audit are retained.
Full processes L; Warm only S. Source is an independent native Prefix-only
prefill, completed before targets; no Full target slice builds a cache. Canonical
Prefix hashes and source-past preservation are checked; every Normal request
clones its incoming Cache inside the timer, including mutable-container creation.
The source-prefill call counter is bound across paired targets. Final Cache length
is L in every path. No Prepared mode replaces Normal in this study.

## Independent correctness axes

Axis A compares each Legacy final-token output to its matched counterpart, plus
exact final KV and the actual last hidden state entering the head. A decoder
hidden/KV mismatch is structural invalidity and stops the run. If these are exact
but Logits differ, one-token head numerical sensitivity is recorded separately;
strict numerical equivalence is not assumed from mathematical equivalence.

Axis B compares matched Full vs Warm one-token Logits, and separately preserves
Legacy Full[:,P:L,:] vs the entire Warm Suffix. All strict comparisons use
`abs(actual-reference) <= 1e-3 + 1e-3*abs(reference)`. Record max/mean absolute
error, relative L2, max relative error where |reference|>1e-12, violation count,
argmax, KL, and inherited bitwise/value information where supported. Numerical
metrics operate outside timers on already-computed outputs; model precision is
never changed to hide errors. The legacy helper decision is retained separately
where B3's explicit reference-relative predicate replaces it.

OUTPUT_MATCHED_VALID means actual one-token computation and structural/adapter/
Cache validity. Native endpoint equivalence PASS/FAIL is another axis. FP32
Smoke fails if native/matched strict endpoint equivalence fails. In FP16 a
shape-valid numerical endpoint FAIL can remain diagnostic data, but performance
is MATCHED_RESULT_UNRESOLVED for native-equivalent interpretation. Full/Warm
Strict FAIL also remains legitimate and never becomes correctness-qualified
because argmax agrees. Qualification additionally requires the Legacy full-Suffix
diagnostic to pass, so last-token matching cannot hide known errors.

## Timing, sessions and inference

Modes share one model, source snapshot, target IDs, adapter, backend, device and
dtype per condition. Attention OFF/eager/use_cache=True is primary; ON is explicit
and separately approved/aggregated. No backend switch is allowed. Loader-default
metadata in inherited provenance is default-only; per-arm execution audits are
authoritative. B1 namespace's legacy output-attention flag is not reused as proof
of actual returned attention. Each arm constructs its own source.

Same-session pairs contain all four modes in seeded randomized cyclic Latin
blocks. With 20 repeats each mode occupies each slot five times. Latency requires
≥3 warmups per mode and ≥20 measured pairs in each of ≥3 independent processes.
Cases are shuffled by order seed in latency; pilot proceeds in increasing length.
No hooks/profiler run inside minimal timing. Integrity hashes, CPU Logit export,
Cache-content checks, output serialization, source build, and matched context
installation are outside the preserved B2 request boundary. The endpoint is
device Logits and final KV, not CPU response latency or generation TTFT.

Raw outputs record:

- delta_legacy = Legacy Full − Legacy Warm
- delta_matched = Matched Full − Matched Warm (primary)
- delta_difference = delta_legacy − delta_matched
- Full/Warm head-policy effects = Legacy − Matched for that mode
- Matched and Legacy Full/Warm ratios, mode walls, clone bytes, prepare host
  intervals, Logit shapes, memory, source/target identity and all quality axes.

Source Prefill/Snapshot and separate PREPARE_ONLY wall are in source_costs.
Standalone preparation and asynchronous in-request host enqueue intervals must
not be added to enclosing request wall. No source amortization is claimed.

Analysis rejects pilot/profile/partial runs, duplicate processes/IDs, incomplete
pairs, GPU/model/adapter/code/endpoint/condition mismatches and invalid structure.
Session means receive equal weight. Bootstrap resamples session means; no pooled
pair independence is assumed. With three sessions uncertainty is exploratory.
All raw outliers remain. Report session mean/p50/p95/std, win rates, between-
session variance and paired policy sensitivity uncertainty.

Performance labels: MATCHED_NO_OBSERVED_SPEEDUP, MATCHED_POSITIVE_MEAN_ONLY,
MATCHED_REPRODUCED_POSITIVE_DIFFERENCE, MATCHED_RESULT_UNRESOLVED. Reproduced
positive needs ≥3 sessions with every mean positive, positive exploratory lower
CI, verified environment and native endpoint equivalence. Qualification also
requires strict Full/Warm correctness, including Legacy suffix diagnostics.
OUTPUT_MATCHED_VALID alone never qualifies speedup.

If the matched delta disappears, retain the observed endpoint sensitivity without
assigning all historical savings to the head. If it remains positive, report the
matched advantage with numerical/structural qualifications. If it shrinks, report
both deltas and uncertainty without speculative component percentages. Native
equivalence failure or structural invalidity is a distinct limitation. R2-B4
remains responsible for verified amortized system costs.

## Staging, capacity and artifacts

Primary six cases: 128:127,512:511,1024:1023,128:64,512:448,1024:960.
Default pilot four: 128:127,512:511,1024:960,1024:1023. Explicit extra pilot
128:64,512:448 approves the remaining primary cases. Extended adds 256:255 and
256:192 only. No 2048+ or large Cartesian sweep.

Pilot requires a supplied COMPLETE GPU FP32 B3.5 Smoke with native endpoint
equivalence, matching code and software. Passing --smoke-inputs/--pilot-inputs is
the user's explicit selection of reviewed evidence; no automatic GPU launching.
Latency/profile require exact per-case B3.5 pilot approvals under matching
code/model/adapter/UUID/software/seed/endpoint, checked before targets and against
actual source/token identity after independent source construction. B3 pilot
artifacts cannot approve the new executor.

The original conservative model/memory preflight is retained. Memory rejection
is not an empirical OOM claim. Peak memory includes gate/source/request work;
OOM/preflight records case/phase/traceback and continues other capacity cases.
Other structural/runtime errors stop. Cleanup/empty_cache occurs only at failed-
case boundaries outside request timers. Partial pairs/gates remain saved; failed
cases cannot be aggregated as complete latency evidence. Use new output tags.

Artifacts as applicable: manifest, environment, workload_config,
output_matched_capability, model_adapter_audit, endpoint_shape_audit,
native_matched_parity, cache_integrity, forward_skip_audit,
correctness_and_gates, source_costs, capacity_pilot JSON;
latency_raw_pairs.csv/.json, latency_session_summary.csv;
latency_aggregate_summary.json, endpoint_ablation_summary.csv,
output_matched_decision.json and decision_summary.json. Failure records retain
traceback and context. Unsupported capabilities are explicit, never zero timings.
No tensor dump. Aggregate experimental model/dtype/cases come from input manifests,
separately from the analysis-process environment; input files are fingerprinted.

## SERAPH CLI (offline)

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/89_run_r2b35_output_matched.py --phase smoke \
  --model facebook/opt-125m --dtype float32 --cases 32:16,32:31 \
  --device cuda:0 --cache-device cuda:0 --attention-arm off \
  --output-dir results/r2/b35/smoke01

# After reviewing Smoke: default four-case pilot.
python scripts/89_run_r2b35_output_matched.py --phase pilot \
  --model facebook/opt-2.7b --dtype float16 --device cuda:0 --cache-device cuda:0 \
  --smoke-inputs results/r2/b35/smoke01 --attention-arm off \
  --warmup-runs 1 --measured-runs 3 --independent-run-id pilot01 \
  --output-dir results/r2/b35/pilot01

# Remaining primary-case capacity/endpoint controls.
python scripts/89_run_r2b35_output_matched.py --phase pilot \
  --model facebook/opt-2.7b --dtype float16 --cases 128:64,512:448 \
  --smoke-inputs results/r2/b35/smoke01 --device cuda:0 --cache-device cuda:0 \
  --independent-run-id pilot_extra01 --output-dir results/r2/b35/pilot_extra01

# Only after all selected pilot conditions are reviewed and validated.
for run in 1 2 3; do
  python scripts/89_run_r2b35_output_matched.py --phase latency \
    --model facebook/opt-2.7b --dtype float16 --case-group primary \
    --device cuda:0 --cache-device cuda:0 --attention-arm off \
    --pilot-inputs results/r2/b35/pilot01 results/r2/b35/pilot_extra01 \
    --warmup-runs 3 --measured-runs 20 --seed 42 --order-seed "$((3500 + run))" \
    --independent-run-id "run${run}" --output-dir "results/r2/b35/latency_run${run}"
done

python scripts/89_run_r2b35_output_matched.py --phase analyze \
  --inputs results/r2/b35/latency_run1 results/r2/b35/latency_run2 results/r2/b35/latency_run3 \
  --output-dir results/r2/b35/aggregate01

# Optional AFTER latency analysis, one trace request per mode and selected case.
python scripts/89_run_r2b35_output_matched.py --phase profile \
  --model facebook/opt-2.7b --dtype float16 --cases 512:511,1024:1023,1024:960 \
  --device cuda:0 --cache-device cuda:0 --attention-arm off \
  --pilot-inputs results/r2/b35/pilot01 results/r2/b35/pilot_extra01 \
  --output-dir results/r2/b35/profile01
```

If a length fails, choose an explicit validated subset with --cases consistently
in all independent runs. Pin revision/tokenizer-revision when reproducing a
specific checkpoint. Profile is separate: B2's Cache/native-forward clocks,
Decoder and actual LM Head ranges, CPU self costs and correlated CUDA activities
are exported. Kernel sums/event intervals/wall are distinct; nested costs are not
summed. Instrumented result must equal a plain reference. Missing profiler reports
PROFILER_UNAVAILABLE. No kernel mechanism/occupancy or exact causal decomposition
is inferred. GPU monitoring reuses B2 UUID matching with no --id invocation.

## Local verification

```bash
PYTHONPATH=src:tests python3 -m unittest test_r2b35_executor \
  test_r2b35_endpoint_audit test_r2b35_benchmark test_r2b35_analysis \
  test_r2b3_workloads test_r2b3_capacity test_r2b3_analysis test_r2b3_benchmark \
  test_r2b2_benchmark test_r2b2_attention_ablation test_r2b2_analysis -q
python3 scripts/89_run_r2b35_output_matched.py --help
git diff --check
```

Synthetic tests cover pre-head hidden selection, extra/all-token head rejection,
native source contracts, LoRA guards/hook cleanup, restore on exception, frozen
timer/Cache inheritance, four-path order/identity, strict FAIL, session bootstrap,
endpoint/hardware/adapter rejection, OOM partial artifacts and aggregate metadata.
Tiny offline actual OPT/PEFT CPU/CUDA tests are dependency-gated. Fake tensors,
timings and approvals exist only in unit-test temporary fixtures and are never
reported as model/GPU evidence.

Local verification on 2026-10-28: 115 tests across new B3.5 and frozen
B3/B2/B1.5/B1/R2-A suites; 94 passed, 21 skipped, zero failures. New B3.5 has
15 tests: 13 passed and two dependency-gated native model tests skipped.
Syntax compilation, CLI help/argument examples and whitespace checks passed.
GPU model execution, FP16 parity, actual capacity and latency are NOT_RUN.
