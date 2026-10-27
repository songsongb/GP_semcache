# R1-B3-B3-C1: GEMM shape causal verification

This is an opt-in correctness experiment, classified `ORACLE_DIAGNOSTIC`.
Implementation and CPU control-flow checks are available locally; Torch,
Transformers, PEFT and CUDA are unavailable here, so no new model/GPU result,
causal finding or kernel mechanism is claimed.

The working checkout was inspected: branch `exp/raw-semcache-r1b3`, HEAD
`4fd4f6b` before these additions. Every run records the actual branch, HEAD,
source SHA256, resolved model/tokenizer revision, actual adapter/weight identity,
source/target token identities, GPU and numerical settings. The user-reported
Pilot (first difference at Layer 0 Base Q, exact all-full-shape control, exact
Legacy/Contiguous parity) motivates C1. It does not bind new runs to an imported
Pilot manifest; `historical_pilot_reproduced` is therefore not asserted.

## Existing call sites

`native_request()` invokes native OPT/PEFT on `[1, sequence_length, hidden]` inputs.
In `batched_v2_projection_path()` the projection replacement gathers
`fresh_inputs = hidden_states.index_select(1, fresh_index)` and calls the captured
`native(fresh_inputs)` callable. The unchanged executor then allocates output,
places fresh outputs, and copies/gathers source QKV into reused positions.
Legacy and V1 have the same fresh-native computation boundary. TOTAL QKV is
captured before head split and Q scaling, as in existing R1 code.

C1 records actual base/LoRA A/LoRA B linear input, weight and output shapes,
strides, dtype, device, storage offsets and pointers. Token count alone is not
used as evidence of a GEMM shape. Weight bytes are hashed for Layer 0; all replay
projection parameters are guarded by identity/layout/version stamps. Vanilla
PEFT validation remains in force. GPU backend settings and package versions are
also recorded. These are operation metadata, not a cuBLAS algorithm ID.

No existing 79/80 runner, executor, selector, reducer, cache policy or tolerance
is changed. New files are `scripts/82_verify_r1b3_b3_gemm_shape.py`,
`src/semcache/diagnostics/r1b3_gemm_shape.py`,
`tests/test_r1b3_gemm_shape.py`, and this document.

## Same-input Layer 0 controls

The native Target-Full forward retains only the original Layer-0 Q/K/V input
references on GPU, preserving actual strides and values. The other retained
reference snapshots are on CPU. Source captures are compared directly against
catalog backing and Target-Full QKV; FULL_A/FULL_B repeatability is checked.
The retained Layer-0 input values are checked before/after shape probes.

For each Q/K/V role, the original PEFT module is called with the same weights,
active `user_a` adapter, eval state and hidden values:

| Control | Reference and operation |
|---|---|
| `FULL_NATIVE` | Actual full native PEFT call, records reference components |
| `FULL_NATIVE_REPEAT` | A second identical-shape native call; tests repeatability |
| `SUBSET_NATIVE` | Native call on the exact gathered fresh rows; compares with Full's same positions |
| `FULL_THEN_SELECT` | Another full native call followed by fresh-position selection |
| `SUBSET_ASSEMBLY` | Same hidden input through ordinary existing Contiguous assembly |
| `FULL_SHAPE_ASSEMBLY` | Full native calculation, then fresh selection, through that identical assembly |
| `FP32_FULL_VS_SUBSET` | Functional FP32 full/subset calls on upcast copies of the same stored weights |
| `FP32_REFERENCE_VS_NATIVE_FULL` | FP32 reference versus native full outputs, with explicit metric dtype alignment |

Actual base output, scaled actual LoRA B output and TOTAL output are compared
separately. Base is copied at its child hook **before** PEFT can add delta
in-place. Capturing an alias until parent-forward completion would confuse Base
and Total. LoRA A call geometry is retained as metadata as well.

The FP32 control uses float32 copies of the stored weights, not a separately
loaded original FP32 checkpoint. It temporarily requests highest float32 matmul
precision and disables TF32 for that functional probe, then restores both
settings in `finally`. FP16 reduction policy and model/adapter tensors remain
unchanged. A lower error in this control is precision-path evidence; the result
must be measured and is not assumed to be exact.

All controls use fixed `atol=rtol=1e-3`, recording exact equality, allclose,
max/mean absolute error, relative L2, element-wise violation count/fraction and
first nonzero/first failing coordinates. Positions in subset comparisons are
reported as original absolute token indices.

## Same assembly mechanism

`FULL_SHAPE_ASSEMBLY` does not reimplement gather, scatter or slice assignment.
It calls the existing B2.1 Contiguous context with a one-layer adapter view and
the original source catalog/index/mapping. The QKV source remains actual source
prefill tensors. No Target-Full QKV is admitted to cache. Its selected fresh
outputs pass through the unchanged executor's fresh assembly; reused positions
still come from the same Source QKV and mapping.

The ordinary same-input subset assembly is also compared with the actual
ordinary model's Layer-0 assembled QKV. A mismatch prevents causal attribution.
Mapping dictionaries must match between the two assembly controls. Base/Delta
comparisons cover actual calculated rows; TOTAL assembly compares all rows.

## Layerwise counterfactual replay

The default is ordinary Contiguous plus the first **one** layer using full-native
computation. Explicit `--replay-prefixes 1,2,4,8,all` expands after inspecting the
initial controls. `all` resolves to the loaded model's actual layer count (32
for OPT-2.7B, 12 for OPT-125M). `--skip-replays` runs only Layer-0 controls and
the ordinary baseline. A no-fresh-row case is rejected as inapplicable rather
than inventing a subset comparison; use the standard lengths or partial coverage.

The intervention is scoped, single-threaded PEFT **class-forward interception**:

1. Before entering the unchanged executor, retain the original PEFT class method
   and provide a native callable which can compute full input then return fresh
   positions. There is no extra module-instance `forward` attribute at validation.
2. After executor validation, transient projection pre-hooks observe the actual
   full hidden input. The incoming native subset must equal its exact selection.
3. Controlled layers call the original PEFT forward on full hidden input;
   other layers call it on the normal fresh subset. No PEFT arithmetic is copied
   or altered by this intervention.
4. Existing Mapping, Fresh indices and Contiguous gather/copy execute unchanged.
5. Release each full hidden reference immediately after its native call and
   restore hooks/class forwards in `finally`, including failed forwards.

This is not a production integration or an optimized executor. There is one
scoped diagnostic forward at a time. Only Layer-0 probe inputs remain on GPU
across calls; replay does not accumulate 32 layers' hidden tensors there.

A separate physical-call ledger counts rows actually sent to original PEFT:

```
actual_native_rows = 3 * (controlled_layers * sequence_length
                        + remaining_layers * fresh_tokens)
actual_skipped_rows = 3 * total_layers * sequence_length - actual_native_rows
```

Counts are checked against observed calls for every layer/role. The original
executor audit still reports its logical fresh-subset interface; it cannot see
extra full rows inside the intercepted native callable. That original audit is
preserved and labeled separately. The C1 `actual_skipped_projection_rows` is the
physical count. All-layer full-native replay skips **zero** rows while retaining
source-data reuse coverage. Reusing values and skipping computation are separate.

Each replay records controlled layers, actual rows, original wrapper audit,
actual GEMM metadata, first nonzero and first tolerance violation, strict logit
correctness, max absolute/relative L2 error and last-token argmax/KL metrics.
`diagnostic_request_wall_ms` includes hooks, CPU comparisons and transfers; it
must never be used as serving latency or an optimized-reuse speedup.

## Evidence interpretation

`h1_verification_summary.json` separates:

- **H1-A supported:** complete Full/Subset/Full-Then-Select component records;
  exact repeated Full and Full-Then-Select; observed Subset difference;
  identical input selection and weights; exact Layer-0 model input, source/cache
  and Source/Target QKV controls; stable Full repeatability.
- **H1-B:** full-shape computation through the exact existing assembly agrees
  with Full TOTAL QKV. This tests assembly/copy corruption as the immediate cause.
- **H1-C:** per-component FP32 and stored-precision shape errors, including
  cases where FP32 still differs.
- **H1-D:** Base differs while actual LoRA Delta is exact for that role. This
  attributes the first controlled projection difference to Base, not LoRA.
- **Causal contribution isolated for the fixture:** ordinary model is Strict FAIL,
  same-input subset assembly reproduces ordinary Layer 0, complete all-layer
  audit is enabled, and all-layer full-native replay yields exact Full logits
  with unchanged source/mapping/weights/assembly and zero actual skipped rows.
- **Specific kernel mechanism:** always `NOT_COLLECTED` / false in this runner.
  Shapes, strides and a successful intervention do not identify a particular
  cuBLAS algorithm, Tensor Core reduction mechanism or CUDA kernel.

Partial layer capture limits conclusions to observed layers and prevents the
all-layer causal gate. Layer-0 intervention alone need not eliminate all final
logit error; subsequent subset projections can introduce new differences. The
all-layer intervention isolates the contribution of the family of reduced-shape
native calls, not necessarily an exclusive Layer-0 cause of every downstream
error. Strict correctness and argmax agreement remain distinct results.

## Output

New output directories only; existing paths are refused. Required scalar outputs:

- `manifest.json`
- `shape_control_comparison.csv`
- `base_lora_total_comparison.csv`
- `counterfactual_layer_replay.csv`
- `first_divergence_control.csv`
- `logit_control_comparison.csv`
- `h1_verification_summary.json`

The first two CSVs provide the same detailed component records for shape-oriented
and component-oriented review. Metadata and errors are serialized; no input,
hidden, attention, QKV or weight tensor dumps are written. Model/code/weight/
adapter/source/target/precision/GPU identities accompany controlled conditions.
Runtime/OOM/missing-evidence failures preserve the manifest, traceback and existing
CSV evidence. `execution_status=COMPLETE` and baseline `correctness_status=STRICT_FAIL`
can coexist. A corrected diagnostic replay does not relabel ordinary reuse PASS.

## SERAPH commands

Configure the same offline cache and deterministic CUDA workspace as previous
experiments. No model download is allowed by default; `--allow-download` is an
explicit override. Use the user's existing scheduling process; no Slurm files
are part of this change.

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Smoke, OPT-125M FP32, 32 tokens:

```bash
python scripts/82_verify_r1b3_b3_gemm_shape.py \
  --model facebook/opt-125m --dtype float32 --prompt-lengths 32 \
  --device cuda:0 --cache-device cuda --coverage max --replay-prefixes 1 \
  --output-dir results/r1b3/b3_c1/smoke125m_v1
```

Initial main diagnosis, OPT-2.7B FP16, Layer 0 intervention only:

```bash
python scripts/82_verify_r1b3_b3_gemm_shape.py \
  --model facebook/opt-2.7b --dtype float16 --prompt-lengths 128,256 \
  --device cuda:0 --cache-device cuda --coverage max --replay-prefixes 1 \
  --output-dir results/r1b3/b3_c1/opt27b_layer0_v1
```

After checking the initial shape/assembly evidence, explicit replay expansion:

```bash
python scripts/82_verify_r1b3_b3_gemm_shape.py \
  --model facebook/opt-2.7b --dtype float16 --prompt-lengths 128,256 \
  --device cuda:0 --cache-device cuda --coverage max \
  --replay-prefixes 1,2,4,8,all \
  --output-dir results/r1b3/b3_c1/opt27b_expanded_v1
```

`--audit-layers 0,1` and `--skip-replays` can limit an initial debugging run, but
cannot establish the all-layer gate. Default host reference budget is 1024 MiB
per reference with at most two references during comparisons; allow several GiB
host headroom in addition to model loading. Layer-0 GPU references are small;
source cache and actual attention/output buffers still consume GPU memory.
The existing model-load and per-length memory preflights run. OOM is not silently
handled by reducing length, coverage or precision. FP32 here concerns only the
functional projection reference, not conversion of the entire OPT-2.7B model.

## Local tests

```bash
PYTHONPATH=src:tests python3 -m unittest test_r1b3_gemm_shape -v
PYTHONPATH=src:tests python3 -m unittest \
  test_r1b3_b3_correctness test_r1b3_b3_break_even \
  test_r1b3_b21_mapping test_r1b3_batched_executor test_r1b3_fast_selection \
  test_r1b3_occurrences test_r1b2 test_r1b1_d1 test_r1b1_impact \
  test_m9a2_impact_audit test_m85_alignment test_r1_raw_semcache -q
```

CPU routing fixtures verify exact selected values, native row accounting, input
mismatch rejection, missing evidence, class restoration after exceptions and
parameter-mutation rejection. Evidence tests distinguish supported, causally
isolated and kernel-identified claims. Optional real-Torch tests check component
capture before in-place LoRA addition, actual existing Contiguous assembly,
FP32 controls, CPU/CUDA paths and two-layer full/subset replay alignment with
exact all-layer logits. Missing Torch/CUDA is SKIP, not numerical verification.

Local validation recorded for this change: 146 tests collected, 100 passed and
46 skipped, no failures, including the existing B3 and earlier R1 regressions.
The C1 suite has 10 tests: 7 CPU evidence/routing tests passed and 3 real-Torch
CPU/CUDA/replay tests skipped. Python compilation and CLI help pass. The real
OPT-125M smoke startup attempt exits at missing Torch, with FAILED/UNVERIFIED
and all required output schemas; it is not a completed model smoke. New GPU
Full/Subset, component, counterfactual or kernel findings remain unverified.

Full-Then-Select uses `index_select` on the original computation device, including
component selection before CPU transfer. Recorded `gemm_calls` are observed
PyTorch Linear operand shapes/strides and outputs; they do not reveal the
backend's lowered cuBLAS arguments, padded work or hardware instruction count.
