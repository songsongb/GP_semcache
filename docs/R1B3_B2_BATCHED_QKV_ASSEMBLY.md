# R1-B3-B2 — Batched QKV assembly (opt-in ORACLE diagnostic)

Status: implementation and offline validation only. Local Torch, Transformers,
PEFT and CUDA execution are unavailable. No new GPU latency, tensor/model parity,
or speedup result is claimed. Production policy/executor and B1 selector are
unchanged. This is **not** production-policy integration or a FP16 accuracy fix.

## Baseline and scope

`mixed_projection_path()` in `edgelora/mixed_projection.py` constructs the reused
mask and fresh positions, validates PEFT projections, patches instance forwards,
and calls native PEFT once on the fresh subset per layer/role (zero native calls
when every row is reused). It allocates the output, scatters fresh rows, then
resolves, converts, and assigns one cached tensor per window. Passive captures
remain on device. Instance forwards are restored in `finally`.

At 256 tokens / MAX, 85 windows × 32 layers × 3 roles gives **8,160 window
iterations**, conditional on that actual coverage. `.to(output)` is a no-op when
device/dtype already match. These iterations are not 8,160 H2D copies or a measured
kernel count. Repeated Python resolution, slice construction, launches, and copies
are bottleneck candidates; attribution requires the new GPU run.

User-provided B1 minimal reference (not reproduced locally): 128/256-token FULL
81.89/81.01 ms, Fast selection 9.97/22.19 ms, warm request 188.12/242.23 ms, coverage
126/128 and 255/256. Strict FULL parity failed there. The new runner compares
**Fast + original executor** to **Fast + batched executor** directly; it does not
compare against B3-A's slower legacy selector.

## Implementation and safety

New independent files:

- `src/semcache/edgelora/batched_mixed_projection.py`
- `scripts/76_benchmark_r1b3_batched_executor.py`
- `tests/test_r1b3_batched_executor.py`
- this document

The runner reuses B1's source-engine cloning and checkpoints, B3-A's completed
source capture/catalog and logit metrics, and R1's row audit, RAW-only guard,
preflight and synchronized request timing. Neither existing runner is modified.

`PositionMapping` receives **already selected** Fast hits. It validates catalog
membership, source query identity, token IDs, bounds, disjoint target positions,
absolute alignment, exact causal prefix, adapter, namespace and weights. It
uses the original Fast index's owner/view/metadata guard, including version and
`.data` rebinding detection. No hash substitutes for prefix equality. No target
FULL QKV enters the catalog. The catalog/index keep backing owners alive.

Selected source and destination positions are flattened into first-seen source
query groups. Selected window order and coverage are not changed. Source query
references point to the original full QKV backing, not concatenated window copies.
Each layer/role executes:

1. Same sorted fresh positions and same native PEFT subset shape as Legacy.
2. Allocate the same output shape/dtype; scatter native fresh rows.
3. For each backing group, gather source rows with `index_select(1, source_index)`.
4. Convert gathered rows with `.to(output)`, then `index_copy_` to target positions.

Indices are reused across layers/roles by group/device for this request. Multiple
source queries are safely grouped; strided backing uses PyTorch's index operations.
Invalid/stale backing fails closed, rather than silently reverting or selecting a
new candidate. Unsupported custom fresh-projection callbacks are rejected.
There is no separate fallback needed for valid multiple/strided backing. Quantized
or incompatible source metadata fails the existing catalog/index checks. CPU
cache uses CPU gather then transfer of the gathered rows; GPU cache gathers on GPU.
The runner supports CPU or the model's CUDA device, not a multi-GPU timing claim.

Mapping creation includes a Fast snapshot guard; the executor checks the snapshot
again to support callers retaining plans. These target-side checks remain in warm
latency; their repeated O(entries × layers × roles) metadata work is **not** hidden
in source build. No Source Index or selector algorithm optimization is introduced.
No full-shape cached output is precomputed. No precision/LoRA algorithm change.

## Timing and memory definitions

`--executor legacy|batched|both` defaults to **legacy**. Fast selection is fixed.
`--instrumentation minimal|breakdown|both` defaults to **minimal**.

- Minimal uses the original R1 synchronized request wall boundary. Source build,
  catalog/index build, independent policy snapshot cloning and correctness checks
  are outside target timing. All target selection, mapping, index transfer,
  executor validation/patching, forward and original policy work are inside it.
- Minimal has no internal CUDA timing events or Python tracer. Small scalar counters,
  selection/mapping/setup host timestamps remain. Unknown internal times are null.
- Breakdown observes the **unchanged** original forward using a scoped Python line
  tracer and uses scoped spans in the new executor. No replica baseline algorithm.
  The tracer is removed before subsequent requests; an existing tracer is rejected.
  This is deliberately expensive, asymmetric diagnostic instrumentation. Always
  use same-executor breakdown-minus-minimal request deltas to inspect its overhead.
- Both use CPU `perf_counter_ns`. CUDA spans record events on the model device's
  current stream; they resolve only after the enclosing request has synchronized.
  No per-window `synchronize()` is added. Pageable CPU transfer can itself block.
  CUDA event elapsed time can include CPU launch gaps; it is not pure kernel time.

Breakdown stages: source reference resolution; source conversion; cached gather;
cached scatter; fresh index preparation (including fresh input gather); native
fresh projection; output allocation; fresh scatter; whole projection forward.
Legacy `.to()` and assignment share a source line, so they are reported as
`window_conversion_assignment`, **not** invented separate copy/scatter costs.
Exclusive Legacy conversion, gather and scatter times remain null.

`executor_ms` / `executor_gpu_ms` sum disjoint Q/K/V forward spans only; they include
native projection, assembly and passive audit work, but **exclude remaining
transformer, selector, mapping, context setup and policy**. Context setup is separately
`executor_context_setup_ms`; it contains `batch_mapping_ms`. Never sum these
nested values or add CPU milliseconds to CUDA milliseconds. The summary's
executor GPU ratio is a breakdown diagnostic, not an uninstrumented speedup.
`fetch_ms`, `assembly_ms`, `native_qkv_ms` are **host** aliases for cached gather,
cached scatter, and fresh-native spans; CUDA counterparts are in breakdown CSV.

Memory accounting distinguishes:

- Catalog physical storage bytes and original source-policy cache bytes.
- Existing Fast index Python-memory estimate and mapping Python-memory estimate.
- Actual logical allocated index bytes (including CPU mask/fresh arrays).
- Gather/conversion allocation-byte totals across calls (allocation volume, not peak).
- `temp_tensor_bytes`: conservative logical live gather/conversion peak + retained
  index tensors. Excludes common fresh/output/passive-capture buffers and allocator
  reservation; not a device-exclusive allocator peak or a net B2-minus-Legacy
  allocation difference (CPU mask/fresh arrays also exist in Legacy).
- CUDA allocated/reserved/peak bytes before/after each request. Peaks are reset per
  request. This observes the whole request, not just assembly. Validation allocations
  are not mixed into timing samples. Physical backing is never cloned for batching.
- Gather/scatter/conversion operation counts, whether conversion allocated, H2D
  payload bytes. These **are not measured kernel counts**; kernel count is null.
  Legacy conversion-copy count is explicitly inferred from source/model dtype/device.

Source request, catalog creation and Fast index build times are recorded separately.
Rows share `source_snapshot_id`; count those build costs **once** per snapshot,
not once per executor/coverage row. Construction is not amortized into warm latency.

## Paired experiment and parity gate

Each repeat/storage has one captured source, one shared catalog/index, one native
FULL reference, and independent target policy state cloned from the same source.
Catalog policy counters are restored for each request. Executor/instrumentation/
coverage order reverses on alternating repeats; warmup rows are separate.
FULL executes before variants and is a supplementary reference, not a fully
order-balanced FULL-vs-reuse study. Same model/tokenizer revision, adapter hashes,
input IDs, eager attention, output_attentions=True, use_cache=False, seed and dtype.

After **all timed cases** for that snapshot, additional untimed Legacy/Batched
forwards retain existing passive QKV captures on device. No hooks or CPU tensor
exports are installed. They verify:

- Ordered occurrence/tensor identity, selected mask, coverage and provenance.
- Reused, fresh and full assembled Q/K/V with `torch.equal`, every layer/role.
- Identical executed row audits and final logits (`torch.equal`).
- Each timed case's selection, row audit and logits exactly match its validation
  forward. Thus minimal/breakdown differences also fail the gate.

Only scalar results are saved. Untimed validation holds two sets of QKV and can
need extra memory: screening is not an OOM guarantee. OOM is recorded, never
silently skipped. Failed conditions remain in outputs and disqualify the paired
minimal condition even when faster. A standalone executor has no paired speedup
claim, although both validation forwards still run.

FULL-vs-executor metrics retain max absolute error, relative L2, last-position KL,
last argmax agreement, and `atol=rtol=1e-3` allclose, separately from Legacy-vs-Batched
exactness. Selector/assembly equivalence does not resolve FP16 full/subset GEMM
error. `warm_improvement` means exact-equivalent to Legacy and faster in minimal;
it **does not** mean strict FULL correctness or faster than FULL. Full ratios and
strict Full pass counts remain separate. Classification stays `ORACLE_DIAGNOSTIC`.

## CLI (no Slurm files)

Offline is default. Only explicit `--allow-download` permits model loader network
access. Use new output directories; B1 result directories are incompatible.

```bash
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# GPU smoke: source/QKV/logit validation, includes zero and full feasible coverage.
python scripts/76_benchmark_r1b3_batched_executor.py \
  --profile smoke --model facebook/opt-125m --dtype float32 \
  --device cuda:0 --cache-devices cuda --prompt-lengths 32 \
  --coverages 0,.5,max --executor both --instrumentation both \
  --output-dir results/r1b3/b2_executor/smoke125m_v1

# Main paired pilot: default 3 warmups / 10 measured, 128/256, GPU, MAX.
python scripts/76_benchmark_r1b3_batched_executor.py \
  --model facebook/opt-2.7b --dtype float16 --device cuda:0 \
  --prompt-lengths 128,256 --cache-devices cuda --coverages max \
  --executor both --instrumentation minimal \
  --output-dir results/r1b3/b2_executor/minimal27b_v1

# Separate diagnostic run, includes its own minimal instrumentation comparison.
python scripts/76_benchmark_r1b3_batched_executor.py \
  --model facebook/opt-2.7b --dtype float16 --device cuda:0 \
  --prompt-lengths 128,256 --cache-devices cuda --coverages max \
  --executor both --instrumentation both \
  --output-dir results/r1b3/b2_executor/breakdown27b_v1
```

For CPU cache use `--cache-devices cpu` (the model can remain on CUDA).
After pilot validation, extend explicitly with `--prompt-lengths 32,64,128,256
--coverages 0,.25,.5,.75,max --cache-devices cpu,cuda`. No full sweep is automatic.
CPU small-model execution uses `--profile smoke --device cpu --cache-devices cpu
--model facebook/opt-125m --dtype float32` and still requires local cached models.

Add `--resume` to the **identical** command/output directory after interruption.
New runs refuse collisions. Config/code/runtime identity must match. Complete
checksummed groups are resumed without duplicate aggregation; incomplete groups
are rerun, and warmup is repeated. Manifest preserves attempts/failure reasons and
records FAILED, INTERRUPTED, MEMORY_PREFLIGHT_FAILED, OOM or PARITY_FAILED. Preflight
is conservative and retains original screening semantics. No automatic length or
coverage reduction. The runner is single-process/single-writer, not thread safe.

Required outputs: `manifest.json`, `executor_parity.json`, `projection_audit.json`,
`latency_comparison.csv`, `executor_breakdown.csv`, `memory_accounting.csv`,
`correctness_comparison.csv`, `summary.json`, plus checksummed group checkpoints.
No attention/hidden/QKV tensor dumps. Summary excludes warmup, reports
mean/p50/p95/std and matched deltas; exact-equivalence failures remain visible.

## Validation and next steps

```bash
PYTHONPATH=src:tests python3 -m unittest \
  test_r1b3_batched_executor test_r1b3_fast_selection test_r1b3_occurrences \
  test_r1b2 test_r1b1_d1 test_r1b1_impact test_m9a2_impact_audit \
  test_m85_alignment test_r1_raw_semcache -v
```

Offline tests exercise randomized selected-hit mappings, 126/128 and 255/256
**metadata fixture** coverage, grouping, overlap/position/prefix/adapter/namespace/
stale-view rejection, CLI collision/resume/failure artifacts, and parity-gated
summary logic. These are not real tensor or GPU coverage measurements.

Torch tests (skipped here) exercise real strided/multiple backings, all/no/partial
reuse, actual QKV/row equality, temporary accounting, mutation, CPU/GPU residency,
and original-observer restoration. Tiny random OPT+PEFT tests (also skipped here)
verify logits, policy snapshot and projections on CPU/CUDA without downloads.
The SERAPH smoke verifies real OPT-125M and the pilot verifies OPT-2.7B FP16.

Final local suite: 106 tests, 68 passed / 38 skipped; no Torch/Transformers/PEFT.
New B2 tests: 5 offline passed / 6 tensor-model skipped. No GPU/model test is
marked passed. Compile/help/diff checks also run locally. An actual CLI run in a
temporary directory exited 1 with `FAILED: No module named torch` and produced
all eight required artifacts; this is failure-path validation, not a benchmark.

Whether batching improves warm latency is unresolved until these commands run.
Potential remaining costs: guarded snapshot scans, index allocation, gather
allocations, model launch overhead, original frontend/policy, and CPU cache transfer.
Only measured results should determine a subsequent optimization; no custom kernel,
source-index change, precision relaxation or production-policy integration is made.
