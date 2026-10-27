# R1-B3-B2.1 — Mapping/setup optimization and latency reproducibility

Implementation status: CLI and offline checks are ready. Local Torch, Transformers,
PEFT and CUDA are unavailable, so no real tensor, model, GPU latency or speedup
result is claimed. The executor classification remains `ORACLE_DIAGNOSTIC`.
FP16 `atol=rtol=1e-3` and the known FULL-vs-reuse failure remain separate from
Legacy-vs-new executor parity.

## Baseline and new code

B2's Legacy executor builds a reused mask and fresh-position list, validates and
patches each Q/K/V module, executes Native PEFT on only fresh rows, allocates the
full output, and then loops through hits doing backing lookup, `.to(output)`, and
window slice assignment. B2 V1 instead builds a per-request PositionMapping,
source-query groups, and position indices, then gathers/scatters. Source and
Target index tensors were created on each request, even for the same workload.

The GPU figures below are user-provided B2 measurements; they were not reproduced
locally. They show why this step adds 30 paired repeats and order control instead
of treating a single 10-sample average as a stable conclusion.

| Tokens | B2 Legacy mean | B2 Batched mean | Paired Batched wins |
|---:|---:|---:|---:|
| 128 | 173.66 ms | 164.71 ms | 10/10 |
| 256 | 269.81 ms | 229.67 ms | 10/10 |
| 128, mixed run | 175.09 ms | 173.09 ms | 6/10 |
| 256, mixed run | 253.10 ms | 277.02 ms | 5/10 |

The earlier 256-token 492/568 ms outliers remain part of the raw sample and will
be flagged by the Tukey 1.5×IQR rule. No row is removed. Their cause is left
unattributed unless the recorded GPU state identifies it.

New files:

- `src/semcache/edgelora/batched_mapping_optimized.py`
- `scripts/77_benchmark_r1b3_b21_setup.py`
- `scripts/78_aggregate_r1b3_b21.py`
- `tests/test_r1b3_b21_mapping.py`
- this document

B2's `mixed_projection_path()` and B2 V1 executor are not changed. The new Runner
keeps the B1 Fast Selector fixed and supports `legacy`, `v1`, `v2`, `contiguous`,
and `all`; the default mode is Legacy. The source capture, Tokenizer workload,
controlled adapters, request timing, row audit, RAW-only guard, and Full logit
metrics reuse R1 code. No CacheGen codec or quantized cache path is used.

## V2 descriptors, exact-target cache, and contiguous ablation

A SourceBackingDescriptor is made once after source prefill, catalog creation, and
the existing validated FastOccurrenceIndex. It stores that index's immutable
Source/Entry/Occurrence references, source owner tensor stamps, window-view stamps,
model/adapter/layout identity, and structural bucket snapshot. It owns **zero extra
QKV bytes**. Descriptor build milliseconds and the Python metadata-size estimate
are recorded separately from FastIndex build and Catalog copy/build.

Every V2 target lookup includes an exact key containing the catalog snapshot ID,
source tensor object identities and stamps, the complete Target token tuple, the
ordered selected entry/source/destination positions and token IDs, model and
adapter identity, sequence length, device, and dtype. The bounded LRU defaults to
four mappings. A changed Target or selected hit sequence cannot reuse a prior
mapping. On every request the executor validates snapshot structure and source
owner versions and validates selected view Tensor backing metadata before
injection. Source/Target causal-prefix equality is established when a mapping is
built; it is reused only for the exact Target token tuple and exact hit sequence.
No hashes decide safety. Stale source owners, metadata, selected views, adapter,
namespace, or mapping hit order fail closed.

The plan stores first-seen Source Query groups, flattened source/target positions,
nonoverlap mask, fresh rows, and prepared CPU/device Long indices. Repeated exact
Targets reuse those index tensors. Mapping memory estimates, actual CPU/device
index bytes, cold build time, cache hit rate, and evictions are reported. Cache
lookup validates the snapshot in executor setup, so validation remains in warm
latency rather than being hidden in source build.

`contiguous` uses a direct source/target slice assignment only when the selected
positions for a single backing are both strictly contiguous. It keeps the same
selected QKV. A non-contiguous group uses V2 Gather/Scatter. Multiple source
backings remain grouped separately. The contiguous mode is an ablation: on the
provided synthetic position-aligned prompts it may find one long contiguous span;
that does not predict performance on arbitrary SemCache traffic.

## Timing, setup breakdown, and memory

Use `--instrumentation minimal` for the primary performance comparison and a
separate `--instrumentation breakdown` invocation for attribution. A run cannot
mix detailed tracing into its minimal requests. The target order is a seeded cycle
through all permutations of the requested executor modes (AB/BA for two modes,
all six/four-mode permutations for three/four modes). The complete schedule and
seed are in `manifest.json`. Warmup is paired across modes. Pilot defaults to 3
warmups and 30 paired measured iterations per Source Snapshot.

The synchronized `request_wall_ms` from R1 runtime is the whole warm request. The
minimal path uses only small host stopwatches for selection, mapping lookup/build,
and executor context enter/cleanup. It does not add CUDA events or the Legacy
Python line tracer.

Breakdown host stages are:

- Source backing descriptor preparation and Source snapshot validation.
- Target hit-to-position mapping and Source group construction.
- CPU index tensor build and CUDA device-index preparation.
- Module patch setup and context cleanup.
- Batched Gather/Scatter host execution and B2 ExecutorBook CUDA event spans.
- Overall warm request.

CUDA index-copy events resolve after the enclosing request has synchronized; there
is no per-index or per-window synchronization. Host submission time and CUDA-event
time stay separate. `target_hit_to_position_mapping` includes some descriptor/key
work, and Source group construction/index construction are nested inside mapping.
`mapping_setup_total_ms` is a separately measured inclusive stopwatch over Target
mapping plus context setup/cleanup. Do not sum it with its component rows. The B2
ExecutorBook's whole Q/K/V projection span contains Fresh Projection, assembly,
and passive capture; its child events are diagnostic and are never added to the
request time. Legacy detailed mode uses the original B2 line observer. Its source
reference and window `.to()`/assignment costs are measured where separable; no
exclusive Copy time is fabricated from the shared `.to`/assignment line.

The Runner samples `nvidia-smi` before and after each length/storage measurement
block, outside requests. It records GPU UUID, utilization, memory, SM clock,
temperature, power, timestamp and compute-process listing. Unavailable values
stay null with a reason. Each request also records CUDA allocator before/after
memory and peak memory. These describe the whole request, not just QKV assembly.

Persistent bytes are separated into Legacy source-policy cache, Catalog physical
bytes, descriptor Python metadata estimate, mapping Python estimate and cached
index Tensor bytes. Temporary Gather/Conversion volume, H2D payload bytes and
maximum logical temporary estimate are distinct fields. The logical estimate is
not an allocator peak and is not a net V2-minus-V1 device-memory delta. Source
prefill, Catalog build/copy, FastIndex build and descriptor build remain
Source-side costs; none is added to warm request latency. Summary includes a
measured-cost amortization estimate for descriptor/cold mapping cost versus warm
paired savings. It is derived arithmetic, not a long-running cache result.

## Parity and statistics

For each repeat the Runner compares Legacy/V1/V2/contiguous selected provenance,
coverage, projection row audits and final logits. It then runs one additional
untimed validation forward per mode on the same source snapshot. Existing passive
QKV captures are compared with `torch.equal` for reused rows, fresh rows, assembled
Q/K/V, every layer and role. The untimed Full Recompute reference supplies max
absolute difference, relative L2, last-position KL, last argmax and strict
`allclose` at `1e-3` tolerances. Those Full metrics do not determine Legacy/new
executor parity. A faster row with failed mapping, QKV, row-audit or logit parity
cannot pass the improvement gate.

Per length/storage/coverage/mode, the summary retains mean, median, standard
deviation, p95, min/max, paired baseline-minus-mode differences, win count,
speedup, and a seeded 5,000-sample paired bootstrap CI. Positive paired difference
means optimized is faster. IQR outliers are tagged and retained. One run's bootstrap
is exploratory; it does not establish reproducibility across independent runs.
`78_aggregate_r1b3_b21.py` performs a hierarchical bootstrap that resamples
independent run IDs and then paired iterations. It reports a reproducible
improvement only when at least three independent runs agree, the parity gate passes,
and the run-cluster bootstrap CI is above zero.

The Source Snapshot is shared across all paired target repeats in one run block.
Warmup requests prebuild exact-target cache entries. Mapping cache hit-rate rows
distinguish warm hits from cold builds; validation forward cache hits are reported
in `mapping_parity.json` after timing and do not alter measured rows. A resumed
incomplete run reconstructs the source and repeats warmups before its remaining
measured pairs. Complete checkpoints are checksummed and not counted twice.

## SERAPH commands

Run minimal and breakdown into distinct output directories. The default is offline;
only `--allow-download` permits model downloads.

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

OPT-125M correctness smoke:

```bash
python scripts/77_benchmark_r1b3_b21_setup.py --profile smoke \
  --model facebook/opt-125m --dtype float32 --device cuda:0 --cache-device cuda \
  --prompt-lengths 32 --coverages max --executor all --instrumentation minimal \
  --output-dir results/r1b3/b21_setup/smoke125m_run01
```

OPT-2.7B FP16 reproducibility pilot, 30 paired iterations:

```bash
python scripts/77_benchmark_r1b3_b21_setup.py --profile pilot \
  --model facebook/opt-2.7b --dtype float16 --device cuda:0 --cache-device cuda \
  --prompt-lengths 128,256 --coverages max --executor all --instrumentation minimal \
  --independent-run-id run01 --seed 42 \
  --output-dir results/r1b3/b21_setup/minimal_run01

python scripts/77_benchmark_r1b3_b21_setup.py --profile pilot \
  --model facebook/opt-2.7b --dtype float16 --device cuda:0 --cache-device cuda \
  --prompt-lengths 128,256 --coverages max --executor all --instrumentation minimal \
  --independent-run-id run02 --seed 42 \
  --output-dir results/r1b3/b21_setup/minimal_run02

python scripts/77_benchmark_r1b3_b21_setup.py --profile pilot \
  --model facebook/opt-2.7b --dtype float16 --device cuda:0 --cache-device cuda \
  --prompt-lengths 128,256 --coverages max --executor all --instrumentation minimal \
  --independent-run-id run03 --seed 42 \
  --output-dir results/r1b3/b21_setup/minimal_run03
```

Run the same three independent output roots with `--instrumentation breakdown`
under a separate `breakdown_runXX` directory to measure tracing overhead. Keep
model revision, adapters, seeds, lengths, device, cache location and requested
coverage the same between the two timing studies. Each pilot output is resumable
with the exact same command plus `--resume`; changed code/config uses a new root.

Aggregate the three minimal runs:

```bash
python scripts/78_aggregate_r1b3_b21.py \
  --run-dirs results/r1b3/b21_setup/minimal_run01,results/r1b3/b21_setup/minimal_run02,results/r1b3/b21_setup/minimal_run03 \
  --output-dir results/r1b3/b21_setup/aggregate_minimal_v1 --seed 9107
```

For a CPU cache ablation, set `--cache-device cpu` while leaving model `--device
cuda:0`. Use one coverage per run; run separate output directories for other
coverage points. No Slurm files are created. Likely pilot duration can exceed an
hour; launch it with the user's existing scheduler procedure.

Every run writes `manifest.json`, `mapping_parity.json`, `executor_parity.json`,
`setup_breakdown.csv`, `latency_comparison.csv`, `paired_differences.csv`,
`gpu_state.json`, `memory_accounting.csv`, `correctness_comparison.csv`, and
`summary.json`, plus checksummed checkpoints. Tensor contents are never serialized.

## Local validation

```bash
PYTHONPATH=src:tests python3 -m unittest test_r1b3_b21_mapping -v
PYTHONPATH=src:tests python3 -m unittest \
  test_r1b3_batched_executor test_r1b3_fast_selection test_r1b3_occurrences \
  test_r1b2 test_r1b1_d1 test_r1b1_impact test_m9a2_impact_audit \
  test_m85_alignment test_r1_raw_semcache -v
```

Offline tests cover randomized mapping identities, 126/128 and 255/256 metadata
coverage, exact-target cache keys, source owner/view/namespace/prefix invalidation,
LRU accounting, four-mode order balancing, bootstrap sign/outlier retention,
checkpoint collision/resume, failure artifacts and independent-run aggregation.
Optional tensor tests exercise Legacy/V1/V2/contiguous QKV exactness, partial/no/max
reuse, row audits, mapping cache hits, stale backing, and CPU/GPU cache. They skip
clearly when Torch/CUDA is absent. No local test result is a GPU benchmark.
