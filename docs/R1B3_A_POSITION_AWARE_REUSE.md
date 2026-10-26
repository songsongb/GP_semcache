# R1-B3-A: position-aware source occurrence diagnosis

This adds an audit and an **opt-in ORACLE_DIAGNOSTIC** source-occurrence catalog.
It does not change production matching, GlobalCache, admission/eviction, CHU/PBR,
the mixed executor, Token Precompute, tolerances, previous experiment artifacts,
or any Slurm file. No batched fetch/assembly optimization is implemented.

## What is known, and what is not

The current code confirms this mechanism:

1. `ExactTokenMatcher.key()` returns `(cluster_id, token_ids)`.
2. `GlobalCache.entries` owns a single entry per key; insertion will not replace
   an existing key.
3. Source admission in `SemCacheEngine.query()` skips a window if its key is
   already resident **or considered earlier in that query**. A rejected first
   occurrence also prevents later occurrences from being reconsidered.
4. `select_position_aligned_diagnostic()` requires the stored source position,
   full source token sequence and user/adapter to match the target.
5. Consequently a repeated window can exist at the correct position in the
   source computation, while its QKV at that position is unavailable from the
   single-entry cache. Dropping wrong-position hits does not recover that QKV.

`semantic/addressing.py` already distinguishes source occurrence/address from
match key, but its descriptors are design-only, not wired into physical storage.
This experiment implements a separate catalog rather than repurposing those
production-neutral descriptors as cache addresses.

The reported 30/54/54/54 strict reused-token counts and 256-token latencies are
**user-provided historical results**. No corresponding B2 artifacts exist in
this local checkout. The code mechanism is verified by inspection and synthetic
provenance tests; the exact historical 54-token limit is **not verified here**.
A periodic 54-token synthetic fixture demonstrates this mechanism at 128/256
without pretending to be the OPT tokenizer's workload or a GPU experiment.

## A1: source occurrence loss audit

Run without `--enable-catalog` first. The runner uses the unchanged B2 setup,
source `engine_request(... LOOKUP_NO_REUSE, detailed=True)`, and minimal targets.
The audit runs **outside timing** on the source result and its resident entries,
before any target policy update. It records actual source token IDs, admissions,
rejections/scores, deduplication and evictions, retained positions, residency and
layer coverage. No audit step mutates the engine or synthesizes QKV.

For every sliding w=3 target window it records:

- Source occurrences with the same key, and whether one exists at this position.
- Prefix equality through the **window end**, not just equality of window tokens.
- Whether aligned source QKV is actually resident in the pre-target legacy cache.
- Retained source position, all loss causes, and an explicitly prioritized cause.
- Strict eligibility, max nonoverlap selection, and overlap exclusion.

Cause flags are non-exclusive, in this reporting priority:

1. `ADAPTER_OR_MODEL_MISMATCH`
2. `PREFIX_CONTEXT_MISMATCH`
3. `NO_MATCHING_SOURCE_OCCURRENCE`
4. `CAPACITY_OR_EVICTION`
5. `ADMISSION_REJECTED`
6. `DUPLICATE_KEY_OCCURRENCE_LOST`
7. `POSITION_MISMATCH`
8. `LEGACY_FULL_PROMPT_MISMATCH` (additional explanation of the unchanged strict selector)

For example, a deduplicated later occurrence may simultaneously have a
wrong-position resident key and an earlier rejection/eviction. None of those
facts is discarded by choosing a primary display label. Capacity rejection is
separated from score rejection. A deduplicated occurrence is not called a
separate admission rejection if its key was never independently evaluated.

The predicted legacy max spans are checked against **executed** legacy max
spans. A discrepancy fails the run rather than endorsing a causal explanation.
`current_run_legacy_max_selection_verified` applies only to the newly observed
run. `historical_54_token_cause_verified` remains false: compact B2 rows alone do
not contain the original admission/dedup evidence needed for that assertion.

`--reference-b2-root` optionally reads existing `tokens_<n>/manifest.json` and
`raw.jsonl`, hashes both, requires the exact saved token sequence, reports
model/runtime/seed/code differences, and compares same-repeat minimal legacy MAX
coverage and selection signatures. `SELECTION_MATCH` is not a claim of historical
latency reproduction or full historical admission provenance. Pin the original
revisions and seed before replay. A new run's causal audit does not retroactively
create missing historical evidence.

## A2: source-only occurrence catalog

The opt-in mode is named `POSITIONAL_OCCURRENCE_DIAGNOSTIC` and classified
`ORACLE_DIAGNOSTIC` because it stores every source occurrence, including keys or
occurrences that original admission did not retain. Its comparison mode is
`LEGACY_POSITION_ALIGNED`, exactly the B2 coverage override plus original strict
selector. Native FULL, LOOKUP_NO_REUSE and Natural RAW are also measured so the
known natural position/prefix errors remain visible.

The occurrence identity is:

```
(cluster_id, token_ids, source_query_id, source_start, source_end,
 adapter_id, model_namespace)
```

Namespace is bound to loaded model/tokenizer provenance; weight identity binds
the resolved model revision and actual controlled adapter parameter SHA256s.
The same frozen model/tokenizer instance is used for source and target. The
catalog is in-process only, never loaded from an unverified disk tensor cache.
Different model, tokenizer, adapter or weight identity is rejected.

For catalog source construction only, a scoped wrapper around existing
`r1_runtime.detailed_mixed` retains its already computed TOTAL Q/K/V outputs.
There are no new projection hooks or extra projection calls. Only **after the
source request returns** does `add_source()` validate and copy those tensors.
The untimed target native reference supplies logits for comparison only; it is
never connected to the capture wrapper or catalog.

One owned full `[1, source_tokens, output_width]` tensor per layer/role is kept on
the requested CPU/GPU device. Every occurrence entry is a view at its actual
source position. Identical token windows at different positions or query IDs
remain distinct, with distinct position-specific data. Multiple queries are
supported by the catalog API; the controlled runner uses one source per target.

Selection verifies:

- cluster/key identity, adapter, model/tokenizer namespace and weight identity;
- absolute source/target position and exact window token IDs;
- equality of source and target causal token prefixes through the window end;
- full layer/role coverage, shape/dtype/device, and backing tensor pointer/stride
  matching that source's slice;
- source completion, floating unquantized finite tensors and capacity at build.

Suffix differences **after** a window are permitted by the catalog when its
prefix and other predicates match; the legacy selector still demands full
prompt equality. The default experiment is same-user exact prompt. Supporting
this predicate does not assert suffix-different FP16 numerical equivalence.
Per-window `prefix_token_ids_equal` and eligibility/selection reasons are saved.

The selector binding is scoped to the runner and restored on errors. Existing
semantic lookup and policy code still executes. In the oracle path, CHU updates
selected sidecar entries, initialized with the actual source window impacts and
source clock; legacy resident entries are not secretly replaced. PBR still
visits only GlobalCache entries. This is deliberately **not policy-equivalent**
to normal SemCache, and is not a new admission strategy. Independent source
rebuilds for each target prevent this diagnostic state from contaminating the
legacy comparison.

## Coverage and correctness

Original earliest-start nonoverlap selection runs on eligible occurrences, then
the B2 deterministic coverage filter keeps at most `floor(n*f/3)` windows. The
CLI uses fractions `0,.25,.5,.75,max`. MAX keeps feasible candidates, not an
assumed 100%. Saved fields include stored occurrences, unique stored token keys (and separately unique source keys),
requested/actual coverage, actual selected spans, reused/recomputed tokens, and
native/skipped rows summed over all Q/K/V roles and layers. Zero reuse must have
zero skipped rows. Native minimal FULL row counts are explicitly derived; all
engine paths require actual executor records.

With complete eligible occurrences and no storage limit, the **theoretical**
upper reused-token counts are 30, 63, 126 and 255 for 32, 64, 128 and 256.
These are not new measured results. Each catalog window records `SELECTED`,
`COVERAGE_FILTER`, `NONOVERLAP_CONFLICT` or `INELIGIBLE` with rejection reasons.
Legacy missing occurrences remain documented in its source loss audit.

Correctness is calculated outside timing: entire-logit allclose at unchanged
`atol=rtol=1e-3`, max absolute/relative L2 difference, last-position KL and argmax,
position/prefix mismatch counts (tokens), adapter identity, and physical row
audit. Selection validity and numerical allclose are independent. Recovering
positions does **not** fix FP16 full/subset GEMM shape sensitivity observed in
D1, and Natural RAW mismatch is not relabeled a success.

Optional `--full-shape-control` reuses D1's native full-shape injection context,
without D1 layer captures. It runs after target timing and saves correctness
only, labeled `CORRECTNESS_ONLY_FULL_SHAPE_NO_SAVINGS`. Its timing is discarded.
Prefer a separate smoke correctness run with this option, since extra forwards
can affect GPU clocks before later measurements. Formal latency runs should
omit it. There are no tensor dumps or full-shape performance claims.

## Storage, lifetime and timing

The primary target is the unchanged R1 **minimal** request: synchronized start
and end, no added per-layer CUDA events, no layer-to-CPU capture hooks. Existing
CPU timers, policy and projection audits remain. B3 deliberately rejects
breakdown mode; B2 remains the independent breakdown runner.

The catalog selector's CPU wall includes validation, indexing, window audit
construction, nonoverlap and coverage filtering. This extra work is **inside**
target warm latency and separately recorded, never subtracted. Higher coverage
can be slower; failed allclose and slow cases remain in output.

`source_request_ms` is the existing detailed source request. `catalog_build_ms`
is subsequent finite/shape validation, owned tensor copies and entry/index
construction, including synchronization. `source_build_ms` is their sum.
`catalog_copy_ms` is nested in build and is **not added again**. Audit hashing,
JSON writing, model initialization and reference/correctness forwards are
separate, outside those costs, just as B2 excludes its provenance packaging.
Source retention bookkeeping is enabled only for oracle source requests and is
identified in each row. No source cost is hidden in target latency.

Physical bytes count unique underlying storages, not the sum of overlapping
occurrence views. Output distinguishes catalog bytes, legacy cache bytes before
and after target policy, and combined unique backing storage bytes. It records
cross-device source-copy bytes and copy wall time; `.to(...).clone()` also makes
an owned same-device copy. CPU transient peak allocation is **not measured**;
resident bytes are not a CPU peak-memory claim. Python catalog/index metadata
allocation is not included in tensor backing-byte totals. GPU snapshots/peak allocator
statistics cover source construction, catalog build and target execution.

Source execution-device QKV references survive until catalog creation, then are
cleared. Owned catalog tensors and views survive through the target and optional
correctness control. All hit references are released before the next source
build. Legacy copies are separately owned and counted. The catalog has its own
explicit capacity; exceeding it fails rather than silently dropping occurrences.
Conservative memory preflight includes the additional GPU catalog budget; actual
OOM still remains possible with eager attentions and temporary copies.

## CLI

No downloads by default. The inherited `--allow-download` is the sole explicit
opt-in. Use the existing `semcache` environment and HF local cache on SERAPH:

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

First audit legacy only; catalog is disabled by default:

```bash
python scripts/74_diagnose_r1b3_occurrences.py --profile pilot \
  --prompt-lengths 128,256 --cache-devices cuda --coverages max \
  --output-dir results/r1b3/a_occurrences/legacy_audit_v1
```

If historical B2 artifacts are present, add
`--reference-b2-root results/r1b2/full27b_v1` (use their actual directory), along
with original `--revision`, `--tokenizer-revision`, `--encoder-revision`, `--seed`.
The old files are read-only.

OPT-125M FP32 smoke and optional D1 full-shape correctness control:

```bash
python scripts/74_diagnose_r1b3_occurrences.py --profile smoke \
  --model facebook/opt-125m --dtype float32 --prompt-lengths 32 \
  --cache-devices cpu,cuda --enable-catalog --full-shape-control \
  --output-dir results/r1b3/a_occurrences/smoke125m_v1
```

Formal OPT-2.7B FP16 comparison (warmup 3, measured 10):

```bash
python scripts/74_diagnose_r1b3_occurrences.py --profile full \
  --prompt-lengths 32,64,128,256 --cache-devices cpu,cuda \
  --coverages 0,.25,.5,.75,max --enable-catalog \
  --cache-capacity-mib 256 --catalog-capacity-mib 256 \
  --output-dir results/r1b3/a_occurrences/full27b_v1
```

Length-specific runs can use separate new directories. Each default group has
13 targets when catalog is enabled (FULL, LOOKUP, Natural, five legacy and five
catalog coverages), 12 independent source builds and one untimed native reference.
This is intentionally more work than one catalog reused across all conditions:
source/target policy preconditions remain controlled. Source construction and
mode order are rotated/reversed across repeats; storage order alternates.
Use the user's existing scheduler configuration for long runs. No Slurm file is
created or modified by this change.

CPU smoke is possible with installed Torch/Transformers/PEFT and cached weights:
`--device cpu --cache-devices cpu --dtype float32`. Missing packages/weights are
an explicit failure, not a simulated model measurement.

Resume with the **identical command** plus `--resume`. Code/config/runtime must
match. Completed measured groups are skipped; warmups rerun. Atomic group
checkpoints have row/audit checksums, and duplicate row IDs are rejected. A new
run on an existing directory fails. Modified code needs a new directory; never
reuse B1/B2 output paths. Output interruption/preflight/OOM/catalog-capacity
failure is preserved in manifest; an incomplete group is not aggregated as a
completed sample. Source audits already collected are persisted separately even
if catalog construction or target execution fails. Hard termination may leave
`RUNNING`; resume still validates checkpoints before continuing.

## Artifacts

Under `results/r1b3/a_occurrences/<tag>/`:

- `manifest.json`: git/code hashes, model/tokenizer/encoder/adapter identities,
  seed, Torch/CUDA/GPU/driver/backend settings, flags, fixtures, raw-only evidence,
  preflight, active condition and status.
- `occurrence_loss_audit.json`: deduplicated full source provenance and
  per-window loss explanations, referenced by audit ID from rows.
- `coverage_comparison.csv`: coverage, spans, selection exclusions and row counts.
- `correctness_comparison.csv`: quality, row audit, optional full-shape control,
  reference B2 selection comparison and CPU/GPU selection/logit parity.
- `storage_accounting.csv`: real resident backing bytes, copies and GPU memory.
- `latency_comparison.csv`: separate source, catalog and warm target times.
- `summary.json`: measured-only mean/p50/p95/std, coverage and allclose counts.
- `source_audits/` and atomic `checkpoints/`: restartable scalar/provenance records.

No QKV, attention, hidden-state or logit tensors are serialized. Within-process
CPU/GPU storage parity compares selected spans and logits at the original
threshold; mismatches set `STORAGE_PARITY_FAILED`. Resumed groups without their
counterpart in memory remain `UNPAIRED`, not fabricated passes.

## Validation and next steps

```bash
PYTHONPATH=src:tests python3 -m unittest \
  test_r1b3_occurrences test_r1b2 test_r1b1_d1 test_r1b1_impact \
  test_m9a2_impact_audit test_m85_alignment test_r1_raw_semcache -v
python3 -m py_compile scripts/74_diagnose_r1b3_occurrences.py \
  src/semcache/diagnostics/r1b3_occurrences.py tests/test_r1b3_occurrences.py
```

Local results: **56 passed, 28 skipped, 84 total**. New B3 tests: six
provenance/structure/lifecycle passes, six tensor/model skips. Torch,
Transformers, PEFT and pytest are absent. An actual dependency-missing CLI
invocation returned exit 1, wrote `FAILED` and all seven required report files.
No OPT GPU run was performed, and no actual coverage recovery, storage parity,
FP16 correctness improvement or latency improvement is claimed.

Optional tensor/model tests cover repeated 128/256 synthetic fixtures, distinct
source values, multiple query identities, wrong-position/prefix/adapter/namespace
rejection, backing-view validation, nonoverlap/coverage, catalog capacity,
CPU/GPU storage data parity, original B2 legacy selection/output parity and
actual CPU/CUDA projection row audits. They require execution on the appropriate
installed runtime before these properties are reported as model-tested.

R1-B3-B candidates remain conditional: repeated per-window transfer/index/assembly
and Python selection/validation costs. First confirm current-run source loss,
actual recovered rows, numerical error and additional storage. Only then use
minimal latency evidence to choose an executor optimization. This stage implements
none of those optimizations and does not assume a break-even exists.
