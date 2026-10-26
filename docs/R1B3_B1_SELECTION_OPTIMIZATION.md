# R1-B3-B1: position-aware occurrence selection optimization

This is an opt-in **ORACLE_DIAGNOSTIC selector experiment**, not a production
cache-policy change or a mixed-executor optimization. New files are
`diagnostics/r1b3_fast_selection.py`, runner 75, its tests and this document.
Runner 74 and its legacy `OccurrenceCatalog.select()` are unchanged. GlobalCache,
Engine, admission/eviction, impact reducer, mixed projection/fetch/assembly,
previous artifacts and Slurm files are unchanged. No push is performed.

## Code-based cost analysis, not attribution of the historical milliseconds

The user reports selection means 76.55/165.67 ms at 128/256 tokens and recovered
126/255 tokens. These historical GPU measurements are not present in the local
checkout and have **not been reproduced here**. Code inspection establishes
which operations repeat, not what fraction of those milliseconds each costs.
The breakdown pass is needed to establish that fraction on SERAPH.

For W=N-2 target windows, K matching occurrences, C accepted occurrences, L layers:

|Existing operation|Actual implementation and cost|
|---|---|
|Occurrence search|Already dictionary lookup by `(cluster, exact tokens, start, end)`; **not** a scan of all source occurrences per window. It traverses the matching query bucket, K total.|
|Prefix validation|`occurrence_reasons()` slices the source prefix, slices/converts the target prefix and compares them for each match. Matching full inputs cause total prefix work proportional to the sum of window ends; single-source worst case is O(N²).|
|Source provenance check|For each accepted candidate, compares its full source-ID tuple against the authoritative source tuple again. O(C × source length) in the equal-but-distinct-tuple case.|
|Backing validation|For every accepted candidate/layer/role, constructs a new source slice and checks pointer/stride/shape/dtype; normalizes device via `torch.device` and string conversion each time. O(3CL) tensor-metadata operations, including 3CL new slice wrappers.|
|Identity/position checks|Repeated scalar/tuple comparisons per matching occurrence; most are source invariants.|
|Selection|Materializes CacheHit objects, stable-sorts candidates for nonoverlap, then sorts selected hits again in the coverage filter. Already ordered inputs help Timsort; do not assume worst-case sorting dominates.|
|Audit|Hashes every accepted occurrence identity using JSON/SHA256 and builds per-window dictionaries/lists on each request.|

For 256 tokens, one valid source and 32 layers, the backing loop constructs
`254 × 32 × 3 = 24,384` slices per target. Prefix ends sum to 32,893 for that
single source, before counting the additional conversions/comparisons. These
are operation counts derived from code, **not measured time shares**.

## Fast index and preserved semantics

`FastOccurrenceIndex` is built only after source-only QKV catalog construction.
It references the **same CacheEntry and exact same QKV tensor objects** as legacy;
no QKV copy or target-FULL-derived cache is introduced. It retains every source
query/position occurrence and original bucket insertion order.

Source-side work validates the complete snapshot: namespace/weight/adapter,
source query and token positions, authoritative source prefix, layer/role
coverage, tensor shape/dtype/device and backing pointer/stride. It caches
occurrence hashes, immutable window/CacheHit descriptors, lookup buckets and
source/query identities. This cost is recorded as `index_build_ms`, separately
from source prefill and catalog build. It is not silently removed from total
cost accounting.

Target-side work:

1. Check that source/catalog structure, identity tuples and layer-to-QKV mappings
   have not changed. Policy frequency/impact/access-clock changes are allowed.
2. Check backing tensor pointer/shape/stride/dtype/device/version/storage size and
   each occurrence view's pointer/shape/stride/dtype/device. This detects view
   rebinding, including `.data` rebinding without a version increment. A mutation
   invalidates the index and raises; no unsafe stale-index fallback is used.
3. Normalize target IDs once and extract windows once.
4. Compute **exact common-prefix length once per source/target query pair**.
   `window.end <= common_prefix_length` is equivalent to zero accumulated
   mismatches through that window. It handles unequal lengths and suffix-only
   differences without a hash-collision assumption. Prefix results are local to
   this request, never cached across changed target contents.
5. Validate target adapter/namespace/weight identity per source query, look up
   position-aware buckets and preserve all legacy audit flags.
6. Select nonoverlapping windows in increasing target order. Legacy CacheHits
   have utility zero and equal keys within one position bucket, so its stable
   sort chooses the first eligible bucket member. The fast scan preserves that
   insertion order, **not lexical query-ID order**.
7. Apply the same `floor(N*coverage/3)` prefix of feasible windows and emit the
   same validation/selection audit rows with precomputed occurrence hashes.

This is specialized to the existing w=3, zero-utility catalog selector. It is not
an alternative general policy for arbitrary utility-ranked hits. Source/window
identity objects must be immutable; source changes require rebuilding. Versioned
owned tensors are required; unsupported inference tensors without version
counters are rejected rather than weakening the guard. The current catalog
clones source tensors outside inference mode.

The target guard still costs O(entries × layers × roles) metadata checks. The
optimization removes repeated backing-slice construction/device normalization,
repeated full-prefix comparisons, identity hashing and redundant sorting; it does
**not** claim constant-time validation or complete elimination of Python work.
Original admission bypass and FP16 numerical limitations remain unchanged.

## Breakdown versus minimal

`--selection-instrumentation minimal` is the primary latency comparison. It uses
the unchanged minimal engine/mixed executor. All target window extraction, index
guards, prefix preparation, lookup, coverage and audit construction occur inside
the timed selector callback and therefore inside target warm request time.
There are no inner CUDA timers or layer tensor capture hooks.

`breakdown` adds CPU `perf_counter_ns` subspans to the selector only. It still uses
the original mixed executor; this is **not** B2's detailed mixed executor pass.
Fine-grained Python instrumentation adds overhead and must not replace minimal
performance evidence. `both` executes separate passes and checks their selection
and output parity; the latency CSV records breakdown-minus-minimal request time.

Legacy breakdown observes the original methods through scoped wrappers. It does
not copy/reimplement the legacy selection algorithm. Tensor mapping iteration
measures the original per-layer validation loop; wrappers are restored **before**
model forward. All wrappers are removed on exceptions as well.

|Requested region|Legacy observation|Optimized observation|
|---|---|---|
|Source-only index construction|Null; existing catalog build combines copies/index/entries|Direct source-only index wall|
|Target window extraction|Direct|Direct, includes one target tuple normalization|
|Candidate lookup|Direct dictionary get|Direct bucket lookup|
|Position filtering|Null exclusive time; part of combined predicates|Null exclusive time; encoded in bucket identity and source proof|
|Prefix context validation|Combined with identity/position predicate|Direct query-pair prefix precompute|
|Model/adapter identity|Combined with prefix/position|Direct per-source target identity setup|
|Non-overlap|Original function interval|Ordered greedy scan intervals|
|Tensor backing validation|Original per-layer loop body|Owner and view change guards|
|Coverage filtering|Original function interval|Equivalent count/slice/mask construction|
|Audit/provenance construction|Digest-only partial interval; full construction is null|Row construction/final selection annotation intervals|

Additional fields expose `snapshot_structure_validation`,
`identity_position_prefix_combined` and `audit_digest_only`. Unsupported exclusive
metrics are **null**, never zero or an invented decomposition. Small loop/dispatch
costs remain outside individual spans but inside total selection. Breakdown
numbers are not an exhaustive additive reconstruction. The optimized
`target_index_validation_ms` is a **derived sum** of disjoint structure, backing,
prefix-precompute and identity-setup intervals in breakdown; it is null in minimal
and does not include every candidate-loop instruction. Source time is never
added to warm request time; nested selector time is never added to request time.

## Shared preconditions and parity gates

Each length/storage/repetition builds one source catalog. Legacy and optimized,
all requested coverages and instrumentation passes share its actual tensor
objects. The fast index is constructed once per source snapshot when requested.
For every target, the runner clones engine policy/history/cluster state outside
timing while explicitly aliasing model, tokenizer, encoder, adapter and all raw
source tensors. Sidecar policy counters/impacts/clocks are reset to source state.
This preserves identical preconditions without copying QKV or letting earlier
queries alter later admission/selection conditions. Candidate restoration cost
is experimental setup, not serving latency.

Source capture uses the same existing detailed no-reuse source path as B3-A.
Model/tokenizer/encoder revisions, controlled adapter parameter hashes, code
hashes, seed and CUDA/GPU/driver/backend flags are recorded. Defaults remain
FP16 OPT-2.7B, eager, output attentions, use_cache false, batch-one prefill.

Selector/pass/coverage order reverses each repeat; warmup and measured groups
are separate. The FULL reference uses the same native minimal helper once per
source group and also provides logits. It precedes selector requests; its ratio
is supplemental, not a fully order-balanced FULL break-even study. The primary
paired comparison is optimized versus legacy at the same repeat/coverage/pass.
Use B2 for a dedicated FULL break-even experiment.

Selection parity checks the full ordered source-query/position/token sequence,
mask, **exact selected tensor object identity**, and all validation/audit rows.
Guarded shared tensor identity establishes equal QKV values without timed
CPU/GPU tensor copies. Actual executor records validate native/skipped rows;
resident bytes must remain unchanged. Model parity is independently evaluated
outside timing at `atol=rtol=1e-3`, with exact equality also recorded.

Save FULL and legacy-relative max error, relative L2, KL and argmax agreement.
`full_error_metrics_equal` reports whether the numerical errors relative to FULL
match across selectors. FULL allclose failure is retained and never counted as
strict-correctness reuse success. Selector equivalence can pass while FULL
correctness fails; these are distinct conclusions.

Missing pairs are UNPAIRED. Selection/output parity failure prevents a validated
selection speedup claim; breakdown parity failures also disqualify the matching
minimal condition. Reports distinguish `selection_optimization_success` from
`end_to_end_improvement`. Faster selection with no warm-request improvement is
retained. No safety claim follows from either flag.

## Storage, lifetime, and cost interpretation

`additional_qkv_bytes` is zero: the index retains original view tensors and
owners. `additional_index_memory_bytes` is a reachable Python object-size
**estimate**, not allocator/RSS peak measurement; shared pre-existing objects and
QKV payloads are excluded where identified. It may conservatively count some
shared metadata. Do not interpret it as CUDA memory. `catalog_bytes` uses the
existing unique-backing-storage physical accounting.

Index owners/views remain strongly referenced through all paired requests and
parity checks. All transient selections and index/catalog references are released
before the next source build. Tensor writes or layout rebinding fail closed.
No tensor/attention/hidden/logit dump is saved.

Source/index costs repeat as metadata on each row with `source_snapshot_id`.
**Count each source snapshot once**, not once per selector or coverage. Index
build includes validation and metadata accounting; catalog build includes raw
copies. Reported warm speedups exclude these one-time costs explicitly. For N
uses, amortization can include `(source + catalog + index)/N` separately; that is
cost arithmetic, not a measured long-running serving cache.

## CLI and artifacts

The default selector is `legacy`. The independent runner itself is an explicit
oracle experiment; `optimized` and `both` opt into the new index. Downloads remain
disabled unless `--allow-download` is supplied. On SERAPH:

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Smoke, with reduced repetitions and cached OPT-125M weights:

```bash
python scripts/75_benchmark_r1b3_selection.py --profile smoke \
  --model facebook/opt-125m --dtype float32 --prompt-lengths 32 \
  --occurrence-selector both --selection-instrumentation both \
  --output-dir results/r1b3/b1_selection/smoke125m_v1
```

Primary 128/256-token GPU-cache MAX comparison, warmup 3 / measured 10:

```bash
python scripts/75_benchmark_r1b3_selection.py \
  --occurrence-selector both --selection-instrumentation minimal \
  --output-dir results/r1b3/b1_selection/minimal27b_v1
```

Combined separate breakdown/minimal passes for overhead and parity:

```bash
python scripts/75_benchmark_r1b3_selection.py \
  --occurrence-selector both --selection-instrumentation both \
  --output-dir results/r1b3/b1_selection/breakdown27b_v1
```

Optional expanded sweep (only after the primary comparison):

```bash
python scripts/75_benchmark_r1b3_selection.py --profile full \
  --prompt-lengths 32,64,128,256 --coverages 0,.25,.5,.75,max \
  --cache-devices cpu,cuda --occurrence-selector both \
  --output-dir results/r1b3/b1_selection/sweep_v1
```

CPU smoke requires installed Torch/Transformers/PEFT and cached models; add
`--device cpu --cache-devices cpu`. Missing packages or weights fail explicitly.
Use original revision/seed flags for historical comparison; no automatic historical
reproduction claim is made. Long jobs use the user's scheduler setup; no job
script is created.

Each new output root contains the eight requested files: `manifest.json`,
`selection_parity.json`, `selection_breakdown.csv`, `latency_comparison.csv`,
`coverage_comparison.csv`, `correctness_comparison.csv`,
`index_storage_accounting.csv`, `summary.json`, plus checksummed atomic pairing
groups under `checkpoints/`. Reports include mean/median(p50)/p95/std, paired
selection/request ratios and deltas, coverage and correctness. Failed parity
rows are preserved; failure status is not hidden by better timing.

Existing directories are rejected. Repeat the exact command with `--resume` to
verify code/config/runtime and checkpoint checksums, rewarm, and skip complete
measured groups. Duplicate row IDs are rejected. Incomplete groups rerun in full.
Preflight/OOM/interruption/failure statuses are saved; no partial group becomes a
measured sample. Changed code needs a new root. Previous B1/B2/D1/B3-A artifacts
are never overwritten.

## Local validation and unexecuted GPU work

```bash
PYTHONPATH=src:tests python3 -m unittest \
  test_r1b3_fast_selection test_r1b3_occurrences test_r1b2 test_r1b1_d1 \
  test_r1b1_impact test_m9a2_impact_audit test_m85_alignment test_r1_raw_semcache -v
python3 -m py_compile scripts/75_benchmark_r1b3_selection.py \
  src/semcache/diagnostics/r1b3_fast_selection.py tests/test_r1b3_fast_selection.py
```

Local suite: **63 passes, 32 skips, 95 total**. New tests: seven dependency-free
passes and four tensor/model skips. The randomized differential test covers
1,600 selector combinations from 80 generated multi-source fixtures, varying
prefixes, target lengths and coverage; separate cases check adapter/namespace,
query tie order, 128/256 coverage, profiling parity/restoration and invalidation.
These use clearly labeled **fake tensor metadata**, not simulated evidence of
real QKV execution. Optional real tensor and tiny OPT CPU/CUDA tests cover QKV
identity/content, lifetime, `.data`/in-place invalidation, policy parity and actual
projection rows. Torch/Transformers/PEFT are absent locally, so they have not run.

No selection speedup, warm-request speedup, GPU 126/255-token preservation or
FP16 correctness success is claimed from this implementation. The historical
FULL mismatch is not resolved by a selector change. B3-B2 fetch/assembly work
remains out of scope; choose its targets only after measuring the remaining
minimal selector/guard and mixed-executor costs on SERAPH.
