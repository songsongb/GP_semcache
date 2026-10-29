# R2-C4 controlled RAW QKV cache-path cost isolation

Inspected checkout: `exp/raw-semcache-r2`, HEAD
`53c4c11d49ac7804fc5f1774c47ec3d7834f729e`, initially clean. Read 92,
C3 Replay/runtime/analysis, C1 runtime, frozen B3 runtime, 77 RequestMapping,
SourceBackingDescriptors, PreparedMappingCache, V2/contiguous and legacy/Batched
executors. Historical GPU artifacts and `jobs/r2c4_samegpu_cache_path.sh` are not
present locally. No frozen Python/results or shell/Slurm file is modified.

The user-reported C3 residual **181.10 ms** at L1024/MAX is an observed RAW
Replay-minus-No-Reuse Replay request difference. It is not an additive set of
independent CPU/GPU costs, nor a new C4 observation.

## Six arms and exact intervention boundaries

| Arm | Timed mapping retrieval | Timed descriptor validation | Assembly |
|---|---|---|---|
| FULL_NATIVE | None | None | Frozen native Full |
| SEMCACHE_NO_REUSE_IMPACT_REPLAY | No physical mapping | None | Frozen native-only Engine wrapper |
| RAW_REPLAY_BASELINE | Original cache.get(validate=False) | Original executor-entry validate | Original contiguous/fallback |
| RAW_REPLAY_SNAPSHOT_ORACLE | Original get | Exact selected snapshot prevalidated outside; one internal call bypassed | Same executor |
| RAW_REPLAY_MAPPING_ORACLE | Verified prebound plan access | Original validate | Same executor |
| RAW_REPLAY_BOTH_ORACLE | Prebound access | One approved call bypassed | Same executor |

Actual frozen path:

```text
semantic/cluster/lookup -> FastOccurrenceIndex.select (includes index.guard)
 -> RequestMapping.select
    -> PreparedMappingCache.get(... mode='contiguous', validate=False)
       -> exact key construction, lookup, prepare_indices if needed, LRU/accounting
 -> RequestMapping.context
    -> batched_v2_projection_path
       -> selected-sequence checks
       -> mapping.descriptor.validate(hits, setup_book)
       -> module patches -> fresh projections + original assembly -> cleanup
```

The **request** cache.get does not perform descriptor validation in this path.
The relevant descriptor call is at executor entry. Selector `index.guard()`,
executor geometry/selected-sequence checks, per-role Source shape/dtype checks,
native fresh GEMMs, LoRA, contiguous slices and grouped fallback remain active.
Snapshot Oracle does not disable every source/selection guard. Mapping Oracle
does not bypass selection or replace the mapping/executor algorithm.

All SemCache arms reuse the unmodified C3 Impact Replay controller. Two actual
untimed ON references per exact condition/base mode provide ordered values,
provenance, policy digests and separate reference costs. No reduction is executed
inside a successful Replay. Model still uses eager attention,
output_attentions=True, use_cache=False and native all-token `[1,L,V]` logits.
There is one OPT model, the same synthetic rank-8 user_a fixture, source QKV
owners and uncompressed RAW storage. No Prefix KV, compression or cross-user mode.

## Oracle preparation, lifetime and fail-closed checks

Before **each** oracle request, outside the primary timer:

1. Use the existing selector to identify expected hits without replacing the
   real semantic/cluster/selection executed later inside the request.
2. Check model/index stamps, descriptor/index/catalog/cache ownership and actual
   canonical Source QKV content against the original hashes.
3. For nonempty hits run the original full descriptor validation on those hits.
4. For Mapping/Both only, call the original cache.get outside timing, then ensure
   device indices are ready. Record actual prebind cache hit/miss and its cost.
5. Verify full target, source/adapter/model namespace, exact hit order,
   absolute positions, prefix tokens, disjoint mask/fresh rows, Source group
   ownership, contiguous/fallback flags and actual index Tensor contents,
   shape/dtype/device. Seal plan geometry, source/index identities and stamps.

During the timed request, actual selection remains. Lightweight exact identity
and hit-sequence checks guard the prevalidated snapshot and prebound plan.
Only the scoped expected descriptor/cache instance may use a shortcut. Unexpected
additional validation, mapping or executor calls fail the experiment. Native
executor sequence/adapter checks still execute.

Afterward, outside timing, run full descriptor/index ownership/stamp checks,
Source QKV content checks and full mapping geometry/index-content checks again.
The executed mapping must be the identical prebound object and unchanged seal.
Original owners remain referenced; no target-shaped QKV copy is prepared.
Warm baseline retrieval is recorded as such. A prebound return uses the original
API compatibility boolean internally, but exported mapping_cache_hit is null and
lookup_kind is PREBOUND_NOT_A_CACHE_LOOKUP. It is never reported as a measured
cache lookup or removal of full mapping construction from an already warm hit.

These oracle shortcuts leave a validation/use gap and are **unsafe to deploy**.
They require an immutable controlled fixture and postchecks; they are not
production configurations. classification=ORACLE_DIAGNOSTIC and all of
safe_reuse_claimed, production_speedup_claimed and correctness_qualified_speedup
remain false.

## Exact parity and zero-reuse controls

Untimed C1/C3 gates validate model/LoRA/head/output settings, actual physical skip,
source capture/backing, exact Impact values/history/CHU/Admission/Eviction/PBR,
and legacy/contiguous parity. C4 then compares every RAW oracle with baseline:
token/window provenance, mapping description, actual coverage and projection
rows, policy proof, reference digest and logits (torch.equal).

Each target's **existing passive** assembled Q/K/V outputs are observed only
after the closed Engine request. Hash one projection at a time; compare every
SHA256 plus shape/dtype/device and final logits. No projection reruns, second
model or QKV dump. This is a byte-fingerprint proof (not an approximate norm).
Real Tensor differential tests additionally use torch.equal. Hash/geometry/Logit
audits are untimed; source/index seals detect .data rebinding and index corruption.
Snapshots retain logical policy clocks; only timing/wall-only metadata is ignored.

FP16 Full-relative Strict FAIL remains distinct from exact baseline/oracle
equivalence. Elementwise atol=rtol=1e-3 is unchanged. A structurally valid
diagnosis can complete with that Full-relative failure. Oracle parity failure,
stale mapping, source mutation, missing evidence, OOM or runtime error cannot
produce COMPLETE. Original exception/traceback, phase/arm, memory and partial
artifacts are retained. No smaller/different workload is retried.

At zero reuse, frozen RequestMapping skips both cache.get and descriptor.validate
and uses the legacy native-only executor. All RAW oracles then perform the same
request path; counters must prove zero get/validation calls. Both interventions
and their interaction are **NOT_APPLICABLE**. Signed timing differences are kept
as negative-control observations, with attributable_difference_ms=null. They are
not called benefits from removing work that never executed.

## Exact timing and separate breakdown

The original synchronized conditional-request wall remains primary. It includes
tokenization, semantic encoding/cluster/lookup/actual selection, request mapping/
validation/setup as executed, Forward/LM Head, Replay and policy plus closing
CUDA synchronization. Source prefill/catalog/index/descriptors, Engine deepcopy/
policy reset/controller preparation, C3 references, oracle prevalidation/prebind,
postchecks, CPU logit transfer and serialization remain outside.

A scoped wrapper marks the **first actual Engine CPUWallTimer** entry/exit, which
is the timer closed by the frozen synchronized engine_request. Method observers
record whether work occurred inside or outside that exact interval. Passive
RequestMapping.report/QKV fingerprinting must occur after it closes. Full keeps
its original native_request timer and never enters the Engine marker.

Minimal arms use only thin counters, boundary markers and necessary oracle
identity checks. They do not use SetupBook/ExecutorBook, CUDA events or profiler
hooks. Wrapper wall remains the broader frozen invoke interval; oracle setup,
postchecks and QKV fingerprints surrounding it are separately recorded and do
not form a service E2E timer. Per-pair quality/equivalence wall, per-condition
untimed gate costs, source and reference costs are explicitly exported and never
amortized into request savings. A controlled exclusion can alter CPU/GPU overlap;
its wall difference is not a kernel-only cost.

Breakdown uses a separate phase at L1024, coverage0/.5/MAX. It executes the same
paths with CPU method spans and existing SetupBook/ExecutorBook. Includes:

- RequestMapping.select and cache.get/prebound dispatch; real cache hit/miss.
- Executor-entry descriptor validation or approved dispatch; retained index guards
  inside selection versus outside-request postchecks.
- Setup/cleanup, module patching, index preparation, full mixed projection,
  fresh native QKV, output allocation and fresh assembly.
- Actual Source resolution, cached gather/slice, conversion (possibly no-op),
  scatter/copy, contiguous/fallback and observed call counters.

CPU host intervals and CUDA event intervals are separate. Events are resolved
after the enclosing request synchronization; there is no per-operator GPU sync.
CUDA intervals can include dispatch gaps and are not kernel self-times. Parent
select contains get; executor setup contains descriptor validation; full mixed
projection contains its children. SetupBook also overlaps method observations.
No nested/inclusive durations are summed into a fictitious total. Unsupported or
unexecuted measurements are null/status-tagged. Inherited assembly byte counters
include logical view volumes and are not allocator peaks. Actual GPU snapshots
and peak observations remain distinct. Profile/breakdown wall is never pooled
with minimal latency.

## Capacity, phases and CLI

Conservative C1 screening, physical Source capacity and inherited R1 preflight are
unchanged; C2 guarded-profile code is untouched. No automatic dtype/length/cache/
attention reduction. Pilot uses reset-aware memory observation. Default primary
cases are L512/L1024 × coverage0/.5/MAX, capacity512MiB. Actual positional limits
are checked. The separate semantic encoder cap is unchanged. Latency requires a
completed fresh **C4** Pilot on matching actual GPU UUID, code/model/adapter/
encoder/settings/storage/capacity/seed, exact condition/source and Impact reference.
Historical C1/C2/C3 Pilot JSON is not repurposed.

Phases: pilot, breakdown, latency, analyze. All requested launcher flags are
implemented. Breakdown defaults to and requires L1024 coverage0/.5/MAX; supplying
matching --pilot-inputs is recommended and validated. A standalone Breakdown can
self-validate with the same conservative preflight/full gates; it cannot approve
a latency run. Analyze requires exactly three completed latency directories and
a compatible completed --breakdown-input. Analyze is CPU-only and never loads OPT.

The separately authored shell file is not in this checkout, so its body cannot
be inspected here. Declared CLI/artifact compatibility is locally tested. No shell
file is created/edited. When that file is present on SERAPH:

```bash
cd /data/khuss/repos/GP_semcache
sbatch jobs/r2c4_samegpu_cache_path.sh
```

Equivalent stages inside one allocated GPU session (separate Python processes;
all output paths must be new; use the existing offline Hugging Face environment):

```bash
python scripts/93_run_r2c4_cache_path.py --phase pilot \
  --model facebook/opt-2.7b --dtype float16 --prompt-lengths 512,1024 \
  --coverages 0,.5,max --device cuda:0 --cache-device cuda --capacity-mib 512 \
  --seed 42 --order-seed 4100 --independent-run-id pilot01 \
  --warmup-runs 1 --measured-runs 3 --output-dir results/r2/c4/pilot01

python scripts/93_run_r2c4_cache_path.py --phase breakdown \
  --model facebook/opt-2.7b --dtype float16 --prompt-lengths 1024 --coverages 0,.5,max \
  --device cuda:0 --cache-device cuda --capacity-mib 512 --seed 42 \
  --pilot-inputs results/r2/c4/pilot01 --warmup-runs 1 --measured-runs 1 \
  --independent-run-id breakdown01 --output-dir results/r2/c4/breakdown01

for run in 1 2 3; do
  python scripts/93_run_r2c4_cache_path.py --phase latency \
    --model facebook/opt-2.7b --dtype float16 --prompt-lengths 512,1024 --coverages 0,.5,max \
    --device cuda:0 --cache-device cuda --capacity-mib 512 --seed 42 \
    --order-seed "$((4100 + run))" --independent-run-id "run${run}" \
    --pilot-inputs results/r2/c4/pilot01 --warmup-runs 3 --measured-runs 20 \
    --output-dir "results/r2/c4/latency_run${run}"
done

python scripts/93_run_r2c4_cache_path.py --phase analyze \
  --inputs results/r2/c4/latency_run1 results/r2/c4/latency_run2 results/r2/c4/latency_run3 \
  --breakdown-input results/r2/c4/breakdown01 --output-dir results/r2/c4/aggregate01
```

Six-arm seeded cyclic Latin blocks counterbalance order; 20 repetitions cannot
divide six perfectly, so position counts differ by at most one. Independent runs
must have distinct process/run IDs and order seeds. Model/source seed stays fixed.
Observed competing GPU processes fail the run; unavailable telemetry is recorded
and cannot establish environmental stability.

## Artifacts, statistics and limitations

All phases: manifest.json, timing_boundary_audit.json. GPU phases additionally
write environment.json, workload_audit.json, correctness_and_gates.json,
impact_references.json (sealed SHA256) and cache_path_decision.json.

- Pilot: capacity_pilot.json; conditions CAPACITY_VALIDATED.
- Breakdown: cache_path_breakdown.json/.csv; conditions DIAGNOSED with verified
  boundaries. Instrumented walls stay separate from latency samples.
- Latency: cache_path_raw_pairs.json, cache_path_session_summary.csv; conditions
  VALID with measured_pairs=20 by default.
- Analyze: cache_path_aggregate_summary.json, cache_path_decision.json;
  GPU_VALIDATION=AGGREGATED_GPU_RUNS describes completed inputs and
  analysis_process.gpu_inference_performed=false describes the analysis process.

COMPLETE requires all gates; successful GPU phases record actual Torch UUID and
GPU_VALIDATION=EXECUTED. Missing dependencies/failures never fabricate that status.
Manifest retains Git/dirty state, code/file hashes, resolved model/tokenizer/
adapter/software/GPU identities, tokens, coverage, policies, reference and memory.

Paired differences: baseline−snapshot, baseline−mapping, baseline−both,
baseline−NoReuseReplay and baseline−Full; interaction = delta_both−delta_snapshot
−delta_mapping. Positive A/B/C favors the oracle; positive D/E means RAW baseline
was slower. Preserve negatives. Equal independent-session weights, per-session
means/wins/variance and exploratory session-mean bootstrap 95% CIs (only three
sessions); no pooled-pair independence or discarded outliers. Zero-reuse oracle
contrasts retain observed noise but no attributed saving. Reject incomplete,
duplicated, mismatched code/GPU/model/endpoint/reference/policy/sample sets and
missing/incompatible diagnostic inputs. No service or correctness-qualified
speedup is inferred from these unsafe oracle controls.

```bash
PYTHONPATH=src:tests python3 -m unittest test_r2c4_cache_path test_r2c4_analysis \
  test_r2c3_impact_ablation test_r2c3_analysis test_r2c2_profile_memory \
  test_r2c1_benchmark test_r2c1_analysis test_r2c1_workloads -q
python3 scripts/93_run_r2c4_cache_path.py --help
git diff --check
```

Scalar/control tests do not constitute GPU evidence. Actual Tensor/executor
differential tests are dependency-gated. Without local Torch/Transformers/PEFT,
GPU_VALIDATION=NOT_RUN: actual CUDA mapping/validation costs, parity, breakdown
overhead and latency outcomes remain to be measured by the separate SERAPH job.

Local verification on 2026-10-29: 16 focused tests, 15 passed and one Torch-gated
Tensor/executor skip. Combined R1/impact, R2-A, B3.5, C1/C2/C3/C4 regression suite:
255 tests, 197 passed, 58 skipped, zero failures. Compilation, CLI help, all six
stage/example argument sets and tracked/new-file whitespace checks passed.
Torch, Transformers and PEFT are absent locally. Existing tracked files and all
shell/Slurm files are unchanged; no GPU job or git push was performed.
