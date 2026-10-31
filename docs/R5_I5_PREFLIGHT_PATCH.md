# R5-I5-P preflight and conversation-aware workload patch

IMPLEMENTED. CPU verification results are recorded below. GPU_NOT_RUN. No jobs,
remote repository actions, downloads, pushes, historical edits or threshold
changes were performed. I3F/I4A numerical fidelity remains ROOT_CAUSE_UNRESOLVED;
I3F-B is FUTURE_WORK. clusters=1, authorized adapter namespaces and LRU persist.

## Routing and packing

Routing version SHA256_CANONICAL_DATASET_VERSION_GROUP_SEED_MOD30_V1 hashes the
canonical JSON list `[dataset, version, group_id, seed]` for CoQA/MultiWOZ and
`[dataset, version, original_sample_id, seed]` for the other four datasets.
Interpret the full hexadecimal SHA-256 as an integer; `user = integer % 30 + 1`.
Users 1–10 own A, 11–20 B, 21–30 C. Mapping is independent of block positions,
tracks, repetitions, prefix lengths and scales. Different authorized users can
share an adapter; cross-adapter sharing remains forbidden. This patch does not
guarantee equal adapter load in each small block. The independent audit recomputes
routing for new manifests and verifies immutable matched A/B/C requests.

Selection keeps deterministic group ordering, but numeric turn IDs sort as
integers (1,2,10). Next-fit packing keeps a selected group whole when it fits.
Oversized groups are split in chronological chunks at the configured target
limit; every involved block records group ID, block ordinals, reason and
`cache_survives=false`. The last sample-selection boundary may include only part
of a group to preserve the exact requested count; this is separately recorded
in `partial_selected_groups`. No duplication or block-filling substitution occurs.
Full historical dialogue text remains in each authentic dataset prompt.

## Scale and cache lifetime

New manifests use INDEPENDENT_BLOCK_DATASET_SCALE. They record distinct scored
samples, repetitions, blocks, scored targets, physical requests including priming,
three condition-specific resets per block, lifetime in measured requests and
conversation/dataset continuity breaks. Within-block occupancy/HIT observations
are retained; independent blocks never imply cache survival. A 1000-sample plan
is NOT a continuously growing 1000-request cache session.

A continuous Natural 20-request session fits the existing 20-target bound only
when context, evidence, memory and runtime preflight pass. Controlled blocks may
have up to 40 physical requests (20 priming plus 20 targets). Longer target
sessions remain NOT_SUPPORTED by this patch. Every block reloads a base model
three times across A/B/C and starts three CPU UD processes per condition. Actual
startup/model/session initialization and warmup are reported separately; their
cost cannot be inferred from matched target latency. Planning duration estimates
exclude unmeasured model-loading time. Summed per-request intervals are not
throughput of a single continuously serving system.

Cross-job live GPU cache restoration is NOT_SUPPORTED. Preserving state would
require a separately validated residency/checkpoint contract, tensor ownership,
model/tokenizer/adapter/source identities, authorization, LRU order, integrity
proofs and UD/session grant reconstruction, or one suitably bounded allocation.
No unsafe checkpoint implementation or larger safety bound is introduced.

## Readiness commands and actual blockers

Run from the repository root, with the intended semcache environment active:

```bash
python scripts/123_preflight_r5_i5.py readiness \
  --registry /ABS/REAL_DATASET_REGISTRY.json \
  --base-prepared /ABS/REAL_BASE_PREPARED \
  --multi-prepared /ABS/EXPANDED_ADAPTERS \
  --budgets configs/r5_i5_smoke_budgets.json --input-limit 512 \
  --allow-context-truncation --output-dir results/r5/i5/readiness_FRESH
```

Registry values can be prepared-directory strings or objects with
`prepared_path`. All six keys remain visible: coqa 1.0, snips
2017-06-custom-intent-engines, multiwoz 2.2, qasper/2wikimqa/multi_news LongBench-v1.
Checks are local only and do not modify sources: prepared and original source
inventories, version/revision/split, IDs/references, ontology, eligible counts,
truncation, generation budgets, adapters, local model/tokenizer weights and
pinned CPU TinyBERT revision 34707a33cd59a94ecde241ac209bf35103691b43.

The actual local invocation with no supplied real registry returned NOT_READY:
all six datasets are DATASET_NOT_READY. No verified CoQA dev path was supplied;
CoQA smoke is blocked, with no automatic dataset switch. Model/tokenizer and
real expanded adapters are unconfigured, and installed OPT/TinyBERT dependencies
are missing. These are readiness blockers, not fabricated dataset results.
Each dataset entry prints exact normalization commands and source/version
acquisition instructions. See the integrated report for native export formats.

Known-reference scorer smokes PASS for CoQA F1/EM, SNIPS accuracy/macro-F1,
MultiWOZ 2.2 JGA/slot counts, Qasper and 2WikiMQA F1. MultiNews is NOT_RUN locally:
install **rouge==1.0.1** outside jobs. No alternative scoring formula is introduced.
The installed CPU OPT contract is NOT_RUN locally because torch/transformers/numpy
are unavailable. Real GPU preflight now calls that contract directly and cannot
silently skip it. Repair the semcache environment with repository-compatible
PyTorch, Transformers, NumPy and PEFT (repository LoRA extra pins peft==0.20.0),
matching the exact versions in the real base-preparation identity. Fetch pinned
weights/tokenizer/semantic snapshots outside inference jobs. No dependency
installation or GPU qualification is claimed here.

## Manually gated operator sequence

1. Supply and verify CoQA **dev** and real model/adapter artifacts; run readiness.
2. Prepare one-dataset Natural SMOKE with the command below; inspect its seal.
3. Run first block only; review the sibling independent audit, actual generated
   text and native task scores, token/text equality and frozen numerical gates.
4. Complete the inspector-derived remaining block range only after that review.
5. Prepare and inspect four-family smoke (CoQA/SNIPS/MultiWOZ/one LongBench subset).
6. Prepare separate Controlled plans for 64/128/256/512, retaining actual eligibility.
7. Add the other LongBench subsets for six-key coverage when eligible.
8. Prepare SMALL, MEDIUM, LARGE with manual review between stages. Insufficient
   unique examples fail or require explicit documented count overrides.

```bash
python scripts/118_prepare_r5_i5.py experiment \
  --registry /ABS/COQA_DEV_REGISTRY.json --base-prepared /ABS/REAL_BASE_PREPARED \
  --scale SMOKE --tracks NATURAL --block-size 5 --repetitions 2 \
  --budgets configs/r5_i5_smoke_budgets.json --input-limit 512 \
  --allow-context-truncation --output-dir results/r5/i5/plan_coqa_FRESH
export I5_EXPERIMENT="$PWD/results/r5/i5/plan_coqa_FRESH"
python scripts/123_preflight_r5_i5.py inspect --experiment "$I5_EXPERIMENT"
```

The inspector prints actual block counts by dataset/track/repetition, valid index
bounds, the exact Slurm range, required clean committed HEAD and environment,
physical lifetime/continuity metadata and forward/storage/duration/GPU estimates.
Commit reviewed execution sources before actual submission and re-inspect.
No real plan bounds can be reported before verified datasets/tokenizer are supplied.
Fixed historical array ranges are obsolete after group packing.

For SERAPH after preparation and review (these commands are instructions only):

```bash
cd /data/khuss/repos/GP_semcache
export I5_EXPERIMENT=/ABS/SEALED_REAL_EXPERIMENT
export I5_MULTI_PREPARED=/ABS/VERIFIED_EXPANDED_ADAPTERS
export I5_RUN_ROOT="$PWD/results/r5/i5/coqa_smoke_FRESH"
export I5_SOURCE_COMMIT="$(git rev-parse HEAD)"
export R5_I5_ENABLE_GPU=1 R5_ENABLE_GPU=1
python scripts/123_preflight_r5_i5.py inspect --experiment "$I5_EXPERIMENT"
sbatch --export=ALL --array=0-0%1 jobs/r5_i5_plan.sh
# AFTER review, copy remaining_blocks from the inspector, or its all_blocks
# command for subsequent NEW experiments; never resubmit verified blocks in place.
```

Use `--tracks CONTROLLED --prefixes 64 128 256 512` in a fresh eligible long-context
plan. For later scales use `--scale SMALL`, `MEDIUM`, `LARGE` and fresh roots;
inspect every plan to derive its array bounds. The generic launcher validates
sealed inputs and rejects negative/out-of-range indices. Existing specialized
launchers retain scale/dataset-count restrictions. No launcher submits other jobs.

## Interpretation and claim safeguards

Native task scores, generated token/text equality and frozen numerical tolerance
remain independent. Reduced TCP traffic alone is never latency speedup; numerical
failure never qualifies a speedup. Latency retains model/application TTFT and E2E,
semantic/lookup/reconstruction/admission and nested instrumentation overhead.
Startup and warmup remain separate. Natural data, Controlled priming and targets
are separately labeled; context-limited LongBench is a derived evaluation, not
an official full-context benchmark. Adapters remain untrained synthetic rank-8
matrices; 30 logical users use three physical CPU UD processes per condition.

## Changed and added files

Changed: `src/semcache/edgelora/r5_i5_workload.py` (routing/packing/lifetime),
`r5_i5_analysis.py` (routing audit and timing scope), `r5_i5_summary.py` (scale
semantics artifact), `scripts/119_run_r5_i5.py` (mandatory installed OPT contract),
`jobs/r5_i5_common.sh` (sealed and nonnegative array bounds),
`tests/test_r5_i5.py` and `tests/test_r5_i5_processes.py` (identity-based fixtures),
and `docs/R5_I5_INTEGRATED_EVALUATION.md` (superseded fixed ranges).
Added: `src/semcache/edgelora/r5_i5_preflight.py`,
`scripts/123_preflight_r5_i5.py`, `jobs/r5_i5_plan.sh`,
`tests/test_r5_i5_preflight.py`, and this report.

The data schema, request contract and physical runtime needed no changes.
All frozen R3/R4/I3P/I3F/I4A source and historical evidence remain untouched.

## Final validation and actual inspection results

Final selected tests: **133 total, 131 PASS, 0 FAIL, 2 SKIP**:
111 unit/regression cases (109 PASS, two optional dependency SKIPs), 12 legacy
I4A process cases PASS, and 10 final I5 process cases PASS. Known scorer and
readiness tests add nine focused cases. Both optional dependencies remain
unverified: installed OPT and pinned ROUGE. They are required readiness gates,
not qualifying PASS results. Shell syntax, bytecode compilation and diff checks
also pass.

The initial combined process run had one failed I5 assertion that assumed
positional requests always covered three adapters. Conversation-sticky routing
correctly invalidated that assumption. The fixture now selects one turn from
each of three independent conversations; all ten I5 process tests were rerun
and passed. The initial failure log is preserved alongside final logs.

Fresh CPU run: `results/r5/i5/preflight_patch_sticky_tcp_20261101/run`.
Readiness: `results/r5/i5/preflight_patch_readiness_20261101/i5_preflight.json`.
Final independent audit: `results/r5/i5/preflight_patch_sticky_audit_20261101/`.
Final plan inspection: `results/r5/i5/preflight_patch_sticky_inspection_20261101/`.
Validation logs/record: `results/r5/i5/preflight_patch_validation_20261101/`.
Independent CPU audit PASS includes matched A/B/C identities, authorization,
physical fixture K/V consumption and shadow selection equivalence. It establishes
no new GPU or native PEFT numerical parity.

The actual inspected CPU fixture plan has one CoQA Controlled block, repetition
0, three distinct scored samples, three priming sources, six measured physical
requests per condition, three condition-specific cache resets, and array bounds
**0-0%1**. This is TEST_FIXTURE_ONLY and must never be submitted as a real plan.
Real CoQA dev / four-family / six-dataset / later-scale bounds are NOT_RUN until
verified snapshots and tokenizer are supplied; use the inspector on each actual
sealed plan rather than a fabricated fixed range.

Remaining blockers before GPU smoke: verified CoQA dev provenance/registry,
real model/tokenizer and pinned TinyBERT snapshots, real expanded adapters,
matching installed OPT/Transformers/PyTorch/NumPy/PEFT environment, and reviewed
committed execution sources. MultiNews additionally requires rouge==1.0.1.
No GPU job or Slurm submission has been executed.
