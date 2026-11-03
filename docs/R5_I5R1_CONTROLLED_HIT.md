# R5-I5R-1: controlled physical Raw-QKV reuse

Status: IMPLEMENTED; CPU validation results are recorded below. GPU_NOT_RUN.
The research experiment is incomplete until the operator runs and audits GPU
evidence. No universal speedup, Native PEFT parity, trained personalization or
multi-model result is claimed. R5-I3F/I4A ROOT_CAUSE_UNRESOLVED remains visible.

## Design and relationship to R0/I5

The operator reports completed R0 RAW_RUNS diagnosis at
results/r5/i5r/diagnosis/raw_review_01: 60 C targets, 28 candidate-bearing requests,
34 ordinary candidates, zero full-source prefixes/HITs/reused tokens and +5.4593%
application E2E. Every candidate failed both length and token-prefix checks.
That SERAPH directory is not present locally. The new preparation command can
verify its inventory, complete coverage and recorded source-blob verdicts using
--r0-diagnostic; otherwise it explicitly records the operator handoff without
claiming a local re-audit. No historical I5 directory or manifest is edited.

Preparation reuses r5_i5_workload.prepare_experiment and its CONTROLLED requests:
source IDs are target_ids[:prefix], never decode/re-encoded. Decoded source text
is solely TinyBERT input. The source is a real standalone, one-generated-token
distributed request, not target tensor extraction. All source and target tasks
are STRICT_AUDIT before execution. There is one source and 1/2/4 scored targets
per independent block; target repetitions have distinct request IDs, identical
input/policy and one common source. Repetitions are labeled repeated observations,
not independent samples. Gold references remain in the scoring data; prompts
use the existing real-data templates, which do not expose current gold answers.

Each condition executes the identical source/target sequence through
scripts/119_run_r5_i5.py. A avoids semantic calls and physical storage. B uses
TinyBERT, w=3 discovery and the existing shadow LRU without tensor consumption.
C runs normal discovery/eligibility, admission and physical reconstruction,
then suffix-only prefill. No static proof creates a HIT flag or bypasses matching.
Cached Q is stored/accounted but **only K/V are consumed**. Projection/correction
kernel calls still execute for suffix rows; an entire QKV kernel is not claimed
skipped. All arithmetic, eligibility, LRU, clusters=1 and frozen thresholds stay
unchanged.

R1 explicitly assigns source groups to prepared adapters and registry users;
it does not change I5 conversation routing. The initial active adapter is A,
User_01. AUTHORIZED_CROSS_USER uses the next permitted user on the same adapter
(e.g. User_02). The 30-user registry and three distinct synthetic rank-8 adapter
matrices stay intact. The existing runtime still launches three separate CPU
UD processes and warms all three adapters, even if measured requests use only A.
These are not thirty physical devices or concurrent serving.

## Profiles, context and bounds

| Profile | Datasets | Cases prefix:input-limit | Source groups per case/dataset | Active adapters | Targets/source | Blocks |
|---|---|---|---:|---|---|---:|
| SMOKE | Qasper | 64:256 | 1 | A | 1 | 1 |
| COMPARISON | Qasper | 64:256,128:256,128:512,256:512 | 3 | A/B/C across groups | 1 | 12 |
| DATASETS | Qasper,2WikiMQA,MultiNews | 64:256,128:512 | 3 | A/B/C across groups | 1 | 18 |
| AMORTIZATION | Qasper | 128:512 | 3 | A/B/C across groups | 1,2,4 | 9 |

CLI overrides are bounded: 1..6 source groups, supported cases only, 1/2/4
targets/source and 1..16 generated tokens (default 8). SMOKE requires exactly one
pair, dataset, case and active adapter with same-user reuse. There is a fresh
cache/index/model/UD environment per A/B/C condition, with no live cross-job
cache restoration. Maximum physical requests per condition is five. Condition
orders use the predetermined six-permutation I5 cycle.

Input limits are upper bounds, not artificial padding or guarantees that every
short example has exactly that length. Actual selected lengths and positive
suffix lengths are recorded. Context-only head/tail truncation follows I5;
questions/instructions/gold answers are never truncated. Truncated LongBench
examples retain LONGBENCH_DERIVED_CONTEXT_LIMITED_EVALUATION labeling, not official
full-context benchmark status. Input plus generation must fit the actual pinned
model configuration, capped at 2048. Short MultiNews generation may score poorly;
native ROUGE-L still requires rouge==1.0.1 and is never substituted.

Default per-adapter capacity is 512 MiB, total provisioned capacity 1.5 GiB.
OPT raw QKV costs 491,520 bytes/token: 64/128/256-token sources consume
31,457,280 / 62,914,560 / 125,829,120 bytes. The longest source fits without
changing eligibility. Model estimate is 5.4 GB; activation, active KV, transient
QKV and all three capacities are included by existing resources(), with its
22-GiB planning ceiling, 32-GiB CPU reserve, <=1-GiB per-adapter capacity and
8-GiB default evidence budget per condition (hard maximum 16 GiB). Exact estimates,
forwards, all-condition disk budgets and duration assumptions come from inspect.
Actual free GPU/CPU memory, disk (3*condition budget+1 GiB), context, installed
OPT, semantic revision and adapter fingerprints are checked by the unchanged
GPU preflight. No estimate is a measured peak or duration guarantee.

The Slurm wrapper retains batch_eebme_ugrad / moana-y4, one GPU (runtime requires
an actual RTX 3090), four CPU cores, 64G RAM, three hours, 9000-second default
block execution bound and one block per array task. Arrays use %1. Both existing
R5 GPU opt-ins, reviewed commit, required branch and clean tracked tree are
mandatory. TinyBERT/model/tokenizer/scorers must be installed/pinned locally;
jobs run offline and do not download assets.

## Evidence and decisions

The existing I5 independent analyzer is always run, including full strict
Raw-QKV storage/reconstruction/consumed-KV validation, projection and component
shapes, source completion, exact prefix, positions/masks, ownership, CPU payload
integrity, two-ended TCP ledgers and session isolation. The R1 analyzer then
checks the planned source receipt, normal PHYSICAL_PREFIX_HIT decision and
PHYSICAL_REUSE event, resident entry, omitted projection token-role-layer rows,
and actual HIDDEN/DELTAS prefill payload differences. CPU correction omissions
are counted from audited HIDDEN frame shapes, not from cached-token counts alone.

The three pairwise comparisons are A-B, B-C and A-C. Actual wire counters retain
directions, prefill/decode/control and full lifecycle scopes. They measure
localhost TCP application framing/payload, not IP retransmissions or a WAN.
Reference quality, token/text equality and numerical fidelity are separate.
All matched requests, including priming, receive existing frozen logit metrics:
prefill/decode max-abs, relative L2, last-token KL, argmax agreement and failures
after divergent token histories. Additional strict last-prefill-position QKV and
attention/FFN/layer captures locate a first **observed** bitwise/tolerance
divergence. They cannot establish its ultimate cause or replace full logit gates.
The likely-cause hypotheses remain unproven; R5-I5R-2 can investigate them.

Separate verdicts are FUNCTIONAL_PHYSICAL_REUSE, CACHE_INTEGRITY,
TOKEN_TEXT_FIDELITY, NUMERICAL_FIDELITY, NONEMPTY_TARGET_OUTPUTS,
WARM_TARGET_BENEFIT, AMORTIZED_E2E_BENEFIT and STATISTICAL_EVIDENCE. Native PEFT
strict parity and correctness-qualified speedup conservatively remain
UNESTABLISHED. CPU fixtures yield CPU_FIXTURE_ONLY, never GPU PASS. A completed
structural independent audit can coexist with numerical FAIL or negative latency.
A missing expected HIT makes functional_assertions=FAIL and the R1 wrapper
exits nonzero after preserving the report. Numerical FAIL is a scientific result,
retained without replacement output or loosened thresholds.

## Timing and amortization

Warm targets use measured target application TTFT/E2E; priming is excluded from
that scope. The source-to-final-target chronological span includes source,
admission, every target and between-request scoring/serialization gaps. It
excludes startup/warmup and the last target's post-completion serialization.
New additive parent lifecycle timestamps measure each full condition process,
including model load, CPU sessions, warmup, final scoring, shutdown and evidence
instrumentation. Parent reconciliation, offline audit and scheduler queue remain
outside that lifecycle scope. Each scope has its own per-target amortization and
throughput denominator. Initialization/warmup are displayed separately, not
double-counted into the lifecycle timer. Nested model/controller timers are not
added together. Admission duration is not attributed wholly to GPU copying.

Repeat-count comparisons contain only measured points (1,2,4); no interpolation
or assumed token-based break-even. CIs resample source groups within each
dataset/case/repeat setting. Shared-source targets are never resampled as separate
independent units. Fewer than three groups yields descriptive results only;
three groups remains a very small sample, not automatic evidence of superiority.

## Changed architecture

New files: r5_i5r1_plan.py, r5_i5r1_analysis.py; scripts/126_prepare_r5_i5r1.py,
127_run_r5_i5r1.py,128_analyze_r5_i5r1.py; jobs/r5_i5r1_controlled.sh;
tests/test_r5_i5r1.py,test_r5_i5r1_processes.py and this document.
Two small shared changes: r5_i5_evidence.fresh additionally permits the narrow
results/r5/i5r/controlled_hit namespace; scripts/119 adds condition-process wall
timestamps to its existing lifecycle receipt. No model forward or old threshold
changes. Existing source inventories automatically include the new src/scripts/
jobs files. Preparation pins one commit/branch/file inventory; run and analyzer
reject a different snapshot, registry or model/tokenizer/adapter identity. The
existing strict historical analyzer source rule is preserved.

## Local CPU commands

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r1 -v
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r1_processes -v
bash -n jobs/r5_i5r1_controlled.sh
```

Final executed validation: 20 R1 unit tests plus 5 R1 process/audit tests PASS;
the combined unit regression command below ran 176 tests (174 PASS, 0 FAIL,
2 SKIP). Existing I5/I4A TCP process regressions ran 22 tests (22 PASS, 0 FAIL,
0 SKIP). Combined distinct total: **201 PASS, 0 FAIL, 2 SKIP (203 tests)**.
The two dependency skips are rouge==1.0.1 and installed OPT
(torch/transformers/numpy); neither is a passed GPU/runtime dependency check.
CLI help, bytecode compilation, shell syntax and git diff --check also passed.

```bash
PYTHONPATH=src:tests python3 -m unittest \
  test_r5_i5r1 test_r5_i5r0 test_r5_i5 test_r5_i5_controller \
  test_r5_i5_preflight test_r5_i5_hotfix test_r5_i5_gpu \
  test_r5_contracts test_r5_semantic test_r5_controller test_r5_multi \
  test_r5_fidelity test_r5_perf_evidence test_r4c_opt_contract -v
PYTHONPATH=src:tests python3 -m unittest test_r5_i5_processes test_r5_multi_processes -v
```

Retained final fixture evidence: results/r5/i5r/controlled_hit/cpu_verified_20261103/,
including the existing legacy _audit and new _i5r1_audit. Both independent audits
PASS; the two repeated targets consume the same admitted 64-token CPU fixture
prefix and omit verified projection/correction rows and prefill payload bytes.
Nine matched fixture numerical comparisons pass frozen thresholds; that is not
OPT FP16 or GPU numerical validation. The earlier failed sandbox/export attempts
are retained separately and are not counted as successful final validation.
Static plans for all four profiles and manifest-derived arrays are under
results/r5/i5r/controlled_hit/cpu_plan_checks_20261103/, marked TEST_FIXTURE_ONLY.

Process tests need localhost socket permission. They use the actual I5 ES/UD
subprocesses and TCP with tiny fixture tensors, not OPT or CUDA. To retain evidence,
set R5_I5R1_TEST_OUTPUT_ROOT to a fresh directory under the controlled_hit
namespace. With no variable they use disposable temporary files. No real local
model/dataset availability is claimed; actual preparation uses verified SERAPH
snapshots. The CPU source-token test deliberately uses a tokenizer whose
decode/re-encode is not identity, proving authoritative token slicing survives.
The tiny Qasper process fixture emits EOS immediately and has empty decoded
text. That negative output is preserved and scored zero; it is not replaced or
used as successful nonempty generation evidence. Unit tests check both nonempty
known-reference scoring and empty-output detection. Real GPU smoke must inspect
NONEMPTY_TARGET_OUTPUTS independently of physical reuse.

## SERAPH preparation (after operator-managed commit/push/pull)

Use the reviewed commit containing all nine new files and both shared changes.
Do not stage unrelated results. Do not reseal an old plan after changing source;
prepare a new immutable plan. Set up the existing environment:

Operator-only local staging/commit commands, after reviewing the diff:

```bash
git add -- scripts/119_run_r5_i5.py src/semcache/edgelora/r5_i5_evidence.py \
  src/semcache/edgelora/r5_i5r1_plan.py src/semcache/edgelora/r5_i5r1_analysis.py \
  scripts/126_prepare_r5_i5r1.py scripts/127_run_r5_i5r1.py scripts/128_analyze_r5_i5r1.py \
  jobs/r5_i5r1_controlled.sh tests/test_r5_i5r1.py tests/test_r5_i5r1_processes.py \
  docs/R5_I5R1_CONTROLLED_HIT.md
git diff --cached --stat
git commit -m "Add R5-I5R-1 controlled physical prefix reuse evaluation"
```

The operator handles push and SERAPH pull separately. No Git mutation or remote
deployment was executed by this task. On SERAPH after that reviewed update:

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export I5R1_MULTI_PREPARED="$PWD/results/r5/multiuser/expanded_152381_20261031T154848Z/adapters"
export I5R1_SOURCE_COMMIT="$(git rev-parse HEAD)"
python scripts/123_preflight_r5_i5.py readiness \
  --registry results/r5/i5/registry_longbench3.json \
  --base-prepared results/r5/multiuser/expanded_152381_20261031T154848Z/base_prepare \
  --multi-prepared "$I5R1_MULTI_PREPARED" --count 1 \
  --budgets configs/r5_i5_smoke_budgets.json --input-limit 256 --allow-context-truncation
```

Readiness must pass for the selected tasks; R1 preparation validates the exact
selected registry/scorers again. MultiNews needs rouge==1.0.1. Repair dependencies
in the approved semcache environment if missing; none are silently skipped by
real GPU preflight. Check source HEAD against the reviewed SHA, not merely an
arbitrary new HEAD. No scripts execute git push/pull or sbatch themselves.

### Phase A: first one-block smoke only

```bash
export I5R1_EXPERIMENT="$PWD/results/r5/i5r/controlled_hit/plan_smoke_v1"
python scripts/126_prepare_r5_i5r1.py prepare \
  --registry results/r5/i5/registry_longbench3.json \
  --multi-prepared "$I5R1_MULTI_PREPARED" --profile SMOKE \
  --r0-diagnostic results/r5/i5r/diagnosis/raw_review_01 \
  --output-dir "$I5R1_EXPERIMENT"
python scripts/126_prepare_r5_i5r1.py inspect --experiment "$I5R1_EXPERIMENT"
export I5R1_RUN_ROOT="$PWD/results/r5/i5r/controlled_hit/run_smoke_attempt_01"
export R5_I5_ENABLE_GPU=1 R5_ENABLE_GPU=1
mkdir -p logs
sbatch --array=0-0%1 --export=ALL jobs/r5_i5r1_controlled.sh
```

The one-block default has six measured requests across A/B/C plus nine separate
warmup requests. Do not equate one target with an independently replicated study.
The script creates block_0_<job-id>, block_0_<job-id>_audit (legacy strict I5) and
block_0_<job-id>_i5r1_audit. Check COMPLETE and both independent audits, then read
the separate numerical/functional/performance decisions before any expansion.

Operator CPU re-analysis of a completed one-block run (derive the real job path):

```bash
i5r1_runs=("$I5R1_RUN_ROOT"/block_[0-9]*_[0-9]*/i5_block_status.json)
test "${#i5r1_runs[@]}" -eq 1
python scripts/128_analyze_r5_i5r1.py --experiment "$I5R1_EXPERIMENT" \
  --runs "${i5r1_runs[0]%/i5_block_status.json}" \
  --output-dir "$PWD/results/r5/i5r/controlled_hit/review_smoke_01"
```

### Phase B and C: separately reviewed manual expansion

After Phase A review, prepare a new plan; do not automatically proceed. B's
COMPARISON default yields 12 blocks; C's DATASETS default yields 18. Actual
array bounds must always come from the sealed plan inspector. For Phase B:

```bash
export I5R1_EXPERIMENT="$PWD/results/r5/i5r/controlled_hit/plan_comparison_v1"
python scripts/126_prepare_r5_i5r1.py prepare \
  --registry results/r5/i5/registry_longbench3.json --multi-prepared "$I5R1_MULTI_PREPARED" \
  --profile COMPARISON --output-dir "$I5R1_EXPERIMENT"
export I5R1_RUN_ROOT="$PWD/results/r5/i5r/controlled_hit/run_comparison_attempt_01"
i5r1_array=$(python scripts/126_prepare_r5_i5r1.py inspect --experiment "$I5R1_EXPERIMENT" | python -c 'import json,sys; print(json.load(sys.stdin)["valid_array"])')
sbatch --array="$i5r1_array" --export=ALL jobs/r5_i5r1_controlled.sh
```

Phase C, only after a separate review decision:

```bash
export I5R1_EXPERIMENT="$PWD/results/r5/i5r/controlled_hit/plan_datasets_v1"
python scripts/126_prepare_r5_i5r1.py prepare \
  --registry results/r5/i5/registry_longbench3.json --multi-prepared "$I5R1_MULTI_PREPARED" \
  --profile DATASETS --output-dir "$I5R1_EXPERIMENT"
export I5R1_RUN_ROOT="$PWD/results/r5/i5r/controlled_hit/run_datasets_attempt_01"
i5r1_array=$(python scripts/126_prepare_r5_i5r1.py inspect --experiment "$I5R1_EXPERIMENT" | python -c 'import json,sys; print(json.load(sys.stdin)["valid_array"])')
sbatch --array="$i5r1_array" --export=ALL jobs/r5_i5r1_controlled.sh
```

For measured repeat-target costs, separately gated:

```bash
export I5R1_EXPERIMENT="$PWD/results/r5/i5r/controlled_hit/plan_amortization_v1"
python scripts/126_prepare_r5_i5r1.py prepare \
  --registry results/r5/i5/registry_longbench3.json --multi-prepared "$I5R1_MULTI_PREPARED" \
  --profile AMORTIZATION --output-dir "$I5R1_EXPERIMENT"
export I5R1_RUN_ROOT="$PWD/results/r5/i5r/controlled_hit/run_amortization_attempt_01"
i5r1_array=$(python scripts/126_prepare_r5_i5r1.py inspect --experiment "$I5R1_EXPERIMENT" | python -c 'import json,sys; print(json.load(sys.stdin)["valid_array"])')
sbatch --array="$i5r1_array" --export=ALL jobs/r5_i5r1_controlled.sh
```

Add --sharing AUTHORIZED_CROSS_USER only to a new non-SMOKE plan after same-user
review. Each plan/run/review directory must be fresh. None of these steps follows
automatically from a successful earlier job.

Complete-study analysis rejects missing, duplicate/retry, corrupt, incomplete or
mixed-source blocks. Select exactly one verified attempt per block, then:

```bash
i5r1_statuses=("$I5R1_RUN_ROOT"/block_[0-9]*_[0-9]*/i5_block_status.json)
i5r1_runs=()
for i5r1_status in "${i5r1_statuses[@]}"; do i5r1_runs+=("${i5r1_status%/i5_block_status.json}"); done
python scripts/128_analyze_r5_i5r1.py --experiment "$I5R1_EXPERIMENT" \
  --runs "${i5r1_runs[@]}" --output-dir "$PWD/results/r5/i5r/controlled_hit/review_comparison_01"
```

Use a distinct review path for each study/reanalysis. For reviewing only the first
block of a larger plan add --block-only and pass exactly that block: the report
will explicitly say the study is incomplete. Failed execution is restarted from
its beginning in a fresh output attempt with an empty cache, not resumed.

## Artifacts and limitations

Plans: existing i5_workload_manifest.json and i5_dataset_manifest.json plus
i5r1_protocol.json, sealed with i5_output_manifest.json. Static pair proofs and
actual lengths are embedded per block. Runtime retains all existing I5 receipts,
tensor blobs, logs, TCP ledgers, quality and frozen-fidelity outputs. R1 analysis:
i5r1_decision.json, i5r1_independent_audit.json, i5r1_physical_hits.json,
i5r1_numerical_fidelity.json, i5r1_layer_diagnostics.json,
i5r1_request_metrics.json, i5r1_quality_per_sample.json,
i5r1_legacy_summaries.json, i5r1_cost_comparisons.json, i5r1_cost_table.csv,
i5r1_report.md and the SHA-256 output inventory. Analysis never edits inputs.

CPU safety tests cover unauthorized/cross-adapter requests, non-prefix shared
windows, equal-length sources, incomplete/evicted entries and invalid mask/
positions. Runtime retains I5's pre-TCP mapping probes and complete registry/
session/namespace audits. These CPU/probe checks are not claims that a real GPU
foreign-cache HIT or every negative scenario was executed. GPU positive reuse,
numerical results, quality and performance remain NOT_RUN locally.

EXACT_PREFIX_ONLY is FUTURE_WORK: it would need its own explicitly matched
metadata/controller cost model, frozen protocol, safety audit and tests. No
fourth runtime path is introduced. More detailed operation-order causality and
admission profiling belong to R5-I5R-2; this stage cannot establish a copying
bottleneck or a performance break-even without actual measurements.
