# I5R-4 integrated cost-aware cache campaign

**LOCAL IMPLEMENTATION; GPU_NOT_RUN.** Existing BE/NF1 evidence is read-only.
The historical BE source `f7157c9b8b97fdce56f87dcbc3566cb677404bb8` is never the
new implementation SHA. The operator reviews and commits all additions first;
only that new commit may prepare/seal the campaign. Generated outputs need no
Git commit. No model/data download, remote deployment, submission or inference
is performed by CPU preparation or validation.

## Architecture and reviewed interfaces

The original I5 runtime, generation, protocol, semantic index, Raw-QKV storage,
CPU LoRA, native projections and numerical thresholds are unchanged. Review
covered BE/3B/common/analysis job wrappers, BE preparation/runtime/analysis,
I5 workload/controller/cache/generation/protocol, multi-controller, R5 controller,
B1 multi-entry storage and B2 proof/KV conversion, and NF1 capture limitations.

Relevant actual interfaces:

- `I5Controller.decide(req)` performs registry/physical validation, TinyBERT
  assignment, w=3 discovery and exact completed-source eligibility; only a valid
  `owner.selected()` returns an entry/grant/actual HF past.
- `I5Controller.admit(..., completed=True)` rejects HITs and oversized requests;
  a normal complete standalone miss is eligible. A1280-token full prompt is too
  large for this512MiB cache. No new full-prefix recomputation is introduced.
- `I5Owner.admit()` validates actual all-layer Raw/attention KV proof **before**
  its `bind()`/reset/insert. `I5Owner.remove()` invalidates the semantic index,
  releases R3-owned tensors/seals, deletes proofs, then sends actual UD
  `EVICT_PREFIX`. `MultiPromptCache.insert()` otherwise evicts implicitly by LRU.
- P1..P4's additive owner pre-selects victims at that post-proof `bind()` boundary
  through the same remove chain. It asserts insert did no additional eviction.
  P0 uses the original owner directly. Process-local factory injection is scoped
  to the dedicated opt-in worker and restored in `finally`; ordinary workers do
  not import it. No tensor arithmetic or protocol replacement is involved.
- `generate()` still completes the actual request on the UD before admission,
  consumes actual prefix K/V, saves full-vocabulary logits and counts real
  suffix-only projection/CPU/TCP work. Cached Q is stored, never consumed.
- The original `audit_adapter()` verifies real TCP ledgers, UD isolation,
  forward order, positions, per-step projection counts and strict sampled tensor
  identities. New CPU replay checks policy state and exact eviction/TTL victims;
  P0/P5 additionally run the unchanged legacy cache replayer. Full tensors on
  every scalable request are **NOT_SUPPORTED**, not silently called a strict audit.

## Predeclared policies

| Policy | Admission | Retention | Capacity eviction | Traffic |
|---|---|---|---|---|
| P0 | Original I5 | No TTL | Original LRU | Sources + scored requests |
| P1 | Causal cost score | No TTL | Original LRU ranking | Same as P0 |
| P2 | Original I5 | Request-idle TTL | Original LRU ranking | Same as P0 |
| P3 | Original I5 | No TTL | Future value per byte | Same as P0 |
| P4 | Causal cost score | Request-idle TTL | Future value per byte | Same as P0 |
| P5 | Disabled | Empty | None | Scored targets only, Full Recompute |

P5 has different arrival traffic; it is a proactive-source economic reference,
not an identical-arrival policy comparison. P0/P4 is primary; other policies are
ablations. Source-inclusive spans versus P0 and versus P5 are separate tables.

Pinned cost estimates come from SHA/Git-verified BE/NF1 records, not numbers
copied from a prompt. Predicted saving is nonnegative mean observed per-HIT
target saving against A0; incremental admission is the measured source admission
interval, explicitly an instrumentation-inclusive proxy. BE contains one context,
so this is **cost calibration only**, not a learned general reuse probability.
The source forward itself is excluded from incremental admission. All proactive
source/reentry forwards remain charged to measured sequence economics.

For candidate bytes `b` and prefix-token digest `h`:

```
expected(h) = prior_expected_reuses + 0.5 * min(2, observed_past_hits(h))
retention  = retention_ms_per_MiB_request * (b / MiB) * idle_prior
admit_net  = expected(h) * pinned_saving_ms - pinned_admission_ms - retention
future_value_per_byte = (expected(h) * pinned_saving_ms - retention) / b
```

Defaults: optimistic prior2 reuses; idle prior4 requests; retention1ms/MiB/request;
exploration every third completed standalone candidate, starting with the first.
Oversized candidates never admit. A nonpositive score admits only on declared
exploration positions. Oversized standalone misses advance the visible candidate
counter but cannot allocate. Priors/thresholds never use future trace, references,
future-use annotations, quality results or future accesses. Reentry preserves
only already observed hit history keyed by actual prefix-token SHA.

TTL is per adapter's request index, excluding warmup. It expires **before lookup**
when `current_index - last_access_index > ttl(entry)`. Hits reset last access.

```
economic_ttl = clamp(floor(expected(h)*saving / retention_cost_per_request), 2, 8)
ttl(entry)   = min(economic_ttl, min(8, idle_prior + observed_entry_hits))
```

Wall-clock residency is descriptive from admission/release timestamps, including
right-censored live entries at the final request; it is never a TTL operand.
Eviction ranks ascending future-value/byte, then older last access, then exact
entry SHA. Paid admission cost is sunk and excluded from retention ranking.
P1/P2 retain LRU for capacity.64-entry and512MiB limits are always enforced.

Receipts separate engine compute, retention/expiration, and original admission
intervals. Receipt IO and synchronization perturb measurements; component timers
are nested in application time and must not be added to it.

CPU validation now seals `i4_policy_discrimination.json` and binds its SHA in
`i4_cpu_audit.json`. `inspect` prints the independently recomputed report. It lists
all admission decisions/scores by policy, capacity denials separately from eligible
512-token source denials, actual-use-of-exploration flags, simulated TTL expiry
and capacity victims, and exact-prefix opportunities retained/lost. Victim keys
are simulated token digests, not physical cache grants. The reporting oracle may
remember previously completed capacity-eligible sources; the on-line Engine is
unchanged and never receives that oracle or future requests.

Pairwise P0..P4 comparisons use action outcomes on identical arrivals, not score
values, unused flags or policy names. P5 is labeled
`NOT_COMPARABLE_DIFFERENT_SOURCE_TRAFFIC`. Measurement and sentinel reports are
separate; sentinel differences cannot make a measured ablation count as exercised.
An ablation with identical actions to P0 is explicitly
`NOT_DISCRIMINATED_BY_WORKLOAD`, even when its admission score was evaluated or
it denied oversized1280-token prompts exactly as the baseline would.

In particular, positive expected value for every eligible512-token source can
make P1 identical to P0. A valid CPU audit is not proof that every mechanism was
discriminated. Review `not_discriminated_ablations` before GPU approval. Parameters
are never adjusted by validation. Any alternative prior/cost requires a scientific
justification, operator pre-registration and a fresh sealed plan under reviewed
source; do not change the old plan or select parameters from future trace accesses.

## Fixed real-data workload

Preparation reconstructs Qasper prompts from sealed original records using the
unchanged question/template and context-only head/tail policy, input limit1280,
and target generation limit32 (explicit bounded option up to64 if all preflights
accept it). Existing original IDs, references, context hashes and30-user/three-
adapter routing are retained. This is **TRANSPORT_LIMITED, CONTROLLED PROACTIVE
REUSE**, not natural whole-prompt reuse. NF1 does no GPU replay here.

The plan excludes BE calibration contexts. Each trace selects three distinct
context groups with distinct512-token prefixes in one original adapter. Each has
two genuine distinct questions routed to distinct authorized same-adapter users.
There is no forced rerouting. Different trace units share neither selected
positive contexts nor source-prefix hashes; their negative contexts are distinct.

Predeclared measurement arrival pattern (19 requests,20 when a real foreign
same-prefix request is available):

```
S0, T0a, T0b, S1, T1a, S2, T2a,
N, N, N, N, N, N,
T1b, T2b, S0_reentry, T0a_repeat, T0b_repeat, N,
[optional original-routed cross-adapter negative]
```

Each execution has a unique request ID and sequence number. Repeats keep actual
sample/group/question/reference/token/route identity. Six genuine negative
repeated requests create request-index idle gaps, not synthetic filler. Source
tokens are sliced directly, never decode/re-encoded; decoded text is semantic
input only. Reentry is a separately computed complete real source, not injection.
All actual semantic discovery, misses, grants/admissions and evictions are left
to the runtime. No HIT is guaranteed by static checks or the simulator.

Three512-token sources each store `512*3*32*2560*2 = 251658240` bytes =240MiB
of Raw Q/K/V. Two occupy480MiB; three cannot fit512MiB. Stored Q remains in byte
accounting even though only K/V is consumed. Each target has768 suffix tokens
if a genuine512-token HIT is granted. No padding/FP32/fidelity intervention.

Per measurement trace/policy P0..P4:19 requests,4 source executions (3 unique
prefixes),15 scored observations,7 unique scored samples,8 repeated observations.
Optional foreign negative adds one scored/unique sample. P5 removes all4 sources
and preserves scored order/policies. Targets in a trace use the same32-token
limit; actual EOS termination is recorded. No score is assumed meaningful merely
because the generation budget is larger than historical8-token smoke.

Initial proposal:4 independent trace units x2 balanced order rotations x6 policies
=48 measurement model/session loads. Six isolated fresh strict sentinel loads
follow, one per policy: source/authorized HIT-opportunity/mismatched MISS; P5 has
only its two scored references. Sentinels use8-token target limits and are excluded
from timing/quality benchmark aggregation. No sentinel can warm measured cache.
The sentinel P0 physical HIT must be independently exercised for campaign audit
completion; numerical failure is a valid result, not tampering.

With no optional foreign request:880 measurement requests +17 sentinel requests
=897 physical executions; independent positive contexts12, independent statistical
trace units4, not8 rotations or hundreds of target requests. More/less favorable
outcomes are never hard-coded. A full genuine selection table and all source/
target IDs/routes/lengths/hashes are printed by `inspect` after real preparation.

If suitable groups, prefix distinctness, original routing or budget are inadequate,
prepare rejects. The operator may explicitly choose `--trace-units 3` or `2` and/or
`--rotations 1` in a **new** preparation before sealing. It never silently shrinks
or pools a partial campaign. Default4x2 feasibility is not established locally.

The current **static template arithmetic**, without real data/calibration, estimates
272.98GiB and23463 maximum forwards for4x2 (no optional foreign targets). Its
4491.6s CPU verification forecast exceeds the3600s bound, so this proposal is
rejected before sealing by the present conservative estimator.3x2 similarly
forecasts3685.3s. Explicit2x2 gives158.95GiB,11791 maximum forwards and2879.0s
verification forecast; it still needs actual group/GPU-time/disk readiness.
These are estimates, never benchmark results. The example below explicitly
selects2x2 for this reason; the CLI retains4x2 as the original proposal default.
Two independent traces cannot support an uncertainty interval under the declared
minimum-three-trace bootstrap rule.3x1 is an explicit alternative with three
independent units but no repeated rotation; neither establishes confirmatory NI.
For2x2 without foreign negatives:24 measured children +6 isolated sentinel
children,457 physical requests,11791 maximum forwards,6 positive context groups,
2 independent trace units. Its hard child budgets total480GiB plus32GiB reserve
(512GiB free required), distinct from158.95GiB estimated output. The declared
full-prefill GPU estimate is12726892032 bytes (11.85GiB), below the unchanged
22GiB limit; actual peak remains unmeasured. Unknown BE rates still determine
whether the conservative GPU-time bound accepts this explicit smaller proposal.

## Resource preflight and evidence

One RTX3090 (compatible class, actual UUID pinned across the campaign),4 CPUs,
64GiB host memory. The allocation check exits before ES starts; the coordinator
has no live CUDA context. Every child independently reloads one bare FP16 model,
starts three CPU-only UD processes, warms, executes sequentially, and exits.
No GPU processes overlap and no live cache is restored across children.

Per-child unchanged I5 resource estimator checks full context<=2048 and GPU peak
<22GiB. Host audit peak estimate is8GiB +6 times the conservative largest child's
JSON/context estimate, required<56GiB. Available host reserve32GiB is checked on
GPU allocation; Slurm allocation must be64GiB. Estimated host memory is not a
guarantee for an arbitrary changed Python/library environment.

GPU time uses the maximum archived per-token prefill/decode/admission/other costs,
no assumed HIT savings, safety factor>=1.5 (default2), and startup floor180s per
child, plus900s provenance/preflight allowance. Total must fit68400s. Independent
verification estimate must fit3600s using a declared128MiB/s hash/read assumption,
.06s per forward receipt and900s provenance margin. Actual analysis/verification
also have hard timeouts. Inspect these assumptions on SERAPH before approval.

Existing bounded16GiB evidence budget per child (including accounting reserve),
54 proposed children =>864GiB hard budget plus32GiB free disk reserve; actual
estimated evidence/logits/forwards are also reported. Per-child budget is enforced
by the original cross-process writer. This intentionally favors fail-closed
planning over claiming that available24h/disk is unlimited. Source inventories
and all child byte hashes are checked; no entire >128GiB campaign `Evidence`
object is constructed. Coordinator seals reference separate <=16GiB child seals.
Compact analysis JSON is bounded, not a copied tensor archive.

One job uses23:50:00. Stage limits: inspect15m, GPU19h, analysis3h, verification1h,
report20m, five30s kill-after reserves =>12m30s remaining allocation headroom.
Time-bound failure/incomplete evidence never becomes campaign PASS. A failed
attempt is preserved and any new run needs a fresh path; no automatic retries.

## Quality, numerical fidelity and decisions

Unchanged Qasper QA F1 is reported per genuine sample and per context on0..100.
Repeated executions/rotations are averaged within original samples, then contexts
and trace clusters. Request-weighted scores are separately descriptive. Predeclared
non-inferiority margin:2 F1 points against P0 and P5 separately;95% paired trace-
cluster bootstrap after rotations collapse,2000 draws. CI uses trace clusters
because three contexts compete in one cache; requests are not independent.
Fewer than8 independent traces => **DESCRIPTIVE_ONLY**, even if a small bootstrap
interval looks favorable. This bounded campaign cannot prove general quality NI.

All scored full-vocabulary prefill/decode logits use the unchanged frozen gates:
atol=.001, rtol=.001, max_abs=.01, relative_L2=.001, last-token KL<=1e-5, last-token
argmax agreement required. Token/text equality, reference F1, and numerical
fidelity remain separate. No numerical FAIL is converted into PASS.

Report raw source-inclusive span, priming costs, negative intervals, target
intervals, between-request gaps, warm-positive and actual-HIT target latencies,
TTFT/E2E, model execution, LoRA, semantics, reconstruction, policy/IO overhead,
wire bytes and peak memory. Initialization/warmup/process lifecycle are separate.
Application timing is instrumentation-inclusive; locality is localhost application
TCP framing, not a remote WAN. Never add nested components to E2E.

`EVIDENCE_VALIDITY=PASS` means independent replay/inventories are valid.
`FUNCTIONAL_PHYSICAL_REUSE`, sampled cache integrity, numerical fidelity,
token/text, task quality, warm-target and amortized benefit have separate results.
Correctness-qualified speedup is NOT_ESTABLISHED in this bounded descriptive
campaign. Native PEFT strict parity and kernel root cause remain unresolved.

## Local CPU commands (no models or GPU needed for fixtures)

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r4 -q
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r4 test_r5_i5r3bbe \
 test_r5_i5r3b test_r5_i5r3a test_r5_i5_controller test_r5_i5_preflight \
 test_r5_multi test_r5_i5r_nf1 test_r5_i5r1 test_r4b1_cache test_r4b2_cache -q
PYTHONPATH=src:tests python3 -m unittest test_r5_i5_processes -q
bash -n jobs/r5_i5r4_integrated.sh
python3 scripts/148_r5_i5r4.py prepare --help
```

## SERAPH CPU preparation (after your reviewed commit/push/pull)

These are operator commands, not commands executed during local implementation.
Read inputs explicitly; no guessed3A/NF1 directory is substituted. Required assets
are the existing real sealed datasets, expanded adapters/base, local OPT tokenizer/
weights and pinned CPU TinyBERT snapshot plus compatible installed OPT packages.
Historical Git objects must be present; missing objects fail, not a source bypass.

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export I5_CONDA_ROOT="$(conda info --base)"
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=''
export I4_SOURCE_COMMIT="$(git rev-parse HEAD)"
test "$I4_SOURCE_COMMIT" != f7157c9b8b97fdce56f87dcbc3566cb677404bb8
test "$(git branch --show-current)" = exp/raw-semcache-r5-integration
test -z "$(git status --porcelain --untracked-files=no)"
git cat-file -e "$I4_SOURCE_COMMIT:src/semcache/edgelora/r5_i5r4_policy.py"

read -r -e -p 'Absolute completed 3A plan directory: ' I4_3A_PLAN
read -r -e -p 'Absolute completed 3A audit directory: ' I4_3A_AUDIT
read -r -e -p 'Absolute completed 3A verification directory: ' I4_3A_VERIFICATION
read -r -e -p 'Absolute completed NF1 forensics output directory: ' I4_NF1_AUDIT
export I4_3A_PLAN I4_3A_AUDIT I4_3A_VERIFICATION I4_NF1_AUDIT
I4_3A_SHA="$(python -c 'import os,json,pathlib; print(json.loads((pathlib.Path(os.environ["I4_3A_PLAN"])/"i5r3a_plan.json").read_text())["source"]["commit"])')"
I4_BE="$PWD/results/r5/i5r/controlled_hit/i5r3b/be"
I4_BE_PLAN="$I4_BE/plan_N3_20261106T064627Z"
I4_BE_RUN="$I4_BE/attempt_N3_20261106T064627Z/block_0_153343"
I4_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export I4_EXPERIMENT="$PWD/results/r5/i5r/controlled_hit/i5r4/plan_$I4_STAMP"
export I4_CPU_AUDIT="$PWD/results/r5/i5r/controlled_hit/i5r4/cpu_$I4_STAMP"
export I4_RUN_ROOT="$PWD/results/r5/i5r/controlled_hit/i5r4/runs_$I4_STAMP"
mkdir -p logs "$I4_RUN_ROOT"
df -h "$PWD/results"
free -h

# Mandatory CPU-only offline model/tokenizer/scorer/semantic readiness check.
python - <<'PY'
import os,json,pathlib,sys
sys.path.insert(0,'src')
from semcache.edgelora.r5_multi_artifacts import validate
from semcache.edgelora.r5_i5_preflight import readiness
p=json.loads((pathlib.Path(os.environ['I4_3A_PLAN'])/'i5r3a_plan.json').read_text())
m,_,_,_=validate(p['multi_prepared'])
r=readiness({'qasper':p['datasets']['qasper']['path']},base_prepared=m['base_prepared'],multi_prepared=p['multi_prepared'],count=1,budgets={'qasper':32},input_limit=1280,allow_truncation=True)
print(json.dumps(r,indent=2))
assert r['status']=='READY_FOR_MANUAL_GPU_SMOKE'
PY

python scripts/148_r5_i5r4.py prepare \
 --output-dir "$I4_EXPERIMENT" --source-commit "$I4_SOURCE_COMMIT" \
 --plan-3a "$I4_3A_PLAN" --audit-3a "$I4_3A_AUDIT" \
 --verification-3a "$I4_3A_VERIFICATION" --expected-3a-commit "$I4_3A_SHA" \
 --be-experiment "$I4_BE_PLAN" --be-run "$I4_BE_RUN" \
 --be-audit "${I4_BE_RUN}_be_audit" --be-verification "${I4_BE_RUN}_be_audit_verification" \
 --nf1-audit "$I4_NF1_AUDIT" --trace-units 2 --rotations 2 \
 --quality-tokens 32 --max-prompt-tokens 1280 --quality-margin-f1 2 \
 --expected-reuses 2 --retention-ms-per-mib-request 1 \
 --ttl-idle-requests 4 --exploration-every 3 --runtime-safety-factor 2 --startup-seconds 180

python scripts/148_r5_i5r4.py cpu-validate \
 --experiment "$I4_EXPERIMENT" --output-dir "$I4_CPU_AUDIT"
python scripts/148_r5_i5r4.py inspect \
 --experiment "$I4_EXPERIMENT" --cpu-audit "$I4_CPU_AUDIT" \
 --source-commit "$I4_SOURCE_COMMIT" --require-cpu-validation

# Inspect the sealed discrimination table before manual approval. These are
# STATIC_ONLY opportunities, NEVER measured semantic/physical HITs.
python - <<'PY'
import os,json,pathlib
r=json.loads((pathlib.Path(os.environ['I4_CPU_AUDIT'])/'i4_policy_discrimination.json').read_text())
print(json.dumps({k:r[k] for k in ('scope','physical_HIT_status','policy_summary','ablations','not_discriminated_ablations')},indent=2))
PY
```

If any limit rejects, review the exact reason. An explicit smaller proposed
campaign must use a new `I4_STAMP`, output plan/audit paths and changed arguments
before sealing. Never edit a sealed plan, lower cache/header thresholds, silently
discard groups, or substitute fixtures. No real campaign was prepared locally.

## Single manual Slurm submission

First check partition/QOS permits23:50:00 and the selected permitted node:

```bash
sinfo -p batch_eebme_ugrad -N -o '%N %G %m %c %l %a'
scontrol show partition batch_eebme_ugrad
sacctmgr show assoc where user="$USER" format=User,Partition,QOS,DefaultQOS
sacctmgr show qos format=Name,MaxWall
read -r -p 'Reviewed permitted node (moana-y3 or moana-y4): ' I4_NODE
export I4_NODE
test "$I4_NODE" = moana-y3 || test "$I4_NODE" = moana-y4
export I4_ENABLE_GPU=1 R5_I5_ENABLE_GPU=1 R5_ENABLE_GPU=1
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
# Preserve the I4_EXPERIMENT/I4_CPU_AUDIT/I4_SOURCE_COMMIT/I4_RUN_ROOT/I5_CONDA_ROOT
# exports above. Review inspect output before this ONE actual submission:
sbatch --export=ALL --nodelist="$I4_NODE" jobs/r5_i5r4_integrated.sh
```

No array, nested submission, automatic retry, alternate node fallback or automatic
campaign extension. The job requires actual Slurm job/node/CPU/RAM, all three GPU
opt-ins, absolute explicit paths, clean tracked source and pinned reviewed commit.
Log directory must exist before submission because Slurm opens logs first.
The existing run-root parent must also exist. Run `mkdir -p logs "$I4_RUN_ROOT"`
before submission; the job explicitly checks both and does not create Slurm's
log directory too late. The run invocation now passes mandatory
`--cpu-audit "$I4_CPU_AUDIT"`, just as inspect/analyze/verify do.

Output names are printed by the job:

```
$I4_RUN_ROOT/run_<SLURM_JOB_ID>/
  i4_run_config.json, i4_run_status.json, i4_bundle_manifest.json
  i4_preflight_input.json, i4_preflight.stdout, i4_preflight.stderr
  children/<sealed_child_id>/i4_child_status.json, i5_output_manifest.json
  children/<sealed_child_id>/condition/<original I5 ES/UD evidence>
$I4_RUN_ROOT/run_<SLURM_JOB_ID>_audit/
  i4_results.json, i4_independent_audit.json, i4_summary.json
  i4_policy_totals.json, i4_numerical_fidelity.json, i4_request_metrics.jsonl
$I4_RUN_ROOT/run_<SLURM_JOB_ID>_verification/i4_verification.json
$I4_RUN_ROOT/run_<SLURM_JOB_ID>_report/i4_final_decision.json, i4_report.md
```

Plan: `i4_plan.json`; CPU audit: `i4_cpu_audit.json` plus
`i4_policy_discrimination.json`; all regular output directories
have separate `i5_output_manifest.json` SHA inventories. Coordinator uses its own
child-referencing bundle seal. Engine receipts live in each `i4_policy.jsonl`;
actual backend/environment receipts are separate from historical identities.

The job automatically invokes analyze/verify/report only after prior stages succeed.
For independent manual CPU reruns, use fresh output paths (never existing job audits):

```bash
read -r -e -p 'Absolute COMPLETE I4 campaign directory: ' I4_COMPLETED_RUN
I4_REVIEW="$PWD/results/r5/i5r/controlled_hit/i5r4/review_$(date -u +%Y%m%dT%H%M%SZ)"
export CUDA_VISIBLE_DEVICES=''
python scripts/148_r5_i5r4.py analyze --experiment "$I4_EXPERIMENT" \
 --run "$I4_COMPLETED_RUN" --cpu-audit "$I4_CPU_AUDIT" --output-dir "${I4_REVIEW}_audit"
python scripts/148_r5_i5r4.py verify --experiment "$I4_EXPERIMENT" \
 --run "$I4_COMPLETED_RUN" --audit "${I4_REVIEW}_audit" --cpu-audit "$I4_CPU_AUDIT" \
 --output-dir "${I4_REVIEW}_verification"
python scripts/148_r5_i5r4.py report --experiment "$I4_EXPERIMENT" \
 --run "$I4_COMPLETED_RUN" --audit "${I4_REVIEW}_audit" \
 --verification "${I4_REVIEW}_verification" --output-dir "${I4_REVIEW}_report"
```

Verification recomputes from sealed raw evidence, rechecks causal decisions,
baseline behavior, occupancy, removal controls, all logits/scores and summaries.
It does not trust free-form flags, policy success, or simulator predictions.
Report revalidates the verified result hash and inventories. Numerical, quality
and economic failures stay in the exported decision.

## Remaining limitations

Real selection, local GPU assets, disk capacity and19h feasibility are NOT_VERIFIED
on this workstation. The new implementation SHA is intentionally unknown until
human review/commit. Default group availability may fail due to disjoint calibration,
prefix distinctness, original routing or insufficient records; do not fabricate.
The conservative disk ceiling is large and the bounded CPU replay still parses one
child's original I5 ledgers in memory. Independent analysis time assumptions need
operator review on the actual filesystem. No successful24h finish is guaranteed.

Eight target tokens in sentinels are functional only.32-token measured outputs
are descriptive QA evidence, not a new official LongBench result. Only Qasper is
in this initial campaign; future3C covers broader datasets. Trained personalized
LoRA,30 physical devices, general multi-model benefit and a numerical correction
are not evaluated. No NF1 GPU causality experiment is run in this stage.

## Original implementation local validation and delivery

Added8 files: the four `r5_i5r4*.py` modules,148 CLI, integrated shell job,
`test_r5_i5r4.py` and this guide. No existing tracked production source or
historical evidence changed. All additions await operator review/manual commit.

- Focused final:43 PASS /0 FAIL /1 SKIP (44 tests).
- Combined relevant policy/3B/BE/3A/I5/preflight/multi/NF1/R1/B1/B2 regression:
  215 PASS /0 FAIL /2 SKIP (217 tests, includes focused tests).
- Original independent I5 CPU process/localhost TCP regression:10 PASS /0 FAIL /
  0 SKIP. The sandbox attempt failed during fixture setup; the approved run
  outside socket restrictions passed on the final source.
- Combined regression plus TCP, without double-counting focused tests:
  **225 PASS /0 FAIL /2 SKIP** (227 tests).
- Skips: installed native OPT CPU integration and actual HF OPT cache factory
  test; Torch/Transformers/NumPy dependencies are absent. No downloads attempted.
- Six Python AST/syntax/whitespace checks, real CLI dispatch/help, `bash -n`,
  shebang/SBATCH23:50/resource/environment assertions and `git diff --check`:PASS.

CPU proofs use real frozen tensor-owner/controller APIs over explicitly synthetic
ArrayTensor fixtures; all five reuse policies pass independent semantic/physical
event replay, including the unmodified legacy P0 replay. They do not establish
OPT FP16 GPU cache reuse or numerical fidelity. Simulation may show fewer late
opportunities for TTL/value policies; that is a predeclared tradeoff, never a
reason to tune parameters using future requests or suppress negative outcomes.

Real BE/NF1/3A inputs are absent locally, so real preparation, CPU asset/resource
validation and new GPU results remain **NOT_RUN**. No plan, preapproval or GPU
success is claimed from fixtures. No Slurm submission, remote access or Git push.

## Pre-GPU review patch validation

The reviewed wrapper had omitted `run --cpu-audit`, lacked the required log
directory check, and used unbounded inspect/120s kill-after settings inconsistent
with its resource accounting. The patch adds the missing argument/check and
uses the already-declared inspect15m/five30s cleanup reserves. The automatic
19h/3h/1h/20m run/analyze/verify/report chain and23:50 allocation are preserved.

New tests execute the actual shell script with parser-only command recorders and
the real148 argparse parser. They never call inference or Slurm. All five public
invocations must parse; intentionally removing `run --cpu-audit` rejects in the
GPU_CAMPAIGN stage and prevents later stages. Missing logs rejects before CLI
dispatch. Existing static assertions were retained, with no weakening.

Final CPU checks for this patch:

- Focused I4:56 PASS /0 FAIL /1 SKIP (57 tests).
- Relevant combined regressions, including focused:228 PASS /0 FAIL /2 SKIP
  (230 tests; do not double-count focused results).
- Required CPU audit invariant entry point:30 PASS /0 FAIL /0 SKIP, overlapping
  the suites above. This now includes policy-discrimination tests.
- CLI help/dispatch, actual shell invocation/parser checks, `bash -n`, Python
  syntax/whitespace and `git diff --check`:PASS.
- Native OPT CPU and actual HF OPT cache tests remain NOT_RUN locally because
  Torch/Transformers/NumPy dependencies are unavailable. These must run in the
  SERAPH semcache environment; real prepare still requires installed OPT checks.

The fixed2x2 template was rechecked:30 child budgets *16GiB +32GiB reserve =
**512GiB required free disk**, and estimated verification **2878.99s** under the
existing declared IO/CPU assumptions. Actual available SERAPH disk and GPU-time
acceptance remain NOT_VERIFIED. No cells or policies were removed.

For the positive-score **synthetic CPU fixture only** (saving100ms,
admission20ms, retention.001ms/MiB/request; not campaign calibration or proposed
new settings), four measured cells produce:

| Policy | Eligible source admissions/denials | TTL expirations | Capacity victims | Prefix opportunities retained/lost | Versus P0 |
|---|---|---|---|---|---|
| P0 |16/0|0|8|32/0|Baseline|
| P1 |16/0|0|8|32/0|NOT_DISCRIMINATED_BY_WORKLOAD|
| P2 |16/0|8|4|24/8|Discriminated; no benefit implied|
| P3 |16/0|0|4|28/4|Discriminated; no benefit implied|
| P4 |16/0|8|4|24/8|Discriminated; no benefit implied|
| P5 |0/0|0|0|0/0|Different-source-traffic economic control|

Exploration admissions are zero in that fixture. Separate loss/cold-start tests
verify that exploration and cost denials are reported when they occur. No real
policy parameters or workload changed. Real-data discrimination is **NOT_RUN**;
it must be recomputed from the fresh sealed plan/CPU audit after review and commit.
Identical P1 behavior is a scientific limitation, not permission to retune it
using future workload accesses. Existing plan/audit seals must never be edited.
