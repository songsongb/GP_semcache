# D5 Light: CPU review and future manual GPU gates

**Current status: PARTIAL_BLOCKED. Do not submit this job template.**

The native Light ES/UD orchestration worker, actual resource/device guards and
native Light parity approval are not implemented. `run` and the shell template
exit with an explicit BLOCKED error, even with an APPROX opt-in. `inspect` cannot
report GPU-ready. Existing Strict workers are never used as a Light fallback.
This runbook supplies preparation commands and the future bounded invocation
shape; it is not authorization to execute GPU work.

## Files, options and defaults

`scripts/159_r5_d5_lightperf.py` exposes `environment`, `fingerprint`, `matrix`,
`prepare`, `cpu-validate`, `inspect`, `run`, `analyze`, `verify`, `aggregate`.
Every command requires `CUDA_VISIBLE_DEVICES=''` in this CPU-only revision.
`prepare` additionally requires offline HF/Transformers settings. It consumes
existing authenticated real I5/model/multi-adapter preparations; it never acquires
data, rebuilds prompts, downloads models or changes original routing.

* Dataset: exactly `qasper`, `2wikimqa`, `multi_news`; count: 5, 20 or 50.
* `--bandwidth-mbps`: omitted = unlimited; choices 200/100/50. Per sending
  channel, per direction; aggregate adapter traffic is not a single shared link.
* `--ud-throughput-factor`: default 1; choices 1/.5/.25. Explicit
  `EMULATED_UD_SERVICE_TIME`, same arithmetic measured on every operation.
* `--ud-operation-seconds`: positive **planning assumption**, not authenticated
  hardware calibration. Unknown slowed-UD budget blocks preflight.
* `--strict-pilot`: optional existing frozen NATURAL `approx_plan.json` directory.
  Exact first-five payload equality gives MATCHED_SELECTION_ONLY. Without it,
  NOT_MATCHED. This does not prove native parity.
* `--continuous-contract`: explicit JSON contract for original blocks merged into
  a new cache lifetime; it binds dataset/count/block IDs/task hash/ordinals/reset
  semantics and `acknowledgement=REVIEWED_NEW_CONTINUOUS_IDENTITY`. It cannot
  authorize duplicate tasks, partial groups, reordered requests or mutated routing.
* `--development`: permits local uncommitted source CPU plans only. Production
  prepare instead requires `--reviewed-commit` and `--reviewed-fingerprint` matching
  this worktree's clean committed HEAD, branch and source content exactly.
* `run --condition` is one condition only. APPROX has a separate explicit
  `--enable-approximate-experiment` option. Execution is currently hard blocked.
  RTT, shared-link scheduling and automatic campaigns are NOT_IMPLEMENTED.

Format/mode: `D5_LIGHTPERF_V1` / `LIGHTWEIGHT_TIMING`. Flat write-once bounded
JSON/CSV/Markdown summaries have a separate `light_inventory.json` with SHA256
hashes. These authenticate local byte consistency and receipt bindings, not an
external signed claim that an experiment happened. No raw tensor/logit files are
accepted. ≤8 MiB per file, ≤2048 files and ≤64 MiB total are hard limits.
`verify` is a Light verifier, not the Strict independent numerical auditor.

## Remaining implementation before any manual GPU command

1. Implement and review native Light ES/isolated UD processes using the audited
   kernel/service. Check source/input fingerprints in each child, matrices and
   model/tokenizer identities, actual resource reserves and original node/backend
   policy, frame/worker/shared allocation deadlines, exception cleanup and seals.
2. Persist the complete compact per-request/run schema, bind request summaries
   to UD compute and ES wire totals, final ACK/STOP/BYE, occupancy and policy
   counts. Measure quality/persistence separately and initialization/sealing outside
   request scope. Compact proof metadata after all original live checks; never
   persist dense tensors. Add measured concurrency, GPU UUID/driver, peak memory,
   errors, deadlines, file/byte usage and contention limitations.
3. Run native tiny OPT Strict-vs-Light CPU parity for FULL/EXACT/APPROX, including
   multi-token decode, ownership/tensor corruption, policy and session continuity.
   Existing Strict native tests alone do not approve the new Light kernel.
4. Complete end-to-end artifact/worker failure tests and receipt reconstruction;
   connect authenticated measured single-axis series to break-even tables/plots.
   Only then implement issuance of CLEAN_NATIVE_CPU_APPROVAL and remove run gates
   in a separately reviewed source change. Current cpu-validate never issues that
   approval, even on a host with native packages installed.

## CPU review on the actual execution host

No installation is part of this task. The current host has no torch,
transformers, numpy, rouge scorer or real prepared manifests. First provision the
already prepared offline assets/environment through the user's normal process.
Do not infer their SERAPH paths from this host or silently substitute remote code.
Read and review the completed worker and commit it manually after authorization.
No commit or push was performed by Codex.

```bash
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 scripts/159_r5_d5_lightperf.py environment
python3 scripts/159_r5_d5_lightperf.py fingerprint
python3 scripts/159_r5_d5_lightperf.py matrix
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest test_r5_d5_lightperf -v
```

The fingerprint includes source/scripts/jobs/tests/configs/docs and root dependency
metadata. After any source change, repeat fingerprint, review, new PLAN and new
CPU audit. Development receipts and old Strict CPU approvals cannot be reused.
An external wrapper must also verify the manual allocation time/node/resource
policy; do not raise budgets to fit a requested count.

## Manual L2 five-request preparation commands for all three datasets

The following are **future manual commands, not executed here**. They remain
blocked by the items above. Assign these variables to verified existing paths on
the actual host, using a fresh unique tag. Qasper and the other pilots may use
different original I5 plans; never guess or rebuild their first-five lists.

```bash
set -euo pipefail
: "${D5_REVIEWED_COMMIT:?reviewed clean Light commit}"
: "${D5_REVIEWED_FINGERPRINT:?fresh reviewed content fingerprint}"
: "${D5_QASPER_I5:?existing original Qasper I5 plan}"
: "${D5_2WIKI_I5:?existing original 2WikiMQA I5 plan}"
: "${D5_MULTINEWS_I5:?existing original MultiNews I5 plan}"
: "${D5_QASPER_STRICT_PLAN:?frozen Qasper five-request plan}"
: "${D5_2WIKI_STRICT_PLAN:?frozen 2WikiMQA five-request plan}"
: "${D5_MULTINEWS_STRICT_PLAN:?frozen MultiNews five-request plan}"
: "${D5_MULTI:?existing expanded multi-adapter preparation}"
: "${D5_TAG:?unique fresh run tag}"
D5_OUTPUT_ROOT="$PWD/results/r5/d5_lightperf"

python3 scripts/159_r5_d5_lightperf.py prepare --dataset qasper --count 5 \
  --original-experiment "$D5_QASPER_I5" --multi "$D5_MULTI" \
  --strict-pilot "$D5_QASPER_STRICT_PLAN" \
  --reviewed-commit "$D5_REVIEWED_COMMIT" --reviewed-fingerprint "$D5_REVIEWED_FINGERPRINT" \
  --output-dir "$D5_OUTPUT_ROOT/qasper_5_plan_$D5_TAG"
python3 scripts/159_r5_d5_lightperf.py prepare --dataset 2wikimqa --count 5 \
  --original-experiment "$D5_2WIKI_I5" --multi "$D5_MULTI" \
  --strict-pilot "$D5_2WIKI_STRICT_PLAN" \
  --reviewed-commit "$D5_REVIEWED_COMMIT" --reviewed-fingerprint "$D5_REVIEWED_FINGERPRINT" \
  --output-dir "$D5_OUTPUT_ROOT/2wikimqa_5_plan_$D5_TAG"
python3 scripts/159_r5_d5_lightperf.py prepare --dataset multi_news --count 5 \
  --original-experiment "$D5_MULTINEWS_I5" --multi "$D5_MULTI" \
  --strict-pilot "$D5_MULTINEWS_STRICT_PLAN" \
  --reviewed-commit "$D5_REVIEWED_COMMIT" --reviewed-fingerprint "$D5_REVIEWED_FINGERPRINT" \
  --output-dir "$D5_OUTPUT_ROOT/multi_news_5_plan_$D5_TAG"

for D5_DATASET in qasper 2wikimqa multi_news; do
  python3 scripts/159_r5_d5_lightperf.py cpu-validate \
    --experiment "$D5_OUTPUT_ROOT/${D5_DATASET}_5_plan_$D5_TAG" \
    --output-dir "$D5_OUTPUT_ROOT/${D5_DATASET}_5_cpu_$D5_TAG"
  python3 scripts/159_r5_d5_lightperf.py inspect \
    --experiment "$D5_OUTPUT_ROOT/${D5_DATASET}_5_plan_$D5_TAG" \
    --cpu-audit "$D5_OUTPUT_ROOT/${D5_DATASET}_5_cpu_$D5_TAG"
done
```

Stop at every failed command; do not use this illustrative loop as permission to
ignore failure or submit a sweep. Current CPU audits are intentionally unapproved.

After all worker/approval blockers are resolved, the user may separately authorize
one manual L2 five-request condition at a time in a suitable existing allocation.
Preserve the same model/GPU UUID/backend/precision/concurrency and exact task list
across FULL/EXACT/APPROX. Cache/model/UD are fresh at condition boundary and persist
for all requests of that condition. One condition cannot be split across jobs.
There is no runnable sbatch command in this revision.

Future invocation shape, currently guaranteed to fail closed:

```bash
python3 scripts/159_r5_d5_lightperf.py run \
  --experiment "$D5_OUTPUT_ROOT/qasper_5_plan_$D5_TAG" \
  --cpu-audit "$D5_OUTPUT_ROOT/qasper_5_cpu_$D5_TAG" \
  --condition FULL_RECOMPUTE --output-dir "$D5_OUTPUT_ROOT/qasper_5_full_$D5_TAG"
# EXACT uses SEMCACHE_D5_EXACT, separate fresh output.
# APPROX uses SEMCACHE_D5_APPROX and --enable-approximate-experiment.
```

`verify --run RUN --experiment PLAN --cpu-audit CPU` checks receipt hashes,
source/plan/audit/knob bindings, request order, token/text hashes, row accounting,
latency definitions and actual resource metadata. Registry JSON for `analyze` or
`aggregate` is an explicit bounded list of `{run, plan, cpu_audit}` triples.
Exports: `analysis.json`, `tables.csv`, `tables.md` and Light inventory in a fresh
output. Current production analysis is unreachable without the missing native
approval; synthetic receipt tests are not experiment results. Plot export and
automatic grouping of L5 measured crossings remain NOT_IMPLEMENTED.

## Later gates and interpretation

Keep L2 20/50 and all L3–L5 jobs on hold. Review actual five-request outputs,
identity matches, native memory, timing overhead, token agreement and task quality
first. A future 20/50 plan needs valid original prefixes, explicit continuous
identity if crossing blocks, and budgets that fit unchanged deadlines. Each
dataset/count/condition/knob tuple is a separate manual decision. Matrix has 12
knob pairs, 324 hypothetical conditions across 3 datasets × 3 scales × 3 modes;
selected/submitted jobs = 0. L5 uses a separately approved targeted subset after
L3/L4 observations. No sign crossing means NO_OBSERVED_BREAK_EVEN; interpolation
is only an estimate within a measured bracket. Unlimited bandwidth is not numeric
zero. APPROX is a separate fidelity-constrained curve; task quality is not pooled
across datasets. N=5 P95 is exploratory and one trial supports no statistical claim.

User-provided historical Strict reference: 221601 files / 34.866 GB; MultiNews
EXACT HIT 2/5, 2880 reused QKV rows, fidelity 5/5; APPROX HIT 2/5, 10944 rows,
fidelity 3/5. These artifacts were not locally available for numerical re-audit.
They remain independent Strict observation data. Do not compare Strict and Light
as one observer-equivalent speed result or claim bitwise cross-mode parity before
matched five-request GPU evidence. Light full-vocabulary independent audit is
NOT_EVALUATED; correctness_qualified_speedup remains NOT_ESTABLISHED.
