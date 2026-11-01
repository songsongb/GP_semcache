# R5-I5 GPU identity policy hotfix

## Failure and policy boundary

User-verified Job 152396_0 failed before inference at the historical predicate
`gpu['uuid'] == pm['gpu']['uuid']` in shared R4A setup. Preparation UUID was
00f9c94e-27f9-a5f9-5ffd-f7dfeeb2105b; allocation UUID was
7eb7dffa-ae05-45cd-7e18-643c3b4a690c. Both were reported RTX 3090. This was an
allocation-identity rejection, not an observed OOM or numerical failure.

`STRICT_PREPARATION_GPU` remains the shared setup default. Its historical UUID
and name predicate is unchanged. No legacy caller passes the new keyword.
`SAME_COMPATIBLE_GPU_CLASS` is explicitly selected by the I5 caller and requires
an I5 scope marker. It requires cuda:0, a usable execution UUID, and RTX 3090
preparation/execution class (rejects unsupported class and 3090 Ti). Unknown
policies fail. There is no global environment bypass. I5 runner exposes
`--gpu-identity-policy`; its default is compatible class, strict remains selectable.

## Identity and integrity

No sealed preparation or adapter artifact is edited. Shared validate_prepared,
model architecture/FP16/seed/software versions, load_bare immutable revisions,
base SHA fingerprint and no-GPU-LoRA checks remain. I5 multi-adapter matrix,
tokenizer vocabulary/revision, registry/user, TinyBERT, installed OPT and all
frozen numerical checks remain active. Inference arithmetic is unchanged.

Preflight uses existing r3a_runtime.memory via a read-only diagnostic and pins
its allocation UUID in the new execution config. Every condition observes its
actual GPU through shared setup and checks this pin before any request or reuse.
A changed allocation fails both the producer and independent analyzer. Model
loading may precede the worker pin check; no measured forward or KV reuse does.

physical()/namespace() already receive gpu['uuid'] from execution, never pm's
historical UUID; I5Owner receives that same physical UUID. Cache contexts, source
receipts, tensors and UD routing retain this namespace. The independent adapter
and ownership replays now require the execution UUID rather than the preparation
UUID, including FULL_RECOMPUTE. Every real adapter requires actual GPU-forward
evidence; C still verifies source/reconstructed/consumed K/V and suffix-only work.
No live cache is restored. Explicit restoration flags/checkpoint paths fail.
Initial cache/index/UD grant emptiness checks remain in the independent audit.

Each completed condition emits gpu_identity.json. It records policy, both UUIDs
and names, equality/class classification, verified base/adapter flags, current
execution KV creation and historical_live_kv_restored=false. It is written only
after runtime completion and the final unchanged-base fingerprint check. CPU
fixtures record null GPU identities/class and physical GPU KV creation=false;
these are not mocked GPU verification. The analyzer recomputes receipts from
sealed preparation and observed adapter identities, checks all condition pins,
and includes receipts in the final independent audit.

## Diagnostics

CPU readiness remains allocation-independent. On a manually allocated GPU:

```bash
python scripts/124_diagnose_r5_i5_gpu.py --base-prepared "$I5_BASE_PREPARED"
```

This uses existing CUDA property/memory detection without model loading or
inference. It reports CUDA availability/visibility, actual name/UUID, preparation
UUID, match, policy and eligibility. Missing CUDA/dependencies return ineligible,
allocated_gpu_checked=false and nonzero exit; no allocated device is invented.
Local diagnostic execution returned NOT_CHECKED because torch is unavailable.

## Existing sealed plan

The existing plan_longbench3_smoke_v1 can be reused **if** normal seal, dataset,
tokenizer/context, adapter and eligibility checks pass on SERAPH. Workload plans
bind selections, routing, policies, budgets and resources; they do not bind the
old setup implementation, preparation UUID or execution GPU policy. This patch
does not modify their construction. A new execution records the new source SHA,
policy and allocation and links to the unchanged plan seal. The local workspace
cannot independently attest the user-reported SERAPH paths or their 12 blocks.
Run the inspector there before submission. Do not edit/reseal the old plan or
reuse the failed block_0_152396 directory. Old executions retain their original
source/audit contract and must be analyzed with their recorded source checkout.

## Changed files

Modified: src/semcache/edgelora/r4a_runtime.py (explicit default-strict keyword),
r5_i5_runtime.py (I5 opt-in, pin, restore rejection, receipt), r5_i5_audit.py and
r5_i5_analysis.py (allocation receipts and matching), scripts/119_run_r5_i5.py
(policy configuration and preflight/condition checks),
docs/R5_I5_INTEGRATED_EVALUATION.md (superseding UUID requirement).
Added: src/semcache/edgelora/gpu_identity.py, r5_i5_gpu.py,
scripts/124_diagnose_r5_i5_gpu.py, tests/test_r5_i5_gpu.py, this report.
No r3a_runtime.py, cache math, dataset/routing/packing/scale code or numerical
threshold was changed. Exact source files are staged explicitly together.

## SERAPH continuation (operator instructions only)

Transfer the reviewed local commit's format-patch file using your approved
method. Stop if tracked changes exist or HEAD differs from the reviewed parent
f05a6229741e7e87c93476e5fd54f027b3a8718d; do not reset or clean untracked files.

```bash
cd /data/khuss/repos/GP_semcache
git status --short
git rev-parse HEAD
git am /tmp/r5_i5_gpu_identity_policy.patch
export I5_SOURCE_COMMIT="$(git rev-parse HEAD)"
export I5_BASE_PREPARED="$PWD/results/r5/multiuser/expanded_152381_20261031T154848Z/base_prepare"
export I5_MULTI_PREPARED="$PWD/results/r5/multiuser/expanded_152381_20261031T154848Z/adapters"
export I5_EXPERIMENT="$PWD/results/r5/i5/plan_longbench3_smoke_v1"
python scripts/123_preflight_r5_i5.py readiness \
  --registry results/r5/i5/registry_longbench3.json \
  --base-prepared "$I5_BASE_PREPARED" --multi-prepared "$I5_MULTI_PREPARED"
python scripts/123_preflight_r5_i5.py inspect --experiment "$I5_EXPERIMENT"
export R5_I5_ENABLE_GPU=1 R5_ENABLE_GPU=1
# Use a NEW attempt root; do not overwrite run_longbench3_smoke_v1/block_0_152396.
export I5_RUN_ROOT="$PWD/results/r5/i5/run_longbench3_gpu_policy_attempt_v2"
sbatch --export=ALL --array=0-0%1 jobs/r5_i5_plan.sh
```

The default I5 policy is explicitly recorded as SAME_COMPATIBLE_GPU_CLASS.
If the inspector confirms 12 blocks, after the new Block 0 job finishes, verify
runtime completion AND independent audit, then review task, token/text and
numerical gates before further submission:

```bash
python - "$I5_RUN_ROOT/block_0_NEW_JOBID" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
assert json.loads((p/'i5_block_status.json').read_text())['status']=='COMPLETE'
a=p.with_name(p.name+'_audit')
r=json.loads((a/'i5_independent_audit.json').read_text())
assert r['status']=='PASS'
for c,v in r['conditions'].items(): print(c,v['gpu_identity'])
print(json.loads((a/'i5_decision.json').read_text()))
PY
python scripts/120_analyze_r5_i5.py --inputs "$I5_RUN_ROOT/block_0_NEW_JOBID" \
  --output-dir results/r5/i5/block0_gpu_policy_reaudit_FRESH
# ONLY after review and a verified 12-block manifest:
sbatch --export=ALL --array=1-11%1 jobs/r5_i5_plan.sh
```

These commands were not submitted. No data, adapter/base artifacts, failed job
outputs or historical evidence are regenerated. Numerical failures, if observed,
remain visible; ROOT_CAUSE_UNRESOLVED / I3F-B Future Work are unchanged. A CPU
policy test or reduced bytes never establishes GPU or correctness-qualified speedup.

## Final CPU verification

Final worktree: **151 tests: 149 PASS, 0 FAIL, 2 SKIP**. This comprises 129
unit/regression cases (127 PASS, installed OPT and ROUGE dependency SKIPs) and
22 actual CPU process/TCP cases (all PASS), including legacy I4A pilot/expanded
independent audits. The 14 new GPU-policy tests are CPU control-plane tests;
they do not qualify actual CUDA execution. They test equal and unequal UUIDs,
strict default rejection, I5-only opt-in, unavailable/wrong device/class, retained
preparation provenance, execution namespace, cross-condition mismatch, forbidden
restore, unavailable CUDA diagnostics, and unchanged model/tokenizer failure gates.
Existing adapter corruption and unauthorized reuse regressions also pass.
The separate I5 independent CPU audit PASS includes all normal physical fixture
KV/source, cache ownership, session, token/logit and matched-condition checks.
Source equality, bytecode compilation and diff checks pass. GPU_NOT_RUN: local
torch/Transformers/NumPy and ROUGE are unavailable; no real GPU test is inferred
from mocks/CPU fixtures. No inference, dataset or preparation artifact is changed.

Fresh evidence roots: results/r5/i5/gpu_policy_tcp_20261101/,
results/r5/i5/gpu_policy_independent_audit_20261101/ and
results/r5/i5/gpu_policy_delivery_20261101/. A local source commit includes all
11 reviewed source/test/doc paths; its exact SHA and format-patch are recorded
in the delivery directory and final response. No push is performed.
