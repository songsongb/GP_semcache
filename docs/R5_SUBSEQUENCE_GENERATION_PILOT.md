# Sparse SemCache autoregressive generation pilot

Implementation connects real native OPT prefill/decode, sparse CPU UD correction,
physical total-QKV cache and streamed M-stage impact. GPU_NOT_RUN. Native OPT
integration tests are NOT_RUN locally because Torch/Transformers are absent.
GPU readiness is pending a **zero-skip SERAPH native CPU approval** and real-asset
preparation. This document supersedes the earlier physical-core blocker list;
the old Exact Prefix runners and historical evidence remain unchanged.

Local verification:197 tests executed,188 PASS,0 FAIL,9 SKIP. The focused
generation/physical/audit/CLI suite ran25 tests (21 PASS,4 native-dependency
SKIP); relevant M bridge, I5/controller/semantic/multi/I4 and R4A regressions
ran172 (167 PASS,5 dependency SKIP). Torch, Transformers and NumPy are absent
on this workstation. CPU fixtures and localhost TCP passed; these results do
not establish native OPT or GPU numerical fidelity. Shell syntax, CLI/parser
contract, Python AST syntax and tracked whitespace checks also passed.

## Execution and evidence

The new151 CLI prepares exactly one real Qasper example using existing sealed
data, tokenizer, full question/reference fields and context-only truncation.
Source IDs are the first64 actual target IDs; the target has at most256 input
tokens and a2-token quality generation limit. Source is a completed independent
one-token request before admission. This is explicitly CONTROLLED, not a natural
reuse observation. It exercises3-token projection spans, **never prefix-forward
skipping**, even though its source is prefix-derived. Initial user/adapter routing
is original; all30 users/three synthetic adapters are verified/configured, but
only the selected adapter/user is exercised by this minimal pilot.

A/B/C each use an independently loaded bare FP16 OPT2.7B model, new CPU UD
process/TCP session and empty owner. A executes all GPU/CPU projection rows with
no semantic/impact/policy/cache; B includes matching, physical materialization,
impact and policy overhead but consumes no cache tensors; C actually consumes
selected total-QKV spans. All use the separately versioned sparse protocol;
this is a fresh comparison, not a replacement for historical A/B/C results.
Native Attention/FFN/head execute full target inputs. Native target-owned past KV
continues decode; source KV is never passed as target past. Actual position-module
lookups and target cache lengths are checked using existing R5 diagnostics.

Sparse TCP V2 binds the original sealed request, generated history, current row
indices, absolute decode positions, masks, layer/roles and matrix/session hashes.
An actual UD COMPLETE_REQUEST acknowledgement is required before physical cache
admission. Decode projection rows are always fresh; reuse applies to prefill.
Matrix loading, CPU isolation, base fingerprint, tokenizer/version and compatible
RTX3090 checks reuse existing R5 utilities. GPU UUIDs must match across the three
conditions; preparation/execution UUID provenance stays separate.

Default EXACT_CONTEXT_VERIFIED checks causal positions and actual per-layer input
bytes before consuming a span. It does not assert FP16 output equivalence.
`prepare --approximate` creates a separate plan for explicitly approximate shifted
or different-context total-QKV reuse. No mixed-mode cache or implicit opt-in.
Cached Q **is** consumed here, unlike Exact Prefix; source/target saved tensor
bytes prove that operation. No code changes to old prefix generation were made.

## Bounded genuine impact

A native attention forward hook consumes the returned probability tensor and
calls the original M paper_row_l2_sum reducer. One detached FP64 conversion is
shared across that layer's windows; the original row/key masks and reduction
definition remain unchanged. The hook replaces only the unused returned telemetry
with None after native PV/output computation. It does not change live attention
operands or output states. Matrices are streamed to sealed evidence one layer at
a time, not retained in the model's all-layer outputs.

Probability matrices are bounded to128MiB each; probability+FP64+window working
memory has a1GiB conservative bound. For this256-token pilot, sizes are much
smaller. Actual capture/reduction time, matrix bytes and working estimate are
recorded; synchronization and I/O are included in application E2E. Physical
materialization, hashing, F/A/I/S updates and CHU/PBR also occur inside that scope.
Owner completion, impact-history/CHU, insert/eviction, nested materialization
and tensor-integrity sealing times are exported separately. Projection rows,
frame bytes and CPU LoRA time are summed from actual per-pass TCP receipts;
summed layer/role rows are not mislabeled as unique reused tokens.
F counts a physically consumed span occurrence (at least one layer), not candidate
lookups; actual layer/role row savings are recorded separately. F/A/I/S formulas
are unchanged. PBR remains at100 queries and therefore is not exercised by this
two-request pilot. Missing probabilities fail admission.

The independent audit verifies every inventory/tensor hash, source-byte slices,
fresh gather/delta/add/scatter, both wire endpoints, UD completion, actual positions,
decode continuation, model/adapter authorization and physical policy event replay.
It independently reduces saved probabilities and checks the original M formula.
FP64 CPU/GPU reduction comparison has a fixed1e-12 absolute/relative allowance;
this is separate from, and does not alter, frozen model/logit thresholds.
Full-vocabulary last-position logits are saved for every generation step. Token
selection, text, Qasper F1, per-step frozen numerical gates and earliest recorded
projection differences are independently reproduced. Earlier numerical FAILs are
not replaced or assumed solved. Evidence PASS is separate from benefit/fidelity.

## CPU commands (after user-managed review, commit/push/pull)

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHONPATH=src:tests python3 -m unittest test_r5_subsequence_generation \
 test_r5_sparse_physical test_r5_subsequence_audit test_r5_subsequence_pilot -q

read -r -e -p 'Existing dataset registry JSON containing qasper: ' SS_REGISTRY
read -r -e -p 'Existing expanded multi-adapter preparation: ' SS_MULTI
export SSG_SOURCE_COMMIT="$(git rev-parse HEAD)"
SS_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export SSG_EXPERIMENT="$PWD/results/r5/i5r/parallel/subseq_gen_plan_$SS_STAMP"
export SSG_CPU_AUDIT="${SSG_EXPERIMENT}_cpu"
export SSG_RUN_ROOT="${SSG_EXPERIMENT}_runs"
python scripts/151_r5_subsequence_generation.py prepare \
 --registry "$SS_REGISTRY" --multi "$SS_MULTI" \
 --source-commit "$SSG_SOURCE_COMMIT" --output-dir "$SSG_EXPERIMENT"
python scripts/151_r5_subsequence_generation.py cpu-validate \
 --experiment "$SSG_EXPERIMENT" --output-dir "$SSG_CPU_AUDIT"
```

The approval runs installed native tiny OPT attention and full A/B/C generation
with a real separate Torch CPU UD process/TCP; no checkpoint is downloaded.
Missing native dependencies or any skipped/failed test rejects approval. It binds
the source snapshot, test SHA and exact plan inventory. The old150 shadow gate
is not used. A changed mode/source requires a fresh plan/approval/output directory.

## Prepared GPU job — NOT approved locally

`jobs/r5_subsequence_generation.sh` is the one non-array allocation wrapper:
compatible RTX3090,4 CPUs,64GiB,3h30m. Workers execute sequentially with1h each;
an outer3h timeout and cleanup headroom remain. Free disk requirement32GiB,
per-condition ES evidence8GiB/UD2GiB, loaded-model free GPU reserve12GiB, static
R5 peak estimate and actual allocation peak<22GiB. Host available reserve32GiB.
The reused conservative R5 resource forecast includes three unused warmup
reserves; actual execution performs zero warmups, recorded explicitly. Timing
forecasts are planning assumptions, not measured GPU latency guarantees.
Required operator variables: SSG_EXPERIMENT, SSG_CPU_AUDIT, SSG_SOURCE_COMMIT,
SSG_RUN_ROOT, SSG_NODE (moana-y3 or moana-y4), I5_CONDA_ROOT and both GPU opt-ins.
Logs and run parent must exist before submission. No automatic submission exists.

No manual GPU launch is authorized by local fixture success. First review a real
SEALED_CPU_PREPARED_GPU_NOT_RUN plan and zero-skip CPU validation on SERAPH. Native
tests/real preparation were unavailable locally; submission is deliberately not
recommended yet. The run command itself requires that sealed approval, Slurm,
explicit opt-in, permitted node, clean source, offline assets and resource checks.

After an operator-approved run, CPU audit commands are:

```bash
export CUDA_VISIBLE_DEVICES=''
read -r -e -p 'Completed fresh sparse-generation run directory: ' SS_RUN
python scripts/151_r5_subsequence_generation.py analyze \
 --experiment "$SSG_EXPERIMENT" --run "$SS_RUN" --output-dir "${SS_RUN}_audit"
python scripts/151_r5_subsequence_generation.py verify \
 --experiment "$SSG_EXPERIMENT" --run "$SS_RUN" --audit "${SS_RUN}_audit" \
 --output-dir "${SS_RUN}_verification"
```

Artifacts: subsequence_plan.json; cpu_validation.json; run status.json and per-
condition identity/requests/results, raw ES/UD tensor/logit/wire/policy evidence;
subsequence_audit.json, subsequence_report.md, subsequence_verification.json;
all independently SHA256-sealed. Failed attempts are preserved, never reused.
SIGTERM requests unwind worker cleanup and seal FAILED evidence. The parent
allows30 seconds to terminate/seal a timed-out worker and reap its CPU UD;
an unresponsive process can still require a forced kill, leaving incomplete
evidence that the independent audit rejects.

## Measurement limits

Application E2E includes semantic, matching, actual impact/copy/synchronization,
tensor evidence, admission and quality/logit persistence. Final metrics JSON
serialization is outside the interval. Model execution timing is instrumented
distributed-forward timing, not pure CUDA compute. Initialization and process
lifecycle are separate; zero unreported warmup requests. Conditions run A/B/C in
fixed order in this smoke, so order/thermal effects are uncontrolled. TCP is
localhost. Source-inclusive matched A/C span and warm-target intervals are
reported separately. A warm target interval is not an independently executed A0
session: true target-only economic break-even remains NOT_ESTABLISHED.

One context and2 target tokens provide functionality/descriptive quality only.
No statistical claim, full-context LongBench claim, trained-personalized adapter
claim, numerical fix or correctness-qualified speedup is established.
