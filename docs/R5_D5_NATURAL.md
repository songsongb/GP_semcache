# D5 on the original I5 NATURAL workload

This separate workflow imports the sealed I5 tasks verbatim. It never calls prompt
selection/truncation, changes routing, adds a source, duplicates a request, restores
historical tensors, or runs all twelve GPU blocks. Phase 3 requires user review.
Historical exact-prefix no-HIT timings remain a separate reference.

`153_r5_subsequence_natural.py` implements `prepare`, `cpu-validate`, `inspect`,
`run`, `analyze`, and `verify`. Private `_worker` and `_ud` actions are subprocess
entry points, not alternative approval routes. All runtime workers check the
pinned plan and immutable native CPU approval again.

## Architecture and invariants

Each condition loads one bare OPT-2.7B FP16 model and starts an isolated CPU UD for
each adapter used in the selected original block. Routing dispatches the original
ordered tasks to their persistent adapter-specific sparse Channel, SemanticIndex
and PhysicalOwner. C uses the original TinyBERT revision, one semantic cluster,
exact w3 windows, original non-overlap selection, F/A/I/S and actual streamed
attention Impact. A has no encoder, owner, impact or admission. Both use the same
native full attention/FFN/head, CPU LoRA, sparse version2 framing and decode path.
Only C reuses authenticated total QKV projection rows; it never grants past KV
from arbitrary subsequences. Cached Q **can** be consumed in this mechanism.

D5 `REQUEST_SCOPED_VERIFIED` and `EXACT_CONTEXT_VERIFIED` are sealed. Approximate
reuse is prohibited. Matching positions and causal token histories are necessary;
per-layer input bytes and actual consumed private tensor snapshots must also pass.
Integrity errors abort; ineligible windows compute fresh rows normally. An entry
is admitted only after native generation and the real UD completion ACK. Cache
state persists only within one condition/block, and disposal clears every owner.
Authorization uses the original expanded30-user/3-adapter registry. The unchanged
D5 threat model is a single owner thread and ordered CUDA stream, without external
concurrent storage writers. Tensor mutation, stale grants and replacement checks
are inherited. Unconsumed bytes are not continuously rehashed after authorization.

PBR remains100 requests **per adapter**: a <=20-request I5 block cannot trigger it.
Zero PBR events are expected, not a disabled mechanism. CHU counts completed actual
consumption, not discovery. No new LRU/admission algorithm is introduced. The
historical I5 LRU exact-prefix implementation remains unchanged; D5 uses its
existing F/A/I/S implementation in this separately labeled experiment.

## CPU feasibility and denominators

`natural_feasibility.json` and its Markdown table cover every original block. Pairs are **ordered**
earlier-source/later-target pairs within that block, across all adapter routes.
A raw matching occurrence-pair count sums `source_count(gram)*target_count(gram)`.
Unique matching target windows and pair counts are separate denominators.
All diagnostic rejection conditions are evaluated, including no exact w3, shifted
position, different causal context and denied sharing/adapter. Rejection reasons
can overlap; they are not additive populations. There is no source-shorter guard.

Static exact-context windows use identical absolute positions and identical token
histories through the window end. The reported greedy non-overlap positions,
capped at64, are an **optimistic upper bound**, not the actual M selection.
Actual M selection sees only resident admitted keys, scans up to64 candidates and
ranks by actual Impact. Cache duplicate-key ownership, actual semantic assignment,
F/A/I/S admission/eviction and per-layer byte equivalence remain unresolved offline.
No zero/fabricated Impact is used to simulate residency. Capacity accounting is
stored all-layer **Q+K+V**, not consumed KV: a three-token FP16 OPT entry occupies
1,474,560 bytes;64 compact entries occupy94,371,840 bytes per adapter, within the
unchanged512MiB bound. Physical/cache metrics are never inferred from this report.
Independent context counts use exact original context hashes; repeated contexts
and original repetition blocks are not independent scientific samples.

Real prepared assets are absent from the local coding environment. Real-data
feasibility is **NOT_RUN** until the commands below succeed on SERAPH. Synthetic
CPU test counts must never be presented as real dataset opportunities or GPU HITs.

## Resource and measurement scope

Original input token IDs and generation budgets are preserved (up to512 input;
Qasper/2Wiki normally32 generation tokens, MultiNews96). Larger original blocks
outside this explicitly supported profile reject, with no automatic shrink/drop.
One512-token FP16 attention layer is16MiB; the streamed FP64 Impact working estimate
is about80MiB, below the original128MiB probability /1GiB reduction bounds. No
all-layer attention matrices are retained in GPU memory. The conservative GPU
forecast and actual peak must remain below22GiB. Host available memory >=32GiB;
Slurm is4 CPU/64GiB RAM/one RTX3090. Each condition is independently initialized.

`prepare` forecasts dense worst-case value evidence (including UD payloads),
full-vocabulary logits, ledgers and files, without assuming HITs or early EOS.
Bounds: ES16GiB and UD4GiB per adapter;32GiB per condition;64GiB per run;
500,000 files;96GiB initial free disk (32GiB reserve). Every JSON <=16MiB.
Per-directory SHA256 inventories bind child inventories; all actual files are
independently streamed and verified. Tensor readers retain the old128MiB bound.
Each request's step metadata is sharded, with at most256MiB reconstructed metadata
and256MiB ledger bytes per request. No global Evidence limit is increased.
The two workers each have a7200s deadline; the job has a241-minute run guard and
4h30 allocation. This is a bounded timeout, not a measured completion forecast.
Preflight must pass on actual assets; budget failure preserves the failed attempt.

Primary E2E includes semantic assignment, matching, validation, inference,
CPU LoRA/TCP, attention capture/reduction, completion, CHU/admission/eviction,
value persistence and scoring. TTFT starts before semantic assignment. Nested
owner/profile scopes are inclusive and must not be added. Request sums and
sequence spans are distinct; initialization, zero warmups, process lifecycle,
final manifest sealing and small final timing sidecars are separate. Both
conditions save strict projection/logit evidence and use wall-only profiling.
C's extra Impact/policy capture is part of its measured overhead. This is an
instrumented comparison, not a production latency claim. Decode layout telemetry
is checked invariant and counted per layer/role instead of redundantly storing
96 identical layout receipts; all per-step tensor values are still saved.

## Exact operator commands

After manual review, commit/push and user-managed pull on SERAPH. The new commit
must include all new source/CLI/job/test files. Never use a historical SHA as the
new implementation identity. Select one original block explicitly; no best-case
selection based on observed GPU results.

```bash
cd /data/khuss/repos/GP_semcache
export I5_CONDA_ROOT=/absolute/path/to/existing/conda_root
source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export D5N_SOURCE_COMMIT="$(git rev-parse HEAD)"
export D5N_ORIGINAL="$PWD/results/r5/i5/plan_longbench3_smoke_v1"
# Set the actual existing expanded preparation; no adapter/model regeneration:
export D5N_MULTI=/absolute/path/to/existing/expanded_30_user_preparation
export D5N_BLOCK=0
export D5N_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export D5N_BASE="$PWD/results/r5/i5r/d5_natural"
export D5N_EXPERIMENT="$D5N_BASE/plan_${D5N_TAG}"
export D5N_CPU_AUDIT="$D5N_BASE/cpu_${D5N_TAG}"
export D5N_RUN_ROOT="$D5N_BASE/attempts_${D5N_TAG}"
mkdir -p logs "$D5N_RUN_ROOT"
df -h "$D5N_BASE"
CUDA_VISIBLE_DEVICES='' python scripts/153_r5_subsequence_natural.py prepare \
 --original-experiment "$D5N_ORIGINAL" --multi "$D5N_MULTI" \
 --source-commit "$D5N_SOURCE_COMMIT" --block-index "$D5N_BLOCK" \
 --output-dir "$D5N_EXPERIMENT"
CUDA_VISIBLE_DEVICES='' python scripts/153_r5_subsequence_natural.py cpu-validate \
 --experiment "$D5N_EXPERIMENT" --output-dir "$D5N_CPU_AUDIT"
CUDA_VISIBLE_DEVICES='' python scripts/153_r5_subsequence_natural.py inspect \
 --experiment "$D5N_EXPERIMENT" --cpu-audit "$D5N_CPU_AUDIT"
```

CPU validation requires installed Torch/Transformers/NumPy and zero FAIL/zero SKIP,
including the native eager OPT tests and actual localhost TCP fixture. No downloads
or GPU initialization are permitted. Inspect the feasibility JSON and selected
block resource/transport report before submission. A GPU allocation is unnecessary
for prepare/cpu-validate/inspect. Do not run heavy CPU validation alongside GPU work.

Manual submission only after those steps pass:

```bash
export D5N_NODE=moana-y3  # or moana-y4
export R5_ENABLE_GPU=1 R5_I5_ENABLE_GPU=1
sinfo -p batch_eebme_ugrad -o '%P %l %N'
scontrol show node "$D5N_NODE"
sbatch --nodelist="$D5N_NODE" --export=ALL jobs/r5_subsequence_natural.sh
```

The job prints/creates `$D5N_RUN_ROOT/run_<JOB_ID>`. It executes only A and C for
one selected block, then stops. A/C order is pinned by original block repetition;
one fixed order cannot establish statistical performance improvement.

After completion, independent CPU-only value replay and a fresh full recomputation:

```bash
export D5N_RUN="$D5N_RUN_ROOT/run_REPLACE_WITH_JOB_ID"
export D5N_AUDIT="$D5N_BASE/audit_${D5N_TAG}"
export D5N_VERIFY="$D5N_BASE/verification_${D5N_TAG}"
CUDA_VISIBLE_DEVICES='' python scripts/153_r5_subsequence_natural.py analyze \
 --experiment "$D5N_EXPERIMENT" --cpu-audit "$D5N_CPU_AUDIT" \
 --run "$D5N_RUN" --output-dir "$D5N_AUDIT"
CUDA_VISIBLE_DEVICES='' python scripts/153_r5_subsequence_natural.py verify \
 --experiment "$D5N_EXPERIMENT" --cpu-audit "$D5N_CPU_AUDIT" \
 --run "$D5N_RUN" --audit "$D5N_AUDIT" --output-dir "$D5N_VERIFY"
```

Analysis intentionally may take substantial CPU time: it independently verifies
all inventories, TCP payloads, gather/add/scatter and actual saved attention
Impact with original M reduction, without running model inference. Verification
repeats that audit. Keep source pinned; no re-sealing or editing historical inputs.

Local focused commands (fixtures only):

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_subsequence_natural -q
PYTHONPATH=src:tests python3 -m unittest test_r5_subsequence_generation test_r5_sparse_physical test_r5_subsequence_audit test_r5_subsequence_validation test_r5_subsequence_diagnostics test_r5_subsequence_pilot -q
bash -n jobs/r5_subsequence_natural.sh
python3 scripts/153_r5_subsequence_natural.py --help
```

## Outputs and acceptance

- Plan: `natural_plan.json`, `natural_feasibility.json`, `natural_feasibility.md`, hierarchical inventories.
- CPU approval: `cpu_validation.json`, inventory binding the exact plan/source.
- Run: root `status.json`; each condition's `identity.json`, `sequence.json`;
  adapter identities/UD isolation records, sharded ES step/results/timing/full
  tensors/logits, UD hidden/delta payloads, directional wire/layer ledgers, policy
  events and sealed failed attempts. No retry or silent fallback.
- Audit: `natural_audit.json`, `natural_report.md`, per-condition index and
  per-request metrics/eligibility/by-layer-role actual consumed windows,
  validation counts/bytes, original quality metrics and policy occupancy/events.
- Verification: `natural_verification.json`.

Evidence PASS is separate from actual physical HIT, frozen numerical PASS and
latency benefit. `Delta_E2E_ms = sum(C request E2E) - sum(A request E2E)`;
percentage divides by sum(A). Negative indicates lower instrumented request time.
A physical HIT requires actual authenticated consumed rows and independently
replayed sparse computation/wire/value receipts, not a metadata match. Full-
vocabulary per-generation-step parity, generated IDs/text and reference quality
are reported separately. Preserve numerical FAIL, even with matching text.
`CORRECTNESS_QUALIFIED_SPEEDUP=NOT_ESTABLISHED` for this single diagnostic block.
Zero HIT or slower E2E are valid results. A0 source-amortization is not applicable
to this unmodified NATURAL request stream; no constructed source is made free.
Do not proceed to the twelve-block GPU campaign without explicit user approval.
