# D5 Phase 3-A: five bounded NATURAL validations

**GPU_NOT_RUN. Remote SERAPH dataset availability is UNVERIFIED.** The user
reports the Qasper pilot is complete; its artifacts were neither accessed nor
changed. This implementation adds the five remaining datasets and no latency
benchmark or QKV algorithm.

`158_r5_natural_phase3a.py` provides read-only discovery, one-dataset preparation,
CPU approval, inspection, manually gated execution, the existing independent
audit/replay, and aggregation of five verified results. `r5_natural_phase3a.py`
delegates to the existing runner and auditor. Four existing modules receive
optional profile/approval/dataset hooks; their default Qasper semantics remain
unchanged. Request-level policy reporting attributes the already replayed
INSERT/EVICT/CHU/PBR events to existing request boundaries; historical reporting
keeps its default. The physical EXACT/APPROX operators, Phase 2.5A, generation, runtime,
sparse protocol, integrity, F/A/I/S, CHU, PBR, TinyBERT and CPU LoRA are unchanged.

One plan contains exactly the first original NATURAL repetition-0 block of one
dataset, with five scored requests and no sources or repeated evaluation added.
Original block ID, request IDs, prepared prompts/tokens, sample provenance,
generation policy, seed and routing remain intact. An explicitly selected later
block, slicing/combining blocks, changed budgets, or Qasper selection is rejected.
The budget authority is each sealed request's `prepared.max_new_tokens` and
matching canonical quality policy, not `BUDGETS[dataset]`. Both recorded budgets
must be integers within the existing 1..96 D5 bound. The original block seal,
payload binding, token/context limits and all resource guards remain required.
CoQA/MultiWOZ require complete, contiguous original conversations in numeric turn
order; partial conversations, even historical block splits, fail explicitly.
There is no search for a more cache-friendly or resource-friendly block.

Each condition runs in a fresh process. One dataset per allocation; its owners,
index and policy state persist throughout the block. A/B/C are FULL_RECOMPUTE,
SEMCACHE_D5_EXACT and SEMCACHE_D5_APPROX, using the existing operator and explicit
approximate opt-in. Source completion, sharing, tensor proofs and native target
past KV protections remain enforced. PBR stays at 100 requests per adapter and
cannot fire in these five-request blocks.

## Availability and blockers

The local workspace has only fixture preparations/plans and lacks Torch,
Transformers, NumPy and rouge. Fixtures are rejected as real assets. No SERAPH
SSH endpoint or mounted filesystem is available. The following are documented
**candidates**, not claims that assets exist:

| Dataset | Candidate original assets | Current finding |
|---|---|---|
| CoQA | `results/r5/i5/data/coqa_v1_dev`; `registry_four.json`; `plan_smoke_four_v1` | Remote UNVERIFIED; complete five-turn-compatible first block required |
| SNIPS | `results/r5/i5/data/snips_test`; same registry/plan candidates | Remote UNVERIFIED |
| MultiWOZ 2.2 | `results/r5/i5/data/multiwoz22_test`; same registry/plan candidates | Remote UNVERIFIED; version/ontology and complete dialogue boundary required |
| 2WikiMQA | `plan_longbench3_smoke_v1`, first dataset block (historically index 4) | User-verified source plan; fresh local revalidation unavailable |
| MultiNews | same LongBench plan, first dataset block (historically index 8) | User-verified source plan; actual resource revalidation pending SERAPH |

Prepared paths for LongBench are read from the sealed plan's dataset manifest,
never guessed. The discovery report records actual found paths, revisions,
split/version/source identity, sealed inventories, registry errors, block IDs,
grouping/resource failures, scorer availability and native dependencies. It
does not load a model, execute GPU work, prepare or modify any artifact.
Transport/tokenizer/native approval is a subsequent preparation/CPU gate.

**Budget-validation correction:** the earlier MultiNews blocker incorrectly
treated the default 96-token budget as the sealed Smoke budget. That conclusion
is withdrawn. The repository's `configs/r5_i5_smoke_budgets.json` records 8 for
2WikiMQA and 32 for MultiNews; production validation reads the sealed requests,
never this configuration or dataset defaults. No workload was regenerated or
budget lowered. Earlier sealed local receipts remain unchanged and describe
the earlier implementation's default-budget assumptions.

For five 512-token targets with a recorded 32-token budget, the unchanged
preflight forecasts **282,432 files**, **11.10 GiB GPU memory**, **12.58 GiB
evidence/condition**, and **37.74 GiB total A/B/C evidence**. UD tensor payload
alone is 2.90 GiB if all requests share one adapter; actual header/ledger and
routing checks remain required. Required free disk is still 96 GiB. This is a
configuration-based upper envelope, **not actual sealed-data validation**:
the original SERAPH plan is absent locally. The exact actual forecast remains
BLOCKED locally until the read-only command below runs against that plan.

For reference, these older analytical examples use I5 **default** budgets,
not necessarily the budgets in the original sealed Smoke plan:

| Dataset | Generation/request | A/B/C file forecast | A/B/C evidence GiB | Status before actual transport |
|---|---:|---:|---:|---|
| CoQA | 32 | 282,432 | 37.74 | Envelope only |
| SNIPS | 16 | 173,952 | 35.96 | Envelope only |
| MultiWOZ | 64 | 499,392 | 41.32 | Only 608 forecast files below limit; runtime can still reject |
| 2WikiMQA | 32 | 282,432 | 37.74 | Envelope only |
| MultiNews | 96 | 716,352 | Not approved | This 96-token example exceeds the file guard |

These are explicitly labeled fixtures for resource arithmetic, not real requests
or measured memory/latency. Forecast GPU allocation is about 11.10–11.12 GiB
for the first four; actual <22 GiB remains required. ES 16 GiB/adapter, UD 4 GiB,
condition 32 GiB, run 64 GiB, free disk 96 GiB, host available RAM 32 GiB,
4 CPU/64 GiB allocation, 7,200 s/condition and 14,460 s shared run deadline are
unchanged. Actual headers and UD ledgers are forecast during preparation.
No token budgets, cache capacities, validation or limits are lowered/increased.

## Read-only SERAPH discovery

Use the existing offline `semcache` environment. The variables below identify
documented candidates; discovery determines whether they actually exist. Use
the existing expanded 30-user preparation recorded by the successful Qasper
`approx_plan.json` field `multi` for `D5N_MULTI`.

```bash
cd /data/khuss/repos/GP_semcache
export I5_CONDA_ROOT=/absolute/path/to/existing/conda_root
source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=42
: "${D5N_MULTI:?set the existing Qasper pilot's sealed expanded preparation path}"
CUDA_VISIBLE_DEVICES='' python scripts/158_r5_natural_phase3a.py discover \
  --search-root "$PWD/results/r5/i5/data" \
  --registry "$PWD/results/r5/i5/registry_four.json" \
  --prepared-dataset "coqa=$PWD/results/r5/i5/data/coqa_v1_dev" \
  --prepared-dataset "snips=$PWD/results/r5/i5/data/snips_test" \
  --prepared-dataset "multiwoz=$PWD/results/r5/i5/data/multiwoz22_test" \
  --original-experiment "$PWD/results/r5/i5/plan_smoke_four_v1" \
  --original-experiment "$PWD/results/r5/i5/plan_longbench3_smoke_v1" \
  --multi "$D5N_MULTI"
```

This command writes only stdout. Save stdout to a **new** JSON file if desired
and set `D5P3A_DISCOVERY` to it. Additional known existing plans can be passed
with repeated `--original-experiment`; do not recursively scan tensor evidence
directories. `STATIC_BLOCK_VALID` is not native CPU or GPU approval. In particular
it does not imply actual cache residency or a physical HIT. Inspect each
dataset's `blockers`, `original_blocks`, scorer and dependency records.

## Missing assets: existing offline commands

Run normalization only if discovery confirms the sealed preparation is missing
and the operator has verified the real local upstream input, split and immutable
revision. No download, normalization, or plan generation was executed here.
Reuse valid existing assets; never overwrite/reseal historical directories.
See `R5_I5_INTEGRATED_EVALUATION.md` for native schemas.

```bash
# Variables below must identify verified existing raw inputs and fresh outputs.
CUDA_VISIBLE_DEVICES='' python scripts/118_prepare_r5_i5.py dataset --dataset coqa \
  --source "$COQA_NATIVE_JSON" --revision "$COQA_ARCHIVE_SHA256" --split dev \
  --output-dir "$NEW_COQA_PREPARED"
CUDA_VISIBLE_DEVICES='' python scripts/118_prepare_r5_i5.py dataset --dataset snips \
  --source "$SNIPS_NATIVE_TEST_DIR" --revision "$SNIPS_COMMIT" --split test \
  --output-dir "$NEW_SNIPS_PREPARED"
CUDA_VISIBLE_DEVICES='' python scripts/118_prepare_r5_i5.py dataset --dataset multiwoz \
  --source "$MWOZ22_NATIVE_TEST_DIR" --ontology "$MWOZ22_SCHEMA_JSON" \
  --revision "$MWOZ_COMMIT" --split test --output-dir "$NEW_MWOZ_PREPARED"
CUDA_VISIBLE_DEVICES='' python scripts/118_prepare_r5_i5.py dataset --dataset 2wikimqa \
  --source "$WIKI_NATIVE_JSONL" --format jsonl --revision "$LONGBENCH_COMMIT" \
  --split test --output-dir "$NEW_WIKI_PREPARED"
CUDA_VISIBLE_DEVICES='' python scripts/118_prepare_r5_i5.py dataset --dataset multi_news \
  --source "$NEWS_NATIVE_JSONL" --format jsonl --revision "$LONGBENCH_COMMIT" \
  --split test --output-dir "$NEW_NEWS_PREPARED"
```

CoQA requires dev gold including additional answers; SNIPS requires a single
native intent-keyed split; MultiWOZ requires native 2.2 USER states and matching
`schema.json`. MultiNews requires installed **rouge==1.0.1**, checked during
discovery, preparation and CPU approval. Missing raw inputs, revisions, ontology
or scorer are prerequisites to report, not reasons to substitute test data.

If valid data exist but no original compatible NATURAL plan exists, the existing
offline I5 plan command is available. Use a fresh registry mapping the needed
dataset keys to verified absolute prepared paths and a fresh counts JSON such
as `{"coqa":5,"snips":5,"multiwoz":5}`; no budget override:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/118_prepare_r5_i5.py experiment \
  --registry "$NEW_VERIFIED_DATASET_REGISTRY" --counts "$NEW_COUNTS_JSON" \
  --base-prepared "$EXISTING_BASE_PREPARED" --scale SMOKE --tracks NATURAL \
  --repetitions 1 --block-size 5 --seed 42 --strict-per-block 1 \
  --allow-context-truncation --input-limit 512 --output-dir "$NEW_I5_PLAN"
```

Use the original I5 selection/template/budgets. A partial selected conversation
or a non-five first block still fails Phase3A; do not adjust the count, seed or
sample ordering to find hits. Re-run discovery on this new plan before D5
preparation. LongBench's existing sealed plan remains the preferred input.

## Dataset-specific CPU preparation and approval

Review/commit the three shared hooks and new module/CLI/job/tests, then transfer
the reviewed source to SERAPH through the operator's normal workflow. No commit
or push is performed here. Pinned preparation requires a clean tracked checkout.
All output paths must be fresh. Set the three original-plan variables to paths
actually validated by discovery, not merely the candidate names above.

```bash
export D5P3A_ROOT="$PWD/results/r5/i5r/d5_natural/phase3a"
export D5NP_SOURCE_COMMIT="$(git rev-parse HEAD)"
: "${D5P3A_DISCOVERY:?saved fresh SERAPH discovery JSON}"
: "${COQA_ORIGINAL_PLAN:?verified original CoQA plan}"
: "${SNIPS_ORIGINAL_PLAN:?verified original SNIPS plan}"
: "${MWOZ_ORIGINAL_PLAN:?verified original MultiWOZ plan}"
export LONGBENCH_ORIGINAL_PLAN="$PWD/results/r5/i5/plan_longbench3_smoke_v1"

prepare_cpu_one() {
  local dataset="$1" original="$2" index
  index=$(python -c 'import json,sys; from pathlib import Path; r=json.load(open(sys.argv[1])); p=str(Path(sys.argv[3]).resolve()); rows=[x for x in r["datasets"][sys.argv[2]]["original_blocks"] if x["plan"]==p]; assert len(rows)==1, "one discovered original block required"; x=rows[0]; assert x["status"]=="STATIC_BLOCK_VALID", x.get("reason"); print(x["block_index"])' "$D5P3A_DISCOVERY" "$dataset" "$original") || return
  CUDA_VISIBLE_DEVICES='' python scripts/158_r5_natural_phase3a.py prepare \
    --dataset "$dataset" --block-index "$index" --original-experiment "$original" \
    --multi "$D5N_MULTI" --source-commit "$D5NP_SOURCE_COMMIT" \
    --enable-approximate-experiment --output-dir "$D5P3A_ROOT/${dataset}_plan_01" || return
  CUDA_VISIBLE_DEVICES='' python scripts/158_r5_natural_phase3a.py cpu-validate \
    --experiment "$D5P3A_ROOT/${dataset}_plan_01" --output-dir "$D5P3A_ROOT/${dataset}_cpu_01" || return
  CUDA_VISIBLE_DEVICES='' python scripts/158_r5_natural_phase3a.py inspect \
    --experiment "$D5P3A_ROOT/${dataset}_plan_01" --cpu-audit "$D5P3A_ROOT/${dataset}_cpu_01"
}
# Separate CPU operations; no GPU submission loop.
prepare_cpu_one coqa "$COQA_ORIGINAL_PLAN"
prepare_cpu_one snips "$SNIPS_ORIGINAL_PLAN"
prepare_cpu_one multiwoz "$MWOZ_ORIGINAL_PLAN"
prepare_cpu_one 2wikimqa "$LONGBENCH_ORIGINAL_PLAN"
prepare_cpu_one multi_news "$LONGBENCH_ORIGINAL_PLAN"
```

Every command evaluates the budget actually sealed in its original block.
Any file/memory/disk guard failure stops preparation without a CPU approval or
job submission. Dialogue dataset commands may similarly stop with the
exact original conversation/block failure. Native approval requires zero skips;
it includes existing native tiny-OPT/TCP and D5 regression tests plus new tests.

## Later manually approved GPU operation, one dataset at a time

These commands are instructions only and were not executed. Run only after that
dataset independently passes source/data/scorer/transport/resource checks and
zero-skip CPU approval. `D5P3A_DATASET` is set to exactly one of `coqa`, `snips`,
`multiwoz`, `2wikimqa`, `multi_news`; do not submit a loop or array.
Each dataset needs its own reviewed allocation and actual resource approval.

```bash
export D5P3A_DATASET=coqa
export D5NP_EXPERIMENT="$D5P3A_ROOT/${D5P3A_DATASET}_plan_01"
export D5NP_CPU_AUDIT="$D5P3A_ROOT/${D5P3A_DATASET}_cpu_01"
export D5NP_RUN_ROOT="$D5P3A_ROOT/${D5P3A_DATASET}_attempts"
export D5N_NODE=moana-y3
export R5_ENABLE_GPU=1 R5_I5_ENABLE_GPU=1
mkdir -p logs "$D5NP_RUN_ROOT"
sbatch --nodelist="$D5N_NODE" --export=ALL jobs/r5_natural_phase3a.sh

# After completion, substitute this dataset's returned job ID.
export D5NP_RUN="$D5NP_RUN_ROOT/run_RETURNED_JOB_ID"
CUDA_VISIBLE_DEVICES='' python scripts/158_r5_natural_phase3a.py analyze \
  --experiment "$D5NP_EXPERIMENT" --cpu-audit "$D5NP_CPU_AUDIT" \
  --run "$D5NP_RUN" --output-dir "${D5NP_RUN}_audit"
CUDA_VISIBLE_DEVICES='' python scripts/158_r5_natural_phase3a.py verify \
  --experiment "$D5NP_EXPERIMENT" --cpu-audit "$D5NP_CPU_AUDIT" \
  --run "$D5NP_RUN" --audit "${D5NP_RUN}_audit" --output-dir "${D5NP_RUN}_verification"
```

Repeat this same one-dataset recipe only after a separate manual decision for
each remaining dataset. Zero physical natural hits are valid. Neither workload
changes nor numerical threshold relaxation follow a zero/negative result.

## Measurements and aggregation

Existing per-request audit JSON records HITs, actual reused/skipped QKV rows,
A/B/C/D category details by layer/role, native/CPU LoRA work, occupancy/policy,
application TTFT/E2E, prefill/decode, wire bytes and profiler components. Numerical
fidelity and generated tokens/text remain independent of evidence integrity.
CoQA F1/EM, SNIPS intent accuracy/macro-F1, canonical MultiWOZ 2.2 JGA/slot
metrics, 2WikiMQA QA F1, and MultiNews ROUGE-L use existing scorers. Approximate
differences are never labeled exact equivalence.

Create a fresh `phase3a_results_registry.json` with **exactly** the five dataset
keys, each mapping to absolute `experiment`, `run`, `cpu_audit`, `audit`, and
`verification` paths from that dataset's completed operation. No Qasper entry.
Then, entirely on CPU:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/158_r5_natural_phase3a.py aggregate \
  --results-registry "$D5P3A_RESULTS_REGISTRY" --output-dir "$D5P3A_ROOT/five_dataset_table_01"
```

Aggregation requires sealed audit/verification receipts bound to the original
plan, CPU approval and run. It emits a 15-row Markdown table plus JSON including
HIT rate, actual skipped CPU LoRA/QKV rows, categories, means, quality/delta,
numerical and generation agreement, wire bytes, occupancy/policy and inclusive
profiler components. It does not create another numerical auditor. Missing or
altered verification fails closed. Inclusive overlapping scopes must never be
added to reconstruct E2E. No pooled quality score or unsubstantiated speedup is
reported; initialization and sealing remain outside the existing application
measurement scope. The final lightweight six-dataset runner is not implemented.

## Local validation and conclusion

Budget fix: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest
test_r5_natural_phase3a test_r5_i5 -v` ran 45 tests: **44 passed, one skipped**
(the installed rouge==1.0.1 check), zero failures. Phase3A's 16 tests all passed.
The fresh sealed receipt and resource envelope are in
`results/r5/i5r/d5_natural/phase3a/budget_fix_cpu_20261109_01/budget_fix_validation.json`.
New regression fixtures cover both Smoke budgets through selection and
preparation, unchanged request payloads/routing/order, mismatched budgets,
altered policy fields/special tokens, malformed budget types, out-of-bound
budgets and unsealed token tampering. They do not establish real-data approval.

To recalculate actual MultiNews file, memory, disk and transport forecasts on
SERAPH, with `D5N_MULTI` pointing at the existing verified preparation, run this
read-only CPU command after updating the reviewed source:

```bash
cd /data/khuss/repos/GP_semcache
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python - <<'PY'
import json, os
from semcache.edgelora.r5_natural_phase3a import original_plan, material
original='results/r5/i5/plan_longbench3_smoke_v1'
w, _, _=original_plan(original)
i=next(i for i,b in enumerate(w['blocks']) if b['dataset']=='multi_news'
       and b['track']=='NATURAL' and b['repetition']==0)
fields, _=material(original, os.environ['D5N_MULTI'], 'multi_news', i)
print(json.dumps(dict(block_id=fields['block_id'],
    original_inventory=fields['original_inventory'],
    request_ids=[t['request_id'] for t in fields['tasks']],
    input_tokens=[len(t['token_ids']) for t in fields['tasks']],
    recorded_budgets=[t['policy']['max_new_tokens'] for t in fields['tasks']],
    resources=fields['resources'], transport=fields['transport']), indent=2))
PY
```

This does not prepare, reseal, submit or run an experiment. Missing assets or
scorers and existing forecast guards still fail closed. Use a fresh Phase3A
plan and zero-skip CPU approval afterward; old source-pinned approvals do not
authorize execution of changed code.

Before the budget fix, the sealed `results/r5/i5r/d5_natural/phase3a/cpu_local_20261109_02/`
receipt, test logs, local discovery and analytical resource envelopes. New
Phase3A tests: 14 passed. Existing D5/I5 tests: 100 passed, 9 dependency-related
skips. Phase 1/2 continuity tests: 34 passed. Total: 148 passed, 9 skipped,
zero failures. The legacy test that executes a job shell was excluded; the new
job was parsed with `bash -n` only. Local
fixture tests establish preparation/rejection, delegation and aggregation
behavior only. Native tests skipped for missing dependencies do not approve GPU
execution. Historical physical/Phase 2.5A code hashes are checked against HEAD.

**Implementation CPU fixtures PASS; remote availability UNVERIFIED; real-data
native approval and actual MultiNews forecast BLOCKED locally by missing assets;
GPU_NOT_RUN.** No GPU/Slurm execution, download, historical resealing,
Qasper rerun, final benchmark implementation or git push occurred.
