# R5-I5 integrated multi-dataset evaluation

Implementation status: **IMPLEMENTED**, CPU validation recorded below, **GPU_NOT_RUN**.
This framework prepares experiments; it has not measured a GPU speedup or real-data
quality improvement. It does not submit jobs, push commits, download in jobs, alter
historical evidence, or change frozen numerical thresholds.

## Research boundary

The user-verified I4A Expanded result is 30 logical users, three adapters, 30
requests, 12 physical HITs, 132 reused tokens, 12 admissions and three evictions.
Generated tokens/text matched; reported communication fell about 18.2%; E2E was
about 7.4% slower in that controlled run. All 12 HITs failed frozen numerical
tolerances. **ROOT_CAUSE_UNRESOLVED**. Native PEFT strict parity and
correctness-qualified speedup remain unestablished. I3F-B is FUTURE_WORK. Task
score equality, generated output equality and numerical tolerance are separate
gates. Clusters remain one, physical policy remains LRU, serving is sequential.

## Architecture and relationship to I4A

I5 is additive. `r5_i5_data`, `quality`, `workload`, `controller`, `runtime`,
`audit`, `analysis`, `evidence`, and the numbered scripts form the new pipeline.
The I5 request contract, forward protocol, driver and generation loop are explicit
versions of the frozen I4A/C1 implementations. Their differences are the I5
version identifiers, input/prefix bounds, I5 protocol imports and I5 cache
destination. No R3/R4/I3P/I3F/I4A file is edited. Actual OPT forwards, eager
attention, CPU LoRA arithmetic, bounded TCP framing, Raw-QKV byte storage,
physical K/V reconstruction and consumed-history checks retain their existing
implementations. Q is stored/accounted but never consumed by reuse.

There is one base model per condition and three persistent independent CPU UD
processes. Adapter_A/B/C use the unchanged deterministic distinct rank-8 matrices
with seeds 104729/130363/155921. They are untrained controlled synthetic research
adapters. User_01–10 map to A, User_11–20 to B, User_21–30 to C. User assignment
cycles through these groups deterministically; 30 users are logical identities,
not 30 physical devices. Within-adapter sharing requires registry authorization.

## Dataset acquisition and local preparation

No real dataset availability is asserted. A prepared snapshot must exist, pass
its inventory SHA-256 checks, contain valid original IDs/references and match the
requested dataset/version. Missing snapshots fail closed. CPU fixtures are
explicitly marked and rejected by GPU preflight.

Acquire the following **outside inference jobs** and keep the download, license,
commit/archive identity and split directory. Supply a 40-character upstream
commit or 64-character published archive SHA-256 as `--revision`; `main`,
`latest` and invented versions are rejected. This argument is a provenance
declaration; the local inventory proves the acquired bytes, not that a remote
server served a declared commit. Upstream revisions must be independently pinned
when acquiring the data.

* CoQA v1.0: [official release](https://stanfordnlp.github.io/coqa/),
  `coqa-dev-v1.0.json` with the native top-level `data` array. Public dev has gold;
  hidden test without references is unsuitable. Preserve all primary and
  additional annotator answers, story, conversation and turn IDs.
* SNIPS: [Sonos/Snips NLU benchmark](https://github.com/sonos/nlu-benchmark),
  `2017-06-custom-intent-engines`. Use native intent-keyed utterance JSON files
  from **one explicit split**, copied into a directory containing only those
  files. The native source has no utterance ID: the stable original identity is
  filename plus original zero-based utterance ordinal. Preserve native chunk and
  slot annotations, but I5 does not score slot filling.
* MultiWOZ **2.2**: [upstream format and ontology](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2).
  Select a single `train`, `dev` or `test` dialogue directory and the matching
  `schema.json`. Use native dialogue JSON, USER frames and `state.slot_values`.
  Version 2.0/2.1/2.4 and differently flattened HF DST schemas are rejected.
* LongBench v1: [official repository](https://github.com/THUDM/LongBench/tree/main/LongBench)
  and `zai-org/LongBench`; export `qasper`, `2wikimqa`, `multi_news` test records
  with `_id`, `context`, `input`, `answers`, `length`. Preserve references exactly.
  JSONL and compatible `datasets.save_to_disk` snapshots are supported.

The normalizer supports native JSON/JSONL and compatible local HF snapshots;
HF snapshots are loaded only through `load_from_disk`, never `load_dataset`.
HF record layouts must match the stated native schema. A native-schema export
is required for incompatible HF adapters.

Examples, after acquiring files and substituting independently verified pins:

```bash
python scripts/118_prepare_r5_i5.py dataset --dataset coqa \
  --source /data/khuss/datasets/coqa/coqa-dev-v1.0.json \
  --revision "$COQA_ARCHIVE_SHA256" --split dev \
  --output-dir results/r5/i5/data/coqa_v1_dev
python scripts/118_prepare_r5_i5.py dataset --dataset snips \
  --source /data/khuss/datasets/snips/test --revision "$SNIPS_COMMIT" --split test \
  --output-dir results/r5/i5/data/snips_test
python scripts/118_prepare_r5_i5.py dataset --dataset multiwoz \
  --source /data/khuss/datasets/MultiWOZ_2.2/test --revision "$MWOZ_COMMIT" --split test \
  --ontology /data/khuss/datasets/MultiWOZ_2.2/schema.json \
  --output-dir results/r5/i5/data/multiwoz22_test
python scripts/118_prepare_r5_i5.py dataset --dataset qasper --format jsonl \
  --source /data/khuss/datasets/longbench/qasper.jsonl --revision "$LONGBENCH_COMMIT" --split test \
  --output-dir results/r5/i5/data/qasper_test
```

Repeat LongBench preparation for `2wikimqa` and `multi_news` if available. Do not
merge splits. Create a registry JSON mapping dataset keys to **absolute prepared
directory paths**. A four-family smoke registry contains coqa/snips/multiwoz/qasper;
the two other supported LongBench subsets can be independent additional runs.

## Scoring and prompt conventions

All scores use 0..1. Per-sample records preserve dataset/version, original ID,
group, references hash, actual text, prompt version and scoring version.

CoQA uses lowercase, punctuation/article removal and whitespace normalization,
multiset token overlap F1 `2*overlap/(prediction_tokens+gold_tokens)` and normalized
exact match. Multiple references use the CoQA leave-one-reference-out convention:
for each excluded annotator take the best remaining reference, then average.
Single-reference examples use their single score. Official domain/source macro
tables are not reproduced; the result is the selected subset's turn mean.
Historical questions and primary answers are teacher-forced dialogue history;
the current gold answer never enters the prompt.

SNIPS outputs exactly one of AddToPlaylist, BookRestaurant, GetWeather,
PlayMusic, RateBook, SearchCreativeWork, SearchScreeningEvent. Only outer
whitespace is stripped. Case changes, explanation, punctuation, multiple labels
or prefixes are invalid and receive zero accuracy; there is no repair. Macro-F1
averages all seven fixed classes, including absent classes with zero F1.
Per-class precision/recall/F1 and invalid rates are retained.

MultiWOZ 2.2 uses the documented 35 tracked slots from the verified ontology,
not response-only schema slots. Gold values are lists of accepted alternatives.
At each USER turn, service frames replace that service's state; other services'
last known state persists. The prompt contains utterance history through the
current USER turn and the fixed slot schema, never current gold state. Output
must be one JSON object mapping tracked slots to nonempty strings. Duplicate
keys, unknown slots, wrong types, prose or broken JSON are invalid. Values are
lowercased and whitespace-collapsed; there is no synonym/time/value repair.
JGA requires exactly all gold slots with each value in its reference alternative
set. Slot micro P/R/F1 count one prediction per slot; an incorrect value is both
FP and FN. Invalid output receives zero JGA and is separately counted. This is
the pinned canonical 2.2 state evaluator; it is not a MultiWOZ 2.1 or response
generation leaderboard metric.

Qasper/2WikiMQA reuse the R4C/LongBench normalized QA F1 and max over references.
MultiNews requires **rouge==1.0.1**, package-default sentence/whitespace processing,
case preserved, no stemming, max reference ROUGE-L F1. No LCS diagnostic is used
for real-data scoring. Empty output scores zero. The official LongBench
[metric implementation](https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py)
is the compatibility reference. Prompt wording is I5's deterministic version,
so results are explicitly derived evaluations.

## Context and generation limits

The actual model config supplies the positional limit, bounded at 2048 for this
OPT implementation. Input IDs plus reserved output budget must fit. I5 supports
1..128 output tokens using the existing variable-length CPU protocol lifecycle,
greedy decoding, EOS or LENGTH_CAP termination, and nonfinite-logit rejection.
Defaults are SNIPS 16, CoQA/QA 32, DST 64, MultiNews 96. Start smoke with the
conservative budget config, then explicitly increase after inspecting outputs.
No budget change alone establishes quality adequacy; clipped DST/summary outputs
remain invalid or incomplete and are reported.

Only context text can be truncated, with explicit opt-in. Head/tail token
retention inserts an omission marker and retokenizes the full prompt until it
fits. Questions, instructions, dialogue history and gold are never truncated.
Long dialogue fixed fields that do not fit are excluded. LongBench truncated
records are labeled `LONGBENCH_DERIVED_CONTEXT_LIMITED_EVALUATION`, never an
official full-context benchmark. Original/retained/removed lengths, exclusions,
truncation rates and effective counts are in manifests. Protocol header/payload
limits and memory checks can further reject a configuration; there is no fallback.

## Conditions, workload and source costs

**A FULL_RECOMPUTE** creates no encoder or semantic index, never admits tensors,
and computes full prefill through the same CPU LoRA/TCP path.

**B SEMANTIC_NO_REUSE** performs TinyBERT assignment and exact-token discovery
against eligible completed sources in a metadata-only shadow index. Its virtual
LRU uses the same per-adapter Raw-QKV byte capacity and 64-entry bound. A virtual
HIT touches LRU and prevents standalone admission, matching C's inability to
admit a suffix-only result. All actual model prefill is full. Shadow serialized
metadata bytes, simulated capacity and unmeasured Python heap are separate from
GPU storage. Tensor integrity failures can make C diverge: the independent audit
compares decisions and reports equivalence or the B–C isolation limitation.

**C SEMCACHE_INTEGRATED** uses real GPU Raw-QKV ownership, exact prefix checks,
same-adapter authorization, completed-source grants, reconstructed K/V and
suffix-only prefill. An integrity failure aborts rather than recomputing secretly.

Natural track uses stable hash-ranked conversation groups and turn ordering,
with authentic dataset prompts and no inserted priming requests. Zero physical
HITs is a valid outcome. Controlled track uses real-data target prompts and
directly slices 64/128/256/512/1024 initial token IDs for one-token priming source
requests. The target has a nonempty suffix. Sources complete before targets and
normal semantic discovery decides reuse. Decoded source text is semantic input
only; it is never re-encoded for model input. Neither source nor target contains
the current gold reference. All three adapters participate through stable routing.

Targets and priming requests are separately labeled. Warm-target latency and
amortized latency including priming plus model warmup are both reported. Model
initialization is separate. One uncached one-token model warmup per adapter
creates no semantic/cache state. Each condition has new ES/UD processes, sessions
and empty caches. Repetition/block orders follow ABC/CBA/BCA/ACB/CAB/BAC. Same
samples, IDs, tokens, generation policy, matrix bytes and mapping are hashed and
matched for A–B, B–C, A–C comparisons. A is the reference for task quality.

## Capacity, runtime and evidence budgets

FP16 OPT-2.7B Raw-QKV costs `3*32*2560*2 = 491520` bytes per cached token
(480 KiB). A 1024-token entry needs 480 MiB. Default capacity is 512 MiB **per
adapter**, total 1.5 GiB, and holds one such prefix. The hard bound is 1 GiB per
adapter; LRU and capacity eligibility are recorded. The historical two-source
workload capacity and <512 source filter do not apply to I5.

The resource manifest estimates 5.4 GB FP16 model storage, active K/V,
transient Raw-QKV, attention/logits workspace plus a 4 GiB reserve, 32 GiB CPU
reserve, maximum forward count, verbose ledger/logit/strict tensor disk size,
and duration. GPU estimate must fit below 22 GiB on 24 GiB hardware; actual free
memory and disk are checked. Estimates are conservative planning assumptions,
not measured runtime or allocator guarantees. OOM aborts the block. Default
evidence bound is 8 GiB per condition, maximum 16 GiB; three conditions require
three times that worst-case free space plus reserve. Per-block request bound is
20 scored targets (40 with priming), generation bound 128. Default block size is
five, so scale requires many independently restartable jobs. Duration planning
defaults to five seconds per forward and must be replaced with observed smoke
costs before scaling. Hard runner bound is three hours.

STRICT_AUDIT stores complete selected Raw-QKV/components and prefill consumed
KV, validates CPU payload receipts and every decode KV hash. Selected HITs also
capture resident cached Raw-QKV, binding it back to the original source hashes,
so natural sources need not all be strictly serialized. SCALABLE_EVALUATION
keeps IDs/tokens/logits, provenance, receipts, all wire headers and projection/
KV hashes; it still executes physical integrity checks but omits full tensor
files for most requests. This reduces storage, not all hashing overhead. Full
tensor audit of every large-run request is NOT_SUPPORTED. Strict sampling is
chosen deterministically before running; source/target pairs share selection.

## Timing, network and statistics

Model TTFT starts immediately before generation; application TTFT starts at
arrival and includes token checks, semantics and lookup. Model E2E includes
distributed forwards, instrumentation, greedy selection, output processing and
CPU completion receipt. Application E2E additionally includes lookup, admission,
eviction and logit evidence writes. Final quality scoring and final row
serialization occur outside this window and are disclosed. Instrumented CUDA
events include the distributed forward/hook span and CPU-induced stream gaps;
they are not isolated GPU kernel time. ES file IO is separately measured as
nested overhead; total hash/UD logging time is not separately isolated.

TCP bytes are actual application frames, including length prefix, JSON and tensor
payload. They exclude kernel IP/TCP headers and retransmissions. Endpoints are
localhost TCP; no geographical network latency or WAN reduction is measured.
Prefill/decode/control and directions are reported separately; final handshake,
warmup and shutdown traffic are audited separately from request totals.

Latency distributions use mean/median/nearest-rank P95/sample variance and
paired candidate-minus-reference absolute/percentage deltas. Negative latency
delta indicates improvement. Task-score differences retain their native metrics;
datasets with incompatible metrics are never pooled. HIT/MISS and truncation
subsets stay separate. Whole blocks retain their repeated observations; blocks
sharing a conversation are merged into a dependence component. Fewer than three
independent components do not yield CIs.
Measured warm/amortized benefits at tested prefix lengths are reported without
interpolation or token-count break-even estimates.

## Validation and execution

Final validation counts and artifact inventory are recorded below. No GPU
submission has been performed by this implementation task.

## Exact staged SERAPH commands

These are instructions for manual use, not commands run during implementation.
Review and commit I5 on `exp/raw-semcache-r5-integration` through your normal
workflow first. No commit or push is performed here. Use the recorded checkout
for subsequent independent audits; source drift fails closed.

Verify real I4A base preparation, model/tokenizer files and TinyBERT revision
`34707a33cd59a94ecde241ac209bf35103691b43` exist locally. `I5_BASE_PREPARED` must
identify a real OPT FP16 preparation on the same GPU UUID. If missing, use the
existing R3A preparation workflow in a separately reviewed allocation. Install
`rouge==1.0.1` for MultiNews, matplotlib for plots, and the compatible datasets
package for HF snapshots outside inference jobs.

```bash
cd /data/khuss/repos/GP_semcache
export I5_BASE_PREPARED=/absolute/path/to/verified/I4A/base_prepare
export I5_MULTI_PREPARED=/data/khuss/repos/GP_semcache/results/r5/i5/adapters_v1
python scripts/118_prepare_r5_i5.py adapters \
  --base-prepared "$I5_BASE_PREPARED" --output-dir "$I5_MULTI_PREPARED"
mkdir -p logs
```

The adapter command runs on CPU and refuses existing destinations. Existing
verified expanded I4A adapters may instead be supplied read-only. Create
`results/r5/i5/registry_one.json` with one dataset, and `registry_four.json` with
these absolute prepared paths, after checking that each exists:

```json
{
  "coqa": "/data/khuss/repos/GP_semcache/results/r5/i5/data/coqa_v1_dev",
  "snips": "/data/khuss/repos/GP_semcache/results/r5/i5/data/snips_test",
  "multiwoz": "/data/khuss/repos/GP_semcache/results/r5/i5/data/multiwoz22_test",
  "qasper": "/data/khuss/repos/GP_semcache/results/r5/i5/data/qasper_test"
}
```

One-dataset smoke: 10 unique scored samples, two repetitions, five targets/block:

```bash
python scripts/118_prepare_r5_i5.py experiment \
  --registry results/r5/i5/registry_one.json --base-prepared "$I5_BASE_PREPARED" \
  --scale SMOKE --tracks NATURAL --input-limit 512 --allow-context-truncation \
  --budgets configs/r5_i5_smoke_budgets.json --block-size 5 --repetitions 2 \
  --output-dir results/r5/i5/plan_smoke_one_v1
export I5_SOURCE_COMMIT="$(git rev-parse HEAD)"
export R5_I5_ENABLE_GPU=1 R5_ENABLE_GPU=1
export I5_EXPERIMENT=/data/khuss/repos/GP_semcache/results/r5/i5/plan_smoke_one_v1
export I5_RUN_ROOT=/data/khuss/repos/GP_semcache/results/r5/i5/smoke_one_v1
# Review block 0 and its sibling _audit before manually continuing.
sbatch --export=ALL --array=0-0%1 jobs/r5_i5_smoke_one.sh
sbatch --export=ALL --array=1-3%1 jobs/r5_i5_smoke_one.sh
```

Four-dataset smoke uses fresh plan and result directories:

```bash
python scripts/118_prepare_r5_i5.py experiment \
  --registry results/r5/i5/registry_four.json --base-prepared "$I5_BASE_PREPARED" \
  --scale SMOKE --tracks NATURAL --input-limit 512 --allow-context-truncation \
  --budgets configs/r5_i5_smoke_budgets.json --block-size 5 --repetitions 2 \
  --output-dir results/r5/i5/plan_smoke_four_v1
export I5_EXPERIMENT=/data/khuss/repos/GP_semcache/results/r5/i5/plan_smoke_four_v1
export I5_RUN_ROOT=/data/khuss/repos/GP_semcache/results/r5/i5/smoke_four_v1
sbatch --export=ALL --array=0-15%1 jobs/r5_i5_smoke_four.sh
```

Controlled comparisons use a separate real-context plan. Choose a one-dataset
registry with enough long examples, such as Qasper. Insufficient eligible
contexts fail; SNIPS is generally unsuitable for long-prefix experiments.

```bash
python scripts/118_prepare_r5_i5.py experiment \
  --registry results/r5/i5/registry_one.json --base-prepared "$I5_BASE_PREPARED" \
  --scale SMOKE --tracks CONTROLLED --prefixes 64 128 256 512 \
  --input-limit 768 --allow-context-truncation --budgets configs/r5_i5_smoke_budgets.json \
  --block-size 3 --repetitions 2 --output-dir results/r5/i5/plan_prefix_v1
export I5_EXPERIMENT=/data/khuss/repos/GP_semcache/results/r5/i5/plan_prefix_v1
export I5_RUN_ROOT=/data/khuss/repos/GP_semcache/results/r5/i5/prefix_v1
# This design has 32 blocks if preparation succeeds with ten eligible samples.
sbatch --export=ALL --array=0-31%1 jobs/r5_i5_controlled.sh
```

For a separately reviewed 1024-token plan use `--prefixes 1024`,
`--input-limit 1280`, `--block-size 1`, `--capacity-bytes 536870912`, and fresh
directories. Tokenizer/context/suffix, TCP framing and resource preflight must
all pass. A capacity rejection never counts as an eligible HIT.

After manually reviewing smoke outputs, prepare 100/500/1000-request plans:

```bash
python scripts/118_prepare_r5_i5.py experiment \
  --registry results/r5/i5/registry_four.json --base-prepared "$I5_BASE_PREPARED" \
  --scale SMALL --tracks NATURAL --input-limit 512 --allow-context-truncation \
  --budgets configs/r5_i5_task_budgets.json --block-size 5 --repetitions 2 \
  --seconds-per-forward 5 --output-dir results/r5/i5/plan_small_v1
# Repeat with --scale MEDIUM and plan_medium_v1 for 500;
# repeat with --scale LARGE and plan_large_v1 for 1000.
```

Replace five seconds/forward with a conservative measured smoke estimate.
Reduce block size if runtime/evidence estimates fail. An override JSON, e.g.
`{"snips":700}`, is supplied with `--counts`; absent unique eligible examples
produce `i5_eligibility_failure.json` and abort. Repetition cannot replace missing
unique samples. For the shown four-dataset, two-repetition, five-target design:

```bash
export I5_EXPERIMENT=/data/khuss/repos/GP_semcache/results/r5/i5/plan_small_v1
export I5_RUN_ROOT=/data/khuss/repos/GP_semcache/results/r5/i5/small_v1
sbatch --export=ALL --array=0-159%1 jobs/r5_i5_small.sh
# Set plan_medium_v1 and medium_v1, then manually:
# sbatch --export=ALL --array=0-799%1 jobs/r5_i5_medium.sh
# Set plan_large_v1 and large_v1, then manually:
# sbatch --export=ALL --array=0-1599%1 jobs/r5_i5_large.sh
```

Always read array bounds from the actual manifest if configuration changes:

```bash
python -c 'import json,os; print(len(json.load(open(os.environ["I5_EXPERIMENT"]+"/i5_workload_manifest.json"))["blocks"]))'
```

Before medium/large scheduling, inspect whole-plan planning estimates as well
as each block's bounds (these are estimates, not measurement results):

```bash
python - "$I5_EXPERIMENT" <<'PY'
import json, sys
from pathlib import Path
m=json.loads((Path(sys.argv[1])/'i5_workload_manifest.json').read_text())
r=[b['resources'] for b in m['blocks']]
print(json.dumps({
  'blocks':len(r),
  'maximum_forwards_all_conditions':3*sum(x['expected_maximum_forwards'] for x in r),
  'duration_estimate_seconds_all_conditions':3*sum(x['duration_estimate_seconds'] for x in r),
  'evidence_estimate_bytes_all_conditions':3*sum(x['estimated_evidence_bytes'] for x in r),
  'evidence_hard_budget_bytes_all_conditions':3*sum(x['evidence_budget_bytes'] for x in r),
  'maximum_gpu_estimate_bytes':max(x['gpu_estimate_bytes'] for x in r)
}, indent=2))
PY
```

`%1` serializes blocks; each element runs just one A/B/C block. Site array
limits may require manually split ranges. Failed blocks restart from their
beginning in fresh job directories. Preserve failed attempts, but supply an
explicit list of successful attempts to consolidation; duplicate retries fail.

Independent audit and complete-plan consolidation:

```bash
python scripts/120_analyze_r5_i5.py --inputs "$I5_RUN_ROOT/block_0_JOBID" \
  --output-dir results/r5/i5/reanalysis_block0_v1
export I5_ANALYSIS_OUT=/data/khuss/repos/GP_semcache/results/r5/i5/small_analysis_v1
sbatch --export=ALL jobs/r5_i5_analyze.sh
```

The analysis job uses CPU only. It rejects missing/failed/mismatched/mutated
blocks. Plot multiple measured scales with `122_plot_r5_i5.py --inputs
smoke_summary small_summary ...`; keep context/generation policies matched.

## Output organization

All new repository evidence lives under `results/r5/i5/`. Previous fidelity and
multiuser evidence is untouched. Plans contain selection/exclusion/truncation
reports and full schedules. Runs carry block-scoped workload identity linked by
hash to the prepared experiment, execution config and adapter registry. Each
condition owns separate accounting, initialization, warmup, wire, request,
generation, ownership and cache event records. The sibling `_audit` contains
the required request/cache/quality/latency/network/numerical/decision/audit
JSON/JSONL artifacts and CSV. Each is sealed by `i5_output_manifest.json`. CSVs and summaries retain `TEST_FIXTURE_ONLY`; generated plot titles and filenames explicitly separate CPU fixtures from GPU measurements.
Adapter preparation keeps the compatible I4A sealed adapter format.

Consolidation supplies native task tables, paired deltas/grouped CIs,
measured-only break-even points and repetition/plot tables. Eight plot families
cover prefix-vs-TTFT, prefix-vs-E2E, scale-vs-latency, scale-vs-HIT rate,
dataset/condition quality, dataset/condition bytes, HIT/MISS latency and quality
delta-vs-reuse. Dataset and track panels remain separate; no unmeasured or
interpolated point is produced.

## Future work

I3F-B numerical causality, native PEFT strict parity, concurrent serving,
alternative cache policies and I4B semantic clustering are outside I5. Larger
generation budgets than 128, alternate model positional extensions, full-context
LongBench, WAN experiments and trained adapters require separate validated work.

## Final local implementation and validation status

| Component | Implementation | Local validation | GPU |
|---|---|---|---|
| I5-0 local loaders, immutable provenance, reference scorers | IMPLEMENTED | CPU_VERIFIED for schemas, QA, intent, DST, scorer-unavailable rejection; real ROUGE package SKIP | GPU_NOT_RUN |
| I5-1 matched A/B/C and independent runtime reset | IMPLEMENTED | CPU_VERIFIED with actual localhost ES/UD processes | GPU_NOT_RUN |
| I5-2 Natural and exact-token Controlled workloads | IMPLEMENTED | CPU_VERIFIED construction, physical Controlled fixture and 1024-token owner/protocol | GPU_NOT_RUN |
| I5-3 scale plans, budgets and independent blocks | IMPLEMENTED | CPU_VERIFIED selection, resource bounds, missing/duplicate rejection | GPU_NOT_RUN |
| I5-4 audit, paired statistics, tables, eight plot families | IMPLEMENTED | CPU_VERIFIED audit/consolidation; plot rendering unavailable locally | GPU_NOT_RUN |

Final selected CPU suite: **112 tests: 110 PASS, 0 FAIL, 2 SKIP**. I5-specific
subset: 45 tests, 44 PASS and one ROUGE-package SKIP. The other SKIP is the
legacy installed OPT/cache API check, because torch/transformers/numpy are absent.
The combined suite includes legacy R5 contracts, semantics, cache controller,
multi-adapter authorization, fidelity, evidence accounting and R4C OPT contracts.

Commands actually run locally:

```bash
PYTHONPATH=src:tests python3 -m unittest \
  test_r5_i5 test_r5_i5_controller test_r5_contracts test_r5_semantic \
  test_r5_controller test_r5_multi test_r5_fidelity test_r5_perf_evidence \
  test_r4c_opt_contract -v
R5_I5_TEST_OUTPUT_ROOT=/home/songsong/SemCache/results/r5/i5/cpu_verified_scoped_delivery \
  PYTHONPATH=src:tests python3 -m unittest test_r5_i5_processes -v
python scripts/120_analyze_r5_i5.py \
  --inputs results/r5/i5/cpu_verified_scoped_delivery/run \
  --output-dir results/r5/i5/independent_audit_scoped_delivery
python scripts/121_summarize_r5_i5.py \
  --experiment results/r5/i5/cpu_verified_scoped_delivery/plan \
  --runs results/r5/i5/cpu_verified_scoped_delivery/run \
  --output-dir results/r5/i5/consolidated_cpu_fixture_scoped_delivery
```

CPU ES/UD TCP needed the sandbox's explicit socket permission; no remote host or
GPU was used. Final integration executes three scored targets and three priming
sources per condition, all three adapters, mixed strict/scalable capture and
separate uncached model warmup. The independent audit is PASS with **128 strict
KV checks per condition**, real fixture K/V consumption on all three targets,
and B–C shadow selection equivalence. Fixture token/text and frozen numerical
checks pass separately; historical GPU tolerance failures remain unresolved.
Unauthorized routes, mutated tensors, partial blocks and duplicate/missing
consolidation blocks are rejected. Read-only reanalysis leaves the source seal
unchanged. CLI help, compilation and all eight shell script syntax checks pass.

No trained GPU execution, real-data inference, GPU latency/quality estimate,
break-even region or graphical plot rendering is claimed. No explicit real
dataset registry was provided. `datasets/`, `/data/khuss/datasets`, and the
default local Hugging Face cache are absent in this workspace; `data/` has no
files discoverable by `rg --files`. The GPU environment must supply and verify
the actual snapshots. Local torch, numpy, transformers, datasets, rouge and
matplotlib are absent. Missing mandatory runtime/scorer resources fail closed.

NOT_SUPPORTED: concurrent serving, alternate clustering/policies, outputs over
128 tokens, OPT positional extension/full-length LongBench, and a full tensor
independent audit on every scalable request. FUTURE_WORK: I3F-B numerical causal
research, native PEFT strict parity and correctness-qualified speedup. The
instrumented localhost timing design and untrained adapters limit inference
beyond this controlled experiment.

All original tracked files remain byte-for-byte unchanged. All implementation
files below are additions; existing untracked historical evidence was preserved.

## Added implementation files

- `configs/r5_i5_smoke_budgets.json`
- `configs/r5_i5_task_budgets.json`
- `docs/R5_I5_INTEGRATED_EVALUATION.md`
- `jobs/r5_i5_analyze.sh`
- `jobs/r5_i5_common.sh`
- `jobs/r5_i5_controlled.sh`
- `jobs/r5_i5_large.sh`
- `jobs/r5_i5_medium.sh`
- `jobs/r5_i5_small.sh`
- `jobs/r5_i5_smoke_four.sh`
- `jobs/r5_i5_smoke_one.sh`
- `scripts/118_prepare_r5_i5.py`
- `scripts/119_run_r5_i5.py`
- `scripts/120_analyze_r5_i5.py`
- `scripts/121_summarize_r5_i5.py`
- `scripts/122_plot_r5_i5.py`
- `src/semcache/edgelora/r5_i5_analysis.py`
- `src/semcache/edgelora/r5_i5_artifacts.py`
- `src/semcache/edgelora/r5_i5_audit.py`
- `src/semcache/edgelora/r5_i5_cache.py`
- `src/semcache/edgelora/r5_i5_contracts.py`
- `src/semcache/edgelora/r5_i5_controller.py`
- `src/semcache/edgelora/r5_i5_data.py`
- `src/semcache/edgelora/r5_i5_driver.py`
- `src/semcache/edgelora/r5_i5_evidence.py`
- `src/semcache/edgelora/r5_i5_generation.py`
- `src/semcache/edgelora/r5_i5_limits.py`
- `src/semcache/edgelora/r5_i5_protocol.py`
- `src/semcache/edgelora/r5_i5_quality.py`
- `src/semcache/edgelora/r5_i5_runtime.py`
- `src/semcache/edgelora/r5_i5_session.py`
- `src/semcache/edgelora/r5_i5_summary.py`
- `src/semcache/edgelora/r5_i5_workload.py`
- `tests/test_r5_i5.py`
- `tests/test_r5_i5_controller.py`
- `tests/test_r5_i5_processes.py`
