# Independent natural audit and controlled physical-reuse pilots

Implementation/CPU fixtures only. GPU_NOT_RUN. No historical evidence is modified.
New outputs: `results/r5/i5r/parallel/`. Review, commit/push manually, then pull on
SERAPH before preparing: source inventories require clean tracked reviewed code.

## Scope and reuse

Experiment2 reuses3A dataset provenance, actual I5 prompts/context-only head/tail
truncation, tokenizer, routing, group packing and sharing checks. Experiment3
reuses the I5 matched ABC runtime, CPU UD/TCP, semantic discovery, Raw-QKV owners,
strict independent tensor/logit audit, BE physical checks/economics and lifecycle
cost scopes. No admission/eviction/inference/threshold code was modified.

The new GPU pilot generalizes the BE method; it is not a replay of the archived
N3 sample selection or8-token quality scope. Dataset generation budgets remain
Qasper32,2WikiMQA32,MultiNews96. There is one real positive sample repeated N
times per independent context and one real mismatched-context negative. The
512-token standalone source is CONTROLLED_EXACT_PREFIX, never a natural request.
N repeats do not create N independent examples. Existing146 remains the route
for exact archived BE reproduction. No historical performance number is pooled.

## CPU audit methodology and limitations

Provide a registry mapping dataset keys to existing sealed prepared directories:
`qasper`, `2wikimqa`, `multi_news`, `snips`, `multiwoz`. Missing keys/paths produce
DATASET_NOT_READY; corrupt/fixture manifests reject. This command does not acquire
or synthesize data. For actual local SNIPS/MultiWOZ sources, first inspect:

```bash
python scripts/118_prepare_r5_i5.py dataset --help
```

Use its existing dataset/revision/split/source/output/ontology options and a fresh
directory. Do not modify existing prepared LongBench artifacts. Default audit
input limit1280 is explicitly context limited; no official full-context claim.
`--max-samples 200` explicitly selects up to200 eligible originals per dataset
using the existing I5 selector. The original total, full eligible count, selected
IDs, unselected IDs and partial-conversation boundaries are reported. Pair rates
apply to that selected workload, not an entire larger SNIPS/MultiWOZ release.
The bound is configurable up to400; pair evaluation rejects beyond400 rather
than silently subsample. No duplicates are inserted to meet the requested bound.

Pairs have chronological source index<target index; denominator m*(m-1)/2 over
eligible requests. Identical prompts are reported separately from strict shorter
complete-prompt prefixes. Common-prefix fraction is not the safe reuse fraction.
Every diagnostic condition is computed independently. Raw w3 candidates use an
inverted index; occurrence multiplicity is frequency multiplication, never a
Cartesian enumeration. Longest contiguous match uses a linear suffix automaton.
Request and pair denominators remain separate. Reported positions are token
starts: prefix<3, suffix>=length-5, otherwise middle. First8 shared grams include
position lists; complete positions can be reconstructed from saved prompt IDs.

Origin classification uses actual-tokenizer offset mappings only when token IDs
agree. Fixed instructions, context, unambiguous question/history spans, inserted
truncation markers, cross-boundary and unknown windows are distinguished. With
slow tokenizers lacking offsets, only a decoded/byte-verified fixed instruction
prefix is labeled TEMPLATE; other positions remain UNKNOWN. No independent
retokenization is assumed to map content boundaries. `non_template_only` removes
TEMPLATE->TEMPLATE matches but includes UNKNOWN; it is not proven content reuse.
`proven_content_occurrences` excludes unknown and cross-boundary windows.

Semantic encoder/cluster acceptance is NOT_EVALUATED. Short-lifetime simulation
uses existing group packing and resets (default5 requests); the separate longer
sequential scenario retains LRU state across eligible records. Both assume
completion/admission/semantic success and enforce512MiB/adapter,64 entries,
original routing and authorization. These are hypothetical, not observed
residency, evictions or GPU HITs. Stored Raw Q/K/V costs491520bytes/token;
consumed K/V size is not substituted. Actual policy still has no TTL.

## SERAPH CPU commands

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=''
read -r -e -p 'Existing registry JSON (missing dataset keys allowed): ' IP_REGISTRY
read -r -e -p 'Existing expanded multi-adapter preparation: ' IP_MULTI
export IP_REGISTRY IP_MULTI
IP_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export IP_CPU="$PWD/results/r5/i5r/parallel/cpu_$IP_STAMP"
python scripts/149_r5_i5r_parallel.py cpu --registry "$IP_REGISTRY" \
 --multi "$IP_MULTI" --input-limit 1280 --block-size 5 --max-samples 200 --output-dir "$IP_CPU"
python scripts/149_r5_i5r_parallel.py cpu-verify --audit "$IP_CPU" \
 --output-dir "${IP_CPU}_verification"
```

CPU outputs: `experiment2_config.json`, `experiment2_summary.json/.csv`,
`<dataset>_pairs.jsonl`, `<dataset>_prompts.jsonl`, `experiment2_report.md` and
SHA256 `i5_output_manifest.json`. Verification recomputes every exported byte
into a fresh sealed directory, including tokenization and pair diagnostics.

## Manual gated GPU preparation and execution

The GPU pilot is independent of Experiment2 completion. It verifies its own
real dataset, model/tokenizer/adapters, scorer, exact512 prefix, context budget,
same original user/adapter, compatible negative and actual TCP envelope with
unchanged8192-byte reserve. Insufficient independent contexts/negative rejects.
All conditions begin with new model/ES/UD/cache state. A0 still runs auxiliary
B/C because the unchanged I5 runner requires ABC; economic reporting uses only
A0 FULL_RECOMPUTE. No fourth inference implementation is introduced.
ABC condition order rotates across context blocks. A0 follows ABC in each
paired pilot; thermal/order drift is a limitation, not controlled by one run.
Fresh repetitions are required before statistical/economic generalization.

First prepare Qasper N3/one-context; do not automatically execute a sweep:

```bash
export IP_SOURCE_COMMIT="$(git rev-parse HEAD)"
export IP_DATASET=qasper
# Later, only after reviewing Qasper: IP_DATASET=2wikimqa, then multi_news.
IP_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export IP_EXPERIMENT="$PWD/results/r5/i5r/parallel/${IP_DATASET}_N3_$IP_STAMP"
export IP_RUN_ROOT="${IP_EXPERIMENT}_runs"
python scripts/149_r5_i5r_parallel.py prepare --registry "$IP_REGISTRY" \
 --multi "$IP_MULTI" --dataset "$IP_DATASET" --contexts 1 --reuses 3 \
 --source-commit "$IP_SOURCE_COMMIT" --output-dir "$IP_EXPERIMENT"
python scripts/149_r5_i5r_parallel.py inspect --experiment "$IP_EXPERIMENT"
```

For a separately reviewed N1/N2/N4 plan, change `--reuses` and use a fresh path.
For expansion use `--contexts 3`; preparation requires three distinct original
context hashes. Submit each block manually with `IP_BLOCK_INDEX=0`, then1, then2;
no automatic context/dataset loop or array. The initial pilot submits only0.

```bash
export I5_CONDA_ROOT="$(conda info --base)"
export IP_NODE=moana-y3  # operator may instead select moana-y4
export IP_BLOCK_INDEX=0 R5_I5_ENABLE_GPU=1 R5_ENABLE_GPU=1
mkdir -p logs "$IP_RUN_ROOT"
df -h "$IP_RUN_ROOT"
sinfo -p batch_eebme_ugrad -N -o '%N %G %m %c %l %a'
scontrol show partition batch_eebme_ugrad
unset CUDA_VISIBLE_DEVICES
sbatch --export=ALL --nodelist="$IP_NODE" jobs/r5_i5r_parallel_pilot.sh
```

One RTX3090,4 CPUs,64GiB,one model at a time;6h20m bounded allocation,
two sequential I5 invocations each with10800s bound. GPU preflight remains the
unchanged119 checks: installed native OPT, offline model/TinyBERT, GPU identity,
22GiB conservative memory estimate, host RAM and disk. Combined hard evidence
reserve128GiB (six16GiB condition budgets plus32GiB reserve). No speed or
completion guarantee. Resource estimates can reject96-token or larger plans;
never silently change their budget. Do not run heavy CPU analysis on this node
concurrently. This script does not submit other jobs or launch analysis.

After the operator obtains the actual run directory (one for the one-context
pilot; all three explicitly listed for a three-context plan):

```bash
export CUDA_VISIBLE_DEVICES=''
read -r -e -p 'Completed fresh run block directory: ' IP_RUN
python scripts/149_r5_i5r_parallel.py analyze --experiment "$IP_EXPERIMENT" \
 --runs "$IP_RUN" --output-dir "${IP_RUN}_audit"
python scripts/149_r5_i5r_parallel.py verify --experiment "$IP_EXPERIMENT" \
 --runs "$IP_RUN" --audit "${IP_RUN}_audit" --output-dir "${IP_RUN}_verification"
```

GPU outputs: sealed `experiment3_plan.json`, `abc/` and `a0/` standard I5 plans;
run `experiment3_status.json` and unchanged child raw artifacts; independent
`experiment3_results.json`, `experiment3_independent_audit.json`,
`experiment3_verification.json` and inventories.
Results include full per-request metrics, native quality, frozen numerical
fidelity, physical K/V proofs, warm A/C target differences, BE positive-only and
negative-inclusive D(N), lifecycle/setup/warmup separately, and independent group
counts. D(N)<0 is measured only for that run/N, not a measured minimum or a CI.
No summing of nested admission/model timers. Localhost TCP is not a WAN model.
Physical HIT requires actual consumed K/V and omitted projections; cached Q is
not consumed. Missing/rejected semantic candidates remain misses, never retries.
Generated token equality, reference quality and numerical tolerance are separate.
NUMERICAL_FIDELITY may FAIL despite economic benefit. Correctness-qualified
speedup stays NOT_ESTABLISHED; unresolved FP16 differences remain unresolved.

## Local checks and current evidence availability

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r_parallel -q
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r_parallel test_r5_i5r3a \
 test_r5_i5r3b test_r5_i5r3bbe test_r5_i5_preflight test_r5_i5 \
 test_r5_i5_controller test_r5_i5r1 -q
bash -n jobs/r5_i5r_parallel_pilot.sh
```

Prepared real datasets/model assets and BE raw evidence are absent locally.
Five-dataset real audit NOT_RUN; all five readiness statuses locally unverified,
not invented zero natural counts. Eligible real GPU context counts NOT_VERIFIED.
Only older historical I5 audit ZIP is locally available, not a dataset substitute.
SNIPS/MultiWOZ acquisition/preparation remains an operator prerequisite when
their actual data are unavailable. No inference, Slurm, downloads, commit or push
was performed during implementation.

Final local validation:21 focused tests PASS; combined relevant suites167 PASS /
0 FAIL /1 SKIP (168 tests, includes focused tests). The skip is real
`rouge==1.0.1` scoring, absent locally; MultiNews preparation requires the real
pinned scorer. Four Python AST checks, CLI help/dispatch and shell fail-closed
test, `bash -n` and whitespace checks PASS. CPU fixtures do not establish native
OPT/GPU compatibility, real-data eligibility or real numerical fidelity.
