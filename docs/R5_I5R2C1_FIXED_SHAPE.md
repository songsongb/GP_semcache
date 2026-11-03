# R5-I5R-2C-1 fixed-shape FP16 feasibility

DIAGNOSTIC_ONLY. Local CPU fixtures verify construction/audit logic, not GPU
numerics. Full-model fidelity, root cause and E2E benefit are NOT evaluated.
Production inference, LoRA, TCP, cache, attention, generation and frozen thresholds
are untouched. Implementation: `r5_i5r2c1.py`, CLI `137_r5_i5r2c1.py`,
`r5_i5r2c1_linear.sh`, focused tests and this document.

Preparation requires explicit sealed S1 experiment, raw run and independent audit
from job 152859. Historical source blobs are verified against Git, separately
from the current pinned source. Archived independent audit/inventory/configuration,
model/weight hashes, input/output shapes and all original repetitions are checked.
This consumes a verified historical audit; it does not rerun the historical S1
analyzer under changed source. No historical file is modified or resealed.

The contiguous S1 FP16 input and immutable A/B outputs are copied into a new plan.
B may contain a synthetic S1 suffix: no claim of the original 256-row hidden state
is made. A is 64 rows; B, ZERO, NONZERO and SAME256 are 256 rows. ZERO uses 192
zero rows; NONZERO uses a versioned S1 xorshift suffix with recorded seed (starting
43, changed if needed to differ from B); SAME256 exactly replays B as a sanity
control only. ZERO identical to B is rejected as a tautological primary control.
All prefixes remain the exact recorded 64 rows. Only isolated Linear hidden states
are padded, never tokens or masks.

One actual bare OPT-2.7B FP16 model is loaded once using existing utilities.
Layer 0 actual Q/K/V native Linear.forward calls preserve weights and bias.
Each kind has two warmups then six measurements, alternating forward/reverse
condition order. Prefix/input/weight/backend/GPU receipts are checked around
execution. Only output positions 0:64 are saved. Complete model fingerprints
before/after must match. GPU UUID may differ from preparation or S1, but must be
one real compatible RTX 3090. Backend settings are explicitly replayed from S1,
with initial/effective/final settings recorded, not tuned for equality.

Comparisons distinguish historical A/B, ZERO/NONZERO versus unchanged S1 B,
current B versus S1 B, current A versus S1 A, SAME256 versus current B, and the
available original historical Base output (qualified: S1 suffix/layout may differ).
Each record reports bitwise equality, differing values/elements, errors, frozen
violations and ULPs using existing S1 metrics. EXACT means that recorded prefix
matches the unchanged S1 B for that repetition. CLOSER requires non-increasing
max-absolute and relative-L2 errors versus historical A/B with at least one strict
improvement. DIFFERENT is bitwise inequality. Repeatability is reported separately
for every role/kind; unstable results do not establish a reliable correction.
Missing/corrupt/incomplete evidence fails before a success report is written.

FLOPs use conventional 2 operations per MAC plus one add per biased output,
using actual projection dimensions. 256 rows cost 4x the 64-row theoretical work;
extra work is 192 rows, with QKV totals reported. CUDA events bracket only the
native forward call after synchronization; copies, hash checks and evidence I/O
are excluded from event time. Raw timings, means and medians are operator-only
measurements affected by instrumentation/cache/clock state, not serving E2E.
Source cost may be amortized over future HITs, but no net benefit is inferred.

Bounds: one RTX 3090, four cores, 64 GiB Slurm RAM, 32 GiB available CPU reserve,
20 minutes allocated/900-second operation deadline, conservative model+5 GiB GPU
headroom, over 1 GiB disk headroom, 256 MiB evidence budget with existing Writer
accounting and 256 files. No FP32 experiment or additional model/dataset sweep.

Local commands:

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r2c1 -v
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r2bs1 -v
bash -n jobs/r5_i5r2c1_linear.sh
python3 scripts/137_r5_i5r2c1.py --help
```

After user review/commit/push and manual SERAPH pull, prepare on CPU:

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export I5_CONDA_ROOT="$(conda info --base)"
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
read -r -p 'Reviewed C1 commit SHA: ' C1_SOURCE_COMMIT
export C1_SOURCE_COMMIT
test "$(git rev-parse HEAD)" = "$C1_SOURCE_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
read -r -e -p 'Sealed S1 experiment: ' S1_PLAN
read -r -e -p 'Original job 152859 raw run: ' S1_RUN
read -r -e -p 'S1 independent audit: ' S1_AUDIT
export C1_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export C1_EXPERIMENT="$PWD/results/r5/i5r/numerical_fidelity/fixed_shape_feasibility/plan_$C1_TAG"
export C1_RUN_ROOT="$PWD/results/r5/i5r/numerical_fidelity/fixed_shape_feasibility/attempt_$C1_TAG"
python scripts/137_r5_i5r2c1.py prepare --s1-experiment "$S1_PLAN" \
  --s1-run "$S1_RUN" --s1-audit "$S1_AUDIT" --output-dir "$C1_EXPERIMENT"
```

Exactly one manual submission (no array, no automatic follow-up):

```bash
read -r -p 'Permitted node (moana-y3 or moana-y4): ' C1_NODE
export C1_NODE C1_ENABLE_GPU=1
export CUBLAS_WORKSPACE_CONFIG="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["backend"]["CUBLAS_WORKSPACE_CONFIG"])' "$C1_EXPERIMENT/i5r2c1_plan.json")"
mkdir -p logs
C1_JOB_ID="$(sbatch --parsable --export=ALL --nodelist="$C1_NODE" jobs/r5_i5r2c1_linear.sh)"
export C1_JOB_ID
```

CPU independent audit after completion, using the same pinned source:

```bash
export C1_RUN="$C1_RUN_ROOT/run_${C1_JOB_ID%%;*}"
python -m json.tool "$C1_RUN/i5r2c1_status.json"
CUDA_VISIBLE_DEVICES='' python scripts/137_r5_i5r2c1.py analyze \
  --experiment "$C1_EXPERIMENT" --run-dir "$C1_RUN" --output-dir "${C1_RUN}_audit_01"
```

New plans/runs/reports use only `fixed_shape_feasibility/`; existing output is
rejected. Plan: `i5r2c1_plan.json`, inputs and historical reference prefixes.
Run: plan, `i5r2c1_records.json`, `i5r2c1_environment.json`,
`i5r2c1_status.json`, prefix outputs and Writer accounting. Audit:
`i5r2c1_comparisons.json`, `i5r2c1_cost.json`, `i5r2c1_summary.json`,
`i5r2c1_provenance.json`. All use SHA-256 `i5_output_manifest.json` seals.

No GPU run, submission, remote access, model download or push is performed locally.
Even exact Layer 0 prefixes do not establish full-model fidelity or production
feasibility; further integration validation requires separate review.

Local validation: focused 7 PASS; S1 regressions 22 PASS / 1 SKIP; total
29 PASS / 0 FAIL / 1 SKIP. The skip is the actual S1 CPU Torch helper because
Torch/NumPy are unavailable locally. The positive fixed-shape audit unit uses
explicit synthetic schema/output fixtures; it is not evidence of GPU equivalence.
CLI help, Python syntax and Slurm shell syntax passed. Real S1 evidence from
152859 is not available locally; GPU numerical and timing results remain NOT_RUN.
All five added files are pending user review/commit:

```bash
git add src/semcache/edgelora/r5_i5r2c1.py scripts/137_r5_i5r2c1.py \
  jobs/r5_i5r2c1_linear.sh tests/test_r5_i5r2c1.py docs/R5_I5R2C1_FIXED_SHAPE.md
git diff --cached --check
git diff --cached --stat
git commit -m "Add diagnostic fixed-shape FP16 projection feasibility experiment"
```
