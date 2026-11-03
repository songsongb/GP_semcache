# R5-I5R-2B: earliest arithmetic divergence isolation

Status: IMPLEMENTED; local validation is CPU_ONLY. GPU_NOT_RUN. The physical
numerical mechanism remains ROOT_CAUSE_UNRESOLVED. This stage makes no performance
or restored-fidelity claim.

The R2A observations supplied by the researcher are historical, previously verified
GPU findings, not findings reproduced on this workstation: standalone-source QKV
already differed from the full-target prefix inside FULL_RECOMPUTE, whereas saved
and consumed source KV matched. That narrows the investigation upstream of cache
storage, without establishing FP16 GEMM as the cause. The original R1 tensors and
completed R2A review are on SERAPH; neither is assumed to exist locally.

## Implementation and production boundaries

New modules are `r5_i5r2b_plan.py`, `r5_i5r2b_capture.py`, and
`r5_i5r2b_analysis.py`. Scripts 130/131/132 prepare, execute, and independently
analyze the diagnostic. `jobs/r5_i5r2b_arithmetic.sh` is a manual Slurm wrapper.
`tests/test_r5_i5r2b.py` and `tests/test_r5_i5r2b_processes.py` contain CPU tests.

There are only two shared-file changes:

* `scripts/119_run_r5_i5.py` accepts `--enable-arithmetic-diagnostic`, requires a
  matching sealed diagnostic plan, and imports the diagnostic worker only when
  that plan is present. Normal workers retain their original imports and calls.
* `r5_i5_evidence.fresh()` additionally permits the narrow new namespace
  `results/r5/i5r/numerical_fidelity/earliest_arithmetic`. Existing-output,
  protected-input, symlink, inventory and source checks remain active.

The original `r3b_runtime.physical_path()` executes the CPU correction callback
before the native GPU Linear, then `(base + delta).to(base.dtype)`. No arithmetic
in that function, `user_device.cpu_deltas()`, the model, or the normal I5 driver
is changed. The diagnostic installs process-local wrappers, restoring them on exit:

| Boundary | Passive observation of the original execution |
| --- | --- |
| Input | Exact token IDs, observed position contract, actual additive attention mask, embedding outputs, Layer 0 input, pre-attention normalization input/output |
| Native base | Python frame observer identifies Layer 0 `miss()`; `TorchFunctionMode` records the actual original Linear return |
| Hidden transfer | Projection input; actual ES send/UD receive payload prefix, whole-payload SHA, byte count and shape; reconstructed UD input |
| CPU LoRA | Original per-role cast input, A output, actual B input, B output and scaled output from the six original CPU Linear calls and scaling operations |
| Delta transfer | Actual returned CPU bundle; UD send/ES receive payload identities; actual correction return after GPU device transfer |
| Total | Actual add return before final cast and original `miss()` return after cast; cross-check against the ordinary Raw-QKV capture |
| KV/attention | Existing head transform, source returned KV, actual HIT consumed prefix KV and preserved returned prefix KV; actual attention module output |

The observer calls each original Torch operation once and returns that operation's
original result. It does not recompute a hypothetical FP32 sum. Dtypes are recorded
from observed tensors. Internal GEMM accumulation precision is not inferred.
Normalization/module hooks return `None`; they never replace outputs. No new
parameters are installed. The normal projection restriction against preexisting
Linear hooks remains intact. Query prefixes do not execute on a HIT: their entries
are explicitly `NOT_CAPTURED: NO_PREFIX_QUERY_ROWS`, rather than fabricated.

Diagnostic OFF neither imports the observer nor alters CPU LoRA arithmetic.
The process regression executes both ON and OFF through the original I5 subprocess
and localhost TCP protocol, checking identical generated IDs/text/logit hashes.
That is CPU fixture evidence only. The small installed-Torch observer API check
is mandatory during real preparation and is separately skipped locally if Torch
or NumPy is absent. It is not a GPU fidelity check.

## Minimal experiment and capture budget

Preparation reads and verifies the immutable historical R1 plan and completed R2A
export. It requires verified historical blobs and matching historical commit,
model/tokenizer identities and three adapter fingerprints. The historical plan
provides token IDs only; no historical tensors become new cache inputs. The actual
offline tokenizer must reproduce the full target IDs. The original files are
never modified or resealed.

The freshly sealed plan has one Qasper CONTROLLED block, Adapter_A/User_01, the
same standalone source of 64 IDs followed by the full 256-ID target, and A/B/C
conditions with independent empty cache/index/model sessions. All three synthetic
adapters and the thirty-user registry remain validated; only A receives measured
requests. Existing adapter warmups remain. The source generates one real token;
the target generates at most two real tokens, permitting one decode transition.
The cache source consists only of its independently computed 64-token prefill.
The integrated HIT must be produced by the existing semantic/controller path.
The analyzer rejects a run without the legitimate 64-token HIT/source receipt.

The primary comparison is S[0:64] versus F[0:64] **within FULL_RECOMPUTE**. Captures
are Layer 0 only. The target's suffix is excluded from tensor evidence; full
input token IDs and context metadata still document the actual shapes. The additive
mask capture keeps 64 query rows and all actual keys, at most 256; analysis aligns
the first 64 keys and verifies every extra future key is masked. Returned/consumed
head-layout KV retains all 64 selected prefix positions. No attention score or
probability matrix is captured.

Each ES or UD capture inventory has a hard 32 MiB / 128 tensor-file cap.
Exhaustion fails before writing another tensor; it never silently samples.
Every tensor records exact retained bytes/SHA, observed shape, dtype, device,
stride/contiguity/storage offset when observable, axis and absolute token interval.
Missing scalar-fixture-only boundaries are explicit. Whole-payload digests bind
selected transfer bytes to the original independent TCP ledger, which still
validates all frames. All output files enter the normal I5 SHA-256 seal.

Normal I5 evidence is SCALABLE_EVALUATION for this diagnostic: no new all-32-layer
full tensor dumps. Ordinary metadata, Raw-QKV hashes/proofs, request completion,
logits, CPU/TCP/session/authorization receipts and frozen comparisons remain.
The independent **byte audit is Layer 0 only**; it is not an all-layer strict tensor
audit. This selection is explicit in the sealed plan and report, not a historical
audit-gate bypass. Existing R1 strict runs are unchanged.

The Slurm contract remains one runtime-verified RTX 3090, 4 CPU cores, 64 GiB RAM,
one hour allocation and a 3,000-second execution budget. Existing GPU estimate
ceiling (22 GiB), 32 GiB available CPU-memory reserve, offline assets, clean tracked
source and disk checks remain. Each condition has a 512 MiB total evidence cap
including diagnostic captures, or at most 1.5 GiB for three conditions, with the
existing free-disk check requiring more than 2.5 GiB. Capture limits are nested
within that cap, not additional disk allowances. The standard 512 MiB per-adapter
cache capacity is retained when present in the historical smoke plan and is
validated using the existing resource estimator. No cross-job live cache is loaded.

## Analysis and interpretation

Script 132 first executes the unchanged independent I5 analyzer, validates current
pinned source, preparation identities, COMPLETE status and every sealed byte, then
checks the targeted captures. It rejects corrupt/ambiguous/missing required normal
evidence, mismatched payload receipts, unsafe masks, wrong alignment, failed
source-to-consumed-KV identity or missing actual HIT. Missing diagnostic boundaries
are reported explicitly; a real run gets `GPU_EVIDENCE_COMPLETE=FAIL` and nonzero
exit. An observed numerical failure does not prevent preserving the diagnostic.

Comparisons record bitwise equality, max/mean absolute error, relative L2, frozen
elementwise violations and frozen max-absolute/relative verdicts. Existing logit
KL, argmax, token/text and frozen numerical gates remain in the attached ordinary
I5 audit. Numerical equality of activations is not substituted for logit fidelity.

The report distinguishes first observed difference, boundaries with matching
recorded inputs/different outputs, and a confirmed mechanism. Computational order
respects that CPU correction executes before the native base Linear. Where branches
are peers, both are visible. A localized Linear boundary does not establish kernel
selection, accumulation precision, shape dependence or nondeterminism as the cause.
CASE_A through CASE_F are derived from actual comparisons. Missing operands never
justify attributing an output difference to that operation. Backend/thread/precision
settings are captured before and after actual S/F execution, not inferred from a
seed or initialization default.

| Observed outcome | Next smallest action after review |
| --- | --- |
| Projection input differs | Inspect embedding, positional and normalization inputs/outputs before blaming projection |
| Input matches, base differs | Propose an isolated same-input 64/256-row Linear test and same-shape repeats; then optional FP32/backend control |
| Base matches, delta differs | Use wire/reconstruction/cast/A/B/scaled boundaries to find the first differing CPU or transport result |
| Base and delta match, total differs | Separate actual sum from final cast; inspect layouts/settings |
| Total matches, attention differs | First compare actual masks/KV/positions; propose an input-controlled attention test if still necessary |
| Incomplete or instrumentation-sensitive | Capture the specifically missing operand, or compare instrumented and uninstrumented GPU execution |

The optional isolated shape/FP32 experiment is deliberately NOT_IMPLEMENTED until
this first GPU diagnostic justifies it. Same-shape repeat variability and GPU
instrumentation effects are NOT_RUN. The analyzer retains ROOT_CAUSE_UNRESOLVED
and NUMERICAL_FIDELITY_RESTORED=NOT_ASSERTED. No diagnostic latency can be used as
a serving speedup measurement; synchronous capture changes scheduling and timing.

## Local CPU validation

No model downloads, CUDA inference or Slurm commands are involved:

```bash
cd /home/songsong/SemCache
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r2b test_r5_i5r2 test_r5_i5r1 -v
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r2b_processes -v
bash -n jobs/r5_i5r2b_arithmetic.sh
python3 scripts/130_prepare_r5_i5r2b.py --help
python3 scripts/131_run_r5_i5r2b.py --help
python3 scripts/132_analyze_r5_i5r2b.py --help
```

The process test needs permission to open localhost TCP sockets. Without
`R5_I5R2B_TEST_OUTPUT_ROOT` it uses a temporary directory; to preserve fixture
inputs, sealed runs, both audits and logs, set that variable to a **fresh** path in
the new namespace. CPU fixtures explicitly label every result TEST_FIXTURE_ONLY.
The stdlib scalar fixture cannot observe Torch normalization, promoted sum or
additive attention mask tensors; those are NOT_CAPTURED, not fabricated PASSes.

Final local validation against the delivered worktree:

| Suite | Executed | PASS | FAIL | SKIP |
| --- | ---: | ---: | ---: | ---: |
| I5R2B, R2A, R1, R0, I5 and cache/controller regressions | 251 | 246 | 0 | 5 |
| I5R2B distributed fixture / OFF comparison | 3 | 3 | 0 | 0 |
| Existing I5R1 distributed strict fixture / audit regressions | 5 | 5 | 0 | 0 |
| Total | 259 | 254 | 0 | 5 |

The five skips are the installed-Torch observer CPU API, pinned ROUGE dependency,
installed OPT/eager-attention API, actual Torch cache ownership, and actual HF OPT
CPU suffix/decode checks. Torch/NumPy/Transformers and `rouge==1.0.1` are absent
locally; these checks are NOT_RUN, not PASS. The real preparation/runner requires
the observer and installed-OPT checks. An initial sandbox socket restriction
prevented TCP startup; the final socket-enabled CPU runs passed. Failed local
attempts were preserved separately. Python syntax, Slurm shell syntax, all three
CLI help commands and tracked diff whitespace checks also passed.

Retained final fixture evidence is in
`results/r5/i5r/numerical_fidelity/earliest_arithmetic/cpu_delivery_03`; the existing
R1 strict regression is in the sibling `legacy_r1_delivery_03`. These include
independent CPU audits, not GPU tensor verification. EOS can end the tiny fixture
before its maximum budget; an unexecuted decode step is not claimed as verified.

All implementation files are intentionally pending user review and commit. Exact
local staging commands (no results, logs or unrelated untracked files):

```bash
git add scripts/119_run_r5_i5.py scripts/130_prepare_r5_i5r2b.py \
  scripts/131_run_r5_i5r2b.py scripts/132_analyze_r5_i5r2b.py \
  src/semcache/edgelora/r5_i5_evidence.py \
  src/semcache/edgelora/r5_i5r2b_plan.py \
  src/semcache/edgelora/r5_i5r2b_capture.py \
  src/semcache/edgelora/r5_i5r2b_analysis.py \
  tests/test_r5_i5r2b.py tests/test_r5_i5r2b_processes.py \
  jobs/r5_i5r2b_arithmetic.sh docs/R5_I5R2B_EARLIEST_ARITHMETIC.md
git diff --cached --stat
git diff --cached --check
git commit -m "Add opt-in R5-I5R-2B arithmetic divergence diagnostic"
```

The user performs deployment separately. No commit or push was performed by the
implementation agent. Real experiment preparation fails closed until all execution
sources are tracked, the reviewed commit is pinned and the tracked tree is clean.

## SERAPH preparation after user-managed source deployment

The user reviews, commits and pushes all the listed source/scripts/job/tests/docs,
then manually updates SERAPH. Do not reuse the historical R1 plan for execution.
Do not change source between preparation, execution and analysis; a new source
revision requires a new sealed plan and fresh attempt. Earlier evidence analysis
under its historical source remains separate and strict.

```bash
cd /data/khuss/repos/GP_semcache
git switch exp/raw-semcache-r5-integration
git pull --ff-only
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export I5R2B_SOURCE_COMMIT="$(git rev-parse HEAD)"
git status --short --untracked-files=no
printf 'Compare this SHA with the reviewed commit: %s\n' "$I5R2B_SOURCE_COMMIT"
export I5R2B_MULTI_PREPARED="$PWD/results/r5/multiuser/expanded_152381_20261031T154848Z/adapters"
export I5R2B_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export I5R2B_EXPERIMENT="$PWD/results/r5/i5r/numerical_fidelity/earliest_arithmetic/plan_$I5R2B_TAG"
export I5R2B_RUN_ROOT="$PWD/results/r5/i5r/numerical_fidelity/earliest_arithmetic/run_$I5R2B_TAG"
python scripts/130_prepare_r5_i5r2b.py \
  --historical-plan results/r5/i5r/controlled_hit/plan_smoke_v1 \
  --r2a-review results/r5/i5r/numerical_fidelity/historical_review_01 \
  --multi-prepared "$I5R2B_MULTI_PREPARED" --output-dir "$I5R2B_EXPERIMENT"
```

This CPU preparation requires installed Torch/NumPy (actual observer API check),
the existing compatible Transformers/PEFT stack and verified offline model/tokenizer,
adapters, dataset manifests and R2A export. The real GPU runner also repeats the
existing installed-OPT and pinned TinyBERT checks. Missing dependencies fail;
there is no download or synthetic fallback.

## One manual GPU submission and independent review

Only after reviewing the preparation output and pinned SHA, the operator submits:

```bash
export R5_I5_ENABLE_GPU=1 R5_ENABLE_GPU=1
mkdir -p logs
I5R2B_JOB_ID="$(sbatch --parsable --export=ALL jobs/r5_i5r2b_arithmetic.sh)"
export I5R2B_JOB_ID
printf 'Submitted one diagnostic job: %s\n' "$I5R2B_JOB_ID"
```

No array or automatic follow-up is submitted. The wrapper permits index 0 only.
After completion, check the status and rerun the CPU analyzer into a new audit
directory under the same pinned source (the wrapper already made the first two
audits automatically):

```bash
export I5R2B_RUN="$I5R2B_RUN_ROOT/block_0_${I5R2B_JOB_ID%%;*}"
python -m json.tool "$I5R2B_RUN/i5_block_status.json"
python -m json.tool "${I5R2B_RUN}_audit/i5_independent_audit.json"
python scripts/132_analyze_r5_i5r2b.py --experiment "$I5R2B_EXPERIMENT" \
  --run-dir "$I5R2B_RUN" --output-dir "${I5R2B_RUN}_arithmetic_review_02"
```

If a shell was restarted, restore the exported plan/run/job paths from the Slurm
log and preparation output; never guess a different plan or reseal old evidence.

Expected outputs in the new independent arithmetic audit:

* `i5r2b_capture_inventory.json`
* `i5r2b_tensor_comparisons.json`
* `i5r2b_first_divergence.json`
* `i5r2b_gpu_base_vs_lora.json`
* `i5r2b_dtype_and_backend.json`
* `i5r2b_hypothesis_verdicts.json`
* `i5r2b_report.md`
* `i5r2b_output_manifest.json`

Inspect COMPLETE, the ordinary independent audit, the targeted byte-integrity
checks, GPU_EVIDENCE_COMPLETE, first observed/localized boundary, frozen logit
verdicts and explicit missing fields separately. An ordinary metadata audit PASS
or identical generated tokens does not imply numerical fidelity. A functional
HIT or localized arithmetic boundary does not imply a confirmed root cause.
