# R5-I3F numerical fidelity forensics

R5-I3F adds a bounded, opt-in tensor investigation around the existing R5-I3P
production path. It does not implement R5-I4A. GPU execution is operator controlled.
Local CPU results do not establish GPU numerical parity or a GPU root cause.
**ROOT_CAUSE_UNRESOLVED** until the actual GPU evidence justifies a narrower claim.

The historical observations supplied for this task remain the starting point:
R5_03 max logit absolute error about 0.0234375 and R5_08 about 0.0200195, each
failing the frozen 0.01 gate in all 16 balanced comparisons, despite exact tokens
and text. Those failures have not been reproduced on this CPU-only host.
FP16 use alone is not a causal explanation. Historical R5-I3P performance
qualification remains false. Native PEFT strict parity and correctness-qualified
speedup remain unestablished.

## Scope and preservation

All implementation files are additive. No existing R3/R4/R5 source, numerical
threshold, model/adapter value, protocol, ownership format, eligibility rule,
semantic policy or benchmark schedule is edited. Job 152301, Diagnostic 152324
and Performance 152325 are never opened for writing. Output is restricted to a
new empty directory under `results/r5/fidelity/` inside this repository; external
empty directories support CPU tests. Nonempty output directories are rejected.

The runner uses the actual I3P block orchestration with a process-local injection
of an observing driver, owner, Channel and UD worker. Both conditions generate
through the unchanged `r5_generation.generate` and `QualityDriver.c_forward`.
The actual model forward, correction arithmetic, CPU-only independent UD,
completion controls and TCP framing remain unchanged. Every temporary class/IO
injection is scoped and restored. Hooks return no replacement output. Tensor
snapshots are owned before subsequent in-place scaling or cache append.

There is one A block (Distributed No-Reuse) and one B block (physical reuse),
each executing the original ten requests in their original order. This is 20
completed requests with G8 by default, rather than another 320-request audit.
G2–G8 are explicitly bounded; G8 matches the historical generation budget.
Each block loads the same prepared weights, matrices, tokenizer and immutable
TinyBERT revision on the same allocated GPU UUID. Blocks have separate ES/UD
processes, sessions, indices and caches. No warmup is included and no performance
conclusion is drawn from their order or timings.

The full ten-request sequence preserves admission/eviction and semantic cluster
state, including requests that are not individually investigated. Four target
IDs select observations, never cache decisions. Normal semantic discovery,
exact-prefix validation and sealed physical ownership still determine HITs.
The analyzer independently requires the observed original lifecycle. Detailed
hooks cover R5_00, R5_01, R5_03, R5_04, R5_07, R5_08 and R5_09.

| Target | Expected historical source relationship, checked after execution |
|---|---|
| R5_03 | R5_00 completed standalone source |
| R5_08 | R5_07 repopulates the evicted library prefix |
| R5_01 | R5_00 completed standalone source, comparison control |
| R5_09 | R5_04 completed standalone source, comparison control |

The recorded grant determines each report's actual source. Unexpected or absent
physical HITs fail diagnostic completeness; they are never relabeled as HITs.
Source completion and admission must precede target arrival. Only the source's
prefill tensors are compared with the target's corresponding prompt positions.
Its generated continuation is retained as completion proof and is not compared
with an unrelated target continuation.

## Captures and alignment

Passive module hooks capture every selected actual OPT layer's incoming hidden
state, pre-attention normalization, attention projection input, corrected Raw
Q/K/V, attention context before out-projection, attention output, second
normalization, FC1 output, activation into FC2, FC2 output, and layer output.
The LM-head hook captures its actual input, including final decoder normalization.
The driver stores actual last-query logits and the generated production tokens.

A scoped Python tracer observes already computed locals in the installed eager
attention/layer methods and the frozen physical projection closure. It never
reexecutes a projection, attention, normalization, FFN or LM head. Assignment
sites are discovered from the installed source, rather than fixed HF line
numbers. Records include assignment expressions and source locations. Where
available this captures raw attention scores, additive masked scores and actual
softmax probabilities. Attention's final actual K/V locals are captured with an
inspected layout and checked against returned KV. The two residual-add results
are captured from real layer locals. The physical closure supplies the actual
native Linear result and actual transported CPU LoRA delta separately from the
corrected projection. Unknown required structures/layouts fail the run and
preserve evidence. Optional unavailable attention intermediates remain a stated
coverage limit; `output_attentions=False` is preserved.

Requested token IDs, full tokenization, positions, attention masks, cache positions,
actual position-embedding observations, actual attention masks, fingerprints,
source completion and cache proofs are retained. Actual additive masks are
checked for correct query/key dimensions and causal visibility. Negative infinity
mask values are encoded explicitly in JSON; ordinary tensor evidence rejects
nonfinite values. The real additive-score intermediate may briefly become
negative infinity before OPT's unchanged clamp. Those bytes are preserved with
an explicit exception limited to verified future masked positions. Its metrics
report nonfinite counts, pattern changes and errors on finite corresponding
values; a finite/nonfinite pattern mismatch has no finite error magnitude.
NaN, positive infinity and negative infinity at visible positions fail validation.
Numerical comparisons verify dtype, layout, shape and selected
positions before reading values. Source-prefix attention scores/probabilities
select both prefix query and prefix key positions; future masked columns are
excluded. Suffix/full comparisons select matching suffix queries and all matching
keys. Prefix positions are never compared against suffix positions.

Decode is compared only while all previously generated IDs agree. The step that
first selects a different token is still directly comparable; later steps are
labeled `UNMATCHED_DECODE_HISTORY_OR_LENGTH`. Such steps cannot pass the complete
frozen numerical gate. Changed tokens and text are reported separately.

Every tensor comparison reports max absolute error, relative L2, cosine similarity
when both norms are nonzero, differing values, bitwise equality, finite checks,
shape, dtype, layout and both output hashes. Zero-valued tensors have no defined
cosine. Signed-zero byte differences remain visible independently of value error.
First-divergence selection orders numeric layer indices and operations, rather
than lexicographic names or final logit differences. Parallel base/delta and
Q/K/V branches retain causal peers. The report says **first observed operation**;
it does not exonerate an internal operation that was not observed.

## Reconstruction and hypothesis controls

At a real owner-selected HIT, evidence separately snapshots the sealed cached
Raw K/V, Raw-derived attention K/V and the actual reconstructed `past_key_values`
returned to the generator. Before the first suffix forward the driver snapshots
the actual consumed past. The unchanged production checks also validate that
suffix/Decode append preserves the cached prefix. The auditor independently
transforms raw bytes to the verified head layout without invoking model code.

It reports six comparisons for every layer and K/V role:

1. Source prefill Raw K/V versus owned cached Raw K/V.
2. Source attention K/V versus independently Raw-derived K/V.
3. Source attention K/V versus reconstructed past.
4. Derived K/V versus reconstructed past.
5. Reconstructed past versus actual consumed input past.
6. Source attention K/V versus the independently executed full target's prefix K/V.

The first five are integrity requirements. The sixth may differ numerically
without violating the first five. A difference in the sixth never becomes an
assertion that conversion corrupted the source. All selected source/target
layers also have a full source-prefix versus full-target-prefix comparison and
a separate full-target versus suffix/Decode comparison.

| Hypothesis | Concrete control/observation | Inference limit |
|---|---|---|
| H1 source/full-target shape | Exact corresponding tokens, positions and prefix masks, every layer's input/projections/attention/output | Different prefix values establish the observation; specific kernel causation needs an intervention |
| H2 Raw conversion | Independent byte permutation; actual used/source KV, derived and constructed KV | Identical bytes exclude conversion-induced value changes for the captured tensors |
| H3 reconstructed-past consumption | Source → derived → constructor → actual input past, plus unchanged-prefix append checks | Preserved contents do not isolate effects of the cache API or attention execution |
| H4 suffix/full path | Matched query slices, actual positions/masks, ordered operations and aligned Decode | Differences may include effects inherited from differing prefix K/V |
| H5 context/layout bug | Actual position observation, actual causal-mask values, API/type contract and full shape/layout/identity checks | Covers the inspected eager OPT path and observed contexts |
| H6 CPU LoRA/TCP | Actual CPU arithmetic input/result before serialization; both endpoint payload files; actual ES correction locals | Excludes byte alteration; equal-input delta differences across shapes remain a separate arithmetic observation |
| H7 numerical/kernel shape | Equal corresponding projection inputs with separately observed native-base and delta differences, fixed weights/dtypes | Supports arithmetic shape sensitivity when measured; no precision sweep or kernel-profiler attribution |

Wire checks connect snapshots to the actual payload digests in the independently
replayed TCP ledger. CPU results are captured before serialization, then checked
against transmitted, received and GPU-reconstructed delta bytes. This is not an
independent reimplementation of CPU GEMM. No higher-precision computation silently
replaces production FP16. No synthetic perturbation is injected into GPU runs.
There is no cache-container intervention, controlled precision sweep or repeated
same-shape GPU baseline in this bounded suite. Unresolved hypotheses stay unresolved.

No localized bug is established by CPU fixtures, so no correction is introduced.
An eventual correction requires direct GPU evidence, a mathematical explanation,
an explicit experimental opt-in, and regression against the unchanged reference.
There is no hidden full-prefix recomputation on a HIT. The unchanged controller
fails closed on corrupted ownership and rejects unsupported identities. The A
block provides an actual no-reuse reference/fallback without consuming cached
KV; an incomplete B block cannot promote any completion/parity gate.

## Evidence, audit and gates

Each block reuses I3P's shared ES/UD writer quota, exclusive tensor writes,
content digests, reconciliations and read-only accounting audit. Default block
cap is 4 GiB, including the existing conservative 64 MiB ledger reservation;
reports/preparation have a separately checked 1 GiB allowance. Budget overflow,
short writes, missing data, unknown files, symlinks, changed sources or corrupted
hashes fail closed. Partial logs and failure stage remain available. A final
R5 content-hashed manifest covers all run artifacts and accounting ledgers.
Reanalysis writes a fresh directory and never edits its inputs.

| Output | Contents |
|---|---|
| `fidelity_sources.json`, `fidelity_config.json` | Execution source hashes, committed identity, fixed settings and selected IDs |
| `fidelity_identity.json`, `fidelity_contexts.json` | Base/adapter/tokenizer/encoder/GPU contract; full source and target contexts and completion proofs |
| `fidelity_dependencies.json` | Actual grants and causal completion/admission dependencies |
| `blocks/*/es/fidelity_steps.jsonl` | Tensor descriptors and files; layer context and source assignment sites |
| `blocks/B/es/fidelity_reconstruction.jsonl` | Actual cached/derived/constructed snapshots and sealed source proof |
| `blocks/*/es/fidelity*wire.jsonl`, `fidelity_ud_math.jsonl` | Actual endpoint payloads and pre-serialization CPU tensors |
| `fidelity_comparisons.json` | Ordered source/full-prefix and matched suffix/Decode layer measurements |
| `fidelity_reconstruction.json` | Independent byte conversion/integrity and source/full-target KV comparisons |
| `fidelity_divergence.json` | Per-request first observations, magnitudes, logit/token consequences and H1–H7 status |
| `fidelity_tolerances.json` | Unchanged R3-A allclose, max_abs, relative L2, KL and argmax gates for each aligned step |
| `fidelity_audit.json`, `fidelity_decision.json`, `fidelity_limitations.json` | Independent physical/TCP/tensor replay, separate gates and explicit limitations |
| `r5_output_manifest.json` | Immutable complete output inventory and content hashes |

The following gates remain independent: `R5_I3F_CPU_TESTS_PASS`,
`R5_I3F_GPU_FORENSICS_COMPLETE`, `R5_I3F_CACHE_INTEGRITY_PASS`,
`R5_I3F_FIRST_DIVERGENCE_IDENTIFIED`, `R5_I3F_FROZEN_NUMERICAL_TOLERANCE_PASS`,
`R5_I3F_INDEPENDENT_AUDIT_PASS`, `R5_I3F_GENERATION_TOKEN_FIDELITY_PASS`.
Capture/analysis never asserts that the unit suite ran: its CPU-tests gate stays
false until an external validation record supplies an actual test result.
Integrity, tolerance, audit and token flags may pass in a fixture but are scoped
by `TEST_FIXTURE_ONLY` and the false GPU gate. Identifying the first observed
GPU divergence requires a real GPU logit difference with aligned preceding
layer observations; it does not set root-cause resolution or numerical parity.

## Local validation

The host has no Torch, NumPy, Transformers or PEFT. CPU unit tests cover exact
source/target and query/key alignment, numeric layer ordering, incompatible
layouts/shapes/dtypes, known numerical metrics, signed zero, nonfinite data,
injected perturbations, changed cache values with valid file hashes, invalid
positions/masks, unmatched Decode histories, duplicate evidence indices and
missing/corrupted/escaped files. Passive hook fixtures exercise shared method
code across distinct layers, unchanged outputs and hook cleanup.

The process tests use real localhost TCP and the existing scalar decoder. They
independently replay physical cache grants and completion, check all selected
layer KV reconstructions, compare final tokens/text/logit bytes with an unchanged
R5 production fixture, reject corruption/extra/missing evidence, test failed
opt-ins/output collisions and revoke previously calculated gates after a late
failure. Two independent analyses must produce byte-identical diagnostic reports.
They explicitly do not test real OPT, FP16 GEMMs or GPU attention instrumentation.

```bash
PYTHONPATH=src:tests python3 -m unittest tests.test_r5_fidelity \
  tests.test_r5_fidelity_processes -v
python3 scripts/114_run_r5_fidelity_forensics.py --phase dry-run \
  --output-dir results/r5/fidelity/my_unique_dry_run
python3 scripts/114_run_r5_fidelity_forensics.py --phase fixture \
  --test-fixture-only --generation-budget 2 --block-budget-bytes 268435456 \
  --output-dir results/r5/fidelity/my_unique_cpu_fixture
python3 scripts/115_analyze_r5_fidelity_forensics.py \
  --inputs results/r5/fidelity/my_unique_cpu_fixture \
  --output-dir results/r5/fidelity/my_unique_cpu_analysis
bash -n jobs/r5_opt_fidelity_forensics.sh
```

CPU TCP tests require localhost socket permission. The sandbox-denied initial
attempt and an exploratory source-change rejection were preserved under `/tmp`;
neither is counted as successful verification. Final retained validation lives
under `results/r5/fidelity/local_verification_20261031/`.

Final verification: **60 tests passed**, with no errors, failures or skips in
`cpu_tests_final.log` (231.374 seconds). This includes 28 new fidelity tests and
32 existing R5 contract/semantic/controller/evidence/schedule regressions.
Compilation, Slurm shell syntax and whitespace checks pass. All tracked existing
source files are unchanged from HEAD. `cpu_tests.log` retains the earlier analyzer
field-name failure; it is not counted as a passing run. The deliberately induced
late-failure test prints a FAILED launcher result while its unittest passes.

Retained final evidence: `cpu_final/fixture/`, `cpu_final/unchanged_reference/`,
`cpu_final/one/` and `cpu_final/two/`. The diagnostic fixture completes 20
requests / 40 production forwards, including four automatically discovered
physical HITs. The additional unchanged reference completes its own 20 requests.
All 20 generated outputs have identical tokens, text and serialized logits
between the observing path and the unchanged reference. The four diagnostic
pairs have 1,928 layer/logit comparison records, 1,536 KV comparison records,
eight passing frozen numerical checks, and 1,792 verified endpoint payloads.
All comparable fixture tensors are bitwise equal. The two independent analyses
produce ten byte-identical diagnostic reports. Every run/analysis has its own
immutable output manifest; `verification.json` records the external CPU-test
gate separately from capture decisions.

| Gate | Final local result |
|---|---|
| `R5_I3F_CPU_TESTS_PASS` | PASS, actual 60-test execution |
| `R5_I3F_GPU_FORENSICS_COMPLETE` | NOT_RUN |
| `R5_I3F_CACHE_INTEGRITY_PASS` | PASS for CPU fixture; GPU NOT_RUN |
| `R5_I3F_FIRST_DIVERGENCE_IDENTIFIED` | False; no fixture divergence, historical GPU cause unresolved |
| `R5_I3F_FROZEN_NUMERICAL_TOLERANCE_PASS` | PASS for CPU fixture; historical GPU failures unresolved |
| `R5_I3F_INDEPENDENT_AUDIT_PASS` | PASS for CPU fixture; GPU NOT_RUN |
| `R5_I3F_GENERATION_TOKEN_FIDELITY_PASS` | PASS for CPU fixture; GPU NOT_RUN |

Created files (no existing implementation file was modified):

| File | Purpose |
|---|---|
| `src/semcache/edgelora/r5_fidelity.py` | Context/tensor alignment, metrics and first observed divergence |
| `src/semcache/edgelora/r5_fidelity_capture.py` | Passive real-layer/locals, owner, TCP and CPU arithmetic snapshots |
| `src/semcache/edgelora/r5_fidelity_analysis.py` | Independent artifact-derived reports and audit |
| `scripts/114_run_r5_fidelity_forensics.py` | Bounded fixture, dry-run, opt-in GPU orchestration and failure gates |
| `scripts/115_analyze_r5_fidelity_forensics.py` | Read-only analysis entrypoint |
| `jobs/r5_opt_fidelity_forensics.sh` | Unsubmitted SERAPH diagnostic job |
| `tests/test_r5_fidelity.py` | 19 alignment, metrics, passive capture and corruption tests |
| `tests/test_r5_fidelity_processes.py` | Nine actual-process/launcher tests, including unchanged-reference regression |
| `docs/R5_I3F_FIDELITY_FORENSICS.md` | Design, observations, commands, limitations and handoff |

## Exact operator-controlled SERAPH commands

No GPU job or Slurm submission is executed by this task. Before submission,
commit every new execution file on `exp/raw-semcache-r5-integration`, including
the scripts, three new source modules and Slurm file. The launcher requires
those paths to be tracked, a matching HEAD and clean tracked source. It does
not require deleting unrelated untracked result artifacts.

SERAPH must have the established `semcache` environment, compatible frozen
Torch/Transformers/PEFT versions, offline OPT-2.7B/tokenizer and the verified
TinyBERT revision **34707a33cd59a94ecde241ac209bf35103691b43** under
`/data/khuss/huggingface`. No download is attempted. Reserve over 10 GiB free
disk. The job uses the established `batch_eebme_ugrad`, `moana-y4`, one RTX 3090,
four CPUs, 64 GiB and three hours. A fresh offline R3-A preparation inside the
allocation verifies the actual GPU UUID and unchanged seed42 rank8 `user_a`
matrices. Preparation is separate from the forensic A/B comparisons and does
not establish Native PEFT parity.

```bash
cd /data/khuss/repos/GP_semcache
mkdir -p logs
R5_ENABLE_GPU=1 R5F_ENABLE_GPU=1 \
  R5F_SOURCE_COMMIT=$(git rev-parse HEAD) \
  sbatch jobs/r5_opt_fidelity_forensics.sh
```

The Slurm script executes these commands after allocation/preflight, using a
unique `r5f_root=results/r5/fidelity/opt_${SLURM_JOB_ID}_<UTC timestamp>`:

```bash
python -u scripts/95_run_r3a_distributed_no_reuse.py --phase prepare \
  --model facebook/opt-2.7b --dtype float16 --prompt-lengths 32 \
  --device cuda:0 --adapter user_a --seed 42 --output-dir "$r5f_root/base_prepare"
python -u scripts/114_run_r5_fidelity_forensics.py --phase run \
  --enable-gpu-experiment --source-commit "$R5F_SOURCE_COMMIT" \
  --base-prepared "$r5f_root/base_prepare" \
  --semantic-revision 34707a33cd59a94ecde241ac209bf35103691b43 \
  --generation-budget 8 --block-budget-bytes 4294967296 \
  --output-dir "$r5f_root/run"
python scripts/115_analyze_r5_fidelity_forensics.py \
  --inputs "$r5f_root/run" --output-dir "$r5f_root/analysis"
```

When transferring evidence for independent analysis, include the small prepared
artifact directory and retain/mount the recorded `base_prepared` path, as with
I3P. The auditor reads identities/matrices/reference manifests, never loads OPT
or launches a GPU. Unsupported installed attention/cache structure must be
reviewed using preserved failure evidence; do not relax contracts to force a run.

## Handoff conditions

R5-I4A is not started. Review the real GPU per-request reports, independent
cache/TCP/tensor audit, reproduced or unreproduced historical errors, all frozen
tolerance failures and token outcomes before considering expansion. If only
compatible-path numerical differences are observed, document their magnitude
and limitations, retain no-reuse operation, and design explicit additional
controls for unresolved kernel/cache/precision hypotheses. Do not change frozen
tolerances or claim parity from root-cause identification. Multiple adapters,
Native PEFT claims and correctness-qualified speedup each require independent
work and evidence.
