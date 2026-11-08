# Sparse QKV numerical forensics and prefill profiling

CPU implementation only. GPU_NOT_RUN. The supplied historical measurements
(189 reused projection rows at layer0, later layers0, numerical FAIL and C-B
prefill +5160.996ms) are a verified operator handoff, not locally reproduced.
The raw pilot and Git object a6fae7d3137935583de38e563eb0d46a396f7eba are absent
locally. Do not infer a CUDA algorithm or a numerical fix from those observations.

## D1: historical CPU-only analysis

The152 reader is independent of the execution loader's current-HEAD check. It
verifies sealed plan/run/child/audit/CPU-approval inventories, original source
SHA256s against recorded Git blobs, prepared dataset/adapters, runtime physical
identity and contract fingerprints, task order and linked independent audit
exports. Missing Git objects or required tensors reject; there is no partial
provenance bypass. No historical file is modified or resealed. Current analyzer
commit, dirty status and file hashes are recorded separately. This is a new
bounded forensic replay of an exported audit, not a fresh original-source GPU audit.

For every layer/role it reads full saved inputs/outputs and compares A/B/C,
standalone C-source versus A/B target-prefix base/delta, actual cached rows,
and actual fresh-row native base/delta/total. Source bytes, received TCP delta
payloads, addition/casting and scatter are independently replayed using the
existing value-audit utilities. Inputs are aligned by recorded absolute positions.
Full-vocabulary per-step frozen logit gates are recalculated and matched to the
old audit. All original atol/rtol/max_abs/L2/KL/argmax checks remain unchanged.

Layer eligibility is reconstructed from saved selected source windows and actual
per-layer input bytes, then checked against consumed positions. Original
per-layer rejection events, unavailable tensor strides, admission reasons for
absent entries and historical hashing durations are NOT_OBSERVABLE. Raw matching
metadata is not a physical grant. Source integrity does not establish equivalence
to dense target projections. Layer1 input mismatch can explain rejection; it does
not isolate the intervening attention/residual/normalization arithmetic. The
report also compares source versus A/B input windows: a mismatch already present
against the dense baseline must not be attributed solely to reused layer0 output.

Outputs are small `subsequence_diagnostics.json`, `subsequence_diagnosis.md` and
`i5_output_manifest.json`. JSON includes comparisons, tensor metadata, per-layer
eligibility, per-step logits, costs, availability, provenance and hypothesis
statuses. No raw tensors are copied. Tensors are bounded to128MiB one at a time;
inventories stream SHA256, JSON retains the existing16MiB limit and input totals
are bounded to128GiB. No Torch/CUDA/model/tokenizer forward is used by D1.

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
git cat-file -e a6fae7d3137935583de38e563eb0d46a396f7eba^{commit}
read -r -e -p 'Original sealed151 experiment: ' HIST_EXPERIMENT
read -r -e -p 'Original complete A/B/C run: ' HIST_RUN
read -r -e -p 'Original independent audit: ' HIST_AUDIT
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DIAG_ROOT="$PWD/results/r5/i5r/parallel/subsequence_diagnostics"
python scripts/152_r5_subsequence_diagnostics.py \
 --experiment "$HIST_EXPERIMENT" --run "$HIST_RUN" --audit "$HIST_AUDIT" \
 --output-dir "$DIAG_ROOT/historical_$STAMP"
```

## D2/D3: opt-in instrumentation, no integrity optimization

The new Profiler is enabled only by a new sealed diagnostic plan. Normal plans
retain default numerical behavior, cache rules, protocol V2 and thresholds.
Dormant wrappers add Python checks; no normal-path CUDA events or additional
synchronization are inserted. Source, model and request identities are unchanged.
The profiler records selected/filtered windows, source and target input proofs,
accepted windows and exact rejection reasons; integrity failure remains fatal.
Failures attempt bounded profile export without masking the primary exception.

Wall scopes include selection, `valid`, full-layer entry proofs, individual tensor
proof serialization/host transfer and SHA256, input proof/gather, native dense or
sparse Linear, sparse TCP send/receive waiting, delta copy/add/cast, scatter,
clones, evidence writes, genuine impact conversion/reduction, completion and
materialization. CPU LoRA computation comes from actual UD receipts; owner
completion/CHU/insertion/sealing costs remain available. Native operand shapes,
strides, contiguity, dtype and storage offsets are recorded without changing them.

Tensor-proof serialization counters measure actual calls/bytes, not hypothetical
network savings. Protocol full-input serialization is counted separately.
`Evidence.save` includes its own copies/hashes; its internals are measured as an
inclusive scope, not falsely added to the tensor-proof byte counter. Wall time
for `.cpu()` includes synchronization/transfer/serialization; it cannot separate
those mechanisms without a lower-level profiler. No guessed kernel times exist.

Per request bounds:500000 calls,8192 aggregated categories/CUDA event pairs,
4096 entry identities and4096 eligibility/layout details. Exhaustion fails closed.
An optional `--cuda-events` plan records native GPU stream spans and synchronizes
once at profile export. These spans can include stream scheduling delays; they
are not pure kernel time. Inclusive scopes and CUDA spans overlap. Exclusive wall
scopes partition only observed work, leaving unobserved model/runtime work.
Do not sum overlapping scopes or compare instrumented latency to the old run as
if measurement scopes matched. Fixed A/B/C order, warm state and variability
remain limitations. No removal or memoization of `valid()` has been implemented.

Code inspection shows that `valid()` hashes all stored layers, even when the
current layer subsequently rejects an input mismatch. For21 selected windows,
32 layers and63 physically consumed Q/K/V spans, the original path would execute
735 validations and serialize1083801600 tensor bytes in those full-entry proofs.
This is a **static reconstruction**, conditional on those verified counts; it is
not measured transfer traffic or an explanation of the entire5.16s difference.
The analyzer recalculates it from evidence and new profiling measures the calls.

## D4: one manually gated unchanged-arithmetic A/B/C diagnostic

After reviewing D1, commit the reviewed implementation manually. Preparation
requires that new clean reviewed commit; it must differ from the historical SHA.
The new sealed plan copies the exact historical tasks rather than regenerating
their selection. Historical and current identities remain separate. CLI/job/test
files omitted by the old I5 inventory glob receive additional Git-verified hashes.
No source, token, generation budget, mode or policy parameter is adjusted.

```bash
export SSG_SOURCE_COMMIT="$(git rev-parse HEAD)"
export SSG_EXPERIMENT="$PWD/results/r5/i5r/parallel/subseq_profile_plan_$STAMP"
export SSG_CPU_AUDIT="${SSG_EXPERIMENT}_cpu"
export SSG_RUN_ROOT="${SSG_EXPERIMENT}_runs"
python scripts/152_r5_subsequence_diagnostics.py \
 --experiment "$HIST_EXPERIMENT" --run "$HIST_RUN" --audit "$HIST_AUDIT" \
 --prepare-revalidation --source-commit "$SSG_SOURCE_COMMIT" --cuda-events \
 --output-dir "$SSG_EXPERIMENT"
PYTHONPATH=src:tests python3 -m unittest \
 test_r5_subsequence_diagnostics test_r5_subsequence_generation \
 test_r5_sparse_physical test_r5_subsequence_audit test_r5_subsequence_pilot -q
python scripts/151_r5_subsequence_generation.py cpu-validate \
 --experiment "$SSG_EXPERIMENT" --output-dir "$SSG_CPU_AUDIT"
```

Omit `--cuda-events` at preparation for wall-only measurement; no runtime override
exists. Native CPU OPT tests must have zero failures AND zero skips in the sealed
approval. Local skips do not authorize GPU execution. The approval includes new
diagnostic fixtures and the existing native attention/generation/TCP tests.

Reuse the existing bounded151 job: one compatible RTX3090,4 CPUs,64GiB,3h30m;
sequential independent model/UD/cache sessions for A/B/C,1h per worker, no retries.
Disk reserve32GiB, ES evidence8GiB/condition, UD2GiB, host available32GiB, GPU peak
<22GiB remain unchanged. No counterfactual or performance-optimization path is
enabled. There is no automatic job submission or dataset/model download.

Only after zero-skip CPU approval and operator review:

```bash
read -r -p 'Permitted node (moana-y3 or moana-y4): ' SSG_NODE
read -r -e -p 'Existing Conda root: ' I5_CONDA_ROOT
export SSG_NODE I5_CONDA_ROOT
export R5_I5_ENABLE_GPU=1 R5_ENABLE_GPU=1
mkdir -p logs "$SSG_RUN_ROOT"
unset CUDA_VISIBLE_DEVICES
sbatch --nodelist="$SSG_NODE" --export=ALL jobs/r5_subsequence_generation.sh
```

Post-run CPU analysis and independent recomputation:

```bash
export CUDA_VISIBLE_DEVICES=''
read -r -e -p 'New completed run_JOBID directory: ' NEW_RUN
python scripts/151_r5_subsequence_generation.py analyze \
 --experiment "$SSG_EXPERIMENT" --run "$NEW_RUN" --output-dir "${NEW_RUN}_audit"
python scripts/151_r5_subsequence_generation.py verify \
 --experiment "$SSG_EXPERIMENT" --run "$NEW_RUN" --audit "${NEW_RUN}_audit" \
 --output-dir "${NEW_RUN}_verification"
python scripts/152_r5_subsequence_diagnostics.py \
 --experiment "$SSG_EXPERIMENT" --run "$NEW_RUN" --audit "${NEW_RUN}_audit" \
 --expected-source-commit "$SSG_SOURCE_COMMIT" --output-dir "$DIAG_ROOT/profile_review_$STAMP"
```

The151 audit checks new receipt counts/input proofs against actual consumption as
well as its unchanged tensor/payload/scatter/logit/quality audit. Profiled timing,
physical reuse, numerical fidelity and evidence validity remain separate results.
No-hit fallback is never a successful physical-reuse verdict. Approximate mode is
not enabled. A0 target-only economic break-even remains NOT_ESTABLISHED.

## Smallest next intervention

Do not weaken hashing based on the static estimate. If measured proof work
dominates, review a separate integrity-preserving validation-lifecycle design
(e.g. request-scoped verification tied to immutable storage and explicit mutation
invalidation), with corruption tests before benchmarking. No such optimization
is part of this task. If layer0 base differences remain, the smallest arithmetic
control is the saved byte-identical layer0 input through actual dense and gathered
Linear shapes with fixed weights, repeats, and separate LoRA/add/scatter checks.
Label it DIAGNOSTIC_ONLY and do not claim saved computation for dense-reference
work. Existing S1/C1 shape diagnostics are prior evidence, not proof for this
specific sparse operand/layout. Attention/residual/normalization remains unresolved
if only later-layer input captures are available. ROOT_CAUSE stays UNRESOLVED
until controlled evidence identifies a mechanism; no numerical fix is asserted.

## Local CPU verification

Final suites:231 tests,221 PASS,0 FAIL,10 SKIP. Diagnostics:18 PASS/1 SKIP;
sparse-generation/physical/audit/CLI regressions:21 PASS/4 SKIP; relevant baseline
regressions:167 PASS/5 SKIP; legacy I5/I5R1 Exact Prefix ES/UD process tests:
15 PASS/0 SKIP. Native Torch/Transformers/NumPy and optional reference-metric
dependencies are unavailable locally. Native OPT and CUDA-event behavior remain
unverified here. Python syntax, CLI dispatch, shell syntax and whitespace checks
passed. An earlier process-test run during source edits had two setup errors;
the frozen-source rerun passed15/15 without changing legacy validation rules.
Real historical diagnosis is NOT_RUN locally: raw evidence and the recorded Git
object are unavailable. No GPU job, remote access, download, commit or push ran.
