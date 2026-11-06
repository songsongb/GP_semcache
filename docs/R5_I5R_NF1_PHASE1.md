# I5R-NF1 Phase 1 — immutable numerical forensics

CPU-only implementation. The specified SERAPH run is **not available locally**.
Three HITs, negative MISS, token/text PASS, numerical FAIL and D(3)=-4817.134836 ms
are researcher-verified handoff observations, not newly reproduced workstation
findings. The analyzer must independently check the sealed records and numerical
values before producing new scientific observations. No GPU experiment, model
forward, semantic encoding, checkpoint download or old-source audit override.

## Prior research review (before designing another GPU experiment)

| Stage | Test / established handoff observation | What it does not establish |
|---|---|---|
| R2B, BS1 | At earlier 64/256 lengths, matching embedding/LN/projection inputs and CPU LoRA intermediates accompanied unequal native Base Linear outputs; isolated native FP16/FP32 shape/repeat controls exist | No specific cuBLAS algorithm or proof for the present 512/1280/768 case |
| C1 | Isolated native fixed-shape Linear controls with zero/nonzero suffixes and unchanged weights/reference | Not whole-model correctness or performance |
| C2 | Source-only and source+suffix native padding aligned Layer0 prefix Base/Raw QKV; module attention output and final logits still differed | Linear alignment did not restore strict numerical fidelity |
| C3/B0 | Full operands were recaptured; native eager replay reproduced historical module output. Earlier scaled QK scores were the first differing internal boundary | First recorded score difference is not a CUDA kernel mechanism |
| C4 | Source code supports query/key shape, layout and FP32 scaled-operand isolation | No C4 GPU export is supplied here; implementation alone is not an experimental conclusion |
| 3B/BE | Actual unoptimized Source512/Target1280/Suffix768 reuse, strict cache integrity, repeated-target control and frozen logit failure | Does not separate Base/Delta/LN inputs or native attention internals |

Reviewed implementation: `r3b_runtime.physical_path()` computes CPU correction,
then the original native Linear, then `(base + delta).to(base.dtype)`. Its passive
clones capture **total** QKV before later attention Q scaling. User-device
`cpu_deltas()` uses A Linear, B Linear and scaling; normal BE records payload
hashes/precision, not these numeric intermediates. `QualityDriver.c_forward()`
executes source512/full1280 or suffix768, observes real positions, stores total
QKV, attention/FFN/layer outputs and first-forward attention-layout KV. The
`r4b2_kv.attention_kv()` transformation is view/transpose/contiguous/clone, with no
rotary or K/V arithmetic; construction/consumption proof is separately recorded.
Normal `generate()` admits a completed source; a HIT uses actual reconstructed
past and suffix-only inputs. No production path is edited by NF1.

These earlier observations are source/report review plus operator handoff unless
actual exports are explicitly supplied; NF1 does not silently fresh-audit them.
`nf1_prior_research.json` preserves this distinction. Existing I5 independent
analysis still requires the original execution source checkout; NF1 verifies
archived Git blobs and historical audit exports instead of weakening that guard.

## Inputs and exact identity

The four input directories must be explicit CLI arguments:

- `plan_N3_20261106T064627Z`
- `attempt_N3_20261106T064627Z/block_0_153343`
- its `block_0_153343_be_audit` sibling
- its `block_0_153343_be_audit_verification` sibling

Historical source defaults to `f7157c9b8b97fdce56f87dcbc3566cb677404bb8`.
The analyzer verifies SHA inventories for the complete input tree, associated
child plan/audit inventories, plan/block/experiment/request hashes, COMPLETE and
real-GPU receipts, Adapter_A fingerprints, model/contract/method hashes and
cross-condition GPU identity. Historical source files are compared to recorded
Git blobs. Reused serialization/context/numerical/transform helper files must
also match their recorded historical SHA; changing current code is not a reason
to bypass historical identities. Current NF1 source identity is recorded separately.

The immutable dataset path comes from its sealed manifest, not a guessed source
file. Dataset files are streamed; only the three selected original records are
retained. Version, source, split/revision, inventory, normalized originals,
references, sample IDs, dataset scope and duplicate IDs are checked. Full original
questions/references are never copied into a new artificial workload.

Source P000000S must contain exactly the first512 IDs of P000000T and P000001T.
Both targets and P000000T_BE03 have1280 IDs. Repeat content must equal original
Target1 except request ID and sequence number. P000002T must not share the source's
complete512 prefix. Actual original masks, position-module observations, recorded
cache grants/completion/admission/residency and authorization are checked. No
array or source identity is fabricated. A0 receipts remain a separate economic
scope and do not enter the numerical reference. D(3) is cross-checked against raw
request timing and the archived economic result, not accepted from this document.

## Reusable historical tensors and precise limits

| Evidence | Positions and layout | Use |
|---|---|---|
| Strict source total Q/K/V | [1,512,2560], all32 layers | Source prefix versus full targets; raw storage identity |
| Strict full target Q/K/V | [1,1280,2560], all32 layers | Prefix0:512; suffix512:1280; repeat control |
| Strict HIT Q/K/V | [1,768,2560], all32 layers | Suffix at absolute512:1280, not prefix0:768 |
| Actual source/target KV | [1,32,512 or1280,80] | Attention-layout comparisons, source versus consumed prefix |
| Strict cached Raw QKV | [1,512,2560] | Storage proof; cached Q is stored but not consumed |
| `input_kv` | Shape/dtype/device/SHA, normally no values | Verify actual input-past receipt against source KV, not invent bytes |
| Attention/FFN/layer outputs | All executed rows on strict requests | Prefix and suffix boundary localization, not internal Softmax/PV |
| Negative scalable activations | Usually full-shape hashes only | Hash equality/inequality is observable; error magnitudes are unavailable |
| Actual logits | [1,generated_steps,50272], float32 export | Full vocabulary at last prefill position and each decode step; not logits for every prefill token |

Live strides/storage offsets, normalized hidden projection inputs, native Base,
CPU LoRA arrays, actual internal4D additive masks, attention scores, Softmax and
PV are **not captured by the normal BE schema**. Missing numerical values are
explicitly NOT_OBSERVABLE_FROM_AVAILABLE_EVIDENCE. A named missing/corrupt file
or missing strict capture rejects; a legitimately hash-only scalable reference
is coverage data, never a fabricated quantitative PASS. Original negative control
logits remain required and are checked under frozen thresholds.

Exports contain canonical logical values. This is not evidence that their live
GPU layout was contiguous. Raw KV is explicitly transformed into head-major
layout before comparison to attention KV; axes and token positions are checked.
No Source-last-token versus unrelated Target-last-token comparison is used.

## Comparisons and interpretation

All aligned elements are compared, not sampled:

- Source C total QKV and actual KV versus A target-prefix values, per layer/role.
- Actual C consumed prefix KV versus A target-prefix KV; source/stored/consumed
  integrity separately, including the repeat.
- A full-target suffix versus C HIT suffix total QKV and module boundaries.
- Source A versus Source C as a no-cache execution control. If identical, a
  source/full-prefix discrepancy also exists in no-cache A; otherwise no such
  equivalence is claimed.
- Full Target1 versus full Repeat1 prefix/suffix, and HIT Target1 versus HIT
  Repeat1 suffix, to test reproduction of the differing trajectories.
- Negative full-versus-C QKV/module values when available, otherwise shape-bound
  full SHA comparisons without invented error metrics.
- Recorded full-vocabulary logits for A/B, B/C, A/C, source and repeat controls.
  Per-step hashes, greedy argmax, token/text equality, frozen atol/rtol/max_abs,
  relative L2 and last-token KL verdicts are recalculated and cross-checked.

Bitwise equality, bitwise/numeric affected-element counts, percentages, max/mean
absolute error, relative L2, elementwise frozen violations and finite FP16 ULP
statistics are retained. Signed-zero byte differences are not confused with
nonzero ULP distance. Activation checks do not substitute for full-logit gates.
All numerical failures remain failures.

`PREFIX_PATH_INSPECTION` and `SUFFIX_PATH_INSPECTION` report first observed
bitwise/tolerance differences across the recorded boundary classes, including
module attention before a later layer's QKV. Q/K/V peers are co-listed; their
listing order is not asserted to be arithmetic chronology. H1 is a recorded-value
hypothesis, not proof of its complete causal logit contribution. H2's projection
shape mechanism stays UNRESOLVED without matching input/Base/Delta operands.
H3 and the kernel-level mechanism remain UNRESOLVED. Repeats are one execution
control, not new independent Qasper samples. No numerical restoration or
correctness-qualified speedup is claimed.

## CPU memory, disk and failure bounds

Standard library only: no Torch/NumPy/Transformers import, no CUDA initialization.
Inventories stream SHA256 in bounded chunks. Read-only `mmap` selects only aligned
regions; at most two32-MiB tensor files are mapped and their bounded selected
bytes/metrics are processed at once. FP16 ULP counts stream rather than allocating
millions of Python integers. Identical finite bytes use an exact zero-error fast
path without changing thresholds. No complete raw archive is copied.

Bounds: 64 GiB per input evidence directory,100000 files,32 MiB per tensor,
16 MiB JSONL record,10000 JSONL records per stream,32 MiB total retained prefill
metadata,4096 comparison records. Only first-step metadata and small per-step
logit receipts are retained; decode tensor arrays are not accumulated. Default
CPU comparison deadline7200 seconds (explicit allowed60..14400); exhaustion
rejects, not a partial PASS. Metadata maps for30k files still consume CPU RAM;
this is not a zero-memory verifier.

Output JSON is capped at24 MiB before writing; CSV/Markdown add only compact
metadata. No tensor copies or original dataset archive. Reusing an output or
writing under a sealed ancestor rejects. Cache source/storage/consumption
mismatches yield a sealed failed diagnostic and nonzero exit. Missing/corrupt
required evidence rejects before successful output. Hash-only optional coverage
is reported. Old numerical FAIL and token/text PASS remain separate.

## Exact SERAPH CPU command

After manual review/commit/push/pull, preserve all historical Git objects and
run in the existing offline semcache environment (no GPU allocation):

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NF_BASE="$PWD/results/r5/i5r/controlled_hit/i5r3b/be"
NF_RUN="$NF_BASE/attempt_N3_20261106T064627Z/block_0_153343"
NF_OUT="$PWD/results/r5/i5r/controlled_hit/i5r3b/nf1/phase1_$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/147_r5_i5r_nf1.py \
 --experiment "$NF_BASE/plan_N3_20261106T064627Z" \
 --run "$NF_RUN" \
 --audit "${NF_RUN}_be_audit" \
 --verification "${NF_RUN}_be_audit_verification" \
 --expected-commit f7157c9b8b97fdce56f87dcbc3566cb677404bb8 \
 --deadline-seconds 7200 --output-dir "$NF_OUT"
```

Initial inventory verification streams about30 GiB; allow CPU time for all-layer
reductions. Do not submit a GPU job while waiting. Inspect exit status and
`nf1_decision.json`, then `nf1_report.md` and comparison details. No missing-file
fallback or source-verification override is provided.

Outputs (all separately SHA-sealed by `i5_output_manifest.json`):
`nf1_tensor_comparisons.json/.csv`, `nf1_layer_summary.json`,
`nf1_numerical_fidelity.json`, `nf1_evidence_availability.json`,
`nf1_source_provenance.json`, `nf1_hypotheses.json`, `nf1_prior_research.json`,
`nf1_phase2_proposal.json`, `nf1_decision.json`, `nf1_report.md`.

## Phase 2 proposal — NOT EXECUTED

First inspect NF1. Existing full QKV/KV must be reused wherever adequate; do not
repeat a large BE experiment merely to obtain values already recorded.

1. **F/H acceptance:** replay unchanged native Layer0 attention with recorded
   Full1280 and real-HIT suffix768/prefix512 operands. Missing actual additive
   masks/live layouts require a small, opt-in genuine F/H prefill capture; no
   silent synthetic-mask substitution. Historical module-output bitwise
   reproduction is mandatory before attributing historical divergence.
2. **O intervention:** fix H suffix QKV and substitute Full-target prefix KV in
   a separate direct native-attention replay. Use request-owned past via observed
   `r4b2_kv.construct_past` API. No global cache, admission, fake controller grant
   or modified production ForwardOrder. Oracle values are diagnostic-only and
   invalid for legitimate-reuse/performance claims. Existing C3 square replay
   needs a separately tested q768 x k1280/past-aware diagnostic extension.
3. **P intervention:** only if NF1 cannot localize pre-projection differences,
   passively capture Layer0 embedding/LN/projection inputs, actual native Base,
   real ES/UD hidden/delta representations, CPU A/B/scaling and final Total QKV.
   `physical_path.miss()`/`cpu_deltas()` hooks observe original operations once,
   changing no shape/math. Compare matching positions, then isolate actual
   native Linear with identical suffix rows in1280/768 shapes and same-shape
   repeats. CPU LoRA remains on the real UD; no artificial corrections.
4. Capture native scaling/QK/mask/Softmax/PV/out_proj only at the first necessary
   attention boundary. Mask future keys causally and align absolute query/key
   positions. No complete all-layer attention dump. Backend/TF32/reduced-precision
   and deterministic settings must be recorded, not guessed from FP16 dtype.

F/H retain exact token IDs, Adapter_A matrices, revision, positions, original
source-one/target-up-to-eight greedy policy and localhost TCP. Prefer recorded
QKV replay with one model load; if authentic missing operands require new F/H
runs, scope them to the existing one pair, Layer0 captures, not all A/B/C repeats.
Two warmups/six balanced native replay repeats bound nondeterminism. One RTX3090,
4 CPU cores,64 GiB RAM,15-minute runtime/30-minute allocation and512 MiB capture
budget (including reserve); additional GPU work needs explicit user approval.
Instrumentation latency is never serving-performance evidence.

Falsifiers/limits:

- Equal source/full-prefix KV across all inspected positions falsifies H1 as a
  recorded-value mismatch. If O still differs from F with other operands fixed,
  source-prefix differences alone are insufficient.
- Unequal hidden inputs do not implicate Linear. Equal inputs plus unequal Base
  bound projection execution as a candidate; changed LoRA/transfer values require
  following that branch. Equal Base and Delta with unequal Total isolate addition/
  conversion, subject to real layout and dtype evidence.
- Matching total prefix/suffix QKV but different native attention outputs points
  to attention/layout/masking/execution, not a particular cuBLAS kernel. O shares
  H query shape; a remaining F/O difference can still involve query-length/Softmax/
  PV ordering and must be localized internally.
- Failed F/H historical replay means HISTORICAL_REPLAY_NOT_ESTABLISHED; keep
  new-run observations separate. No causal result is forced.

Independent audit: fresh source/config/tensor seals, exact operand receipts,
matched semantic regions, real cache-consumption proof for H, explicit oracle
labels for O, repeated native-operation checks and unchanged full-vocabulary
frozen logit checks. No GPU code or Slurm script is added in Phase 1.

If Layer0 replay leaves the full-model contribution unresolved, the separately
reviewed O follow-up would feed request-owned Full-target prefix KV for all32
layers to a native suffix768 forward. It must use an explicitly versioned
diagnostic-only UD/session contract with real adapter authorization and oracle
provenance; it must not fake a production `REGISTER_PREFIX` grant or bypass
`ForwardOrder`. CPU delta arithmetic stays unchanged. Compare F/H/O vocabulary
logits and actual input-past fingerprints. All-layer prefix KV is160 MiB and may
be referenced from verified captures instead of copied. This conditional design
is not implemented, executed, a legitimate cache HIT, or a performance result.

Local commands:

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r_nf1 -q
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r_nf1 \
 test_r5_i5r2 test_r5_i5r2b test_r5_i5r2bs1 \
 test_r5_i5r2c3 test_r5_i5r2c3b0 test_r5_i5r3bbe -q
```

Fixtures exercise actual bounded byte/hash/layout comparisons and stored-logit
reproduction, not native FP16 GPU behavior. A subprocess test forbids Torch and
Transformers imports. Source/hash/path/tensor/context mismatches reject, and a
small sealed-output fixture preserves input inventories and numerical FAIL.

## Local delivery validation

Added files: `r5_i5r_nf1.py`, `r5_i5r_nf1_tensors.py`,
`scripts/147_r5_i5r_nf1.py`, `tests/test_r5_i5r_nf1.py`, and this document.
Existing tracked production files and historical evidence are unchanged.
The new files await operator review and manual commit/push.

Focused tests:21 PASS /0 FAIL /0 SKIP. Combined focused and relevant R2A/R2B/
BS1/C3/B0/BE regressions:188 PASS /0 FAIL /7 SKIP (195 tests total).
The seven native Torch/NumPy/Transformers tests could not run because those
dependencies are unavailable locally; they are not GPU-verification successes.
CLI help, four Python AST/syntax checks and whitespace checks passed.

Real N=3 forensic execution: **NOT_RUN**; the specified raw run directory is
absent locally. Historical numerical FAIL/token PASS and economic observations
remain verified researcher handoff, not newly reproduced local results. Only
the SERAPH command above can establish the new evidence findings and whether
the original capture coverage satisfies this fail-closed reader.
