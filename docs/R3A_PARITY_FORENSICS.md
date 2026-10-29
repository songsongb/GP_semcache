# R3-A isolated native/distributed parity forensics

This is **R3A_PARITY_DIAGNOSTIC**, with
`PRIMARY_PERFORMANCE_ELIGIBLE=false`. It neither changes production R3-A nor
admits the failed pilot into latency. The frozen thresholds, original prepared
adapter/reference, R2 experiments and original failed run remain untouched.
There is no cache reuse, codec, quantization, remote networking or E2E claim.

## Current evidence and unresolved cause

The user reported this actual SERAPH run:

```text
/data/khuss/repos/GP_semcache/results/r3/a/physical_no_reuse_151665_20261029T140923Z
```

The absolute path was checked and is **not accessible in this Codex
environment**. No local fixture has been substituted for its experimental
evidence. The following observations are user-reported, not independently
revalidated here:

| Observation | Reported result |
|---|---|
| prepare / pilot | COMPLETE / FAILED; only warmup_000000 |
| ES / UD PID | 277317 / 277306 |
| Actual forward | GPU inference; 32 layers, 96 QKV outputs, 32 HIDDEN/DELTAS exchanges |
| Max absolute logit error | 0.0546875, above 0.01 |
| Relative L2 | 0.0019029290937084956, above 0.001 |
| Elementwise tolerance violations | 828548 |
| Last-token KL | 1.1583049624620668e-06, passes its individual threshold |
| Last-token argmax | agrees |
| Native versus ES hidden bytes | layer0 MATCH; first mismatch layer1 |
| UD close error | truncated frame reported after ES parity failure |

The strict all-token gate remains **FAIL** even with low last-token KL and
matched argmax. Layer1's hidden SHA mismatch supplies no error magnitude and
does not prove CPU/GPU rounding. Together with a matching layer0 input it bounds
the divergence between layer0's input and layer1's input, including projection,
casting, attention and the rest of the decoder layer. The exact numerical cause
is **UNRESOLVED**. GPU numerical findings in this environment are **NOT_RUN**;
Torch, Transformers, PEFT and NumPy are unavailable here.

## New files and preservation of production provenance

- `scripts/96_diagnose_r3a_parity.py`: read-only audit, no-model dry-run and
  explicitly enabled GPU-worker orchestration.
- `src/semcache/edgelora/parity_forensics_audit.py`: failed-artifact consistency,
  clock/order reasoning, source verification and objective conclusion enums.
- `src/semcache/edgelora/parity_forensics_probe.py`: passive native PEFT probes
  and a separate diagnostic ES execution path.
- `tests/test_parity_forensics.py`: clearly synthetic CPU artifact tests and a
  separately opt-in real-GPU test.

Neither module matches the production `r3a_*.py` source-identity glob. Script 96
is outside the explicit script-95 identity list. All source-identity-tracked
originals are unchanged. The tool compares the recorded source-map SHA and
every original file hash, then calls the **unmodified** `validate_prepared()`.
A different HEAD alone is not corruption; changed captured source bytes are.
If source identity fails, restore the recorded checkout or request a fresh
prepare for the new code. Never rewrite the old manifest's hashes or bypass the
normal gate. Every diagnostic records its own new source hashes separately.

The normal runner 95, CPU UD placement and fail-closed parity behavior are not
patched or called as a benchmark. Diagnostic outputs have a different schema
and cannot serve as normal pilot or latency evidence.

## Pass 1: original failed-pilot audit, CPU only

The audit targets exactly one failing first warmup. It verifies:

- Prepared manifest, complete original code identity, artifact hashes, all
  original A/B matrix offsets/shapes/dtypes/checksums/nonzero content, input
  identity and immutable thresholds.
- FAILED pilot identity and artifact SHA, ES/UD results/environment copies,
  CSV/raw/parity copies, independent recorded PIDs, placement and actual layer
  counts. The error must be the strict parity failure, not a successful pilot.
- All layer HIDDEN/DELTAS frame versions, identity, sequence, dtype/shape,
  payload length, request/layer, application-wire bytes and matching ES/UD frame
  digests. Recorded per-layer and directional row totals must agree.
- All original per-layer/role hidden reference hashes, the first differing
  layer/role, and recorded process cleanup/exit status.
- The next UD receive's incomplete/zero-byte EOF, its position after all
  completed data, the original ES parity error and enclosing monotonic bounds.

The original code logs the request end and a later post-close marker, but **no
exact parity-raise timestamp**. A UD receive can start while waiting for the
next frame, before ES finishes checking parity. Its EOF can be observed before
ES writes its post-close timestamp. Consequently the tool uses verified
control flow (PARITY_FAIL followed by `finally: close()`), complete data frames
and ordering to label a supported shutdown symptom; it does not invent an
exception timestamp. Partial data, earlier errors or insufficient ordering
remain UNRESOLVED. The close symptom is not an explanation for already-received
tensor numerics.

Original payload bytes and ES combined-QKV outputs were not retained. Agreement
of the two endpoints' recorded SHA values is useful integrity evidence, but is
not independent rehashing of stored raw payloads and cannot yield QKV error
magnitudes or prove branch/assembly semantics. The offline audit never declares
H3 proven from that evidence.

## Pass 2: explicitly enabled GPU controls

The parent process loads no model. It starts and fully reaps GPU workers
sequentially, so RTX3090 memory never holds two full OPT copies concurrently.
Each worker enforces the existing immutable prepare gate, matching recorded
runtime versions and original GPU UUID, pinned model/tokenizer snapshots,
complete base fingerprint, input tokens, seed, eager attention,
use_cache=False/output_attentions=True and deterministic/TF32 settings. Original
UD effective thread count is used for CPU comparisons; actual affinity and
thread settings are recorded. Unsupported or missing identity fails visibly.

### A. Reproducibility controls

Two independent native workers each load the original bare OPT and wrap it in
vanilla user_a PEFT. Every constructor-initialized A/B weight is overwritten
with the **exact original prepared binary**, then rehashed before inference.
No fixture generator or newly initialized adapter is used for a forward.
The complete normalized base state is checked before/after PEFT attachment
and after observation; all GPU A/B checksums are checked again at the end.

Each worker runs the native endpoint twice, observing per-role hidden and QKV
hashes with passive hooks, comparing full logits against the original frozen
reference and repeat logits with the same full strict metrics. Cross-process
logit SHA equality is reported separately. All captures are in fresh diagnostic
output; the old reference is never replaced.

If native reproduction differs, the report emits REFERENCE_NONREPRODUCIBLE
with the triggering control and distinguishes bitwise difference from a strict
gate failure. The additional ES replay is deferred until reproducibility is
fixed. Same-hidden component observations, if collected, remain conditional
diagnostics rather than proof of the old run's root cause.

### B. Exact same-hidden arithmetic

The first native worker makes another forward with passive hooks. Hooks return
None and never replace native outputs. Only the current projection/layer's
tensors are retained; hooks and buffers are cleared in `finally`. Selected
layers default to all 32; a smaller selection must include layer0 and all its
Q/K/V roles.

For the identical actual native FP16 hidden and original A/B, records include:

1. Native PEFT output, actual base Linear output (including bias), actual native
   A output and B output/scaled FP32 delta. Clones protect observation from
   possible in-place updates.
2. Independently recomputed frozen `projection_parts()` GPU base/delta/sum;
   actual observed-parts sum; and comparisons against native.
3. Original `user_device.cpu_deltas()` CPU FP32 computation, plus a same-input
   CPU A-intermediate comparison to locate A versus B/delta arithmetic drift.
4. Full bundled QKV through production `tensor_bytes()` /
   `tensor_from_bytes()` and per-role CPU->CUDA transfer, with byte-identity,
   branch/shape/stride and numerical comparisons.
5. Final FP16 GPU-delta and CPU-delta assemblies, actual unequal-element/FP16
   output-flip counts, hashes, max absolute/relative L2, norms, elementwise
   threshold failures, shapes, dtype and device.

The instrumented native logits are also compared against the original and its
native control. Later-layer hidden changes in the old distributed run are not
used as if they were matched-input controls.

`--cpu-fp64-control` additionally evaluates original FP32 weights and hidden
values promoted to CPU FP64, then converts the delta back to FP32. This is
labeled a diagnostic arithmetic A/B only. It does not change the production UD
or establish that its full logit gate would pass. Floating-point additions and
GEMM reductions are nonassociative; different kernels/accumulation paths may
produce different rounding. That general fact is not evidence that it caused
this particular failure without these controls.

### C. Independent diagnostic ES replay

After the native workers have exited, a fresh bare-OPT GPU worker starts the
original CPU-only UD service over real 127.0.0.1 TCP. The original
`ProjectionSession` provides the actual returned corrections. New diagnostic
instance routing preserves base bias and `(base + delta).to(base.dtype)`;
passive output hooks record the actual QKV consumed by attention. No production
module is edited, and no A/B executes on the ES GPU.

For wire verification only, this diagnostic ES also computes a separate local
**CPU** expected delta from the original matrices. That value never supplies
the forward correction. This explicit observer is not part of the production
architecture or a latency measurement. A local-vs-UD mismatch is a consistency
failure; CPU-process numerics versus serialization/layout may still require
further isolation even if frame digests agree.

Records contain actual role order, hidden hashes, bundled reply hash, per-role
received delta, base, expected assembly and observed QKV hash, original native
hash comparisons and full strict logits. The replay sends a clean STOP/BYE
after collecting its observation even if diagnostic logits fail; production 95
still raises immediately on parity failure. All diagnostic timing is ineligible
for performance reporting.

Default output contains **no raw hidden/delta/QKV dumps**. For numerical ES QKV
magnitudes, explicitly add `--capture-qkv-reference`: this stores bounded
layer0/1 native QKV arrays in the fresh native-worker directory. Captures only
qualify when independently checked native controls, hidden and QKV hashes agree
with the frozen original. Their SHA/shape/dtype are verified again before use.
Magnitude records state whether hidden inputs match; unmatched later-layer
differences are consequences, not matched-input arithmetic isolation. No old
tensor is overwritten.

## Conclusion and next-action policy

The JSON emits all objectively triggered findings among:

| Enum | Triggering evidence / next action |
|---|---|
| REFERENCE_NONREPRODUCIBLE | Controlled native repeat/original reproduction differs; fix reference conditions first |
| GPU_RECONSTRUCTION_MISMATCH | Same-hidden native versus independent GPU reconstruction differs; inspect actual cast/order/base/delta |
| CPU_GPU_DELTA_NUMERICS | Same-hidden, same-A/B CPU and actual native GPU deltas differ; inspect A/B magnitudes and FP16 flips, then full replay |
| WIRE_BYTES_OR_LAYOUT_MISMATCH | Production round-trip or actual-reply/local-CPU consistency fails; distinguish transport/layout from process arithmetic |
| ES_ASSEMBLY_MISMATCH | Observed module output differs from independently recorded assembly |
| UNRESOLVED | Insufficient controls or no isolated finding |

Component findings do not force a single causal explanation for the original
run. Multiple sources are retained. A SHA mismatch without magnitude is not a
numeric attribution; an argmax match or low KL is not strict parity. The tool
does not propose a threshold/architecture fix from the currently unavailable
GPU evidence. The fix decision here is **hold production changes; run the
controls on SERAPH**.

The assessment also binds same-hidden/base/CPU-delta/assembly hashes between
the independent native and ES diagnostic workers, alongside the already
measured same-hidden metrics. If matching QKV outputs are followed by a changed
next-layer input, it reports that post-QKV decoder boundary without inventing
an attention error magnitude or specific cause. Observer-dependent native
drift is treated as a failed reference control and defers ES replay.

## Exact commands on SERAPH

Run from the repository root with original captured source files unchanged.
Replace each NEW_* tag with a new timestamp/tag; nonempty outputs are rejected.
No launcher is generated or edited.

First audit (no GPU):

```bash
python3 -u scripts/96_diagnose_r3a_parity.py --phase audit \
  --prepared /data/khuss/repos/GP_semcache/results/r3/a/physical_no_reuse_151665_20261029T140923Z/prepare \
  --pilot-inputs /data/khuss/repos/GP_semcache/results/r3/a/physical_no_reuse_151665_20261029T140923Z/pilot \
  --output-dir results/r3/a/parity_forensics_151665_NEW_AUDIT_TAG
```

Then, on an explicit GPU allocation with the original runtime/local snapshots:

```bash
python3 -u scripts/96_diagnose_r3a_parity.py --phase gpu-probe --enable-gpu-probes \
  --prepared /data/khuss/repos/GP_semcache/results/r3/a/physical_no_reuse_151665_20261029T140923Z/prepare \
  --pilot-inputs /data/khuss/repos/GP_semcache/results/r3/a/physical_no_reuse_151665_20261029T140923Z/pilot \
  --device cuda:0 --layers all \
  --output-dir results/r3/a/parity_forensics_151665_NEW_GPU_TAG
```

The GPU phase repeats the read-only audit before starting any model. It does
not launch pilot/latency or a benchmark. It forces local-files-only/offline
loading. Missing CUDA/dependencies/identity produces a FAILED diagnostic with
GPU findings NOT_RUN (or incomplete if a worker actually began). Worker timeouts
are bounded, process groups are reaped, stdout/stderr and partial evidence are
retained. A no-model readiness command uses the same input/output options with
`--phase dry-run`; it explicitly does **not** audit original evidence or execute
GPU/socket operations.

Required outputs are `r3a_parity_forensics.json` and
`r3a_parity_forensics_summary.txt`. GPU outputs additionally contain isolated
worker plans/results, source/weight checks, per-role metrics, wire observation,
process metadata, lifecycle and stdout/stderr logs. Only explicit QKV capture
adds raw arrays. All diagnostic output paths must be below a new
`results/r3/a/parity_forensics_*` directory.

## Verification scope

CPU tests use explicitly marked synthetic artifact fixtures. They test source
glob preservation, source/artifact tamper rejection, failed-row/frame/copy
accounting, ambiguous shutdown ordering, metric sign/norm, multiple conclusion
classes, no-overwrite, missing real evidence, GPU opt-in and no-model dry-run.
They do not stand in for the actual SERAPH experiment. The real-GPU unittest is
skipped unless SEMCACHE_R3A_PARITY_GPU=1 and original prepared/pilot/output paths
are supplied through its documented environment variables in the test.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_parity_forensics.py' -v
python3 -m compileall -q scripts/96_diagnose_r3a_parity.py src/semcache/edgelora/parity_forensics_audit.py src/semcache/edgelora/parity_forensics_probe.py tests/test_parity_forensics.py
python3 scripts/96_diagnose_r3a_parity.py --help
```

GPU results remain NOT_RUN in this environment. The observed real failure and
its exact original causal mechanism remain unverified locally and UNRESOLVED.

Verification on 2026-10-30: 99 CPU/contract/regression tests ran, **97 passed,
0 failed, 2 skipped**. The new forensic suite has 16 passing CPU tests and one
explicitly skipped real-GPU test; the other skip is an existing optional Torch
tensor test. Compilation and CLI help passed. The original absolute-path audit
was attempted and produced the expected missing-evidence FAILED report, with
no parity metrics imputed and GPU NOT_RUN. A separate readiness dry-run completed
without a model, socket, original-evidence audit or benchmark.
