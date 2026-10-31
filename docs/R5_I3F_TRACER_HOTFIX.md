# R5-I3F decorated OPT tracer hotfix

Slurm Job 152354 failed in Block A before completing real RTX 3090 inference.
This hotfix repairs the observation path. It makes no numerical-fidelity or GPU
completion claim. Job 152354 evidence remains untouched; its artifacts are not
present on this local host. No GPU jobs, Slurm submissions, Git pushes or other
remote repository actions are performed here.

## Exact cause

The prior tracer registered `method.__func__.__code__`. For a method decorated
with `functools.wraps`, that code belongs to a wrapper such as
`wrapper(*args, **kwargs)`. That frame has no local named `self`.

Meanwhile, `inspect.getsourcelines(method)` follows `__wrapped__`, so assignment
sites were analyzed in the original OPT method. The tracer therefore paired the
wrapper's code object with the original function's source locations, then indexed
`frame.f_locals['self']` on the wrapper event. This explains the reported
`KeyError: 'self'` through Transformers' deprecation wrapper. A second weakness
was storing the currently resolved layer in a shared code-to-layer mapping.

The pre-hotfix HEAD implementation was reproduced locally against the new
decorated CPU fixture and raised exactly `KeyError('self')`. This is a direct
regression reproduction, not a new GPU execution. Its result is preserved in
`results/r5/fidelity/tracer_hotfix_20261031/pre_hotfix_reproduction.json`.

## Correction

`original_method()` explicitly unwraps the underlying function with
`inspect.unwrap()`, verifies that it is a Python instance method with a positional
`self`, and supplies that same function to both source analysis and code-object
registration. Source analysis excludes nested definitions belonging to other
code objects. Opaque decorators without an inspectable original fail with a
`FORENSIC_OPT_FRAME` diagnostic.

The code registry contains only fixed attention/layer role and assignment-site
metadata. Actual layer attribution is established exclusively on an original
method's `call` event. The frame's `self` must be the exact registered module
instance, with the expected original code and role. Wrapper frames are not in
the registry and are never assumed to contain `self`.

Each verified frame receives its own layer binding, pending assignment and
residual count. Nested frames that share a function's code cannot overwrite
another frame's layer or residual state. All 32 layers and their attention
methods must complete, and required KV/residual and discovered attention-site
observations must be present. Missing original frames, unknown instances,
missing tensors or ambiguous method identities fail with diagnostics; they
cannot silently create a successful forensic run.

Exception-unwind return events do not attempt to read unfinished computation
locals, preserving the original exception. The same guard covers the observed
native projection closure. The context always restores the previous Python
tracer, removes registered hooks and clears remaining frame state, including
when a tracer callback raises or preparation fails.

No R3/R4 source, model or adapter value, cache decision, generation operation,
numerical threshold, TCP arithmetic, tensor file schema or Evidence contract is
changed. The tracer remains enabled and required observations remain required.

## Files and regression coverage

| File | Change |
|---|---|
| `src/semcache/edgelora/r5_fidelity_capture.py` | Original-function resolution, frame-local attribution, completeness checks and exception cleanup |
| `tests/test_r5_fidelity_tracer.py` | Nine CPU decorator/attribution/cleanup regressions |
| `docs/R5_I3F_TRACER_HOTFIX.md` | Root cause, verification, limitations and resubmission |

The tests use the requested `@wraps` decorator and `wrapper(*args, **kwargs)`,
including two stacked decorators on attention. They verify original source/code
alignment, wrapper frames without `self`, exact attention and both residual
observations for all 32 layers, distinct layers sharing method code, nested
32-layer calls, byte-identical decorated computation with and without capture,
original-method exceptions, tracer callback exceptions, missing required
observations, wrappers that bypass the original, opaque wrappers and unknown
module instances. Every test also asserts tracer restoration and hook removal.

Run the full fidelity and existing R5 CPU suites with localhost TCP permission:

```bash
PYTHONPATH=src:tests python3 -m unittest \
  tests.test_r5_fidelity tests.test_r5_fidelity_tracer \
  tests.test_r5_fidelity_processes tests.test_r5_contracts \
  tests.test_r5_semantic tests.test_r5_controller tests.test_r5_processes \
  tests.test_r5_perf_evidence tests.test_r5_perf_schedule \
  tests.test_r5_perf_processes -v
```

Fresh retained evidence and test logs belong under
`results/r5/fidelity/tracer_hotfix_20261031/`. Prior fidelity validation directories
and the failed GPU job are never overwritten.

Final CPU result: **92 tests passed in 539.244 seconds**, with zero failures,
errors or skips. This comprises all 37 R5-I3F tests (including the nine new
decorator regressions) and all 55 existing R5 tests. The full suites include
actual separate-process localhost TCP execution, unchanged production reference
comparisons, corruption/failure controls, deterministic independent audits and
the four-block ABBA CPU benchmark. The nine decorator tests also compare exact
tensor bytes with uninstrumented computation for 32 sequential and 32 nested
layer calls. The intentionally failing launcher controls print FAILED results
while their unittests correctly pass.

The retained fidelity capture completes 20 requests / 40 forwards and four
automatic physical HITs. All 1,928 layer/logit comparisons and 1,536 KV
comparisons remain bitwise equal in the scalar fixture, all eight frozen
numerical checks pass, and all 20 outputs match the unchanged production
reference's tokens, text and logit bytes. The two independent analyses retain
ten byte-identical diagnostic reports. The earlier sealed R5-I3F validation
directory still matches its immutable inventory. Compilation, Slurm shell syntax
and whitespace checks pass. `verification.json` and the final output manifest
record the result and source hashes.

## Remaining GPU uncertainties

This host has no Torch, NumPy, Transformers or PEFT installation. Decorated
Python tracing and scalar CPU/TCP inference can be tested here; the exact
installed SERAPH Transformers functions, real FP16 tensors and RTX 3090
execution require operator resubmission. Unsupported wrappers or assignment
structures will fail with diagnostics rather than skip required observations.

Fixing the crash does not explain the historical R5_03/R5_08 logit errors.
Their numerical root cause remains `ROOT_CAUSE_UNRESOLVED`. GPU forensics,
frozen GPU numerical gates, Native PEFT strict parity and correctness-qualified
speedup remain unestablished. R5-I4A is not started.

## Exact SERAPH resubmission

Commit the hotfix on `exp/raw-semcache-r5-integration` before submission. The
unchanged job checks clean tracked source, matching committed HEAD, explicit
opt-ins, the allocated RTX 3090, offline checkpoints and the verified TinyBERT
revision `34707a33cd59a94ecde241ac209bf35103691b43`. Each submission creates a
new job/timestamp directory and fresh prepared identity.

```bash
cd /data/khuss/repos/GP_semcache
mkdir -p logs
R5_ENABLE_GPU=1 R5F_ENABLE_GPU=1 \
  R5F_SOURCE_COMMIT=$(git rev-parse HEAD) \
  sbatch jobs/r5_opt_fidelity_forensics.sh
```

This command is prepared for the operator and is not executed by this task.
