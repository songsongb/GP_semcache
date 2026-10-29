# R3-A CPU delta causal intervention (diagnostic only)

This tool asks whether **local CPU FP32 LoRA deltas are sufficient to reproduce
the full failed physical replay**, without TCP or an independent User Device.
It does not implement a production fix, change tolerances, or run a benchmark.
Every report is `R3A_PARITY_DIAGNOSTIC`, with
`PRIMARY_PERFORMANCE_ELIGIBLE=false`. RAW QKV only; no caching, codec,
quantization, trained-adapter quality, remote-network or generation claim.

## Evidence supplied for job 151720

The user reports a same-allocation prepare/pilot/forensic run on moana-y6, RTX
3090. Its original directories are:

```text
/data/khuss/repos/GP_semcache/results/r3/a/physical_no_reuse_repro_151720_20261029T154408Z/prepare
/data/khuss/repos/GP_semcache/results/r3/a/physical_no_reuse_repro_151720_20261029T154408Z/pilot
/data/khuss/repos/GP_semcache/results/r3/a/parity_forensics_repro_151720_20261029T154408Z
```

All three absolute paths were checked and are **unavailable in the development
environment**. The following are supplied empirical observations, not new local
measurements or independently revalidated artifacts:

| Observation | User-reported result |
|---|---|
| Original native / independent repeats | Original logits and all QKV hashes reproduced exactly |
| Independent GPU reconstruction | Exactly equals native PEFT |
| Layer0 GPU/CPU FP32 delta max abs, Q/K/V | 3.91155481338501e-08 / 4.0512531995773315e-08 / 4.7031790018081665e-08 |
| Layer0 final FP16 QKV flips, Q/K/V | 3 / 9 / 17 |
| Layer1 same-native-hidden flips, Q/K/V | 4 / 3 / 38; these are not old-ES matched inputs |
| Local packing / actual TCP / ES assembly | Sampled round trips exact; replay matches local CPU; actual QKV equals assembly |
| Replayed full logits | Strict FAIL: max_abs 0.0546875, relative_l2 0.0019029290937084956, 828548 violations |
| Last-token metrics | Argmax matches and KL passes; strict full-logit gate still fails |
| Existing selected probes | Layers0,1: six roles sampled, not an all-96 numerical census |
| Causal resolution before this intervention | UNRESOLVED |

These distinguish real tiny FP32 differences, real FP16 output flips and a real
strict failure. They do not by themselves prove that the CPU arithmetic is
sufficient for the full logit trace. **New GPU findings here remain NOT_RUN.**
No original experimental evidence is replaced by the CPU fixtures below.

## Source and identity preservation

The implementation is add-only:

- `scripts/97_attribute_r3a_cpu_delta.py`: audit, readiness and explicit GPU orchestration.
- `src/semcache/edgelora/parity_causal_evidence.py`: CPU-only evidence gates and causal assessment.
- `src/semcache/edgelora/parity_causal_probe.py`: isolated native, local oracle and physical replay workers.
- `src/semcache/edgelora/parity_causal_process.py`: bounded child lifecycle and failure cleanup.
- `tests/test_parity_causal.py`: synthetic CPU contracts/arithmetic and an opt-in GPU test.
- This document.

No filename matches `r3a_*.py`; no production source-identity file, script95,
script96, previous forensic module, prepared reference/adapter, threshold or
old generated output is edited. The existing `audit_failure()` and unmodified
`validate_prepared()` run first. They verify the original 15-file source map,
prepared binary/artifact hashes, matrix shapes/dtypes/nonzero values, token
identity, endpoint, failed pilot, clocks, frames and cleanup. HEAD differences
alone are not evidence of corruption. A captured-source mismatch fails closed;
restore the recorded files or prepare a separate future cohort, never rewrite
old hashes or bypass admission.

The prior script96 report is separately verified against its embedded worker
copies, hashed outputs, diagnostic source hashes, original prepare/pilot, native
controls, GPU UUID, CPU affinity and thread settings. Its six selected probes
remain six. Its ES replay must contain all 96 outputs. A prior replay without a
logit hash is explicitly `NOT_ESTIMABLE` for full-logit bitwise reproduction.

Each new worker records the identical prepared/reference/adapter/source hashes,
model/tokenizer revisions, input token IDs/hash, original dtype/scale path,
versions, original GPU UUID, CPU affinity/thread count and strict thresholds.
Existing pinned loaders use `local_files_only=True` with offline environment
variables. Every actual A/B comes from the verified original binary, with
per-matrix rehashes; PEFT constructor initialization is overwritten before any
forward. No adapter fixture generator is called by this CLI.

## Sequential controls and intervention

The parent never loads a model. It fully waits for and reaps each GPU child
before starting the next. There is one OPT GPU copy per child:

1. **native-a**, then **native-b**, each independently loading native PEFT with
   original A/B. Each makes a hook-free forward, a passive hash forward, and an
   all-32-layer same-native-hidden probe forward. All logits must be bitwise
   identical to the original canonical FP32 reference; passive forwards must
   reproduce every original hidden and native QKV SHA. Independent GPU
   `projection_parts()` and native observed-parts assembly must be exact for
   all 96 roles. Native drift stops causal attribution and prevents either
   intervention/replay. A GPU reconstruction mismatch has its own label.
2. **local-cpu**, a separate bare-OPT GPU child. It verifies no LoRA parameters
   on the GPU model. For each actual layer it copies the current incoming FP16
   hidden through the same canonical CPU representation used by production,
   calls the original `user_device.cpu_deltas()` with the original FP32 A/B,
   and transfers each FP32 delta locally back to CUDA. The native base Linear
   keeps its bias. The result is exactly `(base + delta).to(base.dtype)`, with
   base/final FP16, intermediate/delta FP32, scale 1.0. There is **no socket and
   no distinct UD process** in this counterfactual. All three hidden inputs
   must agree within each layer; the bundle is discarded after Q/K/V, before
   moving to the next layer. Passive module hooks verify the actual consumed
   output against independently predicted assembly. Original A/B and bare base
   fingerprints are checked again after the forward.
3. **tcp-replay**, another bare-OPT GPU child, starts an independent CPU-only
   UD running the unchanged original `serve()` implementation. It uses the
   original `ProjectionSession`, actual 127.0.0.1 TCP HIDDEN/DELTAS messages and
   original packing/unpacking. Only real UD replies drive the forward. The ES
   observer computes local CPU deltas solely for consistency checks. No GPU
   LoRA parameters or execution enter this bare ES model. The replay must
   observe 32 exchanges and 96 actual module outputs, with matching role order,
   original matrix identity, CPU placement/configuration and clean STOP/BYE.

All hidden/base/CPU-delta/actual-QKV hashes, shapes/dtypes, output norms,
assembly comparisons and full original `check_parity()` metrics are retained.
All native probes are matched to **native** inputs. A later layer in the local
or TCP trace can have a different hidden than native. Its old same-native-hidden
metrics remain a separate control; they are never applied as arithmetic
magnitudes for the changed ES input. The report binds the first local QKV
divergence to identical native hidden/base/CPU delta and independently predicted
CPU assembly, before making a sufficiency statement.

The original UD effective affinity and thread count are required. New native,
local, ES and UD CPU runtime fingerprints include actual CPU model, Torch build
configuration, thread/inter-op counts, affinity, MKLDNN state and BLAS-related
environment. A mismatch fails closed. The earlier UD did not retain every BLAS
detail: that limitation is not filled by assumption. New environment matching
and exact old payload/QKV reproduction provide distinct evidence.

The UD shares its diagnostic ES process group, while remaining an independent
CPU-only OS process. Normal exit is reaped; a timeout or crashed ES also kills
the inherited group, including a surviving UD. Listeners/sockets close in
`finally`. stdout/stderr, lifecycle, partial results and concrete failure reasons
are preserved. These are diagnostic forwards: no latency CSV, request-latency
summation, pilot eligibility or performance timing is produced.

## Assessment and its limits

`CPU_DELTA_SUFFICIENT_FOR_MATCHED_REPLAY` requires all of:

- Exact hook-free/passive/original native reproducibility in both independent
  controls, all 96 native hashes, and exact GPU reconstruction.
- Matching original identity, GPU UUID, CPU runtime and independently exited
  child PIDs; bare GPU models; original CPU A/B; actual physical UD CUDA absent.
- Local oracle and new physical replay **full-logit canonical FP32 SHA, dtype,
  shape and byte count equal**, plus all 96 QKV hashes and their hidden/base/delta
  component hashes equal in observed role order.
- Both full-logit comparisons fail the unchanged strict original gate; physical
  replies equal local CPU verification and actual outputs equal assembly.
- All earlier replay hidden/base/CPU-delta/QKV hashes reproduced, its strict
  metrics reproduced, and original failed-pilot per-layer hidden/bundled-delta
  payload hashes and strict metrics reproduced.
- First native-to-local QKV change isolated to matching hidden/base/CPU delta
  and predicted CPU assembly, with actual FP16 flips measured in the native
  same-hidden control.

That result would be strong **sufficiency** evidence under these conditions:
TCP and independent-process transport are not needed to reproduce the new
matched physical replay's full-logit divergence. It does not prove necessity,
every delta difference's effect, a production remedy, or bitwise identity to an
older logit byte stream that was never retained. Original pilot failures remain
failures and cannot enter latency. The local oracle is not the physical UD.

If any compared tensor differs, the report gives the first actual layer/role,
hidden equality, component SHAs and whether matched-input attribution is
permitted. Default SHA-only differences have **no invented numerical magnitude**.
Use `--capture-comparison-tensors` for a fresh diagnostic with at most 128 MiB of
local hidden/base/delta/QKV arrays. The replay revalidates each capture SHA and
measures numerical differences for the actual current tensors. When hidden
differs, magnitudes are descriptive downstream differences, never new roots.
Default output stores none of these arrays.

`--capture-logits` optionally saves bounded FP32 arrays (at most 16 MiB per
forward) in new output. By default the hash over the **entire** little-endian
contiguous row-major FP32 serialization, shape, dtype and byte count suffices
for bitwise comparison. Every forward still runs the full strict metric check;
approximate metrics, token IDs, low KL or matching argmax cannot substitute for
the hash or the gate. SHA equality is interpreted under normal cryptographic
integrity assumptions.

`--cpu-fp64-control` runs additional same-native-hidden FP64 arithmetic probes,
cast back to FP32 for comparison. It does not change the all-layer CPU oracle
or UD precision. FP16 midpoint tests illustrate rounding boundaries only. No
claim that this counterfactual fixes full-logit parity is made. A partial
layer0-only propagation ablation is not implemented; it is not needed for the
all-layer sufficiency question.

## SERAPH commands (user executes; no launcher is modified)

Run in `/data/khuss/repos/GP_semcache` on branch `exp/raw-semcache-r3` using the
original Python/PEFT/Torch environment and local pinned snapshots. Use a new
output tag each time; nonempty outputs, and outputs nested inside original
evidence, are rejected. First, an optional CPU-only audit:

```bash
python3 -u scripts/97_attribute_r3a_cpu_delta.py --phase audit \
  --prepared /data/khuss/repos/GP_semcache/results/r3/a/physical_no_reuse_repro_151720_20261029T154408Z/prepare \
  --pilot-inputs /data/khuss/repos/GP_semcache/results/r3/a/physical_no_reuse_repro_151720_20261029T154408Z/pilot \
  --forensic-inputs /data/khuss/repos/GP_semcache/results/r3/a/parity_forensics_repro_151720_20261029T154408Z \
  --output-dir results/r3/a/parity_causal_151720_NEW_AUDIT_TAG
```

From the repository root, request a **fresh** moana-y6 GPU allocation using
the repository's documented partition. This is a diagnostic Slurm invocation,
provided for the user; it was not executed here. The time limit is a resource
cap, not a runtime estimate:

```bash
srun --partition=batch_eebme_ugrad --nodelist=moana-y6 --gres=gpu:1 \
  --cpus-per-task=4 --cpu-bind=none --mem=64G --time=01:00:00 \
  python3 -u scripts/97_attribute_r3a_cpu_delta.py \
  --phase gpu-intervention --enable-gpu-probes \
  --prepared /data/khuss/repos/GP_semcache/results/r3/a/physical_no_reuse_repro_151720_20261029T154408Z/prepare \
  --pilot-inputs /data/khuss/repos/GP_semcache/results/r3/a/physical_no_reuse_repro_151720_20261029T154408Z/pilot \
  --forensic-inputs /data/khuss/repos/GP_semcache/results/r3/a/parity_forensics_repro_151720_20261029T154408Z \
  --device cuda:0 \
  --output-dir results/r3/a/parity_causal_151720_NEW_GPU_TAG
```

Replace `NEW_GPU_TAG` with a fresh timestamp/tag before execution. Node selection
does not guarantee GPU identity on a multi-GPU node. The tool compares the
allocated physical UUID against the immutable original ES UUID and fails before
model inference if it differs. Inspect the original `pilot/r3a_es_result.json`
`gpu.uuid` and Slurm-assigned GPU UUIDs (`nvidia-smi -L` inside the allocation),
then request that physical GPU through the site's scheduler resource mapping.
`cuda:0` refers only to the allocation's exposed GPU. Do not expose an
unallocated device, edit original metadata, or disable the UUID gate. A new
prepare on another GPU creates a different cohort; it cannot establish this
original run's matched attribution. Original CPU IDs must also be available in
the new allocation; `--cpu-bind=none` does not bypass a scheduler cpuset. If
they are unavailable, request an allocation permitting those CPUs rather than
silently substituting affinity. Site-specific GPU IDs/reservation options and
the actual original UUID are unavailable locally and are not invented here.

Use `--phase dry-run` with the same three paths for no-model readiness only.
It explicitly does not audit evidence or execute inference/socket operations.
There is no model download, launcher edit, automatic Slurm submission or push.

## Outputs and verification

Root outputs:

- `r3a_cpu_delta_causal.json`: original audit, prior forensic audit, source/input
  hashes, per-run/compare identity, independent process results, all-role
  comparisons, first differences, strict metrics and structured sufficiency.
- `r3a_cpu_delta_causal_summary.txt`: concise scope and findings, including
  NOT_RUN/UNRESOLVED and concrete failure reasons where applicable.

GPU execution additionally writes per-child `diagnostic_plan.json`,
`worker_result.json`, `lifecycle.json`, `*.stdout.log`, `*.stderr.log`,
`gpu_execution.json`, all-role traces and original-UD protocol artifacts.
Only explicit capture flags add tensor arrays. Diagnostic COMPLETE means the
analysis completed, not parity passed or sufficiency established.

CPU tests use synthetic schema sentinels and a clearly `TEST_FIXTURE_ONLY`
nonzero 32-layer scalar decoder. The latter proves local arithmetic versus raw
production pack/unpack equivalence and detects an intentional delta change;
it is not an OPT/PEFT/CPU-versus-GPU experiment. Gate tests cover positive and
negative sufficiency, all-96 accounting, changed identities/SHA, original versus
fresh evidence, input mismatch, no-overwrite, strict parity, canonical stored
logit descriptors and process cleanup after both normal and crashed leaders.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_parity_causal.py' -v
python3 -m compileall -q scripts/97_attribute_r3a_cpu_delta.py src/semcache/edgelora/parity_causal_evidence.py src/semcache/edgelora/parity_causal_probe.py src/semcache/edgelora/parity_causal_process.py tests/test_parity_causal.py
python3 scripts/97_attribute_r3a_cpu_delta.py --help
```

The GPU unittest is skipped unless `SEMCACHE_R3A_CAUSAL_GPU=1`, with
`SEMCACHE_R3A_CAUSAL_PREPARED`, `SEMCACHE_R3A_CAUSAL_PILOT`,
`SEMCACHE_R3A_CAUSAL_FORENSICS` and `SEMCACHE_R3A_CAUSAL_OUTPUT` explicitly set.
Its success means diagnostic execution, not an assertion of the desired cause.
No actual GPU causal intervention has run in this development environment;
the current fix decision is **hold production changes pending matched evidence**.

Local verification: **135 tests, 132 passed, 0 failed, 3 skipped**, combining
the 125-test causal/forensic/contract/R2-D/M9-A/C3/C4 run (122 passed, 3 skipped)
and the 10 existing CPU localhost protocol tests. The protocol tests initially
hit sandbox socket restrictions; all ten passed with permission outside the
sandbox. Skips are the two explicitly opt-in GPU diagnostics and one existing
optional Torch tensor test. The new suite has 25 passing CPU tests and one
skipped GPU test. Compileall and CLI help passed. All 473 preexisting tracked
files, including all production sources and previous forensic tools/docs, were
verified unchanged by SHA256. Missing-original audit and no-model dry-run reports
are retained separately, with GPU NOT_RUN and no causal result imputed.
