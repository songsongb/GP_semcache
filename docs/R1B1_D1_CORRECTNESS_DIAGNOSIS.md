# R1-B1-D1 correctness diagnosis

This is a standalone correctness investigation, not a latency benchmark or a
production change. The executor, reducer, cache policy, R1-A/B1 measured paths,
and tolerances are unchanged. New code is in `diagnostics/r1b1_d1.py`, script
`71_diagnose_r1b1_correctness.py`, tests, and `jobs/r1b1_d1.sh` (no new `.sbatch`).

The reported OPT-2.7B FP16/64-token observations are **user-provided**, not local
measurements: Natural 63 reused tokens, three displaced windows and max logit
error 13.4453125; strict 54 reused tokens, zero displaced windows and max error
0.046875 with allclose false. No matching R1-B1 raw files exist in the local
checkout, and torch/Transformers/PEFT are absent. Therefore neither these
observations nor the proposed numerical cause have been reproduced locally.

## Artifact binding

The script requires existing B1 `raw.jsonl` files and their sibling
`manifest.json`. It selects the measured `same_user_exact` natural and strict
pair by length, repeat, impact algorithm, storage, and instrumentation. Defaults
are repeat 0, `token_precompute`, `RAW_CPU`, `minimal`; select the actual observed
configuration explicitly if different. Ambiguous/missing rows fail rather than
silently choosing another request. Raw input SHA256 and manifest SHA256 are saved.

For requested length 64, artifact verification requires 63/54 reused tokens,
zero strict position mismatches, and exactly these displaced natural spans:

|Source|Target|
|---|---|
|3:6|54:57|
|6:9|57:60|
|9:12|60:63|

All spans must have width 3, token IDs must match both source and destination
positions in the same exact prompt, executed spans must be nonoverlapping and
agree with recorded coverage, and source/destination adapter identities must
match. Prefix/context inequality is recorded. No tensor execution is implied by
passing artifact verification. Length 32 is a reference-bound control; no 64-token
coverage expectation is imposed on it.

GPU replay uses the reference's resolved model/tokenizer revisions, dtype,
synthetic adapter metadata, seed, encoder snapshot and policy configuration.
The original workload constructor must reproduce the exact saved token IDs.
An unsupported adapter, changed source cache residency/positions/context,
or changed natural/strict hit selections fails before conclusions are drawn.
Same-source identity is verified against both rows' saved source-policy entries.
The script does not tokenize an invented replacement prompt or cast the model
to another dtype behind the user's back.

## Diagnostic interventions

1. **FULL and repeat FULL** establish native reference and repeatability.
2. **Source no-hit capture** uses R1's unchanged `engine_request(detailed=True)`
   and detailed executor, matching B1 source construction even for minimal
   targets. Cache payloads are compared exactly with these
   captured Q/K/V values. Source execution is also compared with native FULL,
   distinguishing source-compute differences from storage changes.
3. **Natural**, **position guard before selection**, **position guard after
   natural selection**, and **strict** run through the original mixed executor.
   The guard only checks `source_start == destination_start` and exists solely
   in this script. Comparing both guard placements avoids confusing filtering
   with a different overlap resolution. Neither is a general safety predicate:
   equal position does not guarantee matching prompt/context/adapter.
4. **Strict FULL vs Mixed** captures each decoder layer's input/output hidden
   states, each projection input, and unscaled TOTAL Q/K/V. Per-layer first
   nonzero and first tolerance-failing differences identify role, token,
   absolute error and relative error. CSV contains all roles; JSON gives first
   hidden/QKV divergences and the first affected layer. “First” refers to this
   defined layer/hook order, not a GPU kernel trace.
5. **Identical-hidden shape probes** compare full rows then select fresh rows
   against selecting inputs first then projecting. This is run on both FULL
   and actual strict hidden inputs. Components are native total PEFT, base
   Linear, LoRA A, and LoRA B given the same full-A input. These distinguish
   shape-sensitive base GEMM behavior from adapter computation. The actual
   strict fresh outputs are also compared with the recomputed subset outputs.
6. **Full-shape injection** computes native full Q/K/V, then a correctness-only
   hook replaces cached spans. Strict and Natural both receive this control.
   No projection FLOP savings are claimed for it.
7. **Full-shape assembly** keeps the original mixed executor's gather/scatter
   and cached fetch. Its existing `fresh_projection` callback obtains fresh
   rows from a full-shape native projection on the current full hidden input.
   It is explicitly a correctness control, not a new production executor.

All controls keep `use_cache=False`, `output_attentions=True`, eager attention,
the same token positions and adapter. The script installs passive capture hooks
only after the original mixed context's validation, then removes them on exit.
These hooks copy tensors but never reproject inside capture. Full-shape controls
are separate forwards. A temporary source-capture function binding is confined
to this diagnostic process and restored; no repository runtime file is edited.
Source timing values produced by that helper are discarded as diagnostic capture
overhead invalidates their use as latency measurements.

## Evidence versus hypothesis

The report separates direct facts from hypotheses. Examples:

* Identical-input full/subset outputs that differ establish shape sensitivity
  for that component, layer, dtype and runtime. They do not alone prove the
  origin of the final 0.046875 logit error.
* Cache equal to captured source and assembled cached rows checks the storage
  and copied-payload path. Cache differing from native target QKV is not alone
  evidence of corruption: the source computation/context may differ.
* If repeat FULL is exact, source no-hit equals FULL, cache/source and fresh/cache
  assembly checks are exact, shape probes differ, and both full-shape strict
  controls eliminate the strict difference, the report records controlled
  evidence that replacing subset projection with full-shape projection removes
  the difference in this run.
* Natural full-shape injection remaining incorrect while strict full-shape
  injection equals FULL provides a control for displaced cached spans without
  subset GEMM in either forward. Guard-vs-strict selections and logits are
  reported independently; disagreement is never hidden.

Specific cuBLAS algorithm choices, Tensor Core accumulation order, or internal
precision mechanisms remain hypotheses without a kernel-level investigation.
There is no automatic claim that every FP16 mismatch has this cause.

Historical reproduction requires exact reported max-logit differences and
allclose outcomes, matched recorded software/GPU/driver/determinism metadata,
and checked execution-file hashes. Differences remain visible under
`DIAGNOSTICS_EXECUTED_REFERENCE_NOT_REPRODUCED`; current-run facts must not be
presented as proof of the historical run. Current TF32/FP16-reduction flags are
recorded, but flags absent from the original manifest cannot be retroactively
verified. Failed source/token/selection identity uses `FAILED`.

Tolerances come directly from each reference row's `correctness.atol/rtol`.
Natural and strict must agree. No CLI tolerance override exists. Allclose and
first failing elements use the same `torch.allclose(reference, candidate)` /
`torch.isclose` operand order as R1. Exact equality is also reported separately;
it means `torch.equal`, not bytewise identity of signed zero representations.

## Run commands

From `/data/khuss/repos/GP_semcache`, first audit the actual artifact without a GPU:

```bash
python3 scripts/71_diagnose_r1b1_correctness.py \
  --raw results/r1b1/opt27b_impact_v1/tokens_64/raw.jsonl \
  --lengths 64 --repeat 0 --impact-execution token_precompute \
  --storage RAW_CPU --instrumentation minimal --audit-only \
  --output-dir results/r1b1_d1/artifact_audit_64_v1
```

Replace the example reference root with the run containing the reported case.
This command must verify the three moves on real input before treating them as
confirmed. Local tests use a clearly labeled synthetic fixture with these spans;
passing those tests is not real-artifact verification.

The `.sh` Slurm job requests one GPU, 23:50:00, and splits 32/64 into separate
array tasks (one at a time). It configures the existing `semcache` environment,
`HF_HOME=/data/khuss/huggingface`, `HF_HUB_CACHE=/data/khuss/huggingface/hub`, both
HF offline flags, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

```bash
export D1_REFERENCE_ROOT=/data/khuss/repos/GP_semcache/results/r1b1/opt27b_impact_v1
sbatch --export=ALL,D1_RUN_TAG=diagnosis_v1 jobs/r1b1_d1.sh
```

To run only the reported 64-token case:

```bash
sbatch --array=1 --export=ALL,D1_RUN_TAG=diagnosis_64_v1 jobs/r1b1_d1.sh
```

Optional selector environment variables are `D1_REPEAT`, `D1_IMPACT_EXECUTION`,
`D1_STORAGE`, `D1_INSTRUMENTATION`. Output roots must be new; rerun failed small
diagnostics with a new tag, never overwrite existing evidence. No job is
submitted by the implementation agent and no runtime estimate is claimed.

For an optional small FP32 control, supply an existing OPT-125M FP32 B1 smoke
artifact and `--lengths 32` to the same Python script in a GPU allocation. It
loads that artifact's model/dtype without overrides. This is a separate small
model control, not proof of OPT-2.7B FP32 equivalence. If such a reference does
not exist, do not substitute another model into the FP16 case.

## Small outputs and tests

Outputs are JSON/CSV only: manifest/input hashes, `provenance_audit.json`, one
`diagnosis_LENGTH.json`, and layer, shape-probe, cache, displaced-cache, and
source-vs-FULL CSVs. Cache checks aggregate windows per layer/role and preserve
the first mismatch. No checkpoint/model weights, logits, attention matrices,
hidden tensors, or full raw input files are saved. CPU captures are temporary
in-memory tensors; capture overhead makes all execution unsuitable for latency
claims. Failure manifests and previously completed per-length files survive.

```bash
PYTHONPATH=src:tests python3 -m unittest test_r1b1_d1 -v
python3 -m py_compile scripts/71_diagnose_r1b1_correctness.py \
  src/semcache/diagnostics/r1b1_d1.py tests/test_r1b1_d1.py
bash -n jobs/r1b1_d1.sh
```

Local results: 4 offline tests passed; 3 optional tensor/model tests skipped
because torch/Transformers/PEFT are absent. Compilation and shell syntax passed.
Optional tests cover exact thresholds/token localization, tiny FP32 CPU and
FP16 CUDA controls, cache assembly, shape probes and hook/forward restoration.
GPU controls, real artifact verification, and causal conclusions remain
unverified until execution against the supplied SERAPH artifacts.
