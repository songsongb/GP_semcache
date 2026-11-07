# Distributed physical subsequence core — Phase1 progress

Physical sparse projection core IMPLEMENTED / CPU_FIXTURE_VERIFIED.
Full R5 generation/campaign integration INCOMPLETE. GPU_NOT_RUN. Phase2 NOT_READY.
This extends the prior shadow bridge; it does not replace the Exact Prefix path.

## New execution APIs

- `r5_sparse_qkv.SparseSession`: ES gather, one CPU correction bundle per layer,
  verified q/k/v consumption order, same hidden-input hash across roles.
- `serve_sparse`: separate CPU UD endpoint over the existing real loopback
  `Channel`; caller supplies matrices verified/loaded by existing
  `r5_multi_artifacts.validate/load`. Production process must use
  `user_device.configure_cpu` and hide CUDA. No ES adapter matrices are required.
- `PhysicalOwner`: M-stage matching/nonoverlap, actual owned all-layer total-QKV
  tensors, original F/A/I/S classes, actual attention reducer and CHU/PBR history.
- `projection_path`: bare GPU Linear hooks. Fresh rows alone execute native
  Linear and CPU LoRA. Full ordered projection outputs are scattered from actual
  fresh results and cached rows. Attention, FFN and head receive full sequences.
- `prefill`: opt-in native eager OPT prefill with actual attention outputs. It
  does not implement autoregressive generation or declare request completion.
  After actual generation succeeds the caller must call owner.complete with
  completed=True and the genuine saved prefill probabilities. No source is
  inserted before this completion barrier. B uses reuse=False but executes real
  sparse full-row correction, materialization/impact/policy overhead.

The new protocol `R5_SPARSE_TOTAL_QKV_TCP_V1` is carried in separate metadata,
never in the existing prefix protocol. It binds complete sealed R5 requests,
model/tokenizer/adapter/GPU identity, absolute fresh token positions, layer and
Q/K/V roles. Existing framing/session/adapter/payload SHA256 and64KiB header
limits are unchanged. Zero fresh rows still send versioned control frames with
layer coordinates in the hashed context; they send no tensor and execute no
CPU correction. Nonzero correction uses unchanged `user_device.cpu_deltas` on
the gathered rows once per layer, returning all three real CPU LoRA deltas.

EXACT_CONTEXT_VERIFIED requires original positions/causal prefix and per-layer
byte-equal projection input spans. This verifies input context, **not numerical
parity**: FP16 shape effects remain subject to frozen logit audits. Approximate
shifted/different-context reuse requires both the explicit
APPROXIMATE_SUBSEQUENCE_TOTAL_QKV mode and allow_approximate=True. It carries
source/target coordinates and never supplies prefix past-KV grants. Exact mode
cannot reuse entries derived from approximate mode. No total-QKV approximation
is silently treated as exact, and no numerical threshold changes were made.

## Physical policy and proof accounting

Admission first evaluates M-stage F/I/S on actual attention-derived impact,
then materializes compact cloned Q/K/V spans covering every layer. Metadata,
input-span hashes, token/source ownership, positions, size and all tensor bytes
are bound by the physical proof. Corruption/stale entries fail, without cold
fallback. Eviction removes both resident tensors and proofs; the cache is the
lookup index, so no second stale posting list remains. Logical and owned physical
storage are both bounded by512MiB/adapter and64 entries. M-stage insertion can
temporarily materialize one extra candidate before eviction; a future GPU
resource plan must include that transient and the active capture buffers.

Reuse F/CHU update only after a completed projection pass, per actually executed
span occurrence. PBR uses genuine matching attention history. Missing actual
attention is a hard error; CPU tests explicitly replace its reducer with labeled
fixture values only where Torch is absent. Physical results never come from the
old metadata shadow's virtual counters. Native/CPU rows are recorded for each
layer and role; TCP counts are actual Channel wire receipts, not predicted savings.

`audit_sparse_receipts` independently checks coordinates, execution coverage,
fresh/reused partitions, UD row counts and actual wire/payload counts. It is
explicitly NOT an independent GPU tensor-value or numerical-equivalence audit.

## CPU tests and exact SERAPH command

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHONPATH=src:tests python3 -m unittest test_r5_sparse_physical -q
PYTHONPATH=src:tests python3 -m unittest test_r5_mstage_bridge test_r5_i5 \
 test_r5_i5_controller test_r5_controller test_r5_semantic test_r5_multi \
 test_r5_i5r4 -q
```

Locally the first suite used a real separate forked CPU UD and localhost TCP
outside the socket-restricted sandbox.5 PASS /0 FAIL /2 SKIP (7 tests). Tests
prove actual ArrayTensor ownership/consumption, smaller transmitted payloads,
full reconstructed projection shapes, exact gather/scatter, explicit shifted
approximation, changed-causal-context rejection, zero-fresh controls, stale and
corrupt tensors, corrupted real TCP packets, and false count rejection. Same
fixture outputs match independently recomputed fresh outputs. This is not
OPT/FP16 evidence. Native bare Torch Linear and actual Torch impact tests skip
because Torch is absent locally; they must run in semcache on SERAPH.

Existing bridge/I5/controller/semantic/multi/I4 regressions:151 PASS /0 FAIL /
3 SKIP (154 tests). Combined without overlap:156 PASS /0 FAIL /5 SKIP (161).
Other skips are actual shadow attention reduction, native OPT and ROUGE. Python
syntax checks PASS. No dependencies/models were downloaded or GPU jobs launched.

## Precise remaining blockers before Phase2

1. Integrate this new session/owner into a separate versioned R5 generation
   driver and UD process launcher, using existing model/adapter preparation,
   actual GPU identity and full model fingerprint preflight. The old
   QualityDriver/119 runner still uses its original protocol and cannot select
   this mode. No dispatch fallback to Exact Prefix is allowed.
2. Preserve model/source/generation completion identities across prefill and
   native decode, and feed actual prefill probabilities to completion only
   after generation. Current standalone prefill API does not do this lifecycle.
3. Bound actual GPU attention capture: the conservative4-byte estimate at
   1280 tokens exceeds the current4GiB attention capture ceiling and rejects.
   Implement audited per-layer streaming impact or a reviewed resource plan,
   rather than silently increasing memory bounds. Full hidden/QKV capture and
   tensor hashing overhead must be included in E2E and peak estimates.
4. Export sealed sparse tensor/input/fresh-output/wire evidence and extend the
   independent R5 analyzer to replay scatter values and actual source/target
   ownership, then full-vocabulary prefill/decode logits with unchanged frozen
   thresholds. The new receipt replay alone does not establish these gates.
5. Prepare new immutable A/B/C plans and scorer/quality artifacts for genuine
   Qasper/2WikiMQA/MultiNews samples; A uses original full distributed recompute,
   B and C have matched semantic/impact/policy overhead. Keep source-inclusive
   timing, natural/controlled labels, generation quality and numerical fidelity
   separate. No correctness-qualified speedup until all required gates pass.

The existing150 readiness gate remains conservative and blocked. No new GPU
submission commands are provided: the CPU-tested physical core is real progress,
but presenting it as an independently audited R5 E2E campaign would be premature.
No historical file, baseline inference path, threshold, cache policy or serializer
was modified; no commit or push was created.
