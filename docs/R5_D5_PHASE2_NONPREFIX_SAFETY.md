# Phase 2: non-prefix exact reuse feasibility and proposed safety specification

This is a code-based feasibility assessment and a proposed later implementation
contract. It does not enable non-prefix physical reuse, change EXACT, or implement
Phase 3. GPU input equality, tensor equivalence, admissions, HITs and latency are
NOT_MEASURED. Real non-prefix opportunity counts require the SERAPH CPU audit.

## What current D5 authorizes

`r5_mstage_bridge.MStageShadow.begin()` extracts w3 windows, updates actual arrival
accounting, and looks up resident keys in the current adapter's owner. Sharing
and physical identity checks restrict candidate selection. Its position/context
reasons are diagnostics; metadata discovery alone never grants a physical HIT.

`r5_subsequence_physical.PhysicalOwner.begin()` additionally filters EXACT
selection to identical source/target absolute spans **and** identical token
histories through the window end. Therefore:

| Static category | Current EXACT outcome |
|---|---|
| Shared prefix, same position | Necessary metadata conditions may pass; ownership, residency, semantic selection, input proofs and consumed-tensor validation still required |
| Same-position non-prefix | Rejected by causal-history check |
| Shifted-position non-prefix | Rejected by position check; causal histories may also differ |
| Shifted occurrence targeting shared-prefix region | Rejected by position check, even when the target prefix is shared |

The filter runs before per-layer projection. A promising layer-zero input does
not rescue a non-prefix candidate under the existing mode. There is no
source-shorter requirement in this D5 subsequence mechanism.

`PhysicalOwner._project()` verifies resident ownership/integrity, requires an
EXACT-provenance source, hashes the gathered target layer input (shape, dtype,
bytes), and compares it with the source admission input proof. A mismatch computes
fresh rows for that layer. Native bare OPT supplies identical inputs to Q/K/V;
the q-role decision is reused for k and v under that model contract. Reused total
QKV rows are authenticated private snapshots and are scattered into full native
projection outputs. Attention, FFN, head, and target autoregressive past are still
computed on that target's full context. Integrity failures abort; they are not
ordinary input-ineligibility fallbacks.

`r5_subsequence_generation._generate()` uses the actual TinyBERT assignment and
owner selection, fresh decode corrections, full native attention/FFN/head, actual
streamed attention Impact, and real completion ACK before admission/CHU. It does
not pass an arbitrary cached subsequence as the target's past KV. The new Phase 2
code never invokes this runtime path.

## What per-layer equality could establish

For a row-local projection with fixed weights and adapter, the mathematical
result depends on its actual input row, weights, LoRA matrices/scaling, bias and
arithmetic contract. Identical tokens alone do not establish identical inputs.

For OPT's learned absolute positions, a same-token/same-position span after a
different prefix could have identical early projection inputs: initial embedding,
position addition and local normalization need not incorporate earlier tokens.
This is a theoretical candidate for a **specific layer**, not a measured result.
After causal attention, different prefixes generally change hidden states; later
layers must be checked independently. A matching layer-zero row never grants
later layers. Equal complete original contexts also do not guarantee equal prompt
histories, questions, truncation, positions, or input tensors.

Shifted tokens generally receive different position embeddings. Even if a rare
equal hidden row occurred despite different histories/positions, it would need a
separately reviewed layer-input authorization contract; current EXACT still
rejects the shift. Position-dependent transformations outside a projection, RoPE,
relative bias or a different model architecture cannot inherit this OPT argument.
Attention and later computation must use the target's current positions/context.

Even byte-equal inputs and fixed weights are not sufficient for a blanket claim
of GPU **bitwise** equality. Shape-dependent GEMM kernels/reduction order and
dense versus gathered LoRA computation can change rounding. The repository's
existing numerical-fidelity work makes this a material constraint. Prove or
control arithmetic layout for the proposed operator/backend, and preserve the
frozen activation/logit/argmax and generated-output audits. Tolerance PASS is not
bitwise PASS; matching text is not numerical PASS.

If source and target layer inputs differ, compute fresh. Reuse of unequal inputs
would require an explicitly approximate mechanism and separate research approval;
it is prohibited in Phase 2. Different adapters/users without authorization,
uncompleted sources, stale/evicted entries and integrity failures remain denied
even with byte-equal inputs.

## Proposed later contract — specification only

1. Preserve `EXACT_CONTEXT_VERIFIED` unchanged. Any future exact layer-input mode
   needs its own reviewed/versioned name, source provenance, plan/approval binding
   and independent auditor. It must not reinterpret historical EXACT or accept
   approximate-source provenance as exact. Do not add it to the Phase 2 CLI.
2. Discover only actual resident completed source entries under original
   TinyBERT/w3/F/A/I/S and per-adapter sharing rules. Token matches remain discovery
   aids. Preserve 64-candidate/entry and 512 MiB owner bounds and report duplicate-key
   ownership limitations; offline source pairs are not all simultaneously resident.
3. At each target layer/role, authenticate model/tokenizer/backend/dtype/layout,
   source and target adapter/LoRA fingerprints, scale/bias and eval configuration.
   Enforce authorized users/groups and no cross-adapter transfer.
4. After computing the target's real upstream hidden state, compare the gathered
   target input with authenticated source admission input bytes, shape and dtype.
   Account for proof/transfer/storage costs. Bind the decision to request digest,
   invocation ID, epoch, entry identity, source/target span, layer, role, weight
   contract, source completion and admission proof. Check each Q/K/V input or prove
   the pinned bare-OPT identical-input contract; never generalize it silently.
5. Prove the matched linear/LoRA arithmetic contract is exact under supported
   shapes, or use a reviewed layout that preserves that contract. Equality testing
   must not silently substitute an approximate kernel. Unsupported layouts and
   input mismatches compute fresh; corrupt proofs/tensors abort.
6. Consume a private authenticated snapshot under existing request-scoped
   ownership/TOCTOU rules. Preserve stale/replacement/view/storage mutation checks.
   Revalidate after checkpoints and for every new request; do not persist live
   authorizations across invocations. Keep the original threat model explicit.
7. Gather fresh rows, scatter approved projection rows at target positions, and
   compute the full target attention/residual/FFN/head. Never transfer source
   attention results or past KV. Recompute every unsupported later layer and
   decode as required by its own exact input proof.
8. Admit only after actual generation/UD completion; use actual Impact. CHU commits
   only after real consumption, with unambiguous once-per-entry/span accounting
   across layer/role receipts. Preserve F/A/I/S and PBR interval 100 per adapter.
   No policy/Impact values come from static matching.
9. Persist complete source/target inputs, consumed outputs, fresh arithmetic,
   gather/scatter/wire receipts, layer decisions and original provenance. Independently
   replay every grant and audit full-vocabulary logits/generation against matched
   recomputation with the frozen rules. A failing layer/logit audit is a FAIL even
   when generated text matches. No correctness-qualified speedup claim follows
   from token overlap or input equality alone.

## Focused later test plan — not implementation

- Same-position different-prefix tokens: verify the current EXACT mode still
  rejects them; in a separate proposed mode, permit only genuinely byte-equal
  verified layer inputs and recompute mismatching later layers.
- Shifted matching tokens and repeated prefix grams: no grant under current
  EXACT; unequal position-dependent inputs compute fresh in the proposed mode.
  Separately test any verified equal-input edge case without changing EXACT.
- Equal tokens/positions with one changed FP16 input element, signed zero, NaN,
  Inf, changed dtype/stride/shape, changed weights/adapter/scale: deny or abort as
  specified. Never promote hash similarity or tolerance to exact equality.
- Dense/full versus sparse/partitioned GEMM and CPU LoRA across empty, all-fresh,
  all-reused, mixed, repeated and boundary spans: preserve supported arithmetic
  and full activation/logit gates. Include real backend/shape regressions before
  any GPU pilot is authorized.
- Cross-adapter identical inputs, denied cross-user groups, foreign source user,
  stale epoch, eviction/replacement, metadata/registry/tensor mutation before and
  after authorization: preserve denial/abort and private-snapshot semantics.
- Layer/role partial acceptance, non-overlap selection, candidate-limit truncation,
  duplicate resident gram keys and failed completion: no double CHU, premature
  admission, phantom HIT, or skipped fresh computation.
- Continuous checkpoints and repeated invocations: counters/history persist while
  authorization epochs expire; dataset/condition state resets completely. Check
  PBR at the actual 100th per-adapter completed request in an explicitly scoped
  test, without changing the production interval.
- Independent evidence replay must detect missing/altered source inputs, wrong
  source layer/role/span, false input equality, reused approximate provenance,
  bad scatter/wire receipts, and falsely claimed numerical PASS.

The Phase 2 fixture tests exercise static classification/lifecycle, not this
proposed operator. No exact non-prefix physical implementation exists as a result
of this phase.
