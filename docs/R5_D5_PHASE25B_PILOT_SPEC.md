# Phase 2.5B bounded pilot proposal — no execution or authorization

This proposal follows the Phase 2.5A operator. No GPU runner, allocation, Slurm
submission or benchmark is implemented here. The user must separately review
this specification and authorize execution. A CPU fixture PASS is not a NATURAL
result or a GPU numerical PASS.

## Prerequisites and scope

Require the original sealed Phase 2 plan/audit, a fresh passing Phase 2.5A CPU
receipt with the native CPU contract test executed (not skipped), reviewed source
identity and actual resource/transport forecasts. Preserve existing offline
OPT-2.7B/FP16/TinyBERT/adapter/GPU identity and deterministic/backend preflights,
frozen thresholds, sharing/integrity model, evidence completeness and failed
attempt seals. Stop on any guard failure. The present local environment lacks
Torch/Transformers and the original real assets, so these gates are not satisfied.

Compare three independently initialized conditions:

| Condition | Behavior |
|---|---|
| A | Native Full Recompute, all original computation and strict evidence |
| B | Original D5 EXACT, unchanged position/causal/input rules |
| C | `LAYER_INPUT_BYTES_VERIFIED_V1` with full-shape reference-output guard |

C must report cached rows consumed separately from work skipped; native skipped
projection/LoRA rows are zero under this contract. C is an expensive numerical
diagnostic, not a speedup hypothesis. Do not enable the scalar fixture work-skipping
contract on a native model. An optimized native contract requires separate
arithmetic proof/validation and review.

## Two tracks with separate plans, outputs and denominators

**Controlled correctness.** Separately labeled source/target fixtures may engineer
matching w3 spans and positive/negative actual layer inputs. Cases include B with
equal initial inputs but divergent later hidden states, C with unequal learned
position inputs, mixed fresh/reused spans, shape-dependent rounding rejection,
fully reused/empty computed-row boundaries and integrity/ownership negatives.
Use native OPT/model forwards for model claims; a direct projection equal-input
edge case must be labeled as an operator fixture and cannot establish native
model input equality. The scalar C equal-input edge remains a CPU fixture only.
Each native correctness attempt is bounded to two requests/condition and one
explicit case; no sweep or cache-policy adjustment.

**NATURAL.** A first pilot may take the first three original invocations of one
predeclared dataset from the sealed Phase 2 sequence. This is a bounded prefix,
not selection by observed overlap. A/B/C must receive those same requests in
exact order with original prompts/token IDs/generation policies/users/adapters.
No artificial source, priming, extra duplicate, altered context, reordered request
or preferred adapter is allowed. Predetermine the dataset and condition order
before outcomes; do not select a replacement dataset or source after zero hits.
Keep one model/per-adapter owner/index/channel/UD per dataset/condition until its
end, and reset completely at boundaries. A broader 50-request session is a later
resource-reviewed attempt, never automatic continuation of this prefix pilot.

Invoke all TinyBERT assignments, w3 discovery, F/A/I/S, actual attention Impact,
ACK barrier, CHU and PBR naturally. Three requests cannot trigger PBR 100 per
adapter. Report zero eligible or consumed NATURAL rows honestly. Do not transplant
controlled sources into the NATURAL cache. Historical experimental replay remains
part of the full workload; the first three-prefix pilot contains whichever
original records occur there without adding the later repeats.

## Evidence and independent audit required before pilot execution

The old one-block approval/worker/analyzer does not support three conditions,
new entry metadata/input storage, transitive restriction or overlapping
computed/reference and cached-consumed rows. It must not silently accept this
schema as APPROX. Build and review a separate bounded evidence/audit route first.

For every request/layer/role, save:

- Source invocation/sample/adapter/entry identity, actual resident candidates,
  candidate-limit/non-overlap exclusions, selected spans and A/B/C/D category.
- Authenticated source input bytes and actual target inputs; literal equality,
  shape/dtype proofs, exact reason for every denial, and per-role shared-QKV-input
  contract receipts. Each layer has its own decision.
- Actual weight/bias/adapter/backend/CPU-precision/session bindings, mode/root
  provenance, source generation completion and UD ACK. Preserve no-transitive
  source eligibility and immutable input storage accounting.
- Full native base projections, actual CPU hidden/delta wire payloads and
  precision receipts, additions/casts, paid full-shape output references, consumed
  private QKV snapshots and final scatter outputs. Input equality without output
  equality cannot authorize C.
- Request-scoped authorization/admission roots, epoch/registry/container/storage
  identity, consumed-snapshot checks, aborts and expired grants; policy histories,
  actual attention evidence, CHU events and occupied logical/QKV/input bytes.
- Actual computed rows, cached output rows consumed by category, reference rows,
  genuinely skipped rows, wire bytes and complete wall timing scopes. Never add
  inclusive scopes or subtract validation/persistence as free work.
- Full native attention/residual/FFN/head activations needed by the frozen audit,
  full-vocabulary logits at every generation step, last-token KL, argmax, generated
  IDs/text, original quality scores, and matched A/B/C numerical PASS/FAIL.

The Phase 2.5A value primitive is one component: it reconstructs source admission
metadata from fresh completed source records, checks literal layer input equality,
private snapshot/value/category, counts and scatter. Extend independent inventory,
wire/UD, policy replay and full-model numerical audits separately. A primitive
PASS does not prove base GEMM or native model equivalence. Independently recompute
the reference condition; charge C's internal reference computation inside C's
primary timing and account for external audit recomputation separately.

No tolerance relaxation, last-token-only replacement for full-vocab checks,
hash-only unavailable values or text-only parity is allowed. A failing numerical
metric remains FAIL even with correct final text. Operator internal byte equality
must not be labeled universal bitwise equivalence of independent GPU runs.

## Resource gates

The Phase 2 50-request upper envelopes already exceed old limits: about
126/150 GiB per condition (32/96-token generation caps), 252/299 GiB per paired
two-condition dataset and 1.45M/4.35M files. A three-condition plan and extra C
evidence are larger; no 50-request GPU session is currently cleared.

For the proposed first-three-request NATURAL prefix, recompute strict forecasts
from its actual lengths/routes/caps for **three**, not two, conditions. At the
512-input/96-generation analytical envelope, the old per-forward file formula
gives roughly 391k files for three requests times three conditions, before the
new references/role inputs/activation/evidence sidecars. Four requests would
already exceed 500k under that old formula. These are planning envelopes, not
measurements or a preflight PASS. If complete numerical/activation evidence makes
even three requests exceed any guard, stop; propose a new explicitly bounded
lossless packing/layout and reviewed budgets before changing pilot scope.

Respect 16 GiB per-adapter ES, 4 GiB per-adapter UD, 32 GiB per-condition,
64 GiB old run inventory, 500k files, 16 MiB JSON and 256 MiB per-request metadata/
ledger limits, 128 MiB tensor-reader bound and 96 GiB initial free-disk guard until
explicitly reviewed otherwise. Cache remains 512 MiB/adapter and 64 entries;
new source input storage adds 30 MiB plus metadata at 64 FP16 entries per adapter
and is part of logical S. Actual tensor/input/host and transient comparison buffers
must be counted. Native projection-weight rehashes and input/output serialization
are material overhead and need host/transfer/disk forecasts.

GPU forecast/actual peak must remain below 22 GiB and host available memory at
least 32 GiB. Reference verification retains full projection results temporarily;
do not assume the old activation forecast accounts for every new buffer. Preserve
7,200s worker and 14,460s paired-run/4h30 allocation limits until a reviewed
three-condition deadline replaces that two-condition assumption. No deadline is
presented as measured completion time. Failure seals and no-retry behavior remain.

## Falsifiable outcomes

Controlled positives must show actual authenticated cached output consumption;
negatives must show fresh computation or abort for the stated reason. Numerical
audits can fail despite equal text. Controlled PASS is reported only in its track.

For NATURAL, report selected residents, exact input/output equality results,
per-category/layer/role consumed rows, fresh/reference/skipped work and all
numerical/overhead metrics. Zero eligible hits and no speedup are valid outcomes.
If static overlap disappears under real admission, duplicate-key ownership or
position/history-dependent inputs, that falsifies the corresponding usefulness
hypothesis; it does not justify altering requests or safety constraints.
