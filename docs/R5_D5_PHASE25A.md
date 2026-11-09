# D5 Phase 2.5A — verified layer-input QKV reuse

`LAYER_INPUT_BYTES_VERIFIED_V1` is implemented as a separate, explicitly selected
owner/operator. Original EXACT, APPROX, generation, wire, policy, integrity,
Phase 1/2 implementation files and sealed artifacts are unchanged. No GPU run,
Slurm job, Phase 3 benchmark or push is part of this work.

The supported native arithmetic path **recomputes all projection/LoRA rows** and
verifies the current full-shape output bytes before consuming any cached row.
It is a correctness/consumption operator, not a native projection accelerator.
Native projection work skipped is zero. Work-skipping positive tests use a
separate, explicitly synthetic scalar row-local contract. Neither input equality
nor those fixtures establish OPT-2.7B GPU numerical correctness.

## Findings

| Finding | Local result |
|---|---|
| Implementation | IMPLEMENTATION_PASS |
| Required portable CPU safety/value tests | CPU_SAFETY_TESTS_PASS |
| Real Phase 2 static extraction | BLOCKED: authoritative SERAPH plan/audit absent locally |
| Native Torch CPU contract | NOT_RUN: Torch/Transformers absent; a dedicated test is supplied |
| GPU numerical validation | GPU_NUMERICAL_VALIDATION_NOT_RUN |
| GPU E2E | GPU_E2E_NOT_RUN |
| GPU pilot approval | Not granted; specification and prerequisite gates only |

The local validator records each test name, pass/failure, dependency skip and
source hash in its sealed output. **32 required tests and 118 existing regressions
passed (150 total, zero failures)**. Eight existing native tests and the new
native CPU test were skipped because Torch/Transformers are absent.
The one historical job-shell execution test is explicitly excluded to honor the
no-jobs restriction. Actual loopback CPU UD tests execute and pass with socket
permission. No fixture metric is a NATURAL result.

## Original pipeline, before this change

These are unchanged historical locations, inspected before implementation:

| Stage | Existing location and behavior |
|---|---|
| TinyBERT/cluster assignment | `r5_subsequence_generation._generate`, line 123 onwards; actual `index.assign` before `owner.begin` |
| Candidate discovery and non-overlap | `r5_mstage_bridge.MStageShadow.begin`, line 51; actual resident gram/cluster lookup, sharing/physical checks, at most 64 examined candidates; `select_nonoverlapping` chooses spans |
| Position/causal restriction | `r5_subsequence_physical.PhysicalOwner.begin`, line 39; EXACT filters to identical absolute spans and identical token histories through window end; APPROX requires its existing explicit opt-in |
| Layer input proof | `PhysicalOwner._project`, line 87; q-role gathered target proof compared to stored layer input proof for EXACT; k/v use the pinned shared-input decision |
| Source authentication | `PhysicalOwner.valid`, line 63; strong resident entry/metadata/tensor proof and source ownership |
| Native projection / CPU LoRA | `_project`, line 87, and `r5_sparse_qkv.SparseSession.correction`, line 108; native Linear on fresh gather, actual CPU deltas, FP32 addition/cast |
| Gather/scatter | `r5_sparse_qkv.gather`, line 15, and `scatter`, line 21; full target output reconstructed, no attention/FFN/past-KV transfer |
| Policy/completion | `PhysicalOwner.complete`, line 142; actual Impact, CHU after completed consumption, original F/A/I/S insert/evict, PBR every 100 adapter arrivals; `_generate.complete` obtains real UD ACK first |
| Request scope / TOCTOU | `r5_subsequence_integrity.RequestValidation`, line 44; request/epoch/registry/entry/container/storage pins, private authenticated consumed snapshots, expired grants after completion/abort |
| Evidence / audit | generation step values, wire/UD ledgers and physical records; `r5_subsequence_audit.replay_projection`, `replay_validation`, `audit_request`, and NATURAL audit compare source bytes, construction, policies, logits and frozen numerical gates |

Old auditors assume the old entry layout and fresh/reused work partition. They
must not be used to approve the new schema. This change does not reinterpret any
historical audit or experimental result.

## Isolated implementation

New files:

- `r5_layer_input_physical.py`: `LayerInputOwner`, `VerifiedCache`, immutable
  `InputBytes`, scoped validator extension, completion-channel adapter, and
  explicit `generate_verified` adapter around unchanged native generation.
- `r5_layer_input_arithmetic.py`: pinned arithmetic contracts.
- `r5_layer_input_audit.py`: independent projection-value primitive and read-only
  sealed Phase 2 static reader.
- `scripts/156_r5_layer_input_verified.py`: CPU-only tests/static-report CLI.
- Three `test_r5_layer_input_*.py` files: portable operator/value tests plus
  optional Torch CPU Linear/real-UD test. No production fixture flag exists.

The new owner calls the real `MStageShadow.begin` directly, without calling
APPROX or changing historical mode dispatch. It retains resident discovery,
original non-overlap selection, semantic assignment from the existing generator,
per-adapter ownership, sharing, entry/candidate bounds and actual F/A/I/S policy.
An explicit `mode=LAYER_INPUT_BYTES_VERIFIED_V1` is required. Passing APPROX,
EXACT, or `allow_approximate=True` to this owner rejects; passing the new mode
to the historical owner also rejects.

At every layer **and role**, the new path validates the current request, pinned
arithmetic, source scope/integrity/provenance and actual gathered target inputs.
It compares shape/dtype/hash **and literal canonical input bytes** with the
authenticated immutable source input snapshot. Each role's actual full input is
also compared with q's input: the bare-OPT identical-input contract is observed,
not assumed. A different k/v input aborts as unsupported, preserving the sparse
protocol's shared-QKV-input requirement. Early-layer acceptance never grants a
later layer. Unsupported dtype/device/shape/layout/callable/weights/backend
rejects. Normal input or guarded-output ineligibility computes fresh rows.

Native output verification can accept/reject each role separately; its CPU LoRA
request still computes all roles/all rows, so role-specific acceptance does not
change wire coordinates. No attention result, FFN value or source past KV is
transferred. The existing generator uses the current target's native attention
and independently initialized request past.

Each decision and consumed span is labeled A/B/C/D using the original disjoint
definitions. Results separate actual cached projection rows consumed by category,
actual native/CPU rows computed, verification-reference rows, and rows skipped.
A full-shape reference may overlap cached-consumed output rows; this overlap is
explicit, and cannot be counted as saved work. Zero eligible NATURAL rows remains
a valid future result.

## Safety, admission and provenance

Only completed sources created by the new mode under the same arithmetic binding
are eligible. Historical EXACT/APPROX sources are not imported or promoted.
Sources must be requests with **no cached tensor consumption anywhere**; requests
that consumed a cached row cannot admit new roots, even for a different window.
This conservative request-level transitive restriction can reduce residency; it
must not be hidden when interpreting NATURAL outcomes.

Source input snapshots are immutable CPU `bytes`, pinned to the resident entry,
and authenticated against the layer input proofs in the strong admission hash.
They are checked again during request-scoped validation. New input payload and
contract metadata bytes are added to the cache entry's logical S **before** the
unchanged admission/eviction policy runs. This preserves policy formulas and
capacity accounting; the new storage cost can legitimately change decisions.
Actual QKV tensor bytes and resident input bytes are reported separately.
Eviction/disposal clears their bindings. At 32 layers/hidden 2560/FP16/w3, inputs
add 491,520 bytes per entry, 30 MiB for 64 entries per adapter, plus bounded
metadata. Stored QKV remains 90 MiB/adapter at 64 entries. The 512 MiB limit and
64-entry limit remain enforced; no capacity increase is made.

`CompletionChannel` binds actual authenticated `READY`/completion digest to the
active request; source admission requires that ACK. Invalid completion, model
failure or abort grants no admission. Unique invocation IDs remain enforced by
the original bridge's seen set and sparse-parent ordering. An aborted request ID
cannot be retried under the same ID. Checkpoints do not dispose owner state;
request authorizations still expire between invocations. CHU only follows rows
that actually reached output construction after full successful completion.
PBR remains 100 per adapter; a 50-total-request dataset cannot exercise it.

The existing single-owner-thread/ordered-stream/no-external-concurrent-writer
threat model remains explicit. Input byte buffers are immutable; QKV consumption
uses existing authenticated private snapshots. Integrity failures abort instead
of silently falling back. Unconsumed mutable tensor bytes are not continuously
rehashed after authorization, as in the original scoped threat model.

## Arithmetic guarantees and limitations

`TEST_FIXTURE_ONLY_ROW_LOCAL_F32_V1` accepts only fixture requests and exact
`ArrayTensor` CPU FP32 inputs. The known fixed row-local scalar Linear/bias and
existing scalar CPU LoRA read canonical FP32 inputs with fixed reductions and
casts; their rows do not depend on the number of other rows. The specific bound
kernel/matrices/bias/algorithm must match. Mixed and empty/all-reused row sets are
tested. This contract does not certify Torch, FP16, CUDA, GEMM or OPT behavior.

`FULL_SHAPE_REFERENCE_OUTPUT_GUARD_V1` binds actual bare eval Linear weights/bias,
layout/dtype/device, model/tokenizer/adapter/LoRA identity, backend fingerprint,
rank/scale and persistent CPU UD session. It verifies CPU LoRA FP32 precision and
scale receipts. It computes full target Linear and real CPU LoRA on all rows,
adds/casts exactly as the native fresh path does, and only substitutes a cached
row when both input bytes and current full-shape output bytes match. Final
assembled output bytes must equal that paid reference tensor. This establishes
an internal per-call output identity; it does **not** prove independently run
model activations/logits bitwise equal across A/B/C.

Weight fingerprints and backend checks, input transfers/proofs, full reference,
snapshot authentication, output comparison/scatter, persistence and actual
attention/policy are charged overhead. They are not free posthoc validation.
Native work skipped is zero. A shape-dependent scalar correctness fixture
demonstrates equal inputs with differing source/target outputs safely falling
back. No assumption of shape-independent GEMM rounding is made. A native
work-skipping contract remains unestablished and must not be enabled by changing
a mode name, boolean or threshold.

The new independent value primitive reconstructs strong admission metadata from
completed source projection records, verifies request/epoch/ownership/token
windows/input bytes/private snapshots/category, replays addition/cast/scatter,
and distinguishes reference work from skipped work. Tests tamper with source
completion/provenance, input/output evidence, auth roots and counts. It is a
**projection value primitive**, not a complete wire/policy/GEMM/full-model auditor.
The later GPU audit must additionally verify inventories, both wire endpoints,
actual CPU LoRA and base GEMM evidence, policy history, all activations, full-vocab
logits, last-token KL, argmax, generated IDs/text and the frozen tolerance gates.
Matching text or primitive PASS cannot override numerical FAIL.

## Real-data feasibility and SERAPH CPU command

Qasper, 2WikiMQA and MultiNews B/C candidate counts, target windows, verified
document/document counts, distinct-sample-only and first-occurrence counts are
**BLOCKED locally**. No equality rate or resident HIT count is estimated.
The reader checks both sealed trees, plan digest/session shards, the real audit
PASS and its exact plan-inventory link, then extracts the prior authorized static
metrics. It does not reseal inputs or rerun a source-pinned historical loader
under new code. This is extraction of existing verified static evidence, not a
new model numerical validation.

One cluster/gram owns one resident entry, potentially at an unhelpful position or
with incompatible inputs. Different static source pairs therefore compete for a
single resident key. F/A/I/S, 64-entry/candidate scan limits, non-overlap selection,
sharing, source-root restriction and numerical contracts can remove all offline
opportunities. B may have equal early inputs; later causal attention can change
them. C generally changes learned absolute-position inputs. Only actual runtime
bytes can establish either case. D is still reported separately, not merged into C.

With the existing CPU environment and new files present on SERAPH:

```bash
cd /data/khuss/repos/GP_semcache
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=42

python scripts/156_r5_layer_input_verified.py cpu-validate \
  --phase2-plan results/r5/i5r/d5_natural/continuous_phase2_seraph_plan_02 \
  --phase2-audit results/r5/i5r/d5_natural/continuous_phase2_seraph_cpu_02 \
  --require-real-data \
  --output-dir results/r5/i5r/d5_natural/layer_input_v1/cpu_seraph_01
```

Use a fresh output path. This runs portable tests, relevant existing regressions,
the native CPU contract test when dependencies exist, and static extraction;
it does not load OPT-2.7B or execute CUDA/Slurm. Source hashes, per-group logs,
`phase25a_validation.json`, `nonprefix_eligibility.json`, Markdown report and tree
inventory are produced. Missing real inputs yield BLOCKED (exit 2 when
`--require-real-data` is supplied); invalid seals/bindings yield FAIL. Dependency
skips are explicit and never GPU validation PASS. Unsafe output overlap rejects
before writing into sources.

[The separately reviewed Phase 2.5B pilot specification](R5_D5_PHASE25B_PILOT_SPEC.md)
lists the prerequisite numerical/resource/evidence work. It is not an executable
runner or GPU approval. Phase 2.5 numerical correctness remains incomplete until
the native CPU prerequisites and future GPU audits pass.
