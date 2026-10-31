# R5-I0–I3 integration

## Implementation map (before code)

Baseline: `076b354373adf9c2528d441fc83d0d95369df80e`, branch
`exp/raw-semcache-r4c`; development: `exp/raw-semcache-r5-integration`.
Existing untracked logs and R3/R4 results are preserved.

| Gate | New boundary | Reused implementation |
|---|---|---|
| I0 | versioned request, physical identity, event contracts | canonical SHA-256 serialization, prepared R3-A identity |
| I1 | metadata-only semantic index | TinyBERT/ControlledEncoder, IntentClusterer immediate Eq9, SubsequenceExtractor(w=3), ExactTokenMatcher |
| I2 | automatic controller, candidate validation, optional admission, synchronized invalidation | R4-C1 PrefixCache, R4-B1 MultiPromptCache, R3-B ExactRawCache and GlobalCache tensor ownership |
| I3 | sequential application/runtime, generation loop, independent audit | QualityDriver.c_forward, existing eager OPT forward, C1 ForwardOrder/CPU UD, real localhost Channel |

The M-series engine cannot directly serve this milestone: it switches adapter
by logical user, owns window tensors, executes prefill only and consumes attention
impact. R5 reuses its semantic primitives rather than its execution engine.
R4-C1's generation loop requires scheduled modes and mandatory source admission;
R5 supplies its own loop while preserving the validated forward implementation.
R4-C1's protocol still names internally selected cold sources `COLD_MISS` and
validated prefixes `PARTIAL_PREFIX_HIT`; these names are produced after discovery,
never supplied by a workload oracle. No R4 source modification is planned.

## Contracts and architecture

Request arrival → model tokenization → CPU semantic text encoding (separate
tokenizer) → cluster assignment/update → bounded cluster + exact-window candidate
lookup → physical prefix/namespace/seal validation → automatic HIT/MISS →
distributed prefill → autoregressive decode → verified completion → optional
full-input source admission/eviction → final output and application metrics.

Admission follows generation completion because existing C1 source grants require
the verified completion receipt. Prefill tensors are retained until that receipt;
Decode tensors are never admitted as source prompt tensors.

Details, executable commands and measured gates are completed alongside the
implementation and recorded in `R5_I0_I3_VERIFICATION.md`.

## Request and cache identities

`R5_REQUEST_V1` requires request_id, user_id, adapter_id, input_text, token_ids,
attention_mask, positions, generation_policy, dataset, sample_id and
request_sequence_number. It also binds the input UTF-8 digest, model-token-ID
digest, full physical namespace, fixture scope and canonical request SHA-256.
IDs are bounded ASCII path-safe identifiers and cannot repeat within a controller
or application run. The controller accepts only its physically prepared namespace.
Masks are all ones, positions start at zero, batch is one, prompts <=512 tokens,
and prompt+generation must fit the loaded positional limit. Unsupported masking
or prior attention context is rejected explicitly.

`R5_PHYSICAL_IDENTITY_V1` contains model ID/immutable revision/base-weight hash,
tokenizer source/revision/vocabulary/config hashes, adapter ID/**matrix-byte hash**,
dtype/layers/hidden/rank, storage device/GPU UUID, installed eager OPT contract
hash and attention semantics. Real identities come from fresh R3-A preparation
and the actual loaded tokenizer, never a new identity string. The source entry's
sealed R4 context additionally binds exact tokens, mask, positions, prepared
manifest, source implementation hashes and endpoint. `r5_ownership.jsonl` binds
that context to source request/user, completed Raw-KV proof, individual Q/K/V
content digests/shapes/dtypes/devices, seal, physical bytes and admission clock.
Admission and access clocks are recorded in index metadata and HIT/reuse events.
The seal's completed source is independently linked to UD completion controls,
source logits/tokens and the full prefill evidence.

Logical user equality is never a reuse condition. A User_B request can consume a
User_A source with the same verified user_a matrices. Another adapter ID is
unsupported even if an operator asserts its matrices are identical. Changing an
ID string cannot prepare an adapter. The explicit negative probe does no model
forward, is labeled separately, and is excluded from successful generation counts.

The model tokenizer uses add_special_tokens=False and has its own immutable OPT
vocabulary/config identity. TinyBERT independently re-tokenizes input text using
the existing CPU float32 masked-mean encoder (max_length512). Its model ID,
operator-supplied immutable revision, resolved revision, full state hash and
vocabulary hash are recorded. The legacy TinyBERT configuration uses `main`, so
R5 deliberately requires `--semantic-revision`; it does not invent a pinned
TinyBERT commit. Missing offline weights fail clearly. Controlled vectors and the
mod5 TinyTokenizer exist only behind explicit TEST_FIXTURE_ONLY.

## Semantic discovery and ownership

The pilot uses `configs/r5_integration.json`: one first_k cluster, immediate Eq9
online mean updates, update interval100 (relevant for buffered mode), exact w=3
windows, <=64 distinct candidates. The first request initializes the centroid
from a real encoder vector. Each request records the vector, pre/post centroid
state, counts, distance and update diagnostics. One cluster is a controlled pilot
configuration, not a claim of improved semantic recall. The index supports
multiple explicitly warmed clusters; tests cover different-cluster exclusion and
the existing buffered policy. Clusters are not silently retrained or reassigned
for admitted entries.

Candidate postings map `(cluster, three exact OPT tokens)` to metadata entry IDs.
Repeated windows deduplicate entry IDs; candidates sort by descending full-source
length then entry ID. A same-cluster middle-window overlap is only a candidate.
Eligibility additionally requires a longer target, full exact source prefix,
matching positions/mask/namespace, completion and residency. The selected entry
then undergoes the original full QKV content and proof validation before any KV
is consumed. A semantic miss simply executes a normal full prefill.

`RawOwner` extends the existing C1 PrefixCache → B1 MultiPromptCache → R3-B
ExactRawCache/GlobalCache/CacheEntry ownership chain. This is the only persistent
Raw-QKV owner. The semantic index stores no tensors. Original compact tensor
cloning, actual unique storage byte accounting, content seals and the validated
Raw-KV proof are reused. R5 selects only the indexed candidate, avoiding a second
unfiltered PrefixCache.find scan. The inherited one-entry primitive admission
always accepts a fitting, complete source; no attention-impact measurement is
claimed. Outer eviction is physical LRU before allocation. The default capacity
is 256MiB; controlled smaller budgets are supported.

Eviction first invalidates metadata postings, clears tensor ownership and seals,
updates byte accounting, and sends the existing EVICT_PREFIX control to UD.
Both endpoints refuse an evicted grant. Candidate validation precedes LRU access;
actual reuse counters update after successful generation. Corruption raises
CacheIntegrityError, records INVALID_CACHE_INTEGRITY, invalidates the unsafe
entry, and stops the run. It is never hidden by a generic MISS exception handler.
OOM, communication, policy, model and evidence errors likewise remain failures.

Ordinary completed full-prefill requests can be admitted. Admission happens only
after all decode steps and the verified COMPLETE_REQUEST receipt. Oversized
requests, length512 sources and requests lacking complete standalone Raw-QKV
(including HITs) return generated output with admission denied. A HIT cannot be
promoted into a new longer source by splicing or slicing cached tensors. No
incomplete source or decoded continuation is admitted. LRUPolicy provides the
extension interface; an impact-based configuration is rejected while actual
attention-impact observations are unavailable (output_attentions=False).

On HIT, the existing attention_kv/construct_past path converts stored corrected
Raw K/V to request-owned attention-ready K/V, verifies its bytes and cache API,
and runs only the suffix in QualityDriver.c_forward. Cached Q remains stored,
sealed and counted but is not consumed. All suffix projection kernels still run;
the saved quantity is prefix projection-token rows and their real HIDDEN/DELTAS
TCP payloads. Every subsequent single-token Decode uses the actual returned KV.
No source output or logits are substituted. No Transformer forward is duplicated.

## Processes, generation and timing

The isolated ES owns bare FP16 OPT and the Raw cache. `r4c_dynamic_ud.serve` runs
in another PID with CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES empty, no full model,
and independently verified CPU adapter matrices. Existing bounded localhost
Channel framing and C1 ForwardOrder enforce layer order, tensor shape/dtype,
namespace, source registration, exact suffix inputs, Decode continuation and
completion. R5's automatic decisions are translated to the existing protocol
mode names only after discovery. Workload records have no expected mode/source.

Fixed greedy generation ignores EOS until the explicit budget. Quality greedy
generation stops at EOS or the budget. Full next-token float32 logits, selected
IDs, decoded text and EOS/length receipts are retained for every actual step.
The pilot has ten ordinary integrated requests, followed by ten independently
generated matched distributed no-reuse controls. Both paths use the same model,
weights, adapter, text and policy; semantic tracing is retained in both arms.
The baseline makes no admissions or physical reuse. It is sequential, ordered and
instrumented, so it is not an unbiased performance study.

Application E2E starts at arrival before model tokenization and semantic encoding,
and ends after generation, completion, admission and evidence storage. Model E2E
starts after discovery/validated KV construction and ends after generated text
and the UD completion receipt. Lookup/KV time is separately reported. TTFT exists
for both scopes. Prefill and Decode are recorded; tokenization, semantic encoding,
cluster, candidate lookup, eligibility, lookup/seal validation, KV construction,
admission/eviction, evidence blob writes, allocated/reserved/peak GPU memory,
occupancy/evictions and both payload/wire directions are identifiable.

Forward host timings include instrumented hashing, transfers, socket waits and
evidence. Tensor/logit blob-write time is an identifiable subset; JSON trace
writes and validation hashes remain inside host/application time. The nested
timers overlap and must not be added together. GPU event elapsed time includes
host/socket idle and is not kernel self-time. Tokens/s uses actual generated
tokens over model wall time. No uninstrumented or correctness-qualified speedup
is inferred. Localhost TCP is not a measured remote mobile network.

STRICT_AUDIT writes QKV, passive OPT component observations, prefill KV, logits,
and Decode digest chains. SCALABLE_EVALUATION uses identical inference semantics
with compact tensor digests and complete next-token logits. It cannot acquire a
full strict tensor-fidelity PASS. Disk budgets preserve partial evidence and
fail rather than silently downgrading profiles. Temporary source captures and
private prefix KV are separate from persistent Raw-QKV occupancy.

## Versioned artifacts and independent audit

Canonical JSON uses the existing sorted-key deterministic serialization and
SHA-256. Runtime clocks/PIDs are observations; analysis of an immutable run is
deterministic. `r5_output_manifest.json` hashes every output except itself.
Outputs refuse nonempty directories and input-directory overlap. Failures retain
stage, cause, worker stderr, partial ledgers and false gates.

| Artifact | Meaning |
|---|---|
| r5_requests.jsonl | sealed R5_REQUEST_V1 ARRIVAL and exactly one COMPLETION |
| r5_identity.json / r5_configuration.json | actual model/adapter/tokenizer/encoder/OPT and policy provenance |
| r5_semantic_events.jsonl | assignments, pre/post state, exact candidate windows, rejection reasons, selection and index insert/invalidate |
| r5_cache_events.jsonl | original owner event sequence, actual bytes, LRU, admission/denial, eviction, HIT/reuse |
| r5_ownership.jsonl | authoritative resident seals/specs/proofs captured after completed source admission |
| r5_generation_outputs.jsonl | generation, source, condition, traffic, latency, memory and skipped work |
| es/r5_tokens.jsonl / logits / tensors | actual forward/QKV/KV/positions/logits and instrumentation |
| es/es_wire.jsonl / ud_wire.jsonl / ud_layers.jsonl | independently recorded TCP exchanges and CPU corrections |
| r5_tcp_summary.json / r5_latency_summary.json | recomputed request-direction bytes and separate candidate/HIT rates/timing scopes |
| r5_fidelity.json | matched no-reuse comparisons, tokens/text/argmax/logit max_abs/relative-L2/KL, strict logits and tensor checks |
| r5_audit.json / r5_decision.json | independently replayed counts and distinct stage/qualification gates |
| r5_negative_controls.json | unsupported adapter probe; not a generated request |
| r5_results.json / r5_limitations.json | launch status, immutable source identity, execution scope and limitations |

The auditor independently replays clustering, index candidates, eligibility,
source completion, Raw-QKV ownership and LRU occupancy. It verifies prefix tokens
and namespace, physical hash/shape/byte seals, no evicted HIT, suffix-only forwards,
both wire ledgers and UD layer receipts, greedy logits/IDs, all completion counts
and metric arithmetic. Strict mode reconstructs every Decode KV from actual
prefill and new corrected Raw K/V, checks prefix bytes and component token shapes.
Artifact corruption and re-sealed fixture promotion are negative tests. Compact
digests alone cannot prove every tensor byte independently.

No invented QA/summarization gold is provided for the controlled integration
texts; quality is UNAVAILABLE. R4-C genuine LongBench gold/registry support remains
untouched for later I5 evaluation. Token agreement, numerical similarity, strict
bitwise equality, native PEFT parity and speedup qualification are distinct.
Native strict parity is historically unresolved; its flag and
correctness_qualified_speedup remain false. Fixture gates explicitly carry
TEST_FIXTURE_ONLY and never set GPU_PHYSICAL_EXECUTION_PASS.

## Reproduction and GPU pilot (operator actions only)

```bash
PYTHONPATH=src:tests python3 -m unittest \
  tests.test_r5_contracts tests.test_r5_semantic tests.test_r5_controller tests.test_r5_processes -v
python3 scripts/110_run_r5_semcache_e2e.py --phase fixture --test-fixture-only \
  --lifecycle-capacity --output-dir results/r5/my_unique_fixture
python3 scripts/111_analyze_r5_semcache_e2e.py --inputs results/r5/my_unique_fixture \
  --output-dir results/r5/my_unique_analysis
python3 scripts/110_run_r5_semcache_e2e.py --phase dry-run --output-dir results/r5/my_unique_preflight
bash -n jobs/r5_opt_semcache_e2e.sh
```

CPU process tests need permission to bind localhost sockets. Neither help nor
dry-run imports Torch, loads a model, creates UD, or starts a GPU experiment.
Use --evidence-profile SCALABLE_EVALUATION and --generation-mode quality for the
other supported evidence/termination policies. Without --lifecycle-capacity the
256MiB default remains available, but no eviction gate is claimed without events.

Before GPU use: commit the implementation on the R5 branch; ensure SERAPH paths,
conda semcache, Torch/NumPy/Transformers/PEFT and eager OPT/cache API are compatible;
cache real OPT and its tokenizer offline under /data/khuss/huggingface; cache real
TinyBERT at a recorded immutable commit; create logs/ before sbatch; provide >=6GiB
disk and the 2GiB run-evidence budget. The job targets moana-y4 and validates one
allocated RTX3090. It freshly runs the existing isolated R3-A preparation, records
this GPU UUID and actual base/adapter hashes, then validates the same GPU, versions,
model/dtype/seed/rank and tokenizer at runtime. Previous RTX2080 or another UUID
preparation cannot pass. UD must remain CPU-only. The pilot re-tokenizes every
input normally and validates actual source-prefix boundaries and capacity before
execution; an incompatible tokenizer boundary fails instead of changing IDs.

```bash
# Set an actual locally cached TinyBERT commit, not main or a fabricated value.
R5_ENABLE_GPU=1 R5_SOURCE_COMMIT=$(git rev-parse HEAD) \
  R5_TINYBERT_REVISION=YOUR_CACHED_IMMUTABLE_COMMIT \
  sbatch jobs/r5_opt_semcache_e2e.sh
```

This command is provided, never submitted by the implementation. GPU execution
also independently requires --enable-gpu-experiment and R5_ENABLE_GPU=1. The
pilot performs 10+10 controlled real-text OPT generations with G8; it is not a
natural traffic HIT-rate benchmark. Real TinyBERT/OPT/API/GPU verification remains
GPU_NOT_RUN until that job executes and the independent audit passes.

I4 extensions: physically prepare/verify multiple adapters, explicit namespace
routing/device lifecycle, richer measured policy inputs and logical user scaling;
only then concurrent scheduling with synchronized ownership. I5: genuine natural
traffic/datasets, gold metrics, semantic-vs-exact controls, larger cohorts,
independent repeat ordering and statistical timing/quality qualification.
