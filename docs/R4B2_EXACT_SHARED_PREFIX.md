# R4-B2: Raw-QKV-backed exact shared-prefix KV reuse

Branch: `exp/raw-semcache-r4b2`, based on `exp/raw-semcache-r4b`.
All implementation files are new; R3, R4-A, R4-B1 and historical results are
frozen. Job 152050 and the October 30 B1 GPU pilot are completed historical
results supplied by the user, not B2 measurements. Their Native strict parity
failures remain deferred. No job is submitted and no git push is performed.

## Architecture and feasibility constraints

Raw projections, attention K/V, HF `past_key_values`, and hidden states are
different objects. B2 admits only explicitly computed standalone source
prefixes, with zero padding, beginning at logical position zero. It does not
slice a longer source prompt into an allegedly validated shorter prefix.

The source executes all transformer layers through the distributed endpoint,
then completes ordinary G-token generation. Its prefill Raw Q/K/V are captured
before attention reshaping/scaling. Each layer's Raw K/V `[1,P,H]` is transformed
with `view(1,P,heads,head_dim).transpose(1,2)`. B2 compares those tensors bitwise
against the actual source model's returned attention K/V, checking every layer,
head dimension, dtype, device and sequence length. Admission requires that proof
and a completed source generation. Cached Q is retained and integrity-hashed
as part of the original Raw QKV object; **it is never consumed by target
attention**. Source logits/generated tokens are never used as target results.

For a partial HIT, B2 derives private attention-ready K/V copies from the sealed
cached Raw K/V, constructs a fresh supported in-request cache, and passes only
the suffix IDs to the actual model. All suffix layers execute QKV projections,
causal attention over prefix plus suffix K/V, output projection, FFN, residuals,
normalization and LM head. Learned positional embeddings start at P; masks cover
the complete current history. The next G-1 forwards each consume one generated
token using the newly completed target KV. Input/output KV shapes, hashes and
append/prefix preservation are checked at every step. No prefix hidden states
need to be restored: each layer's prefix K/V already encodes its causal prefix
history, while suffix hidden states are computed through the actual layers.

The target physically skips prefix embedding, prefix attention/FFN/LM-head work
and prefix projection **token rows**. Each suffix projection module still runs
once per layer: B2 does not claim 96 entire QKV kernel calls were eliminated.
Counts distinguish skipped prefix token rows from executed suffix token rows.
This is Raw-QKV-backed prefix-KV reuse, not partial QKV concatenation or a
relabeling of B1's full-prompt projection substitution.

## Loaded OPT implementation and HF API gate

Torch, NumPy, Transformers, PEFT and SacreBLEU are absent in the local CPU
environment. No installed SERAPH OPT version or CUDA behavior is claimed here.
The actual GPU worker records package versions, config, inspected signatures,
source text/digests of loaded OPT attention, decoder, positional embedding and
LM methods, and the installed cache module digest in `r4b2_opt_contract.json`.

The supported APIs are an actually observed legacy tuple/list cache, or an
actually observed HF `DynamicCache` exposing public `from_legacy_cache` and
`to_legacy_cache`. The factory signature/source digest and source-returned
cache type are recorded. Construction is tested against every derived tensor
before reuse; unknown/static/sliding/cache APIs fail with an explicit
`ARCHITECTURAL_BLOCKER`. No private HF fields are populated and no guessed API
is used. The runtime Raw-KV-to-actual-HF proof is mandatory regardless of version.

The pinned [HF OPT v4.56.2 reference](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/opt/modeling_opt.py)
illustrates unscaled K/V head reshaping and cache updates; its
[public DynamicCache conversion](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py)
provides the supported conversion mechanism. These are reference sources, not
evidence of the installed SERAPH version. Different GEMM shapes, attention
shapes and memory layouts can change FP16 rounding even when source causality
and KV transformations are valid. B2 attempts strict bitwise fidelity and
reports failures with metrics; it does not silently recompute a full target.

## Prefix cache and protocol

`PrefixCache` extends the frozen B1 manager, retaining actual R3-B sealed
`CacheEntry` tensor owners and admission policy. A bounded table scans explicit
source token sequences longest-first. A HIT requires exact prefix IDs at offset
zero, compatible full mask/positions, model/base fingerprints, tokenizer config,
adapter matrices, execution precision, device/GPU UUID, preparation/source
fingerprints and loaded OPT contract. The full-prompt key is never presented as
a prefix authorization: B2 has a separate versioned standalone-prefix context
and source KV proof. Overlapping explicitly admitted prefixes are supported;
the longest valid prefix is chosen. A caller demanding a different boundary is
rejected. Full-prompt HITs and empty suffixes are deliberately unsupported.

The tokenizer namespace includes actual vocabulary/special IDs, fast-backend
serialization (or slow GPT2 BPE merge ranks and byte encoder), tokenizer class
implementation digest, tokenizers version and relevant tokenizer settings.
Preparation and ES configurations must match exactly; an unknown backend that
cannot be fingerprinted fails closed.

The manager enforces shared byte capacity and validated-access LRU. It evicts
before cloning and clears tensor ownership/seals. Prefix proof digests and Raw
contents are checked on every lookup. Persistent occupancy counts Raw QKV once.
Persistent derived KV occupancy is zero: derived private K/V and HF cache storage
belong only to the current request and are reported separately. Model weights,
temporary evidence captures and CUDA allocator reserve are outside cached bytes.

The B2 v2 application contract uses the unchanged bounded R3-A Channel framing.
R4-A validators are not weakened. It explicitly distinguishes `FULL_PREFILL`,
`SUFFIX_PREFILL` and `DECODE`. Every HIDDEN frame carries parent, source cache
entry/request/proof, exact prefix IDs/length, suffix/target lengths, actual input
IDs, full target IDs, masks, positions, cache slots and cohort fingerprint.

The independent CPU-only UD registers a source only after observing all its
cold-generation layer exchanges. REGISTER/EVICT metadata uses acknowledged
HELLO/READY control frames. Every partial forward references a live exact grant;
first suffix prefill has `past_length=P` and `input_length=N-P`. Subsequent
decode transitions, layer order, context hashes and correction shapes are
enforced. The UD has no full model and no CUDA. It loads actual original FP32
LoRA A/B once and computes only the received suffix/decode correction rows.

## Controlled workload and qualification

Default GPU pilot: L128, fixed G8, P32/S96, P64/S64, P96/S32. Each condition has
one standalone source and two distinct targets. The actual tokenizer's prefix
text must round-trip exact IDs; complete target texts are tokenized separately
and validated to have the exact source prefix and different actual suffix IDs.
All 128 retained target IDs are real, with all-one masks and no BOS/EOS padding.
Short texts, collapsed suffixes or boundary changes fail preparation.

Each condition executes:

1. Full NO_REUSE source, Target A and Target B references.
2. Real source COLD computation, completed generation and admission.
3. Target A PARTIAL HIT, Target B PARTIAL HIT.
4. Unrelated NO_REUSE, then Target A PARTIAL HIT again from the same source.

Total: 24 requests and 192 GPU generation forwards; the fixture uses the same
token budgets with tiny scalar vocabulary and G3 (72 forwards). Two original
R3-A-input distributed triages additionally preserve frozen Native comparison
metrics for both cache settings. Fresh frozen Native preparation supplies the
original controls; matched B2 Native target generations are **NOT_RUN_DEFERRED**.
The primary B2 reference is distributed NO_REUSE for identical actual target
IDs/mask/policy. No B2 Native-equivalence claim is made.

The default 128 MiB Raw cache holds explicit 15/30/45 MiB sources (P32/64/96)
for the frozen FP16 OPT-2.7B shape. These are expected tensor sizes, not local
GPU measurements. Cache identity, admission, LRU, occupancy, eviction, lookup,
registration and validation times are retained. Derived request KV copies are
2/3 of that source Raw size before suffix execution. No storage is counted twice.

Negative controls cover absent/one-token-different/non-prefix-offset matches,
adapter/model identity, masks/positions, digests, unsupported boundaries,
corrupted proof/tensor, insufficient actual owner capacity, and evicted entries.
Rejections occur before model forwards and carry zero data frames; an integrity
rejection may send eviction control frames, recorded separately. Failed requests
retain actual target IDs, masks/positions, requested identity/boundary, stage and
reason; proof/tensor corruption has independent digest/snapshot evidence.
Metadata negative controls do not claim execution
of an alternative model or adapter. A bounded single-condition extension omits
the second-entry corruption control after its only entry is evicted; unit tests
cover corruption independently.

Strict equality includes IDs, text, every serialized next-token logit vector,
per-layer suffix QKV, actual full HF KV hashes and prefill component outputs.
The audit also proves cached source K/V equals initial and consumed target KV.
Numerical reports retain original R3 logit thresholds, max-absolute/relative-L2,
KL, argmax, changed-token counts, matched-history status and first observed
divergent layer/operation. Prefill tensor snapshots permit independent numeric
comparison; decode full KV is hash-only, with all decode logits/QKV retained.
First differing hashes are not alone a causal attribution.

TTFT/E2E/Prefill/Decode absolute and percentage deltas are paired against the
correct full reference. Cold prefix cost is compared to the same prefix source,
not to a longer target. Prefill and Decode directional wire/payload bytes and
registration control bytes are separate. Decode metadata can add wire overhead
even though decode tensor payload sizes are unchanged. CUDA memory allocation,
peak and reserve are separate outside-wall observations when available.
Eviction timing covers synchronized tensor-owner release and cache metadata;
control acknowledgements are included in insertion/E2E walls. Reuse events count
only the 64 consumed K/V projections, and explicitly mark cached Q unconsumed.

Only three partial comparisons/two independent full-target references per
condition and one block are present. A-after-U reuses A's reference. The frozen
eight-pair/two-block benefit requirement is not satisfied by pooling conditions.
Benefit flags stay false; diagnostic deltas are still reported. All GPU pass
flags stay false for fixtures. Strict bitwise failure keeps its own flag false;
it does not conceal the legitimate suffix-only execution or quantified failure.
`PRIMARY_PERFORMANCE_ELIGIBLE`, `correctness_qualified_speedup` and Native strict
qualification remain false. No semantic/approximate matching, compression,
CacheGen, remote-network or concurrent-serving claim is made.

## Commands and artifacts

```bash
PYTHONPATH=src:tests python3 -m unittest tests.test_r4b2_cache tests.test_r4b2_session tests.test_r4b2_processes -v
python3 scripts/104_run_r4b2_shared_prefix.py --phase fixture --test-fixture-only \
  --output-dir results/r4/b2/fixture_NEW
python3 scripts/104_run_r4b2_shared_prefix.py --phase analyze \
  --inputs results/r4/b2/fixture_NEW --output-dir results/r4/b2/analysis_NEW
```

Within a separately obtained allocated RTX3090 environment, the opt-in pilot
template is `R4B2_ENABLE_GPU=1 bash jobs/r4b2_opt_shared_prefix.sh`. It requires
partition `batch_eebme_ugrad`, one RTX3090, four CPU cores, 64 GB RAM, three-hour
maximum, conda `semcache`, offline Hugging Face files and the B2 work branch.
Create `logs/` before any future Slurm submission. The implementation task does
not submit. The job runs read-only preflight, fresh original-95 preparation,
frozen Native controls, B2 preparation/run and independent analysis.

The CLI phases are `dry-run`, `fixture`, `prepare`, `run`, `analyze`. `prepare`
takes `--base-prepared <fresh_95>`; `run` takes `--prepared <prepare>/prepared`
and `--enable-gpu-experiment`, with `R4B2_ENABLE_GPU=1`. A separately requested
`prepare --max-new-tokens 16` supports a bounded G16 extension, never an automatic
sweep. L64/P32 is another explicit bounded extension. Each run freezes its suite
and policy. Outputs must be new/empty under `results/r4/b2` and outside inputs.

Expected artifacts: `r4b2_results.json`, `r4b2_decision.json`,
`r4b2_fidelity.json`, `r4b2_latency_pairs.json`, `r4b2_latency_contrasts.csv`,
`r4b2_quality.json`, preflight/summary, prepared suite/source manifests, and
`es/` request/generation/token ledgers, full next-token logits, bounded tensor
snapshots, logit diagnostics, attempted/completed forwards, loaded OPT contract,
cache events, negative controls/failed requests, prefix registrations, strict
original Native diagnostics, independent ES/UD wire/layer logs, process
stdout/stderr and lifecycle records. Tensor snapshots are bounded to 4 GiB per
worker. Copies, hashes, snapshots, diagnostics and cache operations are inside
instrumented serving walls; these are not kernel-only timings. Persistent Raw
cache occupancy excludes temporary evidence copies and in-request KV.

Remaining unverified work: actual RTX3090/OPT-2.7B FP16 execution and cache API
compatibility on that runtime, numerical fidelity under changed GEMM shapes,
real latency/memory/quality measurements, and broader repeated timing samples.
Native parity resolution is separate work. Unknown HF cache APIs require an
explicit new verified adapter, not fallback. CPU scalar fixtures validate the
causal data path and protocol, not HF/CUDA equivalence or performance.

## Changed files (all new)

- `r4b2_cache.py`: versioned standalone-prefix identity, longest-prefix matching, sealed Raw storage and K/V-only reuse events.
- `r4b2_kv.py`: inspected cache API, exact head transformation, actual HF K/V proof, bounded tensor snapshots.
- `r4b2_session.py`: independent prefix-aware v2 execution and grant/order validators.
- `r4b2_dynamic_ud.py`: independent CPU correction worker with acknowledged prefix registration/eviction.
- `r4b2_generation.py`: actual suffix-only prefill, normal decode, finite/position/KV guards and timing/evidence.
- `r4b2_workload.py`: real-token workload, full tokenizer configuration fingerprints and bounded schedule.
- `r4b2_prefix_runtime.py`: physical workers, original Native controls, negative probes and lifecycle.
- `r4b2_evidence.py`: independent wire, computation, cache, Raw/KV and negative-control audit.
- `r4b2_analysis.py`: strict fidelity versus separate numerical, quality and paired latency reports.
- `scripts/104_run_r4b2_shared_prefix.py`: explicit CLI phases, preparation, failure preservation and opt-in gating.
- `jobs/r4b2_opt_shared_prefix.sh`: separately gated SERAPH RTX3090 pilot template.
- `tests/test_r4b2_cache.py`, `tests/test_r4b2_session.py`, `tests/test_r4b2_processes.py`: causal/protocol, negative, physical, numerical and bounded G16 controls.
- `docs/R4B2_EXACT_SHARED_PREFIX.md`: architecture, assumptions, commands, artifacts and limits.

The nine Python implementation modules above live in `src/semcache/edgelora`.
