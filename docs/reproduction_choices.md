# Reproduction choices and unresolved details

Primary source: supplied local SemCache PDF, DOI
10.1109/INFOCOM59046.2026.11571717. Page references below use PDF page numbers.
This ledger distinguishes implementation decisions from paper statements.
Defaults in Table I (p. 6) and §V-A match the requested YAML. No performance
numbers in the paper are used as synthetic measurements.

| Topic | Paper evidence | Milestone 1 reproduction choice / status |
|---|---|---|
| Repository layout | No software layout prescribed | Use existing repository root; preserve supplied PDF and avoid a nested duplicate project. |
| Model identity | OPT-6.7B FP16 (§V-A, p. 6) | `facebook/opt-6.7b`; configurable revision `main` is exploratory, not immutable. Record requested/resolved revisions. Default CPU and local-only loading are operational choices. |
| Architecture | QKV + attention equations (§III) | OPT only, raw biased linear projection outputs before Q scaling/head splitting. Capture actual module inputs, including OPT's own normalization behavior. Attention execution is deferred rather than substituting paper diagrams for architecture-specific order. |
| TinyBERT | TinyBERT in Eq. 7; checkpoint absent | Configurable `huawei-noah/TinyBERT_General_4L_312D`, masked mean including non-padding special tokens, max length 512, no vector normalization; revision recorded. Checkpoint has not been loaded in this environment. |
| Semantic input bridge | Eq. 7 takes base embedding h0 | Available text backend re-tokenizes raw text; it is NOT the unspecified base-embedding bridge and does not establish the paper's privacy claim. Base/encoder embedding dimensional alignment is unresolved. |
| Clustering initialization | Eq. 8 assumes existing centroids | First C warmup vectors, count 1 each; insufficient warmup raises. No k-means randomness. Lowest cluster ID breaks equal-distance ties. Caller-supplied fixture centroids use zero counts. |
| Cluster update interval | Eq. 9 gives online mean; Table I says 100 queries | Assign against frozen centroids during each global batch of 100, then apply Eq. 9 to buffered vectors with their original assignments. No cluster ID remapping or cache migration. `update_interval=1` gives online updates. |
| Windows | Eq. 10 uses an ambiguous end bound | Token-ID windows include the final valid window: zero-based starts 0 through n-w, end exclusive. No windows for n<w. This is the conventional sliding-window interpretation. |
| Match policy | §IV-A/B gives cluster and position indexing but no identity test | `exact_token_ids_within_cluster`; key is (cluster ID, token tuple), source position retained as metadata. Position is not an equality constraint. One model/tokenizer namespace per pool. No semantic threshold matching. |
| Position/context/adapter dependence | Fig. 2 motivates approximate similarity | Exact token equality does not prove equal QKV. No cross-query numerical correctness claim, no actual reuse in demo. Cached base versus LoRA versus combined QKV scope remains unresolved; demo metadata is explicitly unmaterialized. |
| Overlap | §IV-C specifies non-overlapping retrieval | Demo counts all window lookup events and the union of matched positions separately. Actual reuse is zero. A non-overlap selection policy must be chosen before integration. Merge helper rejects overlapping destinations and missing tokens. |
| Admission frequency | Recent appearances, unspecified horizon (§IV-B) | Configurable trailing 100 global queries, counting occurrences, including current request. Caller supplies frequencies to GlobalCache. Count actual reuse separately for eviction. |
| Frequency after insertion | Eviction F counts reuse | Newly admitted entries start at zero reuse count. No artificial admission hit. All request lookups precede insertion. |
| Normalization | F/Fmax etc., maxima across cache (p. 5) | Current pool plus candidate for admission, current pool for eviction; no historical all-time maxima. Cold-start zero maximum maps to zero. Candidate admission F uses appearance counts throughout its normalization population. Tests cover strict threshold and zero maxima. |
| Age/ties | Time since access; largest eviction score | Monotonic query index as configurable supported clock unit, no wall-clock dependence. Lexicographic (cluster, tokens) tie break. Candidate participates in overflow eviction and may itself be evicted. Oversized blocks are rejected without evicting useful entries. Duplicate insertion is rejected. |
| Missing impact | Sum of attention norms (§IV-B, p. 4) | Entry I is null when unknown. Index-demo scoring uses zero contribution; this is not an observed zero norm or a paper-equivalent impact ablation. Attention provider is absent and updater execution raises. |
| CHU initialization | Moving average defined for existing I | If a provider supplies a first observation for unknown I, initialize directly from it; otherwise use rho*I+(1-rho)*Inew. |
| PBR | Eq. 13 conditional mean over most recent Lambda cluster queries, low load | Provider contract returns conditional mean or None (retain old value). Per-cluster count multiple triggers scheduling; low-load scheduling and observation storage are deferred. Table I's Lambda serves as both horizon and interval for this interface. |
| Logical capacity | 20 GB | Decimal GB (10^9); optional tensors independent. Footprint baseline is tokens × layers × 3 × hidden_size × dtype bytes. Metadata overhead excluded. No GPU allocation for logical bytes. |
| Physical bytes | Contiguous QKV described (§IV-B) | Optional per-layer Q/K/V tuples, storage-pointer deduplication; not yet contiguous row-major packed blocks. Physical bytes mean live storage owned/referenced by entries, not process/GPU peak. |
| Demo | No controlled fixtures prescribed | YAML manual whitespace-token queries, vocabulary sorted once over fixture corpus, explicit two-dimensional embeddings and 20 centroids. This is an indexing test, not TinyBERT/MultiWOZ evaluation. |
| Results | Paper reports latency, memory, BLEU | Null unimplemented metrics; per-metric measured/simulated source map. `model`, rank and user count describe configured targets when `model_loaded=false`. Demo dataset/tokenizer/encoder identify fixtures explicitly. |
| Timing/seeding | Full implementation details absent | Fixed seed 42, evaluation/inference mode, deterministic Torch algorithms, synchronized CUDA timing. Inspection total includes forward/hooks/copies/assertions, not projection-only latency. No GPU determinism claim until run on target hardware. |
| Dependencies | HF PEFT reported | Lightweight PyYAML core; optional Torch/Transformers, no PEFT until adapter implementation. Version ranges are compatibility intentions, not a validated locked environment; record installed versions before experiments. |
| Dataset preprocessing | Adjusted MultiWOZ/CoQA/SNIPS queries (§V-A) | Deferred: dataset version, splits, economic/health/weather adjustment, references, ordering, leakage controls and personalization procedure unresolved. No invented dataset pipeline. |
| LoRA training | Rank and Q/K/V targets, HF PEFT | Deferred: adapter data, optimizer, objective, epochs, alpha/scaling, dropout, and distribution across users unresolved. No trained-adapter claims. |
| BLEU | Scores reported without full protocol | Deferred: corpus vs sentence aggregation, tokenizer, smoothing, reference construction and implementation must be recorded. |
| Quantization | LLaMA-7B INT4 | Unsupported: quantizer, calibration, grouping and kernel configuration unresolved. |
| Cost model | Eqs. 18–20; bandwidth 200 Mbps | Abstract interface only. Element-to-byte/bit conversion, compute capacities, overlap schedule, generation length and overhead calibration require explicit choices. No latency estimates yet. |

Unsupported policy values should fail rather than silently change semantics.
Some knobs are intentionally future-facing configuration (e.g. simulated user
count); they do not imply that the corresponding experiment has run. The
`configs/sweeps/planned.yaml` file records future axes only.


## Milestone 2 implementation choices (2026-09-13)

The paper target remains OPT-6.7B FP16. OPT-125m FP32 is an explicitly separate
development configuration. No Fig. 6 implementation or output-quality reuse
claim is added. The existing semantic, admission and eviction formulas remain.

- Capture: loaded OPT projection-module forward hooks, one context per forward;
  actual module inputs include the model's own normalization. Direct independent
  module forward checks run before the captured result is accepted. Float64 CPU
  metric reductions do not change model projection precision.
- Shapes: raw `[batch, tokens, hidden]`; separately described head view
  `[batch, heads, tokens, head_dim]`. Raw Q precedes attention scaling. Loaded
  module dimensions are checked, not inferred from a model name.
- Version scope: Transformers OPT split projection modules are required. Installed
  package versions and requested/resolved revisions are saved; no compatibility
  claim across the entire declared version range has been validated here.
- Controls: deterministic full-prompt candidate tokenization and exact sliding
  windows, excluding special IDs. A enforces equal prefix/position/content;
  B equal position/content with differing prefix; C differing prefix/position
  with equal content; D equal prefix/position with differing content. C changes
  two factors and is not a pure position-effect estimate. No raw-text offsets.
- Physical storage: `from_tensors` clones detached, compact per-layer blocks onto
  CPU by default or explicitly selected CUDA. Existing optional tensor constructor
  detaches tensors; it may retain aliased storage, whose full bytes are counted.
  Logical-only entries allocate no tensor. Logical capacity policy is unchanged.
- Demo: controlled fixture semantic vectors and existing IntentClusterer provide
  a deterministic cluster-scoped physical lookup; this is not TinyBERT validation.
  Stored base projections are compared to fresh projections, never injected.
- Metrics: one row per pair/layer/QKV, full token provenance in CSV and JSON;
  cosine and absolute differences plus reference-relative L2. Zero-reference
  nonzero-candidate relative L2 is null. No averaging across layers before saving.
- Scope: cache retrieval, representation similarity and safe inference reuse are
  separate claims. LoRA, attention replacement, generation quality and experiments
  remain deferred. No artificial LoRA tensor values are generated.

Milestone 1 logical pipeline: complete. Milestone 2 capture, physical storage and
cross-query probe code: implemented, real-model validation pending. Validation:
16 tests passed, 2 tensor-dependent modules skipped; compile and offline logical
smoke passed. Torch/Transformers unavailable; model checks exited explicitly.
No actual OPT forward, numerical projection parity, physical tensor MISS → HIT,
or cross-query measurements have been observed here. Safe inference reuse:
**NOT YET CLAIMED**. README contains the exact SERAPH validation commands.


## Milestone 3 reproduction choices (2026-09-14)

The supplied paper §IV-C, Eq. 14 describes fetching and combining QKV before MHA.
This milestone validates a controlled substitution mechanism, not the paper's
complete multi-user LoRA system or its computation-skipping optimization.

1. Injection point: base_raw_unscaled_linear_projection, the actual OPT
   q_proj/k_proj/v_proj outputs before head reshape/Q scaling. Fresh outputs are
   cloned and only the selected window is overwritten. One layer/forward, batch 1;
   detached device/dtype conversion, inference mode and finally-based hook cleanup.
2. Shared A/B/C/D full-prompt fixtures are development probes, not paper workloads.
   Exact token matching within a fixed controlled cluster remains a reproduction
   choice; the paper does not specify this exact identity test.
3. D is invalid for reuse. Its source key is fetched explicitly for stress injection;
   the CSV labels this separately from a B-window exact-token HIT.
4. The existing physical cache owns CPU copies during development. Window indices
   are zero-based/end-exclusive; source indices in the injection API address the
   supplied cache payload, whereas CSV source indices address the original prompt.
5. No cosine or output threshold defines safety. The default 1e-6 absolute error
   check enforces A identity, B layer-0 identity and causal prefix integrity only.
   A failed invariant aborts; B/C effects otherwise remain measurements.
6. OPT-125m FP32 is development-only. The paper setting remains OPT-6.7B FP16;
   no quantization or large-model execution is added.
7. Preserve original PyTorch checkpoint loading (`use_safetensors=False`): the
   user-validated Transformers 4.57.6/OPT setup had tied-weight/meta-tensor problems
   with the converted safetensors checkpoint. This is compatibility, not a paper rule.
8. Deterministic CUDA requires `export CUBLAS_WORKSPACE_CONFIG=:4096:8` before
   Python. Preflight also accepts PyTorch's supported `:16:8`; it never disables
   deterministic algorithms or edits shell configuration.
9. Logit metrics reduce in CPU float64. KL is baseline || injected using stable
   log-softmax, averaged over target_start through the final position for suffix KL.
   Prefix error uses positions strictly before target_start; empty prefix is zero.
   No generation/BLEU/task-level quality validation is performed; all are deferred.
10. Each case uses a fresh existing GlobalCache with one physical entry and its
    existing admission defaults. No admission/eviction sweep or learned semantic
    candidate selection is introduced. Hooks compute fresh projections before
    replacement, so this measures output effects and cannot establish speedup.

Current local verification supersedes historical local dependency blockers above:
33 tests pass on CPU, including random tiny OPT forwards; 2 pretrained tests skip
without weights even with SEMCACHE_OPT_INTEGRATION=1. No pretrained OPT-125m or
CUDA Milestone 3 measurements were made here. User-reported Milestone 2 SERAPH
validation remains separate. Task quality and safe reuse remain unverified.
