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
| Cluster update timing | Eq. 8 assignment followed immediately by Eq. 9 online mean | Default `immediate_eq9`: assign against current centroids, then update only the assigned centroid and increment its count. Optional legacy `buffered` mode and its interval are `REPRODUCTION_CHOICE`, not paper-faithful timing. |
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

## Milestone 4: LoRA projection structure

This section supersedes earlier statements that all LoRA execution is deferred.
User-reported SERAPH verification covers **both Milestones 2 and 3**, including
pretrained OPT-125m CUDA capture/cache/injection controls. That is separate from
Milestone 4, whose local evidence is random tiny OPT on CPU only.

- **Paper-derived:** Eq. (2), §III-B defines base + user-specific LoRA Q/K/V;
  §V-A applies Hugging Face PEFT to Q/K/V; Table I sets rank 8. Eq. (17)
  counts `nd + 3nd` communication elements per sequence per layer. The supplied
  local PDF is the primary source. Its full deployment includes input/output
  layers on the UD; this milestone emulates projection exchange only.
- **Unspecified training:** learning rate, optimizer, epochs, exact user splits,
  alpha, dropout, initialization and personalized dataset construction are not
  sufficiently specified. No adapters are trained, no datasets are downloaded,
  and no BLEU or personalized-task quality is claimed.
- **Controlled fixture:** alpha=8, dropout=0, seeds user_a=101/user_b=202,
  initialization_scale=0.01 are reproduction choices. After PEFT creation, a
  separate CPU `torch.Generator` seeded per user generates float32 independent
  normal samples with mean 0 and standard deviation 0.01 for **both A and B**.
  Iteration order is increasing layer, q/k/v, then A/B, row-major tensor layout.
  Samples are copied to the actual adapter weight dtype/device. No LoRA bias is
  added. Default scaling is alpha/r=1; both values are configurable. PEFT
  constructor randomness is enclosed in `torch.random.fork_rng`; fixture values
  use only the local generators. Bitwise reproducibility is checked within the
  installed software/hardware environment, not promised across versions.
- **Execution:** eval + inference mode; configured nonzero dropout is disabled
  by eval exactly as in native PEFT. Training-mode decomposition fails. Use
  actual PEFT A/B/dropout/scaling and its input cast helper; add before casting
  the total to the base output dtype. Base-layer bias is included once. Merged,
  disabled, multi-active, DoRA/variant, transposed and quantized paths fail.
  Only the active user's LoRA parameters may be trainable; PEFT freezes inactive
  adapters on switching. All non-adapter parameters remain frozen.
- **Shared base:** one PEFT model, two adapters, no per-user base clone. SHA-256
  over contiguous CPU tensor bytes, shape, dtype and float64 norm for all Q/K/V
  weights **and biases** verifies creation/switching invariance. Tiny-model tests
  also check base parameter object identity and A/B parameter inequality.
- **External interface:** `load_external_adapter(base_model, local_directory,
  adapter_name='external')` loads local QKV-only vanilla PEFT adapters with base
  changes rejected. It neither asserts training provenance nor synthesizes it;
  callers must supply that provenance. Current scripts deliberately use
  `controlled_fixture`, `trained_adapter=false`. The external roundtrip unit
  test saves a synthetic fixture, not a trained adapter.
- **Logical ES/UD:** a single process on the same device computes the base and
  LoRA paths independently. CPU UD offload and actual networking are deferred.
  No host/PCIe timing is interpreted as ES–UD network latency. Full-forward hooks
  still compute native projections before replacement, so no speedup is claimed.
- **Communication:** one hidden tensor sent per layer, three delta tensors
  returned. `paper_comm_elements=4*n*d` is explicitly per sequence (batch one).
  `total_comm_elements=4*batch*n*d`; `measured_tensor_bytes` sums each actual
  tensor's `numel()*element_size()`, including potentially mixed dtypes. There
  is no bandwidth, protocol overhead or latency model. Decomposition CSV repeats
  the complete layer exchange on its Q/K/V rows; do not sum those three copies.
- **Probes:** fixed_hidden_input holds user_a's captured input constant for both
  users. full_user_forward runs the same complete query independently per user;
  later-layer base outputs may differ because hidden states differ. Cross-context
  A/B uses existing exact matched-token windows from full user-specific forwards.
  These observations do not establish safe reuse. No cache is used in M4.
- **Component scope:** base, lora_delta, total are explicit. Capturing a PEFT
  projection records `total_raw_unscaled_linear_projection`, before OPT head
  reshape/Q scaling. Future SemCache integration must identify attention-ready
  combined Q/K/V. Milestone 2/3 base-only cache semantics remain intact.
- **Tolerance:** default absolute and relative decomposition checks are 1e-6;
  full-forward additionally checks KL and last-token argmax agreement. These are
  implementation controls, never learned reuse thresholds. Mismatch aborts before
  writing results; tolerance is not automatically relaxed.
- **Version scope:** local validation uses PEFT 0.20.0, Torch 2.6.0+cpu and
  Transformers 4.57.6. The existing loader, deterministic algorithms and original
  PyTorch checkpoint loading choice are preserved. No CUDA M4 evidence exists
  locally. See [milestone4_status.md](milestone4_status.md) for measured status.

## Milestone 5: integrated prefill system

This section supersedes historical statements above that overlap resolution,
attention impact, observation history and the integrated pipeline are absent.
M2–M4 are **user-reported REAL-GPU VERIFIED** on SERAPH, including pretrained
OPT-125m, Torch 2.6.0+cu118 / Transformers 4.57.6 / PEFT 0.20.0. M5 has separate
local CPU evidence; it is not yet pretrained-model or CUDA verified.

| Concern | Paper definition / setting | Executable reproduction choice |
|---|---|---|
| Semantic encoding | TinyBERT, checkpoint unspecified | Existing `ControlledEncoder` for deterministic tests; existing optional HF masked-mean text encoder requires explicit nonempty checkpoint. The old arbitrary configured checkpoint is removed. The Eq. 7 embedding bridge remains unresolved. |
| Centroids | Eq. 8 nearest Euclidean; Eq. 9 incremental mean after every assignment | M7 defaults to `immediate_eq9`. `first_k` warmup centroids start with count one; explicit synthetic centroids may provide explicit counts. Two fixture clusters are development-only, not a universal paper C. Optional HF mode requires explicit dimension-compatible initial centroids. |
| Subsequences | Overlapping w=3 windows | LLM token IDs, zero-based end-exclusive spans; all valid windows, including any tokenizer special IDs, participate in M5. |
| Matching / index | Cluster-local subsequences, notation (c,p); equality unspecified | Exact token tuple within cluster; collision-safe `(cluster_id, token_tuple)` key. Source positions do not constrain target positions. One model/tokenizer namespace per engine. |
| Overlap | Fetch non-overlapping blocks | Sort target start ascending, semantic impact descending as utility, then lexicographic semantic key; greedy unique-position mask. Utility tie-breaking is a choice, not Eq. 11/12. |
| Block representation | Attention consumes combined Q/K/V, varying by layer | One semantic entry bundles distinct per-layer compact tensor tuples; `component_scope=total_qkv`, source user retained. No base/delta substitution. Entries admitted after mixed forwards contain the projections actually consumed, which may themselves include previous reuse. |
| Frequency / age | Admission appearance F; eviction reuse F; time since access A | Retain trailing 100 global queries of occurrence counters. Lookup counts hits without changing reuse F; only accepted/reused windows increment F and reset last access. One logical tick per query. Multiple reused occurrences of a key each count as reuse/CHU. |
| Normalization | Max-normalized metrics | Reuse existing helper: admission pool plus candidate, appearance F throughout that population; eviction current pool including newly inserted candidate, reuse F. Zero maximum maps to zero. No all-time maxima. |
| Admission / eviction | Eq. 11 strict score > .3; Eq. 12 largest score | Preserve tested policies and weights. Candidate evaluated with logical metadata; physical factory called only after admission. Candidate can itself be the maximum-score eviction victim. Oversized candidates denied without allocation. Same-score eviction uses lexicographic semantic key. Each miss key considered once per query, at leftmost occurrence. |
| Impact | Attention-weight norms over subsequence tokens/layers | Actual eager attention `[1,heads,query,key]`: L2 over each key row, **mean over heads**, sum over tokens and all layers. Float64 reduction. Not QKV cosine. |
| CHU | rho=.8 | Existing updater arithmetic; applied only to actual reused windows, using their current target span attention. |
| PBR | Lambda=100 recent cluster queries; conditional mean; low load | Bounded per-cluster history with ID/order, keys and actual impacts. Repeated occurrences of a key within one query contribute their mean, so each query has one indicator/observation. Explicit manual operation, no low-load scheduler; denominator zero retains old I. Historical `pbr_interval_queries` remains for M1 API compatibility; M5 uses `history_lambda` and manual trigger separately. |
| Mixed projection | Reuse cached tokens; compute unmatched base+LoRA | Inference-only batch-one local instance-forward context; native PEFT receives only unmatched hidden rows for each Q/K/V/all layers; gather/scatter fills full output. No installed source edits. Full projection hooks are rejected because capture validation would recompute cached rows. Context records passive CPU projections for candidate evaluation and restores forwards on success/failure. |
| Exact gate | No numerical implementation tolerance specified | Fixed absolute <=1e-5 and relative-L2 <=1e-6 projection and logit checks, KL <=1e-8 checks and argmax agreement; two disjoint cached windows plus unmatched rows. This is an identity-control tolerance, not a safe reuse threshold. Script 17 repeats the gate before approximate reuse, even if script 18 ran earlier. |
| Storage | Logical default 20 GB | Decimal 20 billion-byte budget, CPU physical storage by default. Logical size uses actual configured development QKV dimensions/dtype, excluding metadata overhead; no 20GB preallocation. Captured forward outputs are transient instrumentation, not admitted cache allocation or peak memory measurement. |
| Savings | 6n'd² base FLOPs; 6n'dr LoRA FLOPs; 4n'd communication elements per layer | Report all-layer sums, separately marked analytical. Communication bytes use one hidden dtype size plus three delta dtype sizes. No network or wall-clock speedup measurement. |
| Trace | Paper workload not used | Controlled health/weather strings, rank-8 untrained M4 fixtures. No safety/personalization claim; `safe_reuse_claimed=false` throughout. Local tiny-model CLI tests use a character tokenizer; pretrained scripts use the actual LLM tokenizer. |
| Decode | Latest-token-only matching after first token | Explicit singleton-key lookup API with separate target position. w=3 prefill entries cannot satisfy a singleton match. Decode cache population, past-key-values execution and the generation loop are deferred to M6; no pretend full algorithm coverage. |

M5 settings extend the existing top-level YAML keys instead of adding duplicate
nested versions of the same paper defaults. `semcache` contains storage, encoder,
fixture centroids, overlap and manual-PBR choices. Unsupported policies fail.
Scripts 15/16 are fixed numerical validators (load the selected config but use
explicit fixture values); script 16 overrides capacity locally to 20 bytes.
Scripts 17/18 use all transformer layers, no `--layers` subset. Main execution is
single-process logical EdgeLoRA, not a physical distributed UD/ES deployment.
Full dataset workloads, trained adapters, BLEU, network emulation and Fig. 6–11 /
Table II reproduction remain out of scope. Optional standalone HF probe 19 was
not needed; the pluggable HF interface is available, with no checkpoint tested.


## Milestone 6A: experimental foundation

M2–M5 pretrained OPT-125m SERAPH GPU verification is user-reported at the
start of M6A; earlier local-only M5 notes are historical. M6A adds explicit
source transformations and cost equations, superseding the earlier deferred
dataset/cost interfaces. See [paper specification](paper_experiment_spec.md)
and [workload reconstruction](workload_reconstruction.md) for the complete
PAPER_DEFINED / REPRODUCTION_CHOICE ledger. No hidden TinyBERT checkpoint,
adapter training recipe or generation protocol is chosen. Logical simulations
have unavailable attention impact; they do not fabricate CHU/PBR. Paper
references remain read-only comparison data and never tune the simulator.

FP32 exact-control uses abs<=1e-5 and relative-L2<=1e-6 for portability
across CPU BLAS/GEMM implementations. CUDA on tested SERAPH GPUs produced
exact zero difference. These are implementation-parity tolerances, not
cache-reuse safety thresholds.
