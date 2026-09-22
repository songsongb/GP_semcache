# SemCache reproduction — Milestones 1–7

**M9-A.2:** [Semantic Impact overhead audit](docs/M9A2_SEMANTIC_IMPACT_AUDIT.md)
adds opt-in blockwise/token-precompute/prefix diagnostics and separate strict-cost
recomposition. Normal inference is unchanged; measured speedups await SERAPH.

**M9-A.1:** [Strict ES base-only profiling](docs/M9A1_STRICT_ES_PROFILE.md) is now the primary
comparison path. PEFT-prefill proxy commands below are historical/exploratory only;
proxy totals are labelled `SIMULATED_PROXY_DOUBLE_COUNTS_LORA`.

**M9-A:** [Single-user UD–ES cost model](docs/M9A_SINGLE_USER_COST_MODEL.md)
consumes M8 artifacts and CPU calibration with explicit provenance; inference
behavior is unchanged. M8.5 is closed on user-reported SERAPH regression evidence.

**M8.5 update:** See [prefill paper alignment](docs/M85_PAPER_ALIGNMENT.md) for the current
reducer default, lookup-only state fix, configurable schedules and scope limits.
Earlier milestone defaults below are historical where they conflict.

This repository implements the foundation of **SemCache: Semantic-Aware Cache
Sharing for Efficient Multi-User LoRA-Adapted LLM Inference at the Edge**, IEEE
INFOCOM 2026, DOI 10.1109/INFOCOM59046.2026.11571717. The local PDF at the repository
root is the primary specification. This root serves as the requested
`semcache-repro/` directory; the supplied paper is preserved in place.

**Current status:** M1–M7 structural components are implemented.
M2–M5 are **user-reported REAL-GPU VERIFIED on SERAPH** with pretrained
OPT-125m. M6A is **CPU-TESTED** using synthetic datasets and random tiny OPT;
real datasets, pretrained M6A and CUDA M6A are **NOT YET VERIFIED**.
**No paper result is reproduced.** See [M6A status and commands](docs/milestone6a_status.md),
[M7 semantic architecture and commands](docs/M7_FULL_SEMANTIC_SEMCACHE.md),
[paper experiment specification](docs/paper_experiment_spec.md), and
[workload reconstruction](docs/workload_reconstruction.md).

| Milestone | Scope |
|---|---|
| M1 | Logical primitives |
| M2 | Real QKV capture and physical cache |
| M3 | Reuse injection and output effects |
| M4 | LoRA / EdgeLoRA decomposition |
| M5 | Integrated controlled SemCache prefill system |
| M6A | Dataset/workload, provenance, logical smoke and analytical experiment foundation |
| M7 | Actual TinyBERT, actual attention impact, CHU/PBR and tiny OPT integration probe |
| Future | Paper evaluation and full-generation validation |


SemCache caches **per-layer Q, K and V projection blocks** across users within
intent clusters. It does not cache final answers, and Hugging Face
`past_key_values` alone is insufficient. The OPT adapter exposes raw projection
outputs at the actual attention input. Context, position, layer history, and
user-specific adapters can change those outputs even for identical token IDs.
An index HIT therefore is not proof that using the cached projections is correct.

## Implemented

- YAML defaults from Table I and the simulation setup in §V-A.
- Pluggable semantic encoder, nearest-centroid assignment (Eq. 8), incremental
  means (Eq. 9), sliding windows and cluster-scoped exact token matching.
- Global logical cache with source positions, optional detached per-layer tensors,
  frequency, age, nullable impact, timestamps, admission (Eq. 11) and eviction
  (Eq. 12). Separate appearance counts and reuse counts.
- CHU arithmetic and provider-based PBR scheduling. An absent attention-impact
  provider raises explicitly; the demo does not execute CHU/PBR.
- Opt-in OPT loader and Q/K/V hooks, position extraction and ordered scatter
  merge. Unsupported architectures fail explicitly.
- Offline MISS → HIT demo, JSON/CSV schema, environment reporting and tests.

## Run Milestone 1

Python 3.10+ is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[test]'
python3 scripts/00_check_environment.py
python3 -m pytest -q
python3 -m compileall -q src scripts tests
python3 scripts/05_check_semantic_clustering.py
python3 scripts/06_check_subsequence_cache.py
python3 scripts/07_run_single_request.py
```

Every configurable script accepts `--config path/to.yaml`; the demo and OPT
inspection also accept `--output path/to.json`. The supplied model/dataset YAML
files are reference fragments; scripts load a **complete config**, without
implicit fragment merging. Copy `configs/base.yaml` when making a variant.

The controlled queries are `find a hotel in cambridge` and
`find a hotel in london`. A deterministic whitespace vocabulary and explicit
fixture embeddings assign both to cluster 0. Expected lookup events:

```text
request 1: MISS MISS MISS
request 2: HIT HIT MISS
logical bytes=7077888; physical tensor bytes=0
```

Three windows are admitted from request 1. The new request-2 window is rejected:
its appearance frequency is 1, while the maximum is 2; with unavailable impact
mapped to zero for scoring and equal sizes, its score is 0.25. The threshold is
strictly greater than 0.3. Two overlapping index hits cover four distinct token
positions; **actual reused tokens remain zero**. The JSON records lookups,
admission decisions, complete configuration and the unavailable impact status.
Results go to `results/raw/milestone1_demo.json` and `.csv`.

The default 50 users and rank 8 are configured targets, not an executed workload.
No generated output, BLEU, GPU latency, or system-latency estimate is fabricated:
unavailable fields are null. `metric_source` is a per-metric mapping with values
`measured` or `simulated`. A measured HIT is an observed dictionary lookup, not
an observed GPU speedup. Logical memory is simulated; optional tensor-storage
bytes are measured separately and do not represent peak GPU allocation.

## Optional OPT inspection

```bash
python3 -m pip install -e '.[models,test]'
python3 -m pytest -q
python3 scripts/04_check_qkv_projection.py --config configs/base.yaml
```

This command requires locally available `facebook/opt-6.7b` weights/tokenizer.
`local_files_only: true` prevents implicit downloads. Set paths/revisions and the
intended device in a copied config. Downloading weights is an explicit user
operation; no test downloads or trains a model. Pin immutable model, tokenizer,
and TinyBERT revisions for reportable experiments. Loader outputs include
requested and resolved revisions (when available), dtype, device and package
versions. The optional TinyBERT implementation is instantiated with
`HuggingFaceTextEncoder(**config['semantic_encoder'])`.

The inspection script hooks one configured layer during an ordinary OPT forward,
then independently recomputes Q/K/V from the captured attention-input hidden
states and asserts parity. It records shapes, tensor bytes and **total inspection
time**, which includes a full forward, copies and validation; this is not a
projection latency benchmark. Model evaluation and inference mode are enabled;
CUDA timing synchronizes before and after. Raw Q is captured before attention
scaling/head splitting. Tests use a random two-layer OPT, never OPT-6.7B weights.

## Paper specification and reproduction choices

Direct specification includes EdgeLoRA's UD/ES roles (§III), Q/K/V reuse (§IV-C),
nearest-centroid/mean equations, fixed windows, max-normalized admission and
largest-score eviction, CHU/PBR, and Table I defaults. These are kept separate in
`models`, `semantic`, `cache`, `edgelora`, and `simulation` packages.

[docs/reproduction_choices.md](docs/reproduction_choices.md) tracks the choices
not fixed by the paper: TinyBERT checkpoint and text interface, cluster warmup
and update timing, token identity and position indexing, cold-start normalization,
frequency horizon, logical memory units, and controlled fixtures.

A logical 20 GB pool means a 20,000,000,000-byte **budget**, not allocating that
much RAM/VRAM. Demo blocks describe all-layer Q/K/V footprints using explicit
OPT metadata; their tensors are absent. Physical accounting deduplicates shared
storage pointers, including views. Tensor storage and logical entry size must
not be mutated after insertion. Each pool belongs to one model/tokenizer
namespace and is currently single-threaded.

## Historical pre-M5 scope and execution blockers

The following list records earlier milestone boundaries. The M5 section below
supersedes overlap selection, integrated policy, attention impact and history
limitations; distributed execution and paper workloads remain deferred.

- Full reuse policy, non-overlapping multi-span selection, distributed
  layer-by-layer integration, and generation correctness. Controlled single-span
  projection substitution is implemented in Milestone 3.
- Physical UD/ES placement, communication, trained per-user LoRA execution.
- Attention-norm extraction, recent-query observation storage for Eq. 13,
  low-load PBR scheduling, and the OPT-embedding → TinyBERT bridge in Eq. 7.
- MultiWOZ preprocessing/personalization, user traces, adapter training and BLEU.
  Scripts 01–03 deliberately exit with an explanatory error.
- Eq. 18–20 cost simulation, physical profiling, baselines and experiment sweeps.
  `CostModel`/`EdgeLoRASimulator` are extension interfaces, not implemented results.
- GPT2 fused QKV, LLaMA and INT4 quantization. Figures 6–10 and Tables II–IV
  remain future milestones; planned YAML axes are not sweep implementations.

For actual OPT-6.7B execution, install PyTorch/Transformers, provision the model
and tokenizer with pinned revisions, and provide sufficient CPU/GPU memory
(the FP16 weights alone are roughly 13.4 GB, before runtime overhead). GPU
measurement requires working CUDA hardware. Full reproduction additionally
requires resolving the dataset, adapters and numerical-reuse items above.

Validation in the supplied environment: Python 3.10.12; offline unit tests,
compile/import checks and controlled scripts run. PyTorch/Transformers are
absent, so tensor and tiny-OPT tests are skipped; neither OPT-6.7B nor GPU timing
has been validated. See `results/raw/environment.json` for the environment report.

## Historical Milestone 2 local status and execution

- Milestone 1: logical SemCache pipeline — complete.
- Actual Q/K/V capture: implemented; real OPT execution and numerical validation pending.
- Physical Q/K/V cache: implemented; tensor tests and real physical hit pending.
- Cross-query similarity: A–D probes and raw result output implemented; measurements pending.
- Safe inference reuse — **NOT YET CLAIMED**. No cached tensor is fed into attention.
- Fig. 6 experiments and distributed/user-specific LoRA remain deferred.

Validation on 2026-09-13 used `/tmp/semcache-m2-venv`: **16 passed,
2 skipped modules**, syntax compilation passed, and the offline logical demo
passed. Torch and Transformers are absent, so scripts 04, 08 and 09 exit with
explicit missing-dependency messages. The opt-in integration invocation skipped
at module collection (pytest exit 5: no collected tests). No real OPT forward,
projection parity measurement, physical tensor hit, or cross-query measurement
occurred. PEFT and Matplotlib are also absent; PEFT is optional for this base-only
milestone. CUDA availability and VRAM are unknown without Torch. No numerical
CSV or figures were fabricated.

`configs/base.yaml` retains the paper target **facebook/opt-6.7b, float16**.
`configs/development.yaml` is a complete config for **facebook/opt-125m,
float32, CPU**. OPT-125m output is development evidence, never paper reproduction
results. Scripts also accept `--model-id`, `--dtype`, `--device`, `--revision`,
`--window-size` and `--layers 0 1 ...`. Omitting `--layers` captures all layers.
Weights load locally by default; `--allow-download` explicitly enables fetching.
No research script installs dependencies and no test downloads weights.

Run these commands on SERAPH from the repository root. Install Torch explicitly
using the machine's approved CUDA-compatible build; the command below uses
PyPI's default build. If SERAPH already has a managed Torch environment, activate
it instead of creating another one.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install torch
python3 -m pip install -e '.[models,test]' matplotlib
python3 -m pytest -q
python3 -m compileall -q src scripts tests plots
python3 scripts/07_run_single_request.py
python3 scripts/00_check_environment.py --config configs/development.yaml --device cuda

# Level A: explicit initial download and projection validation.
python3 scripts/04_check_qkv_projection.py --config configs/development.yaml --device cuda --allow-download --output results/raw/opt125m_projection.json
SEMCACHE_OPT_INTEGRATION=1 python3 -m pytest -q
python3 scripts/08_probe_cross_query_qkv.py --config configs/development.yaml --device cuda --output results/raw/opt125m_probe.csv
python3 scripts/09_real_qkv_cache_demo.py --config configs/development.yaml --device cuda --output results/raw/opt125m_physical.csv
python3 plots/plot_qkv_similarity_by_layer.py --input results/raw/opt125m_probe.csv --output-dir results/figures/opt125m

# Level B: explicit large-model download; sufficient GPU VRAM is required.
python3 scripts/00_check_environment.py --config configs/base.yaml --device cuda --output results/raw/opt67b_environment.json
python3 scripts/04_check_qkv_projection.py --config configs/base.yaml --device cuda --allow-download --output results/raw/opt67b_projection.json
python3 scripts/08_probe_cross_query_qkv.py --config configs/base.yaml --device cuda --output results/raw/opt67b_probe.csv
python3 scripts/09_real_qkv_cache_demo.py --config configs/base.yaml --device cuda --output results/raw/opt67b_physical.csv
python3 plots/plot_qkv_similarity_by_layer.py --input results/raw/opt67b_probe.csv --output-dir results/figures/opt67b
```

For reportable runs append `--revision IMMUTABLE_HF_COMMIT` consistently to the
model scripts; the default `main` is exploratory. Requested and resolved model
and tokenizer revisions and installed package versions are recorded in JSON.
The environment checker does not load weights, so its resolved revision is null.

Capture uses `with qkv_capture(model, layers=[0]) as capture:` around one forward.
Records are indexed by layer and contain detached Q/K/V and projection-input
hidden states. Forward hooks attach to the loaded OPT `q_proj/k_proj/v_proj`
modules and are removed even on failure. Each hook independently executes its
exact module's `forward` on its actual input, asserts Torch numerical closeness,
and records max/mean absolute error and cosine similarity. Failure aborts before
results are saved. Validation is instrumentation parity on the same input, not
cross-query equality. Default capture copies to CPU; `storage_device=None` keeps
the execution device. Validation/copies make this unsuitable for latency claims.

Raw tensors have shape `[batch, token, hidden]`, before Q scaling and head
splitting. Metadata separately reports the equivalent head layout
`[batch, heads, token, head_dim]`; head tensors are not stored. The adapter checks
loaded projection widths, head count and head dimension; unsupported layouts
fail explicitly. This depends on Transformers' OPT module structure, so pin and
record the version validated on SERAPH. The current dependency range is not a
claim that every version has been tested.

Every probe tokenizes complete prompts and selects windows by exact token IDs.
A has identical prefix and position, B differing prefix at the same position,
C differing prefix and position, and D differing content with identical prefix
and position. A bounded deterministic candidate search must satisfy those
constraints or raises. Larger windows may require changing the full-prompt
fixtures. Special-token windows are excluded. CSV preserves each layer and Q/K/V
separately, including full input IDs, selected IDs, and end-exclusive positions;
the JSON sidecar also records decoded windows, shapes and projection validation.
D has no shared IDs and records both differing windows. C combines context and
position changes; comparison with B is diagnostic, not a pure isolated causal
estimate of position alone.

`CacheEntry.from_tensors` owns compact detached copies (CPU by default; CUDA is
optional), preserving the existing logical-only constructor and policies.
Physical bytes count referenced storage; logical bytes govern admission/capacity
and do not measure peak VRAM. Keep entry storage immutable after insertion.
The physical demo uses explicit controlled semantic vectors and the existing
clusterer/matcher, then compares a retrieved A block with freshly computed B
projections per layer. It does not validate learned semantic clustering.

A **cache hit** establishes index retrieval; **Q/K/V similarity** measures numerical
representation differences; **safe inference reuse** additionally needs attention
substitution and output-quality validation, which are not implemented. Similarity
reductions use float64 on CPU: relative L2 is `||A-B|| / ||A||`, with A the reference.
For a zero reference and nonzero candidate it is null; both zero vectors have
cosine 1 and relative L2 0, while one zero vector has cosine 0. Plots show the full
cosine range, including low or negative similarities.


## Milestone 3: controlled cached-QKV output impact

Implemented: one-layer, one-window `q`, `k`, `v`, or `qkv` substitution in a real
OPT forward. Source A projections pass through the existing physical cache:
**MISS → INSERT → HIT → FETCH → INJECT**. Forward hooks replace raw unscaled
`q_proj/k_proj/v_proj` outputs before head reshaping/scaling; weights and
Transformers source are unchanged. Hooks are always removed in `finally`.
This instrumented experiment still computes the fresh projections and therefore
makes no compute-saving or latency claim.

QKV similarity measures representation differences. A physical cache HIT proves
retrieval. Injection feeds retrieved values into attention. Output impact measures
changes to B's numerical logits. **None establishes safe inference reuse**, BLEU
preservation, task quality, cross-user reuse, or reproduction of paper results.
The paper's OPT-6.7B FP16 workload, LoRA, Fig. 6 and full pipeline remain deferred.
OPT-125m FP32 is the development model; an RTX A2000 12GB cannot straightforwardly
hold OPT-6.7B FP16 plus capture/runtime memory.

Scripts 08, 10 and 11 share `semantic/probes.py`. A is the identical-prefix control;
B changes preceding context at fixed positions; C changes context and position.
D is explicitly invalid for reuse: it forcibly retrieves A's source key for a
negative/stress injection, never an exact-token cache match for B.

Script 10 defaults to A/B, layers 0/1/5/11 and all four modes. It enforces A identity
at every selected layer, B identity at layer 0, and causal-prefix integrity for
all rows. Higher-layer B rows are diagnostic measurements. Script 11 defaults to
A/B/C/D with the same layers/modes. Both stop on control violations and print
measured values before checking them. `--tolerance 1e-6` is an absolute float32
implementation-control tolerance, **not a safe-reuse threshold**. Overrides:
`--layers 0 1`, `--all-layers`, `--modes qkv`, `--cases A`, `--output PATH`.

CSV metrics include max/mean absolute logit difference, relative L2, cosine,
last-position and affected-suffix mean **KL(baseline || injected)**, last argmax
IDs/agreement, and prefix maximum error. The affected suffix includes positions
`target_start` through the final position; the prefix is strictly before it
(empty prefix reports zero). Reductions use CPU float64 and stable log-softmax.
Relative L2 reuses the existing zero-reference convention (null for a nonzero
candidate against zero). CSV/JSON record measured provenance, source/target IDs,
positions, cache lookup kind, model revision, dtype, seed and injected bytes.

Local CPU validation (2026-09-14): Python 3.10, Torch 2.6.0+cpu, Transformers
4.57.6; 33 tests passed, 2 optional pretrained tests skipped. Random tiny OPT
forwards verified exact control parity, prefix integrity, physical injection and
nonzero stress effects. These are unit fixtures, not pretrained OPT-125m results.
The integration-enabled suite also skips unavailable local weights. No model
weights were downloaded; no real-GPU Milestone 3 results are available locally.
Historical Milestone 2 notes above describe the previous local environment.
The user separately reports successful SERAPH Milestone 2 validation (22 tests,
OPT-125m CUDA capture parity and physical HIT); this is not Milestone 3 validation.

Exact local commands for the temporary environment used here:

```bash
OMP_NUM_THREADS=1 /tmp/semcache-m3-venv/bin/python -m pytest -q
OMP_NUM_THREADS=1 SEMCACHE_OPT_INTEGRATION=1 /tmp/semcache-m3-venv/bin/python -m pytest -q
/tmp/semcache-m3-venv/bin/python -m compileall -q src scripts tests plots
```

Exact SERAPH commands (existing managed environment and locally provisioned weights):

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
conda activate semcache
cd /data/khuss/repos/GP_semcache
python -m pytest -q
SEMCACHE_OPT_INTEGRATION=1 python -m pytest -q
python scripts/10_validate_qkv_injection.py --config configs/development.yaml --device cuda --revision 27dcfa74d334bc871f3234de431e71c6eeba5dd6
python scripts/11_probe_qkv_reuse_output_impact.py --config configs/development.yaml --device cuda --revision 27dcfa74d334bc871f3234de431e71c6eeba5dd6
python plots/plot_qkv_reuse_output_impact.py
```

The loader preflight explains the required cuBLAS environment for deterministic
CUDA; determinism is never disabled. Loading retains `use_safetensors=False`
(the working original PyTorch checkpoint compatibility choice). Tests use local
files only and skip unavailable pretrained weights. The optional pretrained test
runs on CPU; the two scripts above perform CUDA validation. Default result files
are `results/raw/qkv_injection_control.csv` and
`results/raw/qkv_reuse_output_impact.csv`, each with a JSON sidecar. The plot uses
measured B/C qkv rows; no results or plots are fabricated when weights are absent.

## Milestone 4: PEFT decomposition and logical EdgeLoRA exchange

Milestone 4 validates **base QKV + user-specific LoRA QKV = native PEFT QKV**.
It adds two deterministic, nonzero **synthetic, untrained** adapters on one
frozen OPT model. It does not validate personalized-task quality or safe
cross-user reuse. Fig. 6, TinyBERT clustering, training, the full multi-user
pipeline and workload remain out of scope.

**IMPLEMENTED, CPU-TESTED, and user-reported PRETRAINED-MODEL TESTED /
REAL-GPU VERIFIED on SERAPH for Milestone 4.** Historical local measurements
remain separate from the user-reported SERAPH verification. Current M4 evidence and limitations are
in [docs/milestone4_status.md](docs/milestone4_status.md).

Use `python -m pip install -e '.[lora,test]'` when PEFT is absent. This optional
extra pins PEFT 0.20.0. The loader still uses local files by default and retains
`use_safetensors=False`. No pytest test downloads model weights.

`configs/development.yaml` adds rank=8 and q_proj/k_proj/v_proj targets
(paper-derived). Alpha=8, dropout=0, user seeds 101/202 and normal initialization
scale 0.01 are reproduction choices. Both A and B are explicitly initialized;
these adapters are never called trained. See the
[exact fixture recipe](docs/reproduction_choices.md#milestone-4-lora-projection-structure).
The Python `load_external_adapter` interface accepts a local PEFT adapter later;
its training provenance must be supplied independently.

- Script 12: selected layers 0/1/5/11, nonzero deltas, base + delta parity, base
  fingerprints and user_a → user_b → user_a identity. Outputs
  `results/raw/lora_decomposition_probe.csv` plus JSON metadata.
- Script 13: fixed_hidden_input and full_user_forward experiments, plus existing
  A/B cross-context windows under different users. Outputs
  `results/raw/multiuser_lora_probe.csv` plus JSON. Base equality is required
  only for the same hidden input; later full-forward hidden states can differ.
- Script 14: reconstructs Q/K/V at **all layers by default** with local hooks and
  compares logits to native PEFT. Outputs
  `results/raw/edgelora_projection_parity.csv` plus JSON. This is single-process
  same-device ES/UD projection emulation; native computations still execute.

Scripts accept the existing CLI overrides, `--layers 0 1`, `--output PATH` and
`--tolerance 1e-6`. They stop on nonzero/parity/invariance failures. Communication
uses one hidden send and three LoRA returns per layer: `4*n*d` **elements**, with
actual dtype-dependent bytes recorded separately. Layer communication is repeated
on the three decomposition rows, not three separate exchanges. No timing or
network performance is measured. CPU UD offload is deferred.

Exact SERAPH commands with locally provisioned pretrained weights:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
conda activate semcache
cd /data/khuss/repos/GP_semcache
python -m pytest -q
SEMCACHE_OPT_INTEGRATION=1 python -m pytest -q
python scripts/12_validate_lora_decomposition.py --config configs/development.yaml --device cuda --revision 27dcfa74d334bc871f3234de431e71c6eeba5dd6
python scripts/13_probe_multiuser_lora.py --config configs/development.yaml --device cuda --revision 27dcfa74d334bc871f3234de431e71c6eeba5dd6
python scripts/14_validate_edgelora_projection_path.py --config configs/development.yaml --device cuda --revision 27dcfa74d334bc871f3234de431e71c6eeba5dd6
```

The integration test is CPU/local-only and cleanly skips missing OPT files.
Only actual successful CUDA script runs can establish M4 REAL-GPU VERIFIED.
Synthetic-adapter similarity cannot establish personalized quality or safe reuse.


## Milestone 5: integrated controlled SemCache

`SemCacheEngine` connects tokenizer → fixture/HF encoder → nearest intent cluster
→ w=3 exact token windows → global cache lookup → deterministic non-overlapping
selection → native PEFT projection on **unmatched rows only** → actual attention,
FFN and logits → attention impact → CHU/history → admission/eviction. Manual PBR
recalculates impact from recent per-cluster query observations.

The cache owns TOTAL Q/K/V per layer, including the source user's LoRA. Source
positions are metadata; matching works at different target positions. Cross-user
reuse is approximate and output effects are recorded against the target user's
normal PEFT baseline. It does not replace the source delta with the target delta.
All results retain `safe_reuse_claimed=false`.

Scripts 15/16 validate fixed semantic and numerical fixtures without downloads.
Script 18 checks two disjoint cache windows, native row indices and final logits
against an exact control. Script 17 repeats that gate, then executes controlled
health/weather queries from user_a/user_b. The wrapper records each layer/Q/K/V's
actual native-row subset; it never computes a full projection then overwrites it.
Any gate failure aborts before approximate reuse. The fixed 1e-6 tolerance is an
implementation check, not a safe-reuse threshold.

Logical capacity stays 20 GB; physical tensors are stored on CPU. Optional
`--logical-capacity-bytes N` on script 17 forces development eviction; script 16
uses a separate tiny 20-byte logical fixture. No physical 20 GB allocation occurs.
Block lookup ratio and unique token reuse ratio are separate development metrics.
Analytical saved FLOPs/communication elements and dtype-aware bytes are separate
from measured output errors; there is no wall-clock speedup claim.

The optional HF encoder requires an explicit `semantic_encoder.checkpoint`,
`semcache.encoder_kind: huggingface_text` and dimension-compatible
`semcache.initial_centroids`. No TinyBERT checkpoint is silently selected.
Default M5 uses deterministic fixture vectors with two development clusters.
All initialization, matching, head aggregation and scheduling choices are listed
in [the reproduction ledger](docs/reproduction_choices.md#milestone-5-integrated-prefill-system).

Run on SERAPH with existing local pretrained weights, in this order:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
conda activate semcache
cd /data/khuss/repos/GP_semcache
python -m pytest -q
SEMCACHE_OPT_INTEGRATION=1 python -m pytest -q
python scripts/15_validate_semantic_frontend.py --config configs/development.yaml
python scripts/16_validate_cache_policy.py --config configs/development.yaml
python scripts/18_validate_mixed_projection.py --config configs/development.yaml --device cuda --revision 27dcfa74d334bc871f3234de431e71c6eeba5dd6
python scripts/17_run_semcache_controlled_trace.py --config configs/development.yaml --device cuda --revision 27dcfa74d334bc871f3234de431e71c6eeba5dd6 --compare-baseline
```

Outputs under `results/raw/`: `semantic_frontend_validation.csv`,
`cache_policy_validation.csv`, `mixed_projection_validation.csv` plus JSON,
`semcache_controlled_trace.csv` plus JSON, and `semcache_events.jsonl`.
Summary CSV holds per-query metrics; event JSONL holds keys, source/target spans,
policy scores, metric mutations, allocation/eviction and projection row indices.
Tensors are never serialized into JSON. Optional pretrained pytest remains
CPU/local-only; scripts 18/17 with `--device cuda` provide the GPU evidence.

M5 covers prefill. `lookup_latest_token` explicitly prepares the decode lookup
interface, but does not integrate past-key-values or a generation loop. M6 must
address those and paper-scale experiments. No MultiWOZ/CoQA/SNIPS workload,
trained adapters, BLEU, Fig. 6 or other paper performance reproduction is included.


## Milestone 6A experiment foundation

Paper configs live in `configs/paper/`; dataset files explicitly inherit
`common.yaml` through the M6 loader. All specified defaults are preserved;
unknown encoder/generation/BLEU settings remain null. Each config leaf and
result metric has provenance. Execution modes are explicit and full-generation
execution currently raises a blocker. Paper reference tables never feed the
simulation, and no parameters are fitted to them.

Scripts 20–23 prepare sources, validate hashes and static reuse opportunities,
check unit-safe equations, and consume prepared workloads through the M5 engine
in logical smoke mode. No source is downloaded without `--allow-download` and
an explicit HF ID. Local JSON/JSONL ingestion works without `datasets`.
Optional dependency groups: `datasets` for HF ingestion; `evaluation` for
SacreBLEU. Scripts never install packages. Normal pytest needs no external data,
network or CUDA; numerical BLEU checks skip when SacreBLEU is absent.

```bash
python scripts/22_validate_simulation_model.py --config configs/paper/common.yaml
python scripts/20_prepare_paper_workloads.py --dataset multiwoz --config configs/paper/multiwoz.yaml --input-path /path/to/multiwoz_train.json --output results/workloads/multiwoz.jsonl --seed 42
python scripts/21_validate_paper_workloads.py results/workloads/multiwoz.jsonl --config configs/paper/multiwoz.yaml
python scripts/23_run_workload_smoke.py --workload results/workloads/multiwoz.jsonl --config configs/paper/multiwoz.yaml --max-queries 100
```

The default raw-query transformation, source order and SHA-seeded round-robin
logical assignment are documented reproduction choices. Source and workload
hashes, reconstruction settings and run environments are written to manifests.
Audit windows use an explicit whitespace proxy unless a pinned tokenizer is
supplied. Logical smoke has no tensors or attention-derived impact: it does not
establish CHU/PBR behavior, learned semantic quality or real projection reuse.
It uses dataset-specific C and records an explicit development FP16 logical
activation-payload choice, independent of model weight precision.

20GB is a logical budget. Physical cache tensors, GPU allocated/reserved memory,
parameters, adapters and analytical memory remain separately scoped. Compute
rates and communication element precision must be explicit before latency is
produced. Eq.19 retains all attention/FFN costs; 512/300 is only the simplified
Eq.20 ratio. Measured OPT-125m/A2000 and paper OPT-6.7B/A100 latency are
NOT_COMPARABLE. Full Table II, Fig.6–10, BLEU and paper-model GPU runs remain
future milestones. Earlier README sections describe historical milestone scope;
this section supersedes their deferred dataset/cost-interface statements.

### M6B-1 baseline comparison

The CPU logical/analytical runner now supports UD_ONLY, ES_ONLY, FBC_V1, FBC_V2 and SEMCACHE
on identical prepared workloads. See [semantics, limitations and SERAPH smoke
commands](docs/milestone6b1_status.md). This is baseline infrastructure; M6B and
Table-II/BLEU reproduction are not complete.

M9-B adds a model-free logical multi-user simulator with paired cache traces,
three separate reuse metrics, and isolated M9 fixture latency diagnostics. See
[the M9-B guide](docs/m9b_multi_user.md) for input contracts, commands, provenance,
and the distinction between synthetic smoke outputs and pending dataset runs.

For existing raw SNIPS/MultiWOZ workload JSONL, use the separate
[M9-B.1 semantic preparation stage](docs/m9b_semantic_preparation.md) to produce
pinned-tokenizer/TinyBERT inputs offline before running the model-free simulator.
