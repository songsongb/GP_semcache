# C6-B2: trained SNIPS five-mode causal evaluation

B2 reuses the task-trained B1 adapters and the frozen C6-A 32-SNIPS selection.
It does not train, reselect episodes, run BLEU, or add MultiWOZ/CoQA. B1's prior
capability scores are verified and recorded; their magnitude is never used to
select episodes or admit a run. No measured task improvement is assumed.

## Artifact boundary

The CLI requires the selection, trained adapter root and capability root.
Model-free validation checks exactly 32 SNIPS episodes before applying an
optional smoke prefix. B1's existing planner validates all source/target IDs,
indices, clusters, non-prefix exact-w3 spans, reference hashes and user labels.
Its training-artifact verifier reconstructs the saved training plan and checks
workload, selection, training-plan, holdout and training-ID hashes. No selected
source or target may appear in training.

Both adapter directories must have nonempty standard weights, validated vanilla
QKV LoRA configurations, and exactly the file hashes in the completed training
manifest. B2 uses `load_two_task_users`, never controlled fixtures. Real loading
verifies unchanged base QKV fingerprints, both adapters' decomposition
compatibility, and freezes every parameter after user switching. Configurations
remain rank 8 / alpha 8 / dropout 0 / no bias / causal LM.

The completed B1 capability manifest must reference those exact adapters,
training manifest, selection, workload and label serialization. All capability
output hashes are checked. Its full 32 cases must match frozen target identities
and order. Predicted labels, correctness, margins and greedy exact matches are
recomputed from the saved scores/text, then the summary is checked against those
cases. No minimum accuracy or hard-coded 100% score is required.

The frozen K20/V16 profile SHA is checked even in a dry run (only file bytes are
read). Adapter tensor contents, resolved runtime model weights, and actual GPU
properties cannot be validated by a model-free run and remain runtime checks.

## Physical modes and execution

| Mode | Source resident | Source and target fresh projections |
|---|---|---|
| FULL_RECOMPUTE | No cache or source forward | Target native trained PEFT |
| RAW_SEMCACHE | TOTAL Q/K/V FP16 | Native trained PEFT |
| STORAGE_KV_COMP | TOTAL Q FP16; frozen TOTAL K/V frame | Native trained PEFT |
| TRANSPORT_QKV_COMP | TOTAL Q/K/V FP16 | Real LoRA Q/K/V delta encode/decode, then base recombination |
| FULL_PIPELINE | TOTAL Q FP16; frozen TOTAL K/V frame | Real LoRA Q/K/V delta encode/decode, then base recombination |

The source uses the frozen source user's adapter; fresh target rows use the
frozen target user's adapter. Resident Q is never compressed. Q compression is
still deferred to C7. The physical safety contract remains
`PHYSICAL_EXACT_W3_WITHIN_SEMANTIC_CLUSTER`.

Each episode/mode constructs at most one source entry. Reuse modes perform one
source prefill and one insertion/lookup. They call C6-A's existing
`Storage`, `make_c6_cache` and `insert_and_lookup_c6`. The same frozen codec
instance is registered in compressed caches. `GlobalCache.lookup` owns exactly
one decode; the resident retains Q and compressed KV bytes and the temporary
view contains decoded Q/K/V. The hit is then shared by all seven candidate
forwards and one greedy prompt prefill. There are no seven independent source
prefills, repeated lookups or repeated storage decodes.

Candidate branches are seven independent full target forwards, without shared
mutable past-KV state. They reuse C6-A's mixed projection path and B1's label
scorer. Frozen entry identity, span, owner, lookup count, frequency and logical
charges are checked before and after every candidate and after generation.
Any discrepancy fails the run rather than dropping an episode.

Greedy generation uses the same hit for prompt prefill, then its own incremental
past KV and fresh projection rows, following C6-A's raw-logit argmax path. EOS
and the configured new-token limit are respected. This is sampling-free,
single-beam decoding. Candidate transport packets include all fresh prompt and
candidate rows in that forward; incremental generation uses separate packets.
No equivalence between those transport packet groupings is claimed.

GlobalCache always has the same one-entry raw logical capacity (1,474,560 bytes).
`capacity_charge=None` and zero shared overhead keep physical compression from
changing admission/eviction. Existing guards are reused unchanged. A checkout
with the legacy GlobalCache API fails before OPT loading; `--storage-src` alone
cannot replace the required extended GlobalCache/CacheEntry implementations.

## Task metrics

The primary metric is B1's seven-way mean conditional label-token log-probability
classification. Canonical order is AddToPlaylist, BookRestaurant, GetWeather,
PlayMusic, RateBook, SearchCreativeWork, SearchScreeningEvent. `label_ids()`
serializes one leading ASCII space plus the label and tokenizes it separately.
Frozen prompt IDs remain unchanged. Prompt and EOS tokens are excluded from
scores. The first canonical label wins exact ties.

Each case reports all seven scores, true/predicted labels, correctness,
best-wrong label, correct/best-wrong log probabilities and classification margin.
Per-mode summaries report counts, overall/user/intent accuracy and confusion,
plus means of those log probabilities and margins. User/intent strata with no
cases have null accuracy and zero count. Greedy text/IDs and stripped/casefolded
exact-label accuracy are explicitly secondary; default limit is 12 new tokens.

Paired directions and interpretation are fixed:

- FULL_RECOMPUTE → RAW_SEMCACHE: semantic reuse.
- RAW_SEMCACHE → STORAGE_KV_COMP: storage.
- RAW_SEMCACHE → TRANSPORT_QKV_COMP: transport.
- TRANSPORT_QKV_COMP → FULL_PIPELINE: storage after transport.
- RAW_SEMCACHE → FULL_PIPELINE: combined compression.

Pairs report predictions/correctness, flips, correct-to-wrong, wrong-to-correct,
and differences in correct-label score, best-wrong score and margin. Aggregates
report accuracy deltas in percentage points, transition counts, mean/median
margin deltas and mean correct-label score deltas. Wrong-to-correct changes are
observed perturbations, not evidence of general compression benefit.

## Accounting and outputs

Per-case accounting reports raw QKV/KV reference bytes, resident Q bytes,
resident KV frame bytes, total resident payload and local metadata. Metadata is
already inside the compressed frame and is not added twice. Ratios are raw KV /
stored KV frame and raw QKV / resident payload. Reduction is
`100 * (1 - resident_payload / raw_QKV)`, which may be negative. The shared
profile is recorded once in the manifest, never charged per entry. FULL_RECOMPUTE
has no resident and null compression ratios.

Outputs in a new empty `--output-dir`:

- `per_case.csv`: episode × five modes, task and greedy results, event identity,
  byte accounting and measured execution counters.
- `paired_task_quality.csv`: five directed causal comparisons per episode.
- `summary.json`: per-mode summaries and paired causal aggregates.
- `summary.csv`: per-mode summaries, including explicit case counts/subset flags.
- `manifest.json`: verified input/training/capability hashes, measured B1 summary,
  trained adapter/model/tokenizer provenance, label policy, frozen/full and
  executed-prefix identities, codec source hashes, actual GPU/software data,
  git state, branching policy, counters and output hashes.

Counters include source forwards, candidate forwards, greedy forwards, storage
decodes, transport encode/decode calls, and CDF fits. Transport fit counts follow
the existing codec's `cdf=None` default (one fit per successful encode).
Storage fitting is forbidden by C6-A's call guard; successful runs report zero.
Dry runs record zero executions and leave actual GPU/runtime codec hashes unset.

## SERAPH commands (RTX 3090 24 GB)

Use the existing working Python/CUDA/CacheGen environment on an allocated GPU.
No model/download/install command is included. No runtime estimate is assumed.
The extended cache implementation must already be present in the checkout.
Set the frozen profile path to its existing location if it is in the storage
export instead of the main results tree; its required SHA cannot change.

```bash
cd /data/khuss/repos/GP_semcache
export TORCH_EXTENSIONS_DIR=/data/khuss/.cache/torch_extensions/py311_cu118
export TRITON_CACHE_DIR=/data/khuss/.cache/triton
export CUDA_CACHE_PATH=/data/khuss/.cache/cuda
export TORCH_HOME=/data/khuss/.cache/torch
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export XDG_CACHE_HOME=/data/khuss/.cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
COMMON=(
  --snips /data/khuss/repos/GP_semcache/results/workloads/m9b_snips_semantic.jsonl
  --selection /data/khuss/repos/GP_semcache/results/cachegen/c6/quality_dry_run_32/selection.json
  --adapter-root /data/khuss/repos/GP_semcache/results/cachegen/c6b/snips_pilot200_interleaved
  --capability-root /data/khuss/repos/GP_semcache/results/cachegen/c6b/snips_pilot200_interleaved_capability
  --storage-src /data/khuss/repos/GP_semcache/.c6_storage_src/src
  --profile-path /data/khuss/repos/GP_semcache/results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin
  --seed 42 --max-new-tokens 12 --max-sequence-length 256
)
```

Dry run: validates the complete 32-case cohort and all artifacts without importing
OPT, PEFT, a CacheGen codec or CUDA extension. Writes only the manifest and prints
a concise plan:

```bash
python3 scripts/52_run_cachegen_c6b2_snips.py "${COMMON[@]}" \
  --device cpu --dry-run \
  --output-dir results/cachegen/c6b2/snips_dry_run_32
```

First-four smoke: uses the original first four SNIPS episodes without reordering.
The full 32-case selection and holdout are still validated and remain unchanged:

```bash
python3 scripts/52_run_cachegen_c6b2_snips.py "${COMMON[@]}" \
  --device cuda:0 --max-episodes 4 \
  --output-dir results/cachegen/c6b2/snips_smoke_4
```

Full frozen evaluation after reviewing the smoke:

```bash
python3 scripts/52_run_cachegen_c6b2_snips.py "${COMMON[@]}" \
  --device cuda:0 --max-episodes 32 \
  --output-dir results/cachegen/c6b2/snips_frozen_32
```

CPU regression suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_cachegen_c6*.py' -v
```

The development workspace lacks SERAPH's trained artifacts, workloads, profile,
CUDA and PyTorch. Synthetic tests mock profile verification only for their
explicit non-codec fixture; production always enforces the frozen profile SHA.
Real tensor/codec execution and B1 task scores are not claimed to have been
reproduced locally. All existing C6-A/B1 source and safety guards remain intact.
