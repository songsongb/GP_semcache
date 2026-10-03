# C6-B1: task-trained SNIPS adapters and capability validation

C6-B1 is separate from C6-A. It trains two real PEFT task adapters and evaluates
FULL_RECOMPUTE only. It never invokes SemCache, transport compression, storage
compression, or BLEU. C6-A controlled-fixture code and artifacts remain unchanged.
A successful training job establishes that optimization occurred, not that task
quality is adequate; inspect the capability report before freezing a threshold.

## Frozen data contract

`--selection` is required and may point to any frozen C6 selection artifact.
All SNIPS episodes are read, including their source and target identities. Their
union is the held-out ID set. Every ID must exist exactly once in the canonical
`m9b_snips_semantic.jsonl`; duplicate identities, missing/noncanonical references,
wrong model/semantic provenance, or mismatched selection indices, spans, tokens,
clusters, cache keys, reference hashes and users fail closed. The selected
source and target must also differ in text and full token sequence. MultiWOZ
episodes are ignored. The planner does not require exactly 32 episodes, allowing
small synthetic tests; the real pilot should supply the frozen 32-SNIPS set.

Remove all held-out rows **before** calling the existing
`logical_user_assignment(eligible, users=2, seed=42, dataset='snips')`.
Map `user_000` to `user_a` and `user_001` to `user_b`. This is assignment over
the remaining training rows, not inherited full-workload index parity.
Evaluation uses each frozen episode's `target_user`, without reassignment.

In stable workload order, keep at most the first N rows per user and reference
intent. N defaults to 200: at most 1,400 examples per user and 2,800 total. Class
membership is the canonical training label; no evaluation score or generation
influences selection. No oversampling occurs. Missing class/user examples are
reported as shortfalls; a user with no training examples blocks real training.
Holdout is ID-based as requested; different IDs with identical utterance text
are not silently deduplicated. Plan and manifest record complete held-out and
training ID lists, hashes, counts, assignment seed, and capping rules.

## Training and loading

Use only local `facebook/opt-2.7b` model and tokenizer files pinned to
`905a4b602cda5c501f1b3a2650a4152680238254`.
Both adapters use vanilla causal-LM LoRA: rank 8, alpha 8, dropout 0, Q/K/V-only
projection targets, and no bias. Initialization uses standard PEFT LoRA, not
controlled-fixture initialization. User A and B use seeds `seed` and `seed+1`.
Only the current user's adapter matrices receive gradients; each user has a new
optimizer and scaler. Frozen base QKV fingerprints must remain unchanged.

Prompt IDs are the tokenizer's special-token-enabled encoding of the unchanged
prepared `query_text`, verified against the workload token IDs. Label IDs are
encoded from `completion_text = " " + canonical_label` separately with
`add_special_tokens=False`, then one EOS ID is appended. This renders
`Intent: AddToPlaylist`, while leaving the frozen prompt token IDs unchanged.
The single leading ASCII space belongs to the completion, not the prompt.
Training and candidate scoring share `label_ids()` and this exact serialization;
provenance records it, and validation rejects adapters with a different recorded
serialization. The label array is
`[-100] * prompt_length + label_ids + [eos]`; the causal LM's ordinary shift
therefore supervises the first label token from the last prompt position.
Overlength examples fail rather than dropping examples or truncating labels.

Pilot defaults: batch 1, three epochs, AdamW learning rate 1e-4, zero weight decay,
gradient accumulation 8, max length 256, gradient norm clipping 1, and gradient
checkpointing with non-reentrant checkpointing. Frozen base weights are FP16;
LoRA parameters and Adam moments are FP32, with FP16 autocast and gradient
scaling. Partial accumulation groups divide by their actual example count.
Overflow-skipped updates and successful updates are recorded; zero successful
updates fails. Training order is stable per epoch. These are conservative
starting settings for a 12 GB RTX A2000, not measured memory/runtime guarantees.

Successful training saves standard local PEFT directories `user_a/` and
`user_b/`, plus `training_manifest.json` with `trained_adapter=true` and
`adapter_source="task_finetuned_snips"`. It includes adapter file hashes,
optimizer settings, per-user training losses/update counts, seeds and resolved
model/tokenizer provenance. Dry-run/failed-start manifests do not claim trained
adapters. Output roots must be empty; automatic resume is not implemented.

`load_two_task_users()` validates both configurations, loads them under the two
required names, verifies base QKV fingerprints, and validates every projection
with the existing LoRA decomposition guard. Rank/alpha overrides, saved base
modules, DoRA, RS-LoRA, selective layer targeting and other incompatible variants
are rejected. `activate_task_user()` freezes all parameters after switching,
because PEFT switching may re-enable adapter gradients. C6-A's
`create_controlled_users()` is unchanged.

## Primary metric

For every frozen target, independently score these candidates in this order:

1. AddToPlaylist
2. BookRestaurant
3. GetWeather
4. PlayMusic
5. RateBook
6. SearchCreativeWork
7. SearchScreeningEvent

Each candidate gets one native PEFT forward on `prompt_ids + label_ids`.
For label token j, take its log probability from predictor position
`prompt_length - 1 + j`. Average over label tokens only. Prompt and EOS tokens
are excluded from classification scores. The highest mean log probability wins;
exact ties choose the first canonical label. Training and classification use
the same explicit label-token boundary.

Reports contain overall, per-user and per-intent top-1 accuracy, the complete
true-by-predicted confusion matrix, all seven scores, correct-label and
best-wrong-label mean log probabilities, and their signed margin. Missing strata
have count 0 and accuracy null. Every selected episode is one evaluation case.

Secondary greedy generation stops normally at EOS or `--max-new-tokens`
(default 12), uses no sampling, and reports whitespace-strip/casefold exact-label
match without trimming to the reference. It is not the primary metric.
`--min-accuracy` is optional, defaults to no hard failure, and is explicitly
`REPRODUCTION_CHOICE`. An unmet supplied threshold saves results and exits
nonzero with status `BELOW_MIN_ACCURACY`; it never chooses a threshold from data.

## Artifacts

Training output:

- `train_selection.json`: exact training rows/users/intents and evaluation holdout.
- `training_manifest.json`: hashes, provenance, configuration, updates and status.
- `user_a/`, `user_b/`: standard adapter configurations and weights.

Capability output (separate empty root):

- `train_selection.json`: copy of the verified actual training plan.
- `capability_per_case.csv`: target/user, reference, predicted label, scores,
  margins, greedy text/IDs/length and secondary exact match.
- `capability_summary.json`: primary accuracies/confusion and secondary summaries.
- `capability_manifest.json`: validated training/adapter/input/output hashes,
  resolved revisions and scope flags.

Capability verification re-derives the training plan using the saved training
seed/cap and checks the completed manifest and adapter hashes before model load.
The caller's evaluation seed controls evaluation RNG, not training-plan identity.
Every capability manifest states the task metric, `paper_bleu_claimed=false`,
`task_capability_stage="C6-B1"`, `semantic_reuse_enabled=false` and
`compression_enabled=false`.

## Offline SERAPH commands

Use an existing environment with PyTorch/Transformers/PEFT installed. No command
below installs packages or downloads assets. The user-designated frozen C6-A
32+32 selection below is the same input for the dry run, pilot training holdout,
and FULL_RECOMPUTE capability validation. B1 reads its SNIPS episodes and ignores
MultiWOZ episodes. Do not regenerate or reselect the C6-A episodes. The CLI still
requires an explicit selection path; this documented path is not a model-logic
default. `train_selection.json` is a separate B1 training plan and does not
overwrite or replace the frozen C6-A selection.

```bash
cd /data/khuss/repos/GP_semcache
export SELECTION=/data/khuss/repos/GP_semcache/results/cachegen/c6/quality_dry_run_32/selection.json
export TRAIN_ROOT=/data/khuss/repos/GP_semcache/results/cachegen/c6b/snips_pilot200
export CAPABILITY_ROOT=/data/khuss/repos/GP_semcache/results/cachegen/c6b/snips_pilot200_capability
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
```

CPU-only data plan, with no tokenizer/model/codec import:

```bash
python3 scripts/50_train_cachegen_c6b_snips.py \
  --selection "$SELECTION" --per-intent-per-user 200 --seed 42 \
  --output-root results/cachegen/c6b/snips_pilot200_plan --device cpu --dry-run
```

The plan prints held-out IDs, target count, user/intent counts and shortfalls.
Tokenizer boundary/length validation remains pending until actual execution.
The validator also supports `--dry-run` for data planning; it does not validate
adapter files or claim task capability in that mode.

Submit the conservative pilot (the script exports every cache/offline variable
above; its 12-hour walltime is a scheduler limit, not a runtime estimate):

```bash
sbatch scripts/50_train_cachegen_c6b_snips.sbatch "$SELECTION" "$TRAIN_ROOT"
```

On an allocated GPU after training completes, using the environment exports above:

```bash
python3 scripts/51_validate_cachegen_c6b_snips.py \
  --selection "$SELECTION" --adapter-root "$TRAIN_ROOT" \
  --output-root "$CAPABILITY_ROOT" --device cuda:0 --seed 42 \
  --max-sequence-length 256 --max-new-tokens 12
```

CPU/synthetic checks:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_cachegen_c6b_snips.py -v
bash -n scripts/50_train_cachegen_c6b_snips.sbatch
```

No real training or capability scores were produced during implementation.
The development workspace lacks the SERAPH path, canonical workloads, frozen
selection, PyTorch and PEFT; tensor/local-tiny-OPT tests skip when unavailable.
No C6-B compression mode is implemented in B1.
