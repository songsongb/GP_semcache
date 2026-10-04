# C6-B3-0 MultiWOZ planning

The externally audited SERAPH input is `results/workloads/multiwoz.jsonl`
(approximately 82 MB). It contains conversation/source IDs, query/reference
text, order indices, source split, domain metadata, and metadata.turn_id and
original turn/speaker metadata. The old semantic artifact (~225 MB) is not an
input to this preparation: all prompt tokens, embeddings and clusters are rebuilt.

History uses previous prepared rows of the SAME conversation, preserving their
query and reference text exactly. Current references never enter their own
prompt. Runtime requires unique source IDs, complete per-conversation numeric
turn IDs 0,2,4,... in source order, nonempty references, and unambiguous split
identity. Missing or nonstandard turn sequences fail rather than invent context.
This strict check may identify unsupported source layouts requiring a further audit.

Prompt version: `c6b3_multiwoz_history_v1`:

```
Dialogue:
User: <previous query>
Assistant: <previous reference>
...
User: <current query>
Assistant:
```

No K is predefined. Analysis reports nearest-rank p50/p90/p95/p99/max and fraction
within the specified bound for K=0..4, for prompt and full training sequence.
Full length includes separately tokenized leading-space response plus one EOS.
Analysis loads only the pinned local OPT tokenizer, never OPT weights or TinyBERT.
Choose K from these lengths, never task quality. Generation requires the matching
analysis artifact and explicit K. It refuses ANY over-bound example; no truncation
or length-based row filtering. If needed, analyze an explicitly larger bound.

Generation runs the existing pinned TinyBERT encoder on CPU and unchanged M9-B
first-C/buffered clustering. Resolved snapshot provenance is checked by existing
M9-B validation. C6's existing selector freezes exactly 32 natural earlier-source
exact-w3 events, preferring distinct target conversations in round-robin order.
Special tokens are excluded. Insufficient episodes fail. The old workload and
selection are never used to obtain token spans.

All selected source AND target conversations form the evaluation holdout. All
remaining conversations train, without a cap. Training conversations are assigned
with existing seed-42 group-balanced two-user assignment after holdout exclusion.
Evaluation users retain the existing C6 whole-conversation assignment over the
full workload. These are separate deterministic assignments for disjoint groups.
The plan includes complete IDs/hashes, user counts, domains, expected total and
supervised tokens, response length statistics and future rank-8 QKV LoRA config.
No adapters are trained. Future objective masks prompts and supervises completion
plus EOS. Future primary metric is corpus SacreBLEU with C6-A's explicit protocol;
no BLEU is computed here and paper reproduction is not claimed.

New directories/files are mandatory. Exclusive file creation prevents overwrites.
A failed write can leave partial NEW outputs; use a new destination for retry.
Manifest hashes bind source, semantic workload, selection, training plan and length
analysis. Validate stage verifies source/output bytes without importing models.

## SERAPH execution

Use the existing environment and local snapshots; no installation/download step.

```bash
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export XDG_CACHE_HOME=/data/khuss/.cache TORCH_HOME=/data/khuss/.cache/torch
export TORCH_EXTENSIONS_DIR=/data/khuss/.cache/torch_extensions/py311_cu118
export TRITON_CACHE_DIR=/data/khuss/.cache/triton CUDA_CACHE_PATH=/data/khuss/.cache/cuda
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python3 scripts/53_prepare_cachegen_c6b3_multiwoz.py --stage analyze \
  --source results/workloads/multiwoz.jsonl --max-sequence-length 256 \
  --output-dir results/cachegen/c6b3/multiwoz_lengths_256
```

Read `length_stats.json` first. Set B3_HISTORY_K to the chosen integer 0..4;
there is deliberately no default. If changing the length bound, rerun analysis
in a new directory and use that analysis and bound below.

```bash
: "${B3_HISTORY_K:?Set B3_HISTORY_K after reviewing length statistics}"
python3 scripts/53_prepare_cachegen_c6b3_multiwoz.py --stage generate \
  --source results/workloads/multiwoz.jsonl --max-sequence-length 256 \
  --analysis results/cachegen/c6b3/multiwoz_lengths_256/length_stats.json \
  --history-k "$B3_HISTORY_K" \
  --semantic-output results/workloads/c6b3_multiwoz_history_semantic.jsonl \
  --output-dir results/cachegen/c6b3/multiwoz_plan
python3 scripts/53_prepare_cachegen_c6b3_multiwoz.py --stage validate \
  --source results/workloads/multiwoz.jsonl \
  --output-dir results/cachegen/c6b3/multiwoz_plan
```

Outputs: semantic JSONL plus `training_plan.json`, `evaluation_selection.json`,
`manifest.json`, and `length_stats.json` in the plan directory. Counts, K,
diversity, cross-user statistics and snapshot resolution remain unmeasured until
SERAPH execution. Local tests use synthetic rows/tokens; no actual data statistics
or task quality are asserted.

## V2: current-user content reselection of the existing K=4 / 384 artifact

Do not regenerate embeddings or clustering. `reselect` verifies the old manifest's
output hashes and raw-source hash, reconstructs and compares the same prompts,
then loads only the pinned offline fast tokenizer. Offset mappings must reproduce
all frozen token IDs exactly. A token is eligible only if its nonempty character
interval is entirely inside the CURRENT user's original utterance. Boundary
crossing tokens, speaker markers, separators and historical turns cannot match.

Targets need depth >=1. Eight targets per bucket (1,2,3,4+) and 32 distinct target
conversations are mandatory. Candidate ties use earliest source, source span,
then target span. Stable augmenting bipartite matching allocates conversations to
bucket slots without a greedy allocation falsely exhausting a bucket. There is
no quality input or source-user balancing. Infeasibility raises with distinct
candidate conversation counts per bucket; constraints never relax.

The new selection regenerates training/holdout using the same group-balanced
assignment code. `selection_audit.json` reports all requested diversity, window,
user-direction and leakage diagnostics. `current_user_spans.json` records every
row's content character interval, eligible token indices and history depth.
The existing semantic JSONL remains byte-for-byte unchanged, referenced by hash.

With the offline/cache environment above:

```bash
python3 scripts/53_prepare_cachegen_c6b3_multiwoz.py --stage reselect \
  --source results/workloads/multiwoz.jsonl \
  --existing-plan results/cachegen/c6b3/multiwoz_plan \
  --semantic-input results/workloads/c6b3_multiwoz_history_semantic.jsonl \
  --output-dir results/cachegen/c6b3/multiwoz_plan_v2
python3 scripts/53_prepare_cachegen_c6b3_multiwoz.py --stage validate \
  --source results/workloads/multiwoz.jsonl \
  --output-dir results/cachegen/c6b3/multiwoz_plan_v2
```

The old manifest must explicitly confirm K=4, bound=384 and the unchanged prompt
version/BLEU protocol. Reselect does not use the generate-stage default bound.
It refuses an existing output directory. No TinyBERT, OPT weights, training,
BLEU computation or generation occurs. Local tests are synthetic; real window
and user-direction counts remain pending SERAPH execution.
