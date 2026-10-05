# C6-B3-2 frozen MultiWOZ causal comparison

Capability64 supported the manual pilot/full decision. Frozen32 is now opened;
its official FULL_RECOMPUTE corpus BLEU (reported as2.730391090452228) is evidence,
not a tuning criterion. This stage never trains, selects cases, clusters, or
changes compression settings. It imports the canonical baseline from
`b1_full_frozen32`, preserving its order, tokens, text, references and hashes.
No second FULL_RECOMPUTE generation is performed. Corpus BLEU is recomputed from
imported text with matching SacreBLEU version/signature, checked against the
canonical summary, and the canonical value is reported. The observed score is
not hardcoded as an accuracy threshold.

## Frozen inputs and guards

Defaults point to `b1_full`, `b1_full_freeze_decision.json`, `b1_full_frozen32`,
`multiwoz_plan_v2`, the prepared raw source and history semantic workload.
All adapter files must match the completed two-epoch Full manifest; the two
specified safetensors hashes are additionally enforced. The freeze decision,
completed capability64 evidence, canonical baseline output hashes, selection
file SHA, source and semantic hashes are verified before model loading.

The v2 selection must retain32 distinct target conversations, depth8/8/8/8,
16 cross-user cases,27 distinct windows with maximum frequency3, no turn-zero
targets, and both spans inside audited current-user content. Saved token indices
and character intervals are checked model-free; the pinned tokenizer's offsets
are rechecked at real runtime. There is no selector call or clustering execution.
The raw-source reconstruction comparison only validates unchanged prompt strings.
Holdout/capability/train conversation separation and the full training pool are
checked. The frozen profile SHA is mandatory even in dry-run.

## Physical modes

The new backend subclasses the unchanged C6-B2 backend. Cache construction,
lookup/decode ownership, logical event checks, mixed projection implementation,
transport encode/decode and greedy generation are reused.

| Mode | Fresh projections | Source resident |
|---|---|---|
| FULL_RECOMPUTE | Target native frozen adapter | None |
| RAW_SEMCACHE | Native source/target adapters | TOTAL QKV FP16 |
| STORAGE_KV_COMP | Native source/target adapters | Q FP16 + compressed TOTAL KV |
| TRANSPORT_QKV_COMP | LoRA delta QKV codec, then base recombination | TOTAL QKV FP16 |
| FULL_PIPELINE | LoRA delta QKV codec, then base recombination | Q FP16 + compressed TOTAL KV |

Each reuse mode builds one cold source entry and performs one lookup. Compressed
lookup decodes once, leaving Q and compressed KV resident without a raw KV copy.
The temporary decoded hit serves greedy prefill and one teacher-forced forward.
All modes retain the same raw logical capacity and frozen hit identity. Storage
uses UNIFORM_K20_V16 / B2_ANCHOR_MOD_RESIDUAL_KV with no fitting or recalibration.
Resident Q remains FP16; Q compression is deferred to C7.

The supervised sequence bound remains384 and K remains4. As in B3-1, generated
continuations may extend beyond384, within OPT's actual position limit. Greedy
settings are frozen at160 new tokens, no sampling, one beam, normal EOS. Reuse
modes follow the existing C6 raw-logit argmax implementation.

## Metrics and accounting

Primary quality is corpus SacreBLEU:13a, exp smoothing, lowercase false,
effective_order false, one reference, points0–100. This is the same metric family
under an explicit reproduction protocol, not exact SemCache paper reproduction.
Per-user and history-stratum BLEU are also recorded.

Five signed comparisons always use mode BLEU minus baseline BLEU:

- RAW − FULL_RECOMPUTE: semantic reuse.
- STORAGE − RAW: storage.
- TRANSPORT − RAW: transport.
- FULL_PIPELINE − TRANSPORT: storage after transport.
- FULL_PIPELINE − RAW: combined compression.

Paired token exact match, edit distance, aligned-position agreement and common
prefix length measure mode fidelity, not task quality. Reference-token fidelity
uses B3-1's separately tokenized leading-space response plus EOS.

One teacher-forced forward per mode uses the same canonical FULL_RECOMPUTE
continuation. C6's unchanged reducer computes KL(baseline||comparison), cosine,
top1 and top5 overlap, plus its existing margin/rank diagnostics. Logits are kept
only for the current episode on CPU and discarded after reduction; no vocabulary
arrays are written. Aggregate medians are means of per-case median KL, explicitly
labeled, not pooled medians. These diagnostics describe perturbation conditioned
on the canonical continuation, not the autoregressive cascade.

Transport accounting separates source+greedy task traffic from teacher-forced
traffic. Compressed sizes come from actual codec packets. Runtime packet bytes
and fitted-per-call CDF bytes are separate; the reported transport ratio includes
both. Raw-mode logical delta sizes are derived from fresh projection shapes and
active adapter dtype; no real network transfer is claimed. FULL_RECOMPUTE has no
transport payload. Different generated lengths imply different task traffic;
those totals are not a matched latency/capacity experiment.

Resident accounting reuses C6-B2 definitions, adding raw Q/K/V components and an
explicit compressed-frame field. Local metadata is already inside the frame and
is never added twice. The profile is shared overhead, recorded separately.
Storage summaries are per-entry means; transport totals are sums. Transport CDF
fits are recorded; storage CDF fitting is forbidden. No production latency claim.

## Outputs

- `manifest.json`: STARTING/DRY_RUN/FAILED/COMPLETE; input and output hashes,
  adapters, model/revision, git state, mode/protocol definitions, GPU/software,
  codec source hashes and runtime counters.
- `per_case.csv`: all160 cases with full episode provenance, generation,
  reference fidelity, resident/transport accounting and counters.
- `summary.json`, `summary.csv`: per-mode task metrics; JSON also includes
  causal comparisons, accounting, paired aggregates and selection audit.
- `causal_comparisons.json`: five explicit signed BLEU comparisons.
- `paired_quality.json`: compact per-case and aggregate fidelity/logit metrics.

Existing output roots are refused. Existing result artifacts are read-only.
A dry-run validates artifact bytes and structure without OPT/PEFT/codec/CUDA
imports; it does not recompute BLEU or load even a tokenizer.

## SERAPH command for a manually written job

No shell/job file is supplied or modified by this implementation.

```bash
cd /data/khuss/repos/GP_semcache
export TORCH_EXTENSIONS_DIR=/data/khuss/.cache/torch_extensions/py311_cu118
export TRITON_CACHE_DIR=/data/khuss/.cache/triton
export CUDA_CACHE_PATH=/data/khuss/.cache/cuda
export TORCH_HOME=/data/khuss/.cache/torch
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export XDG_CACHE_HOME=/data/khuss/.cache TMPDIR=/data/khuss/.cache/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$TMPDIR"
python scripts/56_run_cachegen_c6b3_2_multiwoz.py \
  --adapter-root results/cachegen/c6b3/b1_full \
  --freeze-decision results/cachegen/c6b3/b1_full_freeze_decision.json \
  --baseline-root results/cachegen/c6b3/b1_full_frozen32 \
  --plan-dir results/cachegen/c6b3/multiwoz_plan_v2 \
  --source results/workloads/multiwoz.jsonl \
  --semantic results/workloads/c6b3_multiwoz_history_semantic.jsonl \
  --profile-path results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin \
  --storage-src /data/khuss/repos/GP_semcache/.c6_storage_src/src \
  --device cuda:0 --seed 42 \
  --output-root results/cachegen/c6b3/b2_frozen32
```

For CPU-only artifact validation use `--device cpu --dry-run` and a separate new
output root such as `results/cachegen/c6b3/b2_frozen32_dry_run`. No subsets or mode
filters are supported. Frozen32 outputs must never drive further tuning,
selection, stopping or filtering. Local checks use synthetic fixtures; actual
SERAPH artifacts and GPU execution remain to be validated externally.
