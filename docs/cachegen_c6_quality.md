# C6-A matched task quality

C6 measures task quality under controlled **untrained** rank-8 `user_a`/`user_b`
fixtures. Every report records `trained_adapter=false` and
`paper_bleu_claimed=false`. Absolute BLEU is a diagnostic; interpret matched
mode deltas. This is not a SemCache paper reproduction.

The entry point is `scripts/49_run_cachegen_c6_quality.py`. Selection and metrics
live in `c6_quality.py`; model and codec execution is lazy in `c6_runtime.py`.
No dependency installation or downloads are performed. Runtime loading requires
an existing local pinned OPT/tokenizer snapshot. Default generation is greedy,
FP16, seed 42, at most 32 new tokens, stopping normally on EOS.

| Mode | Source/target fresh projection communication | Resident source entry | Target reuse |
|---|---|---|---|
| FULL_RECOMPUTE | native target PEFT | none | none |
| RAW_SEMCACHE | native PEFT, raw-delta equivalent | TOTAL Q/K/V FP16 | frozen w3 |
| STORAGE_KV_COMP | native PEFT, raw-delta equivalent | TOTAL Q FP16 + frozen TOTAL K/V bytes | frozen w3 |
| TRANSPORT_QKV_COMP | actual LoRA delta encode/decode, then ES base recombination | TOTAL Q/K/V FP16 | frozen w3 |
| FULL_PIPELINE | actual LoRA delta encode/decode, then ES base recombination | TOTAL Q FP16 + frozen TOTAL K/V bytes | frozen w3 |

Transport includes source prefill and the target's fresh prefill/decode rows.
Cached projection rows bypass target transport. Source hidden states evolve
through the reconstructed transport path; compression is not retroactively
applied to captured TOTAL projections. Raw native PEFT preserves the existing
raw base-plus-delta semantics. Transmission is the existing single-process
emulation, not a network measurement.

The mixed projection context has an optional fresh-row reconstruction callback;
existing callers retain their native path. C6 uses existing CacheEntry,
GlobalCache insertion/lookup, ExactTokenMatcher and mixed projection execution.
Each mode starts a cold single-entry cache with identical raw logical capacity.
Only the frozen source block is presented for admission (normal admission must
succeed); no other source or target blocks compete. This controlled admission
scope isolates quality from capacity and is not a full workload policy replay.
The compressed resident object owns Q and the real C2 compressed payload;
decoding produces a temporary view for the existing mixed path.

## Frozen selection

Read the prepared measured M9-B workloads in file order. Candidate sources must
precede targets, differ in source ID, query text and full token sequence, and share the measured
cluster and exact three physical token IDs. Both spans start at least at token
3, and contain no special tokens. Choose one candidate per target by ascending
source index, source span, then target span. Round-robin over groups in their
first eligible appearance order, preserving target order within each group:
SNIPS intent, MultiWOZ dialogue. Null dialogue IDs use independent source-ID
groups. Selection never reads generations, codec sizes or quality scores.
User assignment reuses M9-B's two-user seeded assignment and maps `user_000` /
`user_001` to the existing fixtures. M9-B strict safety with no external evidence
is reported rejected and diagnostic-only, not used as the execution gate.

`selection.json` is written before model/codec execution. All requested modes
use it unchanged. The default is 32 episodes per dataset; insufficient eligible
cases or missing references fail rather than reducing the cohort. All input
rows must have string references. Output directories must be empty.

## Metrics and artifacts

* `selection.json`: ordered episodes, IDs, users, cluster, key, w3 IDs, spans,
  reference hashes/text, selection rule, mode factors, scientific contracts.
* `per_case.csv`: episode × mode, generated text/IDs/length, reference, actual hit
  count, logical event hash, transport/storage factors and resident byte counts.
  FULL_RECOMPUTE has zero selected/executed hits.
* `paired_quality.csv`: generated-sequence equality, common prefix, first
  divergence (zero-based, null for identical sequences), position agreement
  over the shorter aligned length, Levenshtein distance divided by maximum
  length, and teacher-forced diagnostics. Empty identical sequences agree.
* `summary.json` / `summary.csv`: dataset × mode corpus BLEU, delta vs RAW,
  SNIPS whitespace-strip/casefold exact-label accuracy and secondary fidelity
  means. JSON also lists all five requested causal comparisons separately.
  There is no mixed-dataset overall BLEU.
* `manifest.json`: model/tokenizer provenance, workload/selection/output hashes,
  profile and codec provenance, fixtures, git/software state, generation/BLEU
  protocols, CDF counters and run status. Failed execution records an error.

SacreBLEU is called once per dataset/mode on the aligned corpus through the
existing evaluator: 13a, exp smoothing, effective_order=false, lowercase=false.
Version and signature are saved in the summaries and manifest. Tests mock the
scorer to verify corpus aggregation without claiming measured BLEU.

Teacher forcing uses RAW's greedy continuation in every mode. A single mixed
forward receives `prompt + raw_generated[:-1]`, extracting logits beginning at
`prompt_length - 1`. Prompt reuse remains fixed and continuation rows are fresh.
This computes mean/median KL(RAW || mode), absolute logit differences, cosine,
top-1 agreement, top-5 set overlap, RAW top-1 rank and top-two margins. Rank uses
one plus the count of strictly larger logits. These are secondary perturbation
diagnostics, not task metrics. The extra FULL_PIPELINE vs TRANSPORT comparison
and RAW vs FULL_RECOMPUTE comparison explicitly name their baseline. Transport
quantization sees the complete fresh teacher-forced sequence in that forward;
its packet grouping differs from incremental free-running decode. This scope is
recorded rather than claimed to reproduce token-by-token transport packets.

## Storage compatibility and discrepancies

Current `main` lacks C2/C4/C5 modules and the frozen profile. `--storage-src`
accepts the existing exported checkout's **src directory**, extending the local
cachegen package search path. It uses C2 `FrozenK20V16Codec` with
`FAST_PY_BITEXACT`; it does not implement an alternative codec. The profile SHA
is enforced before model loading. Runtime storage operations reject the existing
`fit`, `fit_profiles`, `cdf_from_counts` and named CDF-building calls. Transport's
existing per-packet runtime CDF fitting is allowed and counted separately.

Script 48 discards resident Q and recomputes current Q on HIT. C6 intentionally
uses the C2 resident-Q-FP16 design, retaining source Q. These representations
are scientifically different; Q storage compression is deferred to C7.
C4's exact-repeat prefix cases and C5's capacity/admission experiment are not
used as C6's episode set. Their source was inspected read-only using `git show`.

## CPU checks and SERAPH dry run

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_cachegen_c6_quality.py -v
```

Tensor tests skip when PyTorch is absent. No test downloads anything. A dry run
loads no model, encoder or codec, does not require the profile to exist, and
marks profile validation as pending. It writes selection and manifest only,
prints diversity, users, episode IDs, spans/tokens, references and mode plan.

On SERAPH, from the current checkout containing this change:

```bash
cd /data/khuss/repos/GP_semcache
python3 scripts/49_run_cachegen_c6_quality.py \
  --snips results/workloads/m9b_snips_semantic.jsonl \
  --multiwoz results/workloads/m9b_multiwoz_semantic.jsonl \
  --profile-path results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin \
  --per-dataset 2 --max-new-tokens 32 --device cpu --semantic-device cpu \
  --seed 42 --output-dir results/cachegen/c6/quality_dry_run --dry-run
```

Choose a new empty output directory for another run. Real execution additionally
needs the external storage export if still absent on main, compatible installed
CUDA transport dependencies, and local model files. No real C6 experiment has
been run or quality values reported during implementation.
