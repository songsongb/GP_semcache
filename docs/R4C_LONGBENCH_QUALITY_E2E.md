# R4-C0/C1: LongBench-derived quality and physical OPT pilot

Status: IMPLEMENTED / CPU_TESTED / GPU_NOT_RUN. Exact local counts, dependency skips and regression outcomes are recorded in `results/r4/c/local_verification_20261031/verification.json`. This stage does not complete R4-C as a whole. Qwen C2 and repeated independent C3 experiments remain future work. No Slurm job is submitted by these scripts.

The frozen R3, R4-A, B1 and B2 implementations/results are inputs, never rewritten. Historical B2 Partial HIT strict-bitwise failures and Native PEFT strict-parity failures remain failures. All primary-performance and correctness-qualified-speedup flags stay false. The existing user_a adapter is controlled and untrained.

## Offline data and reproducibility

Manually acquire **LongBench v1** before GPU execution using the [official dataset](https://huggingface.co/datasets/zai-org/LongBench). Do not confuse this with LongBench v2. The official [evaluation mapping](https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py) and [metrics](https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py) define QA F1 for qasper/2wikimqa and ROUGE-L for multi_news. This pilot uses its own shorter, versioned prompt templates and context truncation, so every record is labeled **LONGBENCH_DERIVED_TRUNCATED_EVALUATION**, never an official leaderboard result.

Provide a local registry JSON mapping dataset names to:

```json
{"qasper":{"path":"/absolute/path/qasper.jsonl","format":"jsonl","revision":"IMMUTABLE_UPSTREAM_COMMIT","identifier":"zai-org/LongBench","sha256":{"qasper.jsonl":"ACTUAL_SHA256"}}}
```

Add 2wikimqa and multi_news entries. JSONL records must preserve official `_id`, `input`, `context`, `answers`, and `length`. SHA-256 verifies exact source file bytes; each source record has a separate digest. A registry revision is an operator-supplied upstream provenance assertion; hashes do not independently prove the upstream origin. Do not use `main` or `latest`. Verify acquisition externally and retain that revision. `format: hf_saved_dataset` supports an explicitly provided local `datasets.save_to_disk` directory, per-file relative SHA-256 map, and optional `split` for DatasetDict. Ordinary Hugging Face hub cache directories and arbitrary Python dataset loaders are deliberately unsupported. `datasets` is needed only for saved snapshots. No loader downloads data. Missing files raise MISSING_DATASET with acquisition instructions.

Record selection uses SHA-256 rank of seed, dataset, original ID. It is independent of line ordering. Requested counts cannot silently shrink. `--min-input-tokens` filters by actual full formatted input token counts (default prefix+1); optional registry `min_context_tokens` provides a context-length floor. Each registry entry may override the default QA/summary generation budget using `max_new_tokens`. All such filters and actual per-dataset budgets are recorded. Eligibility filters, exclusions and unselected IDs are recorded. `--cohort independent --exclude-cohort previous/r4c_prepared_prompts.jsonl` explicitly excludes previous IDs; an independent label alone is not proof of disjoint cohorts. Selected prompt-preparation failures abort and retain stage evidence, rather than replacing difficult examples.

C0 TokenizerAdapter is model-independent; physical C1 is OPT-only. Preparation loads the actual offline OPT tokenizer and model config from the fresh R3-A base preparation, fingerprints the vocabulary, tokenizer implementation/configuration and config. No padding or synthetic gold is added. Tokenization explicitly uses add_special_tokens=False, with masks of all ones; the actual serving tokenizer object and special IDs must match preparation. The input budget is a **cap**, not padding to an exact length. Context-only token head/tail truncation is deterministic, with an omission marker and whole-prompt retokenization to account for BPE boundary effects. Question/instruction strings survive unchanged; the retained source-token spans, context counts, retained text and final prompt hash are preserved. Counts refer to source context tokens, not the added omission marker. Gold answers are never truncated. Short eligible inputs retain their actual length. Input cap defaults to 256; 512 requires `--enable-512`. Actual `max_position_embeddings` bounds prompt plus maximum generation. Positional embeddings are never extended.

## Execution and cache objects

`CONTROLLED_PROACTIVE_PREFIX_WARMING` executes, in order for each target:

1. COLD_MISS standalone initial token-ID prefix, fixed G1 generation; complete all distributed layer forwards and receive a verified completion receipt.
2. Verify captured corrected Raw K/V against actual HF attention K/V; admit owned GPU Raw QKV into the sealed R3-B/B1 byte-capacity/LRU primitives; register the completed source on UD.
3. Matched full Distributed NO_REUSE target.
4. PARTIAL_PREFIX_HIT target using only actual suffix IDs and a private prefix KV cache, followed by normal one-token Decode.

Prefix extraction never uses text decoding/re-encoding. Targets are bound to prepared-record SHA, source request, cache entry, exact tokens/mask/positions, base weights/model, adapter matrices, tokenizer, source hashes, dtype/device/GPU, inspected OPT/cache API and completed-source proof. Cache lookup fails closed; no full-prefill fallback is counted as HIT. Longest eligible standalone prefix is selected. No slicing of longer sources, semantic matching, concurrency, compression or full-prompt C1 HIT is implemented.

For eager OPT, Raw K/V `[1,P,H]` become attention-ready `[1,heads,P,H/heads]` through view/transpose/contiguous copy. A newly constructed observed tuple/list or supported public DynamicCache.from_legacy_cache owns the target's K/V. Original Raw QKV storage remains sealed and unmodified. Source Q is retained but **never consumed**. Suffix tokens still execute embedding/positions, every attention, FFN, projection and LM head. Prefix token rows are physically removed from those target computations. Projection kernel calls still occur for suffix rows; no whole projection kernel elimination is claimed. Every layer's appended K/V and prior-history digests are checked live, including all Decode steps. No source logits/output are substituted for target inference.

C1's new protocol uses unchanged bounded loopback TCP framing and independent CPU-only UD LoRA. It permits input <=512 and generation <=128, within actual model context limits. A prefix-aware first forward has past=P and input=N-P; later forwards have past=N+step-1 and input=1. UD enforces per-layer ordering, shapes, dtype, model/adapter identity, grants and source completion. An explicit COMPLETE_REQUEST receipt verifies consumed generated IDs and EOS/length termination before a new request or admission. Frozen B2 validators are unchanged.

Both quality (greedy EOS/max) and fixed (greedy exact budget, ignores EOS) policies are explicit in schedules and results. Separate preparation/run directories are required for different policies. Never aggregate unlabeled policies. Decode throughput uses G-1 actual Decode forwards. TTFT starts before lookup and ends after first greedy token/diagnostics; E2E also includes completion receipt, text decoding and source admission where applicable. These are synchronized, **instrumented** timings including hashing and disk writes, not uninstrumented serving benchmarks. TCP is localhost, not remote edge networking. No network emulation is run.

## Evidence and claims

STRICT_AUDIT captures corrected QKV for every forward, prefill actual KV, passive attention/FFN/layer observations, logits, positions, append proofs, protocol and cache lifecycle. Decode KV is validated live and retained as a digest chain. The strict read-only auditor reconstructs every complete per-head Decode KV tensor from prefill KV plus captured Decode Raw K/V and checks the actual HF digest; repeated complete KV snapshots are unnecessary. Independent audit checks source and target prefix/suffix KV bytes, QKV/logits, actual forward and wire ledgers, completion receipts and residency. SCALABLE_EVALUATION keeps critical digests, logits, IDs/text, gold metrics, timing and cache/communication ledgers; it does not claim full tensor-level verification. By default only the first selected example is strict; dataset ordering is explicit. Use a separately bounded strict cohort for broader audit coverage.

Both profiles require the actual eager OPTAttention implementation and set output_attentions=False. This removes retained attention output tensors without selecting a new attention backend; the loaded source/config are preserved. It does not promise that eager OPT avoids internal attention-weight computation. Unsupported cache/attention APIs fail closed. Installed Transformers/OPT inspection and the output-attentions compatibility probe still require the actual GPU environment.

Cache capacity defaults to 256 MiB, counts physical Raw QKV owned bytes once and enforces LRU before cloning. Derived KV is request-owned temporary storage, reported separately. Total run evidence defaults to 2 GiB (maximum configurable 16 GiB); each forward checks actual directory usage and reserves 16 MiB of bounded ledger/control headroom. Every tensor/logit write accounts its exact bytes against that reservation, with another actual disk check after each forward and request. Directory trees are not rescanned for every individual snapshot. Existing nonempty output directories are refused. Budget exhaustion aborts with preserved evidence. No silent compact-profile downgrade occurs. CUDA allocation/peak are separate from owned cache and evidence bytes. Input datasets/base preparation outside the run directory are separate storage costs.

Gold QA uses lowercase, ASCII punctuation removal, article removal, whitespace split, Counter overlap F1; maximum over all gold answers. Empty/empty F1 is zero as in LongBench. Normalized EM is additional. Summary scoring requires `rouge==1.0.1` and uses default ROUGE-L F measure with maximum over golds, case preserved/no stemming. Its exact package version is recorded. Without rouge, only explicitly synthetic fixtures can use labeled token-LCS diagnostic; it is never reported as official summary scoring. SacreBLEU compares HIT against No-Reuse only, with version/signature; it is output fidelity, not task quality. Missing SacreBLEU is recorded as unavailable, never zero. Real GPU preflight requires these metric packages.

Numerical diagnostics retain strict serialized-logit equality separately from max_abs, relative L2, KL and argmax. Only equal-history steps count as direct numerical comparisons; later history divergence is separately counted. Strict suffix QKV comparisons require strict files; compact requests cannot acquire a full strict PASS. First divergent captured projection is reported where available, not claimed to identify an uncaptured causal operation. No numerical threshold changes or bitwise-equivalence inference from output agreement. Equal low-quality outputs do not demonstrate task accuracy.

Analysis reports per-example/dataset gold scores, output agreement, latency, wire versus tensor payload and source costs. Amortization for explicit reuse counts 1/2/4/8 is an analytical model using measured source+hit costs, not a measured natural hit rate/system throughput. The small pilot cannot satisfy the frozen eight-pair benefit requirement. No confidence/significance qualification is claimed. Native PEFT remains historically FAILED and deferred; PRIMARY_PERFORMANCE_ELIGIBLE and correctness_qualified_speedup remain false.

Natural-prefix discovery is a separate bounded offline token-prefix pair table (<=4096 records). It reports useful overlap, fractions, no-overlap records, and optional backend-supplied storage requirements. Template-only prefixes may dominate; discovery records are not physical HITs or traffic hit rates. No natural shared contexts are manufactured.

## Commands and artifacts

```bash
# Local synthetic TCP/causal CPU fixture, never GPU qualification
python3 scripts/106_run_r4c_quality_e2e.py --phase fixture --test-fixture-only \
  --output-dir results/r4/c/my_unique_fixture
python3 scripts/106_run_r4c_quality_e2e.py --phase analyze \
  --inputs results/r4/c/my_unique_fixture --output-dir results/r4/c/my_unique_analysis
PYTHONPATH=src:tests python3 -m unittest tests.test_r4c_dataset tests.test_r4c_metrics tests.test_r4c_session tests.test_r4c_cache tests.test_r4c_opt_contract tests.test_r4c_processes -v

# Explicit local dataset check (no model loading)
python3 scripts/105_prepare_r4c_longbench.py --validate-datasets-only \
  --registry /absolute/path/registry.json --output-dir results/r4/c/my_unique_dataset_check

# Operator command only; not submitted by Codex. Create absolute logs directory first.
R4C_ENABLE_GPU=1 R4C_DATASET_REGISTRY=/absolute/path/registry.json \
  sbatch jobs/r4c_opt_quality_e2e.sh
```

The opt-in SERAPH job uses batch_eebme_ugrad, one allocated RTX3090, 4 CPU cores, 64 GB, 3 hours, conda semcache, offline Hugging Face and absolute durable stdout/stderr. It checks data before model preparation, prepares a fresh model/adapter, actual prompts, CPU contracts, runs QA then summary groups in one persistent ES, independently audits and analyzes. Defaults: two examples/dataset, input cap256/prefix128, QA32/summary64 quality policy. No sweeps or later stages launch. Existing R3-A Native reference artifacts in fresh preparation and all historical parity failures are preserved; no new C-target Native parity is claimed.

A completed run produces r4c_dataset_manifest.json, r4c_prepared_prompts.jsonl, r4c_prefix_coverage.json, r4c_schedule.json, r4c_results.json, r4c_decision.json, r4c_quality_by_dataset.json, r4c_quality_pairs.jsonl, r4c_fidelity.json, r4c_latency_contrasts.csv, r4c_communication.json, r4c_cache_costs.json, r4c_audit.json, r4c_limitations.json, plus immutable r4c_output_manifest.json. `es/` holds both endpoint wire traces, per-step records, requests, cache events, logits, selected tensors, process lifecycle and stage/failure files. An analysis-only directory holds recomputed reports and input identities without changing the source run. Failure outputs keep partial evidence, explicit failing stage and false qualification gates.

## Future boundaries

C2 is not implemented: establish Qwen3-4B distributed No-Reuse projection/CPU-LoRA/TCP baseline first (C2-0), then real generation and installed cache behavior (C2-1), then inspect GQA heads, projection widths, RoPE order, cache offsets, adapter and actual attention-ready KV before prefix reuse (C2-2). OPT reshape helpers are confined to C1 and cannot serve as a generic Qwen KV adapter. Long contexts 1K/2K/4K require runtime config and RTX3090 memory feasibility (C2-3). C0 data/prompt/metrics remain reusable through a new tokenizer/model adapter. No OPT-to-Qwen latency equivalence is assumed.

C3 must supply fresh excluded-ID cohorts, independent repeats/seeds, at least eight independent pairs per qualified condition, and confidence intervals, while separating models/policies/profiles. This task only supplies per-pair/cohort identities and explicit cost analysis; repeated experiment orchestration and confidence qualification are not implemented. SCBench multi-turn/concurrency (D) and optional Mistral/CacheBlend comparison (E) are not implemented.

## Changed files and local validation

All implementation changes are additive. No frozen R3/R4-A/B1/B2 source, test, job, documentation or prior result is edited. Verification compares the pre-task SHA-256 snapshot of 19,416 existing files and the base git tree. New failed fixture diagnostics are preserved alongside successful runs under the separate C directory.

| Files | Purpose |
|---|---|
| `src/semcache/edgelora/r4c_dataset.py` | Verified JSONL/HF snapshot loading, source provenance, deterministic selection |
| `src/semcache/edgelora/r4c_prompts.py` | Tokenizer boundary, derived templates, context truncation, exact token-prefix extraction |
| `src/semcache/edgelora/r4c_prepare.py` | Cohort manifests, OPT preparation adapter, explicit synthetic fixtures |
| `src/semcache/edgelora/r4c_metrics.py` | Independent QA/ROUGE gold scoring and numerical diagnostics |
| `src/semcache/edgelora/r4c_workload.py` | Quality/fixed policies, proactive schedule, natural-prefix discovery |
| `src/semcache/edgelora/r4c_session.py` | C1 versioned protocol, completed-source grants and EOS receipts |
| `src/semcache/edgelora/r4c_cache.py` | Variable-prefix adapter over frozen sealed byte owners and B2 proofs |
| `src/semcache/edgelora/r4c_generation.py` | Actual suffix-only OPT prefill and bounded real decode |
| `src/semcache/edgelora/r4c_dynamic_ud.py` | Independent CPU LoRA endpoint and C1 ordering |
| `src/semcache/edgelora/r4c_runtime.py` | ES lifecycle, actual tokenizer binding, cache/source accounting, invalid HIT controls |
| `src/semcache/edgelora/r4c_evidence.py` | Strict/scalable storage, real disk budget, immutable output hashes |
| `src/semcache/edgelora/r4c_identity.py` | Source fingerprints |
| `src/semcache/edgelora/r4c_analysis.py` | Read-only wire/shape/KV/cache audit, gold/fidelity/performance/cost reports |
| `scripts/105_prepare_r4c_longbench.py` | Offline data integrity and prompt-preparation CLI |
| `scripts/106_run_r4c_quality_e2e.py` | Read-only preflight, CPU fixture, opt-in GPU run, independent analysis CLI |
| `jobs/r4c_opt_quality_e2e.sh` | Opt-in SERAPH launch and stage-aware failure record |
| `tests/test_r4c_dataset.py`, `test_r4c_metrics.py`, `test_r4c_session.py`, `test_r4c_cache.py`, `test_r4c_opt_contract.py`, `test_r4c_processes.py` | C0/C1 fixture, negative, protocol, numerical and lifecycle regressions |
| `docs/R4C_LONGBENCH_QUALITY_E2E.md` | Execution semantics, commands, evidence scopes and future boundaries |

The complete frozen regression attempt includes a historical R4-A test that expects dependency validation on a branch rejected by its frozen branch gate. That failure reproduces from an untouched R4-B2 export. Two legacy R3-C-F calibration setup errors occurred during the broad run; their unchanged classes pass when retried alone, and also pass in the baseline export. No frozen branch or calibration thresholds are adjusted. Exact logs and counts are retained under the verification directory.

This checkout has no configured qasper/2wikimqa/multi_news files in `data/raw` or `data/processed`, and no default HF datasets cache. Torch, Transformers, NumPy, PEFT, datasets, rouge and SacreBLEU are unavailable locally. The optional actual-OPT, saved-HF-snapshot and official-ROUGE checks are therefore skipped. CPU fixtures use no fake real-data provenance; official summarization scoring, actual OPT tokenizer/model loading, installed OPT/cache compatibility and all allocated GPU measurements remain unverified here.

A separate durable fixed-policy fixture can cover the complete initial generation budgets:

```bash
python3 scripts/106_run_r4c_quality_e2e.py --phase fixture --test-fixture-only \
  --fixture-policy fixed --fixture-qa-budget 32 --fixture-summary-budget 64 \
  --output-dir results/r4/c/my_unique_g32_g64_fixture
```

Expect 9 synthetic requests / 259 forwards for this fixture, versus 18 requests / at most 390 forwards for the suggested six-real-example quality pilot (source G1, four QA targets and two summary targets each with No-Reuse and HIT). Natural EOS can shorten the real quality pilot. These are bounded planned counts, not GPU results. Per-condition reports retain model, cohort/seed, policy/profile, input/prefix lengths, sample counts and an explicit unqualified confidence field for future C3 analysis. The optional extra `r4c_performance_groups.json` contains these groups without mixing policies or evidence profiles.
