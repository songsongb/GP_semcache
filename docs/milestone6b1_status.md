# M6B-1: common logical baseline comparison

This establishes baseline semantics and a CPU logical/analytical runner. M6B is
not complete. No answers, BLEU, adapter training, physical projection execution,
RTX3090 timing, direct A100 latency reproduction, or Table-II claims are produced.
Fixed workloads and dataset sources are not changed.

## Paper-defined architecture

- UD_ONLY: full base model and user LoRA inference on each user device; no shared
  cache or EdgeLoRA exchange. Paper reporting excludes OPT for resource reasons;
  this logical runner may account for OPT FLOPs without executing OPT.
- ES_ONLY: centralized base model and user adapters; performance reference that
  ignores local-adapter privacy, with no EdgeLoRA exchange or shared QKV reuse.
- FBC: semantic-unaware QKV caching with frequency + LRU under EdgeLoRA.
- SEMCACHE: semantic-aware hierarchical intent grouping and token subsequence
  reuse. The runner calls the existing M6A engine without replacing its policy.
- Table-II memory covers model parameters, adapters, QKV cache and hidden
  activations summed across ES and all UDs. Table-II latency averages per-query
  across datasets. Registry values remain immutable PAPER_REFERENCE data and
  are never simulator inputs or test targets for generated metrics.

## Reproduction choices

FBC variant `exact_block_frequency_lru_v1` uses the exact token tuple as its key.
The cache is isolated per run/model; user, source position and cluster are absent
from lookup identity. Existing sliding window extraction and earliest-start
non-overlapping hit selection are reused. All lookups precede admissions, so a
block inserted by a query becomes reusable by later queries. Duplicate misses
within a query produce one admission candidate. Overlapping hits count as block
hits but only selected non-overlapping tokens contribute savings.

Every window observation increments a lifetime frequency counter, including
misses and overlapping windows. Counts survive eviction. Admission requires its
first eligible observation (frequency >= 1) and a block that fits capacity.
Frequency has no additional ranking/threshold role in this initial variant.
Resident lookups update recency in window order; insertions become most recent.
Capacity eviction removes the least recently accessed entry, deterministically.
No semantic impact, hybrid scoring, CHU or PBR enters FBC decisions. The paper
does not specify these exact keys, threshold, frequency horizon or update order;
all are REPRODUCTION_CHOICE, not a uniquely recovered paper FBC algorithm.

M6A workload transformations and assignments are inherited and recorded in full.
The CLI uses the M6A whitespace proxy tokenizer and fixture SHA256 scalar encoder
for SemCache, not learned semantic embeddings. QKV payload defaults explicitly to
16 bits as a reproduction choice, independent of model weight precision. Logical
impact remains unavailable, with existing M6A scoring fallback and no CHU/PBR.
These are approximate policy simulations, not evidence of safe physical reuse.

## Result contract and costs

`baseline_runner.run_baseline` and `run_comparison` consume the same rows, model
spec and system config. The common `semcache.baseline.v1` schema includes scalar
metrics, typed per-metric source/scope/unit/comparability, per-query summaries,
resolved config/hash, original and executed workload hashes, model/system specs,
seed, capacity, window, ordered query/user IDs and inherited workload provenance.
Comparison asserts equal fairness metadata. It never reassigns or reshuffles users.
Seed overrides affect run configuration, not the already prepared workload.

Cache counts and ratios are SIMULATED and APPROXIMATE against paper metrics.
Analytical FLOPs, latency and bytes are NOT_COMPARABLE to Table II. Top-level
comparability is a broad logical-run label; individual metrics are authoritative.
No MEASURED timing is emitted; the existing provenance enum and metric envelope
remain compatible with M6A. Per-query summaries are debug simulation records.

Base totals retain all Eq.19 attention/FFN terms. LoRA totals cover QKV
projections. Savings reuse `projection_savings`: all-layer multiples of
6*n_reused*d^2, 6*n_reused*d*r and 4*n_reused*d. No attention/FFN savings are
claimed. Totals are before reuse, not full-model FLOPs (embedding/logits omitted).
UD_ONLY and ES_ONLY have zero exchange elements and zero reuse savings, with
null cache lookup/admission/eviction metrics because those operations do not run.
Undefined ratios (zero denominator) are null.

UD_ONLY and ES_ONLY latency is null: no single-device full-inference timing model
is implemented. FBC/SEMCACHE use the unchanged M6A Eq.19 model only when explicit
ES/UD throughput and wire precision exist. Otherwise latency is null and missing
components are listed. Available latency is a workload **total prefill** estimate,
not Table-II mean generation latency. No throughput is inferred or calibrated.

System/model/adapter/activation memory remains null. `cache_bytes` aliases logical
QKV bytes only; it is not Table-II memory. FBC peak occupancy is tracked; SemCache
peak is null because the unchanged engine exposes final query occupancy, not an
intra-query high-water mark. Zero cache bytes for uncached architectures is a
structural analytical fact. No missing value is filled from the paper registry.

## SERAPH smoke commands

Use the existing environment and repository checkout. These execute only CPU
logical comparisons with model dimensions; no model weights are loaded.

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
python scripts/24_run_baseline_comparison.py --workload results/workloads/multiwoz.jsonl --config configs/paper/multiwoz.yaml --all-baselines --max-queries 100 --seed 42 --output results/baselines/multiwoz_100.json
python scripts/24_run_baseline_comparison.py --workload results/workloads/coqa.jsonl --config configs/paper/coqa.yaml --all-baselines --max-queries 100 --seed 42 --output results/baselines/coqa_100.json
python scripts/24_run_baseline_comparison.py --workload results/workloads/snips.jsonl --config configs/paper/snips.yaml --all-baselines --max-queries 100 --seed 42 --output results/baselines/snips_100.json
```

Use `--baseline UD_ONLY` (or another enum value) instead of `--all-baselines`
for an individual run. `--max-queries` is required to keep runs explicitly bounded.
