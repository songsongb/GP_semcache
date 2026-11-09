# D5 Continuous NATURAL — Phase 2 CPU integration

Implementation/fixture assessment: **PASS**. Local real-data preparation,
expanded workload seal, and scientific sharing conclusions: **BLOCKED**, pending
the existing SERAPH assets. The user reports that Phase 1 real-data validation
passed at `results/r5/i5r/d5_natural/continuous_phase1_seraph_cpu_02`; this phase
requires and independently binds that sealed PASS when run on SERAPH. Its results
and all historical I5/D5 artifacts are unchanged.

GPU physical HITs, measured admissions/evictions/CHU/PBR, attention Impact,
per-layer input equality, GPU tensor equivalence and latency are **NOT_MEASURED**.
There is no Phase 3 implementation, GPU runner modification, Slurm action or push.

## Files and scope

- `r5_continuous_natural_phase2.py`: real-input binding, deterministic expansion,
  sharded sealed plan, independent reconstruction, CPU lifecycle, resource and
  comparison reporting.
- `r5_continuous_natural_sharing.py`: disjoint w3 counts, offset-verified regions,
  coverage, relationships, sharing permissions and three statistical views.
- `scripts/155_r5_continuous_natural_phase2.py`: only `prepare` and `cpu-validate`.
- `tests/test_r5_continuous_natural_phase2.py`: 17 explicitly synthetic CPU fixture
  tests. Fixtures are confined to tests; production has no fixture CLI flag.
- [Non-prefix feasibility and proposed safety specification](R5_D5_PHASE2_NONPREFIX_SAFETY.md):
  inspection of current operators plus a focused later test plan; no new operator.

Phase 1 components are reused without modification. GPU code, policies,
EXACT checks, numerical thresholds, adapters, datasets and job scripts are unchanged.

## Expanded workload and selection

| Dataset | Preserved prefix | New unused samples/invocations | Total invocations | Distinct original samples | Same-sample replay pairs |
|---|---:|---:|---:|---:|---:|
| Qasper | 20 | 30 | 50 | 40 | 10 |
| 2WikiMQA | 20 | 30 | 50 | 40 | 10 |
| MultiNews | 20 | 30 | 50 | 40 | 10 |

These counts are enforced and tested design invariants, not claims about a locally
created real-data plan. A valid real preparation will contain 150 source
invocations, 120 distinct original samples, and 300 matched A/C dispatches.

Preparation verifies the original I5 inventory, dataset preparation provenance,
expanded 30-user/3-adapter preparation and the sealed Phase 1 PASS. It rebuilds
Phase 1 sessions and compares every stored invocation envelope, not just IDs.
Source inventories and the prior CPU audit receipt are included in the new plan.

The I5 manifest did not directly persist literal `input_limit` and
`allow_context_truncation` flags. The CLI therefore requires the original recipe
explicitly; it never guesses a new truncation rule from observed lengths. The
operator must supply the settings used for the original preparation. For the
documented 512-token LongBench profile these are `--input-limit 512
--context-truncation allow`. The implementation calls the existing I5 `select()`
with the original seed, tokenizer, context/generation budgets and that recipe,
then requires exact reproduction of **all original selection-report fields**
(including eligibility/exclusions/order) and the first ten prepared records.
This corroborates the supplied recipe; it does not prove a missing historical
flag had a uniquely recoverable value. A mismatch fails closed.

It selects the first 40 eligible original samples, requires the first ten to
match the original first ten, and appends only ranks 10–39 after the intact
20-invocation Phase 1 prefix. No token-overlap analysis participates in selection.
The existing prompt builder, context-only head/tail truncation, full retokenization,
I5 generation policy and deterministic user/adapter routing are reused. Missing
or insufficient real samples yield BLOCKED; invalid identities/recipe/provenance
yield FAIL. No replacement, padding, priming, deduplication or extra replay occurs.

The original first 20 envelopes and payloads remain exactly equal to Phase 1,
including historical block/request/pair IDs and sequence-number provenance. New
requests use the original I5 task builder, and are explicitly marked
`PHASE2_EXTENSION_FROM_ORIGINAL_DATASET`. They have no historical block ID/index.
New checkpoints have hashed `D5C2_EXTENSION_...` IDs and a distinct checkpoint
kind; they are never fabricated historical blocks. The original hash-ranked
strict-capture selection is retained for extension checkpoints, while future D5
strict evidence still applies to every executed request.

The inherited session invocation namespace extends from indices 0–19 to 0–49;
A/C runtime suffixes remain unique across the 300 dispatches. Prefix IDs are
intentionally stable across versions. Across artifacts, evidence must be joined
by `(plan_id, invocation_pair_id, condition)` and a future run identity, rather
than treating an invocation ID as a globally unique execution across experiments.
New original samples and their complete records remain in the provenance shards.

## Continuous CPU lifecycle

The Phase 1 scheduler/real empty `PhysicalOwner` and `SemanticIndex` probe are
reused. The new checkpoint metadata is adapted only in a temporary scheduling
view. Original provenance is not rewritten. Four historical boundaries plus six
five-request extension boundaries give ten checkpoints per dataset/condition,
60 checkpoints across all six condition sessions. Every checkpoint retains the
same owner/cache/policy/history/semantic state roots. A/C order is identical;
state is freshly initialized at dataset/condition boundaries and disposed on
failure without retry. No request-repeat branch resets state.

The CPU probe uses metadata-only model/channel placeholders. Additional fixture
tests advance real arrival clocks/frequency state across all 50 requests and
extension checkpoints without admitting entries or inventing Impact. Neither
the probe nor these tests establish actual GPU cache continuity. Native target
autoregressive past KV must still start fresh per request in any later runtime.

PBR stays 100 **per adapter**. No adapter can reach that trigger in a 50-total-
request session. CPU reports show actual planned routing counts and state that
the trigger is unreachable at this size; observed policy events remain unmeasured.

## Static matching definitions and denominators

For equal w3 occurrences at source position `sp` and target position `tp`, with
common causal prefix length `L`, the Phase 1 categories remain disjoint:

| Category | Definition |
|---|---|
| A: shared prefix | `sp == tp` and `tp+3 <= L` |
| B: same-position non-prefix | `sp == tp` and `tp+3 > L` |
| C: shifted-position non-prefix | `sp != tp` and `tp+3 > L` |
| D: shifted occurrence targeting prefix | `sp != tp` and `tp+3 <= L` |

Every matching source-occurrence/target-occurrence pair belongs to exactly one
category. Occurrences are counted algebraically by gram/region to avoid expanding
the full Cartesian product. A test compares those counts to a brute-force oracle.
The CPU audit checks totals against the unchanged Phase 1 `static_analysis()` for
both the original 20 and expanded 50 requests.

Every category reports raw occurrence pairs; unique `(target invocation, window
start)` positions; maximum-cardinality greedy non-overlapping w3 windows and their
token coverage; and the token union coverage of all matching target windows.
The two coverage metrics differ: three overlapping windows covering five tokens
have union coverage five, while greedy disjoint w3 coverage is three. These are
static coverage counts, not 64-candidate M-selection forecasts or resident grants.

Each metric has separate all-pair, same-adapter, authorized (same user or permitted
cross-user), authorized cross-user, and sharing-rejection slices. Adapter-pair and
original-sample-relationship axes have the same metrics and ordered-pair
denominators. Source/target invocation and original sample IDs remain in pair
shards. Cross-adapter transfer is forbidden; same-adapter cross-user reuse requires
the original matching share group and both users' permission. Slices overlap and
must not be added as disjoint populations. Distinct categories partition occurrence
pairs, but their target-token coverages can overlap.

| Statistical view | Phase 1 pairs/dataset | Phase 2 pairs/dataset | Interpretation |
|---|---:|---:|---|
| Replay-inclusive | 190 | 1,225 | All earlier-source/later-target invocation pairs |
| Distinct-sample-only | 180 | 1,215 | Excludes ten same-sample experimental replay pairs; remaining invocation multiplicities retained |
| First occurrence per original sample | 45 | 780 | Removes experimental repetition weighting entirely; 10 versus 40 original samples |

The essential distinct-sample-only view excludes replay pairs but is not itself
a fully deduplicated sample-pair denominator. The third view prevents those
remaining multiplicities from being mistaken for independent natural samples.
The actual workload always retains all 50 invocations.

Relationships partition source-target pairs into same-sample experiment replay,
identical request content from distinct original IDs, different requests sharing
the complete original context, and different requests with different contexts.
Identical request content means equal token IDs, prompt text and generation
policy, independent of routing. Equality of derived truncated
prompts need not imply equality of raw contexts, which is reported separately.
Matching-request counts distinguish no-match pairs in those relationships.
Shared templates are an orthogonal token-region classification, not a mutually
exclusive sample relationship.

## Document versus template overlap and MultiNews

Token regions are assigned only after reconstructing the exact preserved prompt
components and verifying the original tokenizer's offset-mapping token IDs.
Labels distinguish instruction template, document/context, question, answer/
summary formatting, inserted truncation marker, mixed boundaries and UNKNOWN.
For truncated prompts, the retained context is reconstructed using the original
context token IDs, retained-token count, original decode options and omission
marker. An unsupported offset tokenizer or failed mapping yields UNKNOWN rather
than independent-tokenization guesses. No alternate tokenizer is downloaded.

`verified_context_to_context` requires all three tokens in **both** windows to be
verified document/context. Known templates, omission markers, questions, mixed
boundaries and UNKNOWN are excluded. This is a conservative verified lower bound.
`verified_non_template_to_non_template` additionally allows question content.
Known-template-involved and unknown-region counters remain explicit; mixed and
unknown slices may overlap. Unknown mappings must not be interpreted as evidence
of no document sharing.

Real-data values for A/B/C/D, replay/template removal and the MultiNews remaining
count are **BLOCKED locally**. No fixture count substitutes for these results.
The SERAPH report provides, for each view, MultiNews shifted non-prefix raw counts,
verified document/document counts, authorized verified document/document counts,
unique target windows, disjoint token coverage and UNKNOWN counts, alongside the
same Phase 1 metrics. It supplies quantitative evidence without inventing a
threshold for “substantial.” A large replay-inclusive C count alone is insufficient.
Even a large verified document C count still fails current EXACT position rules.

## Resource/evidence assessment

Per-dataset reports apply the unchanged Phase 1 resource formula to the actual 50
tasks/routes, preserving every limit. They separately show dense ES and UD payload
forecasts, cumulative condition/paired evidence, file count, GPU memory, original
guard results and individual limit checks. UD payload estimates exclude metadata;
the ES estimate includes conservative reserves. Native memory/timing are unmeasured.

The following are **analytical 512-token upper envelopes**, not selected real
lengths/routes or GPU experiments. The adapter example assumes 18 of 50 requests
route to one adapter; actual counts are recorded by the SERAPH audit.

| Quantity | Qasper / 2Wiki cap 32 | MultiNews cap 96 |
|---|---:|---:|
| Forwards per condition | 1,600 | 4,800 |
| Conservative GPU forecast | 11.10 GiB | 11.12 GiB |
| Condition evidence forecast | 125.82 GiB | 149.66 GiB |
| Paired A/C evidence | 251.63 GiB | 299.31 GiB |
| ES evidence for an 18-request adapter | 34.85 GiB | 42.21 GiB |
| UD dense payload for that adapter | 10.44 GiB | 11.67 GiB |
| Existing run file forecast | 1,454,848 | 4,347,648 |
| 5s/forward planning time, one condition | 8,000s | 24,000s |
| 5s/forward planning time, paired | 16,000s | 48,000s |

The 5s/forward assumption comes from the I5 planning convention; it is explicitly
an estimate and excludes initialization, policy, strict evidence and scoring
overhead. It does not bound measured completion time. Under that assumption even
the 32-token case exceeds the existing 7,200s worker / 14,460s paired guards;
the 96-token case also exceeds the 4h30 allocation substantially.

Existing evidence guards remain 16 GiB/adapter ES, 4 GiB/adapter UD,
32 GiB/condition, 64 GiB/paired run, 500,000 files and 96 GiB initial free disk
(32 GiB reserve). The envelope fails them without any limit increase. Actual
lengths and routing still require assessment. All three paired datasets at these
envelopes would require about 802.58 GiB before extra reserve/metadata beyond the
forecast; do not infer that the old preflight covers the expanded campaign.

The cache remains 512 MiB/adapter with 64 entries/candidates. A three-token total
FP16 QKV entry is 1,474,560 bytes; 64 entries use 90 MiB/adapter, 270 MiB across
three full owners. The conservative GPU forecast reserves the original 1.5 GiB
cache capacity and remains independent of request count at fixed input limits.
At input 512, per-layer attention is 16 MiB and streamed FP64 Impact working
estimate 80.375 MiB, below 128 MiB/1 GiB guards. Peak GPU <22 GiB, host available
>=32 GiB, JSON <=16 MiB, tensor-reader <=128 MiB and per-request metadata/ledger
<=256 MiB guards remain unchanged.

Continuous state exposes more earlier requests but does not grow storage without
bound: admitted keys, capacity, F/A/I/S eviction and duplicate-key ownership still
limit residency. No hit probability, occupancy, eviction or speedup is inferred.
The current one-block GPU approval/runner is also structurally unsuitable for
50-request dataset sessions; extension checkpoints do not override its cap.

Minimum later GPU work: a separately reviewed bounded dataset-session runner and
approval; full provenance-aware UD/evidence/numerical audit; lossless bounded
evidence packing and/or explicitly reviewed budgets if actual forecasts fail;
cumulative accounting across checkpoints, revised free-disk/file/reader preflights;
and deadlines justified by an authorized bounded pilot. Preserve all numerical
evidence and failure seals, target-specific native past, sharing/EXACT rules and
memory checks. Never fix the evidence problem by resetting caches/accounting,
shrinking requests, shortening generation or omitting numerical validation.

## Preparation, validation and outputs on SERAPH

Use the existing offline CPU environment. Use the original `D5N_MULTI` value
from the prior `natural_plan.json` (`multi`); no adapter/dataset preparation is
requested. These commands contain no GPU/Slurm action. Choose new output paths
for every attempt; existing paths are refused, and source/output overlap is
checked before writing to an original asset.

```bash
cd /data/khuss/repos/GP_semcache
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=42

PYTHONPATH=src:tests python -m unittest \
  test_r5_continuous_natural_phase2 test_r5_continuous_natural -v

python scripts/155_r5_continuous_natural_phase2.py prepare \
  --original-experiment results/r5/i5/plan_longbench3_smoke_v1 \
  --multi "$D5N_MULTI" \
  --phase1-audit results/r5/i5r/d5_natural/continuous_phase1_seraph_cpu_02 \
  --input-limit 512 --context-truncation allow \
  --output-dir results/r5/i5r/d5_natural/continuous_phase2_seraph_plan_01

python scripts/155_r5_continuous_natural_phase2.py cpu-validate \
  --experiment results/r5/i5r/d5_natural/continuous_phase2_seraph_plan_01 \
  --output-dir results/r5/i5r/d5_natural/continuous_phase2_seraph_cpu_01
```

The preparation is separately versioned as `D5_CONTINUOUS_NATURAL_PHASE2_CPU_V1`.
It writes `phase2_plan.json`, source/input fingerprints, per-dataset `selection.json`,
`session.json` and 50 individual invocation shards, then seals/verifies the tree.
It does not create an expanded real plan if eligible samples are insufficient.
The CPU validator verifies the tree, source and original bindings, reconstructs
selection/payloads independently, checks all 300 longer-ID transport envelopes,
validates lifecycle, computes metrics and seals a separate audit.

Audit outputs: root `phase2_validation.json` and `phase2_report.md`; per-dataset
`summary.json`, `regions.json`, `phase2/sharing_summary.json` and 1,225 pair shards;
plus `phase1_consistent_metrics/sharing_summary.json` and 190 baseline pair shards.
Every category/axis/slice is in the JSON reports, while the Markdown highlights
replay/template removal and the MultiNews question. Missing metric cells in
non-aggregate sparse axes denote zero; aggregate reporting includes explicit zeros.
Existing 16 MiB JSON limits are preserved by sharding, not relaxed.

PASS means CPU workload/lifecycle/static evidence validated, with
`future_gpu_ready=false`; resource guard failures remain visible. BLOCKED means
missing real inputs/dependencies or insufficient eligible data, exit 2. FAIL means
invalid provenance, recipe, content, source/plan seal or invariants, exit 2. A failed
attempt is sealed separately where a safe output can be created; unsafe output
destinations are rejected without writing. No CPU receipt authorizes GPU execution.

## Local test results and conclusion

The Phase 2 suite has 17 tests covering exact expansion/provenance/IDs, deterministic
next-eligible selection, shortages/invalid recipes/datasets, state persistence and
A/C isolation, offset/mixed/unknown mapping, replay relations, all denominators,
coverage, sharing rejection, guard failures, end-to-end fixture plan reconstruction,
tamper rejection and exclusion of GPU CLI actions. Fixtures are explicitly labeled
and their temporary sharing artifacts are deleted after tests.

Run Phase 1's 17 tests and selected existing D5 corruption, disposal, ownership,
causal/input-rejection regressions alongside them. Native Torch/Transformers/NumPy
are absent locally; no native-model or real-data result is claimed. All original
source files and historical results remain unchanged. The final local sealed
validation receipt records actual test counts and selections.

**PASS** for the CPU implementation and selected fixture regressions. **BLOCKED**
for the sealed real Phase 2 workload and substantive MultiNews/natural-sharing
conclusion until the commands above run on SERAPH. Exact non-prefix physical reuse
is a proposal only, and Phase 3 has not started.
