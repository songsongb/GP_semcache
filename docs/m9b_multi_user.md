# M9-B logical multi-user simulator

`37_run_m9b_multi_user.py` runs 10/25/50 logical users, 200/500/1000 Mbps,
and both requested cost scenarios. It never loads models, tokenizers or datasets
from the network. Rank is 8, w is 3, capacity is 20 GiB (PAPER_REFERENCE).

## Architecture and evidence boundary

One workload/cache pass per dataset/user count feeds six paired output rows.
Source query order is retained. SNIPS uses the existing SHA256-seeded query-level
round-robin assignment. The pre-full-run audit found that MultiWOZ also used
round robin and split conversations; it now uses the existing seeded group-balanced
allocator. Every non-null conversation_id stays on one logical user, including
IDs occurring across source splits. Null/missing IDs become singleton query
groups. Assignment-only tagged group keys do not modify semantic rows. Groups
are ordered by the existing seeded SHA256 rule and assigned to the least-loaded
user (query count; lowest user index breaks ties). Original raw user IDs remain
provenance only. The policy is recorded in summaries and the manifest. Increasing users partitions the **same fixed corpus**;
it does not increase total traffic or independently draw more user corpora.
Consequently candidate hits are invariant with user count under this shared
cluster/key policy. This is a controlled fixed-load experiment, not evidence
about hit growth under increasing aggregate traffic.

Reused components: `SubsequenceExtractor`, `GlobalCache`, `CacheEntry`,
`AdmissionPolicy` (F/I/S), `EvictionPolicy` (F/I/A/S), `CacheMetricManager`,
`QueryImpactHistory`, existing user assignment, M9 network tensor accounting,
and M9-A.2 `recompose`. Policy weights remain the repository defaults.

Admission F counts window appearances over 100 queries. Eviction F counts
actual reuse and therefore remains zero in the conservative workload run.
Candidate hits do not refresh actual-reuse age or frequency. All lookups happen
before this query's admissions. INSERT events count admissions, including
entries subsequently evicted; REJECT counts failed new-key insertion attempts.
Already resident keys are not attempted admissions. Occupancy is post-eviction.
No tensors are allocated. Each logical QKV entry costs 3 projections × 3 tokens
× 2560 hidden units × 32 layers × 2 bytes, excluding index/Python overhead.

I is unavailable in the supported workload schema and remains None; the existing
cache scorer maps missing I to zero. This is **not measured attention**.
Artifacts containing attention need a future validated impact importer; this
version does not consume such fields. rho=.8 and Lambda=100 configure the
existing manager, but CHU has no actual observations to process. PBR is invoked
at 100 global queries as a REPRODUCTION_CHOICE interval approximation and
cannot update missing impact. Cluster assignments replay the semantic artifact;
cluster update interval 100 is recorded, but online clustering is not performed.
No missing paper clustering behavior is synthesized.

## Three levels

* Candidate: a pre-admission lookup finds `(cluster_id, token_ids[ start:start+3 ])`.
  Every sliding window counts, including overlapping occurrences.
* Safety eligible: same logical user, explicit same adapter, identical full
  tokenized prompt, matching source start position, and an external validated
  fixture identity. Same user or position alone is insufficient. The CLI has
  **no correctness-evidence importer** and supplies an empty evidence set, so
  workload safe counts are zero. Untrusted workload flags cannot enable reuse.
* Cost effective: safety eligible AND supported reuse cost strictly less than
  recomputation cost. Ties, missing costs, and unsafe candidates recompute.
  The gate is tested independently; this version cannot execute workload reuse
  because its safety/cost evidence is unavailable.

All three rates divide by lookup_count (zero if there are no windows).
Token counts use distinct covered positions rather than multiplying hit windows
by three. Executed reused/fresh token counts and network savings remain zero/full
under the conservative policy. `candidate_potential_saved_bytes` is a separate
optimistic upper bound obtained from the union of candidate positions, with no
correctness claim. Full h0/hL transfers remain in baseline and reuse bytes.

The pre-full-run audit also adds `same_user_candidate_hit_count` and
`cross_user_candidate_hit_count` to query, per-user and run outputs. Their sum is
`candidate_hit_count`. A cross-user candidate means the resident cache entry's
logical owner **at insertion** differs from the current logical user. Hits do
not transfer ownership; eviction followed by reinsertion establishes a new owner.
`cross_user_candidate_hit_fraction` divides cross-user hits by candidate hits,
with zero for no candidates; aggregates divide summed counters rather than
averaging per-query fractions. Query traces retain owner IDs and cross-user hit
masks for inspection. These are cache-sharing opportunities only, not safe,
physical, or lossless reuse. `safe_reuse_claimed` remains false.

`candidate_cost_effective_if_safe` is null (blank in CSV) with an explicit
unavailability reason in query/per-user/run outputs. The current cost-artifact
interface has no exact workload/candidate comparison mapping. Supplying L32
fixture timings does not populate this metric or workload latency.

## Latency policy

Actual workload lengths drive cache and communication statistics only. Workload
latencies are null even at length 32: equal length alone does not establish a
matching physical fixture. The separate `normalized_fixture_*` columns summarize
strict M9-A fixture repeats, never multiply them by workload query counts, and
never interpolate/extrapolate ES timings. p50/p95 use linear sample quantiles.
These are repeat variation for one fixture, not multi-user service percentiles.

CURRENT_BASELINE preserves original semantic impact. TOKEN_PRECOMPUTE_DIAGNOSTIC
replaces only that component using
`impact_summary.json` → TOKEN_PRECOMPUTE →
`uninstrumented_impact_total_ms.mean`. No timing constant is embedded. Both are
recomposed at each bandwidth from unchanged ES/UD and tensor byte evidence.
The diagnostic is SIMULATED_RESEARCH_EXTENSION. Original M9 artifacts are read
only. Normalized fixture REUSE fractions are distinct from workload execution
fractions (zero). System delta is EdgeLoRA minus SemCache; positive favors reuse.
No cost artifacts means fixture columns are unavailable, not zero latency.

ES evidence is MEASURED strict-base-only; UD is CALIBRATED (the established
CALIBRATED_ON_SERAPH_CPU profile); network and system totals are SIMULATED.
The manifest's provenance map describes component roles, not new measurements.
Source hardware provenance remains in the input artifacts whose SHA256s are
recorded. There is no A100/RTX3090 or paper-UD/CPU equivalence claim.

## Input contract and commands

Local workload JSONL needs one object per query:

```json
{"dataset":"snips","source_id":"q1","token_ids":[2,100,200,300],"cluster_id":4,"model_id":"facebook/opt-2.7b","tokenizer_id":"local-opt-tokenizer-revision","semantic_assignment_source":"existing TinyBERT workload artifact"}
```

Use the [M9-B.1 preparation stage](m9b_semantic_preparation.md) for raw prepared
SNIPS/MultiWOZ workloads. The simulator itself remains an artifact boundary. Export existing
semantic-workload token IDs and assignments into these fields without generating
new tokens/clusters. One model/tokenizer/assignment namespace per input is
required. No such local datasets or M9 cost artifacts were present in this
checkout during implementation, so actual dataset experiments remain pending.

Exact SERAPH smoke command (standard-library Python only):

```bash
python3 scripts/37_run_m9b_multi_user.py --smoke --seed 42 --output-dir results/m9b/smoke
```

Exact SNIPS command once the indicated local artifacts are available:

```bash
python3 scripts/37_run_m9b_multi_user.py --snips results/workloads/m9b_snips_semantic.jsonl --system-summary results/m9a/m9a_summary.json --impact-summary results/m9a2/impact_summary.json --seed 42 --output-dir results/m9b/snips
```

Exact MultiWOZ command:

```bash
python3 scripts/37_run_m9b_multi_user.py --multiwoz results/workloads/m9b_multiwoz_semantic.jsonl --system-summary results/m9a/m9a_summary.json --impact-summary results/m9a2/impact_summary.json --seed 42 --output-dir results/m9b/multiwoz
```

Supply both dataset flags in one invocation for the 36-row actual-data matrix.
Without cost files omit **both** cost flags for cache statistics only. Input
paths above are explicit conventions, not claims these artifacts exist locally.

Outputs: run_raw.jsonl (one aggregate per configuration), summary.csv,
per_user.csv (including inactive users), cache_trace.csv (one query row per cache
pass, hit masks and events), environment.json, manifest.json, and seven plot-ready
CSVs. No matplotlib required. The smoke outputs contain 36 explicitly synthetic
configurations, not SNIPS/MultiWOZ findings.

Model-free test command:

```bash
PYTHONPATH=src python3 tests/test_m9b_multi_user.py
```

Limitations: no validated workload reuse, online cluster updates, measured
attention import, scheduling/queueing, concurrent devices, network contention,
physical QKV execution, adapter training, full autoregressive decode, or full
SemCache reproduction. A future evidence importer and cost support for exact
workload fixtures are required for nonzero executed reuse experiments.


Pre-full-run integrity test commands (no models or downloads):

```bash
PYTHONPATH=src python3 tests/test_m9b_multi_user.py
PYTHONPATH=src python3 tests/test_m9b_semantic_workload.py
```

The audit tests group preservation across 10/25/50 users, repeated deterministic
assignment, unchanged semantic rows/order, original-user independence, owner-hit
partitioning, no-evidence safety, aggregate consistency, and paired traces with
and without normalized fixture cost inputs. Earlier smoke artifacts predate the
assignment/metric audit; future runs record `assignment_policy_version=m9b_integrity_v1`.
