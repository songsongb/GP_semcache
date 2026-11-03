# R5-I5R-0: portable zero-HIT diagnosis

This is an additive CPU-only, read-only diagnostic. It changes no production
acceptance rule, routing, cache policy, numerical tolerance or historical output.
The user's pre-existing modification to scripts/119_run_r5_i5.py is preserved.
No CUDA inference, Slurm submission, deployment, push, model or dataset download
was performed. R5-I3F/I4A ROOT_CAUSE_UNRESOLVED remains a separate boundary.

## Implementation and input contracts

Added files:

* src/semcache/edgelora/r5_i5r0_io.py — streaming ZIP/local input access,
  SHA-256 inventories, strict block selection, source Git blob verification.
* src/semcache/edgelora/r5_i5r0.py — independent candidate predicates, counts,
  token-only opportunity scans, paired timings, coverage and report generation.
* scripts/125_diagnose_r5_i5r0.py — offline CLI.
* tests/test_r5_i5r0.py — CPU-only synthetic schema/adversarial tests.
* docs/R5_I5R0_DIAGNOSIS.md — this implementation and research handoff.

The inspected I5 execution config embeds block.tasks, including prepared token
IDs, actual task identities and generation budgets. Exported request metrics
carry measured input length, immutable matched hashes, physical identity,
candidate_reasons, cache snapshots, timing and wire counters. Exported cache
events include full DISCOVERY candidate metadata and INDEX_INSERT/INVALIDATE
records. These schemas differ from raw requests.jsonl ARRIVAL/COMPLETION and
adapter-tagged events.jsonl; the reader handles them separately.

Raw mode accepts only block_<index>_<numeric_job_id> directories directly under
--runs-root; siblings ending _audit are excluded. It requires COMPLETE block and
condition receipts, verifies sealed inventory, expected task coverage and matched
A/B/C identities, and reads original arrivals/events. FULL_RECOMPUTE may omit events.jsonl
or contain an empty event stream; any nonempty A event stream is rejected.
SEMANTIC_NO_REUSE and SEMCACHE_INTEGRATED must each contain events.jsonl,
including when --allow-partial is selected. SHA-256 inventory verification runs
before these condition-specific checks and remains strict: a file listed in a
sealed manifest cannot be removed or changed, including an empty A event file.
It does not call or weaken the frozen strict I5 GPU independent analyzer.
ZIP mode accepts only exported
block_<index>_<numeric_job_id>_audit roots, verifies reported audit PASS and every
contained manifest/file hash, and clearly retains the exported audit as a
historical claim. Neither mode freshly validates GPU tensor bytes as consumed
by hardware. No extraction occurs: unsafe paths, symlinks, duplicate ZIP members,
encryption, excessive member sizes, duplicate blocks, missing indices, wrong
expected count, corrupt files and mismatched requests are rejected.

--expected-blocks defaults to 12 for this historical study. Override it explicitly
for another complete study; this detects a missing last block as well as holes.
If a raw embedded full plan exists its block IDs must match too. Outputs must be
fresh, cannot overlap inputs, and repository outputs belong under results/r5/i5r/.
Optional evidence gaps require explicit --allow-partial. This never waives corrupt
manifests, ambiguous blocks or contradictory required evidence. Every unknown
predicate is NOT_OBSERVABLE_FROM_AVAILABLE_EVIDENCE, not a PASS or reuse grant.

Each candidate independently evaluates residency, physical identity, completion,
length, full-source token-prefix equality, full sequence equality, LCP, adapter,
positions/mask and registry authorization. The JSONL and CSV keep source and target
identities, rank, recorded window counts, actual rejection and all additional
failures. False predicates can establish denial even when another predicate is
unknown. No partial prefix is substituted for a complete admitted source.

Candidate counts and target-request counts have separate denominators. Candidate
entries are distinct by block/condition/adapter/entry, never globally deduplicated
across reset namespaces. Multiple failures overlap and are labeled accordingly.
The counterfactual scan reports token-prefix pairs within blocks, eviction receipts
when present, and cross-block boundaries only within dataset/track/repetition.
Those pairs do not authorize reuse or establish continuous-session HIT rates.

## Reproduced historical GPU export observations

Primary input: local_evidence/Block_0-11_audit.zip. All twelve audit manifests and
contained inventories verify. All twelve exported independent audits report PASS.
This verifies transfer integrity and recorded claims, not a new GPU audit.

| Integrated metric | Recalculated |
|---|---:|
| Blocks | 12 |
| Scored observations per condition | 60 |
| Scored observations across A/B/C | 180 |
| Distinct source samples per dataset | 10 (two repetitions) |
| Requests with ordinary semantic candidates | 28/60 |
| Ordinary candidate occurrences | 34 |
| Distinct candidate entries | 28 |
| Candidate requests absent | 32/60 |
| Physical HIT requests | 0/60 |
| Reused prefix tokens | 0/30,720 input tokens |
| Physical admissions | 60 |
| Physical evictions | 6 |
| Privileged cross-adapter probe rejections | 82 |
| Truncated integrated scored observations | 60/60 |
| Dispatched input length | exactly 512 for all 60 |

| Dataset | Candidate requests | Candidate occurrences | Full-prefix matches | LCP tokens |
|---|---:|---:|---:|---:|
| Qasper | 8/20 | 8 | 0 | 24 |
| 2WikiMQA | 12/20 | 18 | 0 | 27 |
| MultiNews | 8/20 | 8 | 0 | 16 |

All 34 ordinary integrated candidates have equal 512-token source/target lengths.
None is an identical sequence or full-source prefix. Each independently fails
both TARGET_NOT_LONGER_THAN_SOURCE and EXACT_PREFIX_TOKEN_MISMATCH, even though
the production first-failure reason records only the former. The unchanged
safety policy therefore denies every one. Relaxing the length condition alone
would not make these entries usable. LCP mean is 23.7059 tokens, median/P95 27.

Observed configured cluster count is one. A common prefix of at least 16 tokens
already contains shared w=3 windows; discovering candidates is not proof of a
useful full context prefix or of semantic encoder effectiveness. The common
prefix token span was not decoded into template/context offsets with a tokenizer,
so this report does not claim every shared token belongs to instructions.
Fixed-length truncation structurally eliminates strict shorter-source eligibility
in this run; an intervention changing truncation has not been measured. The
literal preparation input_limit flag was not persisted in the exported config;
512 is user-reported configuration corroborated by actual dispatched lengths.

There are no observed groups split across blocks within a dataset/track/repetition,
and no shorter full-source token-prefix pairs within or across those blocks.
Six evictions are recorded, but there is no eligible token-prefix opportunity
whose loss can be attributed to those evictions. Cache lifetime is five measured
requests per Natural block; there are 36 independent condition resets. The 82
foreign probes are separate privileged aggregates, not additional ordinary
candidate occurrences; they must not be added to the 34 rejection denominator.
Original context length averages 5,052.5667 tokens (median 3,927, P95 16,184).
Retained context averages 472.3667 tokens (median 468, P95 489), with an average
4,580.2 context tokens removed. Context lengths exclude retained prompt instructions.
Recorded current/peak cache occupancy is retained separately per block, condition
and adapter; these snapshots are not fresh GPU memory measurements.

## Latency results and limits

| Condition | Mean application E2E, ms | Mean model execution E2E, ms | Mean admission/eviction, ms |
|---|---:|---:|---:|
| FULL_RECOMPUTE | 36152.8109 | 36113.6499 | 0.0593 |
| SEMANTIC_NO_REUSE | 35955.3570 | 35835.7756 | 17.8289 |
| SEMCACHE_INTEGRATED | 38126.4995 | 35762.7559 | 1904.7799 |

Matched aggregate application E2E differences (candidate minus reference):
A→B -0.5462%, B→C +6.0384%, A→C **+5.4593%**. The reported negative integrated
performance and roughly 1.9-second admission interval are confirmed. Timing
comparisons remain descriptive: repeated documents, conversations and block
state are not independent observations. No significance or speedup is inferred.
Mean, median, nearest-rank P95, per-example paired deltas, dataset and repetition
breakdowns are exported for all available timing/directional wire fields.

Admission spans include checks, hashing, synchronization, bookkeeping/control
and storage work. These records do not isolate GPU-copy cost. Nested timing
fields must not be summed. Model execution E2E is still instrumented distributed
wall time, not pure GPU kernel time. TCP counts are real application framing/
payload over localhost, not WAN measurements; extra control bytes cannot establish
a latency benefit. No actual prefix work was skipped in these observations.

## Provenance and remaining coverage

Blocks 0–10 record 32f5c8083f2443e247850c5bde9cb27876099ca2. All 298 recorded
source-file SHA-256 hashes per block match locally available Git blobs. Block 11
records 04752e9d745588c88a36443275fa26bc2aac2f34, whose Git object is unavailable
locally: verification is UNAVAILABLE, not PASS. Its recorded inventory differs
only at scripts/119_run_r5_i5.py. This does not independently verify the contents
of the missing historical commit. The user's current memory-preflight edit is
not substituted for it. Current diagnostic HEAD and exact source hashes are
recorded separately; the local checkout is not asserted to be either snapshot.

The ZIP has source metadata, prepared tokens, matched request hashes and later
source admissions sufficient to reconstruct the target masks/positions used in
candidate comparisons. It does not include the authorization registry: all 68
ordinary B/C candidate authorization checks are explicitly unknown. They do not
change the definitive length/token denial. --allow-partial is therefore required
for this ZIP; partial observability does not mean a missing block was accepted.
Fresh verification of GPU tensor contents/consumption, wire endpoint payloads,
registry sharing grants and original raw arrivals requires original run artifacts.
A strict fresh GPU evidence audit still requires its recorded source checkout.

The exported numerical/token gates report PASS in all 12 zero-HIT blocks;
physical cache correctness and strict sampled HIT checks are NOT_EXERCISED.
This does not resolve historical HIT-logit failures or native PEFT strict parity.
Reference quality, token equality and numerical fidelity remain separate.

## Commands and output interpretation

No environment variables or external Python packages are required for I5R-0:
Python 3.10+ standard library and local git suffice. Missing historical Git objects
are reported, not fetched. Existing optional OPT/ROUGE regression checks may skip.

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r0 -v
python3 scripts/125_diagnose_r5_i5r0.py \
  --audit-zip local_evidence/Block_0-11_audit.zip --expected-blocks 12 \
  --allow-partial --output-dir results/r5/i5r/diagnosis/ZIP_FRESH
```

On SERAPH (operator-run read-only analysis; no Slurm/GPU job needed):

```bash
cd /data/khuss/repos/GP_semcache
python3 scripts/125_diagnose_r5_i5r0.py \
  --runs-root results/r5/i5/run_longbench3_gpu_policy_attempt_v2 \
  --expected-blocks 12 --output-dir results/r5/i5r/diagnosis/RAW_FRESH
```

Use --allow-partial only if explicitly accepting absent optional artifacts; errors
still reject corrupt/incomplete/duplicate required evidence. Every invocation
requires a new output directory. CLI success with optional gaps says
PARTIAL_DIAGNOSTIC, not independent GPU audit PASS.

Outputs: i5r0_candidate_diagnostics.jsonl/.csv, request_diagnostics.csv,
workload_summary.json, rejection_breakdown.json, latency_diagnostics.json,
source_provenance.json, evidence_coverage.json, recommendations.json, report.md,
opportunity_scan.jsonl and a SHA-256 i5r0_output_manifest.json (all use i5r0_ prefix).
The candidate file evaluates all conditions (68 ordinary B/C candidates here);
filter condition=SEMCACHE_INTEGRATED and scored=true for the historical 34.
The main historical release is results/r5/i5r/diagnosis/longbench3_delivery_20261102/.

## Executed CPU validation

The initial I5R-0 implementation ran 151 applicable tests: **149 PASS, 0 FAIL,
2 SKIP**. This includes 22 I5R-0 diagnostic tests. Existing skips are the pinned
ROUGE dependency and installed OPT contract (torch/transformers/numpy unavailable).
Neither skip is reported as runtime/GPU verification. The exact regression command:

```bash
PYTHONPATH=src:tests python3 -m unittest \
  test_r5_i5r0 test_r5_i5 test_r5_i5_controller test_r5_i5_preflight \
  test_r5_i5_hotfix test_r5_i5_gpu test_r5_contracts test_r5_semantic \
  test_r5_controller test_r5_multi test_r5_fidelity test_r5_perf_evidence \
  test_r4c_opt_contract -v
```

The real ZIP CLI completed successfully with explicit PARTIAL_DIAGNOSTIC status.
Source, manifest, identity and counterfactual checks ran on CPU. No new GPU tensor
audit or process/TCP execution was claimed. The independent audits above are the
twelve verified historical exports, not twelve newly executed GPU audits.

The RAW_RUNS event-loader correction on 2026-11-03 ran **27 PASS, 0 FAIL,
0 SKIP** in test_r5_i5r0, and reran the same combined command above with
**154 PASS, 0 FAIL, 2 SKIP** (156 total). The two existing dependency skips
remain ROUGE and installed OPT. New raw fixtures omit A's events.jsonl and test
its optional empty form, nonempty A rejection, mandatory B/C event files,
malformed B/C JSON rejection and SHA-256 tamper rejection for all conditions.
Only temporary synthetic fixtures are created/resealed by those tests; historical
experiment results and manifests are never modified.

## R5-I5R-1 design handoff

R0 requires **no GPU execution**: the supplied tokens explain zero HITs.
Additional raw artifacts can improve coverage with CPU-only analysis.
R1's measured latency/quality and numerical behavior require a separately reviewed
GPU experiment. Do not infer success from this diagnostic.

1. CPU-enumerate genuine exact-prefix opportunities before scheduling. Retain a
   separate authentic Natural track; report zero natural HITs without injection.
2. Use the existing Controlled real-context design with actual tokenizer ID
   slices for completed standalone sources and a strictly longer target. Do not
   authorize arbitrary slicing of historical full-source cache entries from LCP.
3. Validate 64/128/256/512 prefixes (1024 only within positional, generation,
   cache and evidence budgets). Preserve same-adapter ownership and positions/
   masks, source completion, residency and independent physical consumption audit.
4. Record scored targets, priming costs, measured cache lifetime, model load,
   warmup and admission separately. An independent-block scale is not a continuous
   1000-request cache session. Include amortized costs, not just warm targets.
5. Profile admission/hash/validation/serialization/sync/control/storage components
   before optimizing or attributing the 1.9-second span to copying alone.
6. Keep matched A/B/C inputs/policy/order, native quality and frozen numerical
   gates. Do not relax tolerances, conflate task equality with numerical fidelity,
   or claim correctness-qualified speedup while ROOT_CAUSE_UNRESOLVED persists.
