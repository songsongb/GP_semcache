# R2-D: evidence-gated EdgeLoRA network break-even

R2-D is a CPU-only **ANALYTICAL_DIAGNOSTIC** after R2-C4. It reconciles original
M9-A accounting with separate R2-C4 local diagnostic evidence. It does not run
OPT, load Torch/Transformers, calibrate a device, execute network traffic, or
establish distributed E2E performance. No CacheGen, codec, compression or
quantization is used. Existing experiments, results and launchers are unchanged.

## Invocation and artifacts

The separately supplied launcher can invoke this exact contract:

```bash
python -u scripts/94_run_r2d_edgelora_break_even.py \
  --phase analyze \
  --m9a-summary <PATH_TO_m9a_summary.json> \
  --c4-aggregate-dir <PATH_TO_R2C4_AGGREGATE_DIRECTORY> \
  --bandwidth-mbps '10,50,100,200,500,1000' \
  --rtt-ms '0,1,10,30,50' \
  --output-dir <NEW_OR_EMPTY_OUTPUT_DIR>
```

An environment with only `python3` can use that executable with identical
arguments. The runner refuses a nonempty output directory and never overwrites
inputs. Relative **recorded evidence paths resolve against the repository
root**, including `manifest.breakdown_evidence.directory`; they do not resolve
against the caller's working directory or the aggregate directory. Historical
absolute paths must remain accessible. Restore the referenced evidence tree at
its recorded paths rather than silently replacing or relabeling evidence.

| Output | Contents |
|---|---|
| `r2d_manifest.json` | Version, exact argv, Python version, source HEAD/branch/dirty state, source hashes, input/output hashes, timestamps and execution flags |
| `r2d_evidence_audit.json` | Per-file validity and SHA256, recorded-hash versus snapshot-only integrity, reconstruction gates, compatibility matrix and missing E2E evidence |
| `r2d_bandwidth_rtt_sweep.csv` | Historical per-repetition payload/compute results and separate illustrative C4 threshold rows |
| `r2d_break_even_summary.json` | Directional byte ledgers, roots, monotonic regimes, winners/losers, assumptions, C4 contrasts and separate diagnostic clocks |
| `r2d_decision.json` | Supported versus counterfactual scope, eligibility flags, conclusion and concrete R3 work |

Success sets `execution_status=COMPLETE`, `gpu_inference_performed=false`,
`network_executed=false` and `classification=ANALYTICAL_DIAGNOSTIC`.
COMPLETE describes completion of artifact validation and analysis. It does not
qualify a pooled comparison or assert that reuse wins. Input failures produce a
nonzero exit, FAILED manifest/audit, NOT_ESTIMABLE summary/decision and a visible
failure row in the CSV; invalid records are never silently dropped. Existing
nonempty outputs are refused before any write. The manifest hashes the other
four exports and explicitly excludes its own circular self-hash.

## Required evidence tree and validation

Supply original artifacts from script 34 and the completed script 93 aggregate.
No original M9-A/C4 historical measurement tree is present in this checkout;
the development smoke uses explicitly marked TEST_FIXTURE_ONLY scalar files.

M9-A requires:

- `m9a_summary.json` in `m9a_single_user_v1` format, with `comparisons[].row`,
  component provenance, network ledgers and base profiles.
- Sibling `m9a_environment.json`, including configuration, dimension source and
  input SHA256 values. Its recorded ES input, exact-length CPU calibration and
  local `config.json` must be accessible.
- For strict inputs from script 35, the ES input directory's
  `fresh_m85/inference_raw.jsonl` and `fresh_m85/inference_environment.json`,
  checked against the base profiles' original source hashes. A sibling
  `profile_manifest.json`, when available, must be COMPLETE with the matching
  strict-input SHA256.

The loader reconstructs every historical comparison using the frozen M9-A
`compare_request()` and `communication()` equations. It validates paired
request/model/user/adapter identity, FP16 dtype, strict length-32 base-only
profiles, source field preservation, physical hit spans, endpoint/compute scope,
exact-length four-thread FP32 UD QKV calibration, explicit uncompressed wire
widths, correctness, signs, units, control residual and historical bandwidth
grid. Repeated accounting at different bandwidths is collapsed to one source
measurement per repetition before generating the new sweep.

Missing correctness or an invalid-reuse override is rejected. A valid original
`peft-prefill-proxy` input remains exploratory with
`SIMULATED_PROXY_DOUBLE_COUNTS_LORA` and `primary_comparison_eligible=false`;
it is never promoted to a strict base-only result. No corrected proxy speedup
is invented. M9-A's controlled fixture correctness is not proof that a real
partitioned adapter system produces correct outputs.

C4 requires:

- `manifest.json`, `cache_path_aggregate_summary.json`,
  `cache_path_decision.json` in the supplied aggregate directory.
- All three `input_evidence[].directory` trees, with the recorded manifests,
  `cache_path_raw_pairs.json` and hashed `impact_references.json`.
- The separate `breakdown_evidence.directory`, including its recorded manifest,
  `cache_path_breakdown.json` and hashed `impact_references.json`.

Checks include COMPLETE/AGGREGATED_GPU_RUNS, the recorded
`exp/raw-semcache-r2` branch, OPT-2.7B/FP16, CUDA residency, the six exact arms,
`user_a`, full all-token head, eager attention, use_cache=false,
output_attentions=true, Impact Replay and reference binding/value digests.
All three independent process/run/order-seed identities, warmup/measured counts,
actual physical projection rows, oracle equality, condition/reference identities,
coverage, negative controls and per-arm strict correctness are revalidated by
reconstructing the C4 aggregate with its recorded bootstrap seed/draws. The
decision and aggregate condition copies must agree. Breakdown hashes and timing
boundaries must verify; breakdown request walls are never pooled with minimal
latency.

Recorded source-file maps and implementation hashes are checked against local
source content where recorded. New files and a different git HEAD alone do not
invalidate unchanged historical code. If a full code-file map was not recorded,
the audit explicitly reports the limitation. An unanchored summary's newly
computed SHA256 is a snapshot of the supplied file, not an independently trusted
measurement certificate. Internal hash consistency cannot authenticate the
original experiment operator or hardware by itself.

## Three evidence classes and compatibility

1. **Historical M9-A:** MEASURED base-only ES and exclusive M8.5 control,
   CALIBRATED CPU UD LoRA, SIMULATED tensor payload/network and additive totals.
2. **R2-C4:** MEASURED local conditional six-arm contrasts with exact Impact
   Replay; separate nonminimal CPU/CUDA diagnostic intervals. Oracles only
   exclude controlled request work and require external validation.
3. **Hypothetical sensitivity:** bandwidth assumptions, HYPOTHETICAL_RTT
   schedules and illustrative C4 required-traffic thresholds.

The audit's machine-readable matrix covers model, length, dtype, accelerator,
adapter, LoRA/PEFT placement, reference endpoint, output head, attention/impact
flags, cache residency, timing boundary, source-build exclusion, admission,
correctness and measurement provenance. Matching model/dtype does not remove
the endpoint, length, control-path and correctness differences. These cohorts
remain **NOT_COMPARABLE** for pooled latency. C4 request latency, PEFT prefill,
oracle benefits and diagnostic CPU/GPU spans are never inserted into M9-A
base-only compute totals. There is no length-32 to length-512/1024 compute
extrapolation.

User-supplied historical context at L1024, to be checked against actual files:

| Coverage | RAW−Full ms | RAW−NoReuse ms | Snapshot benefit ms | Mapping benefit ms | Both benefit ms |
|---|---:|---:|---:|---:|---:|
| 510/1024 | +461.00 | +323.60 | +121.30 | +2.34 | Not specified here |
| 1023/1024 | +684.96 | +536.28 | +245.62 | +8.58 | +258.52 |

Nonzero RAW reuse failed strict Full-relative FP16 correctness, while
oracle-versus-RAW equality passed. At zero reuse, mapping/snapshot interventions
are NOT_APPLICABLE; signed contrasts remain negative-control noise. No safe
natural semantic hit rate or deployable oracle optimization follows.

Separate baseline diagnostic means at approximately half/MAX coverage include
snapshot validation 118.19/226.79 ms, Fast Index Guard 130.03/127.89 ms,
mapping retrieval 0.62/0.92 ms and inclusive mixed-projection host time
212.22/313.25 ms. These are nested intervals, not disjoint overheads.
The MAX contiguous-copy counter 502,824,960 bytes is GPU-side logical memory
traffic, not network traffic. The export preserves parent/inclusive/location
labels and summarizes host and CUDA clocks individually. CUDA event intervals
are asynchronous inclusive intervals, not GPU kernel self-times. Their sums
cannot explain an E2E delta.

## Payload and root equations

For N prompt tokens, R actually reused tokens, hidden dimension H, L layers,
hidden width wh and delta width wd, the inherited per-layer traffic is:

```text
ES → UD hidden:          (N − R) × H × wh
UD → ES three deltas: 3 × (N − R) × H × wd
```

With retained boundaries, add **one** full h0 of `N×H×wh` in UD→ES and
**one** full hL of `N×H×wh` in ES→UD. Those terms are unchanged by reuse.
Projection-exchange-only histories retain that explicit omission in the ledger.
Saved payload is exactly `L×R×H×(wh+3×wd)`. Cached QKV size, KV transport,
local GPU copies and protocol headers are never substituted for that savings.
Tensor occurrence counts are not application message counts.

```text
payload_ms(bytes, B_Mbps) = .008 × bytes / B_Mbps
T_no_reuse = ES_base + UD_calibrated + payload_no_reuse
T_reuse = ES_reuse + UD_fresh_calibrated + exclusive_M8_control + payload_reuse
delta(B, RTT) = T_reuse − T_no_reuse
             = delta_local_ms + delta_RTT_ms + .008 × delta_bytes / B
root B* = −.008 × delta_bytes / (delta_local_ms + delta_RTT_ms), when positive/finite
```

Every component appears once. Existing cache transfer/merge is already in ES
prefill. Nested QKV timers and lookup-baseline diagnostics are not added again.
Negative delta favors reuse; positive delta favors no-reuse. Roots classify
finite crossover, always-winning, always-losing or tied for all finite positive
bandwidths. Both compute/traffic tradeoff signs are handled. Zero-byte and
zero-offset cases retain their limiting behavior. Root/domain warnings distinguish
an exact algebraic root from a grid observation. Speedup ratio is
`T_no_reuse/T_reuse`; percent reports `100×(T_no_reuse−T_reuse)/T_no_reuse`.
Negative values are preserved.

All compute and transfer accounting is serialized with **no overlap**. A
compute/transfer overlap model needs the actual dependency schedule; R2-D does
not replace sums with max(). Reuse assumes required projections are resident.
Cold build/fill, refresh, eviction, request frequency and natural admission rates
are unknown, so cold/warm amortized break-even is NOT_ESTIMABLE.

## RTT scenarios and illustrative C4 thresholds

At RTT=0 the payload-only model applies. At nonzero RTT, actual message counts,
RTT charges and actual RTT-adjusted delta/root remain **NOT_ESTIMABLE**.
A separate **HYPOTHETICAL_RTT** sensitivity assumes one serial exchange per
active layer, all three deltas bundled into its reply, plus one exchange for
the retained boundary pair. For any nonzero fresh-token batch, both paths have
`L+1` assumed exchanges (or L without boundaries); partial reuse saves bytes but
does **not** save RTT in this scenario. Full reuse omits the assumed layer
exchanges while retaining the boundary exchange. This is not an observed or
established M9-A message schedule.

The pure `rtt_sensitivity()` function exposes `exchanges_per_layer` and
`boundary_exchanges` for alternative analytical scenarios without changing the
frozen launcher CLI. Separate sends for the three deltas, additional metadata,
different boundary scheduling, partial overlap or cache-fill communication
require R3 evidence. The default's bundling choice does not settle those issues.

C4 reports separate unqualified thresholds. With observed local RAW−NoReuse
penalty O and assumed equal RTT cost, required avoided payload is
`O×B/.008` bytes. With assumed zero payload difference, the separate required
saved-round-trip threshold is `O/RTT`. Fractional thresholds are algebraic
requirements, not invented application message counts. The model supplies no
C4 avoided-byte estimate or 512/1024 distributed compute total; its bandwidth
root remains NOT_ESTIMABLE. At zero reuse these thresholds are NOT_APPLICABLE.
Oracle benefits are preserved as diagnostic contrasts and never subtracted to
construct an optimized implementation.

## CPU verification and R3

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_r2d_*.py' -v
PYTHONPATH=src:tests python3 -m unittest -v test_m9a_cost_model test_m9a1_base_profile test_r2c3_analysis test_r2c4_analysis test_r2c4_cache_path
python3 -m compileall -q scripts/94_run_r2d_edgelora_break_even.py src/semcache/diagnostics/r2d_evidence.py src/semcache/diagnostics/r2d_network.py src/semcache/diagnostics/r2d_analysis.py tests/r2d_fixtures.py tests/test_r2d_evidence.py tests/test_r2d_network.py tests/test_r2d_analysis.py
python3 scripts/94_run_r2d_edgelora_break_even.py --help
```

To create a new clearly labeled scalar smoke tree (choose a new directory):

```bash
PYTHONPATH=src:tests python3 tests/r2d_fixtures.py /tmp/r2d_TEST_FIXTURE_ONLY
python3 -u scripts/94_run_r2d_edgelora_break_even.py --phase analyze --m9a-summary /tmp/r2d_TEST_FIXTURE_ONLY/m9a_TEST_FIXTURE_ONLY/m9a_summary.json --c4-aggregate-dir /tmp/r2d_TEST_FIXTURE_ONLY/c4_TEST_FIXTURE_ONLY/aggregate --bandwidth-mbps '10,50,100,200,500,1000' --rtt-ms '0,1,10,30,50' --output-dir /tmp/r2d_TEST_FIXTURE_ONLY/output
```

TEST_FIXTURE_ONLY propagates into exports and cannot support a measured claim.
The subprocess contract test rejects heavy model/codec imports and socket
construction. Optional older Torch tensor tests can be skipped when Torch is
absent. These checks establish implementation behavior, not GPU/network E2E.

R3 must identify actual adapter partition/weights; trace each message's layer,
direction, type, payload/framed bytes and dependency; measure achieved bandwidth,
RTT and exchange counts; record CPU/GPU handoffs, wire conversion and overlap;
measure cold/warm build/fill/refresh and normal Admission reuse; and compare
matched distributed request walls, TTFT, generation/decode and token quality.
R2-D leaves `primary_claim_eligible=false`, `real_network_measured=false`,
`full_generation_e2e_measured=false` and `correctness_qualified_speedup=false`.

Development validation on 2026-10-29: 24 new tests passed; the combined new and
system-cost/C3/C4 regression suite ran 76 tests with **75 passed, 0 failed,
1 skipped** (optional actual CPU QKV tensor test; Torch absent). Compilation and
CLI help passed. A standalone TEST_FIXTURE_ONLY smoke produced all five
nonempty exports and 120 sweep rows, with verified output hashes. `python` was
unavailable on this host; verification used `python3` 3.10.12. Actual historical
analysis still requires the original M9-A and C4 evidence trees above. No GPU
inference, network execution, Slurm submission or push was performed.
