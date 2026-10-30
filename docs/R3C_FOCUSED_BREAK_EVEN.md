# R3-C-F: fixed-count focused full-request HIT fractions

This add-only diagnostic asks where GPU-resident exact-full-prompt total-QKV
reuse reduces synchronized all-token prefill request wall relative to the same
physical CPU-UD NO_REUSE endpoint. Classification is
`R3C_FOCUSED_APPLICATION_LAYER_DIAGNOSTIC`. Primary performance eligibility,
correctness-qualified reuse speedup, remote network measurement, semantic
generalization, CacheGen, compression, trained-adapter and R2-D root-validation
flags remain false. Prefill is not TTFT, generation or task-quality evidence.

The user reported a completed 151893 R3-C run and exploratory candidate
intervals. Its SERAPH filesystem is unavailable on personal Ubuntu. No quoted
summary numbers are used as rows, controls or runtime expectations. Missing
historical input produces `NOT_RUN_MISSING_INPUT`, never invented evidence.

## Frozen path and isolated extension

All original 95–100/R3-A/B/C sources, PEFT thresholds, historical results and
policies remain untouched. New `r3cf_*` names do not match either the original
`r3a_*.py` source-identity glob or R3-C's `r3c_*.py` glob.

The new 101 ES worker calls the original `r3c_runtime.execute()`. That function
has no experiment-class parameter, so the new worker temporarily binds its
`NetworkExperiment` symbol to an extending class and restores it in `finally`.
This binding exists only in the isolated new worker. It changes schedule
dispatch, role labels and additional audit output; base/LoRA arithmetic,
request-wall implementation, Channel, proxy pacing and UD math are unchanged.
The independent auditor compares every focused row with its original R3-C
component row and rejects any changed numeric/timing/counter evidence.

The actual original bare GPU OPT, `ProjectionSession`, framed TCP Channel,
CPU-only `user_device.serve/cpu_deltas`, transparent loopback proxy, calibration,
R3-B `Experiment.request/physical_path`, and `ExactRawCache` are reused.
SemCache integration is the existing CacheEntry/GlobalCache/ExactTokenMatcher,
Subsequence, admission policy and owned GPU total-QKV storage. There is no
SemCacheEngine.query(), semantic-window authorization or cross-context HIT.

New calibration uses the same proxy PID retained for model requests. Original
zero-credit/per-frame-start, bounded-one-chunk-credit pacing and RTT/2 one-way
frame delay remain as documented in [R3-C](R3C_NETWORK_ONE_SHOT.md). Decimal
Mbps, milliseconds, actual framed application bytes and payload bytes remain
distinct; TCP/IP overhead and remote physical-link RTT are not measured.
Calibration retains the original max(5 ms,35% RTT) residual tolerance and 25%
directional large-frame rate tolerance, plus original per-request checks.
It never adds delay to saved latencies. Invalid realization stops the run.

Prepare runs the unchanged 95 command for OPT-2.7B FP16, batch1/L32, eager,
use_cache=false, output_attentions=true, user_a rank8, seed42. Prepare exits
before one persistent bare ES model starts. One original CPU UD and proxy
remain separate processes; no GPU A/B execution occurs in the physical path.
Within-cohort GPU UUID, snapshots, versions, actual adapter matrix and base
fingerprints, tokens and code hashes must match. Any allocated RTX3090 is
allowed; an earlier run's GPU/node UUID is not required.

## Immutable exact schedule

| Profile | Primary arm | Measured requests | Actual HITs |
|---|---|---:|---:|
| C00: 0 ms/unlimited both ways | NO_REUSE | 16 | 0 |
| C00 | CACHE_02_OF_16 | 16 | 2 |
| C00 | CACHE_04_OF_16 | 16 | 4 |
| C03: 10 ms/200 Mbps both ways | NO_REUSE | 16 | 0 |
| C03 | CACHE_00_OF_16 | 16 | 0 |
| C03 | CACHE_01_OF_16 | 16 | 1 |

Each primary arm has exactly one untimed-as-measurement warmup. Cache warmups
are actual Cold MISS fills; NO_REUSE warms up without caching. C00 is first;
seed42 randomizes primary arm and within-arm exact HIT/MISS order. No grid or
fraction overrides are offered. The manifest, SHA and exact realized order are
audited. Each condition begins with three real NO_REUSE/COLD_MISS/WARM_HIT gates.
Each condition ends with one genuine Cold MISS `aux_fill` followed by eight
`aux_measured` Warm HITs, all in arm `AUX_RESIDENT_HIT`, role `AUXILIARY`.

The exact model total is **126**: 96 primary measurements + 6 primary warmups +
6 gates + 2 auxiliary fills + 16 auxiliary measurements. There are 112 measured
requests across both roles, but only 96 enter primary arm comparisons. The
planned total contains 101 real full-exchange requests and 25 full HITs:
3,232 HIDDEN plus 3,232 DELTAS data frames, conditional on observed gates.
Four session handshake/shutdown frames and calibration probes are separate.
There are no hidden model preconditioning forwards. Resource admission reuses
the original bounded traffic/time estimator; those allowances are assumptions,
never measured latency. Model deadline is 7,200 s, worker deadline 7,500 s.

Every COLD_MISS resets outside request wall, performs actual lookup, transfers
and CPU/GPU math and inserts all 96 total-QKV outputs inside request wall.
Every WARM_HIT validates/clones resident QKV and must have zero HIDDEN/DELTAS,
zero UD LoRA work and 96 physical outputs reused. The rest of OPT still runs.
Full logits and all QKV SHA must equal the physical same-cohort reference on
every gate, warmup, primary and auxiliary request. The unchanged Native PEFT
full-logit strict comparison is separate; its failure is preserved, and does
not falsely become cache corruption or primary performance eligibility.

Prepare, model/adapter loading, handshake and resets are outside request wall.
Lookup/integrity hashing, tensor clones, insertion, forward, socket blocking
and original CPU UD processing stay inside. Full-logit/QKV quality export
follows wall timing. Reset/fill times, admission/occupancy and preconditions
are retained separately. Nested host/CUDA/UD/proxy clocks are not summed.

## Retrospective analysis

`--phase reanalyze` calls the original read-only 100 loader/auditor, validates
COMPLETE/audit PASS, all recorded output/source hashes, exact four calibrated
profiles/default schedule, 228 unique request IDs, bitwise/Native provenance,
and unchanged original unconditional bootstrap values. Git HEAD differences
alone are not tampering; recorded file SHA identities govern the gates.
Original files, classification and decision are retained verbatim alongside
their SHA and the new statistics. Analysis inputs are rehashed at completion.

For each cache arm, bootstrap HIT walls n_hit times and MISS walls n_miss times
separately; independently resample only its matched same-condition NO_REUSE
arm. All 10,000 seeded replicates preserve the realized HIT count. Percentile
95% intervals use linear interpolation at `(B-1)*q`. Sign is cache minus
NO_REUSE; negative favors caching. Original unconditional results remain beside
the fixed-count contrast. Raw walls/IDs/counts and every caveat are exported.

The old 12.5%/8-request arms contain one HIT: its bootstrap HIT variance is
degenerate. Every retrospective candidate is
`RETROSPECTIVE_EXPLORATORY_ONLY`, including apparently opposite CI signs.
One cohort is not eight independent deployments. Different-condition/cohort
controls are never pooled, and Native accuracy is not inferred.

The optional mean-mixture sensitivity uses observed pure-MISS and pure-HIT
arms and matched controls, with `h*=(MISS-NO_REUSE)/(MISS-HIT)`. It explicitly
mixes different arms and reuses controls. Finite bootstrap root intervals may
extend outside [0,1]; nothing is clipped. This linear model is an exploratory
controlled mean mixture, not a natural hit probability or measured exact root.

## Focused inference and conservative one-HIT policy

Primary effects use exactly 16 primary rows per arm and 16 same-condition
NO_REUSE rows. Describe raw means/medians/p90, HIT/MISS counts, bytes/frames,
calibration and bitwise gates. Apply the same fixed-count bootstrap, 10,000
replicates by default, seed42. Bootstrap limits may be explicitly set to
10,000..50,000; no adaptive searches, outlier dropping or retry loop exist.

Adjacent tested fractions need opposite CI-qualified signs **and** identifiable
within-arm uncertainty before `CONFIRMED_DIAGNOSTIC_BRACKET` is allowed.
Otherwise report UNRESOLVED. The C00 two-HIT arm has only two HIT observations:
its interval is still a small-cohort conditional diagnostic, not deployment
proof. Linear interpolation within a qualifying bracket is exploratory. The
95% intervals are pointwise, not simultaneous intervals or guaranteed 95%
coverage for the bracket or interpolated root.

C03's one-HIT arm cannot establish its HIT distribution. This implementation
conservatively never upgrades that primary bracket using auxiliary data.
Secondary `SECONDARY_AUXILIARY_REPLACEMENT_SENSITIVITY` resamples the eight
same-condition auxiliary HITs for the single HIT slot and reports a separately
labeled model-assisted mean/CI and extrema sensitivity. It does not change
the primary point estimate, enlarge its count or claim target-arm replications.
Auxiliary measurements occur after primary arms and use their own fill;
exchangeability across arm/order/fill is an assumption. A favorable sensitivity
alone cannot confirm the C03 bracket.

Resident eight-HIT effects are separate from primary mixtures. Primary
reset-inclusive and warmup/fill/reset-amortized costs are retained separately.
Auxiliary fill cost is additionally amortized across eight target resident
requests with the matched NO_REUSE mean as comparator. Full deployment
lifecycle remains NOT_ESTIMABLE without arrival/refresh/residency evidence.
Focused C00 has no pure-MISS arm: its optional mean-mixture sensitivity pools
MISS observations from its two cache arms and explicitly discloses this.
No R2-D compute/RTT/byte cohort is pooled or numerically validated.

## CLI, Slurm and outputs

Create logs **before** submission; Slurm opens log files before Bash:

```bash
cd /data/khuss/repos/GP_semcache
mkdir -p logs
bash -n jobs/r3c_focused_break_even.sh
sbatch jobs/r3c_focused_break_even.sh
```

Launcher: one unpinned GPU, four CPUs, 64 GiB, three hours on
batch_eebme_ugrad; explicit conda semcache with optional `R3CF_CONDA_ROOT`,
offline HF cache `/data/khuss/huggingface`, bounded temporary paths,
branch/HEAD/source/dependency/cache preflights, immutable unique directories
and failure trap. It runs legacy reanalysis, fresh GPU preparation/126-request
experiment, and independent read-only physical analysis. Set optional
`R3CF_LEGACY_INPUT` to the original directory. Missing legacy input is reported
and does not require fabricated evidence or prevent the fresh independent run.
Corrupt legacy evidence fails; Native FAIL and unresolved roots do not.

```bash
python scripts/101_run_r3c_break_even_focused.py --phase reanalyze \
  --legacy-input results/r3/c/network_one_shot_151893_20261030T065147Z \
  --output-dir results/r3/c/focused_break_even_<unique>_legacy_analysis

python -u scripts/101_run_r3c_break_even_focused.py --phase run \
  --enable-gpu-experiment --device cuda:0 --seed 42 --resamples 10000 \
  --output-dir results/r3/c/focused_break_even_<jobid>_<UTC_timestamp>

python scripts/101_run_r3c_break_even_focused.py --phase analyze \
  --inputs results/r3/c/focused_break_even_<jobid>_<UTC_timestamp> \
  --output-dir results/r3/c/focused_break_even_<jobid>_<UTC_timestamp>_analysis
```

All outputs must be new/empty, under `results/r3/c/focused_break_even_*`, and
outside input directories. CPU `--phase dry-run` executes no model/sockets.
`--phase fixture --test-fixture-only` uses the complete 126-request schedule,
32-layer tiny CPU fixture and real loopback frames; never GPU evidence. Legacy
TEST_FIXTURE_ONLY input also needs explicit `--test-fixture-only` with reanalyze.

Main files: `r3cf_results.json`, `r3cf_summary.txt`, `r3cf_requests.jsonl`,
`r3cf_schedule_manifest.json`, `r3cf_network_calibration.json`,
`r3cf_break_even.csv`, `r3cf_break_even_decision.json`,
`r3cf_arm_statistics.json`. Reanalysis additionally writes
`r3cf_legacy_reanalysis.json` in the separate `<root>_legacy_analysis` sibling.
`prepare/prepare_process/`, `calibration/`, `proxy/` and `es/` retain original
component rows, per-layer framing/UD work, SHA maps, stdout/stderr, gate/reset
events, lifecycle and partial failures. A failed gate stops with FAILED and
NOT_ESTIMABLE conclusions, keeping all available evidence. Analyze rechecks
the 126 requests, original component rows, three independent socket ledgers,
calibration, Native metrics, source/identity and clean process lifecycles.

CPU verification commands (no downloads/models):

```bash
PYTHONPATH=src:tests python3 -m unittest tests.test_r3cf_schedule \
  tests.test_r3cf_analysis tests.test_r3cf_processes -v
python3 -m compileall -q scripts/101_run_r3c_break_even_focused.py \
  src/semcache/edgelora/r3cf_*.py tests/test_r3cf_*.py
python3 scripts/101_run_r3c_break_even_focused.py --help
bash -n jobs/r3c_focused_break_even.sh
```

Loopback socket creation may need sandbox permission. Torch/GPU tests are
explicit opt-in via `SEMCACHE_R3CF_GPU=1` and unique `SEMCACHE_R3CF_OUTPUT` inside
a real Slurm allocation; unavailable dependencies are not synthetic passes.
No SERAPH operation or Slurm submission is performed on personal Ubuntu.

## Implementation-machine verification (2026-10-30)

The CPU integration tests completed a full 126-request focused fixture,
independent read-only analysis, and a separate original-100 228-request tiny
fixture with read-only retrospective analysis. These are TEST_FIXTURE_ONLY,
not 151893 evidence or OPT/GPU findings. Frozen R3-C/B and original TCP tests
also passed, with unavailable tensor/GPU tests explicitly skipped.
New tests: 21 PASS, 0 FAIL, 1 GPU SKIP (22 total). Selected original R3-C/B
and TCP regressions: 59 PASS, 0 FAIL, 3 dependency/GPU SKIP (62 total).
Compilation, CLI help and Bash syntax checks passed.

A separately retained CPU smoke stopped at request 110: C03 UD→ES layer12
recorded a 21.320244 ms one-way wait, outside the unchanged original 5..10 ms
allowed range for configured RTT10. Cache bitwise equality remained PASS; the
run is FAILED/INVALID_IMPAIRMENT, its partial evidence and cleanup are retained,
and no crossover conclusion is allowed. The tool was not rerun to obtain a
favorable result and tolerances were not relaxed. Host scheduling jitter can
therefore stop the SERAPH run too; do not override its gate. The final source
also has a regression ensuring a boundary failure cannot copy a prior success
into a failed request row.

Local evidence/logs:
`results/r3/c/focused_break_even_local_verification_20261030T074143Z/`.
The genuine 151893 reanalysis is NOT_RUN_MISSING_INPUT and GPU findings are
NOT_RUN. The required next action is manual SERAPH execution, not a claim of
an experimentally established focused crossover.
