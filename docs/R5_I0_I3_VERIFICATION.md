# R5-I0–I3 verification

Scope: implementation and stdlib CPU causal fixture; **GPU_NOT_RUN**.
Development branch: `exp/raw-semcache-r5-integration` from baseline
`076b354373adf9c2528d441fc83d0d95369df80e`.

The architecture/interface map was written before code in
`R5_SEMCACHE_E2E_INTEGRATION.md`. No existing tracked implementation or test is
modified. Existing untracked logs/results were included in a 35,061-file
pre-change SHA-256 snapshot at
`results/r5/local_verification_20261031/baseline_files.json`.

## Executable gates

| Gate | Actual path and tests | Scope |
|---|---|---|
| I0 | r5_contracts.request/validate_request/namespace; test_r5_contracts | version, seals, mask/positions, independent user/adapter |
| I1 | SemanticIndex.assign/discover → Controller.decide; test_r5_semantic | ControlledEncoder in CPU fixture; actual TinyBERT path requires offline revision |
| I2 | Controller.admit → RawOwner → unchanged C1/B1/R3-B owners; test_r5_controller | exact-prefix/corruption/byte pressure/eviction/index lifecycle |
| I3 | r5_runtime.execute → r5_generation.generate → QualityDriver.c_forward → independent C1 CPU UD | real TCP, causal prefill/Decode/greedy generation; fixture is not OPT/GPU |
| Independent audit | r5_analysis.audit/analyze; test_r5_processes | saved requests, index/cluster/LRU replay, tensor hashes/KV reconstruction, endpoint receipts, tamper controls |

## Mandatory ordinary-request scenario

Each integrated request is tokenized from its input text; runtime receives no
expected HIT/MISS or source label. For the CPU fixture P=`abcdeabcde` (10 tokens),
Q=`qwertqwert` (10), R=`lmnop lmn` (9). Bytes per source token are
3 roles ×32 layers ×4 hidden ×4 bytes =1,536. The capacity is the largest
two-source sum, **30,720 bytes**, holding any two of P/Q/R but not all three.
Real OPT uses actual BPE lengths and 3×32×2560×2 bytes/token instead. Both
tokenizer prefix relationships and oversized negative-admission requests are
validated before execution. These planned labels are documentation/assertions,
never runtime dispatch inputs.

| Request | Input/condition | Required observed behavior |
|---|---|---|
| R5_00 | User_A, full P | MISS; prefill +3 generated tokens; admit P |
| R5_01 | User_B, P + ` Please explain.` | same-adapter cross-user HIT from R5_00; suffix + Decode |
| R5_02 | User_C, full Q | MISS; generate; admit Q |
| R5_03 | User_A, P + ` Give details.` | HIT P; P becomes MRU |
| R5_04 | User_B, full R | MISS; generate; evict Q (LRU), admit R |
| R5_05 | User_C, Q + repeated explanation | Q is evicted: full prefill/MISS; output succeeds; oversize admission denied |
| R5_06 | User_A, `Yesterday ` + R + repeated changed ending | semantic windows find R; exact prefix rejects; full generation; oversize denial |
| R5_07 | User_B, full P | equal-length source cannot qualify for strict-prefix HIT; full generation; replace completed P source |
| R5_08 | User_C, P + ` Continue.` | HIT new completed R5_07 source |
| R5_09 | User_A, R + ` Continue.` | HIT R5_04 source |
| separate negative probe | other_adapter | rejected before semantic/model/TCP execution; not counted as generated |

Every successful request has a separately generated N_R5_xx no-reuse baseline
with the same text/adapter/policy. All four HITs remove the prefix's projection
rows and HIDDEN/DELTAS bytes. For R5_01, exactly 960 projection-token rows
(10×3×32) are skipped; cached Q is stored, not consumed. Fixed G3 gives 20
completed requests/60 actual forwards. Quality mode can stop at EOS or the cap;
one must not assert every prompt reaches EOS.

## Reproducible commands

```bash
PYTHONPATH=src:tests python3 -m unittest tests.test_r5_contracts tests.test_r5_semantic \
  tests.test_r5_controller tests.test_r5_processes -v
python3 scripts/110_run_r5_semcache_e2e.py --phase fixture --test-fixture-only \
  --lifecycle-capacity --output-dir results/r5/unique_fixture
python3 scripts/111_analyze_r5_semcache_e2e.py --inputs results/r5/unique_fixture \
  --output-dir results/r5/unique_analysis
PYTHONPATH=src:tests python3 -m unittest discover -s tests -v
bash -n jobs/r5_opt_semcache_e2e.sh
```

The repository has a namespace tests directory, so unittest discovery with
`-t .` is invalid. CPU process fixtures require localhost socket permissions;
the initial sandbox attempt records PermissionError and is not counted as a
successful fixture. Fresh output directories are mandatory. Both failed fixture
attempts and successful evidence remain under results/r5.

## Qualification and limitations

The final machine-readable verification record and exact test logs are saved
under `results/r5/local_verification_20261031/`. Fixture stage flags mean the CPU
integration path passed; R5_GPU_PHYSICAL_EXECUTION_PASS stays false. Native PEFT
strict parity and correctness_qualified_speedup remain false regardless of
fixture token/bitwise equality. Compact evidence does not set strict tensor
fidelity. No genuine task references exist for these integration fixtures, so
F1/ROUGE-L are unavailable, not invented.

This host lacks pytest, Torch, NumPy, Transformers, PEFT, datasets, rouge and SacreBLEU.
Actual OPT/eager cache compatibility, genuine TinyBERT encoding, FP16 fidelity,
allocated RTX3090 memory and all GPU timings remain unverified. The real pilot
is implemented, not GPU-verified. See the architecture document for the preflight
checklist and exact operator-only sbatch command. No remote repository operation,
Slurm submission or GPU execution is performed by this task.

Historical R4-A's missing-dependency test expects dependency checking after a
branch gate that intentionally rejects non-R4-A branches. Historical R3-C-F
calibration can fail in combined runs; both failures and any isolated retries
must remain visible. A broad all-module discovery also imports CacheGen modules
that intentionally conflict with existing Raw-only import guards; those failures
must not be attributed to new R5 code without isolated verification.

## Added files (no existing R4 edits)

| Files | Purpose |
|---|---|
| src/semcache/edgelora/r5_contracts.py | request, namespace and distinct stage gates |
| src/semcache/edgelora/r5_semantic_index.py | encoder/cluster/window metadata index, invalidation and eligibility |
| src/semcache/edgelora/r5_controller.py | selected sealed owner, automatic routing, optional admission and LRU |
| src/semcache/edgelora/r5_generation.py | new safe generation loop around unchanged C1 forward |
| src/semcache/edgelora/r5_runtime.py | application path, independent UD, actual no-reuse baseline |
| src/semcache/edgelora/r5_workload.py | ordinary texts, actual-token capacity construction, no dispatch labels |
| src/semcache/edgelora/r5_evidence.py | immutable outputs, source hashes, bounded/timed evidence |
| src/semcache/edgelora/r5_analysis.py | independent cluster/index/cache/TCP/tensor/generation audit and reports |
| scripts/110_run_r5_semcache_e2e.py | dry-run, fixture, opt-in GPU run and analysis |
| scripts/111_analyze_r5_semcache_e2e.py | separately runnable read-only analysis |
| jobs/r5_opt_semcache_e2e.sh | prepared-but-unsubmitted SERAPH RTX3090 pilot |
| configs/r5_integration.json | explicit pilot semantic and LRU policy |
| tests/test_r5_contracts.py | identity, request seals and schema negatives |
| tests/test_r5_semantic.py | candidate, cluster, mismatch, stale and duplicate controls |
| tests/test_r5_controller.py | actual TinyBackend source proofs, automatic HIT/eviction/integrity |
| tests/test_r5_processes.py | full TCP/Decode path, baseline, resource failures, deterministic/adversarial audit |
| docs/R5_SEMCACHE_E2E_INTEGRATION.md | pre-code map, final contracts, architecture and operations |
| docs/R5_I0_I3_VERIFICATION.md | scope, scenario, file list, execution results and limitations |

## Saved final R5 evidence

`results/r5/local_verification_20261031/r5_tests_verified.log`: **36 passed,
0 failures, 0 errors, 0 skipped**. Contract/semantic/controller unit group:
20 tests; application/process/independent-audit group:16. Compilation and
`bash -n jobs/r5_opt_semcache_e2e.sh` pass. R5's earlier failed TCP/duplicate-window
and overstrict EOS-test attempts remain preserved; they are not successful runs.

`verified_fixture/` is the final sealed G3 strict fixture: 20 completed requests,
60 actual forwards, four HITs, four source admissions, one capacity eviction,
30,720-byte peak Raw-QKV occupancy. The auditor checks 7,040 strict tensor files,
3,840 KV reconstruction comparisons and 256 strict source-prefix byte comparisons.
The paired fidelity report contains 4,800 tensor/KV comparisons. All ten pairs
match generated IDs/text and serialized logits; compared fixture tensors match
bitwise. This is explicitly CPU scalar-fixture fidelity, not FP16 OPT parity.

| Field | Final fixture value |
|---|---|
| R5_I0_CONTRACT_PASS | true, TEST_FIXTURE_ONLY |
| R5_I1_SEMANTIC_PATH_PASS | true, ControlledEncoder integrated path |
| R5_I2_CACHE_LIFECYCLE_PASS | true, actual fixture tensor owner and LRU |
| R5_I3_DISTRIBUTED_GENERATION_PASS | true, independent CPU UD/TCP, HIT/MISS prefill and Decode |
| R5_CPU_FIXTURE_PASS / R5_INDEPENDENT_AUDIT_PASS | true |
| R5_TOKEN_FIDELITY_PASS / R5_STRICT_BITWISE_FIDELITY_PASS | true, fixture only |
| R5_GPU_PHYSICAL_EXECUTION_PASS | false; GPU_NOT_RUN |
| NATIVE_PEFT_STRICT_PARITY_PASS / correctness_qualified_speedup | false |
| I4 multi-adapter/concurrency NOT_IMPLEMENTED / I5 large-scale NOT_RUN | true |

`verified_analysis_a/` and `verified_analysis_b/` independently reanalyze the
immutable fixture. The six audit/decision/fidelity/latency/TCP/limitation reports
are byte-identical to each other and the source reports; see
`analysis_reproducibility.json`. `verified_dry_run/` records dependency checks
without heavy imports or GPU execution. `verified_lifecycle_trace.json` preserves
the concise actual request/candidate/rejection/source/eviction/byte/token trace.

The first HIT R5_01 skips 960 prefix projection-token rows and removes 5,120
HIDDEN payload bytes plus 15,360 DELTAS payload bytes versus N_R5_01. Actual
prefill wire bytes fall by 17,566; total request wire falls by 8,045. Three
10-token HITs plus one 9-token HIT skip 3,744 rows in total. Request-scoped and
whole-session control bytes are separate; the latter includes handshake,
condition-reset eviction and shutdown. Headers/control overhead remain measured.
No GPU/network-performance generalization follows from these fixture numbers.

`historical_preservation.json` confirms zero changed/missing files among the
35,061 pre-task source/result/log files. The existing tracked tree diff is empty.

## Complete CPU regression accounting

Final machine record: `results/r5/local_verification_20261031/verification.json`.
Final preservation check: `historical_preservation_final.json`. All saved final
fixture source hashes match the final executable implementation. No existing
tracked file was changed; 18 implementation/config/test/doc files were added.

| Execution | Exact results |
|---|---|
| R5 final dedicated suite | 36 run; 36 passed; zero failures/errors/skips |
| Frozen R4, 18 modules | 109 run; 98 passed; 10 skipped; one historical branch-gate failure; zero errors |
| R4-C subset, 6 modules | 40 run; 37 passed; 3 dependency skips; zero failures/errors |
| All 120 legacy modules, separate processes | 767 run; 636 successful methods; 101 skipped; 3 failed methods (5 failure items); 27 module import errors; 3 additional class-setup errors |
| Isolated retry of the 3 calibration classes | 7 run; 7 passed; zero failures/errors/skips |

The three setup errors do not count as executed test methods in unittest's
767 count. They prevented seven methods from initially running; those seven
pass in the retry. Failure-item count differs from failed-method count because
one CPU-precision test has three failing subtests.

The remaining three failed methods are two **unchanged** CPU-precision tests
(branch mismatch before expected Slurm/mock orchestration) and the unchanged
R4-A missing-dependency branch-gate test. All five failure items reproduce from
the untouched baseline export. Baseline/current logs and comparison JSON are
in `isolated_diagnostics/`. The 27 unavailable modules fail to import because
pytest is absent; all 27 identical import failures reproduce in the baseline.
These are recorded as import errors, not silently reclassified as skips.
Optional Torch/NumPy/Transformers, datasets/ROUGE/SacreBLEU and explicitly opted-in
GPU tests account for the stated skips; per-test reasons are retained in logs.

The earlier combined discovery attempt is also preserved: 807 run, 37 failure
items, 35 errors and 101 skips. It imports CacheGen modules into the same process
as legacy Raw-only guards, producing cross-suite failures. It also collected
R5 before the final EOS-test correction/report schema, so its R5 EOS assertion
and in-memory old analysis schema are superseded by the fresh 36-pass final
suite. It is explicitly exploratory, not a successful regression run. No failing
result is discarded or promoted to PASS.

`isolated_cpu_modules/results.json` lists every exact legacy module command,
return code, count and log path. The executed runner is saved as
`isolated_cpu_modules_runner.py`; it refuses an existing destination directory.
For portable manual repetition in a clean checkout, run each module in its own
process (rather than mixing compression imports and Raw-only guards):

```bash
for test_path in tests/test_*.py; do
  case "$test_path" in tests/test_r5_*) continue ;; esac
  module="${test_path%.py}"
  PYTHONPATH=src:tests python3 -m unittest "${module//\//.}" -v
done
PYTHONPATH=src:tests python3 -m unittest tests.test_r3c_processes.PhysicalGrid \
  tests.test_r3cf_processes.ActualLegacyFormat tests.test_r3cf_processes.PhysicalFocused -v
```

Run timing-sensitive calibration retries without competing experiments. The
initial errors and the successful retry remain separate evidence. No new
persistent CPU regression was found in executable coverage. Missing-dependency
and GPU checks remain outside verified scope; this is not an all-dependencies
green-suite claim. No GPU work, Slurm submission or remote operation occurred.

I4 remains: true prepared multi-adapter routing, measured richer policies,
logical-user scale and concurrent ownership/scheduling. I5 remains: real natural
traffic, genuine quality references, independent repeated timing cohorts,
semantic-vs-exact ablations and statistically qualified system evaluation.
