# D5-L6-A CPU long replay

The experiment fixes 2WikiMQA, 50 Mbps, UD factor 0.25, the v5 eight-token generation policy and 64 resident entries per adapter. FULL, EXACT and APPROX each start with fresh sessions and isolated per-adapter owners. Owners persist between cycles. The CPU-only 128-entry configuration keeps the same 512 MiB byte cap per adapter and the same candidate-selection bound.

The default input is the sealed plan and CPU inventory in `results/r5/d5_lightperf/sweep_v5_20261122T141858Z/prepared/bw050_ud025/{plan,cpu}`. Pass the cell directory to `--reference` if needed. The historical CPU record authenticates the reference only. The planner verifies both inventories and the historical plan ID, checks the reference source commit, obtains the UD planning assumption from that plan and calls Light `material()` on its original offline assets. This reuses `replay_stream()`, original-order selection, deterministic request IDs, tokenizer/model/dataset checks, transport and resource bounds. It compares full tasks and provenance for the first 20 positions, including tokens and generation policy. Cycle boundaries are inclusive ordinals 0–9, 10–19, 20–29, 30–39 and 40–49.

Use the following CPU commands from the repository on the machine with the frozen assets. The entry point enforces hidden CUDA/HIP and offline settings before package imports.

```bash
PY=/data/khuss/anaconda3/envs/semcache/bin/python3
CLI=scripts/164_r5_d5_l6a.py
NS=results/r5/d5_l6a
"$PY" "$CLI" check
"$PY" "$CLI" preflight
```

After source review, supply the approved clean HEAD and fingerprint as shell variables and choose fresh direct children of the L6 namespace. For a development plan omit both review arguments. A development audit stays bound to that development plan.

```bash
"$PY" "$CLI" prepare-cpu --output "$NS/$PLAN_TAG" \
  --reviewed-commit "$REVIEWED_COMMIT" --reviewed-fingerprint "$REVIEWED_FINGERPRINT"
"$PY" "$CLI" cpu-validate --plan "$NS/$PLAN_TAG" --output "$NS/$AUDIT_TAG"
"$PY" "$CLI" inspect --plan "$NS/$PLAN_TAG" --audit "$NS/$AUDIT_TAG"
```

L6 uses its own plan schema, baseline ancestry at 40717c1, pinned experiment branch, code fingerprint, L6 test coverage and inventory-bound audit. Inspection reconstructs the plan from the frozen reference and current source. Entry count, cycles, inputs, resource report and source are included in the plan digest. For structural CPU testing pass `--entry-limit 128`. The phase is `CPU_STRUCTURAL_ONLY`. Physical projection on a non-CPU device rejects this setting. The CLI exposes only CPU commands. All six v5 results remain historical read-only inputs.

The forecast is per condition and includes all-fresh traffic, all forwards, UD slowdown and startup/sealing reserve. A blocked forecast records its reasons and cannot produce CPU approval. The default is 7200 seconds. For the 64-entry long replay, add `--worker-deadline-seconds 9000` to both preflight and prepare-cpu. The V2 plan binds three independent allocations, one fresh 50-request session per condition with all five cycles contiguous. The L6 planner evaluates the existing forecast against 9000 and retains all other blockers. For the supplied SERAPH forecast of 7824.54483968 seconds the time margin is 1175.45516032 seconds per condition. A future scheduler runner must check its own job's EndTime immediately before launch using the L6 remaining-time contract (strictly more than 9120 seconds, at most 16200), and propagate the actual remaining seconds. The existing ES reads the verified plan deadline, writes one absolute monotonic deadline to the UD configuration, and passes it through generation. Different allocations still require official backend, GPU and CPU identity checks before comparison. The L6 native Light plan producer, reviewed CPU approval binding and scheduler integration are pending. Receipt size and memory limits also apply at write time.

## Future read-only analysis

```bash
"$PY" "$CLI" analyze --plan "$NS/$PLAN_TAG" --light-plan "$LIGHT_PLAN" \
  --light-audit "$LIGHT_AUDIT" --full "$FULL_DIR" --exact "$EXACT_DIR" \
  --approx "$APPROX_DIR" --output "$NS/$ANALYSIS_TAG"
```

The three paths are sealed run directories containing run.json. A future reviewed Light producer must bind its 50-request native plan to the L6 plan ID using `l6_plan_id` and explicit `entry_limit=64`. It needs its own matching Light CPU approval. The L6 CPU-only plan/audit alone cannot launch the Light worker. The analyzer uses the official Light verifier, identical Light plan/audit paths, workload routing and official backend/device comparison. EXACT requires all 50 token sequences to agree with FULL. APPROX agreement and original scorer mean are reported separately.

A successful analysis publishes `long_replay_analysis.json`, `cycle_metrics.csv`, `cumulative_metrics.csv` and `light_inventory.json` in a new directory. Each CSV has 15 rows, five per condition. Nested statistics are JSON within CSV cells. Each delta is condition minus FULL in ms at the same ordinal. Speedup divides summed FULL E2E by summed condition E2E over exactly the same rows. The whole-session span is reported separately in JSON. Samples of entry count and physical bytes carry each idle adapter's last state forward from the beginning of the session. Totals are sums across adapters at each request boundary, prior to computing mean, maximum and final. Policy counters already represent per-request differences. Inclusive timers can overlap. Unobserved components remain unmeasured. The eight-token cap and one replay do not establish quality preservation or correctness-qualified speedup.
