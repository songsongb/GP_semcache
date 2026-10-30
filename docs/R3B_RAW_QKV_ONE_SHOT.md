# R3-B1/B2/B3 physical Raw-QKV one-shot experiment

Classification: `R3B_PHYSICAL_RAW_QKV_DIAGNOSTIC`.
`PRIMARY_PERFORMANCE_ELIGIBLE`, `correctness_qualified_reuse_speedup`,
`r2d_break_even_validated`, `cachegen_enabled`, `compression_enabled`,
`semantic_generalization_validated`, and `actual_remote_network_measured` are
always false, even when all three stages pass. This is pretokenized batch-one
L32 prefill, not TTFT, generation, remote hardware or R3-C bandwidth validation.

Codex runs on personal Ubuntu. No SERAPH filesystem/evidence is accessed and no
GPU inference or Slurm job is executed during implementation. Local synthetic
TCP fixtures are explicitly `TEST_FIXTURE_ONLY`, with GPU inference false.

## Actual components integrated

| Existing component | Actual R3-B use |
|---|---|
| `CacheEntry.from_tensors` | Own compact detached clones of all 96 total-QKV outputs on ES GPU |
| `GlobalCache` | Resident dictionary, logical/physical byte accounting, lookup counters, actual reuse frequency, insertion |
| `ExactTokenMatcher` / `Subsequence` | Full-prompt token-key lookup within fixed namespace/cluster 0 |
| `AdmissionPolicy`, normalization | Unchanged default policy, observed candidate frequency 1; unavailable impact maps to zero under existing policy |
| `EvictionPolicy` | Existing normalized score/tie rule; eviction before cloning if capacity requires it |
| `ProjectionSession` | Unchanged one HIDDEN / bundled DELTAS exchange per actual layer on a MISS |
| `Channel`, framing, digest, loopback helpers | Unchanged real 127.0.0.1 TCP, bounded frames, identity/sequence/dtype/shape checks and timeouts |
| `user_device.serve` | Unchanged independent CPU-only service and original FP32 `cpu_deltas`; adapter loaded once |
| `95 --phase prepare`, bare loader/fingerprints | Fresh immutable Native PEFT reference, exact exported A/B and bare GPU model identity |

The existing `mixed_projection_path` validates PEFT Linear modules and
`SemCacheEngine.query()` calls `model.set_adapter()`. Neither can be used
unchanged on an ES that must contain **bare OPT with no GPU LoRA**. R3-B therefore
uses a separate scoped projection callback with the existing physical TCP
session, cache objects and policies. It does not call `SemCacheEngine.query()`
or claim integration of its entire frontend. Intent clustering, semantic
encoding, w=3 candidate selection, CHU/PBR and attention-impact admission are
not executed or claimed. No semantic similarity authorizes reuse.

Only **exact full-prompt replay** is accepted: identical whole token IDs,
logical positions, all-one attention mask, empty causal prefix, no past KV,
model/revision/base fingerprint, tokenizer, adapter/matrix SHA, endpoint,
prepared-manifest SHA and GPU UUID. Logical positions are 0..31; OPT's internal
position offset is preserved by the unchanged forward. The cache is one fixed
prompt/cohort namespace in this experiment, not a multi-prompt service.
No cross-context, natural-hit-rate or semantic-generalization claim follows.

## B1: physical MISS and total-QKV HIT

The ES alone owns embeddings, the complete decoder and LM head. The CPU UD
owns the original user_a FP32 rank8 A/B matrices. The model is OPT-2.7B FP16,
32 layers, batch1 L32, eager, use_cache=False, output_attentions=True, with the
controlled untrained adapter from fresh prepare. There is no trained-user
quality evidence. Actual GPU must be an RTX3090; no node is pinned.

On MISS, each real layer uses original `ProjectionSession.correction()` once
per shared hidden batch, returns bundled Q/K/V FP32 CPU deltas, then computes
the native bias-containing FP16 base and `(base + delta).to(base.dtype)`.
Passive owned captures are taken before later attention transformations.
Insertion is allowed only after the entire forward succeeds, logits are finite,
and all layer/role captures are complete and finite. Partial/failed forwards
never insert. Cache insertion/clone/integrity/admission costs are timed inside
the COLD_MISS request.

On HIT, the actual Transformer still executes attention, FFN, residual/norm and
LM head normally. All 96 QKV forwards return clones of cached **final total
projections** and never invoke the MISS callback, base Linear or UD correction.
The cache does not contain `past_key_values`, LoRA deltas or final logits.
Clones prevent subsequent attention operations from modifying owned storage.
All 32 HIDDEN/DELTAS exchanges must disappear for a full HIT.

The bounded cache defaults to 64 MiB; the expected raw tensor storage for this
shape is 32×3×32×2560×2 = 15,728,640 bytes. This formula is an expected storage
size, not observed network traffic. Physical storage and admission are checked
from actual tensors. Oversized entries are denied without cache materialization.
Resident cache storage is evicted before cloning if necessary. The fixed-prompt
default run normally has zero evictions; no eviction-policy advantage is claimed.

Every lookup validates the full sealed metadata and SHA256 of all stored
tensors, including device/dtype/shape/contiguity, exact layer completeness and
physical byte bounds. SHA validation includes GPU→CPU reads and hashing, which
are **included in lookup and request wall**. These memory-validation bytes are
not TCP bytes. This conservative diagnostic can make HIT expensive; negative
latency results are retained. No optimized/version-only guard is substituted.
Transient forward/capture tensors and model memory are outside the resident
cache-capacity bound.

## B2: separate equality gates

| Scenario | Required physical behavior / correctness |
|---|---|
| NO_REUSE | Real ES/UD TCP, 32 exchanges, 96 outputs; establishes same-run physical reference |
| COLD_MISS | Empty cache, 32 exchanges, completed insertion; entire FP32-serialized logits and all 96 QKV SHA equal NO_REUSE |
| WARM_HIT | Actual validated entry, 96 reused outputs, zero HIDDEN/DELTAS frames and zero UD LoRA processing; full logits/QKV bitwise equal NO_REUSE |
| INVALID_HIT_GUARD | Reject changed tokens, adapter metadata, positions, prefix/context, model metadata, incomplete layer set, incompatible metadata, dtype and shape |

Invalid guards are lookup/validation fixtures. They do **not** execute another
actual adapter, model, token prompt or context, and the report says so explicitly.
Structural guards mutate only the new resident candidate temporarily and restore
the pristine entry; original tensors, weights and historical artifacts are not
changed. Corrupted tensor bytes are separately tested in CPU units.

All paths are also compared against fresh Native PEFT with the unchanged R3-A
`check_parity()` and THRESHOLDS. Native strict FAIL is preserved per request,
independently of COLD/WARM versus physical NO_REUSE bitwise equality. It does
not falsely become a cache-corruption finding. If cache equality passes while
Native fails, B3 is allowed only under this diagnostic classification, with
correctness-qualified reuse-speedup eligibility still false. The old 95 pilot
and latency validators are unchanged and are not used to qualify this result.

B1 failure stops before B2/B3. B2 failure stops before B3. Each stage has a
timestamped START/PASS/FAIL record and a separate projection/wire trace. One ES
worker persists through the stages because GPU cache residency must persist;
worker stdout/stderr identifies stage boundaries. UD work counts are reconciled
against every request and both endpoint frame digests.

## B3: measured scope and sample counts

Defaults: 1 warmup and 5 measured repetitions **per mode**. B3 follows B1/B2
PASS and cycles `NO_REUSE → COLD_MISS → WARM_HIT` for each repetition. This fixed
order is documented, not randomized or a workload distribution. Each cold
request resets cache before its timed wall and repopulates it; the next HIT
therefore starts populated. Reset is an experimental state precondition and
its host duration is reported separately outside the request wall. A final
CUDA synchronization precedes request timing after the reset.

Default output has 3 correctness requests + 18 B3 requests = **21 individual
rows**, with exactly 5 measured rows per mode. Warmups are retained and excluded
from medians/p90. p90 uses linear interpolation at 0.9×(n−1). Counts, signed
latency outcomes, minimum and maximum are retained; no favorable sample clipping.

Synchronized request wall includes cache lookup/validation, scoped routing,
capturing clones, model forward, real blocking TCP, copies/casts, CPU processing,
serialization and cache admission/materialization. Model load/fingerprint,
adapter load, handshake, prepared token tensor, shutdown and post-wall full-logit/
QKV quality export are excluded. Cache integrity checks remain inside timing.
Synchronous per-frame wire-trace writes are also inside the request wall;
these are instrumented diagnostic latencies, not an uninstrumented service claim.

CUDA event elapsed is a forward interval including stream idle/socket gaps,
not GPU kernel self-time. CPU UD, send/recv, copy/cast, framing, lookup/insertion,
host forward and CUDA timings are nested/overlapping and must **not be summed**
or subtracted to invent pure RTT/TCP or GPU compute costs. Cache-copy/hash self
times not separately measured are explicitly marked; their inclusive costs
remain in lookup/insertion/forward/request wall.

Application wire bytes include the existing binary prefix, JSON header and raw
payload accepted/read by TCP. They exclude kernel TCP/IP headers, ACKs and
retransmissions. HIDDEN/DELTAS directions and actual frame counts are reported
separately. HELLO/READY and STOP/BYE are outside requests and remain in session
traces. No h0/hL boundary is fabricated: ES owns those endpoints. Local cache
copy bytes never count as communication savings. This is not an R2-D endpoint
equivalence assertion, remote-link test, bandwidth constraint or R3-C break-even.

## Running on SERAPH (user submits)

No original file or result is changed. New entrypoint:

```bash
python -u scripts/99_run_r3b_raw_qkv_one_shot.py \
  --phase run --enable-gpu-experiment --device cuda:0 \
  --warmup-runs 1 --measured-runs 5 --cache-capacity-mib 64 \
  --output-dir results/r3/b/raw_qkv_one_shot_NEW_JOB_TAG
```

The runner requires a Slurm GPU allocation and branch `exp/raw-semcache-r3`.
It calls existing 95 prepare with OPT-2.7B/FP16/L32/user_a/seed42 in a separate
process and fully reaps it before loading the ES. Then ES starts the original
independent CPU-only UD. Fresh prepare/ES GPU UUID, versions, full base and A/B
identity are verified. Old failed prepares/pilots are not inputs. A different
historical GPU UUID does not invalidate a fresh cohort; within-cohort mismatch
does. Loading is local-files-only with HF offline flags.

The requested `r1b3_b21_setup.sh` is absent in this checkout. The new launcher
uses the actual existing jobs' Slurm/conda/offline conventions, adds repository
logs and a one-allocation lifecycle, and does not edit existing launchers.

After manual synchronization to SERAPH:

```bash
cd /data/khuss/repos/GP_semcache
git switch exp/raw-semcache-r3
git pull --ff-only
mkdir -p logs
sbatch jobs/r3b_raw_qkv_one_shot.sh
```

`logs/.gitkeep` also preserves the directory: Slurm opens logs **before** Bash
starts. The launcher selects partition `batch_eebme_ugrad`, GPU1, CPU4, RAM64G,
2h resource cap, no fixed node, conda `semcache`, HF_HOME=/data/khuss/huggingface,
HF_HUB_CACHE=$HF_HOME/hub and offline mode. If conda is not on batch PATH, set
`R3B_CONDA_ROOT` to its actual installation root. Optional `R3B_REPO` changes
Bash's repository path; adjust Slurm `--chdir` consistently for a different site.
No allocation, job or remote sync is started by Codex.

Run root: `results/r3/b/raw_qkv_one_shot_<jobid>_<UTC timestamp>/`.
No nonempty output can be overwritten. Failures preserve Python traceback,
stage state, raw partial rows, stderr and lifecycle. The reused descendant-aware
cleanup also handles prepare95/UD processes in separate process groups.

## Output contract

- `r3b_one_shot_results.json`: source HEAD/branch/dirty state, source/input/output
  hashes, prepared identity/GPU, worker/process metadata, B1/B2/B3 gates,
  independently reconciled audit and descriptive statistics.
- `r3b_one_shot_summary.txt`: stage status, Native failure count and summaries.
- `r3b_requests.jsonl`: every request separately, including warmups and available
  failure rows; exact full-logit/QKV SHA, native strict metrics, bytes, timings,
  status, cache occupancy/events and physical reuse count.
- `prepare/`: unchanged 95 reference/adapter artifacts and binary tensors.
- `prepare_process/`: wrapper stdout/stderr and descendant lifecycle.
- `es/`: ES stdout/stderr/lifecycle, original UD stdout/stderr/result/ready file,
  `r3b_worker_result.json`, `r3b_es_partial.json`, `r3b_stage_gates.json`,
  `B1_gate.json`, `B2_gate.json`, `B3_gate.json`, per-stage projection and wire
  JSONL, full session ES/UD wire JSONL, bounded physical NO_REUSE reference logits
  and 96 reference SHA. B2 data wire trace is legitimately empty on a full HIT.

Original 95 prepare and original UD service files retain their frozen schema;
the enclosing new experiment/rows use the R3-B diagnostic classification.
An incomplete stage never becomes PASS merely because files exist.

## Local verification and remaining limits

New CPU tests exercise actual CacheEntry/GlobalCache/Matcher/default policies,
ownership, capacity, admission, corruption, exact guard restrictions, bitwise
gates, negative timing outcomes, all-96 counts, B1/B2 failure stopping, no-overwrite
and true independent-process loopback cleanup. The stdlib 32-layer tiny decoder
uses explicit `TEST_FIXTURE_ONLY` CPU tensor storage; it is not OPT/GPU/FP16
evidence. A separate synthetic native-reference perturbation proves Native FAIL
does not suppress cache equality or silently qualify the result. No real
historical reference is altered by that unit fixture.

```bash
PYTHONPATH=src:tests python3 -m unittest -v test_r3b_cache test_r3b_analysis test_r3b_processes
python3 -m compileall -q scripts/99_run_r3b_raw_qkv_one_shot.py src/semcache/edgelora/r3b_cache.py src/semcache/edgelora/r3b_runtime.py src/semcache/edgelora/r3b_analysis.py src/semcache/edgelora/r3b_fixture.py tests/test_r3b_cache.py tests/test_r3b_analysis.py tests/test_r3b_processes.py
python3 scripts/99_run_r3b_raw_qkv_one_shot.py --help
bash -n jobs/r3b_raw_qkv_one_shot.sh
```

Socket tests need permission outside a sandbox that blocks localhost sockets.
Torch/NumPy tensor tests and OPT GPU tests skip explicitly when dependencies/
allocation are absent. Existing pytest-based physical engine tests cannot run
without pytest/Torch; no skipped/uncollected test is called successful.
The opt-in GPU unittest requires SEMCACHE_R3B_GPU=1 and a new
SEMCACHE_R3B_OUTPUT within an actual allocation. Public `--phase fixture
--test-fixture-only` exists solely for CPU integration, never as a production
fallback. `--phase dry-run` performs readiness only, with no model/socket.

Remaining unvalidated: real OPT GPU B1/B2/B3, Native parity of that new cohort,
remote UD/link behavior, semantic/window/context reuse, natural workloads,
trained-personalization quality, full generation and R3-C/R2-D break-even.

Implementation verification: new suite **23 tests: 21 passed, 0 failed,
2 skipped** (actual Torch ownership and allocated OPT GPU). Relevant unittest
regressions including original R3-A socket/process tests, 95–98 diagnostics,
R2-D/M9-A/C3/C4: **183 tests: 174 passed, 0 failed, 9 skipped**. Compilation,
CLI help, shell syntax and the retained CPU-only real-TCP synthetic smoke passed.
That smoke has 12 requests (1 warmup/2 measured per mode plus correctness),
explicit TEST_FIXTURE_ONLY flags and GPU inference false. Socket tests ran
outside the sandbox with permission. All 525 preexisting tracked files are
SHA256-identical; the original 15-file production identity is unchanged.
GPU findings in the development environment remain **NOT_RUN**.
