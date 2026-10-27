# R2-B2 Exact Prefix KV paired latency and runtime profiling

Implementation status: **IMPLEMENTED_LOCAL_ONLY; GPU_SMOKE_NOT_RUN**. Local
Torch, Transformers and PEFT are unavailable. No new model correctness, latency,
kernel attribution or speedup finding is claimed. Historical B1.5 numbers are
motivation only and are never inputs to this benchmark.

Initial checkout: `exp/raw-semcache-r2`, HEAD
`ce1f459621aa7bbe79a0c5ababda7bf99746f0b0`, initially clean. No frozen 85/86,
R1/R2-A/B1/B1.5 runtime, results, tolerances or Slurm files are changed.

## Execution and fairness

New `B2Runtime` inherits B1 loader, controlled rank-8 LoRA fixture, seed, exact
tokenizer fixture, namespace/weight hashes, independent source prefill,
canonical `PrefixSnapshot`, per-request cloning, installed Cache API detection,
negative tests, memory preflight, and mask/position/physical-row audits.

B1 hardcodes returned attentions and all logits in its post-forward checks.
The B2-only forward body preserves its entire timed `with` block (AST equality
regression test) and specializes the post-timer output contracts. `B2API` delegates
to the original input/mask/absolute-position builder, changing only explicit
attention-output/native-logit kwargs. No head, decoder or projection arithmetic
is replaced. B1's GPU monitor is overridden only in B2.

| Mode | Operation | Request timer |
|---|---|---|
| FULL_USE_CACHE | Native full X, no incoming cache | adapter/eval, input/mask/positions, native forward, final KV and logits |
| WARM_NORMAL | Independently built exact X[:P] KV, native X[P:] | same operations plus namespace, per-request KV clone, container/factory and shape validation |
| WARM_PREPARED_DIAGNOSTIC | same suffix using fresh one-use ticket | B1.5 Prepared guard retained; materialize performed before timer; **IDEAL_DIAGNOSTIC_ONLY** |
| PREPARE_ONLY | snapshot materialize; no model forward | separate component/profiler phase only |

All are batch-one eval/inference, `use_cache=True`, same synthetic adapter,
model/tokenizer revision, exact IDs, dtype/device and eager implementation.
P=0 is a native no-hit control, has no source and no Prepared call. Its Prepared
columns are null/NOT_APPLICABLE, and it cannot qualify as cache-reuse speedup.
No source KV is obtained from Full target slicing. Source prefill, snapshot
creation/bytes, and source proof pass remain separately recorded.

## Attention and endpoint capability

`--attention-arm on|off|both` sets only `output_attentions` and independently
builds/audits each arm. Actual config, OPT attention class and installed forward
source hash are checked. Supported eager evidence is `OPTAttention` using
`torch.bmm` or `eager_attention_forward` with eager config. SDPA/Flash/config
fallback is rejected as BACKEND_CONFOUNDED; unknown implementations are
UNVERIFIED_BACKEND, not silently treated as eager. Returned attention shape is
validated ON; OFF must return no attention tensors. OFF does **not** prove that
eager skipped internal probability computation. The saved source/config check
is not kernel-level evidence. Installed API compatibility remains GPU-unverified.

Endpoint policies:

- `b1`: native all-logit output. Full [1,L,V], Warm [1,S,V]; marked
  materialization-confounded and **not output-matched serving**.
- `auto` (default): use an explicit native `logits_to_keep` or
  `num_logits_to_keep` signature parameter if present, otherwise disclosed `b1`
  fallback. No arbitrary **kwargs capability assumption.
- `native-last`: require that native option; otherwise fail UNSUPPORTED_ENDPOINT.

Native last emits [1,1,V] from Full and Warm, with final KV length L in both.
The same output policy is used for source, Full and Warm. Untimed all-logit
controls compare last rows at the unchanged tolerance; failure stops the native
endpoint experiment. Full-suffix strict correctness is also retained in that
control, so returning only last logits cannot turn a known suffix FAIL into a
qualified PASS. With attention ON, returned attention query dimensions still
differ; only required logits/final KV are output-matched. No custom LM head.

## Minimal protocol and statistical unit

`smoke` performs structural gates and three minimal pairs by default; it does
not automatically enable a profiler. `latency` requires >=3 warmups and >=20
measured pairs per case/arm. Run at least three **separate Python processes** with
distinct run IDs/directories/order seeds but identical workload seed (42).
Paired mode schedules use seeded randomized cyclic Latin blocks; each mode
occupies every slot per complete block. Incomplete last blocks may differ by one.
Arm order reverses by order-seed parity; the actual order is recorded.

Primary clock: synchronized CPU request wall, pretokenized input to device
logits and final KV. Adapter selection/eval and cache preparation stay inside.
Tokenization, source construction, heavy source hashes, endpoint/physical hooks,
CPU logit transfer, output integrity checks, frame cleanup and file writes are
outside, preserving B1's boundary. This is not full host-response latency or
generation TTFT. Final caches from earlier modes can remain live until the pair
parity check; order is balanced and peak memory recorded. Outside-timer integrity
work and Prepared prebuild can affect request-entry GPU state: Normal-Prepared
is not a pure clone-cost subtraction.

Raw CSV and a typed JSON companion are checkpointed after each complete pair,
before session summaries. Pair differences are Full minus Normal: positive
favors Normal. Session statistics include mean/p50/p95/std, variance, wins,
Full/Normal ratio and Prepared contrasts. Raw outliers are never discarded.

`analyze` requires COMPLETE latency sessions, >=20 pairs, unique process instance
and run IDs, same code/implementation/model/namespace/token/source snapshot,
software/GPU identity, endpoint/arm conditions and workload seed. It rejects
incomplete and duplicated evidence. Only order seed and run identity may vary.
It preserves each run's result and equally weights **session means**. Bootstrap
resamples complete session means, not pooled within-session pairs; its 95% CI
is exploratory, especially with only three sessions. Between-session variance
is reported separately. Mixed signs are not a stable-speedup conclusion.

Qualification requires nonempty prefix, structural/cache/parity checks,
strict correctness, known CUDA identity, >=3 sessions, all positive session
deltas and positive lower exploratory CI. LATENCY_REPEATED alone is no speedup.
Even qualification is conditional warm-hit evidence, not amortized service
benefit or safe semantic/cross-user reuse. Prepared never qualifies a claim.

## Correctness and failure gates

Untimed B1 hooks prove L native rows versus S rows at every decoder/QKV/FFN,
actual token IDs, causal mask and learned absolute position embeddings. Each
arm validates P→L cache lengths and immutable canonical content; final model
weight hashes are checked. Normal/Prepared final KV **and** logits must be
exact. Prepared cache is fresh, independently cloned and single-use; mutation
or reuse is rejected by frozen B1.5 `PreparedTicket`.

Full-relative strict correctness uses `atol=rtol=1e-3`; Argmax and KL remain
separate metrics. STRICT_FAIL permits mechanism timing but never qualified
speedup. Structural/cache/parity/endpoint failures stop execution with traceback
and active context. Already persisted pairs/gates/profile evidence are retained.
Output collision is rejected; there is no resume/overwrite mode. Use a new tag.

## Profiling boundary and GPU monitoring

`profile` is a separate process/phase, never read by minimal summaries. After
warmup, one small trace per case/arm/mode records CPU+CUDA activities where
available, actual operator shapes/memory and actual module ranges:
adapter/eval, input preparation, namespace, clone/container, native forward,
decoder layers, attention inclusive/output, FFN and LM head. A separate
PREPARE_ONLY trial has no model call. Instrumented outputs/final KV must equal
an uninstrumented reference. Their one-trial latency delta is noisy diagnostic
overhead evidence, not a performance comparison.

CPU operator **self** time, CUDA attributed device self time, raw trace kernel
duration/count, runtime launch calls, memcpy events, and component CUDA stream
event intervals are separate columns. Parent native-forward/attention ranges
contain children. Never sum nested scopes, CPU and GPU clocks, or kernel
duration sums into request wall. Trace GPU activity union and sum differ.
Exclusive CPU attribution uses the nearest real B2 range. Raw GPU correlation
parsing reuses R2-A's tested parser via in-memory annotation-prefix adaptation;
the original trace is unchanged. Unlinked events stay unknown. The enclosing
invocation trace includes outside-request hashes/output transfer and Prepared
prebuild: use the native_forward/materialize subranges for those components.
No cuBLAS algorithm, occupancy, Tensor Core utilization, or causal CPU dominance
is inferred. Missing Kineto reports PROFILER_UNAVAILABLE, not zero kernel cost.

Monitoring is outside request blocks. B2 queries **all** nvidia-smi GPUs, matches
Torch's actual UUID (allowing missing `GPU-` prefix), and never assumes visible
CUDA index equals physical index. No `--id=<UUID>` is passed. Clock/power/temp,
memory/utilization and other processes are retained when available; absent
fields are null and status GPU_MONITOR_UNVERIFIED. Affinity is not changed.

## Artifacts

All model phases: manifest, environment, correctness_and_gates, cache_integrity,
forward_skip_audit, attention_backend_audit, endpoint_parity, source_costs and
decision_summary JSON. Manifest includes current branch/HEAD/dirty status,
source hashes, runtime API signatures/revisions, namespace, seeds, process ID,
actual arms and endpoint settings. Loader provenance default ON is explicitly
marked as default-only; per-arm audit is authoritative.

Smoke/latency: latency_raw_pairs.csv/.json, latency_session_summary.csv.
Profile: profiling_cpu_summary.csv, profiling_cuda_summary.csv,
profiling_kernel_counts.csv (raw runtime/activity counts), profiling_components.csv,
profiling_status.json and profiler_trace/*.json. Analyze:
latency_aggregate_summary.json, manifest and decision_summary. Unsupported phases
do not manufacture missing timings. Trace size is one request per mode; no tensor
dump and no large sweep.

## SERAPH commands (offline)

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/87_run_r2b2_paired_latency.py --phase smoke \
  --model facebook/opt-125m --dtype float32 --cases 32:16,32:31 \
  --device cuda:0 --cache-device cuda:0 --attention-arm both \
  --endpoint-policy auto --warmup-runs 3 --measured-runs 3 \
  --independent-run-id smoke01 --output-dir results/r2/b2/smoke01

# Three separate process launches; identical seed, distinct order seeds.
for run in 1 2 3; do
  python scripts/87_run_r2b2_paired_latency.py --phase latency \
    --model facebook/opt-2.7b --dtype float16 --cases 128:0,128:64,128:127 \
    --device cuda:0 --cache-device cuda:0 --attention-arm both \
    --endpoint-policy auto --warmup-runs 3 --measured-runs 20 \
    --seed 42 --order-seed "$((900 + run))" --independent-run-id "run${run}" \
    --output-dir "results/r2/b2/latency_run${run}"
done

python scripts/87_run_r2b2_paired_latency.py --phase profile \
  --model facebook/opt-2.7b --dtype float16 --cases 128:64,128:127 \
  --device cuda:0 --cache-device cuda:0 --attention-arm both \
  --endpoint-policy auto --warmup-runs 3 \
  --independent-run-id profile01 --output-dir results/r2/b2/profile01

python scripts/87_run_r2b2_paired_latency.py --phase analyze \
  --inputs results/r2/b2/latency_run1 results/r2/b2/latency_run2 results/r2/b2/latency_run3 \
  --order-seed 901 --output-dir results/r2/b2/aggregate01
```

For strict B1 all-row comparison, use `--endpoint-policy b1` consistently in
new isolated runs; never aggregate native-last and all-row sessions. Pin
`--revision` and `--tokenizer-revision` when reproducing a specific snapshot.
512+ tokens, SDPA, offload, multiuser amortization and model optimization are
outside scope. Optional `256:192` is explicit only after primary GPU validation.

Local validation commands:

```bash
PYTHONPATH=src:tests python3 -m unittest test_r2b2_benchmark \
  test_r2b2_attention_ablation test_r2b2_analysis test_r2b15_numeric \
  test_r2b15_cost test_r2b1_prefix_cache test_r2b1_prefix_runtime \
  test_r2a_overhead_audit test_r2a_runtime test_r2a_profiler -v
python3 scripts/87_run_r2b2_paired_latency.py --help
git diff --check
```

Pure-Python tests use explicitly synthetic timing/control fixtures, never GPU
performance evidence. Tiny random OPT/PEFT CPU/CUDA integration tests cover both
arms, no-hit/partial/one-token suffix, native endpoint capability and KV/logit
parity, and SKIP if dependencies/device are missing.

Local verification on 2026-10-28: 80 tests total, 63 passed, 17 skipped, zero
failures across new B2 and existing B1.5/B1/R2-A suites. New B2 suite has 17 tests
(15 passed, two dependency-gated model tests skipped). Syntax compilation and
CLI help succeeded. GPU Smoke remains NOT RUN; missing model dependencies are
not interpreted as GPU validation. Expected traceback text in unit-test logs is
from deliberately injected failures whose provenance is asserted.
