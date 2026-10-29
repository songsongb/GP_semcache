# R3-A physical localhost EdgeLoRA no-reuse baseline

R3-A implements independent ES/UD operating-system processes exchanging real
tensor bytes over IPv4 TCP **127.0.0.1 only**. Its target is OPT-2.7B FP16,
batch one, 32-token prefill, eager attention, use_cache=False,
output_attentions=True and the native all-token LM head. The user_a rank-8
Q/K/V adapter is a **controlled, nonzero, untrained fixture**. Its logits are
correctness references, not task-quality or trained-personalization evidence.

This is a no-reuse baseline. There is no SemCache admission/selection, QKV
source cache, physical HIT/MISS, CacheGen, codec, quantization, bandwidth
emulation or RTT injection. Loopback is not remote networking, genuine UD
hardware, TTFT or full generation. R3-A does not validate the R2-D 344 Mbps
claim or any correctness-qualified reuse speedup.

## Inspection and mapping to the repository

The inspected branch was `exp/raw-semcache-r2`, initially clean. A SHA256
snapshot of all tracked files was taken before implementation. Only the four
explicitly reserved EdgeLoRA stubs are changed; other implementation is added.
No launcher or prior experiment output is changed.

| Inspected contract | Mapping in R3-A |
|---|---|
| `edgelora/edge_server.py` (empty reserved stub) | Bare OPT loader, layer coordinator and scoped QKV instance-forward routing |
| `edgelora/user_device.py` (empty reserved stub) | Independent CPU-only adapter service; raw matrix loading and actual CPU LoRA |
| `edgelora/communication.py` (empty reserved stub) | Bounded framed loopback TCP, per-frame digest/identity and actual byte accounting |
| `edgelora/collaborative_inference.py` (empty reserved stub) | Isolated prepare process, ES/UD lifecycle, gates and phase exports |
| `edgelora/lora_projection.py` | Preserved single-process emulation; not used as physical distributed evidence |
| `models/loader.py` | Existing explicit local-files-only OPT loader, with immutable resolved model/tokenizer revisions pinned after preparation |
| `models/model_adapter.py` | Existing OPT decoder layout and actual q_proj/k_proj/v_proj access; no attention rewrite |
| `models/lora_decomposition.py` | Existing PEFT validation and exact base-plus-delta order checked in preparation |
| `models/lora_fixtures.py` | Existing `initialize_fixture()` fills actual user_a A/B parameters using CLI seed, std=.01 |
| `system_cost/model.py`, `network.py` | Preserved additive analytical accounting; only separate byte reconciliation is reported |
| `system_cost/base_runtime.py`, `base_projection.py` | Preserved profiling-only base-QKV latency surrogate; not reused as personalized inference |
| `14_validate_edgelora_projection_path.py` | Preserved single-process correctness probe; R3-A adds physical transport rather than relabeling it |
| `35_profile_m9a1_es_base.py` | Existing isolated-load pattern informed preparation; its measurements/results are unchanged |
| `94_run_r2d_edgelora_break_even.py`, R2-D documentation | Preserved separate-cohort analytical contract; no R2-D totals enter request walls |

There was no actual historical R2-D decision/sweep under `results` in this
checkout. The available `/tmp/r2d_TEST_FIXTURE_ONLY.gJ7Sur/output` decision and
sweep are explicitly synthetic and NOT_COMPARABLE; they provide no measured
344 Mbps result. Historical full source-tree hashes may include the previously
empty reserved stubs. Revalidating such old evidence requires its original
recorded checkout, rather than relabeling old hashes after these stubs change.

## Architecture and process lifetime

The CLI/orchestrator imports no tensor/model library. Production preparation
starts one isolated child with the native PEFT model on CUDA. It records the
complete bare OPT state_dict fingerprint before attachment, verifies that
normalized base tensors are unchanged after PEFT attachment and native
reference inference, exports actual A/B weights, and captures the reference.
The child must exit successfully before the prepare manifest becomes COMPLETE.
Its CUDA allocations therefore cannot coexist with the distributed ES model.

Pilot/latency start two fresh independent children:

- **UD:** CUDA/HIP devices hidden before importing Torch; only CPU matrices are
  loaded. No Transformers/PEFT or full OPT is loaded. Effective affinity is
  restricted to at most four available CPUs, Torch threads to at most four and
  interop threads to one. CUDA context state and matrix CPU placement are
  checked. PID, affinity, threads, versions and environment are recorded.
- **ES:** one bare OPT model on CUDA, no PEFT module or A/B parameter. The
  original Linear forwards retain their weights and biases. Scoped instance
  forwards obtain each layer's transported correction before attention consumes
  Q/K/V. All forwards are restored in `finally`.

On the first real Q/K/V call for a layer, the ES seals the actual hidden input,
sends it once and receives a single bundled `[3,1,N,H]` delta. The remaining
two calls must use bit-equal hidden input; duplicate projections, changed input,
skipped/out-of-order layers, or unsupported call arguments fail closed. There
is no dependence on Q/K/V call order and no prior-layer activation reuse.
Every request must observe all configured layers and three outputs per layer.
For OPT-2.7B the gate expects 32 layers/96 QKV outputs; those counts are only
reported as observed after matching both endpoint traces and forward records.

UD readiness precedes ES model loading. A bounded 300-second initial accept
wait allows cold model loading; per-frame deadlines are 30 seconds. ES runtime
has a bounded 600-second process wait; preparation has 900 seconds. Errors close
sockets, and the orchestrator terminates/reaps remaining process groups, with
a bounded kill fallback. Exit codes, forced cleanup and PIDs are exported.

## Precision and identity gates

Production preparation explicitly requests PEFT adapter autocasting and then
**checks actual FP32 A/B parameters**. Unsupported precision paths stop, rather
than substituting unrelated FP32 matrices. Rank=alpha=8, scale=1, dropout=0,
vanilla unmerged Q/K/V LoRA and adapter user_a are fixed. Native intermediate
input/output dtypes of both A and B are captured. Actual nonzero deltas and
exact native PEFT decomposition are checked for all reference projections.

The distributed computation follows the same operation order:

```text
ES hidden: FP16
UD x = hidden.to(A.dtype)                      # FP32
UD intermediate = F.linear(x, actual_A)        # FP32
UD delta = F.linear(intermediate, actual_B)*1  # FP32
wire: one bundled FP32 q/k/v delta
ES result = (native_Linear(hidden) + delta).to(native_base_output.dtype)
```

CPU versus CUDA FP32 GEMM rounding can still differ. TF32 is disabled and CUDA
deterministic algorithms are enabled in the native/ES GPU workers. Neither
mathematical equality nor matching token IDs alone can establish parity.
The fixed thresholds are declared in `r3a_artifacts.THRESHOLDS` before loading:

| Gate | Fixed threshold |
|---|---:|
| Full all-token elementwise comparison | atol=1e-3, rtol=1e-3; zero violations |
| Maximum absolute logit difference | <=1e-2 |
| Relative L2, reference norm denominator | <=1e-3 |
| Last-token KL(reference || distributed), stable log-softmax | <=1e-5 |
| Last-token argmax agreement | True |

All gates must pass on every warmup and measured iteration. Original output
dtype must match, and logits must be finite. Any failure is PARITY_FAIL, with
raw evidence retained; remaining iterations stop. No tolerance is weakened and
no failed run becomes an eligible latency result.

Prepared identity includes resolved base and tokenizer revisions/source,
complete base-weight checksum, actual adapter checksum, token IDs/hash, dtype,
shape, precision policy, endpoint, seed, runtime versions and implementation
hashes. The ES checks its newly loaded bare state against the native reference;
the ES and UD independently verify the prepared adapter artifact hash. The UD
also verifies every decoded matrix's dtype, shape, byte range, SHA and nonzero
content. Stable identity, source hashes, pilot device/GPU UUID and all output
copies are rechecked before latency/analysis. Different weights are INVALID.

## Safe artifact and wire formats

`r3a_adapter.bin` is concatenated raw little-endian float32 matrices, in
layer/qkv/A-B order. Its JSON manifest contains offsets, shapes, actual dtypes,
scale and per-matrix/file SHA256. The blob is bounded at 128 MiB. It is neither
pickle nor a torch.load archive. The ES hashes it without constructing adapter
tensors. The UD creates only CPU tensors from bounded byte buffers. The bounded
reference logits artifact is raw little-endian FP32, preserving the original
native logits' values and recording the original dtype. Base checkpoints are
loaded only through the existing local HF cache loader; no socket supplies a
checkpoint. Child environments force HF_HUB_OFFLINE=1 and
TRANSFORMERS_OFFLINE=1; the loader uses local_files_only and trust_remote_code=False.

TCP framing uses a 20-byte `!8sIQ` prefix: magic `R3ATCP01`, JSON-header length
and tensor-payload length. Canonical JSON includes protocol version 1, kind,
session nonce, per-direction sequence, request ID, layer ID, adapter blob SHA,
dtype, shape, payload bytes/SHA256 and bounded metadata. Headers are <=64 KiB;
payloads <=64 MiB. Unknown/duplicate fields, invalid units/shapes, stale identity,
bad digest, version, sequence, order, timeout, unexpected close and truncation
fail closed and close the stream. There is no pickle, arbitrary object decoding,
external bind/address parameter or message-driven model loading.

The preferred observed session schedule is:

```text
outside request: HELLO -> READY
each forward:   HIDDEN(layer 0) -> DELTAS(qkv, layer 0)
               ... actual intervening native attention/decoder work ...
               HIDDEN(layer L-1) -> DELTAS(qkv, layer L-1)
outside request: STOP -> BYE
```

`wire_bytes` measures bytes accepted/read by the application socket, including
prefix and JSON. It excludes kernel TCP/IP headers, retransmissions and NIC
traffic. Each direction's frames, payload and application wire bytes are
recorded independently; ES/UD trace copies must agree and are not added
together as extra traffic. Partial/error records retain actual I/O progress.

## Ownership and M9-A/R2-D disagreements

ES owns prepared IDs, tokenizer, embedding, whole decoder and LM head. **No h0
or hL boundary tensor is physically sent.** Sending artificial boundary tensors
would not validate the analytical model. The boundary audit separately computes
the M9-A scenario with hidden_element_bytes=2, delta_element_bytes=4 and one
full boundary tensor each direction, and compares it with observed payloads.
The historical function's default delta width is configurable (its default is
2); R3-A's separate reconciliation uses the explicitly requested 4-byte scenario.

At N=32, H=2560, L=32, that scenario contains 5,242,880 ES->UD projection bytes
and 31,457,280 UD->ES projection bytes, plus 163,840 boundary bytes each way.
These are **modeled values**, not a claim that the OPT GPU pilot has executed.
Actual per-request totals and framing overhead must come from captured traces.
The tiny CPU test has different dimensions/hidden width and is clearly labeled.

Actual frames do not establish a remote RTT or constrained-link throughput.
The ES exchange wall includes UD deserialization, computation, serialization,
blocking and host scheduling; it is not a pure network RTT. No historical
modeled compute/payload total is compared numerically as the same endpoint.
There is no reuse comparator in R3-A, so no 344 Mbps break-even validation.

## Timing and output contract

The warm ES monotonic wall starts after pre-request CUDA synchronization,
includes coordinator creation, projection-patch setup/teardown, full decoder/LM
head, real socket exchanges and final CUDA synchronization. CUDA events cover
the forward interval and may include stream idle/host gaps. Reference inference,
logit export/parity, model/fingerprint load, CPU adapter load, input preparation,
handshake and shutdown remain outside that wall and are recorded separately.
Warmups are retained separately and excluded from latency means.

Per-request rows and per-layer traces record ES wall/CUDA intervals, UD LoRA
host processing, serialization, synchronized copies/casts, socket calls,
monotonic timestamps, identities and directional bytes/counts. These timers
are nested and use different process/device clocks; they are never subtracted
or summed to manufacture request latency. Unmeasured exclusive base/add,
allocation/dispatch and safety-check spans are explicitly listed and remain
inside the measured enclosing wall where applicable.

| Phase | Required outputs |
|---|---|
| prepare | `r3a_prepare_manifest.json`, `r3a_reference.json`, `r3a_reference_logits.bin`, `r3a_adapter_manifest.json`, `r3a_adapter.bin`, `r3a_preflight.json` |
| pilot / latency | `r3a_manifest.json`, `r3a_environment.json`, `r3a_wire_trace.jsonl`, `r3a_latency_raw.csv`, `r3a_parity.json`, `r3a_boundary_audit.json`, `r3a_decision.json` |
| analyze | `r3a_aggregate_summary.json`, `r3a_decision.json`, `r3a_evidence_audit.json` |

Additional evidence includes the complete base fingerprint, immutable plan,
raw iterations, UD-side wire trace, ES/UD results, readiness/lifecycle records,
and `r3a_{prepare,es,ud}.{stdout,stderr}.log`. Failures add concrete worker
reason/traceback JSON and retain partial traces. Unexecuted prepare phases have
FAILED placeholders rather than invented reference arrays.

All directories must be new/empty. Successful prepare is COMPLETE only after
the preparer exits; pilot is PILOT_VALIDATED only after all gates; latency and
aggregate are COMPLETE only after matching evidence passes. Failed required
evidence produces FAILED with no imputation. A candidate manifest remains
VALIDATING during export checks, never briefly claims successful validation.

Successful physical decisions explicitly contain
`classification=PHYSICAL_LOCALHOST_TCP_NO_REUSE`,
`actual_network_socket_used=true`, `actual_remote_network_measured=false`,
`cache_reuse_enabled=false`, `user_device_cuda_used=false`, `parity_passed=true`.
`gpu_inference_performed=true` belongs to actually executed OPT reference/ES
phases; CPU fixtures and the analysis process use false. Safe reuse and
correctness-qualified reuse speedup flags remain false.

## Exact launcher-compatible CLI

The example directory below must be unused. No launcher is generated here.

```bash
python -u scripts/95_run_r3a_distributed_no_reuse.py \
  --phase prepare --model facebook/opt-2.7b --dtype float16 \
  --prompt-lengths 32 --device cuda:0 --adapter user_a --seed 42 \
  --output-dir results/r3/a/example/prepare

python -u scripts/95_run_r3a_distributed_no_reuse.py \
  --phase pilot --prepared results/r3/a/example/prepare \
  --device cuda:0 --warmup-runs 1 --measured-runs 3 \
  --output-dir results/r3/a/example/pilot

python -u scripts/95_run_r3a_distributed_no_reuse.py \
  --phase latency --prepared results/r3/a/example/prepare \
  --pilot-inputs results/r3/a/example/pilot --device cuda:0 \
  --warmup-runs 3 --measured-runs 10 \
  --output-dir results/r3/a/example/latency

python -u scripts/95_run_r3a_distributed_no_reuse.py \
  --phase analyze --prepared results/r3/a/example/prepare \
  --pilot-inputs results/r3/a/example/pilot \
  --inputs results/r3/a/example/latency \
  --output-dir results/r3/a/example/aggregate
```

`python3` is equivalent where `python` is unavailable. Latency rejects absent,
failed, stale or incompatible pilot evidence before creating a listener. The
optional OPT-125M CUDA variant uses the same endpoint but is not paper-facing.

## CPU tests, executed smoke and GPU opt-in

The development environment has Python 3.10.12 and no Torch, Transformers,
PEFT or NumPy; no GPU allocation was used. Production OPT preparation/reference
and the physical OPT-2.7B pilot are **NOT_RUN, awaiting a configured SERAPH GPU
runtime/allocation**. Local model snapshots must be available there and were
not validated here. Missing dependencies produce explicit FAILED preflight
artifacts and do not launch a reference. No downloads were attempted.

The separate `r3a_tiny` helper builds an explicit TEST_FIXTURE_ONLY scalar causal
decoder with 32 layers, 4-token length, hidden width 4, vocabulary 7 and actual
nonzero rank-8 matrices. It has causal softmax attention/residual nonlinearities
and raw FP32 storage. Its mathematical CPU reference is not PEFT or OPT quality
evidence. The production prepare CLI never falls back to it. Only a prepared
artifact carrying this explicit fixture identity permits `--device cpu`.

CPU verification commands:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_r3a_*.py' -v
PYTHONPATH=src:tests python3 -m unittest -v test_r2d_evidence test_r2d_network test_r2d_analysis test_m9a_cost_model test_m9a1_base_profile test_r2c3_analysis test_r2c4_analysis test_r2c4_cache_path
python3 -m compileall -q src/semcache/edgelora scripts/95_run_r3a_distributed_no_reuse.py tests/test_r3a_protocol.py tests/test_r3a_contracts.py tests/test_r3a_processes.py
python3 scripts/95_run_r3a_distributed_no_reuse.py --help
```

The sandbox blocks even loopback socket creation. CPU protocol/integration
tests and standalone TCP smokes therefore require the environment's approval
for execution outside that sandbox. They still bind only 127.0.0.1 and load no
GPU/model libraries. The tests cover framing/digests, payload/wire accounting,
corruption/truncation/oversize, absolute timeout, wrong version/session/request/
layer/adapter/dtype/shape, stale hidden input, independent PID/CPU isolation,
exact observed counts, tamper rejection and cleanup. A negative parity control
stops after the first warmup and leaves no listeners or child processes.

For a new CPU fixture tree, run the helper and the same three runtime phases
with `--device cpu` for pilot/latency:

```bash
PYTHONPATH=src python3 -m semcache.edgelora.r3a_tiny --output-dir /tmp/r3a_TEST_FIXTURE_ONLY/prepare
python3 -u scripts/95_run_r3a_distributed_no_reuse.py --phase pilot --prepared /tmp/r3a_TEST_FIXTURE_ONLY/prepare --device cpu --warmup-runs 1 --measured-runs 3 --output-dir /tmp/r3a_TEST_FIXTURE_ONLY/pilot
python3 -u scripts/95_run_r3a_distributed_no_reuse.py --phase latency --prepared /tmp/r3a_TEST_FIXTURE_ONLY/prepare --pilot-inputs /tmp/r3a_TEST_FIXTURE_ONLY/pilot --device cpu --warmup-runs 3 --measured-runs 10 --output-dir /tmp/r3a_TEST_FIXTURE_ONLY/latency
python3 -u scripts/95_run_r3a_distributed_no_reuse.py --phase analyze --prepared /tmp/r3a_TEST_FIXTURE_ONLY/prepare --pilot-inputs /tmp/r3a_TEST_FIXTURE_ONLY/pilot --inputs /tmp/r3a_TEST_FIXTURE_ONLY/latency --output-dir /tmp/r3a_TEST_FIXTURE_ONLY/aggregate
```

The GPU-only test is skipped by default. Only on an available allocation with
the local model/runtime configured, set `SEMCACHE_R3A_GPU_PILOT=1` and
`SEMCACHE_R3A_GPU_OUTPUT` to a new directory, then run
`test_r3a_processes.AllocatedGPUIntegration`. It preserves failure outputs and
never converts parity failure into success. It has not been executed locally.

R3-B prerequisites are a successful actual OPT-2.7B physical no-reuse pilot,
matching warm baseline evidence, explicit resident-cache ownership/traffic,
real HIT/MISS traces and independent strict correctness gates. R3-C adds
controlled bandwidth/RTT conditions for network sensitivity. Remote hardware
equivalence and production E2E remain separate evidence requirements.
No prior oracle benefit is a deployable optimization.

Final development verification on 2026-10-29 ran 98 tests: **96 passed, 0
failed, 2 skipped**. The new suite contributed 21 passing CPU tests and one
skipped opt-in GPU pilot; the other skip was the existing optional Torch tensor
test. Compilation, CLI help and whitespace checks passed. The retained
`/tmp/r3a_verified_TEST_FIXTURE_ONLY.9VLIdT` smoke completed tiny preparation,
pilot (1 warmup + 3 measured), latency (3 warmup + 10 measured) and analysis.
Each observed forward had 32 layers, 96 QKV outputs, 32 HIDDEN frames and 32
bundled DELTAS replies. Pilot/latency captured 260/836 total frames respectively,
including four out-of-request control frames in each session. This is CPU tiny
model/socket evidence, not an executed OPT/PEFT GPU result. Test stdout/stderr
and per-worker logs are retained in that tree.
