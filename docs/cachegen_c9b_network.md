# C9-B network / storage-codec break-even

C9-B is CPU-only. It combines **MEASURED** C9-A per-target prompt latency with
exact raw LoRA-delta bytes and **ANALYTICAL** payload transfer time. It never
loads a model, executes a codec, refits profiles, reruns quality or GPU latency,
enables transport compression, or freezes a system policy. Frozen32 remains a
selection-based controlled trace.

Inputs are the completed SELECT_Q24 freeze, completed C8-A capacity trace,
supporting C8-B HIT audit, and completed C9-A READY_FOR_NETWORK_ACCOUNTING run.
The script revalidates their hash chains and exact B2/B8 HIT vectors without
calling the model/quality preparation path. C9-A raw repetitions must reproduce
the saved per-target medians, ALL/HIT/MISS summaries, paired deltas, and runtime
counters. Local primitive values are per-target medians, never condition means.
Existing artifacts stay read-only and a nonempty output root is refused.

The seven conditions remain FULL_RECOMPUTE and B2/B8 crossed with RAW_QKV,
KV_COMP, Q24_KV_COMP. HIT means exact w=3 token IDs within the semantic cluster;
cached TOTAL Q/K/V share the same mask. HIT removes three fresh projection rows
at every layer/role. MISS and FULL have all prompt rows fresh. Prompt lengths
come from the hash-bound semantic workload and must match every measured
C9-A request.

The communication object is **TRANSPORT_LORA_QKV_DELTA**, not raw or compressed
resident TOTAL-QKV. C6-B3-2 accounts each role as:

```text
fresh_rows * module.out_features * module.lora_B[user].weight.element_size()
```

The auditor extracts and checks that exact C6 expression, its fresh-row formula,
and the pure `traffic_summary` function without importing C6's CUDA backend.
It records source symbols and hashes for the transport callback, mixed path,
projection decomposition, and adapter loader. Frozen safetensors headers are
read without loading weight tensors. All32 layers and three roles must have
rank8 A/B matrices with width2560 and FP32 weights. This matches the training
path's saved FP32 adapters and the decomposition's cast to LoRA-A dtype. TOTAL
projection output is FP16, a different object. Unproven adapter dtypes/shapes
fail closed rather than borrowing resident Q's element size.

For the frozen adapters, each fresh row transports:

```text
per role: 32 layers *2560 outputs *4 bytes =327,680 bytes
Q/K/V:   983,040 bytes
one w3 HIT avoids: 2,949,120 bytes
```

This is raw tensor payload accounting, not a physically measured wire packet.
Resident frame metadata/CDF/profile bytes are never substituted for it. Actual
transaction batching, control RTT, and protocol headers are not established by
local callbacks; transaction/RTT counts are null. Logical projection payload
counts are reported separately. There is no assumed numerical RTT.

C6's native FULL path (`hits=None`) does not increment its raw traffic counter.
Its saved zero cannot establish zero EdgeLoRA communication. C9-B derives FULL
and MISS from the same all-fresh projection-delta tensor contract. It labels
this distinction explicitly. If a canonical C6-B3-2 manifest is in C9-A's bound
inputs, its RAW_SEMCACHE and STORAGE_KV_COMP teacher-forward raw byte counters
must match reconstructed `prompt + canonical[:-1]` shapes exactly. Source+greedy
task totals have a different scope and are not compared to a prompt-only request.
No arbitrary artifact directory discovery is used; unavailable cross-check
evidence is recorded explicitly, and contradictory available evidence fails.

The network model is **PAYLOAD_TRANSFER_ONLY**, in decimal Mbps:

```text
transfer_ms = bytes *.008 / bandwidth_Mbps
analytical_e2e_ms = measured_target_total_ms + transfer_ms
```

It evaluates exactly 10/50/100/500/1000Mbps, computing each target before
aggregating mean/p50/p95/population standard deviation. All reference/policy
pairs have paired local, byte, transport and E2E deltas and faster/slower/tied
counts using the existing measured clock tie tolerance. No distributed latency
is claimed.

Exact bandwidth break-even uses aggregate mean quantities and per-target roots:

```text
delta = local_A - local_B + .008*(bytes_A-bytes_B)/BW
BW_break_even = -.008*delta_bytes/delta_local_ms
```

A locally slower, lower-payload policy wins below its positive root. The reverse
tradeoff wins above its root. Dominance, no payload difference, equality and
undefined roots are explicit cases, not fabricated zero/infinite bandwidths.
Root distributions retain per-target reasons and valid-root counts.

**HYPOTHETICAL_ACCELERATION** changes only decode:

```text
nondecode_A = target_total_A - storage_decode_A
accelerated_local_A = nondecode_A + storage_decode_A/S
```

The visualization/table factors are exactly 1/10/50/100/500/1000. Exact minimum
factors are solved independently at every bandwidth. For FULL/RAW comparators,
the comparator stays unchanged. One-sided Q24 acceleration versus current KV
is secondary sensitivity. The primary Q24-vs-KV native-backend comparison applies
the same S to both decode paths:

```text
delta(S) = nondecode_Q24 - nondecode_KV + delta_network
           + (decode_Q24-decode_KV)/S
```

`NO_FINITE_CODEC_SPEEDUP_CAN_BREAK_EVEN` identifies an unbeatable non-decode/
network floor. Negative decode differences can make acceleration erode an
existing relative advantage; that case reports the maximum winning factor too.
These scenarios do not change HIT coverage, lookup/model timing, or payloads.

Source preparation/encode cost remains secondary. For each of KV-minus-RAW,
Q24-minus-RAW and Q24-minus-KV, incremental mean encode cost per admission is
divided by N=1/8/32. Source forward/capture/admission are not folded into the
primary steady-state result. Shared source encoding across budgets is not summed.

The descriptive classification is declared before reading measurements:
per compressed policy/budget, SYSTEM_COMPETITIVE means mean current-codec
payload-only latency matches/beats all relevant comparators at every tested
point; NETWORK_CONDITIONAL means it does at some tested points; otherwise
NOT_LATENCY_COMPETITIVE. Stage SYSTEM requires all four compressed contexts;
stage CONDITIONAL requires at least one competitive context/point. Break-even
regions outside the tested range are separately reported. The native-backend
flag means a finite factor above1 is needed in at least one tested primary
comparison, not that optimization alone can repair every floor or that only
native code could supply the speedup. No arbitrary paper-quality threshold or
final policy decision is introduced.

Run without GPU allocation from `/data/khuss/repos/GP_semcache`:

```bash
CUDA_VISIBLE_DEVICES="" python -u scripts/68_run_cachegen_c9b_network_break_even.py \
  --plan-dir results/cachegen/c6b3/multiwoz_plan_v2 \
  --freeze-decision results/cachegen/c7/q24_freeze/freeze_decision.json \
  --c8a-root results/cachegen/c8/a_capacity_replay \
  --c8b-root results/cachegen/c8/b_quality_b2_b8 \
  --c9a-root results/cachegen/c9/a_local_latency_retry1 \
  --output-root results/cachegen/c9/b_network_break_even
```

Outputs: transport_bytes_per_case.csv, transport_byte_summary.json,
bandwidth_sweep.csv, bandwidth_summary.json, network_break_even.json,
codec_speedup_sweep.csv, codec_break_even.json, source_amortization.json,
pairwise_system_comparisons.json, manifest.json, summary.md. Every estimate and
its evidence source is labeled, input/output hashes are recorded, and no GPU,
shell/Slurm script, network experiment, or new codec is involved.
