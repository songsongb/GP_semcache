# D5 L1–L5: audited design and implementation boundary

Initial audit: 2026-11-13, `/home/songsong/GP_semcache_lightperf`, branch
`exp/d5-l1-l5-lightperf`, clean HEAD
`4916836ce328e119f3a8d85c6c1afc7cf8d363e5`. This is the requested Strict
commit itself. No remote lookup, branch operation, Slurm query or GPU operation
was performed. The other registered worktree is `/home/songsong/SemCache` at
`55587ba…`; it is read-only for this task. SERAPH `/data/khuss/repos/GP_semcache`
does not exist on this host. No applicable AGENTS.md was found.

## Actual call graph

`158_r5_natural_phase3a` → `r5_natural_phase3a.{material,Profile}` →
`r5_natural_approx.{prepare,cpu_validate,run}` →
`r5_subsequence_natural_runtime.worker` → `setup` (native OPT), one isolated
UD process and `PhysicalOwner`/`SemanticIndex` per adapter → original ordered
tasks → `r5_subsequence_generation.generate` → `_generate`:

* `SemanticIndex.assign` → `PhysicalOwner.begin` → `MStageShadow.begin` and
  w=3 matching/authorization/exact position and causal prefix filters.
* Full native prefill with `projection_path` → `PhysicalOwner.project` →
  per-layer input proof, `RequestValidation.verify/consume`, sparse fresh row
  projection plus authenticated private cached QKV scatter. Native attention,
  FFN and LM head still execute on the complete target. Cache QKV is not past KV.
* `SparseSession.correction` → `r5_d5_channel.Channel` → frozen
  `communication.Channel` encode/send/recv → isolated UD `serve_sparse` →
  `SparseOrder.accept`/registry checks → `cpu_deltas` (same A/B matmuls for QKV)
  → DELTAS or READY for zero fresh rows. Both endpoints verify framing, sequence,
  session, adapter, shapes and hashes. Completion HELLO/READY authenticates
  generated history; only the final adapter ACK enables STOP/BYE.
* `StreamingImpact.hooks` reduces actual layer attention in FP64 using original
  `actual_attention_impact`. These values update F/A/I/S history, CHU, admission,
  eviction and PBR in `PhysicalOwner.complete`; they cannot be replaced by zero.
* Greedy `generate_loop`: full-vocabulary host values, finite check, first-index
  tie breaking, termination, target's own native past KV for decode. Strict
  packs every full vocabulary into blobs and synchronizes each token and completion.
* Runtime computes task quality, persists request/step/timing/wire/policy files,
  checks budgets by recursive scans, disposes owners/processes and seals outputs.

## Cost classification

| Class | Actual code / role | Light treatment |
|---|---|---|
| Necessary calculation | Native full prefill/decode, sparse QKV, UD LoRA, semantic assignment, matching, actual attention impact, F/A/I/S and CHU/PBR | Preserve original primitives |
| Strict independent observation | `Evidence.save/blob`, probability/QKV/base/delta/logit dumps, per-frame ledgers, `save_result` step shards, repeated `directory_bytes` and runtime `rglob` | Separate bounded summary sink; no raw tensors or full logits |
| Safety / required owned state | `tensor_proof`, `entry_proof`, protocol full-input hash, per-layer equality, registry/owner checks, captured QKV/hidden clones, private consumed snapshots, admission proof | Preserve; host transfer/hash cost remains visible |

`r5_subsequence_integrity` protects against alias and `.data` mutation without
trusting tensor version counters. Scoped validation pins storage, authenticates
each selected entry and clones/verifies consumed tensors. Skipping these is not
an evidence optimization. Strict source identity requires its original branch;
it stays unchanged. Light uses a separate root/branch/HEAD/content fingerprint
and a fresh clean committed source approval, never a historical Strict CPU audit.

`r5_subsequence_profile` has nested inclusive and observed exclusive wall scopes,
optional bounded CUDA events, and layout details. It does not measure kernel-only
time. `r5_perf_timing`/`r5_perf_analysis` concern earlier prefix experiments and
can measure observer/scanning cost and exclusive forward scopes; they are not a
drop-in full-vocabulary D5 Light auditor. Inclusive components must not be summed.

## Workload and resource contract

I5 `select` ranks groups by SHA256(seed,group), orders turns, and `pack_groups`
packs at most 20 tasks. Each original block has an independent cold cache identity.
Phase3A pins a first repetition-0 five-task block; Qasper uses the approximate pilot.
Light accepts a single original NATURAL block or fails closed. A 20/50 stream
crossing original cold blocks requires a separately reviewed continuous-session
selection contract; copying cold blocks or changing their sequence numbers is
not accepted as proof of continuous identity. No repeated samples, fabricated
overlap, rerouting or truncation is allowed. Prefix checks include whole groups.

Resource forecasts are upper bounds with an explicitly unmeasured 5 s/forward
assumption, per-adapter 512 MiB cache, original input ≤512/context ≤2048/output
≤96 limits, 22 GiB allocation ceiling, ≥12 GiB free after load, ≥32 GiB host
available, ≥4 CPUs, and a manual job deadline. Network bound counts framing and
control allowances in both directions; it does not assume HITs. Failed estimates
produce BLOCKED_INFEASIBLE without increasing limits. A plan is not a measurement.

## Measurement / emulation semantics

Light summaries distinguish application arrival→token availability/completion,
model prefill/decode, admission, persistence and quality. Initialization, final
sealing and warmup scope are explicit. Missing measures are null. Nested component
times retain inclusive/exclusive labels; unobserved time stays unassigned. Light
without full logits reports independent full-vocabulary audit NOT_EVALUATED and
correctness_qualified_speedup NOT_ESTABLISHED.

Link pacing is per sending channel and direction, with monotonic byte scheduling
at bounded socket chunks. It includes the actual encoded frame bytes, applies
identically to FULL/EXACT/APPROX, never changes payload/framing/ACK, never spins,
and keeps the original frame deadline. This is application-level loopback shaping,
not a geographic link or TCP/IP packet-overhead model. RTT is separate and is not
implemented. Multiple adapter channels are explicitly not one shared link.
Even unlimited Light uses the same ≤64 KiB send chunks, introducing Python/socket
call overhead relative to Strict's unbounded send view. Original frame hashes
and protocol serialization remain. Light drops retained per-frame Channel ledgers
after returning each receipt. The native kernel retains host full-vocabulary
transfer/finite checks/greedy selection but omits packing/persisting full logits
and explicit per-token/completion CUDA synchronize. Host logits availability and
proof transfers still synchronize dependencies. This timing change is not yet
native CPU/GPU verified and must be tested before launch approval.

UD service emulation runs the same arithmetic once, measures each operation's
wall cost and paces the total to cost/factor with deadline checks. Requested
1/.5/.25 factors are observable measured service ratios, not CPU-frequency or
hardware equivalence. No sleep is applied outside the supplied LoRA operation.
L5 exports only a CPU matrix; no Cartesian campaign submission exists. Analysis
requires matching identities/knobs/device and authenticated Light receipts;
APPROX remains separate and crossings require measured sign changes.

## Current approval boundary

This host has no torch, transformers, numpy, real prepared dataset/model/adapter
manifests or frozen GPU receipts. Pure stdlib fixtures can test contracts and
protocol behavior, but cannot approve native OPT fidelity, scorer availability,
physical GPU resources, actual sample lists or 50-request feasibility. The CLI
run gate remains fail-closed until a native Light worker is implemented and
validated. See the runbook and CPU report for exact implemented/blocked scope.
