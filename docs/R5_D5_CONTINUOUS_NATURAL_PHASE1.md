# D5 Continuous NATURAL — Phase 1

CPU implementation and fixture validation: **PASS**. Real-data validation:
**BLOCKED**, pending execution on SERAPH against the existing sealed artifacts.
Overall original-data Phase 1 acceptance is pending; no real-data PASS is claimed.
GPU physical HITs, measured policy outcomes, attention Impact, GPU tensor
equivalence, and E2E latency are **NOT_MEASURED**.

The local workspace lacks the original
`/data/khuss/repos/GP_semcache/results/r5/i5/plan_longbench3_smoke_v1` and its
expanded preparation. Available local I5 plans are fixtures or partial diagnostic
manifests. The user confirmed the authoritative inputs remain on SERAPH. They
have not been recreated, substituted, edited, or resealed. Torch, Transformers,
and NumPy are also absent locally; native model/real TinyBERT validation has not
run. The new fixture tests require none of them.

## Implementation and lifecycle

New files only:

- `src/semcache/edgelora/r5_continuous_natural.py`: sealed-input session construction,
  invocation mapping, dataset lifecycle, CPU probes, static counts, resource diagnostics.
- `scripts/154_r5_continuous_natural.py`: explicit-path `cpu-validate` command;
  no GPU runner, worker, Slurm action, or sweep.
- `tests/test_r5_continuous_natural.py`: 17 CPU tests with explicitly labeled fixtures.

The scheduler initializes once for each dataset/condition, dispatches its ordered
requests through persistent per-adapter state, checkpoints at each original block
end, and closes in `finally`. It creates three logical dataset sessions and six
isolated condition executions. The schedule is A then C per dataset; historical
block-level condition counterbalancing does not establish a continuous-session
performance comparison. A/C receive the same source invocation order.

The CPU probe constructs actual empty `PhysicalOwner` and `SemanticIndex` objects
for C. A has neither. It retains references to the owner, cache entries, policy
objects, metric/history manager, integrity state, semantic clusterer/index, and
channel/model handles across checkpoints. It checks that these roots are not
reconstructed. Test-only arrival accounting exercises actual cache clocks and
frequency histories through all four blocks and repeated samples, without
admitting entries or fabricating attention. Other tests verify new state at
dataset/condition boundaries and cleanup on failure.

Probe model and channel handles are explicit metadata-only placeholders. No model
loads, socket connections, encoding, `owner.begin`, admission, CHU update, or reuse
occurs in this probe. Its six initialization calls establish CPU orchestration,
not six measured model initializations. This is an implemented lifecycle design,
not a claim that the existing GPU runner now maintains a continuous physical
cache. The existing one-block GPU runner remains unchanged and still disposes
its owners at block end. Concatenating that runner's outputs is not continuity.

No SemCache arithmetic or policies changed: TinyBERT revision/CPU encoding,
w=3 extraction, one semantic cluster, F/A/I/S, CHU, PBR interval 100 per adapter,
64 entries/candidates, 512 MiB per adapter, sparse total-QKV projection reuse,
completion ACK barrier, and existing sharing/integrity rules remain as implemented.
The plan records `REQUEST_SCOPED_VERIFIED`, `EXACT_CONTEXT_VERIFIED`,
`allow_approximate=False`, and the frozen numerical thresholds. Static candidates
never become grants. Native target-specific autoregressive past KV must still
start fresh for each invocation; it is distinct from persistent SemCache entries.

## Dataset and lifetime summary

These are the required design counts, verified with fixtures; original-data
counts and routing distributions remain BLOCKED until SERAPH validation.

| Dataset | Original blocks | Invocations | Original distinct samples | Evaluations per sample | Initializations per condition | Internal resets |
|---|---|---:|---:|---:|---:|---:|
| Qasper | 0–3 | 20 | 10 | 2 | 1 | 0 |
| 2WikiMQA | 4–7 | 20 | 10 | 2 | 1 | 0 |
| MultiNews | 8–11 | 20 | 10 | 2 | 1 | 0 |

There are 60 logical source invocations, 120 paired A/C dispatches, and 24 block
checkpoints across the six condition executions. No source, warmup, priming,
deduplication, additional repetition, or forced HIT is introduced. A dataset's
cache lifetime is all 20 requests, with adapter-specific clocks advancing only
for that adapter's actual future requests. Even 20 requests to one adapter cannot
reach the unchanged PBR interval of 100. Observed PBR/CHU events remain NOT_MEASURED.

## Request identity and provenance

`r5_i5_workload.requests()` derives `request_id=P{offset+i:06d}T` and
`pair_id=P{offset+i:06d}`. `prepare_experiment()` restarts the cohort offset at
zero for every repetition; `request_sequence_number` restarts in every block.
Thus repetition blocks reuse IDs. `MStageShadow.seen`, sparse request ordering,
ledger paths, result directories, and per-request numerical joins cannot use
these IDs unchanged in a continuous session.

The session ID deterministically hashes the sealed experiment ID and dataset.
Each envelope stores a session sequence number 0–19, unique invocation ID,
invocation pair ID, original block ID/index/repetition, original request/pair/
sequence/sample IDs, full original task, and its digest. Runtime request IDs append
`_A` or `_C` to the invocation ID, fit the existing 60-character request schema,
and are unique across all 120 dispatches. Matched A/C evidence joins use the
condition-independent invocation pair ID. The strict request schema is unchanged;
provenance lives in the sealed sidecar, not extra request-schema fields.

Only runtime request ID, pair ID, and sequence number are remapped in the derived
view. Every token ID, prompt, prepared record, generation policy, user, adapter,
sample ID, and request order is preserved. Ten same-sample pairs per dataset are
experiment-induced repeated evaluations, not independent scientific samples.
Naturally identical content in distinct original samples is counted separately.

## Static opportunities and denominators

Original-data counts for every category below are **BLOCKED**, not zero:

| Dataset | Exact experiment-repeat pairs | Naturally identical distinct-sample pairs | Shared-prefix windows | Same-position non-prefix windows | Shifted-position non-prefix windows | Sharing-eligible counts |
|---|---|---|---|---|---|---|
| Qasper | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| 2WikiMQA | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| MultiNews | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |

The command analyzes all 190 ordered earlier-source/later-target pairs per
dataset, including cross-block pairs. For each equal w3 occurrence at source
position `sp` and target position `tp`, four disjoint counters cover all matches:

1. Shared-prefix: `sp == tp` and `tp + 3 <= longest_common_prefix`.
2. Same-position non-prefix: `sp == tp` and `tp + 3 > longest_common_prefix`.
3. Shifted-position non-prefix: `sp != tp` and `tp + 3 > longest_common_prefix`.
4. Shifted-prefix-target: `sp != tp` and the target window lies in the common prefix.

The fourth category prevents shifted occurrences inside repeated prefix tokens
from inflating non-prefix counts. Counts are raw **occurrence pairs**, including
overlaps; they are not disjoint token coverage or candidate-limit predictions.
Separate target-position unions count each `(target invocation, window start)`
once. Each category is also counted after actual registry sharing eligibility,
with pair-level cross-adapter and cross-user rejection reasons. Sharing requires
the same adapter and, for different users, the same explicitly permitted group.

Exact request content means identical token IDs, prompt, and generation policy;
user/adapter sharing eligibility is reported separately. Same-sample repetitions,
distinct-sample identical-content invocation pairs, distinct original sample
pairs with identical content, and token-only identical pairs have separate counts.
The distinct-original-sample count avoids multiplying natural identity by the
experimental repetitions. Original raw sample records remain attached.

Same-position common causal prefixes are only static necessary EXACT conditions.
Same-position non-prefix and shifted windows fail existing EXACT context/position
rules. Semantic assignment, resident-key ownership, actual Impact, F/A/I/S
admission/eviction, candidate selection, per-layer input bytes, integrity proofs,
and actual tensor consumption are not established by any of these counts.

## Resource and safety assessment

The original I5 preparation packs at most 20 targets per independent block.
`check_block()` and the D5 worker/approval/auditor are bound to one original block;
the new dataset plan must not be passed to those interfaces as a forged block.
The aggregate Phase 1 count is exactly 20 per dataset, so the numeric request cap
alone need not rise for a later Phase 1-sized GPU pilot. Scope/lifetime and strict
evidence are the immediate blockers.

The following **analytical upper envelopes** use the existing resource formula,
20 requests, 512 input tokens each, and the indicated generation cap. They are not
real workload lengths/routing, timing measurements, or synthetic experiment runs.

| Quantity | Qasper / 2Wiki cap 32 | MultiNews cap 96 |
|---|---:|---:|
| Maximum forwards per condition | 640 | 1,920 |
| Existing condition evidence forecast | 50.33 GiB | 59.86 GiB |
| Paired A/C evidence forecast | 100.65 GiB | 119.73 GiB |
| Forecast for an adapter handling 8 requests | 20.13 GiB | 23.95 GiB |
| Dense UD payload alone for 8 requests | 4.64 GiB | 5.19 GiB |
| Existing run file forecast | 583,168 | 1,740,288 |
| Existing conservative GPU forecast | 11.10 GiB | 11.12 GiB |

The forecasts exceed existing 16 GiB per-adapter ES, 4 GiB per-adapter UD, 32 GiB
per-condition, 64 GiB per-run, and 500,000-file bounds. The UD payload lower bound
is `(input + generation - 1) * 32 * 2560 * (2 + 3*4)` bytes per request (one FP16
hidden plus three FP32 deltas); metadata/ledgers add more. The ES planning formula
remains conservative and includes metadata/evidence reserves. Actual sealed
lengths, generation caps and routing must be assessed on SERAPH; no real dataset
is declared feasible or infeasible from these envelopes alone.

The resident cache is bounded independently: a three-token, 32-layer FP16 total
QKV entry is 1,474,560 bytes; 64 entries occupy 90 MiB per adapter, at most 270 MiB
across three full owners, within the existing 512 MiB per-adapter capacity. The
GPU planning formula conservatively reserves 1.5 GiB for caches. Admission can
temporarily materialize entries; existing actual memory checks still apply.
At input 512, one FP16 attention layer is 16 MiB and the FP64 Impact working
estimate is 80.375 MiB, below existing 128 MiB probability/1 GiB working bounds.
No all-layer attention retention is introduced. Peak allocated GPU memory must
remain below 22 GiB; host available memory must remain at least 32 GiB.

Existing free-disk preflight is 96 GiB (64 GiB run plus 32 GiB reserve); it cannot
cover the above paired worst cases. Each worker deadline is 7,200 seconds, paired
run guard 14,460 seconds, and allocation 4h30. Completion time is NOT_MEASURED;
no runtime sufficiency is asserted. Per-JSON 16 MiB, per-request reconstructed
metadata/ledger 256 MiB, and tensor-reader 128 MiB limits also remain in force.

Minimum future GPU work, **documented only**:

1. Add a separately reviewed dataset-session plan/approval/worker route. Initialize
   one real model and per-adapter UD/channel/owner/index per condition/dataset;
   dispatch all four original blocks without shutdown. Checkpoint without
   disposing state. Close completely at condition/dataset end or failure.
2. Bind invocation/provenance sidecars and original sealed inventories in worker,
   UD request allowlists, wire/request ledgers and independent numerical auditor.
   Recheck headers with longer IDs, unchanged framing and the 8 KiB reserve.
3. Forecast evidence from actual assets, including cumulative ES/UD bytes and
   files across checkpoints. If current bounds fail, stop. A new reviewed bounded
   evidence layout with lossless packing/sharding and/or explicitly justified
   resource budgets is required; do not drop strict tensor evidence or silently
   rotate/reset accounting. Update readers/inventories/free-disk preflight together.
4. Measure bounded pilot throughput only in an authorized later phase before
   choosing any new worker/job deadlines. Preserve failure evidence, GPU identity,
   offline model/adapter checks, actual memory guards, and independent EXACT replay.
   No request truncation, generation-budget reduction, or approximation is a fix.

No GPU runtime, job script, policy implementation, or historical result changed.

## Tests and operator command

Local final verification: **17 new tests passed; 11 selected existing D5 scalar
regression tests passed; 0 failures, 0 skips** in the selected suites. The existing
regressions cover same-length NATURAL matching, causal/adapter rejection, CHU and
resident cleanup, per-layer input rejection, and pre/post-authorization corruption,
replacement, metadata/registry tampering, disposal and abort. Existing fixture
Impact values remain test fixtures, not original-workload measurements. No native
Torch test, GPU command, Slurm command/submission, or git push ran.

New unit suite:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests \
  python3 -m unittest test_r5_continuous_natural -v
```

On SERAPH, use the existing offline environment and source checkout containing
these new files. Set `D5N_MULTI` to the existing `multi` path recorded in the prior
`natural_plan.json`; no new preparation is requested. Use a fresh output path:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/154_r5_continuous_natural.py cpu-validate \
  --original-experiment /data/khuss/repos/GP_semcache/results/r5/i5/plan_longbench3_smoke_v1 \
  --multi "$D5N_MULTI" \
  --output-dir results/r5/i5r/d5_natural/continuous_phase1_seraph_cpu_01
```

This verifies original SHA256 inventories, model/tokenizer/prepared dataset
provenance and routing using the existing strict loader; constructs sessions;
checks CPU lifecycle and longer-ID transport envelopes; writes per-dataset
`session.json`, `summary.json`, `static_opportunities.json`, root
`phase1_validation.json` and `phase1_report.md`, then seals/verifies the output.
It does not call the previous one-block GPU approval or runner. An ordinary CPU
process is sufficient. Existing offline Torch/Transformers/NumPy/tokenizer assets
are needed by that loader; no downloads are attempted.

Missing assets produce a sealed BLOCKED record and exit 2. Invalid content fails
closed. A CPU PASS with resource diagnostic FAIL means the static session plan
validated, while future GPU readiness remains false. No output overwrites an
existing directory. Do not use this CPU report as a GPU approval receipt.

## Phase 2 plan only — not implemented

After original-data Phase 1 passes, prepare a new separately sealed workload with
at least 50 **distinct eligible original samples** per dataset (at least 50
invocations; if retaining two repetitions, 100 invocations). Preserve historical
artifacts, prompt/tokenizer/generation/routing rules and predetermined order.
Report eligibility shortages rather than fill by duplication. Generalize the
new session schema's fixed 12-block/20-request checks with explicit reviewed
dataset/session caps; retain block checkpoints and invocation provenance. Repeat
CPU lifecycle, sharing, static matching, header, evidence, memory and deadline
assessments at that declared size. PBR remains 100 **per adapter**, so even 50 total
requests do not imply a PBR event. Phase 3 GPU pilots require separate authorization
and the runtime/evidence work above. Neither phase has started.
