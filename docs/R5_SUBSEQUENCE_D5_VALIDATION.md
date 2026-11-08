# D5: request-scoped validation with authenticated consumed snapshots

GPU_PERFORMANCE=NOT_RUN. The job154175 timings are an operator-verified handoff;
the raw run is unavailable locally. This change does not address numerical
fidelity, and it does not relabel the historical numerical FAIL.

The local checked-out commit is8131800, on exp/raw-semcache-r5-integration.
The requested reviewed5869c9a0ec40b8e0f136b6280f125bba0858c67b object is not
available locally. No checkout/reset/fetch was performed. Review the patch before
deployment; prepare only from the new reviewed, clean committed source. Neither
that reviewed SHA nor the historical a6fae7d SHA is the new implementation SHA.

## Lifecycle and safety review

1. `PhysicalOwner.complete()` materializes independent compact all-layer QKV
   copies through `CacheEntry.from_tensors`, using unchanged F/A/I/S insertion
   and victim rules. It creates the original admission-time `entry_proof` from
   the stored tensors and qkv_metadata/positions/token_ids/size. The digest format
   and its admission semantics are unchanged. Scoped mode additionally pins the
   admitted entry, tensor objects/containers and storage identities, so even an
   identical-byte replacement before the first request verification is rejected.
   These lifetime bindings are deleted through the normal eviction/disposal path;
   they do not freeze policy F/I/access-time updates or replace the byte proof.
2. TinyBERT cluster assignment, exact w3 lookup, authorization, non-overlap and
   exact-context filtering remain unchanged in `begin`. Entries and tensors are
   mutable Python/PyTorch objects, not immutable storage. Public `.cache.entries`,
   tensor aliases, `.data` and views can mutate them. `_version` is not a complete
   detector and is not consulted by the runtime implementation.
3. `EAGER_FULL_PROOF` remains the default. The original q-eligibility call once
   per selected window/layer and the repeated call before every consumed Q/K/V
   still recompute the complete admission digest. For21 windows/32 layers and
   consumption only at layer0:735 full checks. No policy/arithmetic path changes.
4. Explicit `REQUEST_SCOPED_VERIFIED` lazily verifies each selected resident
   entry once, at its first layer eligibility check **inside Prefill**. It checks
   the actual stored values against the admission digest. That same pass exports
   the individual tensor fingerprints; it does not hash them a second time to
   construct the authorization snapshot. The record binds request SHA/epoch,
   cache key, actual resident entry reference, original proof, metadata digest,
   metadata/tensor-container references, policy state, all tensor object references,
   shapes/dtypes/devices/strides/storage offsets/storage pointers and sizes.
   Strong storage references prevent old allocation addresses being recycled
   during a live record. Tensor object references do not replace byte hashing.
5. Every subsequent check rechecks resident identity, proof binding, request and
   registry identity, metadata, policy state and tensor structure. A cache-key
   match, object identity or version counter alone never authorizes reuse.
   EXACT_CONTEXT_VERIFIED's original per-layer input proof still applies.
6. Immediately before consumption, the selected role tensor is copied into a
   private detached snapshot. The actual bytes of that snapshot are SHA256-checked
   against its authenticated fingerprint. Metadata/structure are rechecked, and
   only that private snapshot is supplied to the unchanged ordered scatter.
   No alias of the cache is supplied to scatter in scoped mode. This adds real
   GPU copy cost and temporary memory; both remain within measured request scope.
   The snapshot's values and provenance are saved for independent replay.
7. Verification maps and live references expire before CHU/PBR/insertion/eviction
   at completion. Frequency/impact/access-time updates remain legitimate then;
   those fields are not part of the original admission digest. During an active
   scoped request their saved state must remain unchanged. Eviction/replacement
   is checked against the resident object before every eligibility/consumption.
   Arbitrary direct map removal is not intercepted, but it cannot use an old grant.
8. Scoped projection failures, generation failures and explicit `abort()` close
   verification state and pending request state. `dispose()` also clears cache
   entries and original proofs. New requests have fresh epochs and full checks.
   Persisted JSON authorizations contain hashes/metadata, no live Tensor handles.

### Threat model and limitation

Supported: one owner thread, ordered native CUDA stream, trusted controller/checker
and admission-proof ledger. Ordinary in-place, `.data`, detached/view alias writes,
replacement and metadata changes before consumption are checked. The snapshot
prevents later writes through the original cache alias changing the values sent
to scatter. No arbitrary concurrent CPU/GPU writers, hostile extensions, arbitrary
process-memory writes or other-stream mutation safety is claimed. Such callers
need actual exclusion/ownership enforcement beyond this patch. A foreign-thread
validation API call fails; this is not protection against arbitrary memory writes.

**Unconsumed tensor content is not continuously rehashed after authorization.**
A later-layer content change is detected before that layer's consumption; if all
its windows are rejected, it is detected at a subsequent request's full check.
Eager may detect that unused corruption sooner. Do not claim unrestricted parity
with eager global corruption detection. Structural replacements are checked even
for unconsumed tensors. The safety classification is LIMITED to this explicit
threat model and available CPU evidence, pending native/GPU validation.

## Metrics and independent audit

Physical results and profiling summaries record validation_mode,
request_authorized_entries, full_entry_proof_count/wall_ms/serialized_bytes,
consumed_tensor_proof_count/wall_ms/serialized_bytes, snapshot clone time,
metadata/structure time, integrity failures and request-wide validation wall time.
Admission proof count/time/bytes are separate and remain included in application
E2E. `request_validation_wall_ms` adds disjoint top-level scopes: owner-valid,
consumed-snapshot operations, request setup and scope cleanup. Full-proof, metadata and tensor-proof timings
are nested children, not additional additive costs. Existing per-layer eligibility
and consumed/fresh-row receipts remain available. Setup, Prefill and all request
phases remain inside application TTFT/E2E; proof work is not moved outside them.

For the observed21-entry/layer0-only workload, the expectation to verify is21 full
checks plus63 consumed checks, rather than735 full checks. This is a count
prediction, not a measured speedup. Original tensor-proof/evidence hashing still
costs time. Per-role private copies occupy about315KiB at63 FP16 token rows with
hidden2560; record actual peak GPU allocation and allocator reservation. Existing
32GiB disk reserve,8GiB ES/2GiB UD per-condition evidence budgets and22GiB GPU peak
guard remain unchanged. No model copy or extra forward is introduced.

The independent151 audit reconstructs each scoped admission proof and all-layer
fingerprints from completed source inputs/outputs, checks selected-entry coverage,
request/epoch binding and ownership, then matches consumed snapshot bytes and
scatter output. Counter totals are checked independently. Original payload,
LoRA/addition, position, policy, logit and quality audits remain active. Mode
configuration is sealed and checked against physical/profiling receipts. Older
evidence without D5 receipts remains readable; no old plan becomes scoped.

## CPU preparation on SERAPH, after manual review/commit/push/pull

No data/model download or GPU allocation is needed for this preparation. Use the
verified job154175 plan/run/audit, with its actual audited source SHA, explicitly.

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
read -r -e -p 'Job154175 sealed experiment: ' HIST_EXPERIMENT
read -r -e -p 'Job154175 complete A/B/C run: ' HIST_RUN
read -r -e -p 'Job154175 independent audit: ' HIST_AUDIT
read -r -p 'Job154175 audited execution source SHA: ' HIST_SOURCE_COMMIT
git cat-file -e "$HIST_SOURCE_COMMIT^{commit}"
export SSG_SOURCE_COMMIT="$(git rev-parse HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export D5_ROOT="$PWD/results/r5/i5r/parallel/subsequence_diagnostics/d5_$STAMP"
mkdir -p "$D5_ROOT" logs

PYTHONPATH=src:tests python3 -m unittest \
 test_r5_subsequence_validation test_r5_subsequence_diagnostics \
 test_r5_subsequence_generation test_r5_sparse_physical \
 test_r5_subsequence_audit test_r5_subsequence_pilot -q

for MODE in EAGER_FULL_PROOF REQUEST_SCOPED_VERIFIED; do
 python scripts/152_r5_subsequence_diagnostics.py \
  --experiment "$HIST_EXPERIMENT" --run "$HIST_RUN" --audit "$HIST_AUDIT" \
  --expected-source-commit "$HIST_SOURCE_COMMIT" --prepare-revalidation \
  --source-commit "$SSG_SOURCE_COMMIT" --validation-mode "$MODE" --cuda-events \
  --output-dir "$D5_ROOT/${MODE}_plan"
 python scripts/151_r5_subsequence_generation.py cpu-validate \
  --experiment "$D5_ROOT/${MODE}_plan" --output-dir "$D5_ROOT/${MODE}_cpu"
done
```

Every native CPU test must PASS with zero SKIP in the sealed approvals before
GPU execution. The D5 suite includes real installed tiny OPT autoregressive
generation over an isolated Torch CPU UD/TCP, plus `.data`/view/stride mutation
tests. Locally unavailable native tests remain SKIP, not PASS. Old Exact Prefix
regressions can be rerun without GPU:

```bash
PYTHONPATH=src:tests python3 -m unittest \
 test_r5_i5_processes test_r5_i5r1_processes -q
```

The prepare loop only creates fresh sealed plans/CPU audits. It submits nothing.
Choose CUDA events consistently for both plans; omit the flag from both for wall
only. No runtime flag/environment may silently change validation mode.

## Manual GPU attempts, one per mode, never concurrently

Reuse the existing bounded151 wrapper: one RTX3090,4 CPUs,64GiB,3h30m, fresh model,
cache and CPU UD per A/B/C condition. No arrays or nested submission. Keep the
same selected node for comparison; actual UUIDs and any changes are recorded.
Run/inspect/audit the eager attempt before manually requesting the scoped attempt.

```bash
read -r -p 'Permitted node (moana-y3 or moana-y4): ' SSG_NODE
read -r -e -p 'Existing Conda root: ' I5_CONDA_ROOT
export SSG_NODE I5_CONDA_ROOT
export R5_I5_ENABLE_GPU=1 R5_ENABLE_GPU=1
export SSG_EXPERIMENT="$D5_ROOT/EAGER_FULL_PROOF_plan"
export SSG_CPU_AUDIT="$D5_ROOT/EAGER_FULL_PROOF_cpu"
export SSG_RUN_ROOT="$D5_ROOT/EAGER_FULL_PROOF_runs"
mkdir -p "$SSG_RUN_ROOT"
unset CUDA_VISIBLE_DEVICES
sbatch --nodelist="$SSG_NODE" --export=ALL jobs/r5_subsequence_generation.sh
```

After the eager job completes:

```bash
export CUDA_VISIBLE_DEVICES=''
read -r -e -p 'Eager completed run_JOBID directory: ' EAGER_RUN
python scripts/151_r5_subsequence_generation.py analyze \
 --experiment "$D5_ROOT/EAGER_FULL_PROOF_plan" --run "$EAGER_RUN" --output-dir "${EAGER_RUN}_audit"
python scripts/151_r5_subsequence_generation.py verify \
 --experiment "$D5_ROOT/EAGER_FULL_PROOF_plan" --run "$EAGER_RUN" --audit "${EAGER_RUN}_audit" \
 --output-dir "${EAGER_RUN}_verification"
```

Only after reviewing that attempt and its audit, manually submit the second:

```bash
export SSG_EXPERIMENT="$D5_ROOT/REQUEST_SCOPED_VERIFIED_plan"
export SSG_CPU_AUDIT="$D5_ROOT/REQUEST_SCOPED_VERIFIED_cpu"
export SSG_RUN_ROOT="$D5_ROOT/REQUEST_SCOPED_VERIFIED_runs"
mkdir -p "$SSG_RUN_ROOT"
unset CUDA_VISIBLE_DEVICES
sbatch --nodelist="$SSG_NODE" --export=ALL jobs/r5_subsequence_generation.sh
```

Audit each actual completed run with its own plan, retaining failed attempts:

```bash
export CUDA_VISIBLE_DEVICES=''
read -r -e -p 'Scoped completed run_JOBID directory: ' SCOPED_RUN
python scripts/151_r5_subsequence_generation.py analyze \
 --experiment "$D5_ROOT/REQUEST_SCOPED_VERIFIED_plan" --run "$SCOPED_RUN" --output-dir "${SCOPED_RUN}_audit"
python scripts/151_r5_subsequence_generation.py verify \
 --experiment "$D5_ROOT/REQUEST_SCOPED_VERIFIED_plan" --run "$SCOPED_RUN" --audit "${SCOPED_RUN}_audit" \
 --output-dir "${SCOPED_RUN}_verification"
python scripts/152_r5_subsequence_diagnostics.py --compare-validation \
 --experiment "$D5_ROOT/EAGER_FULL_PROOF_plan" --run "$EAGER_RUN" \
 --audit "${EAGER_RUN}_audit" --verification "${EAGER_RUN}_verification" \
 --other-experiment "$D5_ROOT/REQUEST_SCOPED_VERIFIED_plan" --other-run "$SCOPED_RUN" \
 --other-audit "${SCOPED_RUN}_audit" --other-verification "${SCOPED_RUN}_verification" \
 --expected-source-commit "$SSG_SOURCE_COMMIT" --output-dir "$D5_ROOT/comparison"
```

Outputs: original151 artifacts plus D5 integrity receipts/authorizations inside
physical results, profiling summaries and independent audits. Comparison creates
`d5_validation_comparison.json`, `d5_validation_comparison.md` and SHA256 inventory.
It checks identical tasks/source/model/adapter/backend, selected/reused/fresh rows,
raw QKV/logit/token/text equality across modes, and actual nonzero reuse. Numerical
FAIL under Full Recompute remains FAIL, even when modes match each other.

One diagnostic run per mode cannot establish production speedup or statistical
benefit. Fixed A/B/C order, observer costs, warm state and run-to-run variance
remain confounded. No-hit is not success. A0 break-even remains NOT_ESTABLISHED.
Do not claim a numerical fix or safety against arbitrary concurrent mutation.

## Local verification

Final stable-source run:256 tests,244 PASS,0 FAIL,12 SKIP. D5 plus historical
diagnostics:41 PASS/3 SKIP; sparse generation/TCP/audit/CLI:21 PASS/4 SKIP;
relevant baseline regressions:167 PASS/5 SKIP; legacy I5/I5R1 Exact Prefix
ES/UD process regressions:15 PASS/0 SKIP. The D5 suite itself ran25 tests:
23 PASS,0 FAIL,2 native-dependency SKIP. Parameterized corruption cases include
Q/K/V, pre/post-authorization changes, same-key/object replacement, metadata,
later-layer mutation, replayed request records, disposal, CHU/PBR and counters.
Scalar autoregressive fixture logits/tokens and projection rows match both modes.
The real installed OPT/Torch alias tests remain unexecuted locally. No CPU
fixture result establishes OPT2.7B FP16 GPU fidelity or latency benefit.

Python syntax,152 CLI contract, actual151 Slurm argv regression and `bash -n`
passed. Source artifacts and historical results were not modified. No real
plan/CPU approval was produced locally: real evidence and the required reviewed
historical Git objects are unavailable. No GPU, Slurm, download, commit or push ran.

INTEGRITY_SAFETY=LIMITED (declared serialized mutation model; not global continuous
corruption detection or arbitrary concurrent writes). NUMERICAL_BEHAVIOR=UNCHANGED
in CPU fixtures; real native/GPU behavior NOT_TESTED. GPU_PERFORMANCE=NOT_RUN.
