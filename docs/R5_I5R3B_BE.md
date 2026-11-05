# I5R3B_BE measured source-cost break-even

LOCAL IMPLEMENTATION ONLY; GPU_NOT_RUN. The real completed B2 artifacts are not
available on this workstation. Supplied N=2 totals 147254.331 ms (C), 146286.146 ms
(A0) and +968.185 ms are previously verified handoff observations, **not locally
reproduced measurements**. Individual source/savings/gap values and N=3 estimates
must be calculated from the actual sealed records by the offline command.
No new admission, lifetime, LRU, semantic, authorization or arithmetic policy.

## Offline evidence and economics

All four baseline paths are explicit: 3B experiment, raw paired run, 3B analysis,
and independent verification. Expected baseline source defaults to
`a28e770779568554ad0bac9a49c888a968b6ffed`. The reader verifies every SHA inventory,
archived source Git blobs, plan/experiment IDs, original I5 exported independent
audits and raw GPU completion metadata. It matches raw original request contracts,
actual IDs, tokens, text, routes and policy to sealed tasks, model/adapter
fingerprints, raw per-request timing to exported metrics, and original A0/ABC
sequence costs. The reader reuses exported historical tensor audits; it does not
pretend to rerun the old strict source audit from new code. The existing I5 source
verifier remains strict and unchanged. Missing/altered inputs fail closed.

For each complete measured sequence:

```
D(N) = C source-inclusive sequence span - A0 target-only sequence span
     = source application E2E
       - sum(A0 positive E2E - C positive E2E)
       + (C negative E2E - A0 negative E2E)
       + (C inter-request gaps - A0 inter-request gaps)
```

Negative D is faster. Equality at zero is not break-even. Timings are checked
against monotonic arrival/completion receipts. Source model-forward and admission
fields are nested descriptions of application E2E, never added a second time.
Process lifecycle, initialization and warmup are separate from sequence spans.
Positive-only spans and negative-inclusive sequences are both saved. Per-scored
request amortization includes the negative control in its explicit denominator.

N=2 is measured; N=3+ is **EXTRAPOLATED_NOT_MEASURED** until new GPU evidence exists.
The main extrapolation adds one observed Target1 saving for the requested repeat;
additional scenarios use Target2, mean, min and max savings. Source, negative and
gap overhead are assumed fixed. Nonpositive savings produce no finite estimated
crossing when D(2)>=0. Strictly-negative crossing uses floor(D(2)/saving)+1 extra
requests, not an equality-inclusive ceiling. Two dependent targets are not a
confidence interval; run/order/thermal/instrumentation variance is unmeasured in
the baseline. No interpolation is a measured minimum.

## Immutable repeat contract

The exact verified baseline selection is reused; no dataset reselection,
re-tokenization, new context truncation, rerouting or changed generation policy.
B2 source must have 512 IDs, scored prompts 1280 IDs, Adapter_A and the recorded
original target users User_08, User_06 and User_10. Unexpected baseline selection
fails instead of silently choosing another pair.

| ABC order | Dataset identity | User | Request identity |
|---|---|---|---|
| 1 source | Original source/Target1 provenance | User_08 | Unchanged source ID |
| 2 Target1 | Original genuine question 1 | User_08 | Unchanged original target ID |
| 3 Target2 | Original genuine question 2 | User_06 | Unchanged original target ID |
| 4 Target1 repeat | Same question 1, IDs, text, references and policy | User_08 | Original Target1 request ID + `_BE03` |
| 5 negative | Original different-context question | User_10 | Unchanged original negative ID |

A0 omits only the source and renumbers sequence positions 0..3. It preserves all
four actual scored request IDs and content. The original sample/group IDs stay
unchanged. There are **four scored request observations, three unique scored
samples, one repeated observation, three distinct original users and one shared
positive-context experiment unit**. The negative uses its original other context;
it does not create another independent shared-prefix group. Source is unscored.
Each condition has one cold source and fresh ES/model/cache and three CPU UD
sessions. Unique dataset samples and request identities are deliberately distinct.

Original 3B validation still rejects repeated sample IDs. The separate BE
contract compares complete canonical task dictionaries against baseline clones.
No source grant, fake HIT, injected entry or fallback is supplied. Source admission
and completion are real runtime operations. Every positive must consume that
same physically resident source; no positive admission may introduce another
source. The negative must MISS. Misses/rejections and source loss produce a
functional FAIL with actual evidence, not an altered retry. False grant/integrity
claims fail the evidence audit.

All 32-layer strict capture levels are inherited: source and every positive
(including the repeat) STRICT_AUDIT, negative scalable. Existing I5 audits verify
source Raw-QKV, reconstructed/consumed K/V, omitted projections/CPU LoRA/TCP
prefix work, adapter authorization and completion order. Per-request metrics
include TTFT, application/model E2E, CPU LoRA time, semantic/eligibility/cache/KV
lookup times, actual bytes, reused tokens, skipped rows, occupancy, grants and
real evictions. Zero eviction is valid. Full-vocabulary frozen numerical checks,
KL/argmax, token/text comparisons and available last-prefill-position layer
diagnostics cover the repeated request. These diagnostics do not prove causality.

## Bounds and later repeats

Default: one new 3-positive-request pair block (ABC five requests, A0 four).
A0 retains auxiliary B/C so the unchanged I5 runner/auditor executes six conditions;
only A0 FULL_RECOMPUTE enters economics. No historical N=2 values are pooled with
new N=3 as one paired run. Source512 still stores 251658240 bytes (240 MiB) of
Raw Q/K/V; cached Q is not consumed. Cache capacity remains 512 MiB per adapter,
64 entries, cluster count one, unchanged LRU. No padding/precision repair path.

One RTX3090, four CPU cores, 64 GiB Slurm RAM, three-hour allocation; each ABC/A0
child has a 4800-second bound. Existing resource estimator and GPU/model/TinyBERT
preflights remain mandatory. Evidence budget: 16 GiB per condition, 96 GiB total
worst-case plus 1 GiB free-disk reserve per paired block. GPU estimate must remain
below the existing conservative 22 GiB limit. Transport reuses 65536-byte maximum
and 8192-byte reserved headroom, including every repeat in ABC/A0. Per-request
sizes and utilization are printed by inspect; no runtime truncation/retry.

Optional `--repetitions 2` or `3` creates fresh paired execution blocks, not new
context groups. ABC/A0 execution order alternates; internal A/B/C order rotates
through existing ORDERS. Actual lifecycle timestamps verify the declared order.
All paired comparisons require the same actual GPU within their block. New D
mean/sample variance are reported only across new paired blocks; context-group
count remains one and no population-speedup CI is asserted.

Profiles of 4..8 positive requests require explicit `--validated-be-audit PATH`
pointing to a real sealed, strict-audited successful three-HIT BE run. Extra
requests alternate original Target1/Target2, with `_BE04`, `_BE05`, etc. IDs.
Existing strict evidence/resource guards may reject larger profiles at 16 GiB;
no capture is silently reduced. No automatic promotion or submission occurs.

## SERAPH commands (operator-managed deployment first)

Include all seven new files in one reviewed commit. On SERAPH, manually pull it,
keep historical Git objects and all baseline outputs unchanged, and activate the
offline environment. Paths below are explicitly entered, not guessed:

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export I5_CONDA_ROOT="$(conda info --base)"
read -r -e -p 'Verified B2 experiment directory: ' IBE_BASE_PLAN
read -r -e -p 'Verified B2 raw paired run directory: ' IBE_BASE_RUN
read -r -e -p 'Verified B2 3B analysis directory: ' IBE_BASE_AUDIT
read -r -e -p 'Verified B2 independent verification directory: ' IBE_BASE_VERIFY
read -r -p 'Reviewed current BE source commit: ' IBE_SOURCE_COMMIT
export IBE_SOURCE_COMMIT
test "$(git rev-parse HEAD)" = "$IBE_SOURCE_COMMIT"
IBE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
IBE_ROOT="$PWD/results/r5/i5r/controlled_hit/i5r3b/be"
```

CPU-only actual historical economic analysis **before preparing/submitting**:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/146_r5_i5r3bbe.py offline \
 --experiment "$IBE_BASE_PLAN" --run "$IBE_BASE_RUN" --audit "$IBE_BASE_AUDIT" \
 --verification "$IBE_BASE_VERIFY" \
 --expected-commit a28e770779568554ad0bac9a49c888a968b6ffed \
 --output-dir "$IBE_ROOT/offline_N2_$IBE_TAG"
```

Review measured source, two actual savings, negative/gap overhead, preserved
numerical failure and N=3 sensitivity. If inputs disagree, stop; no GPU retry.

```bash
export IBE_EXPERIMENT="$IBE_ROOT/plan_N3_$IBE_TAG"
export IBE_RUN_ROOT="$IBE_ROOT/attempt_N3_$IBE_TAG"
CUDA_VISIBLE_DEVICES='' python scripts/146_r5_i5r3bbe.py prepare \
 --experiment "$IBE_BASE_PLAN" --run "$IBE_BASE_RUN" --audit "$IBE_BASE_AUDIT" \
 --verification "$IBE_BASE_VERIFY" --positive-requests 3 --repetitions 1 \
 --expected-commit a28e770779568554ad0bac9a49c888a968b6ffed \
 --output-dir "$IBE_EXPERIMENT"
CUDA_VISIBLE_DEVICES='' python scripts/146_r5_i5r3bbe.py inspect \
 --experiment "$IBE_EXPERIMENT"
```

Inspect the exact original sample IDs, request IDs/order, users, token hashes,
resource estimates and all transport headroom before one manual GPU submission:

```bash
read -r -p 'Permitted node (moana-y3 or moana-y4): ' IBE_NODE
export IBE_NODE R5_I5_ENABLE_GPU=1 R5_ENABLE_GPU=1 IBE_BLOCK_INDEX=0
mkdir -p logs
IBE_JOB_ID="$(sbatch --parsable --export=ALL --nodelist="$IBE_NODE" jobs/r5_i5r3bbe.sh)"
```

After raw COMPLETE, CPU-only analysis and independent recomputation:

```bash
IBE_RUN="$IBE_RUN_ROOT/block_${IBE_BLOCK_INDEX}_${IBE_JOB_ID%%;*}"
IBE_AUDIT="${IBE_RUN}_be_audit"
CUDA_VISIBLE_DEVICES='' python scripts/146_r5_i5r3bbe.py analyze \
 --experiment "$IBE_EXPERIMENT" --runs "$IBE_RUN" --output-dir "$IBE_AUDIT"
CUDA_VISIBLE_DEVICES='' python scripts/146_r5_i5r3bbe.py verify \
 --experiment "$IBE_EXPERIMENT" --runs "$IBE_RUN" --audit "$IBE_AUDIT" \
 --output-dir "${IBE_AUDIT}_verification"
```

For an explicitly reviewed repetitions plan, submit each block separately by
setting IBE_BLOCK_INDEX, keep run paths distinct, and pass **all** completed paths
to `--runs` for analyze/verify. Missing/duplicate blocks reject. Do not automatically
submit another block after success or failure.

## Artifacts and interpretation

Offline: `i5r3bbe_offline.json`, `i5r3bbe_baseline_provenance.json`.
Plan: `i5r3bbe_protocol.json`, `i5r3bbe_selection.json`, `i5r3bbe_transport.json`,
`i5r3bbe_N2_economics.json`, separate I5 `abc/` and `a0/` sealed child plans.
Raw: unchanged I5 run/audit subdirectories and `i5r3bbe_attempt_status.json`.
Analysis: `i5r3bbe_results.json`, `i5r3bbe_independent_audit.json`.
Verification: `i5r3bbe_verification.json`. Every directory has SHA256 inventory.
All outputs are fresh; failed attempts and historical data are immutable.

Economic PASS requires actual D(N)<0. A physical-HIT economic PASS additionally
requires every intended positive consumes the resident source and negative MISS.
Evidence audit PASS is separate from functional and numerical PASS. Frozen
numerical FAIL remains FAIL. Correctness-qualified speedup and native PEFT strict
parity remain NOT_ESTABLISHED. One observed N=3 crossing does not establish a
universal/minimum reuse count or statistical inference across datasets/users.
Eight-token Qasper quality is descriptive, repeats do not inflate unique samples,
and all network bytes are localhost TCP measurements. No I5R-4 optimization.

Local test commands:

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r3bbe -q
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r3bbe test_r5_i5r3b \
 test_r5_i5r3a test_r5_i5_preflight test_r5_i5r1 test_r5_i5_controller -q
bash -n jobs/r5_i5r3bbe.sh
```

Tests use synthetic CPU tensors/metadata, including sparse sealed fixtures for
historical-reader checks, with historical source verification mocked only in
those unit fixtures. They do not establish real GPU timing/fidelity. Real N=2
reproduction, numeric N=3 sensitivity and exact selected sample-ID table remain
NOT_RUN locally until the CPU commands access original SERAPH data.

Final local CPU validation: **118 PASS, 0 FAIL, 0 SKIP** (25 BE tests and
93 existing 3B/3A/preflight/R1/controller regressions). Syntax/whitespace, all six
CLI help paths and Slurm shell syntax passed. The I5R-3B duplicate-sample assertion
is still exercised and rejects BE repeats; no original module was modified.

The seven new delivery files are this document, `jobs/r5_i5r3bbe.sh`,
`scripts/146_r5_i5r3bbe.py`, `src/semcache/edgelora/r5_i5r3bbe.py`,
`src/semcache/edgelora/r5_i5r3bbe_analysis.py`,
`src/semcache/edgelora/r5_i5r3bbe_runtime.py` and `tests/test_r5_i5r3bbe.py`.
All must be included in the reviewed commit before actual preparation.

Sequence completion follows unchanged I5 receipts: the final request's later
scoring/client serialization is outside the sequence span, earlier requests'
scoring/serialization can lie in inter-request gaps, and full process lifecycle
includes teardown/final logging. These scopes are not interchangeable with a
production-serving E2E benchmark. No GPU, Slurm, remote, download or Git push
was executed locally; no historical manifest, result or Git object was edited.
