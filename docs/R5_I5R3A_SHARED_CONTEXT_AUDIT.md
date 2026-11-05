# I5R-3A shared-context eligibility audit

Implementation only. Real-data audit NOT_RUN on the local workstation: the
SERAPH registry `results/r5/i5/registry_longbench3.json` is absent locally.
I5R-3A research completion requires a real dataset audit **and** independent
verification on SERAPH. This stage performs no inference, semantic encoding,
cache operations, numerical evaluation, downloads or job submission.

## Method and denominators

Use exactly qasper, 2wikimqa and multi_news from an explicit existing registry
mapping dataset key to prepared directory. Absolute paths are preferred; relative
paths follow existing I5 commands executed from the repository root. Originals,
IDs, group IDs and references remain unchanged. `shared_context_key` is a new
UTF-8 context SHA-256, with exact string equality checked for pair construction.
Empty contexts, if any, remain visible and are not called useful evidence.

`load_dataset()` verifies sealed inventories, sample hashes, duplicate IDs,
original-record hashes and real-data scope. Additional checks require correct
source/version, immutable revision, split and source SHA inventory. Existing
upstream files are rehashed when available; unavailable upstream paths are
explicitly labeled, never claimed independently checked. The original records
embedded in normalized data are still verified. Model/adapter preparation uses
`r5_multi_artifacts.validate()` and the real thirty-user registry; CPU tokenization
uses `opt_tokenizer()` with local-only loading, vocabulary/config identities and
2048 positions. No model weights are loaded for inference. Preparation validation
reads existing adapter bytes for integrity, without evaluating LoRA.

All original prepared samples are audited, without sampling or duplication.
Bounds: 1..2000 records per dataset, 500000 directed shared/prefix records per
dataset; exceeding a bound rejects instead of silently subsampling. Existing
I5 prompts are prepared with QA budget 32, summary budget 96, unchanged fixed
fields and deterministic context-only head/tail truncation. Default input budget
is 2048 minus generation reservation. `--input-limit 512` permits a separately
sealed historical-budget scenario; never merge its counts with another plan.
Group-stable truncation is NOT_EVALUATED (optional policy not implemented).
Retained context counts exclude the omission marker, as in production I5.
No gold field enters a prompt.

Required summaries:

- `total_samples`: original unique prepared IDs, not estimated upstream size.
- `full_context_eligible`: final prompts eligible without truncating context.
- `context_limited_eligible`: all eligible final prompts, including untruncated.
- `excluded_count`, exact reasons, `truncated_count`, removed-token counts/rate.
- `shared_context_groups`: exact original contexts with at least two IDs.
- `shared_context_request_pairs`: unordered distinct-ID pairs, n(n-1)/2.
- `strict_prefix_pairs`: directed distinct complete prompts where source tokens
  are strictly shorter and exactly equal to the initial target tokens. Identical
  full prompts are excluded. These pairs can span different original contexts.
- `same_context_strict_prefix_pairs` and `cross_context_strict_prefix_pairs`
  partition that same directed denominator by exact original context equality;
  their sum equals `strict_prefix_pairs`. Both remain NATURAL, STATIC_ONLY.
- `authorized_prefix_pairs`, `capacity_eligible_pairs`, `within_lifetime_pairs`:
  independent filters over strict-prefix pairs, not a sequential funnel.
- `statically_eligible_pairs`: intersection of prefix, authorization, source
  capacity, proposed time order and same reset block. This is a **necessary
  static-opportunity count**, not admission/residency/completion/physical proof.
- `natural_actual_hit_status=NOT_EVALUATED`. No invented HIT counter or rate.

Proposed scenario: deterministic exact-context key order, then existing group
hash/turn order, followed by existing `pack_groups()` over **original** group IDs.
This may not keep every shared-context group in one block; splits are reported.
Original source ordinal and IDs are retained. Each independent block contains
at most 20 requests and resets its cache. Current `routing()` is applied unchanged;
no same-adapter forcing. `sharing()` and `resolve()` enforce actual authorization.
Sequential completion barriers are proposed prerequisites, not measured receipts.
Source bytes use the verified tokenizer Raw-QKV accounting (491520 bytes/token:
3 x 32 x 2560 x FP16), not consumed K/V size. Capacity is 512 MiB per adapter;
64-entry bound cannot bind in a block of at most 20 requests. Offered bytes per
adapter/block are reported, **not** measured occupancy or an LRU simulation.
Admission, eviction, residency, semantic candidates, actual completion, physical
GPU identity and HITs remain NOT_EVALUATED. An individually fitting source can
still be evicted or never admitted during real execution.

`order_comparison` now evaluates production I5 `select()` ordering and group
packing against the context-grouped proposal. Both scenarios use **all the same
eligible original IDs**, verified identical complete prompt tokens, the same
routing, seed, block size, capacity and reset policy. This is not an attempt to
recreate a historical experiment's selected counts or recorded request order.
Each scenario reports its prefix partition, authorization/capacity filters,
source-precedes-target, lifetime and static-eligibility counts. The comparison
reports gained/lost static pairs. Candidate records retain `original_i5_order`
location/check/rejection receipts alongside the existing proposed-order checks;
`original_i5_order` at dataset level records the complete comparison order.
The summary artifact also includes these comparisons. All results are STATIC_ONLY,
and no order can establish a measured HIT, admission or residency.

Each eligible pair is linked to a context-dependency component. Counts of these
components, distinct contexts, and group-size histograms support planning; they
are not claims of statistical independence of model outputs. Repeated targets
or sources cannot be treated as independent samples.

## Natural versus controlled / I5R-3B recommendations

NATURAL contains only distinct original requests with unmodified I5 prompts.
Every shared pair is retained in both directions for ordering/authorization
inspection; strict-prefix counts use the directed shorter-to-longer relationship.
No shorter source is extracted from a longer natural request.

At most three natural pilot records are proposed per dataset. They contain IDs,
routes, prompt hashes, exact prefix/suffix lengths, order/block assignments,
authorization/capacity checks and missing runtime evidence. No qualifying pair
produces `NO_NATURAL_ELIGIBLE_PAIR`, even if there are shared contexts.

Separately, at most three CONTROLLED_DIAGNOSTIC suggestions per dataset slice
64 token IDs directly from an existing target. They require a new actual
standalone source, same-user/adapter execution, admission and later authorized
reuse through the normal runtime. They are constructed workloads, not natural
requests or natural HIT-rate observations. No synthetic suffix is introduced.
These recommendations do not generate or submit an I5R-3B experiment.

## CPU operator workflow

Review and manually commit/push/pull all four new files first. Preserve historical
outputs. Use the existing semcache environment and offline assets:

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
read -r -e -p 'Existing sealed I5 adapter preparation directory: ' I3A_MULTI
export I3A_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export I3A_ROOT="$PWD/results/r5/i5r/shared_context_audit"
export I3A_PLAN="$I3A_ROOT/plan_$I3A_TAG"
export I3A_AUDIT="$I3A_ROOT/audit_$I3A_TAG"
python scripts/142_r5_i5r3a.py prepare \
 --registry results/r5/i5/registry_longbench3.json \
 --multi-prepared "$I3A_MULTI" --block-size 20 --output-dir "$I3A_PLAN"
python scripts/142_r5_i5r3a.py analyze \
 --plan "$I3A_PLAN" --output-dir "$I3A_AUDIT"
python scripts/142_r5_i5r3a.py verify \
 --plan "$I3A_PLAN" --audit "$I3A_AUDIT" \
 --output-dir "${I3A_AUDIT}_verification"
```

For a 512-input comparison, create a **new** plan/output set with
`prepare --input-limit 512`; no existing report is replaced. There is no Slurm
command in this stage. Missing data/assets/provenance fail closed.

## Output schema and independent verification

All directories are fresh and SHA-256 sealed using `i5_output_manifest.json`.

- `i5r3a_plan.json`: source commit/files, dataset paths/manifests/inventories,
  registry hash, validated adapter inventory/registry, model/tokenizer identities,
  fixed policies, generation budgets, order seed/block size and plan digest.
- `i5r3a_dataset_provenance.json`: source/split/revision inventories and scope.
- `i5r3a_results.json`: per-dataset summary, per-sample original context bytes,
  context/token/prompt hashes and actual final token IDs, exclusions, shared
  groups, directed candidate records and flags, proposed order/block statistics,
  separate natural/controlled bounded pilot suggestions.
- `i5r3a_summary.json`: required counts, denominators, runtime exclusions,
  statistical warnings, plan/source and analysis-source identities.
- `i5r3a_independent_audit.json`: independently recomputed result equality,
  input/source verification and report-inventory binding; CPU static scope only.

`verify` rereads original sealed datasets, recomputes tokenization/routing/order/
pairs, verifies archived Git blobs and checks report equality. It does not trust
a self-reported HIT. Real preparation rejects uncommitted execution sources.
Historical source is verified against its Git blobs, separately from the current
analysis source. The normal inference source verifier is unchanged.

Local tests:

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r3a -q
PYTHONPATH=src:tests python3 -m unittest test_r5_i5_preflight -q
python3 scripts/142_r5_i5r3a.py --help
```

CPU fixtures are synthetic and never enter real-data mode. Full real-tokenizer
analysis and independent verification remain NOT_RUN until the SERAPH workflow
succeeds. Numerical diagnosis/fidelity, performance, optimized cache lifetime and
all later roadmap stages remain outside I5R-3A.

Local final validation after the adapter-manifest patch: 36 tests, 36 PASS,
0 FAIL, 0 SKIP (27 I5R-3A checks and 9 existing I5 preflight regressions).
The CLI regression invokes the real argument parser/dispatch against the actual
prepare signature and enters the real prepare function up to input loading.
The unchanged `--multi-prepared` flag maps explicitly to the internal `multi`
parameter. Synthetic verifier tests exercise
both a sealed successful recomputation and rejection of a resealed false runtime
claim. They do not constitute a real-data independent audit. Reports exceeding
the existing 16 MiB JSON-reader metadata limit fail explicitly before output;
there is no silent reduction of records or candidate pairs.

Create fresh plan/audit/verification directories under the reviewed corrected
source commit before SERAPH execution; never edit/reseal earlier outputs. The
prepare CLI dispatch is CPU-verified, but real offline assets/datasets and the
SERAPH independent audit remain NOT_RUN locally.

Adapter preparation inventories are explicitly verified using their existing
`r5_output_manifest.json` in both prepare and compute. I5 dataset inputs and
I5R-3A outputs retain `i5_output_manifest.json`. Missing/corrupt R5 inventories
reject; there is no fallback, rename or resealing of existing adapter evidence.
