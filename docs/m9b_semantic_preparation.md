# M9-B.1 real-workload semantic preparation

This stage converts prepared raw JSONL to the **existing** M9-B schema. It does
not change the simulator or production inference. Only the preparation command
executes TinyBERT; later simulation and `--validate-only` remain model-free.
No actual encoder/tokenizer execution was performed during implementation.

## Audited consumer schema

The consumer is `read_workload` in `src/semcache/simulation/multi_user.py`, called
by `scripts/37_run_m9b_multi_user.py`. It expects nonempty JSONL objects:

| Field | Consumer requirement/type | Use |
| --- | --- | --- |
| `token_ids` | Required nonempty list of nonnegative Python integers; booleans rejected | Exact overlapping 3-token windows, prompt hash, length/byte accounting |
| `cluster_id` | Required integer; booleans rejected | Cluster namespace in cache key; consumer itself does not bound C |
| `model_id` | Required string exactly `facebook/opt-2.7b` | Cache/byte-accounting namespace |
| `tokenizer_id` | Required truthy value, identical throughout file; preparation emits a string `repo@commit` | Tokenizer namespace provenance |
| `semantic_assignment_source` | Required truthy value, identical throughout file; preparation emits a descriptive versioned string | Semantic namespace provenance |
| `dataset` | Optional string, defaults to selected dataset; if provided its lowercase must match | Dataset identity; consumer normalizes it |
| `source_id` | Optional, defaults to row offset; consumer converts supplied value to string | Trace query identity |
| `adapter_id` | Optional identifier, preparation does not invent one | Same-adapter safety check; insufficient to establish correctness |
| `user_id` | Optional raw field; ignored by simulator assignment | SNIPS seeded round robin; MultiWOZ seeded conversation-group balancing |
| Other fields | Accepted, not required or interpreted for matching | Preserved provenance; included in consumer's full-row query-order digest |

M9-B does not need embeddings. File order determines query order; it does not
sort by `global_query_index`, `original_order_index`, or user. Token windows use
`(cluster_id, token tuple)`. Safety also needs the reassigned logical user, same
adapter, full-token prompt hash, matching occurrence start, and external
correctness evidence. Preparation adds no correctness evidence and does not
change the conservative zero-safe-reuse result.

The stricter **preparation validator**, without weakening the consumer, checks
`0 <= cluster_id < C`, metadata and revisions, source/order preservation, counts
and stream hashes. Each output row additionally contains:

* Preserved `dataset`, `source_id`, `global_query_index`, `original_order_index`,
  `conversation_id`, `domain_or_intent`, `query_text`. Existing types/values are
  retained. Absent optional provenance fields are emitted as null. Raw input
  requires dataset, nonempty string source_id/query_text, and contiguous integer
  global_query_index starting at zero. Other original fields are retained.
* `raw_user_id`: original user ID or null; `user_id` is removed. This is provenance
  only, with no user-count argument to preparation.
* `model_revision`, `model_revision_source`, `tokenizer_source_id`,
  `tokenizer_revision`, `tokenizer_revision_source`, and an object
  `tokenizer_revision_evidence`; OPT target identity does not claim OPT execution.
* `token_count`: integer equal to len(token_ids); `tokenizer_truncation`: false;
  `tokenizer_special_token_policy`: descriptive string.
* `semantic_encoder`: metadata object with checkpoint, resolved revision,
  pooling, max length, device, dtype and backend; `semantic_execution_provenance`:
  MEASURED for the actual CLI, TEST_STUB only for injected unit fixtures.
* `semantic_embedding_sha256`: hash of the actual encoded vector;
  `semantic_input_token_count`: integer TinyBERT token count before truncation;
  `semantic_truncated`: boolean; `semantic_truncation_max_length`: 512.
* `cluster_count`: 30 for SNIPS or 20 for MultiWOZ; `window_size`: 3;
  `cluster_diagnostics`: the existing M7 pre-update distance/count/update
  diagnostics; `actual_attention_impact_available`: false.

## Existing components and procedures

OPT tokenization uses `AutoTokenizer` from the pinned local snapshot for
`facebook/opt-2.7b`, revision
`905a4b602cda5c501f1b3a2650a4152680238254`. This is the tokenizer used by the M7
engine, without loading OPT weights. It explicitly calls
`add_special_tokens=True, truncation=False`, matching M7's default special-token
behavior. BOS/EOS behavior comes from that pinned tokenizer; no manual token
insertion, whitespace approximation, padding, or maximum OPT length is applied.
The existing `resolve_tokenizer_provenance` and `validate_tokenizer_snapshot`
require actual local snapshot/commit evidence; no requested-revision fallback
is used as resolution proof.

`TinyBERTSemanticEncoder` reuses `HuggingFaceTextEncoder.encode` and
`masked_mean_pool`, with `huawei-noah/TinyBERT_General_4L_312D` revision
`34707a33cd59a94ecde241ac209bf35103691b43`, masked_mean, max_length=512, FP32,
eval mode and inference mode. As in M7, text is independently re-tokenized for
TinyBERT. Padding is masked from the mean; semantic inputs longer than 512 tokens
are truncated by the existing encoder. Pre-truncation length and per-row flags
make this explicit. OPT tokens remain complete even when semantic text is
truncated. Semantic embeddings are not OPT attention-impact measurements.

Assignments use the existing `IntentClusterer` unchanged:

1. Encode in fixed source-order batches (default 32). Initialize centroids with
   the first C vectors, counts one, using M7 `first_k`. This is a first-C
   lookahead initialization, not independently trained clustering.
2. Replay **all** rows, including the first C, once in source order. The initial
   anchors already count once in the prior; their replay counts once more when
   flushed. This is explicitly the M7 warmup-plus-observation convention.
3. Assign using Eq.8 minimum Euclidean distance; existing lowest-index tie
   breaking is retained. Buffer each `(assigned cluster, vector)`.
4. After queries 100, 200, etc., apply Eq.9 incremental means in buffered order.
   Pending assignments are not recalculated during flush. No final partial
   flush is added; the remainder cannot affect any already emitted assignment.

`seed_everything` enables deterministic algorithms. No bandwidth, cost scenario,
user count, original user, or dataset label such as domain_or_intent is used as
an embedding/clustering feature. C and w=3 are PAPER_DEFINED. Checkpoint, text
re-encoding/pooling/tokenization, first-C warmup and scheduling details are
REPRODUCTION_CHOICE, not paper-exact preprocessing or Eq.7 bridge reproduction.
Actual TinyBERT execution is MEASURED provenance, not a latency measurement or
physical QKV correctness claim.

## Files and validation

New files:

* `scripts/38_prepare_m9b_semantic_workload.py`: offline model-loading CLI and
  model-free validation mode.
* `src/semcache/experiments/m9b_semantic_workload.py`: conversion, serialization,
  provenance and source/output validation with injectable test components.
* `tests/test_m9b_semantic_workload.py`: standard-library, model-free tests.
* This guide; README links to it.

The output sidecar is `OUTPUT.jsonl.manifest.json`. It records SHA256 of source
raw bytes, output bytes, ordered identities, token sequence stream, cluster ID
stream, embedding hashes, and centroid state; package/device/batch/seed metadata
and scheduling/provenance are also retained. No full embedding export is needed
by the consumer. On-disk validation checks all rows and separately passes the
first ten serialized rows to the unchanged M9-B reader.

Existing output or sidecar files are refused. A rerun needs a new output path;
`--compare-manifest OLD.manifest.json` requires identical source/order/token/
assignment/output hashes before publishing it. Reproducibility is scoped to
fixed packages/device/batch/seed and snapshots; bitwise agreement across CPU/GPU
or library versions is not promised. Raw files are never opened for writing.

## Exact SERAPH commands

From the repository root in the existing environment with local HF assets:

```bash
python3 scripts/38_prepare_m9b_semantic_workload.py --dataset snips --source results/workloads/snips.jsonl --output results/workloads/m9b_snips_semantic.jsonl --device cpu --batch-size 32 --seed 42 --expect-rows 13784
python3 scripts/38_prepare_m9b_semantic_workload.py --dataset multiwoz --source results/workloads/multiwoz.jsonl --output results/workloads/m9b_multiwoz_semantic.jsonl --device cpu --batch-size 32 --seed 42 --expect-rows 56776
```

These expected counts check the particular current prepared artifacts, not
universal dataset sizes. Omitting the count flag permits other prepared splits;
`--check-current-prepared-count` is an alternative spelling for these checks.
All HF access uses `local_files_only=True`, with HF_HUB_OFFLINE and
TRANSFORMERS_OFFLINE set by the preparation loader. Missing snapshots fail;
there is no download option. `--device cuda` is available but is a different
recorded execution configuration; use a fixed device for compared reruns.

Model-free post-preparation validation:

```bash
python3 scripts/38_prepare_m9b_semantic_workload.py --dataset snips --validate-only --expect-rows 13784
python3 scripts/38_prepare_m9b_semantic_workload.py --dataset multiwoz --validate-only --expect-rows 56776
```

Subsequent real cache/workload runs, with no models:

```bash
python3 scripts/37_run_m9b_multi_user.py --snips results/workloads/m9b_snips_semantic.jsonl --seed 42 --output-dir results/m9b/snips
python3 scripts/37_run_m9b_multi_user.py --multiwoz results/workloads/m9b_multiwoz_semantic.jsonl --seed 42 --output-dir results/m9b/multiwoz
```

To include normalized fixture cost diagnostics, append
`--system-summary results/m9a/m9a_summary.json --impact-summary results/m9a2/impact_summary.json`
to either command when those local timing files are available at those paths.
Supply both dataset flags in one invocation for the combined 36-configuration
matrix. The simulator and its original input validator are unchanged.

## Tests and limitations

```bash
PYTHONPATH=src python3 tests/test_m9b_semantic_workload.py
PYTHONPATH=src python3 tests/test_m9b_multi_user.py
```

Tests use explicitly labeled TEST_STUB components: schema/order/count retention,
repeat hashes, cluster bounds and buffered counts, unchanged source bytes,
user-count/raw-user independence, consumer acceptance, invalid provenance,
truncation accounting, overwrite refusal, and current-artifact count checks.

Real prepared inputs and model caches are on SERAPH and are not present in this
implementation checkout. Preparation was deliberately not executed with actual
models during Codex work. Real row counts, checkpoint loading, and device runtime
must therefore be validated by the commands above. The implementation holds the
raw and output rows in memory and encodes in batches. It exports cluster IDs and
embedding hashes, not full vectors. M9-B latency/correctness limitations remain:
semantic preparation adds neither attention impacts nor safe cross-user QKV
reuse evidence, and does not change arbitrary-length latency support.
