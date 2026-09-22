# M7: full semantic SemCache path

**M8.5 update:** See [prefill paper alignment](M85_PAPER_ALIGNMENT.md) for the current
reducer default, lookup-only state fix, configurable schedules and scope limits.
Earlier milestone defaults below are historical where they conflict.

M7 completes the structural semantic path without claiming paper-scale numerical
reproduction. M1–M6 already supplied OPT Q/K/V capture, EdgeLoRA-style rank-8
base/user decomposition, physical projection injection and skipping, exact
cluster-scoped token matching, the global cache and its admission/eviction
policy, workloads/baselines, analytical accounting, and provenance.

M7 adds a persistent Hugging Face TinyBERT encoder, explicit L2 cluster
diagnostics, a replaceable actual-attention reducer, current-query impact
initialization, hit-only CHU, bounded scalar-history PBR, optional periodic PBR,
and an opt-in OPT-125M end-to-end probe. It does not retain attention matrices in
history. The old controlled encoder and logical simulation remain available.

## Paper-defined versus reproduction choice

| Component | Setting | Provenance |
|---|---|---|
| Semantic encoder family | TinyBERT | `PAPER_DEFINED` |
| Encoder checkpoint | `huawei-noah/TinyBERT_General_4L_312D` default, overrideable | `REPRODUCTION_CHOICE` |
| Encoder revision | `main` unless overridden | `REPRODUCTION_CHOICE` |
| Sentence pooling | attention-mask-aware mean of final hidden states | `REPRODUCTION_CHOICE` |
| Tokenizer/max length | checkpoint tokenizer, truncation, 512 | `REPRODUCTION_CHOICE` |
| Assignment/update | nearest Euclidean centroid (Eq. 8), incremental mean (Eq. 9) | `PAPER_DEFINED` |
| Cluster counts | MultiWOZ 20, CoQA 40, SNIPS 30 | `PAPER_DEFINED` |
| Initialization | first C embeddings (`first_k`) | `REPRODUCTION_CHOICE` |
| Centroid schedule | assign using current centroids, then immediately apply Eq. 9 | `PAPER_DEFINED` |
| Optional buffered centroid mode | flush every configured interval | `REPRODUCTION_CHOICE` |
| Attention reducer | `mean_layer_head_frobenius_v1` | `REPRODUCTION_CHOICE` |
| Cold block impact | current query's reduced attention | `REPRODUCTION_CHOICE` |
| CHU rho | 0.8 | `PAPER_DEFINED` |
| PBR history Lambda | 100 | `PAPER_DEFINED` |
| PBR scheduler | manual by default; optional fixed query interval | `REPRODUCTION_CHOICE` |
| Exact token IDs within selected cluster | executable reuse authorization | `REPRODUCTION_CHOICE` |

The encoder calls `AutoTokenizer` and `AutoModel` once at construction, switches
the model to evaluation mode, disables parameter gradients, and encodes batches
under `torch.inference_mode()`. No sentence-transformers substitution is used.
Its metadata exposes model/revision, resolved revision, tokenizer identity,
hidden size, pooling, maximum length, device, and dtype.

## Clustering and attention impact

Assignment is

`c = argmin_c ||e_u - mu_c||_2`.

The selected cluster's diagnostic includes the pre-update unsquared L2 distance.
Immediately after assignment, the selected centroid applies

`mu_c <- (N_c mu_c + e_u) / (N_c + 1)`

and increments `N_c` exactly once. Only that centroid changes, and the next
query observes the updated value. The normal M7 mode is `immediate_eq9`; the
legacy `buffered` mode remains available only as an explicit reproduction
choice and is not described as the paper's update timing.

Per-query output records the pre-update distance, count before/after, whether
the update was applied, and centroid-shift L2 without serializing centroids.
The backward-compatible `cluster_distance` is the pre-update distance, while
`cluster_updated` is an alias for `centroid_update_applied` in normal M7 output.

With `first_k` initialization, the first C embeddings become the C centroids and
each initializes its membership count to one because it has already been
incorporated. Synthetic callers supplying externally chosen centroids may pass
explicit counts (including zero), but subsequent observations still increment
the selected count exactly once.

Intent-centroid updating is independent of semantic-impact PBR. Clustering
updates after every query. PBR instead retains the most recent Lambda scalar
impact observations per cluster; its paper-default Lambda remains 100.

For each layer `l` and head `h`, the default reducer selects all valid query
positions and block key positions `[start,end)`. Padding and cells where the key
position is later than the query position are excluded. It computes

`r_lh = sqrt(sum_valid a[l,h,q,k]^2)`

and returns `A_q(c,p) = mean_l,h r_lh`. Model attention probabilities already
respect causal masking; the reducer enforces that mask again. This deterministic
layer/head aggregation is our choice, not a claim about the authors' hidden
implementation. OPT uses explicit eager attention for this path.

## Impact lifecycle and cache policy

The forward order is embedding → cluster → windows → lookup → mixed fresh/cached
QKV forward → attentions → scalar block impacts → CHU/admission → optional PBR.
Thus admission of an unseen block uses its current-query actual impact without a
circular pre-forward decision.

On an actually selected physical hit, CHU performs

`I <- rho I + (1-rho) A_current`, with `rho=0.8`.

It records old impact, current attention impact, new impact, query, cluster and
block key. Passive lookup does not trigger CHU.

Each cluster retains at most Lambda records containing only query identity and
`block_key -> scalar impact`. Forced or scheduled PBR applies Eq. 13: the new
entry impact is the arithmetic mean of scalar impacts among recent queries that
contain that block. If none contain it, the old impact remains unchanged. The
public `recalculate_cluster(cluster_id)` method permits a tiny forced probe.

These values update the existing entry's `impact`; no second policy exists.
Consequently existing max-normalized admission Eq. 11 and eviction Eq. 12 see
the dynamic value while F, A and S retain their previous meanings and weights.
Logical fixture execution remains a distinct cheap capability without actual
attention impact.

## Safety and scope

TinyBERT only chooses a coarse cluster. Reuse still requires the existing exact
token-ID subsequence key within that cluster. Context, positions, layer history,
and user LoRA can make equal token IDs produce unequal QKV, so
`safe_reuse_claimed` remains `false`. The exact-identity control establishes
implementation parity for the physical mechanism; it is not a safety threshold
for approximate cross-query reuse.

This is a structural reproduction on a tiny trace, not a MultiWOZ/CoQA/SNIPS
evaluation, parameter sweep, latency benchmark, BLEU run, or reproduction of
paper figures. OPT-6.7B remains an analytical/future target.

## Commands

From the repository root:

```bash
python3 -m pytest -q tests/test_m7_semantics.py tests/test_mixed_semcache.py tests/test_semcache_system.py
python3 -m pytest -q
SEMCACHE_TINYBERT_INTEGRATION=1 python3 -m pytest -q tests/test_m7_semantics.py -rs
SEMCACHE_OPT_INTEGRATION=1 python3 -m pytest -q tests/test_mixed_semcache.py -rs
python3 scripts/29_run_m7_semcache_integration.py --device cuda
```

Models are local-only by default. Add `--allow-download` to the final command
only when network/model provisioning is explicitly intended. The smoke probe
uses Lambda=4 to force bounded-history behavior without 100 model queries; omit
or change `--pbr-history-lambda` as needed. Its JSON is written to
`results/m7/m7_semcache_integration.json` and contains no attention matrices.
