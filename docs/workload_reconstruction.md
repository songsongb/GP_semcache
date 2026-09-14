# Reconstructable workloads, not the paper's undisclosed adjusted queries

**The exact SemCache personalization and adjustment procedure is unknown.**
M6A does not train adapters, invent user personas or relabel another dataset as
MultiWOZ, CoQA or SNIPS. Logical users identify requests only.

## Source ingestion

Script 20 accepts explicitly supplied JSON/JSONL, an HF `save_to_disk` directory,
or an explicit HF ID/config/revision with `--allow-download`. Without a source
it fails with actionable instructions. There is no default arbitrary dataset
release, dataset substitution or automatic network access. JSON split wrappers
select the requested split. An unsplit file is understood to be the supplied
split; users must select the correct source file. Directory input means HF
saved format, not a guess among assorted raw files.

Local files are fingerprinted by exact SHA256; HF sources record version and
fingerprint. Requested HF revision and resolved revision are separate; resolved
revision stays null when unavailable. Keep source files/versions alongside the
manifests. Source IDs are preserved, with deterministic row indexes only when
an ID is absent; duplicate normalized IDs raise. Unknown schemas fail loudly.

| Dataset | Supported source and output |
|---|---|
| MultiWOZ | Official dialogue-ID keyed `log` (alternating user/system); HF `dialogue_id` with list/columnar `turns` and explicit speaker. Each user turn becomes one query; immediate following system turn is its reference if available. Dialogue ID, turn ID, services/domains, goal and turn metadata are preserved. |
| CoQA | Official `data` wrapper or record list; story, questions and answers by turn ID (positional IDs only when missing). Each question becomes one query; answer text is reference. Story, original question/answer and conversation ID remain in metadata. HF string-list questions and columnar answer records are supported. |
| SNIPS | Utterance/intent, text/label, or official `intents -> utterances -> data` text chunks. Each utterance is one query; reference is the original intent label (numeric labels remain numeric strings unless source exports names). Original intent and metadata remain available. |

An empty query is dropped and its source index/ID/reason recorded. Missing
reference is retained as null. Malformed structure fails rather than being
silently dropped. Source example counts, transformed query counts, affected
source-example drops, dropped-query counts and max-query truncation are distinct.

## Transformation version 1

Both modes are REPRODUCTION_CHOICE, with no claim to the hidden paper procedure:

- `raw_query` (default): MultiWOZ user text; CoQA question **without story in the
  query** (story retained in metadata); SNIPS utterance with intent reference.
- `paper_reproduction_v1`: MultiWOZ `User: {text}\nAssistant:`; CoQA
  `Story: {story}\nQuestion: {question}\nAnswer:`; SNIPS
  `Classify the intent of this utterance: {text}\nIntent:`.

No previous conversation turns are inserted into prompts. Original story/turn
metadata permits future audited transformations. No query rewriting uses user
identity. Source selection is explicit; no train/eval repartition, random split
or adapter training occurs. This does not establish a valid quality benchmark:
task/generation/reference choices still need validation.

## Users, order and serialization

Default seed=42, logical users=50, assignment=`seeded_round_robin`:
sort user indexes by SHA256 of canonical `[seed,"user",index]`, then cycle
through that permutation in transformed source order. `deterministic_hash`
uses SHA256 of `[seed,dataset,split,source_id]` modulo user count. No built-in
Python hash or platform-dependent RNG is used. With at least 50 rows the
round-robin mode includes all 50 users; shorter limits may leave inactive users.

Assignment precedes ordering and limiting. `source_order` is default.
`seeded_shuffle` sorts by SHA256 of `[seed,"order",dataset,split,source_id]`,
with source ID as a deterministic tie break. No ordering is selected for a
better hit rate. `--max-queries` takes a prefix of the chosen order; null means
all transformed queries. There is no invented paper query count.

Rows contain global index, source/turn identity, split, query/reference,
domain/intent, logical user, source ordering, transformation name/version and
optional model token length. Stable UTF-8 JSONL uses sorted keys and compact
separators with LF line endings. SHA256 covers exact serialized bytes. The
sidecar `<workload>.manifest.json` records reconstruction settings, counts,
source hash/revision, user/order rule, C and creation time. `manifest_sha256`
covers manifest fields except creation time and the hash itself; thus identical
preparations have identical workload and reconstruction-manifest hashes.
Readers verify both. Timestamps do not alter workload hashes. Manifests reject
credential-bearing fields and never dump environment variables.

## Static reuse audit

Script 21 validates without writing or mutating inputs. It reports active and
configured users; queries/user min, median and max including zero-query users;
domain distributions; empty texts/references; duplicate exact query count; token
length p50/p90/p95/max; and per-domain/global `static_reuse_opportunity`.

With `--tokenizer-id` and `--tokenizer-revision`, a local HF tokenizer supplies
model token IDs. Otherwise explicitly labeled `whitespace_proxy_v1` is used;
these are not claimed to be model token statistics. Special tokens, if supplied
by a tokenizer, participate. No implicit truncation is performed.

For w=3, audit counts total windows, unique windows, occurrences beyond first,
occurrences with prior same-user evidence, occurrences with prior different-user
evidence, and the union of cross-user eligible positions in each target query.
Same-user and cross-user counts may overlap. Coverage respects trace order and
counts overlapping token positions once. It is an opportunity bound, not
SemCache hit rate, cache capacity behavior, safe QKV reuse or measured speedup.
Domain audits restrict prior evidence to that domain grouping.

## Smoke semantics

Script 23 requires an explicit limit and invokes `SemCacheEngine.logical_query`,
using existing M5 window extraction, matching, overlap selection, appearance
counts, admission and eviction. Fixture SHA scalar embeddings are not TinyBERT.
C comes from dataset config; fixture centroids span [0,1]. Real encoder mode
requires explicit ID/revision and first-C warmup, retaining resolved model
revision. Text retokenization/masked mean is the existing reproduction choice.

Logical entries have no tensors. Attention impact is unavailable, stays null,
and contributes zero only during policy scoring as in the existing logical
policy. CHU/PBR are not fabricated. Selected token reuse and savings are
SIMULATED; full M5 physical subsets can provide actual attention through the
Python runner's injected model engine. Logical smoke does not validate the full
paper policy's impact behavior or learned semantic encoder quality.
