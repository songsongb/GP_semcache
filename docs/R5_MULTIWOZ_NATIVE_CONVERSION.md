# M-stage MultiWOZ 2.2 to I5 native JSONL

CPU-only converter: `scripts/159_convert_r5_multiwoz.py` and
`src/semcache/edgelora/r5_multiwoz_native.py`. The I5 loader/scorer, QKV operators,
GPU runners, cache policies, independent GPU auditors and historical experiments
are unchanged. **GPU_NOT_RUN. SERAPH conversion and I5 sealing are pending.**

The provided SERAPH source is
`/data/khuss/datasets/semcache/multiwoz_v22/test.json`. It is read explicitly as
JSONL, regardless of extension. It is not mounted in this workspace. The user
reports that its directory has no `schema.json`; no authentic schema, official
local Git checkout or independently verified schema pins were supplied here.
These remain prerequisites, not assumed installed assets. No data or schema was
downloaded.

## Conversion contract

The converter transposes equally sized turn columns in their original order.
It requires explicit integer speakers **0=USER, 1=SYSTEM**, consistent with the
existing M-stage mapping; it never infers roles from turn position. Confirm that
encoding against the existing export metadata before real conversion.

Each turn's `frames` must be a dictionary of parallel `service`, `state`, and
`slots` arrays. Every state is preserved, replacing only
`slots_values.{slots_values_name,slots_values_list}` with `slot_values`.
Alternatives, case, whitespace, ordering and types are preserved. I5 requires
nonempty lists of nonblank strings for state values: numeric/null/nested values
fail rather than being coerced or dropped. A USER frame must contain its actual
state; an explicitly empty slot-value map remains empty. Empty/null SYSTEM
states remain unchanged. Empty SYSTEM frames are permitted.

Per-frame slot annotations may be column dictionaries or existing row lists.
Column annotations become native row objects; all annotation fields and types
are retained. Dialogue/turn metadata and dialogue acts remain intact. Unknown
services, wrong-service or unsupported I5 state slots, duplicate service/slot
names, malformed columns/values, duplicate IDs and I5 USER sample-ID collisions
fail closed. No sorting, deduplication, label inference or service merging occurs.

Converted frame states are **per-service snapshots**, not accumulated labels.
The existing I5 `normalize()` replaces the state of a mentioned USER service,
retains unmentioned services, and ignores SYSTEM states when constructing gold.
The converter compares those actual I5 labels and histories with a separate
source-column accumulation check. The existing scorer performs its own
lowercase/whitespace matching; conversion does not normalize gold values.

## Authentic ontology prerequisite

Production conversion requires all of:

* A local checkout/object database of the official
  `https://github.com/budzianowski/multiwoz` repository.
* An independently verified **full 40-character upstream commit** containing
  `data/MultiWOZ_2.2/schema.json`.
* That exact schema file and its independently verified SHA-256.

The converter checks the official Git origin, pinned commit object, exact Git
blob bytes and SHA-256, service/slot structure, and coverage of I5 tracked slots.
Git replacement objects, lazy object fetching and transport protocols are
disabled during proof checks. Missing objects fail; there is no fetch fallback.
This is an offline proof against operator-verified pins, not a remote upstream
attestation. Merely changing a repository's origin or hashing an arbitrary
schema is not authentication. A fabricated list of I5 slots is never accepted
as production ontology. Test schemas are explicitly fixture-only and cannot
authorize a real conversion through the CLI.

If only the official Git object is already available locally, materialize it
into a new staging file without editing the dataset directory:

```bash
# Supply actual existing checkout and independently verified pins first.
: "${MWOZ_SCHEMA_REPO:?existing official local Git checkout}"
: "${MWOZ_SCHEMA_REV:?verified full upstream commit}"
: "${MWOZ_SCHEMA_SHA256:?verified schema blob SHA256}"
: "${MWOZ_SCHEMA_JSON:?fresh staging filename outside historical assets}"
( set -o noclobber
  GIT_NO_LAZY_FETCH=1 GIT_NO_REPLACE_OBJECTS=1 \
    git -c protocol.allow=never -C "$MWOZ_SCHEMA_REPO" \
    show "$MWOZ_SCHEMA_REV:data/MultiWOZ_2.2/schema.json" > "$MWOZ_SCHEMA_JSON"
)
```

If that checkout/object/pin is absent, stop and report it. This task does not
authorize acquiring data or substituting an unverified ontology.

## SERAPH conversion and CPU verification

Use the existing offline environment and reviewed source. All outputs must be
fresh. The schema variables identify verified inputs; the output paths below
are proposed new directories, not claims that existing assets are present.

```bash
cd /data/khuss/repos/GP_semcache
export MWOZ_SOURCE=/data/khuss/datasets/semcache/multiwoz_v22/test.json
export MWOZ_SOURCE_SHA256="$(sha256sum "$MWOZ_SOURCE" | cut -d ' ' -f1)"
export MWOZ_CONVERTED="$PWD/results/r5/i5/data/multiwoz22_mstage_native_01"

CUDA_VISIBLE_DEVICES='' python scripts/159_convert_r5_multiwoz.py convert \
  --source "$MWOZ_SOURCE" --source-sha256 "$MWOZ_SOURCE_SHA256" \
  --ontology "$MWOZ_SCHEMA_JSON" --schema-repository "$MWOZ_SCHEMA_REPO" \
  --schema-revision "$MWOZ_SCHEMA_REV" --schema-sha256 "$MWOZ_SCHEMA_SHA256" \
  --output-dir "$MWOZ_CONVERTED"

CUDA_VISIBLE_DEVICES='' python scripts/159_convert_r5_multiwoz.py verify \
  --conversion-dir "$MWOZ_CONVERTED"
```

The converter hashes the original file before/during/after reading. It never
edits it. It publishes `native.jsonl` only after all dialogue checks pass.
Failed attempts retain sealed FAILED metadata and a partial file, never an
approved native output; use a fresh attempt after resolving the error.

Keep the entire conversion directory. It contains:

* `native.jsonl`: ordered native dialogues, with original IDs/text/state.
* `schema.json`: byte-for-byte copy of the pinned authentic ontology.
* `records.jsonl`: original line/record hashes, dialogue ID, ordinal, native
  record hash, turn/USER/frame counts and conversion validation metadata.
* `conversion_manifest.json`: source SHA-256/path, ontology provenance/pins,
  converter hash, output hash, totals, scope and completion status.
* `i5_output_manifest.json`: integrity inventory of this conversion artifact;
  this alone is **not** an I5 prepared dataset.

Verification rechecks inventory and source/schema/output hashes, compares each
saved native record with its original columns, and checks actual I5 labels,
histories, identities and aggregate counts. It does not run inference or claim
real-data correctness before these checks execute on SERAPH.

## Existing I5 normalization and sealing

Only after conversion verification succeeds, use the existing command with an
explicit **file path** and `--format jsonl`. Do not pass the whole conversion
directory, which also contains provenance JSONL.

```bash
# Recover the actual dataset upstream revision/archive pin from existing
# M-stage provenance. This is distinct from the ontology commit; do not invent it.
: "${MWOZ_DATA_REVISION:?verified original MultiWOZ2.2 dataset revision/archive pin}"
export MWOZ_I5_PREPARED="$PWD/results/r5/i5/data/multiwoz22_mstage_i5_01"
CUDA_VISIBLE_DEVICES='' python scripts/118_prepare_r5_i5.py dataset \
  --dataset multiwoz --source "$MWOZ_CONVERTED/native.jsonl" --format jsonl \
  --revision "$MWOZ_DATA_REVISION" --split test \
  --ontology "$MWOZ_CONVERTED/schema.json" --output-dir "$MWOZ_I5_PREPARED"

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python - <<'PY'
import os
from semcache.edgelora.r5_i5_data import load_dataset
rows, manifest = load_dataset(os.environ['MWOZ_I5_PREPARED'])
print({'dataset': manifest['dataset'], 'version': manifest['version'],
       'user_samples': len(rows), 'dialogues': len({s['group_id'] for s in rows}),
       'GPU_status': 'GPU_NOT_RUN'})
PY
```

The counts must match the conversion manifest's USER/dialogue counts. Then use
the existing Phase 3-A discovery, original I5 workload preparation and zero-skip
CPU approval. Complete dialogue boundaries and all resource/scorer gates still
apply. Conversion success does not establish a compatible five-request block,
native CPU approval, physical reuse or GPU readiness.

## Local tests and status

Run `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest
test_r5_multiwoz_native test_r5_i5 -v`.

Local result: **43 passed, one skipped, zero failures** (44 tests). All 15
converter tests passed. The sole skip is the unrelated installed ROUGE package
check. The sealed CPU log, source hashes and prerequisite report are in
`results/r5/i5/multiwoz_native_conversion_cpu_20261109_01/`.

Fixtures explicitly match the observed column layout and test dialogue/turn/
USER counts, speaker mapping, values/alternatives/types, annotation preservation,
per-service replacement versus accumulation, SYSTEM-state exclusion, actual
I5 normalization/scoring, provenance hashes, fresh outputs, malformed input,
duplicate identities and missing/corrupt/untrusted schema proofs. Fixture
expected outputs are never evidence of real SERAPH conversion correctness.

**SERAPH_VALIDATION_PENDING; I5_SEALING_NOT_RUN; PHASE3A_CPU_APPROVAL_NOT_RUN;
GPU_NOT_RUN.** Missing prerequisites are the locally inaccessible raw source,
an authentic pinned schema with its existing official Git object database,
and the dataset revision needed for subsequent I5 provenance. No GPU/Slurm job,
final latency benchmark, historical modification, data download or git push was
performed.
