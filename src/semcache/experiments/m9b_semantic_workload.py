"""Offline M9-B semantic artifact preparation; model imports live in the CLI only."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import tempfile

from semcache.models.tokenizer_provenance import validate_tokenizer_snapshot
from semcache.semantic.intent_clusterer import IntentClusterer
from semcache.simulation.multi_user import digest, read_workload

MODEL_ID = 'facebook/opt-2.7b'
MODEL_REVISION = '905a4b602cda5c501f1b3a2650a4152680238254'
ENCODER_ID = 'huawei-noah/TinyBERT_General_4L_312D'
ENCODER_REVISION = '34707a33cd59a94ecde241ac209bf35103691b43'
CLUSTERS = {'snips': 30, 'multiwoz': 20}
CURRENT_PREPARED_COUNTS = {'snips': 13784, 'multiwoz': 56776}
PRESERVED = ('dataset', 'source_id', 'global_query_index', 'original_order_index',
             'conversation_id', 'domain_or_intent', 'query_text')
SCHEMA = 'm9b_semantic_preparation_v1'


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def read_raw(path, dataset):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if dataset not in CLUSTERS or len(rows) < CLUSTERS[dataset]:
        raise ValueError('Need a supported dataset with at least C rows for first-C initialization')
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or row.get('dataset') != dataset:
            raise ValueError(f'Invalid raw dataset at row {i}')
        if not isinstance(row.get('source_id'), str) or not row['source_id']:
            raise ValueError(f'Need nonempty source_id string at row {i}')
        if type(row.get('global_query_index')) is not int or row['global_query_index'] != i:
            raise ValueError('Raw global_query_index must follow file order starting at zero')
        if not isinstance(row.get('query_text'), str) or not row['query_text'].strip():
            raise ValueError(f'Need nonempty query_text at row {i}')
    return rows


def order_hash(rows):
    return digest([[r.get(k) for k in ('dataset', 'source_id', 'global_query_index', 'original_order_index')]
                   for r in rows])


def assignment_source(dataset):
    return f'{SCHEMA}:{ENCODER_ID}@{ENCODER_REVISION}:masked_mean:first_k:buffered100:C{CLUSTERS[dataset]}'


def validate_metadata(metadata):
    if not isinstance(metadata, dict) or not isinstance(metadata.get('tokenizer'), dict) or not isinstance(metadata.get('semantic_encoder'), dict):
        raise ValueError('Missing tokenizer or semantic provenance object')
    token = metadata['tokenizer']
    validate_tokenizer_snapshot(dict(model_id=MODEL_ID, model_revision=MODEL_REVISION, **token))
    if token['tokenizer_source_id'] != MODEL_ID or token['tokenizer_revision'] != MODEL_REVISION:
        raise ValueError('Need the pinned OPT tokenizer snapshot')
    semantic = metadata['semantic_encoder']
    for key, expected in dict(model_id=ENCODER_ID, resolved_revision=ENCODER_REVISION,
                              pooling='masked_mean', max_length=512, backend='tinybert_huggingface').items():
        if semantic.get(key) != expected:
            raise ValueError(f'Missing or incompatible semantic provenance: {key}')
    if metadata.get('execution_provenance') not in ('MEASURED', 'TEST_STUB'):
        raise ValueError('Need explicit actual execution or TEST_STUB provenance')


def convert_rows(raw, dataset, tokenizer, encoder, metadata, batch_size=32):
    """Dependency-injected conversion; tests use marked stubs, CLI uses actual TinyBERT."""
    validate_metadata(metadata)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    c = CLUSTERS[dataset]
    if len(raw) < c:
        raise ValueError('Need at least C raw queries')

    def vectors():
        for offset in range(0, len(raw), batch_size):
            texts = [r['query_text'] for r in raw[offset:offset+batch_size]]
            batch = encoder.encode(texts)
            if len(batch) != len(texts):
                raise ValueError('Encoder row count mismatch')
            for vector in batch:
                if not vector or any(not math.isfinite(v) for v in vector):
                    raise ValueError('Invalid semantic vector')
                yield vector

    stream = vectors()
    anchors = list(itertools.islice(stream, c))
    clusterer = IntentClusterer(c, initialization='first_k', update_mode='buffered', update_interval=100)
    clusterer.initialize(anchors)  # Existing M7 semantics: initial counts are one.
    output = []
    for row, vector in zip(raw, itertools.chain(anchors, stream)):
        text = row['query_text']
        ids = list(tokenizer(text, add_special_tokens=True, truncation=False)['input_ids'])
        if not ids or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError('Invalid OPT tokenizer output')
        # Count before the existing encoder's truncation; no alternate semantic encoder.
        semantic_count = len(encoder.tokenizer(text, add_special_tokens=True, truncation=False)['input_ids'])
        diagnostics = clusterer.observe_with_diagnostics(vector)
        prepared = {**row, **{k: row.get(k) for k in PRESERVED}}
        prepared.pop('user_id', None)
        prepared.update(raw_user_id=row.get('user_id'), model_id=MODEL_ID,
            model_revision=MODEL_REVISION, model_revision_source='pinned_target_tokenizer_snapshot_no_OPT_model_loaded',
            tokenizer_id=f'{MODEL_ID}@{MODEL_REVISION}', **metadata['tokenizer'],
            tokenizer_special_token_policy='add_special_tokens=True; tokenizer checkpoint defaults',
            tokenizer_truncation=False, token_ids=ids, token_count=len(ids),
            semantic_assignment_source=assignment_source(dataset),
            semantic_encoder=dict(metadata['semantic_encoder']),
            semantic_execution_provenance=metadata['execution_provenance'],
            semantic_embedding_sha256=digest(vector), semantic_input_token_count=semantic_count,
            semantic_truncated=semantic_count > 512, semantic_truncation_max_length=512,
            cluster_id=diagnostics['cluster_id'], cluster_count=c, window_size=3,
            cluster_diagnostics=diagnostics, actual_attention_impact_available=False)
        output.append(prepared)
    # No terminal partial-buffer flush: it cannot affect emitted assignments.
    return output, dict(initial_centroids_sha256=digest(anchors), final_centroids_sha256=digest(clusterer.centroids),
        final_counts=clusterer.counts, pending_update_count=len(clusterer.pending),
        embedding_stream_sha256=digest([r['semantic_embedding_sha256'] for r in output]))


def validate_generated(source, output, manifest, *, smoke_rows=10, previous_manifest=None):
    """Model-free full validation, including the unchanged M9-B consumer boundary."""
    dataset = manifest['dataset']
    raw = read_raw(source, dataset)
    rows = read_workload(output, dataset)
    validate_metadata(manifest['metadata'])
    if len(rows) != len(raw) or manifest['row_count'] != len(raw):
        raise ValueError('Source/output row count mismatch')
    expected = manifest.get('expected_current_artifact_rows')
    if expected is not None and len(raw) != expected:
        raise ValueError('Current prepared artifact row count mismatch')
    if manifest['source_raw_sha256'] != file_hash(source) or manifest['output_semantic_sha256'] != file_hash(output):
        raise ValueError('Source/output file hash mismatch')
    for original, row in zip(raw, rows):
        if any(row.get(k) != original.get(k) for k in PRESERVED):
            raise ValueError('Source identity, text or order changed')
        if row.get('raw_user_id') != original.get('user_id') or 'user_id' in row:
            raise ValueError('Raw user must be provenance only')
        if type(row.get('token_count')) is not int or row['token_count'] != len(row['token_ids']):
            raise ValueError('token_count inconsistent')
        if not 0 <= row['cluster_id'] < CLUSTERS[dataset] or row.get('cluster_count') != CLUSTERS[dataset]:
            raise ValueError('Cluster outside dataset bounds')
        if row.get('model_revision') != MODEL_REVISION or row.get('tokenizer_id') != f'{MODEL_ID}@{MODEL_REVISION}':
            raise ValueError('Missing pinned target identity')
        if row.get('semantic_assignment_source') != assignment_source(dataset):
            raise ValueError('Missing semantic assignment provenance')
        if row.get('semantic_encoder') != manifest['metadata']['semantic_encoder']:
            raise ValueError('Semantic encoder metadata changed')
        if row.get('semantic_execution_provenance') != manifest['metadata']['execution_provenance']:
            raise ValueError('Semantic execution provenance mismatch')
        for key, value in manifest['metadata']['tokenizer'].items():
            if row.get(key) != value:
                raise ValueError('Tokenizer provenance mismatch')
        if (row.get('tokenizer_truncation') is not False or row.get('window_size') != 3
                or row.get('tokenizer_special_token_policy') != 'add_special_tokens=True; tokenizer checkpoint defaults'
                or row.get('semantic_truncation_max_length') != 512):
            raise ValueError('Token/window policy mismatch')
        count = row.get('semantic_input_token_count')
        if type(count) is not int or count < 1 or row.get('semantic_truncated') != (count > 512):
            raise ValueError('Invalid semantic truncation accounting')
    hashes = dict(query_order_sha256=order_hash(rows), token_sequence_stream_sha256=digest([r['token_ids'] for r in rows]),
                  semantic_assignment_stream_sha256=digest([r['cluster_id'] for r in rows]))
    if order_hash(raw) != hashes['query_order_sha256'] or any(manifest[k] != v for k, v in hashes.items()):
        raise ValueError('Stream/order hash mismatch')
    if previous_manifest is not None:
        for key in ('source_raw_sha256', 'query_order_sha256', 'token_sequence_stream_sha256',
                    'semantic_assignment_stream_sha256', 'output_semantic_sha256'):
            if previous_manifest[key] != manifest[key]:
                raise ValueError(f'Rerun differs: {key}')
    # Full input has already passed read_workload; verify an actual serialized prefix too.
    if type(smoke_rows) is not int or smoke_rows < 1:
        raise ValueError('smoke_rows must be positive')
    with tempfile.TemporaryDirectory() as directory:
        prefix = Path(directory)/'prefix.jsonl'
        prefix.write_text(serialize(rows[:smoke_rows]), encoding='utf-8')
        accepted = read_workload(prefix, dataset)
    return dict(valid=True, validated_rows=len(rows), simulator_prefix_rows=len(accepted), **hashes)


def serialize(rows):
    return ''.join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
                   for row in rows)


def prepare(source, output, dataset, tokenizer, encoder, metadata, *, batch_size=32,
            seed=42, expected_rows=None, previous_manifest=None, environment=None):
    source, output = Path(source), Path(output)
    sidecar = Path(str(output)+'.manifest.json')
    if output.resolve() == source.resolve() or sidecar.resolve() == source.resolve():
        raise ValueError('Never overwrite the raw workload')
    if output.exists() or sidecar.exists():
        raise ValueError('Output already exists; choose a new path for a rerun')
    raw_hash = file_hash(source)
    raw = read_raw(source, dataset)
    if expected_rows is not None and len(raw) != expected_rows:
        raise ValueError(f'Expected {expected_rows} rows in current prepared artifact; got {len(raw)}')
    rows, state = convert_rows(raw, dataset, tokenizer, encoder, metadata, batch_size)
    manifest = dict(schema=SCHEMA, dataset=dataset, source_path=str(source), row_count=len(rows),
        expected_current_artifact_rows=expected_rows, source_raw_sha256=raw_hash,
        output_semantic_sha256=hashlib.sha256(serialize(rows).encode('utf-8')).hexdigest(),
        query_order_sha256=order_hash(raw), token_sequence_stream_sha256=digest([r['token_ids'] for r in rows]),
        semantic_assignment_stream_sha256=digest([r['cluster_id'] for r in rows]),
        metadata=metadata, seed=seed, batch_size=batch_size, cluster_count=CLUSTERS[dataset], window_size=3,
        cluster_initialization='first C query vectors; N_c=1; replay all queries including anchors',
        cluster_assignment='Eq.8 minimum Euclidean distance; lowest cluster index wins ties',
        cluster_update='Eq.9 buffered 100-query updates; pending assignments retained; no final partial flush',
        cluster_update_interval=100, state=state, environment=environment or {},
        provenance=dict(semantic_execution=metadata['execution_provenance'], checkpoint='REPRODUCTION_CHOICE',
            tokenizer='REPRODUCTION_CHOICE', cluster_count='PAPER_DEFINED', window_size='PAPER_DEFINED',
            initialization='REPRODUCTION_CHOICE', scheduling='REPRODUCTION_CHOICE', pooling='REPRODUCTION_CHOICE'),
        raw_user_policy='raw_user_id provenance only; M9-B deterministically reassigns logical users',
        safe_reuse_claimed=False, paper_exact_preprocessing=False, opt_model_executed=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate before publishing output, and never touch raw source bytes.
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory)/'semantic.jsonl'
        temporary.write_text(serialize(rows), encoding='utf-8')
        manifest['validation'] = validate_generated(source, temporary, manifest, previous_manifest=previous_manifest)
        if file_hash(source) != raw_hash:
            raise ValueError('Raw input changed during preparation')
        with output.open('x', encoding='utf-8') as stream:
            stream.write(temporary.read_text(encoding='utf-8'))
    with sidecar.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return manifest
