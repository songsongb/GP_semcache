"""Deterministic transformations and serialization, with a separate timed sidecar."""
import json
from pathlib import Path
from .dataset_adapters import normalize
from .user_assignment import assign_users, stable_digest
from .manifest import canonical, sha256, now, write_manifest

CLUSTERS = {'multiwoz': 20, 'coqa': 40, 'snips': 30}


def serialize(rows):
    return ''.join(canonical(r)+'\n' for r in rows).encode('utf-8')


def build_workload(dataset, examples, source, *, seed=42, user_count=50,
                   assignment='seeded_round_robin', order='source_order',
                   transformation='raw_query', max_queries=None, tokenizer=None):
    if order not in ('source_order', 'seeded_shuffle'):
        raise ValueError('Unsupported workload ordering')
    if max_queries is not None and (not isinstance(max_queries, int) or isinstance(max_queries, bool) or max_queries < 1):
        raise ValueError('max_queries must be positive or null')
    rows, dropped = normalize(dataset, examples, source['source_split'], transformation)
    transformed_count = len(rows)
    users = assign_users(rows, user_count, seed, assignment)
    for i, (row, user) in enumerate(zip(rows, users)):
        row.update(user_id=user, original_order_index=i,
            query_token_length=len(tokenizer(row['query_text'])['input_ids']) if tokenizer else None,
            reproduction_transformation=dict(name=transformation, version='1', provenance='REPRODUCTION_CHOICE'))
    if order == 'seeded_shuffle':
        rows.sort(key=lambda r: (stable_digest(seed, 'order', dataset, r['source_split'], r['source_id']), r['source_id']))
    rows = rows[:max_queries]
    for i, row in enumerate(rows):
        row['global_query_index'] = i
    manifest = dict(dataset=dataset, source=source, source_split=source['source_split'],
        transformation_name=transformation, transformation_version='1', source_example_count=len(examples),
        transformed_query_count=transformed_count, output_query_count=len(rows),
        dropped_example_count=len({r['source_example_index'] for r in dropped}),
        dropped_query_count=len(dropped), dropped_reasons=dropped,
        limited_query_count=transformed_count-len(rows), max_queries=max_queries,
        user_assignment_rule=assignment, user_count=user_count, ordering_rule=order, seed=seed,
        sha256=sha256(serialize(rows)), cluster_target=CLUSTERS[dataset], creation_time=now(),
        tokenizer=getattr(tokenizer, 'name_or_path', None),
        provenance=dict(cluster_target='PAPER_DEFINED', user_count='PAPER_DEFINED' if user_count == 50 else 'REPRODUCTION_CHOICE',
            transformation='REPRODUCTION_CHOICE', ordering_rule='REPRODUCTION_CHOICE',
            user_assignment_rule='REPRODUCTION_CHOICE', seed='REPRODUCTION_CHOICE', counts='MEASURED',
            source='REPRODUCTION_CHOICE', max_queries='REPRODUCTION_CHOICE'),
        leakage_policy='single explicit source split; no train/eval repartition or adapter training')
    manifest['manifest_sha256'] = sha256(canonical({k:v for k,v in manifest.items() if k != 'creation_time'}).encode('utf-8'))
    return rows, manifest


def save_workload(path, rows, manifest):
    data = serialize(rows)
    if sha256(data) != manifest['sha256']:
        raise ValueError('Manifest does not match workload')
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Validate sidecar before writing workload.
    write_manifest(str(p)+'.manifest.json', manifest)
    p.write_bytes(data)


def read_workload(path):
    p = Path(path)
    data = p.read_bytes()
    manifest = json.loads(Path(str(p)+'.manifest.json').read_text(encoding='utf-8'))
    expected_manifest = sha256(canonical({k:v for k,v in manifest.items() if k not in ('creation_time','manifest_sha256')}).encode('utf-8'))
    if expected_manifest != manifest['manifest_sha256']:
        raise ValueError('Workload manifest SHA256 mismatch')
    if sha256(data) != manifest['sha256']:
        raise ValueError('Workload SHA256 mismatch')
    rows = [json.loads(line) for line in data.decode('utf-8').splitlines()]
    if len(rows) != manifest['output_query_count']:
        raise ValueError('Query count mismatch')
    for i, row in enumerate(rows):
        if row['global_query_index'] != i or row['dataset'] != manifest['dataset']:
            raise ValueError('Invalid workload identity/order')
    return rows, manifest
