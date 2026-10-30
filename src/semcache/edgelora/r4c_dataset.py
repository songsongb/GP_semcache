"""Offline, revision-pinned LongBench v1 inputs. No download code lives here."""
import json
from pathlib import Path

from .communication import digest, digest_bytes, require

DATASETS = ('qasper', '2wikimqa', 'multi_news')
ACQUISITION = ('Acquire LongBench v1 manually from https://huggingface.co/datasets/zai-org/LongBench '
               'outside the GPU job. Export official records as JSONL, or datasets.save_to_disk; '
               'record the immutable upstream revision and SHA-256 in a local registry. Never use synthetic gold for a real run.')


def file_hashes(path):
    path = Path(path).resolve()
    files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
    require(files, 'MISSING_DATASET: empty snapshot ' + str(path) + '; ' + ACQUISITION)
    return {p.name if path.is_file() else str(p.relative_to(path)): digest_bytes(p.read_bytes()) for p in files}


def load_dataset(name, spec, *, fixture=False):
    require(name in DATASETS, 'unsupported LongBench dataset')
    path = Path(spec['path']).resolve()
    require(path.exists(), f'MISSING_DATASET: expected {path} ({name}.jsonl or explicit HF save_to_disk snapshot); ' + ACQUISITION)
    require(isinstance(spec.get('revision'), str) and spec['revision'] and spec['revision'] not in ('main', 'latest'),
            'immutable dataset revision required')
    actual = file_hashes(path)
    require(spec.get('sha256') == actual, 'dataset file/snapshot SHA-256 differs from configured registry')
    kind = spec.get('format', 'jsonl')
    if kind == 'jsonl':
        raw = [(json.loads(line), digest_bytes(line.encode('utf-8'))) for line in path.read_text().splitlines() if line.strip()]
    else:
        require(kind == 'hf_saved_dataset', 'only JSONL or explicit local HF saved Dataset supported')
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ValueError('MISSING_DEPENDENCY: datasets required for local HF snapshot') from exc
        dataset = load_from_disk(str(path))  # Local only, never load_dataset / network.
        if spec.get('split') is not None:
            dataset = dataset[spec['split']]
        require(hasattr(dataset, 'column_names') and not isinstance(dataset.column_names, dict), 'explicit snapshot split required')
        raw = [(dict(r), digest(dict(r))) for r in dataset]
    rows = []; seen = set()
    for record, source_sha in raw:
        require(all(k in record for k in ('_id', 'input', 'context', 'answers', 'length')), 'invalid official LongBench record schema')
        require(isinstance(record['_id'], str) and record['_id'] and record['_id'] not in seen, 'duplicate/invalid original example ID')
        require(isinstance(record['input'], str) and isinstance(record['context'], str) and record['context'], 'invalid question/context')
        require(isinstance(record['answers'], list) and record['answers'] and all(isinstance(a, str) and a.strip() for a in record['answers']),
                'MISSING_GOLD_REFERENCE: all original nonempty answers required')
        require(type(record['length']) is int and record['length'] >= 0, 'invalid original prompt length')
        require(bool(record.get('TEST_FIXTURE_ONLY', False)) == fixture, 'synthetic versus real-data scope mismatch')
        seen.add(record['_id'])
        rows.append(dict(dataset=name, dataset_revision=spec['revision'], dataset_identifier=spec.get('identifier', 'zai-org/LongBench'),
                         source_path=str(path), source_record_sha256=source_sha, original=record, TEST_FIXTURE_ONLY=fixture))
    require(rows, 'MISSING_DATASET: no records; ' + ACQUISITION)
    return rows, dict(dataset=name, revision=spec['revision'], path=str(path), format=kind, files=actual,
                      records=len(rows), TEST_FIXTURE_ONLY=fixture)


def select(rows, count, seed, *, min_context_tokens=0, tokenizer=None, cohort='development', excluded_ids=(),input_lengths=None,min_input_tokens=0):
    """Hash-ranked IDs are stable across input ordering and Python versions."""
    require(type(count) is int and count > 0 and cohort in ('development', 'independent'), 'invalid selection/cohort')
    eligible = []; exclusions = []
    for row in rows:
        rid = row['dataset'] + ':' + row['original']['_id']
        n = len(tokenizer.encode(row['original']['context'])) if tokenizer is not None else None
        length=None if input_lengths is None else input_lengths[row['original']['_id']]
        reason = 'previous_cohort' if rid in excluded_ids else 'context_too_short' if min_context_tokens and (n is None or n < min_context_tokens) else 'input_too_short' if min_input_tokens and (length is None or length<min_input_tokens) else None
        if reason:
            exclusions.append(dict(id=rid, reason=reason, context_tokens=n,original_input_tokens=length))
        else:
            eligible.append(row)
    eligible.sort(key=lambda r: (digest([seed, r['dataset'], r['original']['_id']]), r['original']['_id']))
    require(len(eligible) >= count, f'INSUFFICIENT_ELIGIBLE_DATA: requested {count}, eligible {len(eligible)}; no silent cohort shrink')
    return eligible[:count], dict(seed=seed, cohort=cohort, requested=count, total=len(rows), eligible=len(eligible),min_context_tokens=min_context_tokens,min_input_tokens=min_input_tokens,
                                 selected_ids=[r['original']['_id'] for r in eligible[:count]], exclusions=exclusions,
                                 not_selected_ids=[r['original']['_id'] for r in eligible[count:]])
