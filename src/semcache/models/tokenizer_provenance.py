"""Offline tokenizer snapshot evidence. Never loads a model or contacts the Hub."""
from pathlib import PurePath
import re

INHERITED = 'same_repo_same_revision_inherited_from_model_snapshot'
INDEPENDENT_SOURCES = frozenset({'tokenizer.init_kwargs._commit_hash', 'tokenizer._commit_hash',
                                 'tokenizer_resolved_snapshot_path'})
TOKENIZER_ARTIFACT_FIELDS = ('tokenizer_revision', 'tokenizer_revision_source', 'tokenizer_source_id',
                            'model_revision_requested', 'tokenizer_revision_requested',
                            'tokenizer_revision_evidence')


def is_snapshot_commit(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{40}', value) is not None


def _path_snapshot(value, source_id):
    if not isinstance(value, str):
        return None
    parts = PurePath(value).parts
    repo = 'models--' + source_id.replace('/', '--')
    for i in range(len(parts)-2):
        if parts[i] == repo and parts[i+1] == 'snapshots' and is_snapshot_commit(parts[i+2]):
            return parts[i+2]
    return None


def resolve_tokenizer_provenance(tokenizer, *, model_source_id, tokenizer_source_id,
        model_requested_revision, tokenizer_requested_revision, model_resolved_revision,
        model_revision_explicit, tokenizer_revision_explicit):
    """Priority: retained tokenizer commit, resolved HF asset path, conditional inheritance.

    A model config's commit alone is NOT tokenizer evidence. It is used only by
    the final same-repository/explicit-same-requested-revision rule.
    """
    kwargs = getattr(tokenizer, 'init_kwargs', {}) or {}
    revision, source, evidence = None, 'unresolved', {}
    for candidate, candidate_source in ((kwargs.get('_commit_hash'), 'tokenizer.init_kwargs._commit_hash'),
            (getattr(tokenizer, '_commit_hash', None), 'tokenizer._commit_hash')):
        if is_snapshot_commit(candidate):
            revision, source = candidate, candidate_source
            evidence = dict(resolved_commit=candidate)
            break
    if revision is None:
        paths = []
        for name in ('tokenizer_file', 'vocab_file', 'merges_file', 'name_or_path'):
            for path in (kwargs.get(name), getattr(tokenizer, name, None)):
                commit = _path_snapshot(path, tokenizer_source_id)
                if commit:
                    paths.append((path, commit))
        commits = {commit for _, commit in paths}
        if len(commits) > 1:
            raise ValueError('Conflicting resolved tokenizer snapshot paths')
        if commits:
            revision, source = commits.pop(), 'tokenizer_resolved_snapshot_path'
            evidence = dict(resolved_commit=revision, resolved_asset_paths=sorted({p for p, _ in paths}))
    if (revision is None and model_source_id == tokenizer_source_id
            and model_revision_explicit is True and tokenizer_revision_explicit is True
            and isinstance(model_requested_revision, str) and model_requested_revision
            and model_requested_revision == tokenizer_requested_revision
            and is_snapshot_commit(model_resolved_revision)):
        revision, source = model_resolved_revision, INHERITED
        evidence = dict(model_source_id=model_source_id, tokenizer_source_id=tokenizer_source_id,
            model_requested_revision=model_requested_revision,
            tokenizer_requested_revision=tokenizer_requested_revision,
            model_resolved_revision=model_resolved_revision,
            model_revision_explicit=True, tokenizer_revision_explicit=True)
    return dict(tokenizer_revision=revision, resolved_tokenizer_revision=revision,
        tokenizer_revision_source=source, tokenizer_source_id=tokenizer_source_id,
        model_revision_requested=model_requested_revision,
        tokenizer_revision_requested=tokenizer_requested_revision, tokenizer_revision_evidence=evidence)


def tokenizer_artifact_fields(metadata):
    result = {key: metadata.get(key) for key in TOKENIZER_ARTIFACT_FIELDS}
    # Old loader metadata used tokenizer_revision for the request, not resolution.
    # Never reinterpret an old branch name as a resolved snapshot.
    result['tokenizer_revision'] = metadata.get('resolved_tokenizer_revision')
    return result


def validate_tokenizer_snapshot(row):
    revision, source = row.get('tokenizer_revision'), row.get('tokenizer_revision_source')
    source_id = row.get('tokenizer_source_id')
    evidence = row.get('tokenizer_revision_evidence') or {}
    if not isinstance(evidence, dict):
        raise ValueError('Tokenizer revision evidence must be an object')
    if not is_snapshot_commit(revision) or not isinstance(source_id, str) or not source_id:
        raise ValueError('Resolved tokenizer snapshot commit and source repository are required')
    if source in INDEPENDENT_SOURCES:
        if evidence.get('resolved_commit') != revision:
            raise ValueError('Missing independent tokenizer resolution evidence')
        if source == 'tokenizer_resolved_snapshot_path':
            paths = evidence.get('resolved_asset_paths') or []
            if not paths or any(_path_snapshot(p, source_id) != revision for p in paths):
                raise ValueError('Tokenizer snapshot paths do not prove the declared repository/commit')
    elif source == INHERITED:
        model_id, model_commit = row.get('model_id'), row.get('model_revision')
        requested = row.get('model_revision_requested')
        if (source_id != model_id or revision != model_commit or not requested
                or requested != row.get('tokenizer_revision_requested')
                or evidence.get('model_source_id') != model_id
                or evidence.get('tokenizer_source_id') != source_id
                or evidence.get('model_resolved_revision') != model_commit
                or evidence.get('model_requested_revision') != requested
                or evidence.get('tokenizer_requested_revision') != requested
                or evidence.get('model_revision_explicit') is not True
                or evidence.get('tokenizer_revision_explicit') is not True):
            raise ValueError('Tokenizer inheritance requires explicit same-repo/same-requested-revision evidence')
    else:
        raise ValueError('Missing or unsupported tokenizer revision provenance source')
    return dict(tokenizer_source_id=source_id, tokenizer_revision=revision,
                tokenizer_revision_source=source, tokenizer_revision_evidence=evidence)


def compare_snapshot_provenance(fresh, base_metadata, *, require_match=True):
    base = dict(model_id=base_metadata['model'], model_revision=base_metadata['resolved_model_revision'],
                **tokenizer_artifact_fields(base_metadata))
    fresh_tokenizer, base_tokenizer = validate_tokenizer_snapshot(fresh), validate_tokenizer_snapshot(base)
    result = dict(fresh=dict(model_source_id=fresh['model_id'], model_revision=fresh['model_revision'], **fresh_tokenizer),
        base_only=dict(model_source_id=base['model_id'], model_revision=base['model_revision'], **base_tokenizer),
        model_source_matches=fresh['model_id'] == base['model_id'],
        model_revision_matches=fresh['model_revision'] == base['model_revision'],
        tokenizer_source_matches=fresh_tokenizer['tokenizer_source_id'] == base_tokenizer['tokenizer_source_id'],
        tokenizer_revision_matches=fresh_tokenizer['tokenizer_revision'] == base_tokenizer['tokenizer_revision'])
    result['matched'] = all(result[k] for k in ('model_source_matches', 'model_revision_matches',
                                               'tokenizer_source_matches', 'tokenizer_revision_matches'))
    if require_match and not result['matched']:
        raise ValueError('Fresh M8.5 and base-only model/tokenizer snapshot provenance mismatch')
    return result
