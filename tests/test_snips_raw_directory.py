"""Synthetic raw-layout fixtures; counts match the M6A ingestion contract."""
import hashlib
import json
from pathlib import Path

import pytest

from semcache.experiments.dataset_adapters import load_source, SNIPS_INTENTS
from semcache.experiments.workload import build_workload, save_workload, read_workload


@pytest.fixture
def raw_snips(tmp_path):
    root = tmp_path / '2017-06-custom-intent-engines'
    for index, intent in enumerate(SNIPS_INTENTS):
        directory = root / intent
        directory.mkdir(parents=True)
        # Deliberately duplicated records prove that ingestion does not deduplicate.
        utterance = {'data': [{'text': 'request '}, {'text': '🍕 café', 'entity': 'test'}]}
        count = 2000 if index < 6 else 1784
        raw = json.dumps({intent: [utterance] * count}, ensure_ascii=False).encode('utf-8')
        if intent == 'PlayMusic':
            raw = raw.replace('🍕'.encode('utf-8'), b'\xed\xa0\xbc\xed\xbd\x95')
        (directory / f'train_{intent}_full.json').write_bytes(raw)
        # Exact train/validate overlaps must remain in the workload.
        (directory / f'validate_{intent}.json').write_text(json.dumps({intent: [utterance] * 100}))
        (directory / f'train_{intent}.json').write_text('not a selected source')
    return root


def test_snips_raw_counts_encoding_hashes_and_manifest(raw_snips, tmp_path, monkeypatch):
    paths = sorted(raw_snips.glob('*/train_*_full.json'))
    before = {p.relative_to(raw_snips).as_posix(): p.read_bytes() for p in paths}
    original_read = Path.read_bytes
    def selected_only(path):
        assert path.name.startswith('train_') and path.name.endswith('_full.json')
        return original_read(path)
    with monkeypatch.context() as m:
        m.setattr(Path, 'read_bytes', selected_only)
        examples, source = load_source('snips', raw_snips, revision='supplied-repository-commit')
    assert len(examples) == source['source_count'] == source['expected_source_count'] == 13784
    assert len(source['intents']) == 7 and set(source['intents']) == set(SNIPS_INTENTS)
    assert set(e['intent'] for e in examples) == set(SNIPS_INTENTS)
    assert source['source_split'] == 'train_full'
    assert source['source_format'] == 'snips_raw_directory'
    assert source['repository_revision'] == source['requested_revision'] == 'supplied-repository-commit'
    assert source['repository_revision_provenance'] == 'user_provided'
    assert source['encoding_repair_occurred']
    assert source['encoding_repaired_files'] == ['PlayMusic/train_PlayMusic_full.json']
    assert all(e['utterance'] == 'request 🍕 café' for e in examples)
    assert len({e['id'] for e in examples}) == 13784
    assert len(source['source_files']) == 7
    for file in source['source_files']:
        assert file['sha256'] == hashlib.sha256(before[file['path']]).hexdigest()
        assert (raw_snips / file['path']).read_bytes() == before[file['path']]
        assert file['encoding_repaired'] == file['path'].startswith('PlayMusic/')
    rows, manifest = build_workload('snips', examples, source)
    again, repeated = build_workload('snips', examples, source)
    assert rows == again and manifest['sha256'] == repeated['sha256']
    assert manifest['manifest_sha256'] == repeated['manifest_sha256']
    assert len(rows) == manifest['source_example_count'] == 13784
    assert manifest['dropped_query_count'] == 0
    assert manifest['user_assignment_rule'] == 'seeded_round_robin'
    assert manifest['assignment_unit'] == 'query' and manifest['grouping_field'] is None
    assert manifest['user_count'] == 50 and len({r['user_id'] for r in rows}) == 50
    assert set(manifest['per_user_record_counts'].values()) == {275, 276}
    assert all(r['conversation_id'] is None and r['source_split'] == 'train_full' for r in rows)
    assert [r['domain_or_intent'] for r in rows] == [e['intent'] for e in examples]
    assert manifest['cluster_target'] == 30
    assert manifest['provenance']['cluster_target'] == 'PAPER_DEFINED'
    assert manifest['provenance']['user_assignment_rule'] == 'REPRODUCTION_CHOICE'
    assert manifest['source']['provenance']['source_split'] == 'REPRODUCTION_CHOICE'
    assert manifest['source']['provenance']['selection_rule'] == 'REPRODUCTION_CHOICE'
    output = tmp_path / 'workload.jsonl'
    save_workload(output, rows, manifest)
    assert read_workload(output) == (rows, manifest)


def test_snips_raw_strict_utf8(raw_snips):
    file = raw_snips / 'PlayMusic/train_PlayMusic_full.json'
    file.write_bytes(file.read_bytes().replace(b'\xed\xa0\xbc\xed\xbd\x95', '🍕'.encode('utf-8')))
    examples, source = load_source('snips', raw_snips, split='train_full')
    assert not source['encoding_repair_occurred'] and source['encoding_repaired_files'] == []
    assert all('🍕' in e['utterance'] for e in examples)
    assert source['repository_revision'] is None


@pytest.mark.parametrize('bad', [b'\xff', b'\xed\xa0\xbc'])
def test_snips_raw_rejects_non_cesu8_and_lone_surrogates(raw_snips, bad):
    file = raw_snips / 'PlayMusic/train_PlayMusic_full.json'
    file.write_bytes(file.read_bytes().replace(b'\xed\xa0\xbc\xed\xbd\x95', bad))
    with pytest.raises(ValueError, match='Invalid UTF-8/CESU-8.*PlayMusic'):
        load_source('snips', raw_snips)


@pytest.mark.parametrize('failure', ['missing', 'eighth_intent', 'count', 'wrong_intent', 'split'])
def test_snips_raw_contract_failures(raw_snips, failure):
    file = raw_snips / 'AddToPlaylist/train_AddToPlaylist_full.json'
    if failure == 'missing':
        file.rename(file.with_suffix('.backup'))
        match = 'exactly 7.*missing='
    elif failure == 'eighth_intent':
        directory = raw_snips / 'ExtraIntent'
        directory.mkdir()
        (directory / 'train_ExtraIntent_full.json').write_text('{}')
        match = 'exactly 7.*unexpected='
    elif failure == 'count':
        body = json.loads(file.read_text())
        body['AddToPlaylist'].pop()
        file.write_text(json.dumps(body))
        match = 'expected 13784, found 13783'
    elif failure == 'wrong_intent':
        file.write_text('{"OtherIntent": []}')
        match = 'expected object containing only'
    else:
        match = 'only source_split=train_full'
    with pytest.raises(ValueError, match=match):
        load_source('snips', raw_snips, split='validate' if failure == 'split' else 'train')


def test_snips_raw_cli(raw_snips, tmp_path):
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'cli.jsonl'
    subprocess.run([sys.executable, str(root / 'scripts/20_prepare_paper_workloads.py'),
                    '--dataset', 'snips', '--config', str(root / 'configs/paper/snips.yaml'),
                    '--input-path', str(raw_snips), '--revision', 'supplied-revision',
                    '--output', str(output)], check=True, capture_output=True, text=True)
    rows, manifest = read_workload(output)
    assert len(rows) == 13784 and manifest['source_split'] == 'train_full'
    assert manifest['source']['repository_revision'] == 'supplied-revision'
    assert manifest['source']['encoding_repaired_files'] == ['PlayMusic/train_PlayMusic_full.json']
    assert manifest['user_assignment_rule'] == 'seeded_round_robin'
