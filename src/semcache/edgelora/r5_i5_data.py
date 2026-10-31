"""I5 local-only datasets. Acquisition and normalization happen before inference."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

from .communication import digest, require
from .r3a_artifacts import read, sha, write
from .r4c_dataset import file_hashes

DATASETS = ('coqa', 'snips', 'multiwoz', 'qasper', '2wikimqa', 'multi_news')
LABELS = ('AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
          'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent')
DST_SLOTS = set('''attraction-area attraction-name attraction-type bus-day bus-departure bus-destination bus-leaveat
hospital-department hotel-area hotel-bookday hotel-bookpeople hotel-bookstay hotel-internet hotel-name hotel-parking
hotel-pricerange hotel-stars hotel-type restaurant-area restaurant-bookday restaurant-bookpeople restaurant-booktime
restaurant-food restaurant-name restaurant-pricerange taxi-arriveby taxi-departure taxi-destination taxi-leaveat
train-arriveby train-bookpeople train-day train-departure train-destination train-leaveat'''.split())
VERSIONS = dict(coqa='1.0', snips='2017-06-custom-intent-engines', multiwoz='2.2',
                qasper='LongBench-v1', **{'2wikimqa': 'LongBench-v1', 'multi_news': 'LongBench-v1'})
SOURCES = dict(coqa='stanfordnlp/CoQA', snips='sonos/nlu-benchmark',
               multiwoz='budzianowski/multiwoz', qasper='zai-org/LongBench',
               **{'2wikimqa': 'zai-org/LongBench', 'multi_news': 'zai-org/LongBench'})


def validate_sample(s):
    require(s['dataset'] in DATASETS and s['version'] == VERSIONS[s['dataset']], 'unsupported dataset/version')
    require(all(isinstance(s[k], str) and s[k] for k in ('id', 'group_id', 'source_record_sha256')), 'missing original identity')
    require(all(isinstance(s[k], str) for k in ('context', 'question', 'history')), 'invalid prompt fields')
    refs = s['references']
    if s['dataset'] == 'multiwoz':
        require(isinstance(refs, dict) and isinstance(s.get('ontology'), list) and s['ontology'], 'missing DST gold/ontology')
        require(all(k in s['ontology'] and isinstance(v, list) and v and
                    all(isinstance(x, str) and x.strip() for x in v) for k, v in refs.items()), 'invalid DST gold')
    else:
        require(isinstance(refs, list) and refs and all(isinstance(a, str) and a.strip() for a in refs), 'MISSING_GOLD_REFERENCE')
        if s['dataset'] == 'snips':
            require(len(refs) == 1 and refs[0] in LABELS, 'invalid SNIPS label')
    require(type(s['TEST_FIXTURE_ONLY']) is bool, 'explicit data scope required')
    require(s['source_record_sha256']==digest(s['original']), 'original record hash differs')
    return s


def normalize(name, records, *, fixture=False, ontology=None):
    """Accept native upstream records; retain complete originals and original IDs."""
    require(name in DATASETS, 'unsupported dataset')
    rows = []
    def add(original, rid, group, context, question, history, refs, **extra):
        rows.append(validate_sample(dict(dataset=name, version=VERSIONS[name], id=str(rid), group_id=str(group),
            context=context, question=question, history=history, references=deepcopy(refs), original=deepcopy(original),
            source_record_sha256=digest(original), TEST_FIXTURE_ONLY=fixture, **extra)))
    for n, r in enumerate(records):
        require(bool(r.get('TEST_FIXTURE_ONLY', False)) == fixture, 'fixture/real dataset scope mismatch')
        if name == 'coqa':
            answers = {a['turn_id']: a for a in r['answers']}; history = []
            for q in r['questions']:
                tid = q['turn_id']; a = answers[tid]
                refs = [a['input_text']]
                for other in r.get('additional_answers', {}).values():
                    refs.extend(x['input_text'] for x in other if x['turn_id'] == tid)
                add(r, f'{r["id"]}:{tid}', r['id'], r['story'], q['input_text'], '\n'.join(history), refs, turn_id=tid)
                history.extend(['Question: ' + q['input_text'], 'Answer: ' + a['input_text']])
        elif name == 'snips':
            # Native intent-keyed files are flattened by read_native with a stable file+ordinal ID.
            add(r, r['id'], r['id'], '', r['text'], '', [r['intent']])
        elif name == 'multiwoz':
            require(ontology, 'MultiWOZ 2.2 schema.json ontology required')
            history = []; by_service = {}
            for turn in r['turns']:
                history.append(turn['speaker'] + ': ' + turn['utterance'])
                if turn['speaker'] != 'USER':
                    continue
                frames = turn['frames']; require(frames and all('state' in f for f in frames), 'missing USER belief state')
                for f in frames:
                    require('slot_values' in f['state'], 'unsupported MultiWOZ annotation format')
                    by_service[f['service']] = f['state']['slot_values']
                state = {k: v for d in by_service.values() for k, v in d.items()}
                add(r, f'{r["dialogue_id"]}:{turn["turn_id"]}', r['dialogue_id'], '', '', '\n'.join(history),
                    state, turn_id=turn['turn_id'], ontology=sorted(ontology))
        else:
            require(all(k in r for k in ('_id', 'context', 'input', 'answers', 'length')), 'invalid LongBench v1 schema')
            add(r, r['_id'], r['_id'], r['context'], r['input'], '', r['answers'])
    require(rows and len({r['id'] for r in rows}) == len(rows), 'empty/duplicate original sample IDs')
    return rows


def read_native(path, name, split, kind='json', ontology_path=None):
    path = Path(path).resolve(); require(path.exists(), 'MISSING_DATASET: ' + str(path) + '; see docs/R5_I5_INTEGRATED_EVALUATION.md')
    if kind == 'hf_saved_dataset':
        from datasets import load_from_disk
        ds = load_from_disk(str(path))
        if hasattr(ds, 'keys'):
            require(split in ds, 'explicit HF split missing'); ds = ds[split]
        records = [dict(r) for r in ds]
    else:
        files = [path] if path.is_file() else sorted(path.glob('*.json'))
        require(files, 'no native files')
        records = []
        for p in files:
            if kind == 'jsonl':
                data = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            else:
                require(kind == 'json', 'unsupported local format'); data = read(p)
            if name == 'coqa' and isinstance(data, dict): data = data['data']
            if name == 'snips' and isinstance(data, dict):
                for label, utterances in data.items():
                    require(label in LABELS, 'unknown SNIPS native label')
                    records.extend(dict(id=f'{p.name}:{i}', text=''.join(c['text'] for c in u['data']), intent=label,
                                        native_utterance=u) for i, u in enumerate(utterances))
            else:
                require(isinstance(data, list), 'native file must contain a record list'); records.extend(data)
    ontology = None
    if name == 'multiwoz':
        require(ontology_path and Path(ontology_path).is_file(), 'MISSING_ONTOLOGY: explicit MultiWOZ_2.2/schema.json required')
        available = {s['name'] for service in read(ontology_path) for s in service['slots']}
        require(DST_SLOTS <= available, 'MultiWOZ 2.2 ontology missing tracked slots')
        ontology = sorted(DST_SLOTS)
    return records, ontology


def prepare_dataset(name, path, output, *, revision, split, kind='json', ontology_path=None):
    from .r5_i5_evidence import fresh, seal
    require(re.fullmatch('[0-9a-f]{40,64}', revision or ''), 'immutable upstream revision or published archive SHA256 required')
    require(isinstance(split, str) and split, 'explicit split required')
    out = fresh(output, (path, ontology_path)); records, ontology = read_native(path, name, split, kind, ontology_path)
    rows = normalize(name, records, ontology=ontology)
    with (out/'samples.jsonl').open('x') as f:
        for row in rows: f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
    manifest = dict(schema='I5_DATASET_V1', dataset=name, source=SOURCES[name], version=VERSIONS[name], revision=revision,
        split=split, source_path=str(Path(path).resolve()), source_files=file_hashes(path),
        ontology_path=str(Path(ontology_path).resolve()) if ontology_path else None,
        ontology_sha256=sha(ontology_path) if ontology_path else None, samples_sha256=sha(out/'samples.jsonl'),
        samples=len(rows), selection='all normalized records; selection occurs in experiment preparation', TEST_FIXTURE_ONLY=False)
    write(out/'i5_dataset_manifest.json', manifest); seal(out); return manifest


def load_dataset(path, *, fixture=False):
    from .r5_i5_evidence import verify
    path = Path(path); verify(path); m = read(path/'i5_dataset_manifest.json')
    require(m['TEST_FIXTURE_ONLY'] == fixture and sha(path/'samples.jsonl') == m['samples_sha256'], 'dataset scope/hash differs')
    rows = [validate_sample(json.loads(x)) for x in (path/'samples.jsonl').read_text().splitlines()]
    require(rows and len(rows) == m['samples'] and len({s['id'] for s in rows}) == len(rows), 'dataset count/IDs differ')
    require(all(s['TEST_FIXTURE_ONLY'] == fixture and s['dataset'] == m['dataset'] and s['version'] == m['version'] for s in rows), 'mixed dataset/version/scope')
    return rows, m
