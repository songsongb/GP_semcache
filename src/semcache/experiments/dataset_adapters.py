"""Explicit local JSON/JSONL, HF saved datasets, or opt-in HF loading.

Supported native schemas: MultiWOZ dialogue_id/turns or keyed log dialogues;
CoQA official data/story/questions/answers; SNIPS intent/utterance, text/label,
or official intents/*/utterances/data chunks. Unknown structures fail loudly.
"""
import hashlib
import json
from pathlib import Path

DATASETS = ('multiwoz', 'coqa', 'snips')
TRANSFORMATIONS = ('raw_query', 'paper_reproduction_v1')


def _parse_local_json(path, text):
    """Detect JSON first, then JSONL, independently of the filename suffix."""
    try:
        return json.loads(text), 'json'
    except json.JSONDecodeError as full_error:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as line_error:
                raise ValueError(
                    f'Cannot parse local dataset {path} (suffix={path.suffix!r}): '
                    f'full-JSON parse error: {full_error}; '
                    f'JSONL line {line_number} error: {line_error}') from line_error
        return records, 'jsonl'


def load_source(dataset, input_path=None, split='train', *, hf_id=None, hf_config=None,
                revision=None, allow_download=False):
    if dataset not in DATASETS:
        raise ValueError('Unsupported dataset')
    source = dict(dataset=dataset, source_split=split, requested_revision=revision,
                  source_version=None, hf_id=hf_id, hf_config=hf_config)
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f'Dataset unavailable: {path}. Supply a local JSON/JSONL or HF saved dataset.')
        source['input_path'] = str(path.resolve())
        if path.is_file():
            raw = path.read_bytes()
            source['source_sha256'] = hashlib.sha256(raw).hexdigest()
            data, source['source_format'] = _parse_local_json(path, raw.decode('utf-8'))
        else:
            try:
                from datasets import load_from_disk
            except ImportError as exc:
                raise RuntimeError('HF saved datasets require optional dependency .[datasets]') from exc
            data = load_from_disk(str(path))
            if hasattr(data, 'keys'):
                data = data[split]
            source['source_version'] = str(data.info.version)
            source['source_fingerprint'] = data._fingerprint
            data = list(data)
    else:
        if not allow_download or not hf_id:
            raise FileNotFoundError('No dataset supplied. Set --input-path (or dataset_source.input_path); external loading requires --allow-download and explicit --hf-id. No substitute dataset is used.')
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError('HF ingestion requires optional dependency .[datasets]') from exc
        data = load_dataset(hf_id, name=hf_config, split=split, revision=revision)
        source['source_version'] = str(data.info.version)
        source['source_fingerprint'] = data._fingerprint
        # Fingerprint identifies cached content; requested revision is not called resolved.
        source['resolved_revision'] = None
        data = list(data)
    if isinstance(data, dict) and split in data:
        data = data[split]
    if dataset == 'coqa' and isinstance(data, dict) and 'data' in data:
        source['source_version'] = data.get('version', source['source_version'])
        data = data['data']
    if dataset == 'snips' and isinstance(data, dict) and 'intents' in data:
        data = [dict(utterance=''.join(c['text'] for c in u['data']), intent=intent,
                     id=f'{intent}:{i}', original_metadata=u)
                for intent, body in data['intents'].items() for i, u in enumerate(body['utterances'])]
    if isinstance(data, dict):
        # A one-record JSONL file is also valid JSON. Preserve its record
        # container using existing schema keys, without relying on the suffix.
        record_keys = {'multiwoz': ('turns', 'log'), 'coqa': ('story',),
                       'snips': ('utterance', 'text')}
        if any(key in data for key in record_keys[dataset]):
            return [data], source
        if dataset != 'multiwoz':
            raise ValueError('Unsupported source object; expected records')
        data = [dict(value, dialogue_id=key) for key, value in data.items()]
    return list(data), source


def _turns(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and value:
        lengths = {len(v) for v in value.values()}
        if len(lengths) != 1:
            raise ValueError('Unequal column lengths')
        return [dict(zip(value, row)) for row in zip(*value.values())]
    raise ValueError('Unsupported turns schema')


def normalize(dataset, examples, split='train', transformation='raw_query'):
    if dataset not in DATASETS or transformation not in TRANSFORMATIONS:
        raise ValueError('Unknown dataset or transformation')
    records, dropped = [], []
    def emit(i, sid, query, reference, domain, conversation=None, **metadata):
        if not isinstance(query, str) or (reference is not None and not isinstance(reference, str)):
            raise ValueError(f'Invalid text at source {sid}')
        if not query.strip():
            dropped.append(dict(source_id=str(sid), source_example_index=i, reason='empty_query'))
            return
        if transformation == 'paper_reproduction_v1':
            if dataset == 'coqa':
                query = f'Story: {metadata["story"]}\nQuestion: {query}\nAnswer:'
            elif dataset == 'snips':
                query = f'Classify the intent of this utterance: {query}\nIntent:'
            else:
                query = f'User: {query}\nAssistant:'
        records.append(dict(dataset=dataset, source_split=split, source_id=str(sid),
            query_text=query, reference_text=reference, domain_or_intent=domain,
            conversation_id=conversation, metadata=dict(source_example_index=i, **metadata)))
    for i, ex in enumerate(examples):
        sid = str(ex.get('id', ex.get('dialogue_id', f'row:{i}')))
        if dataset == 'coqa':
            story = ex['story']
            questions, answers = _turns(ex['questions']), _turns(ex['answers'])
            answer_map = {str(a.get('turn_id', j+1)): a for j, a in enumerate(answers)}
            for j, q in enumerate(questions):
                if isinstance(q, str):
                    q = dict(input_text=q, turn_id=j+1)
                tid = str(q.get('turn_id', j+1))
                a = answer_map.get(tid)
                emit(i, f'{sid}:{tid}', q['input_text'], a['input_text'] if a else None,
                     ex.get('source'), sid, story=story, question=q, answer=a,
                     context_in_query=transformation == 'paper_reproduction_v1', turn_id=tid)
        elif dataset == 'multiwoz':
            legacy = 'log' in ex
            turns = _turns(ex['log'] if legacy else ex['turns'])
            def is_user(t, j):
                speaker = t.get('speaker')
                if speaker is None and not legacy:
                    raise ValueError('MultiWOZ turns need speaker; only official log schema uses alternating roles')
                return j % 2 == 0 if legacy else speaker in (0, 'USER', 'user')
            for j, turn in enumerate(turns):
                if not is_user(turn, j):
                    continue
                nxt = turns[j+1] if j+1 < len(turns) and not is_user(turns[j+1], j+1) else None
                tid = str(turn.get('turn_id', j))
                emit(i, f'{sid}:{tid}', turn.get('utterance', turn.get('text')),
                     nxt.get('utterance', nxt.get('text')) if nxt else None,
                     ex.get('domains', ex.get('services', turn.get('domain'))), sid,
                     turn_id=tid, dialogue_metadata={k:v for k,v in ex.items() if k not in ('turns','log')},
                     turn_metadata=turn, role_rule='alternating_official_log' if legacy else 'explicit_speaker')
        else:
            text = ex.get('utterance', ex.get('text'))
            intent = ex.get('intent', ex.get('label'))
            if intent is None:
                raise ValueError('SNIPS requires intent or label')
            emit(i, sid, text, str(intent), intent, original_metadata=ex,
                 reference_mode='intent_label')
    identities = [(r['source_split'], r['source_id']) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate source IDs; supply stable unique source identities')
    return records, dropped
