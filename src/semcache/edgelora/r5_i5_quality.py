"""Reference quality only; token equality and frozen logit fidelity live elsewhere."""
from collections import Counter
import json
import statistics
from .communication import digest, require
from .r4c_metrics import qa, normalize, gold_score
from .r5_i5_data import LABELS, validate_sample

SCORING_VERSION = 'I5_QUALITY_V1'


def validate_scorers(names):
    if 'multi_news' in names:
        import importlib.metadata
        try:
            from rouge import Rouge
            require(importlib.metadata.version('rouge')=='1.0.1','MultiNews requires rouge==1.0.1')
        except ImportError as exc:
            raise ValueError('SCORER_UNAVAILABLE: install rouge==1.0.1 outside the GPU job') from exc


def state_value(s):
    return ' '.join(s.lower().split())


def score(sample, generated_text, template_version):
    validate_sample(sample); require(isinstance(generated_text, str) and template_version, 'actual output/template required')
    name = sample['dataset']; refs = sample['references']; invalid = False; parsed = None
    if name == 'snips':
        parsed = generated_text.strip(); invalid = parsed not in LABELS
        result = dict(metric='intent_accuracy', score=float(not invalid and parsed == refs[0]), gold=refs[0], prediction=parsed)
    elif name == 'multiwoz':
        def unique(pairs):
            d = {}
            for k, v in pairs:
                require(k not in d, 'duplicate state key'); d[k] = v
            return d
        try:
            parsed = json.loads(generated_text, object_pairs_hook=unique)
            require(isinstance(parsed, dict) and all(k in sample['ontology'] and isinstance(v, str) and v.strip()
                    for k, v in parsed.items()), 'invalid state slot/value')
        except (ValueError, TypeError):
            invalid = True; parsed = {}
        correct = sum(k in parsed and state_value(parsed[k]) in {state_value(v) for v in vs} for k, vs in refs.items())
        result = dict(metric='canonical_multiwoz22_joint_goal_accuracy', score=float(not invalid and correct == len(refs) == len(parsed)),
                      slot_tp=correct, slot_fp=len(parsed)-correct, slot_fn=len(refs)-correct, prediction=parsed, gold=refs)
    elif name == 'coqa':
        # CoQA leave-one-reference-out max, averaged over omitted annotators.
        scores = [qa(generated_text, r) for r in refs]
        if len(scores) == 1:
            f1 = scores[0]['f1']; em = scores[0]['exact_match']
        else:
            f1 = statistics.mean(max(s['f1'] for j, s in enumerate(scores) if j != i) for i in range(len(scores)))
            em = statistics.mean(max(s['exact_match'] for j, s in enumerate(scores) if j != i) for i in range(len(scores)))
        # Official CoQA punctuation/article-only normalized empty answers match.
        if not normalize(generated_text) and any(not normalize(r) for r in refs):
            f1 = em
        result = dict(metric='coqa_leave_one_out_token_f1', score=f1, exact_match=em)
    else:
        result = gold_score(name, generated_text, refs, fixture=sample['TEST_FIXTURE_ONLY'])
    return dict(result, dataset=name, dataset_version=sample['version'], original_sample_id=sample['id'],
        group_id=sample['group_id'], references_sha256=digest(refs), generated_text=generated_text,
        prompt_template_version=template_version, scoring_version=SCORING_VERSION, invalid_prediction=invalid,
        empty_output=not generated_text.strip(), TEST_FIXTURE_ONLY=sample['TEST_FIXTURE_ONLY'])


def aggregate(rows):
    require(rows and len({r['metric'] for r in rows}) == 1, 'cannot pool incompatible task metrics')
    out = dict(metric=rows[0]['metric'], samples=len(rows), score=statistics.mean(r['score'] for r in rows),
        invalid_rate=statistics.mean(r['invalid_prediction'] for r in rows), parsing_failures=sum(r['invalid_prediction'] for r in rows))
    if rows[0]['dataset'] == 'snips':
        table = {}
        for label in LABELS:
            tp = sum(r['prediction'] == label == r['gold'] for r in rows)
            fp = sum(r['prediction'] == label != r['gold'] for r in rows)
            fn = sum(r['gold'] == label != r['prediction'] for r in rows)
            table[label] = prf(tp, fp, fn)
        out.update(per_intent=table, macro_f1=statistics.mean(v['f1'] for v in table.values()))
    if rows[0]['dataset'] == 'multiwoz':
        out['slot_micro'] = prf(*(sum(r[k] for r in rows) for k in ('slot_tp', 'slot_fp', 'slot_fn')))
    if 'exact_match' in rows[0]: out['exact_match'] = statistics.mean(r['exact_match'] for r in rows)
    return out


def prf(tp, fp, fn):
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0., f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
