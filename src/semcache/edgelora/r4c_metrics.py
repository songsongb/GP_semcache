"""Gold quality and output fidelity are deliberately separate measurements."""
from collections import Counter
import importlib.metadata
import math
import re
import string

from .communication import require


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def qa(prediction, gold):
    p, g = normalize(prediction).split(), normalize(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    f1 = 2*common/(len(p)+len(g)) if common else 0.0  # LongBench empty/empty F1 is zero.
    return dict(f1=f1, exact_match=float(normalize(prediction) == normalize(gold)))


def diagnostic_rouge_l(prediction, gold):
    """Token LCS fixture diagnostic, NOT official rouge-package summary ROUGE-L."""
    p, g = prediction.split(), gold.split(); previous = [0]*(len(g)+1)
    for a in p:
        current = [0]
        for j, b in enumerate(g): current.append(previous[j]+1 if a == b else max(previous[j+1], current[-1]))
        previous = current
    return 2*previous[-1]/(len(p)+len(g)) if p and g else 0.0


def gold_score(dataset, output, answers, *, fixture=False):
    require(answers and all(isinstance(a, str) and a.strip() for a in answers), 'MISSING_GOLD_REFERENCE')
    if dataset in ('qasper', '2wikimqa'):
        scores = [qa(output, a) for a in answers]
        return dict(metric='normalized_answer_f1', score=max(s['f1'] for s in scores),
                    exact_match=max(s['exact_match'] for s in scores), implementation='LongBench-v1-normalize/Counter-max-gold', scale='0..1')
    require(dataset == 'multi_news', 'unsupported gold metric')
    try:
        from rouge import Rouge
    except ImportError as exc:
        require(fixture, 'MISSING_DEPENDENCY: rouge==1.0.1 required for official-compatible multi_news gold scoring')
        return dict(metric='TEST_FIXTURE_ONLY_TOKEN_LCS', score=max(diagnostic_rouge_l(output, a) for a in answers), official_compatible=False, scale='0..1')
    version = importlib.metadata.version('rouge'); require(version == '1.0.1', 'pin rouge==1.0.1 for reproducible LongBench ROUGE-L')
    scorer = Rouge(); scores = []
    for answer in answers:
        if not output.strip(): scores.append(0.0)
        else:
            try:scores.append(scorer.get_scores([output], [answer], avg=True)['rouge-l']['f'])
            except ValueError as exc:
                # LongBench assigns zero to empty ROUGE inputs. Unexpected
                # package errors remain visible rather than becoming scores.
                require('empty' in str(exc).lower(),'ROUGE_METRIC_FAILURE: '+str(exc));scores.append(0.0)
    return dict(metric='rouge-l', score=max(scores), package='rouge', package_version=version,
                normalization='rouge default sentence/whitespace handling; case preserved; no stemming', official_compatible=True, scale='0..1')


def numerical(reference, actual, reference_ids, actual_ids):
    """Aligned histories only: downstream divergence has its own accounting."""
    direct = []; history_matches = True; first = None
    for step, (a, b) in enumerate(zip(reference, actual)):
        if a != b and first is None: first = step
        require(len(a) == len(b) and a, 'logit vector dimension differs')
        require(all(math.isfinite(v) for v in a+b), 'nonfinite numerical evidence')
        if history_matches:
            error = sum((x-y)**2 for x,y in zip(a,b)); norm = sum(x*x for x in a)
            def logsoftmax(x):
                m=max(x); z=m+math.log(sum(math.exp(v-m) for v in x)); return [v-z for v in x]
            lp,lq=logsoftmax(a),logsoftmax(b)
            direct.append(dict(step=step,max_abs=max(abs(x-y) for x,y in zip(a,b)),relative_l2=math.sqrt(error)/max(math.sqrt(norm),1e-30),
                kl=sum(math.exp(x)*(x-y) for x,y in zip(lp,lq)),argmax_agreement=max(range(len(a)),key=a.__getitem__)==max(range(len(b)),key=b.__getitem__)))
        history_matches = history_matches and step < len(reference_ids) and step < len(actual_ids) and reference_ids[step] == actual_ids[step]
    token_first=next((i for i,(a,b) in enumerate(zip(reference_ids,actual_ids)) if a!=b),None)
    if token_first is None and len(reference_ids)!=len(actual_ids): token_first=min(len(reference_ids),len(actual_ids))
    return dict(first_divergent_logit_step=first, first_divergent_generated_token=token_first, aligned_history_steps=direct,
        downstream_divergent_steps=max(len(reference),len(actual))-len(direct), generation_histories_match=reference_ids==actual_ids,
        max_abs=max((s['max_abs'] for s in direct),default=None),relative_l2=max((s['relative_l2'] for s in direct),default=None),
        kl=max((s['kl'] for s in direct),default=None), argmax_agreement=all(s['argmax_agreement'] for s in direct) if direct else None)
