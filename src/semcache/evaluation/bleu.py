"""Explicit optional corpus SacreBLEU interface; protocol is a reproduction choice."""
import importlib.metadata


def compute_bleu(generated_texts, reference_texts, *, implementation, tokenizer,
                 smoothing, scope, effective_order, lowercase):
    if implementation != 'sacrebleu' or scope != 'corpus':
        raise ValueError('Supported explicit BLEU protocol: sacrebleu, corpus')
    if not tokenizer or smoothing not in ('none','floor','add-k','exp'):
        raise ValueError('Explicit tokenizer and smoothing required')
    if not isinstance(effective_order,bool) or not isinstance(lowercase,bool):
        raise ValueError('Explicit effective_order and lowercase required')
    if not generated_texts or len(generated_texts) != len(reference_texts) or any(not isinstance(s,str) for s in [*generated_texts,*reference_texts]):
        raise ValueError('Aligned generated and reference strings required')
    try:
        from sacrebleu.metrics import BLEU
    except ImportError as exc:
        raise RuntimeError('BLEU requires optional dependency .[evaluation]; no automatic install') from exc
    evaluator = BLEU(tokenize=tokenizer,smooth_method=smoothing,effective_order=effective_order,lowercase=lowercase)
    score = evaluator.corpus_score(generated_texts,[reference_texts])
    return dict(value=score.score,unit='BLEU points (0–100)',metric_source='MEASURED',
        metric_scope='corpus BLEU of supplied generations and references',
        implementation=implementation,version=importlib.metadata.version('sacrebleu'),
        tokenizer=tokenizer,smoothing=smoothing,scope=scope,effective_order=effective_order,
        lowercase=lowercase,signature=str(evaluator.get_signature()),protocol_provenance='REPRODUCTION_CHOICE',
        paper_bleu_claimed=False)
