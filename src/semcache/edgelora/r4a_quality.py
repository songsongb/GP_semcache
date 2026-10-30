"""Aligned corpus SacreBLEU and distinct output-fidelity / real-gold scopes."""
import importlib.metadata

from semcache.evaluation.bleu import compute_bleu
from .communication import require

# Same convention as C6's BLEU, without importing any CacheGen experiment code.
BLEU=dict(implementation='sacrebleu',scope='corpus',tokenizer='13a',smoothing='exp',effective_order=False,lowercase=False)


def corpus(hypotheses,references):
    try:score=compute_bleu(hypotheses,references,**BLEU)
    except RuntimeError as exc:return dict(status='MISSING_DEPENDENCY',dependency='sacrebleu',reason=str(exc),score=None)
    return dict(status='MEASURED',score=score['value'],scale='0-100',n_references=1,**{k:v for k,v in score.items() if k!='value'})


def aligned_fidelity(candidates,references):
    require(candidates and len(candidates)==len(references),'aligned nonempty corpus required')
    ref={r['comparison_id']:r for r in references};require(len(ref)==len(references),'duplicate reference sample IDs')
    seen=set();hyps=[];refs=[];exact_tokens=exact_text=equal_tokens=total=changed=0
    for c in candidates:
        key=c['comparison_id'];require(key not in seen and key in ref,'duplicate/missing sample alignment');seen.add(key);r=ref[key]
        require(c['sample_id']==r['sample_id'] and c['prompt_sha256']==r['prompt_sha256'] and c['policy']==r['policy'],
            'BLEU prompt/policy/identity mismatch')
        require(isinstance(c['generated_text'],str) and isinstance(r['generated_text'],str),'text, not token IDs, required for BLEU')
        a,b=c['generated_token_ids'],r['generated_token_ids'];exact_tokens+=a==b;exact_text+=c['generated_text']==r['generated_text']
        changed+=a!=b or c['generated_text']!=r['generated_text']
        equal_tokens+=sum(x==y for x,y in zip(a,b));total+=max(len(a),len(b));hyps.append(c['generated_text']);refs.append(r['generated_text'])
    return dict(corpus_bleu=corpus(hyps,refs),samples=len(candidates),exact_generated_token_match_fraction=exact_tokens/len(candidates),
        exact_text_match_fraction=exact_text/len(candidates),token_positional_agreement=equal_tokens/total if total else 1.,
        changed_output_count=changed,scope='output fidelity against named model endpoint; not independent task quality')


def evaluate(outputs,native_outputs=None,gold=None,dataset_metric=None):
    measured=[r for r in outputs if r['phase']=='measured'];groups={};signatures={}
    for cell in sorted({r['cell_id'] for r in measured}):
        block=[r for r in measured if r['cell_id']==cell];ref=[r for r in block if r['mode']=='NO_REUSE'];modes={}
        for mode in ('COLD_MISS','WARM_HIT'):
            candidate=[r for r in block if r['mode']==mode];modes[mode]=dict(bleu_vs_distributed_full_recompute=aligned_fidelity(candidate,ref))
            if native_outputs:
                indexed={(r['cell_id'],r['sample_id']):r for r in native_outputs}
                references=[]
                for c in candidate:
                    require((cell,c['sample_id']) in indexed,'missing matched native generation')
                    references.append(dict(indexed[cell,c['sample_id']],comparison_id=c['comparison_id']))
                modes[mode]['bleu_vs_native_peft']=aligned_fidelity(candidate,references)
            else:modes[mode]['bleu_vs_native_peft']=dict(status='NOT_RUN_NO_MATCHED_NATIVE_GENERATION')
        groups[cell]=modes
    gold_result=dict(status='NOT_ESTIMABLE_NO_GOLD_REFERENCES')
    if gold is not None:
        require(gold and (dataset_metric is None or callable(dataset_metric)),'invalid dataset metric plugin')
        records={r['sample_id']:r for r in gold};require(len(records)==len(gold),'duplicate external gold IDs')
        scores={}
        for cell in sorted({r['cell_id'] for r in measured}):
            for mode in ('NO_REUSE','COLD_MISS','WARM_HIT'):
                selected=[r for r in measured if r['mode']==mode and r['cell_id']==cell];hyp=[];refs=[]
                for r in selected:
                    require(r['sample_id'] in records,'incomplete external gold corpus');g=records[r['sample_id']]
                    require(g['prompt_sha256']==r['prompt_sha256'] and isinstance(g['reference_text'],str) and g['dataset'] and g['revision'],
                        'invalid external gold provenance/alignment')
                    hyp.append(r['generated_text']);refs.append(g['reference_text'])
                scores[f'{cell}:{mode}']=dict(corpus_bleu=corpus(hyp,refs),dataset_metric=None if dataset_metric is None else dataset_metric(selected,gold))
        gold_result=dict(status='EXTERNAL_GOLD_SUPPLIED',modes=scores,task_quality_caveat='controlled untrained adapter; not trained task capability evidence')
    for cell,modes in groups.items():
        for mode,metrics in modes.items():
            for name,value in metrics.items():
                score=value.get('corpus_bleu',{})
                signatures[f'{cell}:{mode}:{name}']={k:score.get(k) for k in ('status','version','signature','tokenizer','smoothing','effective_order','lowercase','n_references','scale')}
    if gold_result['status']=='EXTERNAL_GOLD_SUPPLIED':
        for name,value in gold_result['modes'].items():
            score=value['corpus_bleu'];signatures[name+':bleu_vs_ground_truth']={k:score.get(k) for k in
                ('status','version','signature','tokenizer','smoothing','effective_order','lowercase','n_references','scale')}
    return dict(by_cell=groups,bleu_vs_ground_truth=gold_result,trained_adapter=False,
        paper_numeric_comparability='NOT_ESTABLISHED: corpus/model/decoding/reference scope differ from C6/C8 and paper',
        dataset_metric_plugin='callable(selected_generation_records, external_gold_records); use original dataset metric and record its version'),signatures
