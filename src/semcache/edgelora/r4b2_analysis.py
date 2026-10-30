"""Strict bitwise outcomes and separately quantified shape-change diagnostics."""
import math
from pathlib import Path
import statistics

from .communication import digest_bytes,require
from .r3a_artifacts import THRESHOLDS,check_parity,read,unpack,write,write_csv
from .r3a_evidence import jsonl
from .r4a_quality import aligned_fidelity
from .r4b2_prefix_runtime import flags


def tensor_data(root,s):
    root=Path(root).resolve();p=(root/s['file']).resolve()
    require(p.is_relative_to(root),'tensor evidence escaped output');blob=p.read_bytes()
    require(digest_bytes(blob)==s['sha256'] and len(blob)==math.prod(s['shape'])*{'float16':2,'float32':4}[s['dtype']],
        'tensor evidence SHA/shape/length differs');unpack(blob,s['dtype']);return blob


def token_slice(blob,s,start,stop):
    shape=s['shape'];require(len(shape) in (3,4) and 0<=start<=stop<=shape[-2],'invalid evidence token slice')
    w=shape[-1]*{'float16':2,'float32':4}[s['dtype']];n=shape[-2];heads=1 if len(shape)==3 else shape[1]
    return b''.join(blob[(h*n+start)*w:(h*n+stop)*w] for h in range(heads))


def raw_head_bytes(blob,shape,heads,width):
    _,p,h=shape;require(h%heads==0,'head layout invalid');d=h//heads
    return b''.join(blob[(i*h+j*d)*width:(i*h+(j+1)*d)*width] for j in range(heads) for i in range(p))


def tensor_metrics(a,b,dtype):
    x,y=unpack(a,dtype),unpack(b,dtype);require(len(x)==len(y) and x,'matched nonempty tensors required')
    delta=[u-v for u,v in zip(x,y)];den=sum(v*v for v in y)
    return dict(max_abs=max(abs(d) for d in delta),relative_l2=math.sqrt(sum(d*d for d in delta)/max(den,1e-30)),
        unequal_elements=sum(u!=v for u,v in zip(x,y)),elements=len(x))


def fidelity(es,rows,tokens):
    es=Path(es);by={r['request_id']:r for r in rows};steps={r['request_id']:[t for t in tokens if t['request_id']==r['request_id']] for r in rows};checks=[]
    for r in rows:
        if r['mode']=='NO_REUSE':continue
        ref=by[r['reference_request_id']];p=r['reused_prefix_tokens'];rs,bs=steps[r['request_id']],steps[ref['request_id']]
        a=(es/r['logits_file']).read_bytes();b=(es/ref['logits_file']).read_bytes();width=r['logits_shape'][-1]
        require(r['policy']==ref['policy'] and r['target_token_ids']==ref['target_token_ids'] and r['prompt_sha256']==ref['prompt_sha256'],'unmatched B2 comparator')
        logit_checks=[];operations=[]
        for i,(s,t) in enumerate(zip(rs,bs)):
            aa,bb=a[i*width*4:(i+1)*width*4],b[i*width*4:(i+1)*width*4]
            logit_checks.append(dict(step=i,same_history=r['generated_token_ids'][:i]==ref['generated_token_ids'][:i],bitwise_equal=aa==bb,
                original_threshold_metrics=check_parity(unpack(aa,'float32'),unpack(bb,'float32'),[1,1,width])))
            offset=p if i==0 else 0
            for key,spec in s['qkv'].items():
                base=t['qkv'][key];aa=tensor_data(es,spec);bb=token_slice(tensor_data(es,base),base,offset,base['shape'][1])
                operations.append(dict(step=i,layer=int(key.split('.')[0]),operation='suffix_raw_'+key.split('.')[1],bitwise_equal=aa==bb,
                    same_history=logit_checks[-1]['same_history'],metrics=tensor_metrics(aa,bb,spec['dtype'])))
            if i==0:
                for key,spec in s['components'].items():
                    base=t['components'][key];aa=tensor_data(es,spec);bb=token_slice(tensor_data(es,base),base,offset,base['shape'][1])
                    operations.append(dict(step=i,layer=int(key.split('.')[0]),operation='suffix_'+key.split('.')[1],bitwise_equal=aa==bb,
                        metrics=tensor_metrics(aa,bb,spec['dtype'])))
            for layer,(sp,tp) in enumerate(zip(s['kv'],t['kv'])):
                for role in 'kv':
                    exact=sp[role]['sha256']==tp[role]['sha256'];item=dict(step=i,layer=layer,operation='full_actual_past_'+role,bitwise_equal=exact,
                        same_history=logit_checks[-1]['same_history'])
                    if i==0:item['metrics']=tensor_metrics(tensor_data(es,sp[role]),tensor_data(es,tp[role]),sp[role]['dtype'])
                    else:item['numeric_snapshot']='hash-only decode KV; next-token logit metrics retained'
                    operations.append(item)
        order={name:i for i,name in enumerate(('suffix_raw_q','suffix_raw_k','suffix_raw_v','full_actual_past_k','full_actual_past_v',
            'suffix_attention','suffix_ffn','suffix_layer'))}
        first=next((o for o in sorted(operations,key=lambda o:(o['step'],o['layer'],order[o['operation']])) if not o['bitwise_equal']),None)
        strict=(a==b and r['generated_token_ids']==ref['generated_token_ids'] and r['generated_text']==ref['generated_text'] and all(o['bitwise_equal'] for o in operations))
        checks.append(dict(request_id=r['request_id'],reference_request_id=ref['request_id'],strict_bitwise_pass=strict,
            generated_ids_equal=r['generated_token_ids']==ref['generated_token_ids'],text_equal=r['generated_text']==ref['generated_text'],serialized_logits_equal=a==b,
            changed_generated_tokens=sum(u!=v for u,v in zip(r['generated_token_ids'],ref['generated_token_ids'])),
            per_step_logits=logit_checks,operations=operations,first_divergent_layer_operation=first,
            interpretation='shape/scheduling changed; unmatched-history contrasts are downstream propagation, not independent numerical equivalence'))
    return checks


def analyze(es,suite,rows,tokens,*,fixture,audit):
    es=Path(es);checks=fidelity(es,rows,tokens);indexed={r['request_id']:r for r in rows};pairs=[]
    for r in rows:
        if r['mode'] not in ('PARTIAL_PREFIX_HIT','COLD_MISS'):continue
        ref=indexed[r['reference_request_id']];d={}
        for metric in ('ttft_ms','e2e_wall_ms','prefill_ms','decode_ms'):
            d[metric]=dict(reference=ref[metric],candidate=r[metric],improvement=ref[metric]-r[metric],
                improvement_percent=100*(ref[metric]-r[metric])/ref[metric] if ref[metric] else None)
        for stage in ('prefill_accounting','decode_accounting'):
            for direction in ('es_to_ud_wire_bytes','ud_to_es_wire_bytes','hidden_payload_bytes','delta_payload_bytes'):
                a,b=ref[stage][direction],r[stage][direction]
                d[f'{stage}.{direction}']=dict(reference=a,candidate=b,reduction_bytes=a-b,reduction_percent=100*(a-b)/a if a else None)
        pairs.append(dict(request_id=r['request_id'],reference_request_id=ref['request_id'],mode=r['mode'],condition_id=r['condition_id'],deltas=d,
            scope='cold standalone-prefix versus same-prefix full reference' if r['mode']=='COLD_MISS' else 'same complete target/policy versus distributed full NO_REUSE'))
    summaries={}
    for c in suite['conditions']:
        selected=[p for p in pairs if p['condition_id']==c['id'] and p['mode']=='PARTIAL_PREFIX_HIT']
        summaries[c['id']]=dict(comparisons=len(selected),unique_references=len({p['reference_request_id'] for p in selected}),repeat_blocks=1,
            mean_ttft_improvement_ms=statistics.mean(p['deltas']['ttft_ms']['improvement'] for p in selected),
            mean_e2e_improvement_ms=statistics.mean(p['deltas']['e2e_wall_ms']['improvement'] for p in selected),
            qualification='INSUFFICIENT_INDEPENDENT_PAIRS: fewer than eight per condition/two blocks; A-after-U reuses A reference')
    candidates=[r for r in rows if r['mode']!='NO_REUSE'];refs=[dict(indexed[r['reference_request_id']],comparison_id=r['request_id']) for r in candidates]
    quality=aligned_fidelity([dict(r,comparison_id=r['request_id']) for r in candidates],refs)
    quality=dict(bleu_vs_distributed_no_reuse=quality,bleu_vs_native_peft=dict(status='NOT_RUN_DEFERRED'),
        bleu_vs_ground_truth=dict(status='NOT_ESTIMABLE_NO_GOLD_REFERENCES'),trained_adapter=False)
    measured=not fixture and audit['status']=='PASS';strict=all(c['strict_bitwise_pass'] for c in checks)
    decision=dict(execution_status='COMPLETE',R4B2_PHYSICAL_E2E_PASS=measured,R4B2_EXACT_PREFIX_MATCH_PASS=measured,
        R4B2_PREFIX_KV_INTEGRITY_PASS=measured,R4B2_SUFFIX_ONLY_EXECUTION_PASS=measured,R4B2_MULTI_TARGET_REUSE_PASS=measured,
        R4B2_STRICT_BITWISE_FIDELITY_PASS=measured and strict,R4B2_NUMERICAL_FIDELITY_DIAGNOSTIC=dict(status='MEASURED' if measured else 'TEST_FIXTURE_ONLY',
            strict_bitwise=strict,comparisons=len(checks),strict_passes=sum(c['strict_bitwise_pass'] for c in checks),original_thresholds=THRESHOLDS),
        R4B2_TTFT_BENEFIT_OBSERVED=False,R4B2_E2E_BENEFIT_OBSERVED=False,fixture_checks_passed=fixture,
        fixture_strict_bitwise=fixture and strict,by_condition=summaries,**flags(fixture))
    return decision,checks,pairs,quality


def export(output,decision,checks,pairs,quality):
    write(output/'r4b2_decision.json',decision);write(output/'r4b2_fidelity.json',checks);write(output/'r4b2_latency_pairs.json',pairs);write(output/'r4b2_quality.json',quality)
    write_csv(output/'r4b2_latency_contrasts.csv',[dict(request_id=p['request_id'],reference_request_id=p['reference_request_id'],condition_id=p['condition_id'],mode=p['mode'],
        metric=k,**v) for p in pairs for k,v in p['deltas'].items()])
