"""Deterministic, actually tokenized, zero-padding standalone-prefix workload."""
import importlib.metadata
import inspect
from pathlib import Path

from .communication import digest,digest_bytes,require
from .r4a_evidence import TEXTS
from .r4a_generation import generation_policy
from .r4b2_session import PREFIX_LENGTHS


def tokenizer_configuration(tokenizer=None,*,fixture=False):
    if fixture:return dict(kind='TEST_FIXTURE_ONLY_INTEGER_IDS',padding=False,add_special_tokens=False)
    backend=getattr(tokenizer,'backend_tokenizer',None)
    if backend is not None and callable(getattr(backend,'to_str',None)):
        model=dict(kind='FAST_BACKEND',serialized_backend_sha256=digest_bytes(backend.to_str().encode()))
    else:
        ranks=getattr(tokenizer,'bpe_ranks',None);encoder=getattr(tokenizer,'byte_encoder',None)
        require(ranks is not None and encoder is not None,'ARCHITECTURAL_BLOCKER: tokenizer configuration cannot be fully fingerprinted')
        model=dict(kind='SLOW_GPT2_BPE',merge_ranks_sha256=digest([[list(pair),rank] for pair,rank in sorted(ranks.items(),key=lambda x:x[1])]),
            byte_encoder_sha256=digest({str(k):v for k,v in encoder.items()}))
    file=inspect.getsourcefile(type(tokenizer));require(file is not None,'tokenizer implementation source unavailable')
    try:version=importlib.metadata.version('tokenizers')
    except importlib.metadata.PackageNotFoundError:version=None
    return dict(model=model,implementation_sha256=digest_bytes(Path(file).read_bytes()),class_name=type(tokenizer).__name__,
        module=type(tokenizer).__module__,tokenizers_version=version,
        settings={name:getattr(tokenizer,name,None) for name in ('padding_side','truncation_side','model_max_length','add_bos_token','add_eos_token','clean_up_tokenization_spaces')},
        special_tokens_map={k:str(v) for k,v in getattr(tokenizer,'special_tokens_map',{}).items()})


def make_suite(tokenizer=None,*,fixture=False,target_length=128,new_tokens=8,prefix_lengths=PREFIX_LENGTHS):
    require(target_length in (64,128) and new_tokens in (3,8,16) and (fixture or new_tokens in (8,16)),'bounded B2 length/policy required')
    require(prefix_lengths and len(set(prefix_lengths))==len(prefix_lengths) and all(p in PREFIX_LENGTHS and p<target_length for p in prefix_lengths),
        'unsupported exact prefix boundary')
    special=dict(bos_token_id=0,eos_token_id=6,pad_token_id=0) if fixture else {k:getattr(tokenizer,k) for k in ('bos_token_id','eos_token_id','pad_token_id')}
    policy=generation_policy(special,'fixed',new_tokens);prompts=[];conditions=[]
    def encode(text):
        out=tokenizer(text,add_special_tokens=False,truncation=False,padding=False)
        require(out['attention_mask']==[1]*len(out['input_ids']),'actual tokenizer introduced padding');return list(out['input_ids'])
    def prompt(name,ids,text,role):
        require(ids and len(ids)<=128 and all(type(t) is int and t>=0 for t in ids),'invalid controlled token IDs')
        if not fixture:require(not set(ids).intersection(special.values()),'controlled real IDs include a special/pad token')
        item=dict(sample_id=name,token_ids=list(ids),attention_mask=[1]*len(ids),positions=list(range(len(ids))),source_text=text,role=role,
            real_tokens=len(ids),padding_tokens=0,prompt_sha256=digest(dict(tokens=ids,mask=[1]*len(ids))))
        prompts.append(item);return item
    for index,p in enumerate(prefix_lengths):
        cid=f'P{p}_S{target_length-p}_G{new_tokens}'
        if fixture:
            shared=[1+(i+index)%5 for i in range(p)]
            a=shared+[1+(i+3)%5 for i in range(target_length-p)];b=shared+[1+(i+4)%5 for i in range(target_length-p)]
            texts=['TEST_FIXTURE_ONLY_INTEGER_SOURCE','TEST_FIXTURE_ONLY_TARGET_A','TEST_FIXTURE_ONLY_TARGET_B']
            unrelated=[5]*target_length
        else:
            tokenizer.truncation_side='right';source_ids=encode(TEXTS[index]);require(len(source_ids)>=target_length,'source text too short')
            shared=source_ids[:p];shared_text=tokenizer.decode(shared,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            require(encode(shared_text)==shared,'prefix text does not round-trip exact tokenizer IDs')
            texts=[shared_text,shared_text+' '+TEXTS[2],shared_text+' '+TEXTS[3]]
            av,bv=encode(texts[1]),encode(texts[2]);require(len(av)>=target_length and len(bv)>=target_length,'suffix source text too short')
            a,b=av[:target_length],bv[:target_length];unrelated=encode(TEXTS[3])[:target_length]
        require(len(shared)==p and len(a)==len(b)==len(unrelated)==target_length and a[:p]==b[:p]==shared and a[p:]!=b[p:],
            'tokenized targets do not share the declared exact prefix with distinct suffix IDs')
        require(unrelated[:p]!=shared,'unrelated control accidentally shares prefix')
        src=prompt(cid+'_P',shared,texts[0],'standalone_source');pa=prompt(cid+'_A',a,texts[1],'target');pb=prompt(cid+'_B',b,texts[2],'target')
        pu=prompt(cid+'_U',unrelated,'unrelated controlled text' if fixture else TEXTS[3],'unrelated')
        conditions.append(dict(id=cid,prefix_length=p,suffix_length=target_length-p,target_length=target_length,
            source=src['sample_id'],targets=[pa['sample_id'],pb['sample_id']],unrelated=pu['sample_id']))
    suite=dict(schema='r4b2_suite_v2',TEST_FIXTURE_ONLY=fixture,target_length=target_length,policy=policy,prompts=prompts,conditions=conditions,
        tokenizer_settings=dict(add_special_tokens=False,padding=False,truncation='explicit exact token prefix'),special_tokens=special,
        tokenizer_configuration=tokenizer_configuration(tokenizer,fixture=fixture))
    validate_suite(suite);return suite


def validate_suite(s):
    require(s['schema']=='r4b2_suite_v2' and s['prompts'] and len({p['sample_id'] for p in s['prompts']})==len(s['prompts']),'invalid B2 suite')
    lookup={p['sample_id']:p for p in s['prompts']}
    for p in s['prompts']:
        n=len(p['token_ids']);require(n==p['real_tokens'] and p['padding_tokens']==0 and p['attention_mask']==[1]*n and p['positions']==list(range(n)) and
            p['prompt_sha256']==digest(dict(tokens=p['token_ids'],mask=p['attention_mask'])),'suite token/mask/position/count identity differs')
    for c in s['conditions']:
        p=lookup[c['source']]['token_ids'];a,b=[lookup[k]['token_ids'] for k in c['targets']]
        require(len(p)==c['prefix_length'] in PREFIX_LENGTHS and len(a)==len(b)==c['target_length']==s['target_length'] and
            a[:len(p)]==b[:len(p)]==p and a[len(p):]!=b[len(p):] and c['suffix_length']==len(a)-len(p),'declared exact prefix/suffix invalid')


def schedule(suite):
    tasks=[]
    for c in suite['conditions']:
        p=c['source'];a,b=c['targets'];u=c['unrelated']
        for sample,mode,tag in ((p,'NO_REUSE','P_REF'),(a,'NO_REUSE','A_REF'),(b,'NO_REUSE','B_REF'),(p,'COLD_MISS','P_COLD'),
                (a,'PARTIAL_PREFIX_HIT','A_HIT'),(b,'PARTIAL_PREFIX_HIT','B_HIT'),(u,'NO_REUSE','UNRELATED'),(a,'PARTIAL_PREFIX_HIT','A_AFTER_U')):
            rid=f'B2_{c["prefix_length"]}_{tag}'
            tasks.append(dict(request_id=rid,sample_id=sample,mode=mode,condition_id=c['id'],prefix_length=c['prefix_length'],tag=tag,
                reference_request_id=f'B2_{c["prefix_length"]}_'+('P_REF' if sample==p else 'A_REF' if sample==a else 'B_REF' if sample==b else 'UNRELATED')))
    require(len(tasks)<=24 and len(tasks)*suite['policy']['max_new_tokens']<=384,'bounded B2 pilot budget exceeded')
    return tasks
