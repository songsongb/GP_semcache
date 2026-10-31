"""Ordinary sequential requests. No cache labels or source IDs enter dispatch."""
from .communication import require
from .r4c_workload import policy

LABEL='CONTROLLED_REAL_TEXT_INTEGRATION_NOT_NATURAL_TRAFFIC'
P='The library opens at nine. Visitors can borrow books.'
Q='The garden grows green herbs. Rain waters the plants.'
R='The station serves local trains. Tickets are sold here.'


def workload(*,fixture=False):
    p,q,r=('abcdeabcde','qwertqwert','lmnop lmn') if fixture else (P,Q,R)
    texts=[p,p+' Please explain.',q,p+' Give details.',r,q+' Please explain.'*12,
        'Yesterday '+r+' A different opening.'*12,p,p+' Continue.',r+' Continue.']
    return [dict(request_id=f'R5_{i:02d}',user_id=('User_A','User_B','User_C')[i%3],adapter_id='user_a',
        input_text=t,dataset='TEST_FIXTURE_ONLY' if fixture else LABEL,sample_id=f'controlled_{i:02d}',request_sequence_number=i)
        for i,t in enumerate(texts)]


def lifecycle_capacity(tokenizer,raw,bytes_per_token):
    """Budget holds any pair of P/Q/R, but never all three, using observed IDs."""
    ids=[list(tokenizer.encode(raw[i]['input_text'],add_special_tokens=False)) for i in (0,2,4)]
    lengths=[len(x) for x in ids]
    for source,target in ((0,1),(0,3),(2,5),(7,8),(4,9)):
        a=list(tokenizer.encode(raw[source]['input_text'],add_special_tokens=False))
        b=list(tokenizer.encode(raw[target]['input_text'],add_special_tokens=False))
        require(len(b)>len(a) and b[:len(a)]==a,'TOKENIZER_BOUNDARY: controlled text failed actual prefix check')
    capacity=max(lengths[i]+lengths[j] for i in range(3) for j in range(i))*bytes_per_token
    require(sum(lengths)*bytes_per_token>capacity>0,'invalid lifecycle capacity')
    require(all(len(tokenizer.encode(raw[i]['input_text'],add_special_tokens=False))*bytes_per_token>capacity for i in (5,6)),
        'controlled recomputation requests must exceed lifecycle admission capacity')
    return capacity,dict(source_lengths=lengths,bytes_per_token=bytes_per_token,capacity_bytes=capacity,
        policy='maximum observed two-source tensor size; excludes third source',workload_label=LABEL)


def generation_policy(tokenizer,mode='fixed',budget=3):
    return policy({k:getattr(tokenizer,k) for k in ('bos_token_id','eos_token_id','pad_token_id')},mode,budget)
