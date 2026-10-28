"""Small explicit length groups; no Cartesian or unreviewed 2048+ sweep."""
import random

GROUPS={
    'A':[(n,n-64) for n in (128,256,512,1024)],
    'B':[(n,n-1) for n in (128,256,512,1024)],
    'C':[(128,0),(512,0),(1024,0),(512,256),(1024,768)],
    'pilot':[(128,64),(256,192),(512,448),(1024,960),(1024,1023)],
    'smoke':[(32,16),(32,31)],
}


def validate_case(length,prefix,max_positions=1024):
    if type(length) is not int or type(prefix) is not int or not 0<=prefix<length:
        raise ValueError('require integer 0<=P<L and S>=1')
    if length>min(1024,max_positions): raise ValueError('context exceeds B3 reviewed 1024 cap or model position limit')
    if length<4: raise ValueError('L must be >=4 for the inherited tokenizer fixture')
    return dict(L=length,P=prefix,S=length-prefix,prefix_ratio=prefix/length,
        suffix_ratio=(length-prefix)/length,cache_hit=prefix>0)


def cases(explicit=None,group='AB',phase='latency'):
    if explicit:
        result=[tuple(map(int,x.split(':'))) for x in explicit.split(',')]
        if any(len(x)!=2 for x in result): raise ValueError('cases require L:P')
    else:
        names=['smoke'] if phase=='smoke' else ['pilot'] if group=='pilot' else list(group)
        if any(n not in GROUPS for n in names): raise ValueError('unknown group')
        result=[c for n in names for c in GROUPS[n]]
    if not result or len(result)>13 or len(set(result))!=len(result): raise ValueError('need 1..13 distinct cases')
    for c in result: validate_case(*c)
    return result


def ordered_conditions(workloads,arm,seed,phase):
    arms=['off','on'] if arm=='both' else [arm]
    if any(a not in ('off','on') for a in arms): raise ValueError('invalid arm')
    result=[]
    # OFF completes capacity testing first. Other phases randomize case ordering.
    rng=random.Random(seed)
    for a in arms:
        block=[(n,p,a) for n,p in workloads]
        if phase=='pilot': block.sort()
        else: rng.shuffle(block)
        result.extend(block)
    return result
