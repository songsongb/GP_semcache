"""Whole blocks are the statistical units. Requests within blocks stay ordered."""
from .communication import digest,require

TINYBERT_REVISION='34707a33cd59a94ecde241ac209bf35103691b43'


def schedule(repetitions=1,seed=42,start='ABBA'):
    require(type(repetitions) is int and 1<=repetitions<=16 and start in ('ABBA','BAAB'),'invalid balanced design')
    blocks=[]
    for repetition in range(repetitions):
        order=start if repetition%2==0 else ('BAAB' if start=='ABBA' else 'ABBA')
        for position,condition in enumerate(order):
            blocks.append(dict(block_id=f'B{len(blocks):03d}',condition=condition,repetition=repetition,position=position,
                independent_pair=f'P{repetition:02d}_{position//2}',pair_order=order[(position//2)*2:(position//2)*2+2],seed=seed,
                state_initialization='FRESH_ES_AND_FRESH_MEASURED_UD',request_order=list(range(10))))
    result=dict(schema='R5_BALANCED_ORDER_V1',seed=seed,repetitions=repetitions,start=start,blocks=blocks,
        statistical_unit='independent adjacent A/B pair; requests and four-block schedules are not 10 independent samples')
    result['sha256']=digest(result);return result


def validate_schedule(value):
    require(value==schedule(value['repetitions'],value['seed'],value['start']),'order manifest changed');return value
