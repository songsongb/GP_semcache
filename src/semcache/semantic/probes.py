"""Select controls exclusively from complete-prompt tokenizations."""
from .subsequence import SubsequenceExtractor


def select_pair(ids_a, ids_b, window_size, case, special_ids=()):
    if case not in 'ABCD' or len(case) != 1:
        raise ValueError('Unknown probe case')
    windows = SubsequenceExtractor(window_size)
    for a in windows.extract(ids_a):
        for b in windows.extract(ids_b):
            if set(a.token_ids+b.token_ids).intersection(special_ids):
                continue
            same = a.token_ids == b.token_ids
            prefix = ids_a[:a.start] == ids_b[:b.start]
            position = a.start == b.start
            valid = {'A': same and prefix and position,
                     'B': same and not prefix and position,
                     'C': same and not prefix and not position,
                     'D': not same and prefix and position}[case]
            if valid:
                return a, b
    raise ValueError(f'No valid case {case} window of size {window_size}; adjust full prompts')


def construct_probes(tokenizer, window_size):
    # Search complete sentences; no separately tokenized phrase or inferred offset.
    target = ' I need a quiet hotel near the station for tonight.'
    pairs = {
        'A': [('Hello.'+target+' Thank you.', 'Hello.'+target+' Please help.')],
        'B': [(a+target, b+target) for a,b in [('Hello.', 'Great.'), ('Today.', 'Hello.'), ('Yes.', 'No.')]],
        'C': [('Hello.'+target, 'After a long journey this morning.'+target)],
        'D': [('Hello. I need a quiet hotel near the station.', 'Hello. We want some fresh fruit from the market.')],
    }
    result = []
    for case, candidates in pairs.items():
        for qa, qb in candidates:
            ia, ib = tokenizer(qa)['input_ids'], tokenizer(qb)['input_ids']
            try:
                a,b = select_pair(ia, ib, window_size, case, tokenizer.all_special_ids)
            except ValueError:
                continue
            result.append(dict(probe_case=case, query_a=qa, query_b=qb,
                               input_ids_a=ia, input_ids_b=ib, window_a=a, window_b=b))
            break
        else:
            raise ValueError(f'Cannot construct case {case} for this tokenizer/window size')
    return result
