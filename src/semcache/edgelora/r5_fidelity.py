"""R5-I3F alignment and numerical measurements; no model/GPU imports."""
import math
from pathlib import Path
import struct

from .communication import digest_bytes, require
from .r3a_artifacts import unpack
from .r4c_session import validate_context

TARGETS = ('R5_03', 'R5_08', 'R5_01', 'R5_09')
SELECTED = frozenset((*TARGETS, 'R5_00', 'R5_04', 'R5_07'))


def gates():
    return {k: False for k in (
        'R5_I3F_CPU_TESTS_PASS', 'R5_I3F_GPU_FORENSICS_COMPLETE',
        'R5_I3F_CACHE_INTEGRITY_PASS', 'R5_I3F_FIRST_DIVERGENCE_IDENTIFIED',
        'R5_I3F_FROZEN_NUMERICAL_TOLERANCE_PASS', 'R5_I3F_INDEPENDENT_AUDIT_PASS',
        'R5_I3F_GENERATION_TOKEN_FIDELITY_PASS', 'NATIVE_PEFT_STRICT_PARITY_PASS',
        'correctness_qualified_speedup')}


def historical_id(rid):
    return rid[2:] if rid.startswith('N_') else rid


def align(a, b, *, kind, prefix=0, history_a=(), history_b=()):
    """Return comparable query ranges only after validating actual contexts."""
    validate_context(a); validate_context(b)
    require(a['policy'] == b['policy'], 'generation policies differ')
    require(a['pass_index'] == b['pass_index'], 'unmatched Decode step')
    step = a['pass_index']
    if kind == 'source_prefix':
        require(step == 0 and a['past_length'] == b['past_length'] == 0,
                'source prefix comparison requires independent full prefill')
        require(a['target_token_ids'] == b['target_token_ids'][:prefix] and
                len(a['target_token_ids']) == prefix > 0, 'source/target exact prefix differs')
        left, right = (0, prefix), (0, prefix)
    else:
        require(kind == 'matched' and a['target_token_ids'] == b['target_token_ids'],
                'matched target tokenization differs')
        require(list(history_a[:step]) == list(history_b[:step]) and
                len(history_a) >= step and len(history_b) >= step, 'unmatched Decode histories')
        if step:
            require(a['input_token_ids'] == b['input_token_ids'], 'unmatched Decode input')
            left = right = (0, 1)
        else:
            require(a['past_length'] == 0 and b['past_length'] == prefix and
                    b['input_token_ids'] == a['input_token_ids'][prefix:], 'incorrect suffix alignment')
            left, right = (prefix, a['input_length']), (0, b['input_length'])
    require(a['positions'][left[0]:left[1]] == b['positions'][right[0]:right[1]] and
            a['cache_positions'][left[0]:left[1]] == b['cache_positions'][right[0]:right[1]],
            'incorrect positions or cache positions')
    if kind == 'matched':
        require(a['attention_mask'] == b['attention_mask'], 'incorrect attention masks')
    else:
        require(a['attention_mask'] == b['attention_mask'][:prefix], 'prefix masks differ')
    return left, right


def tensor_blob(root, item):
    root = Path(root).resolve(); path = root / item['file']
    require(path.resolve().is_relative_to(root) and not path.is_symlink(), 'tensor evidence escape')
    blob = path.read_bytes()
    require(item['dtype'] in ('float16', 'float32'), 'unsupported tensor evidence dtype')
    require(all(type(n) is int and n > 0 for n in item['shape']), 'invalid tensor shape')
    require(len(blob) == math.prod(item['shape']) * (2 if item['dtype'] == 'float16' else 4) and
            digest_bytes(blob) == item['sha256'], 'corrupted tensor evidence')
    return blob


def slice_axis(blob, shape, axis, start, stop, element):
    require(0 <= axis < len(shape) and 0 <= start < stop <= shape[axis], 'tensor shape alignment differs')
    outer = math.prod(shape[:axis]); inner = math.prod(shape[axis+1:]) * element
    stride = shape[axis] * inner
    data = b''.join(blob[i*stride+start*inner:i*stride+stop*inner] for i in range(outer))
    shape = list(shape); shape[axis] = stop-start
    return data, shape


def select(root, item, query=None, key_stop=None):
    blob = tensor_blob(root, item); shape = list(item['shape']); e = 2 if item['dtype'] == 'float16' else 4
    if query is not None:
        blob, shape = slice_axis(blob, shape, item['token_axis'], *query, e)
    if key_stop is not None and item.get('key_axis') is not None:
        blob, shape = slice_axis(blob, shape, item['key_axis'], 0, key_stop, e)
    return blob, shape


def compare_bytes(reference, actual, dtype, *, allow_negative_infinity=False):
    require(len(reference) == len(actual) and reference, 'numerical tensor sizes differ')
    require(dtype in ('float16','float32'),'unsupported numerical dtype')
    if allow_negative_infinity:
        width=2 if dtype=='float16' else 4
        require(len(reference)%width==0,'misaligned masked score tensor')
        fmt='<e' if width==2 else '<f'
        a,b=([v[0] for v in struct.iter_unpack(fmt,blob)] for blob in (reference,actual))
        require(all(math.isfinite(v) or v==-math.inf for v in a+b),'unexpected nonfinite masked score')
    else:
        a,b=unpack(reference,dtype),unpack(actual,dtype)  # rejects NaN/Inf
    nonfinite_a=sum(not math.isfinite(v) for v in a);nonfinite_b=sum(not math.isfinite(v) for v in b)
    mismatch=sum((not math.isfinite(x) or not math.isfinite(y)) and x!=y for x,y in zip(a,b))
    finite_pairs=[(x,y) for x,y in zip(a,b) if math.isfinite(x) and math.isfinite(y)]
    all_a,all_b=a,b
    a=[x for x,y in finite_pairs];b=[y for x,y in finite_pairs]
    error = math.fsum((x-y)**2 for x, y in zip(a, b))
    na = math.fsum(x*x for x in a); nb = math.fsum(x*x for x in b)
    dot = math.fsum(x*y for x, y in zip(a, b))
    return dict(max_abs=max((abs(x-y) for x, y in zip(a,b)),default=None) if not mismatch else None,
                relative_l2=math.sqrt(error)/max(math.sqrt(na), 1e-30) if a and not mismatch else None,
                cosine_similarity=max(-1., min(1., dot/math.sqrt(na*nb))) if na and nb else None,
                bitwise_equal=reference == actual, finite=nonfinite_a==nonfinite_b==0,elements=len(all_a),
                reference_nonfinite=nonfinite_a,actual_nonfinite=nonfinite_b,nonfinite_pattern_mismatches=mismatch,
                error_scope='finite corresponding values; nonfinite pattern differences have undefined magnitude' if nonfinite_a or nonfinite_b else 'all values',
                differing_values=sum(x != y for x, y in zip(all_a,all_b)),
                reference_sha256=digest_bytes(reference), actual_sha256=digest_bytes(actual))


def compare(root_a, a, root_b, b, query_a=None, query_b=None, key_stop=None):
    require(a['dtype'] == b['dtype'] and a['layout'] == b['layout'] and
            a['token_axis'] == b['token_axis'] and a.get('key_axis') == b.get('key_axis'),
            'incompatible dtype/layout')
    av, ashape = select(root_a, a, query_a, key_stop)
    bv, bshape = select(root_b, b, query_b, key_stop)
    require(ashape == bshape, 'incompatible tensor shapes')
    require(a.get('allow_negative_infinity',False)==b.get('allow_negative_infinity',False),'incompatible finite-value contract')
    return dict(shape=ashape, dtype=a['dtype'], layout=a['layout'],
                **compare_bytes(av,bv,a['dtype'],allow_negative_infinity=a.get('allow_negative_infinity',False)))


# Parallel Q/K/V branches have the same causal rank; do not imply a dependency
# between their base and delta calculations just because one hook returned first.
ORDER = {'layer_input': 0, 'norm1': 1, 'projection_input': 2,
         **{f'{kind}_{role}': 3 for kind in ('base', 'delta') for role in 'qkv'},
         **{f'raw_{role}': 4 for role in 'qkv'},
         'attention_scores': 5, 'attention_masked_scores': 6,
         'attention_probabilities': 7, 'attention_context': 8,
         'attention_output': 9, 'attention_residual': 10, 'norm2': 11,
         'fc1': 12, 'activation': 13, 'ffn': 14, 'ffn_residual': 15, 'layer_output': 16,
         'final_hidden': 17, 'logits': 18}


def operation_order(name):
    if name.startswith('L') and '.' in name:
        layer, op = name.split('.', 1)
        return int(layer[1:]), ORDER.get(op, 100), op
    return 10**6, ORDER.get(name, 100), name


def first_divergence(records):
    divergent = [r for r in records if not r['metrics']['bitwise_equal']]
    if not divergent:
        return dict(status='NO_OBSERVED_DIVERGENCE',layer=None,operation=None)
    r = min(divergent, key=lambda x: (x['step'], operation_order(x['operation'])))
    layer, _, _ = operation_order(r['operation'])
    return dict(status='FIRST_OBSERVED_DIVERGENCE', layer=None if layer == 10**6 else layer,
                operation=r['operation'], step=r['step'], type=('NONFINITE_PATTERN_DIFFERENCE' if r['metrics'].get('nonfinite_pattern_mismatches') else
                    'VALUE_DIFFERENCE' if r['metrics']['max_abs'] else 'BIT_PATTERN_ONLY'),
                metrics=r['metrics'], causal_peers=[x['operation'] for x in divergent if x['step'] == r['step'] and
                    operation_order(x['operation'])[:2] == operation_order(r['operation'])[:2]],
                scope='earliest captured operation; unobserved operations are not exonerated')
