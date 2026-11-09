"""Bounded Phase 2 static w3 analysis. No encoder, policy, inference or HITs.

All counts describe token occurrences, independent of physical residency. Offset
mapping is conservative: unsupported or unverifiable offsets yield UNKNOWN.
"""
from collections import Counter, defaultdict
from .communication import require
from .r5_i5_workload import parts
from .r5_i5r_parallel import grams, lcp
from .r5_continuous_natural import content_identity
from .r5_multi_artifacts import sharing, resolve

CATEGORIES = ('shared_prefix_windows', 'same_position_nonprefix_windows',
              'shifted_position_nonprefix_windows', 'shifted_prefix_target_windows')
SCOPES = ('replay_inclusive', 'distinct_sample_only', 'first_occurrence_per_original_sample')
RELATIONS = ('EXPERIMENT_REPLAY', 'NATURALLY_IDENTICAL_DISTINCT_SAMPLE',
             'SAME_ORIGINAL_CONTEXT_DIFFERENT_REQUEST', 'DIFFERENT_CONTEXT_DIFFERENT_REQUEST')
MARKER = '\n[...context omitted...]\n'
NON_TEMPLATE = {'DOCUMENT_CONTEXT', 'QUESTION'}


def token_regions(prepared, tokenizer):
    ids = prepared['token_ids']; text = prepared['input_text']; sample = prepared['sample']
    unknown = ['UNKNOWN'] * len(ids)
    try:
        before, after = parts(sample)
        require(text.startswith(before) and text.endswith(after), 'prompt components differ')
        end = len(text)-len(after)
        middle = text[len(before):end]
        spans = [(0, len(before), 'TEMPLATE_INSTRUCTION')]
        if prepared['truncated']:
            context_ids = list(tokenizer.encode(sample['context'], add_special_tokens=False))
            keep = prepared['retained_context_tokens']; head, tail = (keep+1)//2, keep//2
            left = tokenizer.decode(context_ids[:head], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            right = tokenizer.decode(context_ids[-tail:], skip_special_tokens=False, clean_up_tokenization_spaces=False) if tail else ''
            require(middle == left+MARKER+right, 'retained context reconstruction differs')
            a = len(before)+len(left); b = a+len(MARKER)
            spans += [(len(before), a, 'DOCUMENT_CONTEXT'), (a, b, 'TEMPLATE_OMISSION'), (b, end, 'DOCUMENT_CONTEXT')]
        else:
            require(middle == sample['context'], 'original context differs')
            spans.append((len(before), end, 'DOCUMENT_CONTEXT'))
        question = sample['question']
        if sample['dataset'] == 'multi_news':
            prefix, suffix = '\n', '\nConcise summary:'
        else:
            prefix, suffix = '\nQuestion: ', '\nAnswer:'
        require(after == prefix+question+suffix, 'question/format components differ')
        qstart = end+len(prefix); qend = qstart+len(question)
        spans += [(end, qstart, 'TEMPLATE_FORMAT'), (qstart, qend, 'QUESTION'),
                  (qend, len(text), 'TEMPLATE_FORMAT')]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        require(list(encoded['input_ids']) == ids and len(encoded['offset_mapping']) == len(ids),
                'offset token IDs/length differ')
        labels = []
        for a, b in encoded['offset_mapping']:
            if not (type(a) is int and type(b) is int and 0 <= a < b <= len(text)):
                labels.append('UNKNOWN'); continue
            regions = [kind for lo, hi, kind in spans if lo <= a and b <= hi]
            labels.append(regions[0] if len(regions) == 1 else 'MIXED_BOUNDARY')
        return dict(token_labels=labels, status='VERIFIED_OFFSETS',
                    region_tokens=dict(Counter(labels)), offset_scope='original tokenizer, exact input IDs and prompt reconstruction')
    except (TypeError, AttributeError, NotImplementedError, KeyError, ValueError) as exc:
        return dict(token_labels=unknown, status='UNKNOWN', reason=str(exc),
                    region_tokens={'UNKNOWN': len(ids)})


def window_region(labels, p):
    kinds = set(labels[p:p+3])
    if 'UNKNOWN' in kinds or len(labels[p:p+3]) != 3: return 'UNKNOWN'
    return next(iter(kinds)) if len(kinds) == 1 else 'MIXED_BOUNDARY'


class Metric:
    def __init__(self):
        self.occurrences = 0; self.windows = set(); self.pairs = set(); self.regions = Counter()

    def add(self, count, positions, target, pair, origin):
        self.occurrences += count
        self.windows.update((target, p) for p in positions)
        if count: self.pairs.add(pair)
        self.regions[origin] += count

    def record(self):
        starts = defaultdict(list)
        for target, p in self.windows: starts[target].append(p)
        greedy = coverage = 0
        for positions in starts.values():
            end = -1; tokens = set()
            for p in sorted(positions):
                tokens.update(range(p, p+3))
                if p >= end: greedy += 1; end = p+3
            coverage += len(tokens)
        return dict(raw_matching_occurrence_pairs=self.occurrences,
                    unique_target_window_positions=len(self.windows),
                    nonoverlapping_target_token_coverage=3*greedy,
                    nonoverlapping_target_windows=greedy, target_token_union_coverage=coverage,
                    matching_request_pairs=len(self.pairs), region_occurrence_pairs=dict(sorted(self.regions.items())))


def relation(source, target):
    if source['sample_id'] == target['sample_id']: return RELATIONS[0]
    if content_identity(source) == content_identity(target): return RELATIONS[1]
    if source['prepared']['sample']['context'] == target['prepared']['sample']['context']: return RELATIONS[2]
    return RELATIONS[3]


def pair_windows(source, target, source_labels, target_labels):
    """Aggregate Cartesian occurrences by region without enumerating their product."""
    sg, tg = grams(source), grams(target); prefix = lcp(source, target)
    cells = {}
    for gram in sorted(sg.keys() & tg.keys()):
        source_positions = set(sg[gram])
        source_origins = Counter(window_region(source_labels, p) for p in sg[gram])
        for tp in tg[gram]:
            tr = window_region(target_labels, tp)
            same = tp in source_positions
            sr_same = window_region(source_labels, tp) if same else None
            for sr, total in source_origins.items():
                equal = int(same and sr == sr_same)
                counts = ((CATEGORIES[0] if tp+3 <= prefix else CATEGORIES[1], equal),
                          (CATEGORIES[3] if tp+3 <= prefix else CATEGORIES[2], total-equal))
                for category, count in counts:
                    if not count: continue
                    key = (category, sr, tr)
                    cell = cells.setdefault(key, [0, set()])
                    cell[0] += count; cell[1].add(tp)
    return cells


def analyze(session, registry, regions):
    inv = session['invocations']; first = {}; first_indices = set()
    require(0 < len(inv) <= 50 and len(regions) == len(inv) and
            all(3 <= len(i['task']['token_ids']) <= 512 and len(r['token_labels']) == len(i['task']['token_ids'])
                for i, r in zip(inv, regions)), 'bounded aligned Phase 2 static inputs required')
    for n, i in enumerate(inv):
        resolve(registry, i['task']['user_id'], i['task']['adapter_id'])
        if i['task']['sample_id'] not in first:
            first[i['task']['sample_id']] = n; first_indices.add(n)
    buckets = defaultdict(Metric); denominators = {s: Counter() for s in SCOPES}; pairs = []
    axis_denominators = {s: Counter() for s in SCOPES}
    for ti, target in enumerate(inv):
        t = target['task']
        for si, source in enumerate(inv[:ti]):
            s = source['task']; rel = relation(s, t)
            denied = sharing(registry, s['user_id'], t['user_id'], s['adapter_id'], t['adapter_id'])
            scopes = [SCOPES[0]]
            if s['sample_id'] != t['sample_id']: scopes.append(SCOPES[1])
            if si in first_indices and ti in first_indices: scopes.append(SCOPES[2])
            adapter = s['adapter_id']+'->'+t['adapter_id']
            for scope in scopes:
                denominators[scope]['ordered_request_pairs'] += 1
                denominators[scope][rel] += 1
                denominators[scope][denied or 'AUTHORIZED'] += 1
                for axis in ('aggregate', 'adapter:'+adapter, 'relationship:'+rel):
                    axis_denominators[scope][axis] += 1
            cells = pair_windows(s['token_ids'], t['token_ids'], regions[si]['token_labels'], regions[ti]['token_labels'])
            for scope in scopes:
                denominators[scope]['matching_request_pairs'] += bool(cells)
                denominators[scope]['no_exact_w3_request_pairs'] += not cells
            pair_metrics = defaultdict(Metric)
            for (category, sr, tr), (count, positions) in cells.items():
                origin = sr+'->'+tr
                slices = ['all']
                if s['adapter_id'] == t['adapter_id']: slices.append('same_adapter')
                if denied is None:
                    slices.append('authorized')
                    if s['user_id'] != t['user_id']: slices.append('authorized_cross_user')
                else: slices.append('rejected:'+denied)
                if sr == tr == 'DOCUMENT_CONTEXT':
                    slices.append('verified_context_to_context')
                    if denied is None: slices.append('authorized_verified_context_to_context')
                if sr in NON_TEMPLATE and tr in NON_TEMPLATE: slices.append('verified_non_template_to_non_template')
                if 'UNKNOWN' in (sr, tr): slices.append('unknown_region')
                if sr.startswith('TEMPLATE') or tr.startswith('TEMPLATE'): slices.append('known_template_involved')
                if sr == 'MIXED_BOUNDARY' or tr == 'MIXED_BOUNDARY': slices.append('mixed_boundary')
                pair_metrics[category].add(count, positions, ti, (si, ti), origin)
                for scope in scopes:
                    for axis in ('aggregate', 'adapter:'+adapter, 'relationship:'+rel):
                        for selection in slices:
                            buckets[(scope, axis, selection, category)].add(count, positions, ti, (si, ti), origin)
            pairs.append(dict(source_invocation_id=source['invocation_id'], target_invocation_id=target['invocation_id'],
                source_sample_id=s['sample_id'], target_sample_id=t['sample_id'], relationship=rel,
                adapter_pair=adapter, sharing_rejection=denied,
                same_original_context=s['prepared']['sample']['context'] == t['prepared']['sample']['context'],
                identical_request_content=content_identity(s) == content_identity(t),
                common_prefix_tokens=lcp(s['token_ids'], t['token_ids']),
                categories={c: pair_metrics[c].record() for c in CATEGORIES}))
    views = {}
    for scope in SCOPES:
        axes = {}
        for sc, axis, selection, category in sorted(buckets):
            if sc != scope: continue
            axes.setdefault(axis, {}).setdefault(selection, {})[category] = buckets[(sc, axis, selection, category)].record()
        for axis in ['aggregate', *('relationship:'+r for r in RELATIONS)]:
            axes.setdefault(axis, {}).setdefault('all', {})
            for c in CATEGORIES: axes[axis]['all'].setdefault(c, Metric().record())
        for selection in ('authorized', 'same_adapter', 'authorized_cross_user', 'verified_context_to_context',
                          'authorized_verified_context_to_context', 'verified_non_template_to_non_template',
                          'known_template_involved', 'unknown_region', 'mixed_boundary'):
            for c in CATEGORIES: axes['aggregate'].setdefault(selection, {}).setdefault(c, Metric().record())
        for r in RELATIONS: denominators[scope].setdefault(r, 0)
        views[scope] = dict(denominators=dict(denominators[scope]),
                           axis_ordered_pair_counts=dict(axis_denominators[scope]), axes=axes)
    return dict(dataset=session['dataset'], invocations=len(inv), distinct_samples=len(first), views=views, pairs=pairs,
        region_mapping=dict(Counter(r['status'] for r in regions)),
        scope='STATIC_ONLY; no residency, policy, model execution or physical authorization',
        physical_hits='NOT_MEASURED', measured_policy_outcomes='NOT_MEASURED', tensor_equivalence='NOT_MEASURED',
        attention_impact='NOT_MEASURED', e2e_latency='NOT_MEASURED')
