"""Static reuse opportunities, independent of cache/clustering policy.

Repeated counts are occurrences beyond the first. Cross-user counts require a
PRIOR occurrence from a different user. Token coverage is the union of such
windows in the target query; overlapping positions count once.
"""
from collections import Counter, defaultdict
from .manifest import canonical
from .user_assignment import assignment_counts


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*p
    lo = int(index)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(index-lo)


def static_reuse_opportunity(rows, tokenize, w=3):
    if w < 1:
        raise ValueError('w must be positive')
    seen = defaultdict(Counter)
    total = cross = within = coverage = tokens = 0
    for row in rows:
        ids = tokenize(row['query_text'])
        tokens += len(ids)
        covered = set()
        user = row['user_id']
        for start in range(len(ids)-w+1):
            key = tuple(ids[start:start+w])
            previous = seen[key]
            total += 1
            within += int(previous[user] > 0)
            if any(other != user and count for other, count in previous.items()):
                cross += 1
                covered.update(range(start, start+w))
            previous[user] += 1
        coverage += len(covered)
    return dict(total_windows=total, unique_windows=len(seen), repeated_windows=total-len(seen),
        repeated_across_users=cross, repeated_within_user=within,
        potential_cross_user_reused_token_coverage=coverage,
        potential_cross_user_token_coverage_ratio=coverage/tokens if tokens else 0,
        total_tokens=tokens, window_size=w, metric_scope='static_reuse_opportunity; prior occurrences, no cache policy',
        metric_source='MEASURED')


def validate_workload(rows, manifest, tokenizer=None, w=3):
    tokenize = (lambda text: tokenizer(text)['input_ids']) if tokenizer else str.split
    users = Counter(r['user_id'] for r in rows)
    counts = list(users.values()) + [0]*max(0, manifest['user_count']-len(users))
    assignment, conversations = assignment_counts(rows, manifest['user_count'])
    users_per_conversation = [len(u) for u in conversations.values()]
    split_count = sum(n > 1 for n in users_per_conversation)
    if (manifest.get('user_assignment_rule') == 'seeded_group_balanced'
            or manifest.get('assignment_unit') == 'conversation'):
        if any(not r.get('conversation_id') for r in rows):
            raise ValueError('Conversation-preserving assignment requires conversation_id on every record')
        if split_count:
            raise ValueError(f'Conversation-preserving assignment violated: '
                             f'conversations_assigned_to_multiple_users={split_count}; required 0')
    conversation_counts = list(assignment['per_user_conversation_counts'].values())
    domains = defaultdict(list)
    for r in rows:
        domains[canonical(r['domain_or_intent'])].append(r)
    lengths = [len(tokenize(r['query_text'])) for r in rows]
    return dict(query_count=len(rows), configured_users=manifest['user_count'], active_users=len(users),
        queries_per_user=dict(min=min(counts, default=0), median=percentile(counts, .5), max=max(counts, default=0)),
        total_conversation_count=len(conversations),
        conversations_assigned_to_multiple_users=split_count,
        conversation_split_ratio=split_count/len(conversations) if conversations else 0,
        users_per_conversation=dict(max=max(users_per_conversation, default=0),
            mean=sum(users_per_conversation)/len(conversations) if conversations else 0),
        conversations_per_user=dict(min=min(conversation_counts, default=0),
            median=percentile(conversation_counts, .5), max=max(conversation_counts, default=0)),
        **assignment,
        domain_distribution={k:len(v) for k,v in domains.items()},
        empty_query_count=sum(not r['query_text'].strip() for r in rows),
        empty_reference_count=sum(not (r['reference_text'] or '').strip() for r in rows),
        duplicate_exact_queries=len(rows)-len({r['query_text'] for r in rows}),
        tokenization=getattr(tokenizer, 'name_or_path', 'whitespace_proxy_v1; NOT model tokens'),
        token_lengths=dict(p50=percentile(lengths,.5), p90=percentile(lengths,.9), p95=percentile(lengths,.95), max=max(lengths, default=0)),
        static_reuse_opportunity=static_reuse_opportunity(rows, tokenize, w),
        by_domain={k:static_reuse_opportunity(v, tokenize, w) for k,v in domains.items()},
        metric_source='MEASURED', metric_scope='prepared workload audit', reported_hit_rate_definition=None)
