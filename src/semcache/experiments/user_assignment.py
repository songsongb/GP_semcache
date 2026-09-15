"""Stable SHA256-based logical assignment; no adapter training/personalization."""
import hashlib
import json
from collections import Counter, defaultdict

ASSIGNMENT_MODES = ('deterministic_hash', 'seeded_round_robin', 'seeded_group_balanced')


def stable_digest(*values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode('utf-8')).hexdigest()


def assign_users(records, user_count=50, seed=42, mode='seeded_round_robin'):
    if not isinstance(user_count, int) or isinstance(user_count, bool) or user_count < 1:
        raise ValueError('user_count must be a positive integer')
    if mode not in ASSIGNMENT_MODES:
        raise ValueError('Unsupported user assignment')
    if mode == 'seeded_group_balanced':
        groups = defaultdict(list)
        for i, record in enumerate(records):
            if not record.get('conversation_id'):
                raise ValueError('seeded_group_balanced requires a nonempty conversation_id')
            groups[conversation_key(record)].append(i)
        users = [''] * len(records)
        loads = [0] * user_count
        # Seeded group order; only record counts influence balancing, never text/reuse.
        for key in sorted(groups, key=lambda key: (stable_digest(seed, 'conversation', *key), key)):
            user = min(range(user_count), key=lambda u: (loads[u], u))
            for i in groups[key]:
                users[i] = f'user_{user:03d}'
            loads[user] += len(groups[key])
        return users
    permutation = sorted(range(user_count), key=lambda u: stable_digest(seed, 'user', u))
    return [f'user_{(permutation[i % user_count] if mode == "seeded_round_robin" else int(stable_digest(seed, r["dataset"], r["source_split"], r["source_id"]), 16) % user_count):03d}'
            for i, r in enumerate(records)]


def conversation_key(record):
    return (record.get('dataset'), record.get('source_split'), record['conversation_id'])


def assignment_counts(records, user_count):
    """Counts for emitted records, including inactive configured users."""
    queries = Counter({f'user_{u:03d}': 0 for u in range(user_count)})
    conversations = defaultdict(set)
    for record in records:
        queries[record['user_id']] += 1
        if record.get('conversation_id') is not None:
            conversations[conversation_key(record)].add(record['user_id'])
    per_user_conversations = Counter({user: 0 for user in queries})
    for users in conversations.values():
        per_user_conversations.update(users)
    return dict(per_user_record_counts=dict(sorted(queries.items())),
                per_user_conversation_counts=dict(sorted(per_user_conversations.items()))), conversations
