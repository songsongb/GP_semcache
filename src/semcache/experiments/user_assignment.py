"""Stable SHA256-based logical assignment; no adapter training/personalization."""
import hashlib
import json


def stable_digest(*values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode('utf-8')).hexdigest()


def assign_users(records, user_count=50, seed=42, mode='seeded_round_robin'):
    if not isinstance(user_count, int) or isinstance(user_count, bool) or user_count < 1:
        raise ValueError('user_count must be a positive integer')
    if mode not in ('deterministic_hash', 'seeded_round_robin'):
        raise ValueError('Unsupported user assignment')
    permutation = sorted(range(user_count), key=lambda u: stable_digest(seed, 'user', u))
    return [f'user_{(permutation[i % user_count] if mode == "seeded_round_robin" else int(stable_digest(seed, r["dataset"], r["source_split"], r["source_id"]), 16) % user_count):03d}'
            for i, r in enumerate(records)]
