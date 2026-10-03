"""Epoch permutations of frozen B1 training rows; never select new examples."""
import random

from semcache.experiments.dataset_adapters import SNIPS_INTENTS
from semcache.models.task_adapters import USERS
from semcache.simulation.multi_user import digest

POLICY = 'deterministic per-epoch within-intent shuffle + class-balanced interleave'
SEED_RULE = 'int(SHA256(canonical JSON ["c6b_order_v1", global_seed, zero_based_user_index, one_based_epoch]), 16)'


def epoch_order(plan, user, epoch, seed):
    """Shuffle each class locally, then take one row/class in canonical order.

    Exhausted queues are skipped for previously supported shortfall plans.
    Balanced pilot cohorts have exactly one of each intent in every seven rows.
    The plan and Python's global RNG state are never mutated.
    """
    if user not in USERS or type(epoch) is not int or epoch < 1 or type(seed) is not int:
        raise ValueError('Expected a task user, integer seed and positive one-based epoch')
    all_ids = [r['source_id'] for r in plan['train_rows']]
    if len(set(all_ids)) != len(all_ids) or set(all_ids).intersection(plan['holdout_ids']):
        raise ValueError('Duplicated training rows or holdout leakage in frozen plan')
    queues = {label: [] for label in SNIPS_INTENTS}
    for row in plan['train_rows']:
        if row['user'] not in USERS or row['intent'] not in queues:
            raise ValueError('Invalid user/intent in frozen training plan')
        if row['user'] == user:
            queues[row['intent']].append(row['source_id'])
    derived_seed = int(digest(['c6b_order_v1', seed, USERS.index(user), epoch]), 16)
    rng = random.Random(derived_seed)
    for queue in queues.values():
        rng.shuffle(queue)
    ordered = [queue[index] for index in range(max(map(len, queues.values())))
               for queue in queues.values() if index < len(queue)]
    expected = [r['source_id'] for r in plan['train_rows'] if r['user'] == user]
    if len(ordered) != len(expected) or set(ordered) != set(expected):
        raise ValueError('Epoch ordering changed the frozen training cohort')
    return ordered, dict(epoch=epoch, derived_rng_seed=derived_seed,
                        row_count=len(ordered), ordered_row_ids_sha256=digest(ordered))


def order_manifest(plan, seed, epochs):
    """Record planned order hashes before model loading, including dry runs."""
    if type(epochs) is not int or epochs < 1:
        raise ValueError('Positive epoch count required')
    return dict(training_order_policy=POLICY, training_order_seed_derivation=SEED_RULE,
        training_order_intents=list(SNIPS_INTENTS),
        training_order_epochs={u: [epoch_order(plan, u, e, seed)[1]
                                   for e in range(1, epochs+1)] for u in USERS})
