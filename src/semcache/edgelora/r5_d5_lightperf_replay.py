"""Explicit cyclic replay of validated original NATURAL requests; no HIT selection."""
from collections import Counter
from copy import deepcopy

from .communication import digest, require
from .r5_d5_lightperf_workload import COUNTS, original_pool

MODE = 'CONTINUOUS_REPLAY'
SCHEMA = 'D5_CONTINUOUS_REPLAY_V1'


def replay_enabled(plan):
    mode = plan.get('workload_mode', 'NATURAL')
    require(mode in ('NATURAL', MODE), 'unknown Light workload mode')
    if mode == MODE:
        require(plan.get('enable_continuous_replay') is True, 'explicit continuous replay plan opt-in required')
    else:
        require(not plan.get('enable_continuous_replay', False), 'replay flag requires CONTINUOUS_REPLAY mode')
    return mode == MODE


def replay_stream(workload, dataset, count, *, enable_continuous_replay=False,
                  baseline_tasks=None, baseline_seed=None, strict_pilot_tasks=None):
    require(enable_continuous_replay is True, 'explicit --enable-continuous-replay required')
    require(type(count) is int and count in COUNTS, 'unsupported replay scale')
    blocks, pool = original_pool(workload, dataset)
    require(pool, 'EMPTY_ORIGINAL_REPLAY_POOL')
    # All sources are used in original order before cycling. No ranking by HIT,
    # length, user, adapter or measured performance is permitted.
    pool_records = []
    for block_index, block in enumerate(workload['blocks']):
        if not any(block is b for b in blocks):
            continue
        for task_index, task in enumerate(block['tasks']):
            pool_records.append(dict(original_request_id=task['request_id'], original_block_id=block['block_id'],
                original_block_index=block_index, original_task_index=task_index,
                original_task_sha256=digest(task), sample_id=task['sample_id'],
                user_id=task['user_id'], adapter_id=task['adapter_id']))
    residual = count % len(pool)
    if residual:
        chosen_groups = {t['group_id'] for t in pool[:residual]}
        require(not any(t['group_id'] in chosen_groups for t in pool[residual:]), 'GROUP_SPLIT_AT_REPLAY_BOUNDARY')
    tasks, occurrences = [], []
    source_ids = {t['request_id'] for t in pool}
    for ordinal in range(count):
        cycle, index = divmod(ordinal, len(pool))
        task = deepcopy(pool[index])
        if cycle:
            rid = f'D5REPLAY_{ordinal:03d}_{digest(pool_records[index])[:16]}'
            require(rid not in source_ids, 'replay execution ID collides with original request')
            task['request_id'] = rid
        occurrence = dict(pool_records[index], request_id=task['request_id'], session_ordinal=ordinal,
                          pool_index=index, cycle_index=cycle, occurrence_index=cycle,
                          original_request_sequence_number=pool[index]['request_sequence_number'])
        tasks.append(task)
        occurrences.append(occurrence)
    require(len({t['request_id'] for t in tasks}) == count, 'duplicate replay execution request ID')
    reasons = []
    if baseline_tasks is None:
        reasons.append('NO_FROZEN_LIGHT_FIVE_REQUEST_BASELINE')
    else:
        if tasks[:5] != baseline_tasks:
            reasons.append('FIRST_FIVE_TASK_PAYLOAD_OR_ORDER_DIFFERS')
        if baseline_seed != workload['seed']:
            reasons.append('BASELINE_SEED_DIFFERS')
    unique = len({t['sample_id'] for t in tasks})
    return dict(schema=SCHEMA, workload_mode=MODE, dataset=dataset, request_count=count,
                tasks=tasks, tasks_sha256=digest(tasks), occurrences=occurrences,
                source_pool=pool_records, source_pool_sha256=digest(pool_records),
                source_blocks=[b['block_id'] for b in blocks], seed=workload['seed'],
                policy='CYCLIC_ALL_REPETITION0_NATURAL_REQUESTS_ORIGINAL_ORDER_V1',
                session_ordinals=list(range(count)), reset='EMPTY_ONCE_PER_CONDITION',
                cache_lifetime_requests=count, original_continuous_identity='NEW_CONTINUOUS_REPLAY_NOT_ORIGINAL_COLD_BLOCK_IDENTITY',
                first_five_match=dict(status='MATCHED_SELECTION_ONLY' if not reasons else 'NOT_MATCHED', reasons=reasons),
                strict_five_parity='MATCHED_SELECTION_ONLY' if strict_pilot_tasks is not None and tasks[:5] == strict_pilot_tasks else 'NOT_MATCHED',
                statistics=dict(original_request_count=len(pool), unique_samples=unique,
                    unique_original_requests=len({o['original_request_id'] for o in occurrences}),
                    repeated_requests=count - unique, repeat_ratio=(count - unique) / count,
                    complete_cycles=count // len(pool), partial_cycle_requests=residual,
                    cycles_started=(count + len(pool) - 1) // len(pool),
                    per_adapter_requests=dict(Counter(t['adapter_id'] for t in tasks)),
                    per_user_requests=dict(Counter(t['user_id'] for t in tasks))))


def validate_replay_stream(selected):
    """Offline plan/receipt consistency, independent of native packages/assets."""
    require(selected['schema'] == SCHEMA and selected['workload_mode'] == MODE and
            selected['reset'] == 'EMPTY_ONCE_PER_CONDITION' and
            selected['original_continuous_identity'] == 'NEW_CONTINUOUS_REPLAY_NOT_ORIGINAL_COLD_BLOCK_IDENTITY' and
            selected['policy'] == 'CYCLIC_ALL_REPETITION0_NATURAL_REQUESTS_ORIGINAL_ORDER_V1', 'replay contract differs')
    tasks, occurrences, pool = selected['tasks'], selected['occurrences'], selected['source_pool']
    n = selected['request_count']
    require(type(n) is int and n in COUNTS and len(tasks) == len(occurrences) == n and pool and
            selected['session_ordinals'] == list(range(n)) and selected['cache_lifetime_requests'] == n and
            digest(tasks) == selected['tasks_sha256'] and digest(pool) == selected['source_pool_sha256'], 'replay counts/hashes differ')
    require(len({t['request_id'] for t in tasks}) == n and
            len({s['sample_id'] for s in pool}) == len({s['original_request_id'] for s in pool}) == len(pool),
            'duplicate execution ID/original pool source')
    block_ids = list(dict.fromkeys(s['original_block_id'] for s in pool))
    require(selected['source_blocks'] == block_ids, 'replay source block list differs')
    for ordinal, (task, occurrence) in enumerate(zip(tasks, occurrences)):
        cycle, index = divmod(ordinal, len(pool))
        source = pool[index]
        require(all(occurrence[k] == v for k, v in source.items()) and occurrence['session_ordinal'] == ordinal and
                occurrence['pool_index'] == index and occurrence['cycle_index'] == occurrence['occurrence_index'] == cycle and
                occurrence['request_id'] == task['request_id'] and
                all(task[k] == source[k] for k in ('sample_id', 'user_id', 'adapter_id')), 'replay provenance/routing/order differs')
        restored = dict(task, request_id=source['original_request_id'])
        require(digest(restored) == source['original_task_sha256'] and
                task['request_sequence_number'] == occurrence['original_request_sequence_number'], 'original replay request payload changed')
        expected = source['original_request_id'] if not cycle else f'D5REPLAY_{ordinal:03d}_{digest(source)[:16]}'
        require(task['request_id'] == expected, 'non-deterministic replay execution ID')
    unique = len({t['sample_id'] for t in tasks})
    require(selected['statistics'] == dict(original_request_count=len(pool), unique_samples=unique,
        unique_original_requests=len({o['original_request_id'] for o in occurrences}), repeated_requests=n - unique,
        repeat_ratio=(n - unique) / n, complete_cycles=n // len(pool), partial_cycle_requests=n % len(pool),
        cycles_started=(n + len(pool) - 1) // len(pool), per_adapter_requests=dict(Counter(t['adapter_id'] for t in tasks)),
        per_user_requests=dict(Counter(t['user_id'] for t in tasks))), 'replay statistics differ')
    return True
