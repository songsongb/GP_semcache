"""Original-order NATURAL streams and conservative, non-inflating CPU plans."""
from collections import Counter
from copy import deepcopy
import math

from .communication import digest, require
from .r5_subsequence_natural import check_block
from .r5_i5_workload import routing, TEMPLATE
from .r4c_workload import policy

DATASETS = ('qasper', '2wikimqa', 'multi_news')
COUNTS = (5, 20, 50)
CONDITIONS = ('FULL_RECOMPUTE', 'SEMCACHE_D5_EXACT', 'SEMCACHE_D5_APPROX')


def knobs(bandwidth_mbps=None, ud_throughput_factor=1.0):
    require(bandwidth_mbps is None or type(bandwidth_mbps) in (float, int) and
            math.isfinite(bandwidth_mbps) and bandwidth_mbps in (200, 100, 50),
            'bandwidth must be unlimited (None), 200, 100 or 50 Mbps')
    require(type(ud_throughput_factor) in (float, int) and math.isfinite(ud_throughput_factor) and
            ud_throughput_factor in (1., .5, .25), 'UD factor must be 1, .5 or .25')
    return dict(bandwidth_mbps=bandwidth_mbps, ud_throughput_factor=float(ud_throughput_factor),
                link_scope='APPLICATION_LOOPBACK_PER_CHANNEL_PER_DIRECTION', rtt_ms=None,
                ud_scope='EMULATED_UD_SERVICE_TIME')


def stream(workload, dataset, count, *, continuous_contract=None, strict_pilot_tasks=None):
    require(dataset in DATASETS and type(count) is int and count in COUNTS, 'unsupported dataset/scale')
    require(workload['template_version'] == TEMPLATE and workload['clusters'] == 1, 'original template/cluster differs')
    blocks = [b for b in workload['blocks'] if b['dataset'] == dataset and
              b['track'] == 'NATURAL' and b['repetition'] == 0]
    require(blocks, 'MISSING_ORIGINAL_NATURAL_BLOCK')
    all_tasks = []
    for b in blocks:
        require(b['routing_seed'] == workload['seed'] and not b.get('continuity_breaks'),
                'group split/routing seed mismatch')
        for t in check_block(b):
            s = t['prepared']['sample']
            require(s['id'] == t['sample_id'] and s['group_id'] == t['group_id'] and s['dataset'] == dataset,
                    'sample/group/dataset changed')
            require(all(t[k] == v for k, v in routing(s, workload['seed']).items()), 'mutated routing')
            require(t['token_ids'] == t['prepared']['token_ids'] and
                    t['prepared']['template_version'] == TEMPLATE and
                    t['input_text'] == t['prepared']['input_text'] and
                    t['policy'] == policy(workload['special_tokens'], 'quality', t['prepared']['max_new_tokens']),
                    'prompt/generation policy changed')
        all_tasks.extend(b['tasks'])
    require(len({t['sample_id'] for t in all_tasks}) == len(all_tasks) and
            len({t['request_id'] for t in all_tasks}) == len(all_tasks), 'duplicate original request/sample')
    require(len(all_tasks) >= count, 'INSUFFICIENT_ORIGINAL_REQUESTS; no replication or alternative selection')
    chosen = all_tasks[:count]
    groups = {t['group_id'] for t in chosen}
    require(not any(t['group_id'] in groups for t in all_tasks[count:]), 'GROUP_SPLIT_AT_PREFIX_BOUNDARY')
    source_blocks, accumulated = [], 0
    for b in blocks:
        if accumulated >= count:
            break
        source_blocks.append(b['block_id'])
        accumulated += len(b['tasks'])
    # Every scale uses the same complete original prefix. No task is rewritten.
    binding = dict(dataset=dataset, request_count=count, source_blocks=source_blocks,
                   tasks_sha256=digest(chosen), session_ordinals=list(range(count)),
                   reset='EMPTY_ONCE_PER_CONDITION', cache_lifetime_requests=count,
                   session_contract='NEW_CONTINUOUS_SESSION_FROM_ORIGINAL_ORDER_V1')
    if len(source_blocks) > 1:
        require(continuous_contract == dict(binding, acknowledgement='REVIEWED_NEW_CONTINUOUS_IDENTITY'),
                'BLOCKED_CONTINUOUS_IDENTITY: explicitly review new session contract; cold blocks are not continuous proof')
    matched = strict_pilot_tasks is not None and chosen[:5] == strict_pilot_tasks
    return dict(**binding, tasks=deepcopy(chosen), strict_five_parity='MATCHED_SELECTION_ONLY' if matched else 'NOT_MATCHED',
                prefix_policy='original repetition0 order; whole groups; no rewritten tasks',
                original_continuous_identity='SINGLE_BLOCK' if len(source_blocks) == 1 else 'NEW_EXPLICIT_CONTRACT')


def preflight(tasks, performance_knobs, *, slurm_seconds=16200, seconds_per_forward=5.,
              ud_operation_seconds=None, available=None):
    """Fail closed when UD costs are unknown for slowdown; no estimated HIT savings.

    Network includes a conservative 64KiB header EACH direction plus framing,
    empty controls, final ACK and STOP/BYE. Bounds are per condition, never a
    promise that a job will complete. Caller must also run exact transport checks.
    """
    knobs(performance_knobs['bandwidth_mbps'], performance_knobs['ud_throughput_factor'])
    require(tasks and len(tasks) <= 50 and 0 < slurm_seconds <= 16200 and
            type(seconds_per_forward) in (int, float) and math.isfinite(seconds_per_forward) and
            seconds_per_forward > 0, 'invalid fixed resource/time envelope')
    if ud_operation_seconds is not None:
        require(type(ud_operation_seconds) in (float, int) and math.isfinite(ud_operation_seconds) and
            ud_operation_seconds > 0, 'positive UD operation planning bound required')
    reasons = []
    n = max(len(t['token_ids']) for t in tasks)
    context = max(len(t['token_ids']) + t['policy']['max_new_tokens'] for t in tasks)
    forwards = sum(t['policy']['max_new_tokens'] for t in tasks)
    per_adapter = dict(Counter(t['adapter_id'] for t in tasks))
    if not all(3 <= len(t['token_ids']) <= 512 and 1 <= t['policy']['max_new_tokens'] <= 96 for t in tasks) or context > 2048:
        reasons.append('UNCHANGED_INPUT_CONTEXT_GENERATION_BOUND')
    probability = 32 * n * n * 2
    working = probability + 32 * n * n * 8 + 32 * 3 * n * 8
    cache = len(per_adapter) * 512 * 2**20
    gpu = 5400000000 + cache + 4 * 2**30 + 2 * 32 * 2560 * context * 2 + n * 491520 + working + n * 50272 * 4
    if probability > 128 * 2**20 or working > 2**30 or gpu >= 22 * 2**30:
        reasons.append('GPU_IMPACT_MEMORY_BOUND')
    exchanges = forwards * 32
    rows = sum(len(t['token_ids']) + t['policy']['max_new_tokens'] - 1 for t in tasks)
    es_payload = rows * 32 * 2560 * 2
    ud_payload = rows * 32 * 2560 * 12
    framing_each = (exchanges + 2 * len(tasks) + 2 * len(per_adapter)) * (65536 + 20)
    es_wire, ud_wire = es_payload + framing_each, ud_payload + framing_each
    rate = performance_knobs['bandwidth_mbps']
    network_s = 0. if rate is None else (es_wire + ud_wire) * 8 / (rate * 1e6)
    factor = performance_knobs['ud_throughput_factor']
    ud_pacing_s = 0. if factor == 1. else (None if ud_operation_seconds is None else exchanges * ud_operation_seconds * (1. / factor - 1.))
    if ud_pacing_s is None:
        reasons.append('UD_OPERATION_CALIBRATION_REQUIRED')
    # One fresh condition, no concurrent conditions, initialization and seal reserve.
    forecast = None if ud_pacing_s is None else 600 + forwards * seconds_per_forward + network_s + ud_pacing_s
    worker_deadline = min(7200, slurm_seconds - 300)
    if worker_deadline <= 0 or forecast is not None and forecast > worker_deadline:
        reasons.append('DEADLINE_OR_SLURM_BOUND')
    summary_bytes = len(tasks) * 512 * 1024 + 4 * 2**20
    file_count = len(tasks) + len(per_adapter) + 8
    if summary_bytes > 64 * 2**20 or file_count > 2048:
        reasons.append('LIGHT_DISK_FILE_BOUND')
    requirements = dict(host_available_bytes=32 * 2**30, cpus=4, disk_free_bytes=2**30,
                        gpu_free_after_load_bytes=12 * 2**30)
    if available is not None:
        for key, value in requirements.items():
            if available.get(key, -1) < value:
                reasons.append('INSUFFICIENT_' + key.upper())
    return dict(status='BLOCKED_INFEASIBLE' if reasons else 'STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING', reasons=reasons,
                requests=len(tasks), max_input_tokens=n, max_context_tokens=context, max_forwards=forwards,
                per_adapter_requests=per_adapter, per_adapter_capacity_bytes=512 * 2**20,
                cache_capacity_total_bytes=cache, gpu_estimate_bytes=gpu, gpu_allocation_ceiling_bytes=22 * 2**30,
                probability_layer_bytes=probability, impact_layer_working_bytes=working,
                requirements=requirements, max_projection_exchanges=exchanges, es_to_ud_wire_bound_bytes=es_wire,
                ud_to_es_wire_bound_bytes=ud_wire, link_pacing_upper_seconds=network_s,
                ud_extra_service_seconds=ud_pacing_s, summary_bytes_bound=summary_bytes, file_count_bound=file_count,
                ud_operation_bound_provenance='USER_PLANNING_ASSUMPTION_NOT_CALIBRATION_RECEIPT' if ud_operation_seconds is not None else 'NOT_PROVIDED',
                host_transient_reserve_bytes=4 * 2**30, worker_deadline_seconds=worker_deadline,
                slurm_seconds=slurm_seconds, forecast_seconds=forecast, seconds_per_forward=seconds_per_forward,
                forecast_scope='UNMEASURED_PLANNING_ASSUMPTION; no HIT savings; verify with manual 5-request evidence',
                uncertainty='host proofs/semantic/quality/model load may exceed reserves; runtime deadline remains mandatory',
                measured_resources=None, performance_knobs=performance_knobs)


def matrix():
    return dict(status='CPU_PLAN_ONLY_NO_SUBMISSION', datasets=list(DATASETS), counts=list(COUNTS),
                bandwidth_mbps=[None, 200, 100, 50], ud_throughput_factors=[1., .5, .25],
                candidate_knob_pairs=[knobs(b, f) for b in (None, 200, 100, 50) for f in (1., .5, .25)],
                full_grid_condition_count=3 * 3 * 12 * 3, selected_jobs=[],
                gate='L2 five, then separately approved L2 20/50; L3/L4 evidence before targeted L5 subset')
