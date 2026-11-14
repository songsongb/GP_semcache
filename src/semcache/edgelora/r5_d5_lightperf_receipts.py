"""Compact native receipts. No raw tensors/logits or per-frame files."""
from collections import Counter
from copy import deepcopy
import time

from .communication import canonical, digest, require


class WireSummary:
    """Endpoint-local totals; send/recv scopes include peer wait and pacing."""
    def __init__(self, request_ids):
        require(len(set(request_ids)) == len(request_ids) <= 50, 'bounded unique request IDs required')
        self.ids = tuple(request_ids)
        self.rows = {}
        self.errors = []
        self.observer_ms = 0.

    def __call__(self, record):
        started = time.monotonic_ns()
        h = record.get('header', {})
        pid = h.get('request_id', '')
        matched = [rid for rid in self.ids if pid == rid + '_COMPLETE' or
                   pid.startswith(rid + '_s') and pid[len(rid) + 2:].isdigit()]
        require(len(matched) <= 1, 'ambiguous request wire binding')
        key = matched[0] if matched else 'CONTROL'
        require(matched or pid == 'shutdown' or not record['complete'], 'unbound successful wire frame')
        row = self.rows.setdefault(key, self.empty())
        if not record['complete']:
            require(len(self.errors) < 16, 'wire error count bound exceeded')
            self.errors.append(str(record.get('error'))[:2048])
        else:
            s = row[record['operation']]
            s['frames'] += 1
            s['wire_bytes'] += record['wire_bytes']
            s['payload_bytes'] += h['payload_bytes']
            s['frame_overhead_bytes'] += record['wire_bytes'] - h['payload_bytes']
            for k in ('socket_call_wall_ms', 'framing_serialization_ms', 'framing_deserialization_ms', 'added_pacing_ms'):
                s[k] += record.get(k, 0.)
            s['frame_hash_chain'] = digest([s['frame_hash_chain'], record['frame_sha256']])
        self.observer_ms += (time.monotonic_ns() - started) / 1e6

    @staticmethod
    def empty():
        return {op: dict(frames=0, wire_bytes=0, payload_bytes=0, frame_overhead_bytes=0,
                         socket_call_wall_ms=0., framing_serialization_ms=0.,
                         framing_deserialization_ms=0., added_pacing_ms=0., frame_hash_chain=None)
                for op in ('send', 'recv')}

    def request(self, request_id):
        require(request_id in self.rows, 'missing request wire summary')
        return deepcopy(self.rows[request_id])


def wire_equal(es, ud):
    for a, b in (('send', 'recv'), ('recv', 'send')):
        require(all(es[a][k] == ud[b][k] for k in
                    ('frames', 'wire_bytes', 'payload_bytes', 'frame_overhead_bytes', 'frame_hash_chain')),
                'ES/UD wire accounting or frame hashes differ')


def compact_profile(profile):
    """Collapse layer/role dimensions only after the live proof calculations."""
    bins = {}
    for item in profile['components']:
        key = (item['phase'], item['component'])
        b = bins.setdefault(key, dict(count=0, bytes=0, inclusive_wall_ms=0., exclusive_wall_ms=0.))
        for k in b:
            b[k] += item[k]
    return dict(components=[dict(phase=k[0], component=k[1], **v) for k, v in sorted(bins.items())],
                detail_counts=profile['detail_counts'], profiler_overhead_ms='NOT_MEASURED',
                gpu_kernel_ms='NOT_MEASURED',
                scope='observed exclusive bins only; inclusive parents overlap; wall is not GPU kernel duration')


def compact_request(task, ordinal, result, owner, policy_counts, tasks, wire, ud):
    from .r5_d5_lightperf_analysis import categories
    physical = result['completion']['physical']
    cats, sources = categories(physical, tasks, task['request_id'])
    wire_equal(wire, ud['wire'])
    count = len(result['generated_token_ids'])
    rows = result['row_summary']
    require(rows['fresh_qkv_rows'] == ud['cpu_projection_rows'], 'ES/UD projection row counts differ')
    require(sum(cats.values()) == rows['skipped_qkv_rows'], 'physical categories/sparse skips differ')
    row = {k: result[k] for k in ('generated_token_ids', 'generated_token_ids_sha256', 'generated_text',
                                 'generated_text_sha256', 'termination', 'first_available_ns',
                                 'model_execution_e2e_ms', 'model_ttft_ms', 'prefill_ms', 'decode_ms',
                                 'admission_completion_ms', 'condition')}
    row.update({k: task[k] for k in ('request_id', 'sample_id', 'user_id', 'adapter_id')})
    row.update(session_ordinal=ordinal, input_tokens=len(task['token_ids']), output_tokens=count,
               generation_e2e_ms=result['application_e2e_ms'], **rows,
               physical_hit=rows['skipped_qkv_rows'] > 0, physical_reuse_sources=sources,
               categories=cats, cache_occupancy_bytes=0 if owner is None else owner.cache.physical_tensor_bytes,
               cache_entry_count=0 if owner is None else len(owner.cache.entries), policy_counts=dict(policy_counts),
               matching_candidate_count=result['matching_candidate_count'],
               semantic_encoding_ms=result.get('semantic_encoding_ms', 'NOT_MEASURED'),
               cluster_assignment_ms=result.get('cluster_assignment_ms', 'NOT_MEASURED'),
               cache_completion_costs=None if physical is None else physical['costs'],
               integrity_validation=None if physical is None else
                   {k: v for k, v in physical['integrity_validation'].items() if k != 'authorizations'},
               attention_impact=None if result['completion']['impact'] is None else
                   dict(layers=len(result['completion']['impact']['records']),
                        elapsed_ms=result['completion']['impact']['elapsed_ms'],
                        values_sha256=digest([r['span_values'] for r in result['completion']['impact']['records']])),
               profile=compact_profile(result['components']), wire=wire, ud_compute=ud['compute'],
               timing_scope='wall; application includes quality and primary receipt I/O; init, final sealing excluded',
               unavailable=dict(gpu_qkv_kernel_ms='NOT_MEASURED', observer_total_ms='NOT_MEASURED',
                                cuda_sync_ms='NOT_MEASURED', network_queue_ms='NOT_MEASURED'),
               correctness_qualified_speedup='NOT_ESTABLISHED')
    require(len(canonical(row)) <= 96 * 1024, 'compact request receipt exceeds96KiB')
    return row


class PolicyCounter:
    def __init__(self):
        self.counts = Counter({k: 0 for k in ('INSERT', 'EVICT', 'CHU', 'PBR')})

    def __call__(self, event):
        if event.get('kind') in self.counts:
            self.counts[event['kind']] += 1


def bind_replay_provenance(row, occurrence):
    require(all(row[k] == occurrence[k] for k in ('request_id', 'sample_id', 'user_id', 'adapter_id', 'session_ordinal')),
            'replay occurrence does not match executed request')
    row['replay_provenance'] = deepcopy(occurrence)
    require(len(canonical(row)) <= 96 * 1024, 'replay request receipt exceeds96KiB')
