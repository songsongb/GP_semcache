"""Phase 1 only: sealed NATURAL sessions, CPU lifecycle probes and static counts.

This module has no GPU runner. Probe dispatch does not encode, begin a request,
advance a policy clock, admit an entry, or authorize physical reuse.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import os

from .communication import digest, require
from .r3a_artifacts import ROOT, THRESHOLDS
from .r5_multi_artifacts import resolve, sharing, validate_registry
from .r5_subsequence_natural import CONDITIONS, check_block, resources
from .r5_subsequence_natural_io import output, put, seal, verify
from .r5_subsequence_physical import PhysicalOwner, EXACT
from .r5_subsequence_integrity import SCOPED
from .r5_semantic_index import SemanticIndex
from .r5_i5r_parallel import grams, lcp
from .r5_i5r2_tensors import Evidence, file_sha

DATASETS = ('qasper', '2wikimqa', 'multi_news')
SCHEMA = 'D5_CONTINUOUS_NATURAL_CPU_V1'
NOT_MEASURED = 'NOT_MEASURED'


def build_sessions(workload, registry):
    """Copy the original tasks verbatim; identities live in a separate envelope."""
    copy = dict(workload)
    experiment_id = copy.pop('experiment_id')
    require(digest(copy) == experiment_id, 'workload content identity differs')
    validate_registry(registry, ('Adapter_A', 'Adapter_B', 'Adapter_C'))
    require(len(registry['users']) == 30, 'expanded original registry required')
    require(len(workload['blocks']) == 12 and workload['clusters'] == 1,
            'exact original twelve-block NATURAL plan required')
    sessions = []
    for dataset_index, dataset in enumerate(DATASETS):
        sid = 'D5C_' + digest([experiment_id, dataset])[:20]
        invocations = []
        samples = Counter()
        occurrences = {}
        for bi in range(dataset_index * 4, dataset_index * 4 + 4):
            block = workload['blocks'][bi]
            require(block['dataset'] == dataset and block['repetition'] == (bi % 4) // 2,
                    'original dataset/block/repetition ordering differs')
            check_block(block)
            for task in block['tasks']:
                resolve(registry, task['user_id'], task['adapter_id'])
                require(task['token_ids'] == task['prepared']['token_ids'] and
                        task['input_text'] == task['prepared']['input_text'] and
                        task['policy']['max_new_tokens'] == task['prepared']['max_new_tokens'],
                        'original payload differs from prepared task')
                samples[task['sample_id']] += 1
                occurrences.setdefault(task['sample_id'], []).append((block['repetition'], task))
                ordinal = len(invocations)
                invocations.append(dict(invocation_id=f'{sid}_I{ordinal:03d}',
                    invocation_pair_id=f'{sid}_P{ordinal:03d}', session_sequence_number=ordinal,
                    original_block_index=bi, original_block_id=block['block_id'],
                    original_repetition=block['repetition'], original_task_sha256=digest(task),
                    original_request_id=task['request_id'], original_pair_id=task['pair_id'],
                    original_sample_id=task['sample_id'],
                    original_request_sequence_number=task['request_sequence_number'], task=deepcopy(task)))
        require(len(invocations) == 20 and len(samples) == 10 and set(samples.values()) == {2},
                'require ten distinct original samples, each twice')
        for rows in occurrences.values():
            require([r for r, _ in rows] == [0, 1] and rows[0][1] == rows[1][1],
                    'experiment repetition changed original request content')
        sessions.append(dict(session_id=sid, dataset=dataset,
            original_block_indices=list(range(dataset_index * 4, dataset_index * 4 + 4)),
            invocations=invocations))
    require(len({i['invocation_id'] for s in sessions for i in s['invocations']}) == 60,
            'invocation identity collision')
    return sessions


def runtime_task(invocation, condition):
    """Strict request-schema compatible view. Provenance stays in the envelope.

    Pair ID is not in I5_REQUEST_V1; use invocation_pair_id to join matched A/C
    evidence, never the historical pair ID alone.
    """
    require(condition in CONDITIONS, 'unknown condition')
    t = deepcopy(invocation['task'])
    suffix = 'A' if condition == CONDITIONS[0] else 'C'
    t.update(request_id=invocation['invocation_id'] + '_' + suffix,
             pair_id=invocation['invocation_pair_id'],
             request_sequence_number=invocation['session_sequence_number'])
    require(len(t['request_id']) <= 60, 'runtime request ID exceeds original schema')
    return t


@dataclass
class AdapterState:
    channel: object
    owner: object
    index: object


@dataclass
class SessionState:
    model: object
    adapters: dict


def state_roots(state):
    """Hold actual references, including policy/history roots, across checkpoints."""
    roots = [state.model, state.adapters]
    for aid, a in sorted(state.adapters.items()):
        roots.extend((a, a.channel, a.owner, a.index))
        if a.index is not None:
            roots.extend((a.index.encoder, a.index.clusterer, a.index.extractor,
                          a.index.matcher, a.index.entries, a.index.postings))
        if a.owner is not None:
            o = a.owner
            roots.extend((o.cache, o.cache.entries, o.bridge, o.bridge.metrics,
                          o.cache.admission, o.cache.eviction, o.bridge.metrics.updater,
                          o.bridge.metrics.history, o.bridge.metrics.appearances,
                          o.bridge.seen, o.proofs, o.admission_bindings, o.validation))
    return tuple(roots)


def execute_sessions(sessions, backend):
    """One fresh initialization per dataset AND condition; blocks only checkpoint.

    backend owns model/channels, execution and exception-safe initialization.
    close must dispose owners and terminate channels/model. CPUProbe is the only
    backend supplied in Phase 1. No GPU dispatch is exposed by this module/CLI.
    """
    trace = []
    for session in sessions:
        for condition in CONDITIONS:
            state = backend.initialize(session, condition)
            roots = state_roots(state)
            try:
                require(set(state.adapters) == {i['task']['adapter_id'] for i in session['invocations']},
                        'adapter coverage differs')
                for n, invocation in enumerate(session['invocations']):
                    task = runtime_task(invocation, condition)
                    adapter = state.adapters[task['adapter_id']]
                    backend.execute(state, adapter, invocation, task, condition)
                    trace.append(dict(session_id=session['session_id'], condition=condition,
                                      request_id=task['request_id'],
                                      invocation_id=invocation['invocation_id'],
                                      original_block_index=invocation['original_block_index']))
                    last = n == len(session['invocations']) - 1
                    if last or session['invocations'][n+1]['original_block_index'] != invocation['original_block_index']:
                        backend.checkpoint(state, invocation['original_block_index'])
                    require(all(a is b for a, b in zip(roots, state_roots(state))) and
                            len(roots) == len(state_roots(state)), 'session state reconstructed inside dataset')
            finally:
                backend.close(state)
    return trace


class NoInferenceEncoder:
    def encode(self, texts):
        raise RuntimeError('Phase 1 lifecycle probe must not encode or simulate TinyBERT')


class CPUProbe:
    """Real empty D5 owners/indexes, metadata-only model/channel placeholders.

    Nothing here represents an actual model initialization, inference request,
    admission, policy outcome or physical HIT. Initialization counts describe the
    implemented lifecycle orchestration only.
    """
    def __init__(self, registry, identities):
        self.registry = registry
        self.identities = identities
        self.initializations = []
        self.closed = []
        self.checkpoints = []

    def initialize(self, session, condition):
        adapters = {}
        for aid in sorted({i['task']['adapter_id'] for i in session['invocations']}):
            owner = index = None
            if condition == CONDITIONS[1]:
                owner = PhysicalOwner(self.identities[aid], self.registry, validation_mode=SCOPED)
                index = SemanticIndex(NoInferenceEncoder(), clusters=1)
            adapters[aid] = AdapterState(object(), owner, index)
        state = SessionState(object(), adapters)
        self.initializations.append((session['session_id'], condition, state))
        return state

    def execute(self, state, adapter, invocation, task, condition):
        resolve(self.registry, task['user_id'], task['adapter_id'])
        require(adapter is state.adapters[task['adapter_id']], 'wrong adapter state')
        original = invocation['task']
        require({k: v for k, v in task.items() if k not in ('request_id', 'pair_id', 'request_sequence_number')} ==
                {k: v for k, v in original.items() if k not in ('request_id', 'pair_id', 'request_sequence_number')},
                'runtime payload/routing changed')
        if adapter.owner is not None:
            require(adapter.owner.bridge.identity['adapter_id'] == task['adapter_id'] and
                    not adapter.owner.cache.entries and adapter.owner.cache.now == 0 and
                    not adapter.owner.bridge.seen and not adapter.index.entries,
                    'CPU probe must not fabricate resident entries or policy activity')
        else:
            require(condition == CONDITIONS[0] and adapter.index is None, 'A/C state mixed')

    def checkpoint(self, state, block):
        self.checkpoints.append((state, block, state_roots(state)))

    def close(self, state):
        for a in state.adapters.values():
            if a.owner is not None:
                a.owner.dispose()
        self.closed.append(state)


def validate_lifecycle(sessions, registry, identities):
    probe = CPUProbe(registry, identities)
    trace = execute_sessions(sessions, probe)
    require(len(probe.initializations) == len(probe.closed) == 6 and len(probe.checkpoints) == 24,
            'dataset/condition initialization or block checkpoints differ')
    require(len({r['request_id'] for r in trace}) == len(trace) == 120, 'runtime ID collision')
    roots = [state_roots(s) for _, _, s in probe.initializations]
    for i, left in enumerate(roots):
        for right in roots[:i]:
            require(not ({id(x) for x in left if x is not None} &
                         {id(x) for x in right if x is not None}), 'dataset/condition state shared')
    for session in sessions:
        sequences = [[r['invocation_id'] for r in trace if r['session_id'] == session['session_id'] and
                      r['condition'] == c] for c in CONDITIONS]
        require(sequences[0] == sequences[1], 'A/C order differs')
    return dict(status='PASS', scope='CPU orchestration and empty real owners/indexes; no model/network execution',
                dataset_sessions=3, source_invocations=60, paired_condition_dispatches=120,
                fresh_lifecycle_initializations=6, block_checkpoints=24, internal_cache_resets=0,
                actual_model_initializations=NOT_MEASURED, trace=trace)


def content_identity(t):
    return digest({k: t[k] for k in ('token_ids', 'input_text', 'policy')})


def static_analysis(session, registry):
    """Ordered earlier-source/later-target occurrence pairs, w=3, no residency.

    Four disjoint occurrence categories; target-window unions have a separate
    denominator. Sharing is registry eligibility, not a physical authorization.
    """
    inv = session['invocations']
    totals = Counter()
    allowed = Counter()
    pairs = []
    unique = set()
    eligible_unique = set()
    repeats = Counter()
    natural_sample_pairs = set()
    for ti, target in enumerate(inv):
        t = target['task']; tg = grams(t['token_ids'])
        for source in inv[:ti]:
            s = source['task']; sg = grams(s['token_ids']); prefix = lcp(s['token_ids'], t['token_ids'])
            reason = sharing(registry, s['user_id'], t['user_id'], s['adapter_id'], t['adapter_id'])
            counts = Counter()
            for gram in sorted(sg.keys() & tg.keys()):
                for tp in tg[gram]:
                    unique.add((ti, tp))
                    if reason is None: eligible_unique.add((ti, tp))
                    # Count equal and shifted occurrences without quadratic expansion.
                    same = int(tp in sg[gram])
                    if same:
                        counts['shared_prefix_windows' if tp+3 <= prefix else 'same_position_nonprefix_windows'] += 1
                    shifted = len(sg[gram]) - same
                    counts['shifted_prefix_target_windows' if tp+3 <= prefix else 'shifted_position_nonprefix_windows'] += shifted
            totals.update(counts)
            if reason is None: allowed.update(counts)
            exact = content_identity(s) == content_identity(t)
            repeated = s['sample_id'] == t['sample_id']
            if exact:
                repeats['experiment_repeated_sample_pairs' if repeated else 'distinct_sample_identical_request_pairs'] += 1
                if not repeated:
                    natural_sample_pairs.add(tuple(sorted((s['sample_id'], t['sample_id']))))
                if reason is None: repeats['sharing_eligible_exact_request_pairs'] += 1
            if s['token_ids'] == t['token_ids']: repeats['identical_token_id_pairs'] += 1
            pairs.append(dict(source_invocation_id=source['invocation_id'], target_invocation_id=target['invocation_id'],
                source_adapter=s['adapter_id'], target_adapter=t['adapter_id'], sharing_rejection=reason,
                common_prefix_tokens=prefix, exact_request=exact, experiment_repeated_sample=repeated,
                occurrence_counts=dict(counts)))
    keys = ('shared_prefix_windows', 'same_position_nonprefix_windows',
            'shifted_position_nonprefix_windows', 'shifted_prefix_target_windows')
    return dict(dataset=session['dataset'], ordered_request_pairs=len(pairs),
        occurrence_counts={k: totals[k] for k in keys}, sharing_eligible_occurrence_counts={k: allowed[k] for k in keys},
        matching_target_window_union=len(unique), sharing_eligible_target_window_union=len(eligible_unique),
        exact_requests={k: repeats[k] for k in ('experiment_repeated_sample_pairs',
            'distinct_sample_identical_request_pairs', 'sharing_eligible_exact_request_pairs', 'identical_token_id_pairs')} |
            dict(distinct_original_sample_identical_request_pairs=len(natural_sample_pairs)),
        sharing_pair_counts=dict(Counter(p['sharing_rejection'] or 'ELIGIBLE' for p in pairs)), pairs=pairs,
        scope='STATIC_ONLY; same-position prefix is necessary, not sufficient, for EXACT. Nonprefix/shifted fail EXACT.',
        semantic_assignment=NOT_MEASURED, resident_entries=NOT_MEASURED, measured_policy_outcomes=NOT_MEASURED,
        attention_impact=NOT_MEASURED, gpu_physical_hits=NOT_MEASURED,
        gpu_tensor_equivalence=NOT_MEASURED, e2e_latency=NOT_MEASURED)


def resource_assessment(tasks):
    """Diagnostic forecast using the existing formula, without changing its gates."""
    by_adapter = Counter()
    ud_payload = Counter()
    for t in tasks:
        n = len(t['token_ids']); g = t['policy']['max_new_tokens']
        by_adapter[t['adapter_id']] += (n+g-1)*32*2560*44 + 32*32*n*n*2 + g*(50272*4+4*2**20) + 64*2**20
        # Actual UD saves one FP16 hidden and three FP32 deltas per layer/pass.
        # Payload only; ledgers, headers and inventories require additional space.
        ud_payload[t['adapter_id']] += (n+g-1)*32*2560*(2+3*4)
    files = 2*sum(t['policy']['max_new_tokens']*(32*14+4)+64 for t in tasks)+2048
    n = max(len(t['token_ids']) for t in tasks)
    total = max(len(t['token_ids'])+t['policy']['max_new_tokens'] for t in tasks)
    probability = 32*n*n*2
    working = probability+32*n*n*8+32*3*n*8
    gpu = (5400000000+3*512*2**20+4*2**30+2*32*2560*total*2+
           n*491520+working+n*50272*4)
    try:
        bounded = resources(tasks)
        gate = dict(status='PASS', existing_resource_check=bounded)
    except ValueError as exc:
        gate = dict(status='FAIL', reason=str(exc))
    return dict(existing_gates=gate, condition_evidence_forecast_bytes=sum(by_adapter.values()),
        paired_run_evidence_forecast_bytes=2*sum(by_adapter.values()),
        per_adapter_forecast_bytes=dict(by_adapter), estimated_run_files=files,
        per_adapter_ud_dense_payload_bytes=dict(ud_payload),
        ud_payload_limit_failures=[a for a, v in sorted(ud_payload.items()) if v >= 4*2**30],
        max_input_tokens=n, max_context_tokens=total, gpu_estimate_bytes=gpu,
        gpu_limit_bytes=22*2**30, probability_layer_bytes=probability,
        impact_layer_working_estimate_bytes=working,
        limit_checks=dict(file_count=files <= 500000, condition_evidence=sum(by_adapter.values()) < 32*2**30,
            paired_run_evidence=2*sum(by_adapter.values()) < 64*2**30,
            per_adapter_forecast=all(v < 16*2**30 for v in by_adapter.values()),
            ud_payload_only=all(v < 4*2**30 for v in ud_payload.values()),
            gpu_memory=gpu < 22*2**30, attention=probability <= 128*2**20,
            impact_working=working <= 2**30),
        forwards_per_condition=sum(t['policy']['max_new_tokens'] for t in tasks),
        resident_qkv_bytes_per_w3_entry=1474560, resident_64_entry_bytes_per_adapter=94371840,
        condition_limit_bytes=32*2**30, paired_run_limit_bytes=64*2**30, file_limit=500000,
        required_free_disk_bytes=96*2**30, per_adapter_es_limit_bytes=16*2**30,
        per_adapter_ud_limit_bytes=4*2**30, worker_deadline_seconds=7200,
        runtime_completion_time=NOT_MEASURED,
        note='Forecast uses the unchanged conservative formula, including reserves. UD payload lower bound excludes ledgers/metadata. No limits bypassed.')


def cpu_validate(original_experiment, multi, output_dir):
    """Strict real-assets command. Missing assets write a sealed BLOCKED report."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'set CUDA_VISIBLE_DEVICES empty')
    out = output(output_dir, (original_experiment, multi))
    record = dict(schema=SCHEMA, phase1_status='FAIL', real_data_validation='NOT_RUN',
        gpu_physical_hits=NOT_MEASURED, measured_policy_outcomes=NOT_MEASURED,
        gpu_tensor_equivalence=NOT_MEASURED, e2e_latency=NOT_MEASURED,
        validation_mode=SCOPED, context_mode=EXACT, allow_approximate=False, thresholds=THRESHOLDS,
        original_experiment=str(Path(original_experiment).resolve()), multi=str(Path(multi).resolve()))
    paths = [Path(__file__), ROOT/'scripts/154_r5_continuous_natural.py',
             ROOT/'tests/test_r5_continuous_natural.py']
    record['implementation_files'] = {str(p.relative_to(ROOT)): file_sha(p) for p in paths}
    try:
        from .r5_subsequence_natural import inputs, transport
        from .r5_i5r3b_transport import predicted_request
        require(Path(original_experiment).is_dir() and Path(multi).is_dir(),
                'existing sealed workload and expanded preparation directories required')
        w, d, receipt, inventories, m, pm, specs, users, tok = inputs(original_experiment, multi)
        sessions = build_sessions(w, users)
        identities = {}
        for s in sessions:
            for i in s['invocations']:
                t = runtime_task(i, CONDITIONS[1])
                identities[t['adapter_id']] = predicted_request(t, pm, specs[t['adapter_id']], tok.identity)['physical_identity']
        lifecycle = validate_lifecycle(sessions, users, identities)
        summaries = []
        for session in sessions:
            name = session['dataset']
            put(out/name/'session.json', session)
            static = static_analysis(session, users)
            put(out/name/'static_opportunities.json', static)
            tasks = [i['task'] for i in session['invocations']]
            forecast = resource_assessment(tasks)
            network = {c: transport([runtime_task(i, c) for i in session['invocations']], pm, specs, tok.identity, users)
                       for c in CONDITIONS}
            summary = dict(dataset=name, original_blocks=session['original_block_indices'],
                invocations=20, distinct_samples=10, repeated_sample_evaluations=10,
                original_request_id_collisions=20-len({t['request_id'] for t in tasks}),
                original_pair_id_collisions=20-len({t['pair_id'] for t in tasks}),
                adapter_requests=dict(Counter(t['adapter_id'] for t in tasks)),
                cache_lifetime_requests=20, fresh_initializations_per_condition=1,
                internal_cache_resets=0, resource_assessment=forecast, transport=network,
                static_counts={k: static[k] for k in ('occurrence_counts', 'sharing_eligible_occurrence_counts', 'exact_requests', 'sharing_pair_counts')})
            put(out/name/'summary.json', summary); summaries.append(summary)
        record.update(phase1_status='PASS', real_data_validation='PASS', sessions=summaries,
            lifecycle=lifecycle, original_inventory=receipt, dataset_inventories=inventories,
            multi_inventory=Evidence(multi).verify('r5_output_manifest.json'),
            fixture_only=False, future_gpu_ready=False)
    except (ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        record['blocker'] = str(exc)
        if (not Path(original_experiment).is_dir() or not Path(multi).is_dir() or
                isinstance(exc, (FileNotFoundError, ImportError))):
            record.update(phase1_status='BLOCKED', real_data_validation='BLOCKED')
    put(out/'phase1_validation.json', record)
    lines = ['# D5 Continuous NATURAL Phase 1', '',
             'Phase 1: **'+record['phase1_status']+'**. Real-data validation: '+record['real_data_validation']+'.', '',
             'GPU physical HITs, tensor equivalence, measured policy outcomes, and E2E latency: NOT_MEASURED.', '',
             'CPU probes retain empty real D5 owners/indexes; model/channel placeholders do not execute inference.']
    if 'blocker' in record:
        lines += ['', 'Blocker: '+record['blocker'], '',
                  'Dataset request counts and static opportunities are NOT_RUN. No fixture counts substitute for original data.']
    for s in record.get('sessions', []):
        lines += ['', '## '+s['dataset'], '',
                  f"Blocks {s['original_blocks']}: 20 invocations, 10 samples each twice; 1 lifecycle initialization per condition; 0 internal resets.",
                  '', 'Static occurrence counts: '+str(s['static_counts']), '',
                  'Resource forecast: '+str(s['resource_assessment'])]
    (out/'phase1_report.md').write_text('\n'.join(lines)+'\n')
    seal(out); verify(out)
    return record
