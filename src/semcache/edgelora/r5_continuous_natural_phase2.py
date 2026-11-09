"""Phase 2 CPU-only real NATURAL expansion. No GPU/runtime or Slurm actions."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import os

from .communication import digest, require
from .r3a_artifacts import ROOT, THRESHOLDS
from .r5_i5r2_tensors import Evidence, file_sha
from .r5_subsequence_natural_io import output, put, seal, verify, safe_path
from .r5_subsequence_natural import inputs, transport
from .r5_continuous_natural import (DATASETS, CONDITIONS, SCOPED, EXACT, build_sessions,
    runtime_task, execute_sessions, CPUProbe, state_roots, resource_assessment, static_analysis)
from .r5_continuous_natural_sharing import analyze, token_regions, CATEGORIES, SCOPES
from .r5_i5_workload import select, pack_groups, requests, EligibilityError
from .r5_i5r3a import provenance

SCHEMA = 'D5_CONTINUOUS_NATURAL_PHASE2_CPU_V1'
FILES = ('src/semcache/edgelora/r5_continuous_natural_phase2.py',
         'src/semcache/edgelora/r5_continuous_natural_sharing.py',
         'scripts/155_r5_continuous_natural_phase2.py', 'tests/test_r5_continuous_natural_phase2.py',
         'src/semcache/edgelora/r5_continuous_natural.py', 'src/semcache/edgelora/r5_i5_workload.py',
         'src/semcache/edgelora/r5_subsequence_physical.py', 'src/semcache/edgelora/r5_subsequence_integrity.py',
         'src/semcache/edgelora/r5_subsequence_generation.py', 'src/semcache/edgelora/r5_mstage_bridge.py')


class Blocked(ValueError):
    pass


def implementation():
    return {p: file_sha(ROOT/p) for p in FILES}


def cpu_only():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'set CUDA_VISIBLE_DEVICES empty')


def declared_dataset_paths(original_experiment):
    """Protect declared sources before even a failed loader can create output.

    This preliminary read grants no trust: full sealed provenance is still
    checked by load_original. Invalid metadata fails before any write.
    """
    path = Path(original_experiment)/'i5_dataset_manifest.json'
    if not path.is_file(): return []
    require(path.stat().st_size <= 16*2**20, 'original dataset metadata exceeds16MiB')
    records = json.loads(path.read_bytes())
    require(isinstance(records, dict) and all(isinstance(r, dict) and
            isinstance(r.get('prepared_path'), str) and r['prepared_path'] for r in records.values()),
            'invalid original dataset preparation paths')
    return [r['prepared_path'] for r in records.values()]


def load_original(original_experiment, multi, phase1_audit):
    for p in (original_experiment, multi, phase1_audit):
        if not Path(p).is_dir(): raise Blocked('missing existing sealed asset: '+str(p))
    receipt = verify(phase1_audit)
    phase1 = json.loads((Path(phase1_audit)/'phase1_validation.json').read_bytes())
    require(phase1['schema'] == 'D5_CONTINUOUS_NATURAL_CPU_V1' and
            phase1['phase1_status'] == phase1['real_data_validation'] == 'PASS', 'real Phase 1 PASS required')
    require(phase1['context_mode'] == EXACT and phase1['validation_mode'] == SCOPED and
            phase1['allow_approximate'] is False and phase1['thresholds'] == THRESHOLDS, 'Phase 1 safety differs')
    loaded = inputs(original_experiment, multi)
    w, d, original_receipt, inventories, m, pm, specs, users, tok = loaded
    require(phase1['original_inventory'] == original_receipt and phase1['dataset_inventories'] == inventories and
            phase1['multi_inventory'] == Evidence(multi).verify('r5_output_manifest.json'), 'Phase 1 original input bindings differ')
    baseline = build_sessions(w, users)
    for session in baseline:
        stored = json.loads((Path(phase1_audit)/session['dataset']/'session.json').read_bytes())
        require(stored == session, 'Phase 1 original invocation provenance differs')
    datasets = {name: provenance(d[name]['prepared_path'], name)[0] for name in DATASETS}
    return loaded, baseline, datasets, receipt


def expand(workload, dataset_manifest, baseline, datasets, tokenizer, *, input_limit, context_truncation):
    """Use I5 selection, with an explicit recipe corroborated against sealed data."""
    require(type(input_limit) is int and 3 <= input_limit <= 512 and context_truncation in ('allow', 'forbid'),
            'explicit original bounded input/truncation recipe required')
    sessions = []; selections = {}
    for base in baseline:
        name = base['dataset']; samples = datasets[name]
        require(len({s['id'] for s in samples}) == len(samples) and all(s['dataset'] == name for s in samples),
                'invalid dataset sample identities')
        records = [s for s in dataset_manifest[name]['selections'] if s['track'] == 'NATURAL' and s['prefix'] is None]
        require(len(records) == 1, 'one sealed NATURAL selection per dataset required')
        original_selection = records[0]
        tasks = [i['task'] for i in base['invocations']]
        require(tasks == [t for b in workload['blocks'] if b['dataset'] == name for t in b['tasks']] and
                all(i['original_task_sha256'] == digest(i['task']) for i in base['invocations']),
                'Phase 1 prefix differs from original workload')
        require(all(b['routing_seed'] == workload['seed'] for b in workload['blocks']), 'original routing seed differs')
        require(len({t['policy']['max_new_tokens'] for t in tasks}) == 1, 'original dataset generation budget differs')
        budget = tasks[0]['policy']['max_new_tokens']
        settings = dict(seed=workload['seed'], context_limit=workload['context_limit'], budget=budget,
                        allow_truncation=context_truncation == 'allow', input_limit=input_limit)
        try:
            first, report = select(samples, tokenizer, count=10, **settings)
            require(all(original_selection[k] == v for k, v in report.items()),
                    'explicit recipe does not reproduce sealed original selection/eligibility report')
            selected, extended_report = select(samples, tokenizer, count=40, **settings)
        except EligibilityError as exc:
            raise Blocked(name+': '+str(exc)) from exc
        require(selected[:10] == first, 'extension changed first ten selected prompts')
        require([p['sample']['id'] for p in first] == [t['sample_id'] for t in tasks[:10]],
                'original first ten distinct samples/order differ')
        for p, t in zip(first, tasks[:10]):
            require(p == {k: v for k, v in t['prepared'].items() if k != 'strict'},
                    'original first ten prepared payloads differ from deterministic recipe')
        session = deepcopy(base); inv = session['invocations']; checkpoints = []
        for bi in base['original_block_indices']:
            indices = [n for n, i in enumerate(inv) if i['original_block_index'] == bi]
            require(indices and indices == list(range(indices[0], indices[-1]+1)), 'original block not contiguous')
            checkpoints.append(dict(checkpoint_id='HISTORICAL_'+inv[indices[0]]['original_block_id'],
                kind='ORIGINAL_I5_BLOCK', original_block_index=bi,
                original_block_id=inv[indices[0]]['original_block_id'], start=indices[0], stop=indices[-1]+1))
        strict_counts = {sum(t['strict'] for t in b['tasks']) for b in workload['blocks'] if b['dataset'] == name}
        require(len(strict_counts) == 1 and next(iter(strict_counts)) >= 1, 'original strict capture selection differs')
        strict_count = next(iter(strict_counts))
        cohorts = pack_groups(deepcopy(selected[10:]), workload['block_size'])
        require([p['sample']['id'] for cohort in cohorts for p in cohort] == [p['sample']['id'] for p in selected[10:]],
                'extension checkpoint packing reordered deterministic samples')
        offset = 10
        for ordinal, cohort in enumerate(cohorts):
            chosen = set(sorted(range(len(cohort)), key=lambda n: digest([workload['seed'], cohort[n]['sample']['group_id']]))[:strict_count])
            for n, p in enumerate(cohort): p['strict'] = n in chosen
            new_tasks = requests(cohort, track='NATURAL', prefix=None, offset=offset,
                                 special=workload['special_tokens'], routing_seed=workload['seed'])
            cp_id = 'D5C2_EXTENSION_'+digest([base['session_id'], ordinal, new_tasks])[:24]
            start = len(inv)
            for task in new_tasks:
                n = len(inv); sid = base['session_id']
                require(3 <= len(task['token_ids']) <= 512 and len(task['token_ids'])+budget <= 2048 and budget <= 96,
                        'original D5 input/generation resource profile exceeded')
                inv.append(dict(invocation_id=f'{sid}_I{n:03d}', invocation_pair_id=f'{sid}_P{n:03d}',
                    session_sequence_number=n, original_block_index=None, original_block_id=None,
                    original_repetition=None, original_task_sha256=digest(task),
                    original_request_id=task['request_id'], original_pair_id=task['pair_id'],
                    original_sample_id=task['sample_id'], original_request_sequence_number=task['request_sequence_number'],
                    source_kind='PHASE2_EXTENSION_FROM_ORIGINAL_DATASET', extension_checkpoint_id=cp_id, task=task))
            checkpoints.append(dict(checkpoint_id=cp_id, kind='PHASE2_EXTENSION_CHECKPOINT',
                original_block_index=None, original_block_id=None, start=start, stop=len(inv)))
            offset += len(cohort)
        session.update(checkpoints=checkpoints, schema=SCHEMA)
        counts = Counter(i['original_sample_id'] for i in inv)
        require(inv[:20] == base['invocations'] and len(inv) == 50 and len(counts) == 40 and
                Counter(counts.values()) == Counter({1: 30, 2: 10}), 'expanded counts or Phase 1 prefix differ')
        require(len({i['original_sample_id'] for i in inv[20:]}) == 30 and
                not ({i['original_sample_id'] for i in inv[:20]} & {i['original_sample_id'] for i in inv[20:]}),
                'extension reuses original Phase 1 sample')
        selections[name] = dict(settings=settings, original_selection_reproduced=True, selection=extended_report,
                                new_sample_ids=[i['original_sample_id'] for i in inv[20:]])
        sessions.append(session)
    require(len(sessions) == 3 and len({i['invocation_id'] for s in sessions for i in s['invocations']}) == 150,
            'expanded invocation/session identity differs')
    return sessions, selections


def lifecycle(sessions, registry, identities):
    # Phase 1's scheduler accepts reporting-boundary keys under its legacy field
    # name. Adapt only a temporary scheduling view; never rewrite provenance.
    scheduled = deepcopy(sessions)
    for s in scheduled:
        for cp in s['checkpoints']:
            for i in s['invocations'][cp['start']:cp['stop']]: i['original_block_index'] = cp['checkpoint_id']
    probe = CPUProbe(registry, identities); trace = execute_sessions(scheduled, probe)
    require(len(trace) == len({r['request_id'] for r in trace}) == 300, 'expanded runtime ID collision/count')
    require(len(probe.initializations) == len(probe.closed) == 6 and
            len(probe.checkpoints) == 2*sum(len(s['checkpoints']) for s in sessions), 'expanded initialization/checkpoints differ')
    roots = [state_roots(s) for _, _, s in probe.initializations]
    for n, group in enumerate(roots):
        for earlier in roots[:n]:
            require(not ({id(x) for x in group if x is not None} & {id(x) for x in earlier if x is not None}),
                    'dataset/condition state shared')
    for s in sessions:
        orders = [[r['invocation_id'] for r in trace if r['session_id'] == s['session_id'] and r['condition'] == c] for c in CONDITIONS]
        require(orders[0] == orders[1] == [i['invocation_id'] for i in s['invocations']], 'A/C order differs')
    return dict(status='PASS', scope='CPU orchestration, real empty owner/index, placeholder model/channel',
                sessions=3, invocations=150, condition_dispatches=300, fresh_initializations=6,
                checkpoints=len(probe.checkpoints), internal_cache_resets=0,
                actual_model_initializations='NOT_MEASURED', actual_gpu_cache_continuity='NOT_MEASURED')


def write_sessions(out, sessions):
    for s in sessions:
        meta = {k: v for k, v in s.items() if k != 'invocations'}; meta['invocation_files'] = []
        for n, i in enumerate(s['invocations']):
            name = f'invocations/{n:03d}.json'; put(out/s['dataset']/name, i)
            meta['invocation_files'].append(name)
        put(out/s['dataset']/'session.json', meta)


def read_sessions(root):
    sessions = []
    for dataset in DATASETS:
        base = Path(root)/dataset; s = json.loads((base/'session.json').read_bytes())
        files = s.pop('invocation_files')
        require(files == [f'invocations/{n:03d}.json' for n in range(50)], 'invocation shard order differs')
        s['invocations'] = [json.loads(safe_path(base, f).read_bytes()) for f in files]; sessions.append(s)
    return sessions


def preparation_failure(out, exc):
    status = 'BLOCKED' if isinstance(exc, (Blocked, FileNotFoundError, ImportError, EligibilityError)) else 'FAIL'
    record = dict(schema=SCHEMA, status=status, real_data_validation=status, reason=str(exc),
                  gpu_physical_hits='NOT_MEASURED', policy_outcomes='NOT_MEASURED',
                  tensor_equivalence='NOT_MEASURED', attention_impact='NOT_MEASURED', e2e_latency='NOT_MEASURED')
    put(out/'status.json', record)
    (out/'phase2_report.md').write_text('# D5 Continuous NATURAL Phase 2\n\n**'+status+'**: '+str(exc)+
        '\n\nNo expanded real-data workload or sharing conclusion is claimed. GPU metrics: NOT_MEASURED.\n')
    seal(out); verify(out); return record


def prepare(original_experiment, multi, phase1_audit, input_limit, context_truncation, output_dir):
    cpu_only(); out = None; protected = [original_experiment, multi, phase1_audit]
    protected += declared_dataset_paths(original_experiment)
    try:
        loaded, baseline, datasets, phase1_receipt = load_original(original_experiment, multi, phase1_audit)
        w, d, receipt, inventories, m, pm, specs, users, tok = loaded
        protected += [v['prepared_path'] for v in d.values()]
        sessions, selections = expand(w, d, baseline, datasets, tok.tokenizer,
                                      input_limit=input_limit, context_truncation=context_truncation)
        plan = dict(schema=SCHEMA, implementation=implementation(), TEST_FIXTURE_ONLY=w['TEST_FIXTURE_ONLY'],
            original_experiment=str(Path(original_experiment).resolve()), multi=str(Path(multi).resolve()),
            phase1_audit=str(Path(phase1_audit).resolve()), phase1_inventory=phase1_receipt,
            original_inventory=receipt, dataset_inventories=inventories,
            dataset_preparations={name: d[name]['prepared_path'] for name in DATASETS},
            multi_inventory=Evidence(multi).verify('r5_output_manifest.json'),
            recipe=dict(input_limit=input_limit, context_truncation=context_truncation),
            session_digests={s['dataset']: digest(s) for s in sessions},
            conditions=list(CONDITIONS), context_mode=EXACT, validation_mode=SCOPED,
            allow_approximate=False, thresholds=THRESHOLDS, warmup_requests=0,
            source_invocations=150, distinct_samples_per_dataset=40,
            measurement_scope='CPU_ONLY_NO_GPU_APPROVAL')
        plan['plan_id'] = digest(plan)
        out = output(output_dir, protected)
        write_sessions(out, sessions)
        for name, selection in selections.items(): put(out/name/'selection.json', selection)
        put(out/'phase2_plan.json', plan)
        record = dict(schema=SCHEMA, status='PASS_CPU_PREPARED', plan_id=plan['plan_id'],
                      real_data_validation='PENDING_CPU_VALIDATE', invocations=150, future_gpu_ready=False,
                      TEST_FIXTURE_ONLY=w['TEST_FIXTURE_ONLY'],
                      gpu_physical_hits='NOT_MEASURED', measured_policy_outcomes='NOT_MEASURED',
                      tensor_equivalence='NOT_MEASURED', attention_impact='NOT_MEASURED', e2e_latency='NOT_MEASURED')
        put(out/'status.json', record); seal(out); verify(out); return record
    except (ValueError, KeyError, TypeError, FileNotFoundError, ImportError) as exc:
        if out is None: out = output(output_dir, protected)
        return preparation_failure(out, exc)


def load_plan(experiment):
    receipt = verify(experiment)
    require(json.loads((Path(experiment)/'status.json').read_bytes())['status'] == 'PASS_CPU_PREPARED', 'preparation did not complete')
    p = json.loads((Path(experiment)/'phase2_plan.json').read_bytes()); content = dict(p); pid = content.pop('plan_id')
    require(p['schema'] == SCHEMA and digest(content) == pid and p['implementation'] == implementation(), 'Phase 2 plan/source differs')
    require(p['context_mode'] == EXACT and p['validation_mode'] == SCOPED and p['allow_approximate'] is False and
            p['thresholds'] == THRESHOLDS and p['conditions'] == list(CONDITIONS) and p['warmup_requests'] == 0 and
            p['source_invocations'] == 150 and p['distinct_samples_per_dataset'] == 40 and
            p['measurement_scope'] == 'CPU_ONLY_NO_GPU_APPROVAL', 'Phase 2 safety/scope differs')
    loaded, baseline, datasets, phase1_receipt = load_original(p['original_experiment'], p['multi'], p['phase1_audit'])
    w, d, original_receipt, inventories, m, pm, specs, users, tok = loaded
    require(p['TEST_FIXTURE_ONLY'] == w['TEST_FIXTURE_ONLY'], 'workload fixture classification differs')
    require(p['phase1_inventory'] == phase1_receipt and p['original_inventory'] == original_receipt and
            p['dataset_inventories'] == inventories and p['multi_inventory'] == Evidence(p['multi']).verify('r5_output_manifest.json') and
            p['dataset_preparations'] == {name: d[name]['prepared_path'] for name in DATASETS}, 'Phase 2 inputs changed')
    sessions, selections = expand(w, d, baseline, datasets, tok.tokenizer, **p['recipe'])
    require(read_sessions(experiment) == sessions and p['session_digests'] == {s['dataset']: digest(s) for s in sessions}, 'Phase 2 invocation contents differ')
    for name, expected in selections.items():
        require(json.loads((Path(experiment)/name/'selection.json').read_bytes()) == expected, 'Phase 2 selection receipt differs')
    return p, receipt, loaded, sessions


def cpu_validate(experiment, output_dir):
    cpu_only(); out = None; protected = [experiment]
    try:
        if not Path(experiment).is_dir(): raise Blocked('missing sealed Phase 2 preparation: '+str(experiment))
        verify(experiment)
        preflight = json.loads((Path(experiment)/'phase2_plan.json').read_bytes())
        protected += [preflight[k] for k in ('original_experiment', 'multi', 'phase1_audit')]
        protected += list(preflight['dataset_preparations'].values())
        p, receipt, loaded, sessions = load_plan(experiment)
        w, d, original_receipt, inventories, m, pm, specs, users, tok = loaded
        out = output(output_dir, protected)
        from .r5_i5r3b_transport import predicted_request
        identities = {}
        for session in sessions:
            for i in session['invocations']:
                t = runtime_task(i, CONDITIONS[1])
                identities[t['adapter_id']] = predicted_request(t, pm, specs[t['adapter_id']], tok.identity)['physical_identity']
        life = lifecycle(sessions, users, identities); summaries = []
        for session in sessions:
            name = session['dataset']; inv = session['invocations']; tasks = [i['task'] for i in inv]
            regions = [token_regions(t['prepared'], tok.tokenizer) for t in tasks]
            sharing_report = analyze(session, users, regions)
            baseline = dict(session, invocations=inv[:20]); phase1 = analyze(baseline, users, regions[:20])
            for ss, current in ((baseline, phase1), (session, sharing_report)):
                legacy = static_analysis(ss, users)
                require(legacy['occurrence_counts'] == {c: current['views']['replay_inclusive']['axes']['aggregate']['all'][c]['raw_matching_occurrence_pairs'] for c in CATEGORIES},
                        'Phase 1/2 category definitions differ')
            require(sharing_report['views'][SCOPES[0]]['denominators']['ordered_request_pairs'] == 1225 and
                    sharing_report['views'][SCOPES[1]]['denominators']['ordered_request_pairs'] == 1215 and
                    sharing_report['views'][SCOPES[2]]['denominators']['ordered_request_pairs'] == 780, 'sharing denominators differ')
            for label, report in (('phase2', sharing_report), ('phase1_consistent_metrics', phase1)):
                for n, pair in enumerate(report.pop('pairs')): put(out/name/label/'pairs'/f'{n:04d}.json', pair)
                put(out/name/label/'sharing_summary.json', report)
            put(out/name/'regions.json', dict(invocations=[dict(invocation_id=i['invocation_id'], **r) for i, r in zip(inv, regions)]))
            forecast = resource_assessment(tasks)
            # Existing I5 5s/forward assumption; excludes initialization, strict
            # evidence, policy overhead and scoring. Not a runtime measurement.
            forecast.update(runtime_estimate_seconds_per_condition=5*forecast['forwards_per_condition'],
                runtime_estimate_seconds_paired=10*forecast['forwards_per_condition'],
                runtime_estimate_assumption='I5 planning assumption 5 seconds/forward, not measured, excludes extra overhead',
                adapter_request_counts=dict(Counter(t['adapter_id'] for t in tasks)), pbr_interval_per_adapter=100,
                pbr_trigger_reachable=False)
            forecast['per_adapter_es_dense_forecast_bytes'] = {a: v-forecast['per_adapter_ud_dense_payload_bytes'][a]
                for a, v in forecast['per_adapter_forecast_bytes'].items()}
            forecast['limit_checks'].update(
                runtime_worker_planning_estimate=forecast['runtime_estimate_seconds_per_condition'] <= 7200,
                runtime_paired_planning_estimate=forecast['runtime_estimate_seconds_paired'] <= 14460)
            network = {c: transport([runtime_task(i, c) for i in inv], pm, specs, tok.identity, users) for c in CONDITIONS}
            summary = dict(dataset=name, invocations=50, distinct_original_samples=40, preserved_phase1_invocations=20,
                new_distinct_samples=30, checkpoints=len(session['checkpoints']), internal_cache_resets=0,
                distinct_original_contexts=len({digest(t['prepared']['sample']['context']) for t in tasks}),
                provenance_original_blocks=session['original_block_indices'], resource_assessment=forecast,
                transport=network, region_mapping=sharing_report['region_mapping'],
                comparison={scope: dict(phase1=phase1['views'][scope]['axes']['aggregate'],
                                        phase2=sharing_report['views'][scope]['axes']['aggregate']) for scope in SCOPES})
            put(out/name/'summary.json', summary); summaries.append(summary)
        record = dict(schema=SCHEMA, status='PASS',
            real_data_validation='TEST_FIXTURE_ONLY' if p['TEST_FIXTURE_ONLY'] else 'PASS',
            TEST_FIXTURE_ONLY=p['TEST_FIXTURE_ONLY'], plan_id=p['plan_id'],
            plan_inventory=receipt, lifecycle=life, datasets=summaries, future_gpu_ready=False,
            gpu_physical_hits='NOT_MEASURED', measured_policy_outcomes='NOT_MEASURED',
            tensor_equivalence='NOT_MEASURED', attention_impact='NOT_MEASURED', e2e_latency='NOT_MEASURED')
        record['combined_ac_evidence_forecast_bytes'] = sum(s['resource_assessment']['paired_run_evidence_forecast_bytes'] for s in summaries)
        record['combined_evidence_file_forecast'] = sum(s['resource_assessment']['estimated_run_files'] for s in summaries)
        put(out/'phase2_validation.json', record); write_report(out, record)
        seal(out); verify(out); return {k: v for k, v in record.items() if k != 'datasets'}
    except (ValueError, KeyError, TypeError, FileNotFoundError, ImportError) as exc:
        if out is None: out = output(output_dir, protected)
        return preparation_failure(out, exc)


def write_report(out, record):
    lines = ['# D5 Continuous NATURAL Phase 2 CPU report', '', '**'+record['status']+'** CPU validation; GPU readiness: false.', '',
        '150 invocations; 40 original samples / 50 invocations per dataset. The original first 20 invocation envelopes are unchanged.', '',
        'Physical HITs, measured policy outcomes, tensor equivalence, attention Impact and latency: NOT_MEASURED.', '',
        'Counts are static matching occurrences. Context-to-context counts require verified offsets on BOTH windows; UNKNOWN/mixed windows are excluded from those conservative lower bounds.', '',
        'Distinct-sample-only excludes 10 same-sample replay pairs but retains invocation multiplicity (1,215 pairs). First-occurrence-per-sample also removes repeat weighting (780 pairs). Replay-inclusive: 1,225 pairs.', '',
        '| Dataset / scope | P1 shifted nonprefix | P2 shifted nonprefix | P2 verified context/context | P2 authorized context/context | P2 unknown region |',
        '|---|---:|---:|---:|---:|---:|']
    c = 'shifted_position_nonprefix_windows'
    for d in record['datasets']:
        for scope in SCOPES:
            a, b = d['comparison'][scope]['phase1'], d['comparison'][scope]['phase2']
            selections = ('all', 'verified_context_to_context', 'authorized_verified_context_to_context', 'unknown_region')
            values = [b[s][c]['raw_matching_occurrence_pairs'] for s in selections]
            lines.append('| '+d['dataset']+' / '+scope+' | '+str(a['all'][c]['raw_matching_occurrence_pairs'])+' | '+' | '.join(map(str, values))+' |')
        lines += ['', '## '+d['dataset'], '', 'Region mapping: '+str(d['region_mapping']), '',
                  'Resource assessment: '+str(d['resource_assessment'])]
        lines += ['', '| Scope / category | Raw occurrence pairs | Unique target windows | Non-overlap token coverage | Authorized pairs | Verified document pairs |',
                  '|---|---:|---:|---:|---:|---:|']
        for scope in SCOPES:
            axes = d['comparison'][scope]['phase2']
            for category in CATEGORIES:
                metric = axes['all'][category]
                lines.append('| '+scope+' / '+category+' | '+str(metric['raw_matching_occurrence_pairs'])+' | '+
                    str(metric['unique_target_window_positions'])+' | '+str(metric['nonoverlapping_target_token_coverage'])+' | '+
                    str(axes['authorized'][category]['raw_matching_occurrence_pairs'])+' | '+
                    str(axes['verified_context_to_context'][category]['raw_matching_occurrence_pairs'])+' |')
        if d['dataset'] == 'multi_news':
            axes = d['comparison'][SCOPES[2]]['phase2']
            doc = axes['verified_context_to_context'][c]; raw = axes['all'][c]['raw_matching_occurrence_pairs']
            lines += ['', 'MultiNews, first occurrence per original sample: '+str(doc['raw_matching_occurrence_pairs'])+
                ' verified document-to-document shifted nonprefix occurrence pairs out of '+str(raw)+
                ' total shifted nonprefix occurrence pairs; '+str(doc['unique_target_window_positions'])+
                ' unique target windows; '+str(doc['nonoverlapping_target_token_coverage'])+
                ' tokens in greedy non-overlapping document windows. Unknown-region occurrence pairs: '+
                str(axes['unknown_region'][c]['raw_matching_occurrence_pairs'])+'. This is a conservative static count, not measured reuse.']
    lines += ['', 'MultiNews: use the distinct-sample-only and first-occurrence context/context columns to assess document sharing beyond replay/template overlap. If offsets are UNKNOWN, template exclusion is unresolved; zero verified context counts do not prove absence of document sharing.', '',
        'The tables do not establish residency, admission, per-layer input equality or physical reuse. Existing EXACT rejects same-position nonprefix causal mismatches and all shifted windows. No physical restriction or GPU runner changed.']
    (out/'phase2_report.md').write_text('\n'.join(lines)+'\n')
