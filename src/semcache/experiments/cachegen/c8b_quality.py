"""Selection-based controlled B2/B8 quality replay; provenance and reporting."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from . import c7b3_q_freeze as provenance
from . import c8a_capacity as capacity

require = provenance.require
POLICIES = capacity.POLICIES
BUDGETS = (2, 8)
LABEL = 'SELECTION_BASED_CONTROLLED_QUALITY_REPLAY'
REFERENCE = 'FULL_RECOMPUTE'
# Assertions about the supplied canonical experiment, never substitutes for its trace.
EXPECTED_HITS = {2: (2, 4, 13), 8: (8, 18, 32)}
SAFETY = ('runtime_q_profile_fit_count', 'runtime_storage_kv_cdf_fit_count',
          'transport_encode_calls', 'transport_decode_calls')
MISS_MAX_ABS = 1e-6
MISS_MEAN_ABS = 1e-7
REVIEW_RULE = (
    'Support requires exact C8-A residency/HIT reproduction and faithful MISS paths. '
    'At either budget, concurrent task/generation degradation (lower BLEU AND '
    '(lower exact FULL match rate OR higher edit distance)) and teacher degradation '
    '(higher mean KL AND lower top1 agreement) relative to KV_COMP requires further '
    'review. Otherwise report descriptive support for C9 review only. No weighted '
    'score, paper-quality threshold, or system-policy freeze is used.')


def condition(budget, policy):
    require(budget in BUDGETS and policy in POLICIES, 'Only B2/B8 and RAW/KV/Q24 policies allowed')
    return f'B{budget}_{policy}'


def csv_form(row):
    return {k: json.dumps(v, sort_keys=True, separators=(',', ':')) if isinstance(v, (list, dict))
            else ('' if v is None else str(v)) for k, v in row.items()}


def verify_capacity(args):
    """Revalidate A's hash-bound measurements and replay; never choose new budgets."""
    frozen = provenance.validate_freeze(args.freeze_decision)
    aargs = SimpleNamespace(freeze_decision=args.freeze_decision, plan_dir=args.plan_dir,
        c7b2_root=Path(frozen['b2_manifest_path']).parent)
    episodes, sizes, shared, hashes = capacity.prepare(aargs)
    root = args.c8a_root
    manifest, bindings = provenance.load_manifest(root, dict(stage='C8-A', status='COMPLETE',
        replay_protocol=capacity.PROTOCOL, eviction_policy='BYTE_AWARE_LRU', policies=list(POLICIES),
        budgets_raw_entry_equivalent=[2, 4, 8, 16], target_lookups=32,
        recommended_c8b_budgets_raw_entry_equivalent=list(BUDGETS),
        physical_safety_contract=provenance.CONTRACT, cached_payload='TOTAL_QKV',
        model_inference_performed=False, quality_evaluated=False, latency_evaluated=False,
        transport_compression_enabled=False, c7b3_freeze_decision_sha256=provenance.sha(args.freeze_decision),
        frozen32_selection_sha=provenance.SELECTION_SHA, q24_profile_sha256=provenance.Q_SHA,
        kv_profile_sha256=provenance.KV_SHA), required=('summary.json', 'per_event.csv', 'residency_trace.json'))
    provenance.verify_files(manifest['input_hashes'])
    require(all(manifest['input_hashes'].get(p) == h for p, h in hashes.items()), 'C8-A input chain differs')
    hashes.update(bindings)
    summary = provenance.read(root/'summary.json')
    provenance.fields(summary, dict(replay_protocol=capacity.PROTOCOL, eviction_policy='BYTE_AWARE_LRU',
        policies=list(POLICIES), budgets_raw_entry_equivalent=[2, 4, 8, 16],
        duplicate_key_semantics=capacity.DUPLICATE_RULE,
        recommended_c8b_budgets_raw_entry_equivalent=list(BUDGETS)), 'C8-A summary')
    events = provenance.read_csv(root/'per_event.csv')
    snapshots = provenance.read(root/'residency_trace.json')['runs']
    require(len(summary['results']) == len(snapshots) == 12 and len(events) == 12*64, 'Incomplete C8-A trace')
    expected = {}
    for budget in BUDGETS:
        for policy, expected_hits in zip(POLICIES, EXPECTED_HITS[budget]):
            name = condition(budget, policy)
            rows = [r for r in summary['results'] if r['budget_raw_entry_equivalent'] == budget and r['policy'] == policy]
            traces = [r for r in snapshots if r['budget_raw_entry_equivalent'] == budget and r['policy'] == policy]
            saved_events = [r for r in events if int(r['budget_raw_entry_equivalent']) == budget and r['policy'] == policy]
            require(len(rows) == len(traces) == 1 and len(saved_events) == 64, 'Duplicate/missing C8-A condition: '+name)
            calculated, replay_events, residency = capacity.replay(episodes, sizes[policy], policy,
                budget*capacity.RAW_ENTRY_BYTES, budget)
            require(rows[0] == calculated and traces[0] == residency, 'C8-A resident/summary replay mismatch: '+name)
            require(all(all(saved.get(k) == v for k, v in csv_form(event).items())
                for saved, event in zip(saved_events, replay_events)), 'C8-A HIT/event vector mismatch: '+name)
            require(calculated['target_hits'] == expected_hits, 'Canonical C8-A HIT count differs: '+name)
            expected[name] = dict(summary=calculated, admissions=replay_events[:32],
                lookups=replay_events[32:], residency=residency)
    provenance.verify_files(hashes)
    return SimpleNamespace(frozen=frozen, episodes=episodes, sizes=sizes, shared=shared,
        expected=expected, input_hashes=hashes, c8a_manifest=manifest)


def prepare(args):
    prepared = verify_capacity(args)
    from . import c7b2_q_quality as gate
    from . import c6b3_2_multiwoz as b3
    from . import c7b_q_profiles as qcodec
    from .c7b2_runtime import forbid_fitting, runtime_counts
    provenance.check_hash(args.kv_profile, provenance.KV_SHA)
    require(args.kv_profile.resolve() == Path(prepared.frozen['kv_profile_path']), 'Frozen KV path differs')
    for path, expected in ((args.adapter_freeze_decision, gate.FREEZE_SHA),
                           (args.source, gate.SOURCE_SHA), (args.semantic, gate.SEMANTIC_SHA)):
        provenance.check_hash(path, expected)
        prepared.input_hashes[str(path.resolve())] = expected
    bargs = SimpleNamespace(**vars(args))
    bargs.freeze_decision = args.adapter_freeze_decision
    bargs.baseline_root = args.canonical_full_root
    bargs.profile_path = args.kv_profile
    rows, episodes, official, summary, chain = b3.prepare(bargs)
    require(episodes == prepared.episodes, 'C8-A and canonical FULL selections differ')
    b2 = provenance.read(Path(prepared.frozen['b2_manifest_path']))
    provenance.verify_files(b2['input_hashes'])
    prepared.input_hashes.update(b2['input_hashes'])
    saved = b2['teacher_forced_continuation_provenance']
    for key in ('canonical_frozen32_manifest_sha256', 'plan_manifest_sha256', 'training_manifest_sha256',
                'semantic_workload_sha256', 'freeze_decision_sha256'):
        require(chain[key] == saved[key], 'Canonical FULL provenance differs from C7-B2: '+key)
    canonical_root = args.canonical_full_root
    canonical_manifest = provenance.read(canonical_root/'capability_manifest.json')
    prepared.input_hashes.update(provenance.output_hashes(canonical_root, canonical_manifest,
        ('capability_per_case.csv', 'capability_summary.json', 'capability_validation.json')))
    canonical_path = canonical_root/'capability_per_case.csv'
    require(provenance.sha(canonical_path) == saved['teacher_forced_canonical_source']['per_case_sha256'],
        'Canonical teacher continuation differs from C7-B2')
    for path in (canonical_root/'capability_manifest.json', args.adapter_root/'training_manifest.json',
                 args.adapter_root/'train_selection.json', args.adapter_root/'capability_validation.json'):
        prepared.input_hashes[str(path.resolve())] = provenance.sha(path)
    decision = provenance.read(args.adapter_freeze_decision)
    cap_path = Path(decision['capability_manifest_path'])
    prepared.input_hashes[str(cap_path.resolve())] = provenance.sha(cap_path)
    prepared.input_hashes.update(provenance.output_hashes(cap_path.parent, provenance.read(cap_path)))
    for user in ('user_a', 'user_b'):
        for path in sorted((args.adapter_root/user).rglob('*')):
            if path.is_file(): prepared.input_hashes[str(path.resolve())] = provenance.sha(path)
    counts = runtime_counts()
    with forbid_fitting(counts, 'q'):
        backend = qcodec.load_backend(args.storage_src)
    calibration = provenance.read(Path(prepared.frozen['b1_manifest_path']))
    require(backend['provenance']['source_hashes'] == calibration['transform_provenance']['source_hashes'],
        'Frozen Q backend source differs')
    prepared.input_hashes.update(backend['provenance']['source_hashes'])
    # Recompute only the saved texts' metric to validate the installed BLEU protocol.
    gate.verify_score(b3.mode_summary(REFERENCE, official)['corpus_bleu'], summary['corpus_bleu'])
    prepared.rows, prepared.official, prepared.full_summary = rows, official, summary
    prepared.provenance, prepared.q_backend = chain, backend
    prepared.canonical_source = saved['teacher_forced_canonical_source']
    provenance.verify_files(prepared.input_hashes)
    return prepared


def generation_check(tokens, text, official, hit):
    from .c6_quality import generation_metrics
    metrics = generation_metrics(official['generated_token_ids'], tokens)
    exact_text = text == official['generated_text']
    require(hit or (metrics['exact_generation'] and exact_text), 'INVALID: MISS generation differs from canonical FULL_RECOMPUTE')
    return dict(metrics, exact_text_match=exact_text)


def teacher_metrics(full, actual, hit):
    from .c6_quality import logit_metrics
    result = logit_metrics(full, actual)
    result['top5_agreement'] = result.pop('top5_overlap')
    result['output_margin_delta'] = result['margin_delta']
    require(all(math.isfinite(v) for v in result.values()), 'Nonfinite teacher metrics')
    negligible = result['max_abs_logit_difference'] <= MISS_MAX_ABS and result['mean_abs_logit_difference'] <= MISS_MEAN_ABS
    require(hit or (negligible and result['top1_agreement'] == result['top5_agreement'] == 1),
        'INVALID: MISS teacher logits differ from FULL_RECOMPUTE')
    return dict(result, numerically_equivalent=negligible)


def aggregate(group, full_by_id):
    if not group:
        return dict(cases=0, corpus_bleu=None, reference_subset_bleu=None, teacher_forced=None)
    from semcache.evaluation.bleu import compute_bleu
    from .c6_quality import BLEU
    refs = [r['reference_text'] for r in group]
    score = compute_bleu([r['generated_text'] for r in group], refs, **BLEU)
    full_score = compute_bleu([full_by_id[r['episode_id']]['generated_text'] for r in group], refs, **BLEU)
    teachers = [r['teacher_forced'] for r in group]
    return dict(cases=len(group), corpus_bleu=score, reference_subset_bleu=full_score,
        delta_bleu_vs_full=score['value']-full_score['value'],
        exact_generation_match_count=sum(r['generation_fidelity']['exact_generation'] for r in group),
        exact_generation_match_rate=mean(r['generation_fidelity']['exact_generation'] for r in group),
        mean_normalized_edit_distance=mean(r['generation_fidelity']['normalized_edit_distance'] for r in group),
        mean_token_position_agreement=mean(r['generation_fidelity']['position_agreement'] for r in group),
        teacher_forced={k: mean(r[k] for r in teachers) for k in teachers[0]
                        if k != 'numerically_equivalent'},
        max_abs_logit_difference=max(r['max_abs_logit_difference'] for r in teachers),
        aggregation='unweighted per-episode fidelity means; corpus BLEU recalculated per subset')


def results(cases, official):
    full_by_id = provenance.unique(official, 'episode_id')
    full_cases = [r for r in cases if r['condition'] == REFERENCE]
    require(len(full_cases) == 32 and set(provenance.unique(full_cases, 'episode_id')) == set(full_by_id),
        'Incomplete canonical FULL reference condition')
    reference = dict(condition=REFERENCE, policy=REFERENCE, budget_raw_entry_equivalent=None,
        budget_bytes=None, target_hits=0, hit_rate=0., generation_source='imported_hash_bound_FULL_RECOMPUTE',
        **aggregate(full_cases, full_by_id))
    rows = []
    for budget in BUDGETS:
        for policy in POLICIES:
            name = condition(budget, policy)
            group = [r for r in cases if r['condition'] == name]
            require(len(group) == 32 and set(provenance.unique(group, 'episode_id')) == set(full_by_id),
                'Incomplete condition: '+name)
            subsets = {label: aggregate([r for r in group if r['hit'] is hit], full_by_id)
                       for label, hit in (('HIT', True), ('MISS', False))}
            rows.append(dict(condition=name, policy=policy, budget_raw_entry_equivalent=budget,
                budget_bytes=budget*capacity.RAW_ENTRY_BYTES, target_hits=subsets['HIT']['cases'],
                hit_rate=subsets['HIT']['cases']/32, **aggregate(group, full_by_id), subsets=subsets))
    deltas = []
    for budget in BUDGETS:
        by_policy = {r['policy']: r for r in rows if r['budget_raw_entry_equivalent'] == budget}
        for before, after in (('RAW_QKV', 'KV_COMP'), ('KV_COMP', 'Q24_KV_COMP'), ('RAW_QKV', 'Q24_KV_COMP')):
            a, b = by_policy[before], by_policy[after]
            deltas.append(dict(budget_raw_entry_equivalent=budget, comparison=after+' - '+before,
                target_hits=b['target_hits']-a['target_hits'], hit_rate=b['hit_rate']-a['hit_rate'],
                corpus_bleu=b['corpus_bleu']['value']-a['corpus_bleu']['value'],
                **{k: b[k]-a[k] for k in ('exact_generation_match_rate', 'mean_normalized_edit_distance')},
                teacher_forced_mean_kl=b['teacher_forced']['mean_kl']-a['teacher_forced']['mean_kl'],
                teacher_forced_top1_agreement=b['teacher_forced']['top1_agreement']-a['teacher_forced']['top1_agreement']))
    recommendation, review = review_quality(deltas)
    return dict(evaluation_label=LABEL, reference_mode=REFERENCE, reference=reference, conditions=rows,
        recommendation=recommendation, review_rule=REVIEW_RULE, review_evidence=review), deltas


def review_quality(deltas):
    evidence = []
    for budget in BUDGETS:
        row = next(r for r in deltas if r['budget_raw_entry_equivalent'] == budget and r['comparison'] == 'Q24_KV_COMP - KV_COMP')
        task = row['corpus_bleu'] < 0 and (row['exact_generation_match_rate'] < 0 or row['mean_normalized_edit_distance'] > 0)
        teacher = row['teacher_forced_mean_kl'] > 0 and row['teacher_forced_top1_agreement'] < 0
        evidence.append(dict(budget_raw_entry_equivalent=budget, task_generation_degradation=task,
            teacher_fidelity_degradation=teacher, concurrent_degradation=task and teacher))
    label = 'C8_B_REQUIRES_FURTHER_QUALITY_REVIEW' if any(r['concurrent_degradation'] for r in evidence) else 'C8_B_SUPPORTS_Q24_KV_FOR_C9'
    return label, evidence


def write_reports(root, cases, build, hits, manifest, summary=None, deltas=None):
    """Write partial evidence safely, retaining the original execution failure."""
    values = dict(summary=summary or dict(status=manifest['status'], checks='NOT_COMPLETED',
        recommendation='C8_B_REQUIRES_FURTHER_QUALITY_REVIEW'), cache_build_audit=dict(runs=build),
        hit_audit=dict(cases=hits), paired_quality=dict(reference_mode=REFERENCE,
            cases=[{k: r[k] for k in ('episode_id', 'condition', 'hit', 'generation_fidelity', 'teacher_forced')}
                   for r in cases if r['condition'] != REFERENCE],
            aggregation=None if summary is None else summary['conditions']), causal_comparisons=deltas or [])
    for name, value in values.items():
        (root/(name+'.json')).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    capacity.write_csv(root/'per_case.csv', cases)
    capacity.write_csv(root/'summary.csv', [] if summary is None else [summary['reference'], *summary['conditions']])
    lines = ['# C8-B controlled quality replay', '', LABEL, 'Status: '+manifest['status'], '',
        'Frozen32 was used for Q-profile selection. This is bounded mechanistic evidence on the controlled trace, '
        'not unbiased final generalization evidence. An untouched cohort is required for that claim.',
        'The exact-w3-within-cluster HIT and cached TOTAL Q/K/V payload are unchanged. Transport is disabled.',
        'Storage codec perturbation and capacity-induced HIT coverage are distinct mechanisms.', '',
        'Predeclared review rule: '+REVIEW_RULE, '']
    if summary is not None:
        lines += ['C8-A resident sets and HIT/MISS vectors reproduced exactly: True. All MISS paths faithfully reproduce FULL: True.', '',
            'Canonical imported FULL_RECOMPUTE corpus BLEU: '+str(summary['reference']['corpus_bleu']['value'])+'.', '',
            '| Budget | Policy | Hits / 32 | BLEU | Delta vs FULL | Exact FULL match | Edit distance | Teacher KL | Top1 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in summary['conditions']:
            lines.append(f"| B{r['budget_raw_entry_equivalent']} | {r['policy']} | {r['target_hits']} | {r['corpus_bleu']['value']:.8g} | {r['delta_bleu_vs_full']:.8g} | {r['exact_generation_match_count']}/32 | {r['mean_normalized_edit_distance']:.8g} | {r['teacher_forced']['mean_kl']:.8g} | {r['teacher_forced']['top1_agreement']:.8g} |")
        for evidence in summary['review_evidence']:
            lines.append(f"\nB{evidence['budget_raw_entry_equivalent']} concurrent task/generation and teacher fidelity degradation vs KV_COMP: {evidence['concurrent_degradation']}.")
        lines += ['', 'The table and paired comparisons describe how added reuse coverage changes quality; positive BLEU perturbations do not show compression improves quality.',
        'Support is descriptive evidence for later C9 review, not a system-policy freeze or a latency result.']
    else:
        lines += ['Residency, HIT coverage, quality, and MISS fidelity checks: NOT_COMPLETED.',
            'Primary failure: '+manifest.get('failure_type', '')+': '+manifest.get('failure_message', '')]
    lines += ['', (summary or values['summary'])['recommendation'], '']
    (root/'summary.md').write_text('\n'.join(lines))
    manifest['output_hashes'] = {p.name: provenance.sha(p) for p in root.iterdir() if p.is_file() and p.name != 'manifest.json'}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(adapter_root='results/cachegen/c6b3/b1_full',
        adapter_freeze_decision='results/cachegen/c6b3/b1_full_freeze_decision.json',
        plan_dir='results/cachegen/c6b3/multiwoz_plan_v2', source='results/workloads/multiwoz.jsonl',
        semantic='results/workloads/c6b3_multiwoz_history_semantic.jsonl',
        freeze_decision='results/cachegen/c7/q24_freeze/freeze_decision.json',
        c8a_root='results/cachegen/c8/a_capacity_replay',
        kv_profile='results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin',
        canonical_full_root='results/cachegen/c6b3/b1_full_frozen32',
        storage_src='/data/khuss/repos/GP_semcache/.c6_storage_src/src', output_root='results/cachegen/c8/b_quality_b2_b8')
    for name, default in defaults.items(): parser.add_argument('--'+name.replace('_', '-'), type=Path, default=Path(default))
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    require(args.seed == 42 and args.device.startswith('cuda:'), 'Seed42 and one CUDA device required')
    require(Path('/data/khuss').is_dir() and args.output_root.resolve().is_relative_to(Path('/data/khuss')),
        'GPU execution requires SERAPH with outputs under /data/khuss')
    require(not args.output_root.exists(), 'New C8-B output root required; no overwrite')
    from . import c7b_q_capture as b0
    b0.seraph_paths(args.output_root)
    args.profile_path = args.kv_profile; args.max_sequence_length = 384; args.max_new_tokens = 160
    prepared = prepare(args)
    from . import c8b_runtime as runtime
    runtime.run(args, prepared)
