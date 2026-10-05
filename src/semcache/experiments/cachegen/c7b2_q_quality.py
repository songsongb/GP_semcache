"""Bounded frozen32 incremental resident-Q quality gate; no profile selection."""
import argparse
import csv
import importlib.metadata
import json
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from . import c6b3_2_multiwoz as b3
from . import c7b_q_capture as b0
from . import c7b_q_profiles as b1
from .c7b2_runtime import MODES, CANDIDATES, Backend
from .c6_quality import generation_metrics, logit_metrics, BLEU, PROFILE, PROFILE_SHA, write_csv
from .c6b3_capability import aggregate

require, sha, read = b0.require, b0.sha, b0.read
FREEZE_SHA = '45259f01a920371654122f6c8e3c27eb3aac39ae959c08fbe215d7a7d839c8a9'
SOURCE_SHA = 'b3a5b00397348a48dbeed0f49a2b016b19e73fa6ca00c97b8d4a776ea3a33e14'
SEMANTIC_SHA = 'a8cd31863c9aff560dacda29fe65fd009500dc897b02be92635b34ae8907f665'
CAPTURE_SHA = '2141a4413634030015bf1cc4c2d2749a79bd361c9b1cd5f5174dd97ea1fa1b70'
COHORT_SHA = '842c3e411637a8f22ccf88b07a8b3b6f1a1b9968d318830bb7d1ccf2b3ce5012'
Q_SHAS = dict(Q24='ca82734cb720a7cc27ce21bcbf214415e296d2bf4a57182eb9b2b88caa075885',
              Q32='37071f85757bb4a38973eb917d807322de27da89e76a1774ad8dab726b80a788')
CANONICAL_BLEU = 3.1478254770301533
CONTRACT = 'PHYSICAL_EXACT_W3_WITHIN_SEMANTIC_CLUSTER'


def check_hash(path, expected):
    require(path.is_file() and sha(path) == expected, 'Artifact hash mismatch/missing: '+str(path))


def bound_outputs(root, manifest):
    require(manifest.get('output_hashes'), 'Missing output hash bindings: '+str(root))
    files = {}
    for name, expected in manifest['output_hashes'].items():
        path = Path(name)
        if not path.is_absolute():
            path = root/path
        check_hash(path, expected)
        files[str(path.resolve())] = expected
    return files


def verify_b1(root):
    """Read calibration provenance/profiles; never load a fitter or evaluate it."""
    m = read(root/'manifest.json')
    expected = dict(stage='C7-B1', status='COMPLETE', q_candidates=[16,20,24,32], fit_blocks=96,
        candidate_select_blocks=32, fit_select_conversation_overlap_count=0, frozen32_overlap_count=0,
        capability64_overlap_count=0, runtime_profile_fit_count_on_candidate_select=0,
        downstream_quality_evaluated=False, q_profile_frozen=False, selected_q_candidate=None,
        existing_kv_profile_modified=False, model=b0.MODEL, model_revision=b0.REVISION,
        tokenizer_revision=b0.REVISION, adapter_hashes=b0.WEIGHTS,
        capture_artifact_sha256=CAPTURE_SHA, cohort_sha256=COHORT_SHA)
    b3.check_fields(m, expected, 'C7-B1')
    b3.check_fields(m['transform_provenance'], dict(transform=b1.TRANSFORM, role_generic_verified=True), 'Q transform')
    require(m['existing_kv_profile']['sha256'] == PROFILE_SHA, 'Calibration KV profile changed')
    files = bound_outputs(root, m)
    for path, expected_sha in ((root/'cohort.json', COHORT_SHA), (root/'capture/q_blocks.pt', CAPTURE_SHA),
        (root/'capture/capture_manifest.json', m['capture_manifest_sha256'])):
        check_hash(path, expected_sha)
        files[str(path.resolve())] = expected_sha
    cohort = read(root/'cohort.json')
    b0.validate_cohort(cohort)
    capture = read(root/'capture/capture_manifest.json')
    b3.check_fields(capture, dict(stage='C7-B0', status='COMPLETE', model=b0.MODEL, model_revision=b0.REVISION,
        tokenizer_revision=b0.REVISION, prompt_version=b0.VERSION, adapter_hashes=b0.WEIGHTS,
        cohort_sha256=COHORT_SHA, layers=32, hidden_dim=2560, dtype='float16', blocks=128,
        frozen32_overlap_count=0, capability64_overlap_count=0, fit_select_conversation_overlap_count=0), 'C7-B0')
    require(capture['output_hashes'].get('q_blocks.pt') == CAPTURE_SHA and
            capture['input_hashes'] == cohort['provenance']['input_hashes'], 'Capture hash binding changed')
    files.update(capture['input_hashes'])
    files.update(m['transform_provenance']['source_hashes'])
    files[str((root/'manifest.json').resolve())] = sha(root/'manifest.json')
    b0.verify_files(files)
    profiles = {}
    fit = [r for r in cohort['blocks'] if r['split'] == 'profile_fit']
    for candidate in CANDIDATES:
        path = root/'profiles'/(candidate.lower()+'.bin')
        check_hash(path, Q_SHAS[candidate])
        require(m['profile_hashes'].get(candidate) == Q_SHAS[candidate] and
                files.get(str(path.resolve())) == Q_SHAS[candidate], 'Unbound Q profile: '+candidate)
        profiles[candidate] = path, Q_SHAS[candidate]
    return m, cohort, profiles, files, fit


def verify_score(actual, saved):
    for key in ('version','signature','tokenizer','smoothing','effective_order','lowercase','scope','implementation'):
        require(actual[key] == saved[key], 'Baseline SacreBLEU environment/protocol mismatch: '+key)
    require(math.isfinite(actual['value']) and abs(actual['value']-saved['value']) < 1e-10,
            'Baseline BLEU mismatch')


def read_c6_baseline(root, rows, episodes, provenance, official):
    m = read(root/'manifest.json')
    b3.check_fields(m, dict(stage='C6-B3-2', status='COMPLETE', physical_safety_contract=CONTRACT,
        model=b0.MODEL, revision=b0.REVISION, prompt_version=b0.VERSION, seed=42, dtype='float16',
        storage_profile_sha256=PROFILE_SHA, generation=b3.GENERATION, bleu_protocol=BLEU), 'C6-B3-2 baseline')
    for key in ('training_manifest_sha256','freeze_decision_sha256','evaluation_selection_sha256',
                'semantic_workload_sha256','plan_manifest_sha256','adapter_hashes'):
        require(m[key] == provenance[key], 'Canonical baseline binding mismatch: '+key)
    b3.check_fields(m['model_tokenizer_provenance'], dict(resolved_model_revision=b0.REVISION,
        resolved_tokenizer_revision=b0.REVISION), 'C6 resolved revisions')
    require(all(name in m.get('output_hashes',{}) for name in ('per_case.csv','summary.json')),
            'Canonical C6 inputs lack required output hash bindings')
    files = bound_outputs(root, m)
    # Only storage code executes here. Historical transport code is context,
    # not an input dependency of this storage-isolation experiment.
    source = m.get('codec_source_hashes',{}).get('storage')
    if source:
        check_hash(Path(source['path']), source['sha256'])
        files[str(Path(source['path']).resolve())] = source['sha256']
    if m.get('storage_source_tree_hashes'):
        storage_root = Path(m['codec_source_hashes']['storage']['path']).parents[1]
        for name, expected in m['storage_source_tree_hashes'].items():
            path = storage_root/name
            check_hash(path,expected)
            files[str(path.resolve())] = expected
    files[str((root/'manifest.json').resolve())] = sha(root/'manifest.json')
    with (root/'per_case.csv').open(newline='') as stream:
        cases = list(csv.DictReader(stream))
    baseline = [c for c in cases if c['mode'] == 'STORAGE_KV_COMP']
    full = [c for c in cases if c['mode'] == 'FULL_RECOMPUTE']
    require(len(baseline) == len(full) == len(episodes) == 32, 'Canonical C6 case count changed')
    for e, c, f, o in zip(episodes, baseline, full, official):
        for key, value in e.items():
            # These are the same CSV episode fields emitted by C6's writer.
            if isinstance(value, (dict,list)):
                matches = json.loads(c[key]) == value
            else:
                matches = c.get(key) == ('' if value is None else str(value))
            require(matches, 'Canonical C6 frozen episode changed: '+key)
        for case in (c, f):
            case['generated_token_ids'] = json.loads(case['generated_token_ids'])
            require(case['reference_text'] == rows[e['target_index']]['reference_text'], 'Canonical reference changed')
            require(case['user'] == e['target_user'], 'Canonical adapter assignment changed')
        require(f['generated_token_ids'] == o['generated_token_ids'] and f['generated_text'] == o['generated_text'],
                'C6 canonical teacher continuation changed')
        require(c['logical_event_hash'] == b0.digest(e), 'Canonical C6 logical HIT changed')
    summary = read(root/'summary.json')
    saved = next(r for r in summary['modes'] if r['mode'] == 'STORAGE_KV_COMP')
    require(abs(saved['corpus_bleu']['value']-CANONICAL_BLEU) < 1e-10, 'Canonical C6 KV BLEU differs')
    # Check installed SacreBLEU against saved canonical outputs before GPU work.
    texts = [dict(c, history_depth=int(c['history_depth']), generated_length=int(c['generated_length']),
        normalized_edit_distance=float(c['normalized_edit_distance']), position_agreement=float(c['position_agreement'])) for c in baseline]
    verify_score(aggregate(texts)['corpus_bleu'], saved['corpus_bleu'])
    context = [r for r in summary['modes'] if r['mode'] in ('FULL_RECOMPUTE','RAW_SEMCACHE','STORAGE_KV_COMP')]
    return baseline, saved, files, context


def prepare(args):
    required = {args.freeze_decision: FREEZE_SHA, args.source: SOURCE_SHA, args.semantic: SEMANTIC_SHA,
        args.plan_dir/'evaluation_selection.json': b3.SELECTION_SHA, args.kv_profile: PROFILE_SHA}
    for path, expected in required.items():
        check_hash(path, expected)
    # Reuse C6-B3-2's full adapter/plan/freeze/canonical FULL validation. Its
    # prepare function audits the frozen selection; it never reselects it.
    decision = read(args.freeze_decision)
    c6args = SimpleNamespace(**vars(args))
    c6args.profile_path = args.kv_profile
    c6args.baseline_root = Path(decision['final_output_root'])
    rows, episodes, official, _, provenance = b3.prepare(c6args)
    provenance.update(adapter_hashes=read(args.adapter_root/'training_manifest.json')['adapter_hashes'])
    calibration, cohort, profiles, files, fit = verify_b1(args.q_calibration_root)
    hold = {e[s+'_conversation_id'] for e in episodes for s in ('source','target')}
    require(not hold & {c['conversation_id'] for c in cohort['blocks']}, 'Calibration/frozen32 leakage')
    require(cohort['provenance']['dataset_workload_hashes']['plan_manifest_sha256'] == provenance['plan_manifest_sha256'],
            'Calibration belongs to a different B3 plan')
    require(cohort['provenance']['adapter_hashes'] == b0.WEIGHTS, 'Calibration adapter weights differ')
    baseline, saved, baseline_files, imported = read_c6_baseline(args.c6b3_baseline_root, rows, episodes, provenance, official)
    files.update(baseline_files)
    files.update({str(p.resolve()): expected for p, expected in required.items()})
    # Include canonical continuation and decision-bound capability evidence.
    for root, name in ((c6args.baseline_root, 'capability_manifest.json'),
                       (Path(decision['capability_manifest_path']).parent, Path(decision['capability_manifest_path']).name)):
        path = root/name
        files[str(path.resolve())] = sha(path)
        files.update(bound_outputs(root, read(path)))
    continuation_file = c6args.baseline_root/'capability_per_case.csv'
    provenance['teacher_forced_canonical_source'] = dict(mode='FULL_RECOMPUTE',
        per_case_path=str(continuation_file.resolve()),per_case_sha256=sha(continuation_file),
        manifest_path=str((c6args.baseline_root/'capability_manifest.json').resolve()),
        manifest_sha256=provenance['canonical_frozen32_manifest_sha256'],
        adapter_freeze_decision_sha256=FREEZE_SHA)
    backend = b1.load_backend(args.storage_src)
    require(backend['provenance']['source_hashes'] == calibration['transform_provenance']['source_hashes'],
            'Q coding backend differs from B1')
    # Read frozen Q metadata and verify its fit cohort membership, never fit.
    for candidate, (path, _) in profiles.items():
        profile = b1.QProfile.from_bytes(path.read_bytes(), backend)
        b3.check_fields(profile.metadata, dict(bins=int(candidate[1:]), layers=32, tokens=3, hidden=2560,
            transform=b1.TRANSFORM, fit_cohort_sha256=b0.digest(fit),
            fit_block_ids=[r['calibration_id'] for r in fit]), 'Q profile fit provenance')
    return SimpleNamespace(rows=rows, episodes=episodes, official=official, baseline=baseline,
        baseline_summary=saved, imported_context=imported, profiles=profiles, q_backend=backend,
        input_hashes=files, provenance=provenance)


def baseline_case_check(actual, canonical):
    text = actual['generated_text'] == canonical['generated_text']
    tokens = actual['generated_token_ids'] == canonical['generated_token_ids']
    require(text and tokens, 'INVALID: STORAGE_KV_COMP baseline generation mismatch')
    return dict(generated_text_match=text, generated_token_id_match=tokens)


def storage_summary(cases):
    result = {}
    keys = ('raw_q_bytes','raw_k_bytes','raw_v_bytes','raw_qkv_bytes','resident_raw_q_bytes',
        'compressed_q_bitstream_bytes','local_q_metadata_bytes','compressed_q_frame_bytes',
        'compressed_kv_frame_bytes','local_kv_metadata_bytes','total_resident_qkv_bytes',
        'incremental_resident_byte_reduction_vs_kv_baseline')
    for mode in MODES:
        group = [c['storage_accounting'] for c in cases if c['mode'] == mode]
        totals = {k: sum(c[k] for c in group) for k in keys}
        raw, resident = totals['raw_qkv_bytes'], totals['total_resident_qkv_bytes']
        result[mode] = dict(totals=totals, mean_resident_bytes=resident/len(group), total_resident_bytes=resident,
            whole_qkv_compression_ratio=raw/resident, whole_qkv_byte_reduction_percentage=100*(1-resident/raw),
            q_compression_ratio=totals['raw_q_bytes']/(totals['resident_raw_q_bytes']+totals['compressed_q_frame_bytes']),
            shared_profile_bytes_charged_per_entry=0)
    return result


def results(cases, paired, saved_baseline, shared_profiles, imported_context):
    storage = storage_summary(cases)
    modes = []
    for mode in MODES:
        group = [c for c in cases if c['mode'] == mode]
        require(len(group) == 32, 'Each mode must independently execute32 cases')
        quality = aggregate(group)
        modes.append(dict(mode=mode, cases=32, corpus_bleu=quality['corpus_bleu'],
            user_a_bleu=quality['user_bleu'].get('user_a'), user_b_bleu=quality['user_bleu'].get('user_b'),
            history_depth_bleu=quality['history_depth_bleu_diagnostic'], storage=storage[mode],
            raw_q_resident_after_insert=mode == MODES[0]))
    verify_score(modes[0]['corpus_bleu'], saved_baseline['corpus_bleu'])
    require(abs(modes[0]['corpus_bleu']['value']-CANONICAL_BLEU) < 1e-10, 'INVALID: canonical KV baseline BLEU changed')
    pairs = []; causal = []
    for mode in MODES[1:]:
        group = [p for p in paired if p['mode'] == mode]
        require(len(group) == 32, 'Missing paired cases')
        exact = sum(p['generation_fidelity']['exact_generation'] for p in group)
        teacher = {key: mean(p['teacher_forced'][key] for p in group)
            for key in ('mean_kl','mean_logit_cosine','top1_agreement','top5_agreement')}
        pairs.append(dict(baseline_mode=MODES[0], mode=mode, cases=32, exact_generation_match_count=exact,
            exact_generation_match_rate=exact/32,
            mean_normalized_edit_distance=mean(p['generation_fidelity']['normalized_edit_distance'] for p in group),
            mean_token_position_agreement=mean(p['generation_fidelity']['position_agreement'] for p in group),
            teacher_forced=teacher, aggregation='unweighted means of per-episode metrics'))
        index = MODES.index(mode)
        modes[index]['exact_generation_match_count_vs_kv_baseline'] = exact
        causal.append(dict(baseline_mode=MODES[0], mode=mode,
            baseline_bleu=modes[0]['corpus_bleu']['value'], mode_bleu=modes[index]['corpus_bleu']['value'],
            delta_bleu=modes[index]['corpus_bleu']['value']-modes[0]['corpus_bleu']['value'],
            effect='incremental resident TOTAL-Q compression; fixed K20/V16; transport disabled'))
    return dict(executed_c7b2_modes=list(MODES), modes=modes, paired_quality=pairs,
        causal_comparisons=causal, storage_accounting=storage, shared_profile_bytes=shared_profiles,
        imported_context=imported_context, q_profile_frozen=False, selected_q_candidate=None, manual_freeze_required=True)


def write_outputs(root, summary, cases, paired, manifest):
    for name, value in (('summary.json',summary), ('paired_quality.json',dict(per_case=paired, aggregates=summary['paired_quality'])),
        ('causal_comparisons.json',summary['causal_comparisons']),
        ('storage_accounting.json',dict(modes=summary['storage_accounting'], shared_profile_bytes=summary['shared_profile_bytes'],
            scope='sum of isolated single-entry source caches across frozen32; not simultaneous cache occupancy',
            accounting_policy='KV frame includes local KV metadata; Q frame includes local Q metadata; never double-charge metadata'))):
        b0.write(root/name, value)
    write_csv(root/'per_case.csv', cases)
    write_csv(root/'summary.csv', summary['modes'])
    lines = ['# C7-B2 frozen32 resident-Q quality gate', '',
        'The HIT unit remains the exact w=3 token subsequence within a semantic cluster. Q/K/V remain the shared SemCache payload.',
        'The baseline already contains frozen K20/V16 compression. Q24/Q32 comparisons isolate the incremental effect of compressing resident TOTAL Q.',
        'Transport compression is disabled. Q24/Q32 were fitted previously on C7-B1 and are never refitted here.',
        'Frozen32 is used only as this bounded quality gate. No profile is automatically frozen. A manual freeze decision is required after reviewing this stage.',
        'Positive BLEU perturbation must not be interpreted as evidence that compression improves task quality.', '',
        '| Mode | BLEU | Delta vs KV baseline | Resident ratio | Mean resident bytes |', '|---|---:|---:|---:|---:|']
    for i, mode in enumerate(summary['modes']):
        delta = 0 if i == 0 else summary['causal_comparisons'][i-1]['delta_bleu']
        lines.append(f"| {mode['mode']} | {mode['corpus_bleu']['value']:.10g} | {delta:.10g} | {mode['storage']['whole_qkv_compression_ratio']:.7g} | {mode['storage']['mean_resident_bytes']:.10g} |")
    lines += ['', 'Generation fidelity and canonical teacher-forced fidelity are reported separately in paired_quality.json.',
        'Shared profile bytes are separate from resident-entry bytes. Imported C6 context modes were not executed in C7-B2.', '']
    (root/'summary.md').write_text('\n'.join(lines))
    manifest.update(status='COMPLETE', baseline_matches_c6b3_2=True,
        baseline_reproduction=dict(generated_text_match_count=32, generated_token_id_match_count=32, BLEU_match=True),
        output_hashes={p.name: sha(p) for p in root.iterdir() if p.is_file() and p.name != 'manifest.json'})
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n')


def run(args, prepared, manifest):
    backend = Backend(args, prepared.profiles, prepared.q_backend)
    backend.rows = prepared.rows
    manifest.update(model_inference_performed=True, counters=backend.counts,
        model_tokenizer_provenance=backend.metadata, shared_profile_bytes=backend.shared_profile_bytes,
        q_codec_provenance=prepared.q_backend['provenance'],
        software={name: importlib.metadata.version(name) for name in ('torch','transformers','peft','sacrebleu')})
    cases, paired = [], []
    for index, (e, official, previous) in enumerate(zip(prepared.episodes, prepared.official, prepared.baseline)):
        canonical = list(official['generated_token_ids'])
        require(0 < len(canonical) <= 160, 'Invalid canonical teacher continuation')
        episode_cases = {}; baseline_logits = None
        for mode in MODES:
            before = dict(backend.counts); backend.reuse_audits.clear()
            context = backend.prepare(e, mode)
            identity = backend.event_identity(context, e)
            require(identity == b0.digest(e), 'Logical frozen32 HIT changed')
            tokens, text = backend.greedy(context, e)
            generated = dict(generated_token_ids=tokens, generated_text=text)
            if mode == MODES[0]:
                baseline_case_check(generated, previous)  # Fail before candidate work.
            logits = backend.teacher_logits(context, e, canonical)
            require(backend.event_identity(context, e) == identity, 'Logical HIT changed during execution')
            delta = {k: backend.counts[k]-before[k] for k in before}
            require(delta['source_forward_count'] == delta['storage_decode_count'] == 1, 'Source/lookup count changed')
            require(delta['runtime_q_profile_fit_count'] == delta['runtime_storage_kv_cdf_fit_count'] == 0
                and delta['transport_encode_calls'] == delta['transport_decode_calls'] == 0, 'Forbidden fitting/transport')
            require(delta['q_encode_count'] == delta['q_decode_count'] == int(mode != MODES[0]), 'Q codec count changed')
            require(len(backend.reuse_audits) == 2, 'Greedy prefill and teacher forward must both reuse QKV')
            encoded = b3.d.encode_example(backend.tokenizer, prepared.rows[e['target_index']])
            fidelity = generation_metrics(encoded['input_ids'][encoded['prompt_length']:], tokens)
            case = dict(e, episode_index=index, mode=mode, user=e['target_user'],
                reference_text=prepared.rows[e['target_index']]['reference_text'], **generated, **fidelity,
                logical_event_hash=identity, storage_accounting=context.accounting, runtime_counters=delta,
                reuse_audit=list(backend.reuse_audits), teacher_forced_canonical_token_ids=canonical,
                teacher_forced_canonical_sha256=b0.digest(canonical), generation_source='executed_c7b2_independent_greedy')
            cases.append(case); episode_cases[mode] = case
            if mode == MODES[0]:
                baseline_logits = logits
            else:
                teacher = logit_metrics(baseline_logits, logits)
                teacher['top5_agreement'] = teacher.pop('top5_overlap')
                pair = dict(episode_id=e['episode_id'], baseline_mode=MODES[0], mode=mode,
                    generation_fidelity=generation_metrics(episode_cases[MODES[0]]['generated_token_ids'], tokens),
                    teacher_forced=teacher, canonical_continuation_sha256=b0.digest(canonical))
                require(all(math.isfinite(v) for v in teacher.values()), 'Nonfinite teacher fidelity')
                paired.append(pair)
                case['fidelity_vs_kv_baseline'] = pair
            del logits, context
        del baseline_logits
        print(f'Completed frozen32 {index+1}/32: {e["episode_id"]}', flush=True)
    b0.verify_files(prepared.input_hashes)
    summary = results(cases, paired, prepared.baseline_summary, backend.shared_profile_bytes, prepared.imported_context)
    manifest.update(runtime_q_profile_fit_count=backend.counts['runtime_q_profile_fit_count'],
        runtime_storage_kv_cdf_fit_count=backend.counts['runtime_storage_kv_cdf_fit_count'])
    write_outputs(args.output_root, summary, cases, paired, manifest)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    defaults = dict(adapter_root='results/cachegen/c6b3/b1_full',
        freeze_decision='results/cachegen/c6b3/b1_full_freeze_decision.json',
        plan_dir='results/cachegen/c6b3/multiwoz_plan_v2', source='results/workloads/multiwoz.jsonl',
        semantic='results/workloads/c6b3_multiwoz_history_semantic.jsonl',
        q_calibration_root='results/cachegen/c7/b_q_calibration_retry2', kv_profile=PROFILE,
        storage_src='/data/khuss/repos/GP_semcache/.c6_storage_src/src',
        c6b3_baseline_root='results/cachegen/c6b3/b2_frozen32',
        output_root='results/cachegen/c7/b2_q_quality_gate_frozen32')
    for name, default in defaults.items():
        p.add_argument('--'+name.replace('_','-'), type=Path, default=Path(default))
    p.add_argument('--device', default='cuda:0'); p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)
    require(args.seed == 42 and args.device.startswith('cuda:'), 'Frozen seed42 and one CUDA device required')
    require(Path('/data/khuss').is_dir() and args.output_root.resolve().is_relative_to(Path('/data/khuss')),
            'Real C7-B2 execution requires SERAPH with output under /data/khuss')
    require(not args.output_root.exists(), 'New C7-B2 output root required; no overwrite')
    b0.seraph_paths(args.output_root)
    args.profile_path = args.kv_profile
    args.max_sequence_length = 384; args.max_new_tokens = 160
    prepared = prepare(args)
    m = dict(stage='C7-B2', status='STARTING', research_scope='frozen32 incremental resident TOTAL-Q compression quality gate',
        source_stage='C7-B1', physical_safety_contract=CONTRACT, cached_payload='TOTAL_QKV',
        executed_modes=list(MODES), transport_compression_enabled=False, q_candidates=list(CANDIDATES),
        q_profile_frozen=False, selected_q_candidate=None, manual_freeze_required=True, frozen32_cases=32,
        model=b0.MODEL, model_revision=b0.REVISION, tokenizer_revision=b0.REVISION, prompt_version=b0.VERSION,
        adapter_hashes=b0.WEIGHTS, adapter_file_hashes=prepared.provenance['adapter_hashes'],
        adapter_freeze_decision_sha256=FREEZE_SHA, frozen32_selection_sha256=b3.SELECTION_SHA,
        q_profile_hashes=Q_SHAS, kv_profile_sha256=PROFILE_SHA, c7b1_manifest_sha256=sha(args.q_calibration_root/'manifest.json'),
        source_sha256=SOURCE_SHA, semantic_sha256=SEMANTIC_SHA, capture_artifact_sha256=CAPTURE_SHA, cohort_sha256=COHORT_SHA,
        runtime_q_profile_fit_count=0, runtime_storage_kv_cdf_fit_count=0, baseline_matches_c6b3_2=None,
        paper_bleu_claimed=False, protocol_provenance='REPRODUCTION_CHOICE', bleu_protocol=BLEU, generation=b3.GENERATION,
        teacher_forced_policy='C6-B3-2: identical imported canonical FULL_RECOMPUTE continuation for all three modes; prompt plus canonical[:-1]',
        teacher_forced_continuation_provenance=prepared.provenance,
        model_inference_performed=False, training_performed=False, input_hashes=prepared.input_hashes, git=b0.git(), output_hashes={})
    b0.write(args.output_root/'manifest.json', m)
    try:
        run(args, prepared, m)
    except Exception as exc:
        counts = m.get('counters', {})
        m.update(status='INVALID' if 'baseline' in str(exc).lower() else 'FAILED', error=str(exc),
            runtime_q_profile_fit_count=counts.get('runtime_q_profile_fit_count',0),
            runtime_storage_kv_cdf_fit_count=counts.get('runtime_storage_kv_cdf_fit_count',0))
        if m['status'] == 'INVALID':m['baseline_matches_c6b3_2'] = False
        (args.output_root/'manifest.json').write_text(json.dumps(m, indent=2, sort_keys=True, allow_nan=False)+'\n')
        raise
