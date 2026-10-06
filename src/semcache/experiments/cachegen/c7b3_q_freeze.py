"""Manual SELECT_Q24 artifact generator. Standard-library-only, read-only evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

B1_SHA = 'fc8bde2d4d0eb1b2a0a736377dedf9ccb0db49dfcb805db97343e7e09cbb0ab0'
B2_SHA = '0959257d98eb2a0446cf14ba483d1a5743f88894680b9574b191b92630b40509'
Q_SHA = 'ca82734cb720a7cc27ce21bcbf214415e296d2bf4a57182eb9b2b88caa075885'
KV_SHA = '8defa27a83e8ad16e9c7efa0f40e68c302aef90363e14a1451288a3bf433cb2c'
SELECTION_SHA = 'e1b5d2156cc548b712cfb09b805ad33ddab04dcbd7b8d55656333bf7b5ac3876'
CONTRACT = 'PHYSICAL_EXACT_W3_WITHIN_SEMANTIC_CLUSTER'
TRANSFORM = 'B2_ANCHOR_MOD_RESIDUAL_Q'
BASELINE, Q24, Q32 = ('STORAGE_KV_COMP_BASELINE', 'STORAGE_Q24_KV_COMP', 'STORAGE_Q32_KV_COMP')
RATIONALE = ('Q24 provided greater resident compression than Q32 while both showed no observed frozen32 quality degradation; '
    'C7-B2A verified that compressed Q is genuinely consumed and that the observed downstream invariance is causal isolation '
    'in this tested path, not an inactive-Q implementation bug.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def output_location(root):
    path = root.resolve()
    require(not path.is_relative_to(Path('/home/khuss')), 'Runtime artifacts must not be written under /home/khuss')
    if Path('/data/khuss').is_dir():
        require(path.is_relative_to(Path('/data/khuss')), 'SERAPH runtime artifacts must stay under /data/khuss')


def fields(obj, expected, label):
    for key, value in expected.items():
        require(key in obj and obj[key] == value and type(obj[key]) is type(value), label+' mismatch: '+key)


def check_hash(path, expected):
    require(Path(path).is_file() and sha(path) == expected, 'Hash mismatch/missing: '+str(path))


def verify_files(hashes):
    for path, expected in hashes.items():
        check_hash(path, expected)


def output_hashes(root, manifest, required=()):
    require(manifest.get('output_hashes'), 'Missing output bindings: '+str(root))
    files = {}
    for name, expected in manifest['output_hashes'].items():
        path = Path(name)
        path = path if path.is_absolute() else root/path
        check_hash(path, expected)
        files[str(path.resolve())] = expected
    for name in required:
        require(str((root/name).resolve()) in files, 'Unbound output: '+str(root/name))
    return files


def load_manifest(root, expected, pinned_sha=None, required=()):
    path = root/'manifest.json'
    if pinned_sha:
        check_hash(path, pinned_sha)
    m = read(path)
    fields(m, expected, str(path))
    hashes = output_hashes(root, m, required)
    hashes[str(path.resolve())] = sha(path)
    return m, hashes


def unique(rows, field):
    result = {}
    for row in rows:
        require(row[field] not in result, 'Duplicate '+field+': '+str(row[field]))
        result[row[field]] = row
    return result


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def git():
    return {key: subprocess.check_output(['git', *args], text=True).strip() for key, args in
            (('branch', ['branch', '--show-current']), ('commit', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain']))}


def causal_evidence(root, m, baseline_rows):
    selected = read(root/'selected_cases.json')
    fields(selected, dict(status='COMPLETE', frozen32_selection_sha256=SELECTION_SHA), 'B2A selection')
    ordered = sorted(baseline_rows, key=lambda r: int(r['episode_index']))
    first = [next(r for r in ordered if int(r['history_depth']) == depth) for depth in (1, 2, 3, 4)]
    require(len(selected['cases']) == 4, 'B2A requires four cases')
    for i, (case, original) in enumerate(zip(selected['cases'], first)):
        fields(case, dict(episode_id=original['episode_id'], history_depth=i+1, audit_case_index=i,
                         canonical_episode_index=int(original['episode_index'])), 'B2A selected identity')
        for key, value in case.items():
            if key in ('audit_case_index', 'canonical_episode_index', 'canonical_selection_index'):
                continue
            require(key in original, 'B2A metadata missing from canonical B2: '+key)
            actual = json.loads(original[key]) if isinstance(value, (dict, list)) else original[key]
            expected = value if isinstance(value, (dict, list)) else ('' if value is None else str(value))
            require(actual == expected, 'B2A selected metadata differs: '+key)
    ids = {r['episode_id'] for r in first}
    reports = {}
    for name in ('q_distortion', 'injection_audit', 'internal_causal_effect', 'continuation_causal_effect'):
        report = read(root/(name+'.json'))
        fields(report, dict(status='COMPLETE'), name)
        reports[name] = unique(report['cases'], 'episode_id')
        require(set(reports[name]) == ids, 'B2A case coverage mismatch: '+name)
    fields(m['tolerances'], dict(logit_max_abs=1e-6, logit_mean_abs=1e-7, fresh_attention_max_abs=1e-6), 'B2A tolerances')
    for episode_id in ids:
        distortion = reports['q_distortion'][episode_id]
        for candidate in ('Q24', 'Q32'):
            layers = unique(distortion['candidates'][candidate], 'layer')
            require(set(layers) == set(range(32)), 'Incomplete Q distortion layers')
            require(any(r['exact_equal'] is False and r['raw_sha256'] != r['comparison_sha256'] and
                        math.isfinite(r['mse']) and r['mse'] > 0 for r in layers.values()), 'Decoded Q must differ: '+candidate)
        injection = reports['injection_audit'][episode_id]
        fields(injection, dict(q24_injection_verified=True, q32_injection_verified=True, qzero_injection_verified=True,
            hit_mask_equal=True, fresh_q_rows_equal=True, k_rows_equal=True, v_rows_equal=True), 'B2A injection')
        masks = []
        for mode in ('KV_BASELINE', 'Q24', 'Q32', 'Q_ZERO_COUNTERFACTUAL'):
            evidence = injection['modes'][mode]
            fields(evidence, dict(hit_mask_equal=True, q_injection_verified=True, fresh_q_rows_equal=True,
                                 k_rows_equal=True, v_rows_equal=True), mode)
            if mode in ('Q24', 'Q32'):
                fields(evidence, dict(raw_q_resident_after_insert=False), mode+' residency')
            layers = unique(evidence['per_layer'], 'layer')
            require(set(layers) == set(range(32)), 'Incomplete injection layers')
            for layer in layers.values():
                fields(layer, {k: True for k in ('q_injection_verified', 'k_injection_verified', 'v_injection_verified',
                    'fresh_q_rows_equal', 'k_rows_equal', 'v_rows_equal')}, 'Layer injection')
            masks.append(evidence['hit_positions'])
        target_start = int(next(r for r in first if r['episode_id'] == episode_id)['target_start'])
        require(all(mask == list(range(target_start, target_start+3)) for mask in masks), 'Different QKV masks')
        internal = reports['internal_causal_effect'][episode_id]
        fields(internal, dict(hit_row_causal_effect_verified=True, fresh_attention_numerically_negligible=True), 'Internal causality')
        layers = unique(internal['per_layer'], 'layer')
        require(set(layers) == set(range(32)), 'Incomplete internal layers')
        require(all(math.isfinite(r[part]['max_absolute_error']) and r[part]['max_absolute_error'] >= 0
            for r in layers.values() for part in ('hit_attention', 'hit_post_layer_hidden')), 'Nonfinite internal effect')
        require(any(r[part]['max_absolute_error'] > 0 for r in layers.values()
                    for part in ('hit_attention', 'hit_post_layer_hidden')), 'Q_ZERO must change hit-row computation')
        require(all(0 <= r['fresh_attention_max_abs_diff'] <= 1e-6 for r in layers.values()), 'Fresh attention changed')
        continuation = reports['continuation_causal_effect'][episode_id]
        fields(continuation, dict(causal_isolation_supported=True), 'Continuation isolation')
        for label in ('Q24', 'Q32', 'Q_ZERO_COUNTERFACTUAL'):
            effect = continuation['comparisons'][label]
            fields(effect, dict(numerically_negligible=True), 'Continuation effect')
            require(0 <= effect['max_abs_logit_difference'] <= 1e-6 and 0 <= effect['mean_abs_logit_difference'] <= 1e-7
                and effect['top1_agreement'] == effect['top5_agreement'] == 1, 'Continuation changed')
    summary = (root/'summary.md').read_text()
    confirmations = (
        'Decoded Q24/Q32 different from raw TOTAL Q in every case?',
        'Decoded tensors actually injected into the selected HIT rows?',
        'Zeroing cached Q changed hit-row attention/internal states?',
        'Fresh-row attention remained numerically unchanged?',
        'Canonical teacher-forced continuation remained numerically unchanged?',
        'Compressed Q is genuinely consumed but causally isolated here?')
    for i, question in enumerate(confirmations, 1):
        require(f'{i}. {question} True' in summary.splitlines(), 'B2A summary does not confirm: '+question)
    return dict(decoded_q_differs_from_raw=True, decoded_q_injected_into_hit_rows=True,
                cached_q_causally_consumed=True, fresh_attention_isolated=True, continuation_isolated=True)


def decision(q_root, b2_root, b2a_root, kv_profile):
    """Validate immutable evidence; record the user's manual decision, never select/fit."""
    b1, hashes = load_manifest(q_root, dict(stage='C7-B1', status='COMPLETE', selected_q_candidate=None,
        q_profile_frozen=False, runtime_profile_fit_count_on_candidate_select=0, q_candidates=[16, 20, 24, 32],
        fit_blocks=96, candidate_select_blocks=32, fit_select_conversation_overlap_count=0,
        frozen32_overlap_count=0, capability64_overlap_count=0, existing_kv_profile_modified=False), B1_SHA,
        ('profiles/q24.bin', 'candidate_summary.json'))
    fields(b1['transform_provenance'], dict(transform=TRANSFORM, role_generic_verified=True), 'Q transform')
    require(b1['profile_hashes']['Q24'] == Q_SHA and b1['existing_kv_profile']['sha256'] == KV_SHA, 'Calibration profile binding mismatch')
    q_profile = q_root/'profiles/q24.bin'
    check_hash(q_profile, Q_SHA); check_hash(kv_profile, KV_SHA)
    hashes[str(kv_profile.resolve())] = KV_SHA
    rates = unique(read(q_root/'candidate_summary.json')['candidates'], 'candidate')
    require(set(rates) == {'Q16', 'Q20', 'Q24', 'Q32'}, 'Calibration candidates changed')
    require(math.isfinite(rates['Q24']['q_resident_compression_ratio']) and
        rates['Q24']['q_resident_compression_ratio'] > rates['Q32']['q_resident_compression_ratio'] > 0,
        'Manual compression rationale unsupported')
    b2, files = load_manifest(b2_root, dict(stage='C7-B2', status='COMPLETE', baseline_matches_c6b3_2=True,
        runtime_q_profile_fit_count=0, runtime_storage_kv_cdf_fit_count=0, transport_compression_enabled=False,
        c7b1_manifest_sha256=B1_SHA, kv_profile_sha256=KV_SHA, frozen32_selection_sha256=SELECTION_SHA,
        physical_safety_contract=CONTRACT, cached_payload='TOTAL_QKV'), B2_SHA,
        ('summary.json', 'paired_quality.json', 'per_case.csv', 'storage_accounting.json'))
    hashes.update(files)
    require(b2['q_profile_hashes']['Q24'] == Q_SHA, 'B2 Q24 profile mismatch')
    namespace = {k: b1[k] for k in ('model', 'model_revision', 'tokenizer_revision', 'adapter_hashes')}
    fields(b2, namespace, 'B1/B2 model namespace')
    require(all(b1['profile_hashes'][c] == b2['q_profile_hashes'][c] for c in ('Q24', 'Q32')), 'B1/B2 Q profile binding differs')
    summary = read(b2_root/'summary.json'); paired = read(b2_root/'paired_quality.json')
    modes = unique(summary['modes'], 'mode'); pairs = unique(paired['aggregates'], 'mode')
    require(set(modes) == {BASELINE, Q24, Q32}, 'B2 mode set changed')
    rows = read_csv(b2_root/'per_case.csv')
    by_mode = {mode: unique([r for r in rows if r['mode'] == mode], 'episode_id') for mode in modes}
    baseline_ids = set(by_mode[BASELINE])
    require(len(baseline_ids) == 32, 'B2 baseline case count changed')
    for mode in (Q24, Q32):
        require(set(by_mode[mode]) == baseline_ids, 'B2 candidate identities differ')
        require(modes[mode]['cases'] == modes[BASELINE]['cases'] == 32, 'B2 case count changed')
        delta = modes[mode]['corpus_bleu']['value'] - modes[BASELINE]['corpus_bleu']['value']
        require(math.isfinite(delta) and delta == 0, 'Observed B2 BLEU delta is not zero')
        fields(pairs[mode], dict(baseline_mode=BASELINE, cases=32, exact_generation_match_count=32), 'B2 paired quality')
        case_pairs = unique([r for r in paired['per_case'] if r['mode'] == mode], 'episode_id')
        require(set(case_pairs) == baseline_ids and all(r['generation_fidelity']['exact_generation'] is True
            for r in case_pairs.values()), 'B2 per-case generation evidence differs')
        for episode_id in baseline_ids:
            actual, baseline = by_mode[mode][episode_id], by_mode[BASELINE][episode_id]
            require(actual['generated_text'] == baseline['generated_text'] and
                json.loads(actual['generated_token_ids']) == json.loads(baseline['generated_token_ids']), 'B2 generation mismatch')
    evidence = dict(delta_bleu_vs_kv_baseline=modes[Q24]['corpus_bleu']['value']-modes[BASELINE]['corpus_bleu']['value'],
                    exact_generation_match_count=pairs[Q24]['exact_generation_match_count'], cases=32)
    b2a, files = load_manifest(b2a_root, dict(stage='C7-B2A', status='COMPLETE', audit_cases=4,
        causal_isolation_supported=True, runtime_q_profile_fit_count=0, runtime_storage_kv_cdf_fit_count=0,
        transport_encode_calls=0, transport_decode_calls=0, c7b2_manifest_sha256=B2_SHA,
        kv_profile_sha256=KV_SHA, frozen32_selection_sha256=SELECTION_SHA,
        physical_safety_contract=CONTRACT, cached_payload='TOTAL_QKV'), required=(
        'selected_cases.json', 'q_distortion.json', 'injection_audit.json', 'internal_causal_effect.json',
        'continuation_causal_effect.json', 'summary.md'))
    hashes.update(files)
    fields(b2a, namespace, 'B2A model namespace')
    require(b2a['q_profile_hashes'] == b2['q_profile_hashes'], 'B2A Q profile binding mismatch')
    causal = causal_evidence(b2a_root, b2a, list(by_mode[BASELINE].values()))
    verify_files(hashes)
    return dict(stage='C7-B3', status='FROZEN', selected_q_candidate='Q24', q_profile_frozen=True,
        q_bins=24, k_bins=20, v_bins=16, q_transform=TRANSFORM,
        q_profile_path=str(q_profile.resolve()), q_profile_sha256=Q_SHA,
        kv_profile_path=str(kv_profile.resolve()), kv_profile_sha256=KV_SHA,
        physical_safety_contract=CONTRACT, cached_payload='TOTAL_QKV',
        selection_basis=['C7-B1 held-out TOTAL-Q rate-distortion calibration',
            'C7-B2 frozen32 incremental resident-Q compression quality gate', 'C7-B2A cached-Q causal sanity audit'],
        frozen32_used_for_selection=True, additional_q_fitting_after_freeze=False, additional_kv_fitting_after_freeze=False,
        training_performed=False, model_inference_performed=False,
        model_namespace=namespace,
        b1_manifest_path=str((q_root/'manifest.json').resolve()), b1_manifest_sha256=B1_SHA,
        b2_manifest_path=str((b2_root/'manifest.json').resolve()), b2_manifest_sha256=B2_SHA,
        b2a_manifest_path=str((b2a_root/'manifest.json').resolve()), b2a_manifest_sha256=sha(b2a_root/'manifest.json'),
        q24_quality_evidence=evidence, causal_audit=causal, manual_decision='SELECT_Q24',
        decision_rationale=RATIONALE, input_hashes=hashes)


def freeze(args):
    output_location(args.output_root)
    path = args.output_root/'freeze_decision.json'
    require(not path.exists(), 'Refusing to overwrite freeze_decision.json')
    result = decision(args.q_calibration_root, args.c7b2_root, args.c7b2a_root, args.kv_profile)
    write(path, dict(result, git=git()))
    return result


def validate_freeze(path):
    saved = read(path)
    fields(saved, dict(stage='C7-B3', status='FROZEN', manual_decision='SELECT_Q24', selected_q_candidate='Q24',
        q_profile_frozen=True, q_bins=24, k_bins=20, v_bins=16, q_profile_sha256=Q_SHA, kv_profile_sha256=KV_SHA,
        additional_q_fitting_after_freeze=False, additional_kv_fitting_after_freeze=False), 'Frozen policy')
    verify_files(saved['input_hashes'])
    expected = decision(Path(saved['b1_manifest_path']).parent, Path(saved['b2_manifest_path']).parent,
                        Path(saved['b2a_manifest_path']).parent, Path(saved['kv_profile_path']))
    fields(saved, expected, 'Freeze evidence')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in dict(q_calibration_root='results/cachegen/c7/b_q_calibration_retry2',
        c7b2_root='results/cachegen/c7/b2_q_quality_gate_frozen32', c7b2a_root='results/cachegen/c7/b2a_q_causal_audit_retry1',
        kv_profile='results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin',
        output_root='results/cachegen/c7/q24_freeze').items():
        parser.add_argument('--'+name.replace('_', '-'), type=Path, default=Path(default))
    args = parser.parse_args(argv)
    freeze(args)
    print('FROZEN Q24/K20/V16: '+str(args.output_root/'freeze_decision.json'))
