"""Calibration-only matched-rate selection; independent role fits and encodes.

The pure engine accepts a calibration loader for tiny CPU tests. The real CLI
uses existing hashed OPT captures and the recorded CUDA contract on SERAPH.
"""
from collections import Counter
import math
from pathlib import Path

from ..common import digest, file_hash, load_fixture, read_json, write_csv, write_json
from ..harness import verify_capture
from ..b1 import atomic_bytes
from ..shared.device_contract import resolve_contract
from .harness import ROOT, RESULTS, SUM_FIELDS, git_state, metric_sums, metrics
from .policy import CACHEGEN_RELEASED_QL2, UniformKVPolicy, validate_candidates
from .reference import source_manifest
from . import rate_storage as rs

OUTPUT = RESULTS/'rate_calibration'
SELECTION_RULE = dict(primary='Independently minimize abs(uniform role payload - QL2 role payload)/QL2 role payload',
    tie_break='Smaller bins; secondary total-rate ties use ascending (K bins,V bins)',
    secondary='Minimize absolute full physical storage gap over all pairs, by addition only',
    brackets='Strictly below and strictly above target; exact matches are reported separately',
    quality_used_for_selection=False, close_match='Both primary absolute role gaps <=1%; no policy changes otherwise')
ACCOUNTING_RULE = dict(role_payload='Anchor payload + residual payload; measured arithmetic bytes including each stream termination',
    local_metadata='Existing B2 block header + four uint32 lengths + SHA256 + 2*32*3 FP32 maxabs slots',
    global_profile='One serialized full four-CDF B2 profile per policy/workload, never per block',
    attribution='Two CDF tables per K/V role; shared profile header is not charged twice',
    total='Sum of actual K/V role payload bytes + all local metadata + one global profile',
    pair_sizes='Exact composition from measured role lengths and identical fixed framing; no pair re-encoding',
    candidate_full_rate='Candidate role paired with the QL2 counterpart; actual frames measured in memory',
    raw='FP16 K+V only; Q is excluded', metadata='FP16 maxabs exactly widened to FP32, identical for all policies')
CDF_RULE = dict(partition='calibration only', window_size=3, mode=rs.MODE,
    roles=['K_anchor', 'K_mod255_residual', 'V_anchor', 'V_mod255_residual'],
    fitting='Fresh CDFs per policy and role from its own calibration symbols; entire partition reused for rate estimation',
    smoothing='Existing shared.core.cdf_from_counts; Laplace +1, 255 symbols, total 65536',
    evaluation_access=False, old_B2_CDF_reused=False)


def calibration_blocks(capture):
    """Filter partition first. Existing verifier sees no evaluation blocks/samples."""
    selected = [b for b in capture['blocks'] if b['partition'] == 'calibration' and b['token_group_size'] == 3]
    if not selected:
        raise ValueError('Nonempty existing calibration w=3 partition required')
    if (capture['model_config']['name'] != 'facebook/opt-2.7b' or capture['model_config']['dtype'] != 'float16' or
            capture.get('scope') != 'base_raw_unscaled_linear_projection; no LoRA adapter'):
        raise ValueError('Expected frozen OPT-2.7B FP16 base captures')
    sampling = {}
    for dataset in sorted({b['dataset'] for b in selected}):
        original = capture['sampling'][dataset]
        sampling[dataset] = dict(calibration=original['calibration'], evaluation=[],
            hashes=dict(calibration=original['hashes']['calibration'], evaluation=digest([])))
    view = dict(capture, blocks=selected, sampling=sampling, sampling_sha256=digest(sampling))
    verify_capture(view)
    return sorted(selected, key=lambda b: b['block_id'])


def workload_shape(block_count, candidates):
    candidates = validate_candidates(candidates)
    if type(block_count) is not int or block_count < 1:
        raise ValueError('Positive calibration block count required')
    per_role = 1+len(candidates)
    return dict(calibration_blocks=block_count, K_candidates=len(candidates), V_candidates=len(candidates),
        role_distribution_fit_passes=2*per_role, role_distribution_encode_passes=2*per_role,
        fixture_scan_passes=2, fixture_loads=2*block_count,
        quantization_calls=4*per_role*block_count, arithmetic_encode_calls=4*per_role*block_count,
        arithmetic_decode_calls=4*per_role*block_count, cartesian_pair_encodes=0)


class ObservationError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__('Invalid released quantization observation; no zero guard/repair: '+str(report))


def observe(encoded, block_id, phase):
    import torch
    zero_vectors = encoded.maxabs == 0
    zero_symbols = encoded.symbols[zero_vectors.expand_as(encoded.symbols)]
    values, counts = torch.unique(zero_symbols, return_counts=True)
    report = dict(block_id=block_id, role=encoded.role, policy=encoded.policy, phase=phase,
        zero_maxabs_vector_count=int(zero_vectors.sum().item()),
        zero_vector_symbol_histogram={str(int(v)): int(n) for v, n in zip(values.tolist(), counts.tolist())},
        invalid_symbol_count=int(((encoded.symbols < 0) | (encoded.symbols > 2*encoded.limits[:, None, None])).sum().item()),
        nonfinite_reconstruction_count=int((~torch.isfinite(encoded.dequantize())).sum().item()),
        nonfinite_final_cast_count=int((~torch.isfinite(encoded.reconstructed)).sum().item()))
    if report['invalid_symbol_count'] or report['nonfinite_reconstruction_count'] or report['nonfinite_final_cast_count']:
        raise ObservationError(report)
    return report


def _new_stats():
    return dict(**{key: 0 for key in SUM_FIELDS}, max_abs_error=0,
        anchor_payload_bytes=0, residual_payload_bytes=0, payload_bytes=0, per_block_payload_bytes=[],
        per_block_framed_bytes_with_ql2_counterpart=[],
        zero_maxabs_vector_count=0, zero_vector_symbol_histogram={}, storage_symbol_mismatch=0)


def _finite_metrics(sums):
    if any(not math.isfinite(sums[k]) for k in (*SUM_FIELDS, 'max_abs_error')):
        raise ValueError('Nonfinite pooled sufficient statistics')
    result = metrics(sums)
    if any(v is not None and not math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite reconstruction summary')
    return result


def calibrate(blocks, loader, *, device, candidates, scope_hash, progress=None):
    """Two fixture scans; 2*(1+B) role encodes per block, never B*B pairs.

    Candidate frames use the already encoded QL2 counterpart to measure exact
    fixed framing costs. Selected/secondary pairs are composed by byte addition.
    No bitstreams or tensors survive this function; only small summaries/CDFs.
    """
    import torch
    candidates = validate_candidates(candidates)
    blocks = list(blocks)
    if not blocks or any(b['partition'] != 'calibration' or b['token_group_size'] != 3 for b in blocks):
        raise ValueError('Only nonempty calibration w=3 blocks may reach the loader')
    ids = [b['block_id'] for b in blocks]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate calibration block IDs')
    plans = {role: [('ql2', CACHEGEN_RELEASED_QL2)]+[(b, UniformKVPolicy(b, b)) for b in candidates] for role in ('K', 'V')}
    counts = {(role, key): [[0]*255 for _ in range(2)] for role in plans for key, _ in plans[role]}
    shapes = []
    for phase in ('fit', 'measure'):
        if phase == 'measure':
            profiles = {key: rs.fit_role(key[0], hist, scope_hash) for key, hist in counts.items()}
            stats = {key: _new_stats() for key in counts}
            ql2_profile = rs.compose_profile(profiles['K', 'ql2'], profiles['V', 'ql2'])
        for index, block in enumerate(blocks):
            fixture = loader(block)
            shape = tuple(fixture['k'].shape)
            rs.fmt.validate_shape(shape, expected_tokens=3)
            if tuple(fixture['v'].shape) != shape or any(fixture[r].dtype != torch.float16 for r in ('k', 'v')):
                raise ValueError('Matching captured FP16 K/V required')
            if phase == 'fit':
                shapes.append(shape)
            elif shape != shapes[index]:
                raise ValueError('Fixture shape changed between fit and measurement')
            cache = {}
            for role in ('K', 'V'):
                x = fixture[role.lower()].to(device)
                for key, policy in plans[role]:
                    encoded = policy.quantize(x, role)
                    observation = observe(encoded, block['block_id'], phase)
                    if phase == 'fit':
                        for target, hist in zip(counts[role, key], rs.stream_counts(rs.role_streams(encoded))):
                            for i, value in enumerate(hist):
                                target[i] += value
                    else:
                        payloads, metadata = rs.encode_role(encoded, profiles[role, key])
                        cache[role, key] = (payloads, metadata)
                        row = stats[role, key]
                        sums = metric_sums(fixture[role.lower()], encoded.reconstructed.cpu())
                        _finite_metrics(sums)
                        for field in SUM_FIELDS:
                            row[field] += sums[field]
                        row['max_abs_error'] = max(row['max_abs_error'], sums['max_abs_error'])
                        row['anchor_payload_bytes'] += len(payloads[0])
                        row['residual_payload_bytes'] += len(payloads[1])
                        row['payload_bytes'] += sum(map(len, payloads))
                        row['per_block_payload_bytes'].append(sum(map(len, payloads)))
                        row['zero_maxabs_vector_count'] += observation['zero_maxabs_vector_count']
                        hist = Counter(row['zero_vector_symbol_histogram'])
                        hist.update(observation['zero_vector_symbol_histogram'])
                        row['zero_vector_symbol_histogram'] = dict(hist)
            if phase == 'measure':
                # Maxabs metadata is policy-independent. Verify this invariant,
                # then measure actual B2 frames without additional coder calls.
                for role in ('K', 'V'):
                    expected = cache[role, 'ql2'][1]
                    if any(cache[role, key][1] != expected for key, _ in plans[role]):
                        raise ValueError('Policies changed maxabs metadata accounting')
                for role in ('K', 'V'):
                    for key, _ in plans[role]:
                        kp, km = cache['K', key if role == 'K' else 'ql2']
                        vp, vm = cache['V', key if role == 'V' else 'ql2']
                        profile = rs.compose_profile(profiles['K', key if role == 'K' else 'ql2'],
                                                     profiles['V', key if role == 'V' else 'ql2'])
                        blob = rs.fmt.frame_payloads(profile, kp+vp, km+vm, shape, expected_tokens=3)
                        actual_shape, actual_metadata, _, sizes = rs.fmt.inspect(blob, profile, expected_tokens=3)
                        account = rs.physical_accounting([sum(map(len, kp))], [sum(map(len, vp))], [shape],
                                                        profile_bytes=len(profile.to_bytes()))
                        if actual_shape != shape or actual_metadata != km+vm or len(blob) != account['per_block_bitstream_bytes'][0]:
                            raise ValueError('Independent role accounting differs from actual B2 frame')
                        if sizes['local_metadata_bytes'] != account['local_metadata_bytes']:
                            raise ValueError('B2 local metadata accounting differs')
                        stats[role, key]['per_block_framed_bytes_with_ql2_counterpart'].append(len(blob))
            if progress:
                progress(phase, index+1, len(blocks))
    qaccount = rs.physical_accounting(stats['K', 'ql2']['per_block_payload_bytes'],
        stats['V', 'ql2']['per_block_payload_bytes'], shapes, profile_bytes=len(ql2_profile.to_bytes()))
    if qaccount['per_block_bitstream_bytes'] != stats['K', 'ql2']['per_block_framed_bytes_with_ql2_counterpart']:
        raise ValueError('QL2 workload accounting differs from measured frames')
    def role_summary(role, key):
        row = stats[role, key]
        return dict(role=role, policy='CACHEGEN_RELEASED_QL2' if key == 'ql2' else f'UNIFORM_{role}{key}',
            bins=None if key == 'ql2' else key, integer_limit=None if key == 'ql2' else key//2-1,
            calibration_block_count=len(blocks), relevant_profile_bytes=rs.profile_attribution()['role_cdf_bytes'],
            **row, reconstruction_metrics=_finite_metrics(row), storage_roundtrip_reconstruction_equal=True,
            calibration_counts=counts[role, key],
            calibration_counts_sha256=digest(counts[role, key]))
    qsummary = dict(policy=CACHEGEN_RELEASED_QL2.name, layer_profile=CACHEGEN_RELEASED_QL2.profile(),
        calibration_block_ids=ids, calibration_block_count=len(blocks), **qaccount,
        roles={role: role_summary(role, 'ql2') for role in ('K', 'V')})
    rows = []
    for role in ('K', 'V'):
        for bins in candidates:
            row = role_summary(role, bins)
            target = qsummary['roles'][role]['payload_bytes']
            row.update(target_payload_bytes=target, target_gap_percent=100*(row['payload_bytes']-target)/target,
                       absolute_target_gap_percent=100*abs(row['payload_bytes']-target)/target)
            k = stats['K', bins if role == 'K' else 'ql2']['per_block_payload_bytes']
            v = stats['V', bins if role == 'V' else 'ql2']['per_block_payload_bytes']
            row['full_physical_accounting_with_ql2_counterpart'] = rs.physical_accounting(k, v, shapes,
                profile_bytes=len(ql2_profile.to_bytes()))
            if row['full_physical_accounting_with_ql2_counterpart']['per_block_bitstream_bytes'] != row['per_block_framed_bytes_with_ql2_counterpart']:
                raise ValueError('Candidate workload accounting differs from measured frames')
            rows.append(row)
    selection = select_policy(qsummary, rows)
    kb, vb = selection['primary']['K_bins'], selection['primary']['V_bins']
    selected_profile = rs.compose_profile(profiles['K', kb], profiles['V', vb])
    selection['primary']['physical_accounting'] = rs.physical_accounting(
        stats['K', kb]['per_block_payload_bytes'], stats['V', vb]['per_block_payload_bytes'], shapes,
        profile_bytes=len(selected_profile.to_bytes()))
    sk, sv = selection['secondary_best_total_rate_pair']['K_bins'], selection['secondary_best_total_rate_pair']['V_bins']
    selection['secondary_best_total_rate_pair']['physical_accounting'] = rs.physical_accounting(
        stats['K', sk]['per_block_payload_bytes'], stats['V', sv]['per_block_payload_bytes'], shapes,
        profile_bytes=len(selected_profile.to_bytes()))
    return dict(ql2=qsummary, candidates=rows, selection=selection, ql2_profile=ql2_profile,
                matched_profile=selected_profile, workload=workload_shape(len(blocks), candidates))


def select_policy(ql2, rows):
    """Storage bytes only. Reconstruction metrics are deliberately inaccessible."""
    targets = {r: ql2[r.lower()+'_payload_bytes'] for r in ('K', 'V')}
    if any(type(v) is not int or v <= 0 for v in targets.values()):
        raise ValueError('Positive measured QL2 K/V payload targets required')
    by_role = {r: [x for x in rows if x['role'] == r] for r in ('K', 'V')}
    brackets, nearest = {}, {}
    for role, candidates in by_role.items():
        if not candidates or len({c['bins'] for c in candidates}) != len(candidates):
            raise ValueError('Distinct candidates for both K/V required')
        for c in candidates:
            validate_candidates([c['bins']])
            if type(c['payload_bytes']) is not int or c['payload_bytes'] <= 0:
                raise ValueError('Positive measured candidate payload required')
        target = targets[role]
        def choose(pool):
            if not pool:
                return None
            best = min(pool, key=lambda c: (abs(c['payload_bytes']-target), c['bins']))
            return dict(bins=best['bins'], payload_bytes=best['payload_bytes'],
                        signed_gap_percent=100*(best['payload_bytes']-target)/target,
                        absolute_gap_percent=100*abs(best['payload_bytes']-target)/target)
        nearest[role] = choose(candidates)
        brackets[role] = dict(lower=choose([c for c in candidates if c['payload_bytes'] < target]),
            upper=choose([c for c in candidates if c['payload_bytes'] > target]), nearest_absolute=nearest[role],
            exact_match_bins=sorted(c['bins'] for c in candidates if c['payload_bytes'] == target))
    overhead = ql2['local_metadata_bytes']+ql2['global_profile_bytes']
    total_target = ql2['total_physical_bytes']
    if total_target != sum(targets.values())+overhead:
        raise ValueError('QL2 total accounting inconsistent')
    def pair(k, v):
        total = k['payload_bytes']+v['payload_bytes']+overhead
        return dict(K_bins=k['bins'], V_bins=v['bins'], K_payload_bytes=k['payload_bytes'], V_payload_bytes=v['payload_bytes'],
            K_rate_gap_percent=100*(k['payload_bytes']-targets['K'])/targets['K'],
            V_rate_gap_percent=100*(v['payload_bytes']-targets['V'])/targets['V'],
            overall_physical_storage_gap_percent=100*(total-total_target)/total_target,
            total_calibration_bytes=total, compression_ratio=ql2['original_fp16_kv_bytes']/total)
    primary = pair(nearest['K'], nearest['V'])
    pairs = [pair(k, v) for k in by_role['K'] for v in by_role['V']]
    secondary = min(pairs, key=lambda p: (abs(p['total_calibration_bytes']-total_target), p['K_bins'], p['V_bins']))
    return dict(primary_policy=f"UNIFORM_K{primary['K_bins']}_V{primary['V_bins']}",
        primary=primary, secondary_best_total_rate_pair=secondary, brackets=brackets,
        both_primary_role_gaps_within_one_percent=all(
            abs(nearest[r]['payload_bytes']-targets[r])*100 <= targets[r] for r in ('K', 'V')),
        selection_rule=SELECTION_RULE, secondary_pair_count=len(pairs), cartesian_pair_encodes=0)


def run_rate_calibration(args):
    """Real workload entry point: implementation only, run manually on SERAPH."""
    import torch
    candidates = validate_candidates(args.candidate_bins)
    reference = source_manifest(args.cachegen_repo)
    capture_path = args.capture_manifest.resolve()
    root = capture_path.parent
    capture = read_json(capture_path)
    blocks = calibration_blocks(capture)
    for block in blocks:
        if not (root/block['file']).resolve().is_relative_to(root):
            raise ValueError('Calibration fixture path escapes capture root')
    contract = resolve_contract(root)
    if not contract['quantization_device_resolved'].startswith('cuda:'):
        raise ValueError('Real calibration requires recorded C1 CUDA contract; no CPU fallback')
    if args.device is not None and args.device not in (contract['quantization_device_resolved'], contract['reference_c1_device']):
        raise ValueError('--device must equal the recorded C1 device')
    out = args.output_dir.resolve()
    if not out.is_relative_to(OUTPUT.resolve()):
        raise ValueError('Rate calibration outputs must stay in results/cachegen/c1_5c/rate_calibration')
    for source in (root, args.cachegen_repo.resolve()):
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError('Rate output overlaps frozen inputs')
    out.mkdir(parents=True, exist_ok=False)
    paths = [*Path(__file__).parent.glob('*.py'), ROOT/'src/semcache/experiments/cachegen/b2/format.py',
             ROOT/'src/semcache/experiments/cachegen/b1.py', ROOT/'src/semcache/experiments/cachegen/shared/core.py',
             ROOT/'src/semcache/experiments/cachegen/common.py', ROOT/'src/semcache/experiments/cachegen/harness.py',
             ROOT/'src/semcache/experiments/cachegen/shared/device_contract.py']
    manifest = dict(stage='C1.5C-2', status='RUNNING', gp_semcache_git=git_state(), cachegen_reference=reference,
        capture_manifest_path=str(capture_path), capture_manifest_sha256=file_hash(capture_path),
        capture_manifest_content_hash=digest(capture), calibration_block_ids=[b['block_id'] for b in blocks],
        calibration_block_count=len(blocks), calibration_blocks=blocks,
        source_capture_paths=[str((root/b['file']).resolve()) for b in blocks],
        calibration_sampling_hashes={d: capture['sampling'][d]['hashes']['calibration'] for d in {b['dataset'] for b in blocks}},
        model=capture['model_config'], model_revision=capture.get('model_metadata', {}).get('resolved_model_revision'),
        dtype='float16', w=3, seed=args.seed, seed_use='Recorded only; complete sorted calibration partition, no sampling',
        candidate_bins=list(candidates), QL2_layer_profile=CACHEGEN_RELEASED_QL2.profile(),
        device_contract=contract, torch_version=torch.__version__, cuda_version=torch.version.cuda,
        implementation_sha256={str(p): file_hash(p) for p in paths}, selection_rule=SELECTION_RULE,
        byte_accounting_rule=ACCOUNTING_RULE, CDF_fitting_rule=CDF_RULE,
        workload=workload_shape(len(blocks), candidates), evaluation_blocks_loaded=0,
        storage_symbol_mismatch=0, no_model_inference=True, selected_profiles={})
    scope_hash = digest(dict(capture_sha256=manifest['capture_manifest_sha256'],
        block_ids=manifest['calibration_block_ids'], device_contract=contract, w=3,
        byte_accounting=ACCOUNTING_RULE, CDF_rule=CDF_RULE))
    manifest['calibration_scope_sha256'] = scope_hash
    write_json(out/'calibration_manifest.json', manifest)
    def progress(phase, done, total):
        if done == total or done % 16 == 0:
            manifest['progress'] = dict(phase=phase, completed_blocks=done, total_blocks=total)
            write_json(out/'calibration_manifest.json', manifest)
            print(f'{phase}: {done}/{total} calibration blocks', flush=True)
    try:
        result = calibrate(blocks, lambda b: load_fixture(root, b), device=contract['quantization_device_resolved'],
                           candidates=candidates, scope_hash=scope_hash, progress=progress)
        if file_hash(capture_path) != manifest['capture_manifest_sha256'] or resolve_contract(root) != contract:
            raise ValueError('Capture/device contract changed during calibration')
        for b in blocks:
            if file_hash(root/b['file']) != b['sha256']:
                raise ValueError('Calibration fixture changed during calibration')
        for path, expected in manifest['implementation_sha256'].items():
            if file_hash(path) != expected:
                raise ValueError('Implementation changed during calibration')
        source_manifest(args.cachegen_repo)
        write_json(out/'ql2_calibration_summary.json', result['ql2'])
        write_json(out/'uniform_rate_search.json', dict(candidate_bins=list(candidates), candidates=result['candidates'],
            calibration_scope_sha256=scope_hash, byte_accounting_rule=ACCOUNTING_RULE))
        csv_rows = []
        for row in result['candidates']:
            account = row['full_physical_accounting_with_ql2_counterpart']
            csv_rows.append({key: row[key] for key in ('role', 'bins', 'integer_limit', 'calibration_block_count',
                'anchor_payload_bytes', 'residual_payload_bytes', 'payload_bytes', 'relevant_profile_bytes',
                'target_payload_bytes', 'target_gap_percent', 'absolute_target_gap_percent',
                'zero_maxabs_vector_count', 'storage_symbol_mismatch')} | row['reconstruction_metrics'] | {
                    'shared_profile_header_bytes': rs.profile_attribution()['shared_profile_header_bytes'],
                    'counterpart_policy': 'CACHEGEN_RELEASED_QL2',
                    'local_metadata_bytes': account['local_metadata_bytes'],
                    'local_transform_metadata_bytes': account['local_transform_metadata_bytes'],
                    'scale_maxabs_metadata_bytes': account['scale_maxabs_metadata_bytes'],
                    'global_profile_bytes': account['global_profile_bytes'],
                    'total_physical_bytes_with_ql2_counterpart': account['total_physical_bytes'],
                    'compression_ratio_with_ql2_counterpart': account['compression_ratio']})
        write_csv(out/'uniform_rate_search.csv', csv_rows, csv_rows[0].keys())
        (out/'profiles').mkdir()
        primary = result['selection']['primary']
        selected_policy = UniformKVPolicy(primary['K_bins'], primary['V_bins'])
        for key, filename, policy, profile in (
            ('released_ql2', 'cachegen_released_ql2.bin', CACHEGEN_RELEASED_QL2, result['ql2_profile']),
            ('matched_uniform', f'matched_uniform_k{primary["K_bins"]}_v{primary["V_bins"]}.bin', selected_policy, result['matched_profile'])):
            relative = 'profiles/'+filename
            atomic_bytes(out/relative, profile.to_bytes())
            manifest['selected_profiles'][key] = dict(file=relative, sha256=profile.sha256,
                serialized_bytes=len(profile.to_bytes()), policy=policy.name, layer_profile=policy.profile(),
                mode=profile.mode, calibration_scope_sha256=scope_hash,
                CDF_fit='Own calibration symbols, same blocks, fresh role-separated CDFs')
        result['selection']['profiles'] = manifest['selected_profiles']
        write_json(out/'selection.json', result['selection'])
        manifest['output_sha256'] = {name: file_hash(out/name) for name in (
            'ql2_calibration_summary.json', 'uniform_rate_search.json', 'uniform_rate_search.csv', 'selection.json')}
        manifest.update(status='COMPLETED', profiles_status='FROZEN_FOR_C15C3',
            storage_roundtrip_reconstruction_equal=True,
            zero_vector_counts={r: result['ql2']['roles'][r]['zero_maxabs_vector_count'] for r in ('K', 'V')})
        print('Primary rate match:', result['selection']['primary_policy'])
        print('Primary gaps:', {k: v for k, v in primary.items() if k != 'physical_accounting'})
        print('Both role gaps <=1%:', result['selection']['both_primary_role_gaps_within_one_percent'])
        print('Secondary diagnostic:', {k: v for k, v in result['selection']['secondary_best_total_rate_pair'].items()
                                         if k != 'physical_accounting'})
    except BaseException as exc:
        manifest.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        if isinstance(exc, ObservationError):
            manifest['invalid_quantization_observation'] = exc.report
        raise
    finally:
        write_json(out/'calibration_manifest.json', manifest)
