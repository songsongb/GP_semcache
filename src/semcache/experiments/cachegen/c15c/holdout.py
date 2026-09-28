"""C1.5C-3: two frozen policies, one evaluation fixture scan, encode only."""
from collections import Counter, defaultdict
import math
from pathlib import Path
import struct
import time

from ..common import digest, file_hash, load_fixture, read_json, write_csv, write_json
from ..harness import verify_capture
from ..b2 import format as fmt
from ..shared import core
from ..shared.device_contract import resolve_contract
from ..shared.source_audit import REVISION
from .harness import ROOT, RESULTS, SUM_FIELDS, git_state, metric_sums, metrics
from .policy import CACHEGEN_RELEASED_QL2, UniformKVPolicy
from .rate_calibration import ObservationError, observe
from .rate_storage import MODE, physical_accounting, role_streams

OUTPUT = RESULTS/'holdout'
REQUIRED_WINDOW_SIZE = 3
QL2 = 'CACHEGEN_RELEASED_QL2'
UNIFORM = 'UNIFORM_K20_V16'
POLICIES = (QL2, UNIFORM)
PROFILE_FILES = {QL2: 'profiles/cachegen_released_ql2.bin',
                 UNIFORM: 'profiles/matched_uniform_k20_v16.bin'}


def evaluation_blocks(capture):
    """Select the w=3 holdout matching C1.5C-2; never inspect T=10 fixtures."""
    counts = Counter(b['partition'] for b in capture['blocks'])
    if set(counts) - {'calibration', 'evaluation'}:
        raise ValueError('Unexpected capture partition')
    evaluation = [b for b in capture['blocks'] if b['partition'] == 'evaluation']
    selected = [b for b in evaluation if b['token_group_size'] == REQUIRED_WINDOW_SIZE]
    if not selected:
        raise ValueError('No evaluation w=3 blocks in capture manifest')
    if any(b['token_group_size'] != REQUIRED_WINDOW_SIZE for b in selected):
        raise ValueError('Selected holdout block is not w=3')
    if (capture['model_config']['name'] != 'facebook/opt-2.7b' or
            capture['model_config']['dtype'] != 'float16' or
            capture.get('scope') != 'base_raw_unscaled_linear_projection; no LoRA adapter'):
        raise ValueError('Expected frozen OPT-2.7B FP16 base captures')
    sampling = {}
    for dataset in sorted({b['dataset'] for b in selected}):
        original = capture['sampling'][dataset]
        sampling[dataset] = dict(calibration=[], evaluation=original['evaluation'],
            hashes=dict(calibration=digest([]), evaluation=original['hashes']['evaluation']))
    view = dict(capture, blocks=selected, sampling=sampling, sampling_sha256=digest(sampling))
    verify_capture(view)
    return sorted(selected, key=lambda b: b['block_id']), dict(
        total_capture_blocks=len(capture['blocks']), total_calibration_blocks=counts['calibration'],
        total_evaluation_blocks=counts['evaluation'], evaluation_w3_count=len(selected),
        evaluation_non_w3_count=len(evaluation)-len(selected),
        selected_holdout_count=len(selected), required_window_size=REQUIRED_WINDOW_SIZE)


def frozen_profiles(calibration_dir, capture_sha256):
    """Read only frozen C1.5C-2 metadata and two selected profile files."""
    directory = Path(calibration_dir).resolve()
    manifest_path, selection_path = directory/'calibration_manifest.json', directory/'selection.json'
    manifest = read_json(manifest_path)
    selection = read_json(selection_path)
    if (manifest.get('status') != 'COMPLETED' or manifest.get('profiles_status') != 'FROZEN_FOR_C15C3'
            or manifest.get('capture_manifest_sha256') != capture_sha256
            or manifest.get('w') != 3 or manifest.get('dtype') != 'float16'
            or manifest.get('calibration_block_count', 0) < 1
            or manifest.get('cachegen_reference', {}).get('commit') != REVISION
            or selection.get('primary_policy') != UNIFORM
            or selection.get('primary', {}).get('K_bins') != 20
            or selection.get('primary', {}).get('V_bins') != 16
            or manifest.get('output_sha256', {}).get('selection.json') != file_hash(selection_path)):
        raise ValueError('Matching completed C1.5C-2 K20/V16 calibration selection required')
    expected = dict(released_ql2=(QL2, CACHEGEN_RELEASED_QL2),
                    matched_uniform=(UNIFORM, UniformKVPolicy(20, 16)))
    if set(manifest.get('selected_profiles', {})) != set(expected):
        raise ValueError('Exactly the two frozen C1.5C-2 profiles required')
    profiles, entries = {}, {}
    scope = manifest.get('calibration_scope_sha256')
    for key, (name, policy) in expected.items():
        item = manifest['selected_profiles'][key]
        if (item.get('file') != PROFILE_FILES[name] or item.get('policy') != name or
                item.get('mode') != MODE or item.get('layer_profile') != policy.profile() or
                item.get('calibration_scope_sha256') != scope):
            raise ValueError(f'Frozen profile policy/scope mismatch: {key}')
        path = (directory/item['file']).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f'Missing frozen profile: {key}')
        data = path.read_bytes()
        profile = fmt.Profile.from_bytes(data, item['sha256'])
        if len(data) != item['serialized_bytes'] or profile.mode != MODE:
            raise ValueError(f'Frozen profile format/length mismatch: {key}')
        profiles[name] = profile
        entries[name] = dict(path=str(path), sha256=profile.sha256, bytes=len(data),
            policy=name, layer_profile=policy.profile(), calibration_scope_sha256=scope)
    if selection.get('profiles') != manifest['selected_profiles']:
        raise ValueError('Selection/profile provenance mismatch')
    provenance = dict(calibration_manifest_path=str(manifest_path), calibration_manifest_sha256=file_hash(manifest_path),
        selection_path=str(selection_path), selection_sha256=file_hash(selection_path),
        calibration_scope_sha256=scope, calibration_block_count=manifest['calibration_block_count'],
        cachegen_reference_commit=manifest.get('cachegen_reference', {}).get('commit'),
        profiles=entries)
    return profiles, provenance, manifest.get('device_contract')


def _new_group():
    return dict(block_count=0, shapes=[], k_payload=[], v_payload=[],
        k_anchor_payload_bytes=0, k_residual_payload_bytes=0,
        v_anchor_payload_bytes=0, v_residual_payload_bytes=0,
        roles={r: dict(**{field: 0 for field in SUM_FIELDS}, max_abs_error=0,
                       zero_maxabs_vector_count=0) for r in ('K', 'V')})


def _metrics(sums):
    if sums['element_count'] < 1 or any(not math.isfinite(sums[k]) for k in (*SUM_FIELDS, 'max_abs_error')):
        raise ValueError('Nonfinite or empty holdout sufficient statistics')
    result = metrics(sums)
    if any(v is not None and not math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite holdout reconstruction metric')
    return result


def _summarize(group, profile_bytes):
    account = physical_accounting(group['k_payload'], group['v_payload'], group['shapes'],
        profile_bytes=profile_bytes, expected_tokens=REQUIRED_WINDOW_SIZE)
    return dict(block_count=group['block_count'],
        k_anchor_payload_bytes=group['k_anchor_payload_bytes'],
        k_residual_payload_bytes=group['k_residual_payload_bytes'],
        v_anchor_payload_bytes=group['v_anchor_payload_bytes'],
        v_residual_payload_bytes=group['v_residual_payload_bytes'],
        physical_storage=account,
        reconstruction={role: dict(element_count=sums['element_count'],
            zero_maxabs_vector_count=sums['zero_maxabs_vector_count'], **_metrics(sums))
            for role, sums in group['roles'].items()})


def compare_quality(ql2, uniform):
    """Signed errors: Uniform minus QL2; cosine reverses so negative is better."""
    out = {}
    for role in ('K', 'V'):
        q, u = ql2[role], uniform[role]
        def subtract(key):
            return None if q[key] is None or u[key] is None else u[key]-q[key]
        def percent(key):
            return None if q[key] in (None, 0) or u[key] is None else 100*(u[key]-q[key])/q[key]
        rel_delta = subtract('relative_l2')
        out[role] = dict(relative_l2_difference=rel_delta,
            relative_l2_absolute_difference=abs(rel_delta) if rel_delta is not None else None,
            relative_l2_percent_difference=percent('relative_l2'),
            cosine_difference=None if q['cosine_similarity'] is None or u['cosine_similarity'] is None
                else q['cosine_similarity']-u['cosine_similarity'],
            MSE_percent_difference=percent('MSE'),
            max_abs_difference=subtract('max_abs_error'))
    return out


def compare_results(summary):
    overall = summary['overall']
    q, u = overall[QL2], overall[UNIFORM]
    qa, ua = q['physical_storage'], u['physical_storage']
    def gap(field):
        if qa[field] <= 0:
            raise ValueError('Positive QL2 storage denominator required')
        return 100*(ua[field]-qa[field])/qa[field]
    qm, um = q['reconstruction'], u['reconstruction']
    result = dict(policy_A=QL2, policy_B=UNIFORM, calibration_selected_policy=UNIFORM,
        comparison_partition='evaluation', required_window_size=REQUIRED_WINDOW_SIZE,
        selected_policy_changed=False,
        sign_convention='Error deltas: Uniform minus QL2; negative means Uniform better. '
                        'Cosine delta: QL2 minus Uniform, so negative also means Uniform better.',
        K_payload_gap_percent=gap('k_payload_bytes'), V_payload_gap_percent=gap('v_payload_bytes'),
        overall_physical_storage_gap_percent=gap('total_physical_bytes'),
        quality_delta=compare_quality(qm, um),
        uniform_overall_storage_smaller=ua['total_physical_bytes'] < qa['total_physical_bytes'])
    for role in ('K', 'V'):
        for key, field in (('rel_l2_lower', 'relative_l2'), ('MSE_lower', 'MSE')):
            result[f'uniform_{role}_{key}'] = (um[role][field] < qm[role][field]
                if um[role][field] is not None and qm[role][field] is not None else None)
    return result


def evaluate(blocks, profiles, loader, *, device, progress=None):
    """Exactly one loader call and eight arithmetic encodes per evaluation block."""
    import torch
    blocks = list(blocks)
    if not blocks or set(profiles) != set(POLICIES):
        raise ValueError('Nonempty evaluation set and exactly the two frozen policies required')
    if any(b['partition'] != 'evaluation' or b['token_group_size'] != REQUIRED_WINDOW_SIZE for b in blocks):
        raise ValueError('Only evaluation w=3 blocks allowed in the selected holdout')
    if len({b['block_id'] for b in blocks}) != len(blocks):
        raise ValueError('Duplicate evaluation block ID')
    policy_objects = ((QL2, CACHEGEN_RELEASED_QL2), (UNIFORM, UniformKVPolicy(20, 16)))
    groups = defaultdict(_new_group)
    encode_calls = 0
    with torch.inference_mode():
        for completed, block in enumerate(blocks, 1):
            qkv = loader(block)
            shape = tuple(qkv['k'].shape)
            fmt.validate_shape(shape, expected_tokens=REQUIRED_WINDOW_SIZE)
            if tuple(qkv['v'].shape) != shape or any(qkv[role].dtype != torch.float16 for role in ('k', 'v')):
                raise ValueError('Matching captured FP16 K/V required')
            for name, policy in policy_objects:
                profile = profiles[name]
                encoded = {role: policy.quantize(qkv[role.lower()].to(device), role) for role in ('K', 'V')}
                observations = {role: observe(encoded[role], block['block_id'], 'evaluation') for role in ('K', 'V')}
                role_sums = {role: metric_sums(qkv[role.lower()], encoded[role].reconstructed.cpu()) for role in ('K', 'V')}
                for sums in role_sums.values():
                    _metrics(sums)
                streams = role_streams(encoded['K'])+role_streams(encoded['V'])
                payloads = tuple(core.arithmetic_encode(s, cdf) for s, cdf in zip(streams, profile.cdfs))
                encode_calls += 4
                maxima = [v for role in ('K', 'V') for v in encoded[role].storage_metadata.cpu().flatten().tolist()]
                metadata = struct.pack('<'+'f'*len(maxima), *maxima)
                blob = fmt.frame_payloads(profile, payloads, metadata, shape, expected_tokens=REQUIRED_WINDOW_SIZE)
                expected_bytes = core.BLOCK_HEADER.size+16+32+len(metadata)+sum(map(len, payloads))
                if len(blob) != expected_bytes:
                    raise ValueError('Frozen B2 framing/accounting mismatch')
                for group_name in ('overall', block['dataset']):
                    group = groups[name, group_name]
                    group['block_count'] += 1
                    group['shapes'].append(shape)
                    group['k_payload'].append(len(payloads[0])+len(payloads[1]))
                    group['v_payload'].append(len(payloads[2])+len(payloads[3]))
                    group['k_anchor_payload_bytes'] += len(payloads[0])
                    group['k_residual_payload_bytes'] += len(payloads[1])
                    group['v_anchor_payload_bytes'] += len(payloads[2])
                    group['v_residual_payload_bytes'] += len(payloads[3])
                    for role in ('K', 'V'):
                        sums = role_sums[role]
                        target = group['roles'][role]
                        for field in SUM_FIELDS:
                            target[field] += sums[field]
                        target['max_abs_error'] = max(target['max_abs_error'], sums['max_abs_error'])
                        target['zero_maxabs_vector_count'] += observations[role]['zero_maxabs_vector_count']
                del blob, payloads, streams, encoded, role_sums
            if progress:
                progress(completed, len(blocks))
    if encode_calls != 8*len(blocks):
        raise ValueError('Encode-call count differs from two frozen policies/four roles')
    summary = dict(overall={name: _summarize(groups[name, 'overall'], len(profiles[name].to_bytes())) for name in POLICIES},
        datasets={dataset: {name: _summarize(groups[name, dataset], len(profiles[name].to_bytes())) for name in POLICIES}
                  for dataset in sorted({b['dataset'] for b in blocks})},
        evaluation_block_count=len(blocks), required_window_size=REQUIRED_WINDOW_SIZE,
        arithmetic_encode_calls=encode_calls,
        arithmetic_decode_calls=0, cdf_fit_calls=0,
        pooling='Float64 element-pooled sufficient statistics; profile counted once per reported population')
    return summary, compare_results(summary)


def per_dataset_csv_rows(summary):
    rows = []
    for dataset, policies in summary['datasets'].items():
        for name in POLICIES:
            item = policies[name]
            account = item['physical_storage']
            for role in ('K', 'V'):
                m = item['reconstruction'][role]
                rows.append(dict(dataset=dataset, policy=name, role=role, block_count=item['block_count'],
                    **m, k_anchor_payload_bytes=item['k_anchor_payload_bytes'],
                    k_residual_payload_bytes=item['k_residual_payload_bytes'],
                    v_anchor_payload_bytes=item['v_anchor_payload_bytes'],
                    v_residual_payload_bytes=item['v_residual_payload_bytes'],
                    k_payload_bytes=account['k_payload_bytes'], v_payload_bytes=account['v_payload_bytes'],
                    local_metadata_bytes=account['local_metadata_bytes'],
                    local_transform_metadata_bytes=account['local_transform_metadata_bytes'],
                    scale_maxabs_metadata_bytes=account['scale_maxabs_metadata_bytes'],
                    global_profile_bytes=account['global_profile_bytes'],
                    total_physical_bytes=account['total_physical_bytes'],
                    original_fp16_kv_bytes=account['original_fp16_kv_bytes'],
                    compression_ratio=account['compression_ratio']))
    return rows


def run_holdout(args):
    capture_path = args.capture_manifest.resolve()
    capture = read_json(capture_path)
    blocks, counts = evaluation_blocks(capture)
    capture_sha = file_hash(capture_path)
    profiles, frozen, calibration_contract = frozen_profiles(args.rate_calibration_dir, capture_sha)
    expected = 8*len(blocks)
    print(f"capture={counts['total_capture_blocks']}")
    print(f"calibration={counts['total_calibration_blocks']}")
    print(f"evaluation_total={counts['total_evaluation_blocks']}")
    print(f"evaluation_w3={counts['evaluation_w3_count']}")
    print(f"evaluation_excluded_non_w3={counts['evaluation_non_w3_count']}")
    print(f"selected_holdout={counts['selected_holdout_count']}")
    print(f'expected arithmetic encode calls={expected}')
    if args.dry_run:
        print('Dry run complete; no fixtures loaded.')
        return dict(dry_run=True, counts=counts, expected_arithmetic_encode_calls=expected)
    import torch
    root = capture_path.parent
    for block in blocks:
        if not (root/block['file']).resolve().is_relative_to(root):
            raise ValueError('Evaluation fixture path escapes capture directory')
    contract = resolve_contract(root)
    if contract != calibration_contract or not contract['quantization_device_resolved'].startswith('cuda:'):
        raise ValueError('Holdout requires same recorded C1 CUDA contract as frozen calibration; no CPU fallback')
    out = args.output_dir.resolve()
    if not out.is_relative_to(OUTPUT.resolve()):
        raise ValueError('Holdout outputs must stay under results/cachegen/c1_5c/holdout')
    for source in (root, args.rate_calibration_dir.resolve()):
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError('Holdout output overlaps frozen inputs')
    out.mkdir(parents=True, exist_ok=False)
    code_paths = [Path(__file__), Path(__file__).parent/'policy.py', Path(__file__).parent/'rate_storage.py',
        Path(__file__).parent/'rate_calibration.py', Path(__file__).parent/'harness.py',
        ROOT/'src/semcache/experiments/cachegen/b1.py',
        ROOT/'src/semcache/experiments/cachegen/b2/format.py',
        ROOT/'src/semcache/experiments/cachegen/shared/core.py',
        ROOT/'src/semcache/experiments/cachegen/common.py',
        ROOT/'src/semcache/experiments/cachegen/shared/device_contract.py']
    manifest = dict(stage='C1.5C-3', status='RUNNING', gp_semcache_git=git_state(),
        capture_manifest_path=str(capture_path), capture_manifest_sha256=capture_sha,
        evaluation_block_ids=[b['block_id'] for b in blocks], **counts,
        model=capture['model_config'], model_metadata=capture.get('model_metadata'), dtype='float16',
        frozen_calibration=frozen, selected_policy=UNIFORM, device_contract=contract,
        seed=args.seed, seed_use='Recorded only; complete sorted evaluation w=3 partition, no sampling',
        policies=list(POLICIES), expected_arithmetic_encode_calls=expected,
        arithmetic_decode_limit=0, no_cdf_fit=True, no_model_inference=True,
        fixture_scan_passes=1, implementation_sha256={str(p): file_hash(p) for p in code_paths},
        completed_blocks=0, actual_arithmetic_encode_calls=0)
    write_json(out/'manifest.json', manifest)
    start = time.monotonic()
    def progress(done, total):
        if done % 25 == 0 or done == total:
            manifest.update(completed_blocks=done, actual_arithmetic_encode_calls=8*done)
            write_json(out/'manifest.json', manifest)
            print(f'[{done}/{total}] elapsed={time.monotonic()-start:.1f}s', flush=True)
    try:
        summary, comparison = evaluate(blocks, profiles, lambda b: load_fixture(root, b),
                                        device=contract['quantization_device_resolved'], progress=progress)
        if file_hash(capture_path) != capture_sha:
            raise ValueError('Capture manifest changed during holdout')
        if (file_hash(frozen['calibration_manifest_path']) != frozen['calibration_manifest_sha256'] or
                file_hash(frozen['selection_path']) != frozen['selection_sha256'] or
                any(file_hash(item['path']) != item['sha256'] for item in frozen['profiles'].values())):
            raise ValueError('Frozen calibration/profile changed during holdout')
        for path, expected_hash in manifest['implementation_sha256'].items():
            if file_hash(path) != expected_hash:
                raise ValueError('Holdout implementation changed during evaluation')
        write_json(out/'evaluation_summary.json', summary)
        write_json(out/'evaluation_comparison.json', comparison)
        rows = per_dataset_csv_rows(summary)
        write_csv(out/'per_dataset_summary.csv', rows, rows[0].keys())
        manifest.update(status='COMPLETED', completed_blocks=len(blocks),
                        actual_arithmetic_encode_calls=summary['arithmetic_encode_calls'],
                        actual_arithmetic_decode_calls=0, cdf_fit_calls=0,
                        output_sha256={name: file_hash(out/name) for name in (
                            'evaluation_summary.json', 'evaluation_comparison.json', 'per_dataset_summary.csv')})
    except BaseException as exc:
        manifest.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        if isinstance(exc, ObservationError):
            manifest['invalid_quantization_observation'] = exc.report
        raise
    finally:
        write_json(out/'manifest.json', manifest)
    return comparison
