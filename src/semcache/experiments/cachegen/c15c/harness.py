"""Parity, capture smoke, rate calibration, and frozen-policy holdout."""
import argparse
from collections import defaultdict, deque
import math
from pathlib import Path
import random
import subprocess

from ..common import digest, file_hash, load_fixture, read_json, write_json, write_csv
from ..harness import verify_capture
from ..shared.device_contract import resolve_contract
from .parity import run_parity, print_parity
from .policy import CACHEGEN_RELEASED_QL2, DEFAULT_CANDIDATE_BINS
from .reference import source_manifest
from .storage import load_profile, roundtrip

ROOT = Path(__file__).resolve().parents[5]
RESULTS = ROOT/'results/cachegen/c1_5c'
SUM_FIELDS = ('element_count', 'error_sq_sum', 'original_sq_sum', 'reconstruction_sq_sum', 'dot_sum')


def select_blocks(capture, count, seed):
    if not 1 <= count <= 16:
        raise ValueError('Smoke is limited to 1..16 existing blocks; default 12')
    verify_capture(capture)
    if (capture['model_config']['name'] != 'facebook/opt-2.7b' or
            capture['model_config']['dtype'] != 'float16' or
            capture.get('scope') != 'base_raw_unscaled_linear_projection; no LoRA adapter'):
        raise ValueError('Expected frozen base OPT-2.7B FP16 C1 captures')
    groups = defaultdict(list)
    for block in capture['blocks']:
        if block['partition'] == 'evaluation' and block['token_group_size'] == 3:
            groups[block['dataset']].append(block)
    rng = random.Random(seed)
    buckets = []
    for dataset in sorted(groups):
        pool = sorted(groups[dataset], key=lambda b: b['block_id'])
        rng.shuffle(pool)
        buckets.append(deque(pool))
    if sum(map(len, buckets)) < count:
        raise ValueError('Insufficient existing evaluation w=3 captures; no capture generation is available')
    selected = []
    while len(selected) < count:
        for bucket in buckets:
            if bucket and len(selected) < count:
                selected.append(bucket.popleft())
    return selected


def metric_sums(original, reconstructed):
    x, y = original.detach().double().cpu(), reconstructed.detach().double().cpu()
    error = x-y
    return dict(element_count=x.numel(), error_sq_sum=(error*error).sum().item(),
        original_sq_sum=(x*x).sum().item(), reconstruction_sq_sum=(y*y).sum().item(),
        dot_sum=(x*y).sum().item(), max_abs_error=error.abs().max().item())


def metrics(sums):
    mse = sums['error_sq_sum']/sums['element_count']
    denominator = math.sqrt(sums['original_sq_sum']*sums['reconstruction_sq_sum'])
    return dict(MSE=mse, RMSE=math.sqrt(mse),
        relative_l2=math.sqrt(sums['error_sq_sum']/sums['original_sq_sum']) if sums['original_sq_sum'] else None,
        cosine_similarity=sums['dot_sum']/denominator if denominator else None,
        max_abs_error=sums['max_abs_error'])


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        for group in ('overall', f"layer_{row['layer']:02}", f"bins_{row['bins']}"):
            groups[row['role'], group].append(row)
    result = []
    for (role, group), entries in sorted(groups.items()):
        sums = {field: sum(r[field] for r in entries) for field in SUM_FIELDS}
        sums['max_abs_error'] = max(r['max_abs_error'] for r in entries)
        result.append(dict(role=role, group=group, block_count=len({r['block_id'] for r in entries}),
            element_count=sums['element_count'], fp16_raw_bytes=2*sums['element_count'],
            nominal_quantized_symbol_count=sums['element_count'], **metrics(sums)))
    return result


def git_state():
    def git(*args):
        return subprocess.run(['git', '-C', str(ROOT), *args], check=True, text=True, capture_output=True).stdout.strip()
    return dict(commit=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'), status=git('status', '--short'))


def run_smoke(args):
    """SERAPH ONLY. Reads captures, never loads a model or regenerates a dataset."""
    import torch
    source = source_manifest(args.cachegen_repo)  # Strictly verify live pinned reference.
    parity = run_parity(args.cachegen_repo)
    if parity['status'] != 'PASS':
        raise ValueError('Released CPU reference parity must pass before smoke')
    capture_path = args.capture_manifest.resolve()
    root = capture_path.parent
    capture = read_json(capture_path)
    selected = select_blocks(capture, args.num_blocks, args.seed)
    for block in selected:
        if not (root/block['file']).resolve().is_relative_to(root):
            raise ValueError('Capture path escapes capture directory')
    contract = resolve_contract(root)
    if not contract['quantization_device_resolved'].startswith('cuda:'):
        raise ValueError('Real smoke requires recorded C1 CUDA device; no CPU fallback')
    profile, profile_info = load_profile(args.b2_profile_dir) if args.b2_profile_dir else (None, None)
    out = args.output_dir.resolve()/'smoke'
    for input_root in (root, args.cachegen_repo.resolve(), args.b2_profile_dir.resolve() if args.b2_profile_dir else root):
        if out.is_relative_to(input_root) or input_root.is_relative_to(out):
            raise ValueError('Output may not overlap existing capture/reference/storage inputs')
    # Fresh namespace only. Reruns require a different --output-dir subdirectory.
    out.mkdir(parents=True, exist_ok=False)
    code_paths = [Path(__file__), *Path(__file__).parent.glob('*.py'),
                  ROOT/'src/semcache/experiments/cachegen/b2/format.py',
                  ROOT/'src/semcache/experiments/cachegen/b1.py',
                  ROOT/'src/semcache/experiments/cachegen/shared/core.py']
    manifest = dict(stage='C1.5C-1', status='INCOMPLETE', classification='SMALL_DIAGNOSTIC_ONLY',
        model='facebook/opt-2.7b', model_dtype='float16', window_size=3, seed=args.seed,
        model_config=capture['model_config'], model_metadata=capture.get('model_metadata'),
        policy=CACHEGEN_RELEASED_QL2.name, layer_profile=CACHEGEN_RELEASED_QL2.profile(),
        reference=source, gp_semcache_git=git_state(), device_contract=contract,
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        selected_block_ids=[b['block_id'] for b in selected],
        selected_blocks=selected, capture_manifest_path=str(capture_path),
        capture_manifest_sha256=file_hash(capture_path), capture_manifest_content_hash=digest(capture),
        source_capture_paths=[str((root/b['file']).resolve()) for b in selected],
        input_environment_path=str(root/'environment.json'), b2_profile=profile_info,
        implementation_sha256={str(p): file_hash(p) for p in code_paths},
        reconstruction_metrics='Original FP16 vs final FP16 reconstruction; float64 pooled sums; no model-quality claims',
        aggregation='Pooled concatenated elements, not mean layer/block ratios; zero-norm ratios are null',
        storage_policy='Same B2 four roles, +127 signed mapping, modulus 255, framing and frozen CDF; explicit T=3',
        storage_executed=False, completed_blocks=0, no_model_inference=True)
    write_json(out/'manifest.json', manifest)
    write_json(out/'c15c_release_parity.json', parity)
    rows, storage_rows = [], []
    try:
        if profile:
            (out/'storage').mkdir()
            (out/'storage/b2_profile.bin').write_bytes(profile.to_bytes())
        for block in selected:
            qkv = load_fixture(root, block)  # Existing hash, shape, dtype and finiteness checks.
            encoded = tuple(CACHEGEN_RELEASED_QL2.quantize(qkv[role.lower()].to(contract['quantization_device_resolved']), role)
                            for role in ('K', 'V'))
            reconstruction = tuple(e.reconstructed.cpu() for e in encoded)
            for role, e, y in zip(('K', 'V'), encoded, reconstruction):
                for layer in range(32):
                    sums = metric_sums(qkv[role.lower()][layer], y[layer])
                    rows.append(dict(block_id=block['block_id'], dataset=block['dataset'], role=role,
                        layer=layer, bins=int(e.bins[layer].item()), quantization_limit=int(e.limits[layer].item()),
                        fp16_raw_bytes=2*sums['element_count'], nominal_quantized_symbol_count=sums['element_count'],
                        zero_maxabs_vector_count=int((e.maxabs[layer] == 0).sum().item()),
                        **sums, **{k: v for k, v in metrics(sums).items() if k != 'max_abs_error'}))
            if profile:
                blob, restored, proof = roundtrip(encoded, profile)
                if not all(torch.equal(a.cpu(), b) for a, b in zip(restored, reconstruction)):
                    raise ValueError('Storage reconstruction differs from quantization-only reconstruction')
                relative_path = f"storage/block_{manifest['completed_blocks']:03}.bin"
                (out/relative_path).write_bytes(blob)
                storage_rows.append(dict(block_id=block['block_id'], file=relative_path,
                    sha256=file_hash(out/relative_path), actual_bitstream_bytes=len(blob), **proof))
                manifest['storage_executed'] = True
            manifest['completed_blocks'] += 1
            write_json(out/'manifest.json', manifest)
        # Inputs must still match their frozen hashes after use.
        if file_hash(capture_path) != manifest['capture_manifest_sha256']:
            raise ValueError('Capture manifest changed during smoke')
        for block in selected:
            if file_hash(root/block['file']) != block['sha256']:
                raise ValueError('Capture changed during smoke')
        for path, expected in manifest['implementation_sha256'].items():
            if file_hash(path) != expected:
                raise ValueError('Implementation changed during smoke')
        source_manifest(args.cachegen_repo)
        if profile and load_profile(args.b2_profile_dir)[1] != profile_info:
            raise ValueError('Frozen storage profile changed during smoke')
        summary = aggregate(rows)
        write_csv(out/'c15c_smoke_raw.csv', rows, rows[0].keys())
        write_json(out/'c15c_smoke_summary.json', dict(quantization_reconstruction_error=summary))
        write_json(out/'c15c_storage_roundtrip.json', dict(
            status='PASS' if profile else 'NOT_REQUESTED', blocks=storage_rows,
            storage_roundtrip_symbol_mismatch=0 if profile else None,
            quantization_reconstruction_error_file='c15c_smoke_summary.json',
            all_reconstructions_equal_to_quantization_only=True if profile else None))
        manifest.update(status='COMPLETED', storage_roundtrip_symbol_mismatch=0 if profile else None)
        print('role group    relative-L2 cosine max-abs')
        for r in summary:
            if r['group'] == 'overall' or r['group'].startswith('bins_'):
                print(r['role'], r['group'], r['relative_l2'], r['cosine_similarity'], r['max_abs_error'])
        print('Storage symbol mismatch:', manifest['storage_roundtrip_symbol_mismatch'])
    except BaseException as exc:
        manifest.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(out/'manifest.json', manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description='C1.5C QL2 parity, smoke, rate calibration, and frozen-policy holdout')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('parity', 'smoke'):
        p = sub.add_parser(name)
        p.add_argument('--output-dir', type=Path, default=RESULTS)
        p.add_argument('--cachegen-repo', type=Path, default=None if name == 'parity' else Path('/data/khuss/repos/CacheGen'))
        if name == 'smoke':
            p.add_argument('--capture-manifest', type=Path, default=ROOT/'results/cachegen/c1/capture_manifest.json')
            p.add_argument('--num-blocks', type=int, default=12)
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--b2-profile-dir', type=Path, help='Optional existing frozen B2 profile directory; never fits profiles')
    p = sub.add_parser('rate-calibrate', help='SERAPH calibration captures only; fits fresh policy-specific CDFs')
    p.add_argument('--capture-manifest', type=Path, default=ROOT/'results/cachegen/c1/capture_manifest.json')
    p.add_argument('--cachegen-repo', type=Path, default=Path('/data/khuss/repos/CacheGen'))
    p.add_argument('--output-dir', type=Path, default=RESULTS/'rate_calibration')
    p.add_argument('--candidate-bins', type=int, nargs='+', default=list(DEFAULT_CANDIDATE_BINS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', help='Optional assertion: must match recorded C1 device, no fallback')
    p = sub.add_parser('holdout-compare', help='SERAPH evaluation only; frozen QL2 versus frozen K20/V16')
    p.add_argument('--capture-manifest', type=Path, default=ROOT/'results/cachegen/c1/capture_manifest.json')
    p.add_argument('--rate-calibration-dir', type=Path, default=RESULTS/'rate_calibration')
    p.add_argument('--output-dir', type=Path, default=RESULTS/'holdout')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dry-run', action='store_true', help='Manifest/profile preflight only; no fixtures, CUDA or output files')
    args = parser.parse_args(argv)
    if not args.output_dir.resolve().is_relative_to(RESULTS.resolve()):
        parser.error('All new outputs must be under results/cachegen/c1_5c')
    if args.command == 'parity':
        report = run_parity(args.cachegen_repo)
        report['gp_semcache_git'] = git_state()
        write_json(args.output_dir/'c15c_release_parity.json', report)
        print_parity(report)
        if report['status'] != 'PASS':
            raise SystemExit(1)
    elif args.command == 'smoke':
        run_smoke(args)
    elif args.command == 'rate-calibrate':
        from .rate_calibration import run_rate_calibration
        run_rate_calibration(args)
    else:
        from .holdout import run_holdout
        run_holdout(args)
