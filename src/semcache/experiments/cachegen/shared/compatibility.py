"""Read-only C1 compatibility diagnostics. Never fit profiles or encode entropy."""
import ast
import hashlib
import inspect
from pathlib import Path
import subprocess

from ..codecs import Baseline
from ..common import file_hash, read_json


class UniformCompatibilityError(ValueError):
    def __init__(self, report):
        self.report = report
        differences = []
        for component, result in report['components'].items():
            if not result.get('exact_contract_match'):
                differences.append(f'{component}: first_unequal_index={result.get("first_unequal_index")}, '
                                   f'A={result.get("A_value")}, B={result.get("B_value")}, '
                                   f'A_dtype={result["A"].get("dtype")}, B_dtype={result["B"].get("dtype")}')
        super().__init__('C1.5 quantization differs from existing C1 UNIFORM_INT8 reconstruction; '
                         f'block_id={report["block_id"]}; '+ '; '.join(differences))


def tensor_info(tensor):
    import torch
    if not isinstance(tensor, torch.Tensor):
        return dict(type=type(tensor).__name__, is_tensor=False)
    return dict(is_tensor=True, shape=list(tensor.shape), dtype=str(tensor.dtype),
                device=str(tensor.device), element_count=tensor.numel(),
                stride=list(tensor.stride()), contiguous=tensor.is_contiguous())


def compare_tensors(a, b):
    """A is the saved/replayed C1 value, B the fresh C1.5 value; no tolerance."""
    import torch
    result = dict(A=tensor_info(a), B=tensor_info(b), torch_equal=False,
                  exact_contract_match=False, max_abs_diff=None, mean_abs_diff=None,
                  unequal_elements=None, first_unequal_index=None, A_value=None, B_value=None)
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return result
    ac, bc = a.detach().cpu(), b.detach().cpu()
    result['torch_equal'] = torch.equal(ac, bc)
    result['exact_contract_match'] = result['torch_equal'] and a.dtype == b.dtype
    if ac.shape != bc.shape:
        result['shape_mismatch'] = True
        return result
    unequal = ac != bc
    result['unequal_elements'] = int(unequal.sum())
    # Float64 reductions avoid FP16 subtraction/reduction overflow.
    diff = (ac.double()-bc.double()).abs()
    if diff.numel():
        def finite_value(value):
            value = value.item()
            import math
            return value if math.isfinite(value) else str(value)
        result['max_abs_diff'] = finite_value(diff.max())
        result['mean_abs_diff'] = finite_value(diff.mean())
        if result['unequal_elements']:
            index = tuple(unequal.nonzero()[0].tolist())
            result.update(first_unequal_index=list(index), A_value=finite_value(ac[index]),
                          B_value=finite_value(bc[index]))
    return result


def failure_report(root, item, block, saved, expected):
    keys = list(saved) if isinstance(saved, dict) else None
    components = {n: compare_tensors(saved.get(n) if isinstance(saved, dict) else None, tensor)
                  for n, tensor in zip('kv', expected)}
    return dict(status='MISMATCH', provenance='DIAGNOSTIC_ONLY', primary_result_eligible=False,
        block_id=block['block_id'], dataset=block['dataset'], token_group_size=block['token_group_size'],
        reconstruction_entry=item, reconstruction_path=str((root/item['file']).resolve()),
        original_fixture_path=str((root/block['file']).resolve()),
        original_fixture_sha256=block.get('sha256'), reconstruction_sha256=item['sha256'],
        saved_container_type=type(saved).__name__, saved_key_order=keys,
        expected_key_set=['k', 'v'], components=components,
        comparison='A=saved C1 UNIFORM_INT8; B=fresh CPU C1.5 Baseline reconstruction',
        symbols_scales_persisted=False,
        intermediate_comparison='Unavailable in C1 reconstruction: benchmark persisted K/V only; '
                                'replay below is current C1 source, not recovered historical intermediates')


def source_provenance():
    from .. import harness as c1_harness
    from . import tensors
    source = Path(inspect.getfile(Baseline)).resolve()
    repo = source.parents[4]
    relative = source.relative_to(repo).as_posix()
    def git(*args):
        result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else None
    def baseline_hash(text):
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'Baseline')
        return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    current = baseline_hash(source.read_text())
    history = []
    for revision in (git('log', '--format=%H', '--', relative) or '').splitlines():
        old = git('show', f'{revision}:{relative}')
        if old is not None:
            history.append(dict(revision=revision, baseline_ast_sha256=baseline_hash(old),
                                baseline_matches_current=baseline_hash(old) == current))
    return dict(baseline_source_path=str(source), baseline_source_sha256=file_hash(source),
        baseline_ast_sha256=current, current_git_head=(git('rev-parse', 'HEAD') or '').strip(),
        c1_harness_source_sha256=file_hash(inspect.getfile(c1_harness)),
        c15_tensor_boundary_sha256=file_hash(inspect.getfile(tensors)), history=history,
        historical_artifact_source_identity='C1 did not persist its Baseline source hash; '
                                            'local git history alone cannot prove SERAPH source identity')


def enrich_report(report, root, qkv, current_encoded, current_reconstructed):
    """Replay only this block through the actual C1 measured_baseline helper."""
    import torch
    from ..harness import measured_baseline
    env_path = root/'environment.json'
    env = read_json(env_path) if env_path.exists() else {}
    recorded = env.get('benchmark') or {}
    report.update(source=source_provenance(), c1_benchmark_environment=recorded,
                  current_torch=torch.__version__, current_cuda=torch.version.cuda,
                  original_components={n: tensor_info(qkv[n]) for n in 'kv'},
                  c15_intermediates={n: dict(symbols=tensor_info(symbols), scales=tensor_info(scales))
                                     for n, (symbols, scales) in zip('kv', current_encoded)},
                  static_path_checks=dict(
                      C1_layout='load_fixture validates FP16 [32,T,2560]; K/V .to(args.device); no layout transforms',
                      C15_layout='same load_fixture; direct CPU Baseline.encode before entropy flattening',
                      component_order='explicit K then V; saved dictionary accessed by key, never insertion order',
                      quantization_axis='Baseline.encode: amax(-1, keepdim=True), over hidden_dim',
                      scale_dtype='input.float(), FP32 scale; no FP16 cast before decode multiplication',
                      reconstruction_dtype='Baseline.decode: (symbols.float()*scale).half()',
                      rounding='same Baseline.encode torch.round + clamp(-127,127).to(int8)',
                      entropy_stage='not reached by this diagnostic'))
    # Recheck same-device delegation independently; this does not alter any data.
    direct = Baseline('UNIFORM_INT8').encode(qkv['k'], qkv['v'])
    report['direct_c1_cpu_vs_c15'] = {
        n: dict(symbols=compare_tensors(a[0], b[0]), scales=compare_tensors(a[1], b[1]))
        for n, a, b in zip('kv', direct, current_encoded)}
    recorded_device = recorded.get('device')
    devices = ['cpu']
    if recorded_device and recorded_device != 'cpu':
        devices.append(recorded_device)
    report['recorded_c1_device'] = recorded_device
    report['device_replays'] = {}
    saved = torch.load(report['reconstruction_path'], map_location='cpu', weights_only=True)
    for device in devices:
        replay = dict(is_recorded_c1_device=device == recorded_device)
        report['device_replays'][device] = replay
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            replay.update(status='UNAVAILABLE', reason='Recorded C1 CUDA device unavailable; no CPU substitution')
            continue
        try:
            encoded, reconstructed, _, _, _ = measured_baseline(
                Baseline('UNIFORM_INT8'), qkv['k'].to(device), qkv['v'].to(device), device)
            replay.update(status='REPLAYED_CURRENT_C1_SOURCE',
                saved_vs_replay={n: compare_tensors(saved.get(n), x) for n, x in zip('kv', reconstructed)},
                replay_vs_c15={n: compare_tensors(a, b) for n, a, b in zip('kv', reconstructed, current_reconstructed)},
                intermediates_vs_c15={n: dict(symbols=compare_tensors(a[0], b[0]),
                                             scales=compare_tensors(a[1], b[1]))
                                       for n, a, b in zip('kv', encoded, current_encoded)},
                swapped_components={n: compare_tensors(saved.get(n), reconstructed[1-i])
                                    for i, n in enumerate('kv')})
        except (RuntimeError, ValueError) as exc:
            replay.update(status='REPLAY_FAILED', reason=f'{type(exc).__name__}: {exc}')
    matched = [device for device, replay in report['device_replays'].items()
               if replay.get('saved_vs_replay') and
               all(x['exact_contract_match'] for x in replay['saved_vs_replay'].values())]
    report['devices_reproducing_saved_tensors_exactly'] = matched
    if recorded_device in matched:
        replay = report['device_replays'][recorded_device]
        report['proven_recorded_device_differences'] = {
            n: dict(symbols_differ=not replay['intermediates_vs_c15'][n]['symbols']['exact_contract_match'],
                    scales_differ=not replay['intermediates_vs_c15'][n]['scales']['exact_contract_match'],
                    reconstruction_differs=not replay['replay_vs_c15'][n]['exact_contract_match']) for n in 'kv'}
    report['conclusion'] = ('Recorded C1 device replay reproduces saved tensors exactly; inspect '
                            'intermediates_vs_c15 to identify symbols, scales, or reconstruction differences.'
                            if recorded_device in matched else
                            'No exact recorded-device reproduction established. Do not change codec or artifacts; '
                            'artifact/source provenance needs investigation.')
    return report
