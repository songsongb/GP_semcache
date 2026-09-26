"""Offline C1.5-A CLI; immutable C1 inputs, isolated research outputs."""
import argparse
import csv
from contextlib import contextmanager, nullcontext
from pathlib import Path
import statistics
import time

from ..common import (digest, environment, file_hash, load_fixture, percentile,
                      raw_bytes, read_json, write_json)
from ..harness import verify_capture, quality_hits
from ..codecs import Baseline
from .core import (CONFIG, MODES, Profile, cdf_from_counts, decode, encode, fit_profiles,
                   inspect_block, partition_ids, pool_accounting)
from .source_audit import audit
from .tensors import check_roundtrip, histograms, prepare, quantize

C1_FILES = ('capture_manifest.json', 'reconstruction_manifest.json', 'c1_block_raw.csv',
            'c1_summary.csv', 'c1_quality.csv')
PROFILE_FILES = dict(zip(MODES, ('shared_cdf_global.bin', 'shared_cdf_layergroup.bin')))
TIMING = dict(timing_device='cpu', timing_method='synchronous perf_counter_ns wall time; no CUDA work',
              warmup=5, measured_repetitions=20,
              timing_scope='entropy encode + framing/checksum; entropy decode + integrity checks; '
                           'quantization, tensor/byte conversion, reconstruction, I/O excluded')


def contained(path, parent):
    return Path(path).resolve().is_relative_to(Path(parent).resolve())


def protect_output(output, c1_root):
    output, c1_root = Path(output).resolve(), Path(c1_root).resolve()
    forbidden = (c1_root, Path('results/cachegen/c1').resolve())
    if any(contained(output, p) or contained(p, output) for p in forbidden):
        raise ValueError('C1 artifacts are immutable; use an isolated c1_5 output directory')
    # Existing symlinks may point outside an otherwise safe directory.
    if output.exists() and any(p.is_symlink() for p in output.rglob('*')):
        raise ValueError('C1.5 output must not contain symlinks')
    return output


def new_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)


def new_json(path, data):
    import json
    new_file(path, (json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())


def new_csv(path, rows):
    if not rows:
        raise ValueError('No rows to write')
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def snapshot(c1_root):
    return {name: file_hash(c1_root/name) for name in C1_FILES}


def implementation_hashes():
    import inspect
    from . import core, tensors
    return {name: file_hash(inspect.getfile(obj)) for name, obj in
            (('c1_baseline', Baseline), ('arithmetic_codec', core), ('tensor_boundary', tensors))}


def quality_selection(c1_root, blocks):
    """Reuse C1's recorded fixtures, never choose based on C1.5 outcomes."""
    with (c1_root/'c1_quality.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    by_id = {}
    for row in rows:
        if row['compression_mode'] in ('NATIVE', 'FP16_RAW', 'UNIFORM_INT8'):
            if row['status'] != 'MEASURED':
                raise ValueError('C1 quality controls must have passed')
            modes = by_id.setdefault(row['fixture_id'], set())
            if row['compression_mode'] in modes:
                raise ValueError('Duplicate C1 quality fixture/mode')
            modes.add(row['compression_mode'])
    expected = {'NATIVE', 'FP16_RAW', 'UNIFORM_INT8'}
    selected = [b for b in blocks if b['block_id'] in by_id]
    if (len(by_id) != 4 or len(selected) != 4 or any(m != expected for m in by_id.values())
            or any(b['partition'] != 'evaluation' for b in selected)
            or {(b['dataset'], b['token_group_size']) for b in selected} !=
               {('snips', 3), ('snips', 10), ('multiwoz', 3), ('multiwoz', 10)}):
        raise ValueError('Expected the same four completed C1 controlled fixtures')
    return selected


@contextmanager
def inputs(args):
    root = args.capture_manifest.resolve().parent
    if args.capture_manifest.name != 'capture_manifest.json':
        raise ValueError('Use the immutable C1 capture_manifest.json')
    out = protect_output(args.output_dir, root)
    before = snapshot(root)
    capture = read_json(args.capture_manifest)
    verify_capture(capture)
    if capture.get('scope') != 'base_raw_unscaled_linear_projection; no LoRA adapter':
        raise ValueError('Expected C1 bare-OPT base projection capture scope')
    ids = partition_ids(capture['blocks'])
    rec = read_json(root/'reconstruction_manifest.json')
    if rec['capture_manifest_sha256'] != digest(capture):
        raise ValueError('C1 reconstruction/capture identity mismatch')
    selected = quality_selection(root, capture['blocks'])
    out.mkdir(parents=True, exist_ok=True)
    try:
        yield root, out, capture, rec, ids, selected, before
    finally:
        if snapshot(root) != before:
            raise ValueError('Immutable C1 inputs changed during execution')


def update_state(out, **fields):
    path = out/'manifest.json'
    state = read_json(path) if path.exists() else dict(stage='C1.5-A',
        description='CacheGen-inspired fixed/shared entropy-model adaptation for SemCache KV storage',
        provenance='RESEARCH_EXTENSION', quantization_changed=False, production_cache_changed=False)
    state.update(fields)
    write_json(path, state)


def record_environment(out, stage):
    path = out/'environment.json'
    env = read_json(path) if path.exists() else {}
    env[stage] = environment()
    write_json(path, env)


def profile(args):
    with inputs(args) as (root, out, capture, rec, ids, selected, hashes):
        if (out/'manifest.json').exists() or (out/'profile_manifest.json').exists():
            raise ValueError('Profile already exists; frozen profiles cannot be refit in place')
        source = audit(args.cachegen_repo)
        def load(block):
            assert block['partition'] == 'calibration'
            qkv = load_fixture(root, block)
            return histograms(quantize(qkv['k'], qkv['v']))
        profiles, fitted, counts = fit_profiles(capture['blocks'], load)
        if fitted != ids['calibration'] or set(fitted) & set(ids['evaluation']):
            raise ValueError('Calibration-only fitting invariant failed')
        entries = {}
        for mode, model in profiles.items():
            new_file(out/PROFILE_FILES[mode], model.to_bytes())
            entries[mode] = dict(file=PROFILE_FILES[mode], sha256=model.sha256,
                logical_shared_profile_tensor_bytes=model.logical_bytes,
                serialized_shared_profile_bytes=len(model.to_bytes()),
                calibration_counts=counts[mode], calibration_counts_sha256=digest(counts[mode]))
        result = dict(schema_version=1, status='FROZEN_BEFORE_EVALUATION', config=CONFIG,
            implementation_sha256=implementation_hashes(),
            c1_file_sha256=hashes, capture_manifest_sha256=hashes['capture_manifest.json'],
            capture_manifest_canonical_sha256=digest(capture),
            calibration_block_ids=ids['calibration'], evaluation_block_ids=ids['evaluation'],
            calibration_ids_sha256=digest(ids['calibration']), evaluation_ids_sha256=digest(ids['evaluation']),
            fitted_block_ids=fitted, fitted_ids_sha256=digest(fitted), fitting_partition='calibration',
            model_metadata=capture['model_metadata'], quality_fixture_ids=[b['block_id'] for b in selected],
            profiles=entries, official_source_audit=source,
            fitting_policy='Each calibration block contributes once, including both T granularities; '
                           'overlapping tokens across T are counted per captured block; no evaluation reads')
        new_json(out/'profile_manifest.json', result)
        update_state(out, profile_status='FROZEN', benchmark_status='NOT_RUN', quality_status='NOT_RUN',
                     profile_manifest_sha256=file_hash(out/'profile_manifest.json'))
        record_environment(out, 'profile')


def load_profiles(out, hashes, ids):
    state = read_json(out/'manifest.json')
    if file_hash(out/'profile_manifest.json') != state['profile_manifest_sha256']:
        raise ValueError('Profile manifest SHA256 mismatch')
    pm = read_json(out/'profile_manifest.json')
    if (pm['config'] != CONFIG or pm['status'] != 'FROZEN_BEFORE_EVALUATION' or pm['c1_file_sha256'] != hashes
            or pm['implementation_sha256'] != implementation_hashes()):
        raise ValueError('Frozen profile configuration/C1 provenance mismatch')
    for partition in ('calibration', 'evaluation'):
        if pm[partition+'_block_ids'] != ids[partition] or pm[partition+'_ids_sha256'] != digest(ids[partition]):
            raise ValueError('Profile partition ID mismatch')
    if pm['fitted_block_ids'] != ids['calibration'] or pm['fitted_ids_sha256'] != digest(ids['calibration']):
        raise ValueError('Profile must fit exactly calibration IDs')
    models = {}
    for mode in MODES:
        item = pm['profiles'][mode]
        if item['file'] != PROFILE_FILES[mode]:
            raise ValueError('Unexpected shared profile filename')
        model = Profile.from_bytes((out/item['file']).read_bytes(), item['sha256'])
        expected = Profile(mode, tuple(cdf_from_counts(c) for c in item['calibration_counts']))
        if (model != expected or digest(item['calibration_counts']) != item['calibration_counts_sha256']
                or model.logical_bytes != item['logical_shared_profile_tensor_bytes']
                or len(model.to_bytes()) != item['serialized_shared_profile_bytes']):
            raise ValueError('Profile differs from recorded calibration counts/accounting')
        models[mode] = model
    return models, pm


def measure(call):
    for _ in range(TIMING['warmup']):
        call()
    times = []
    value = None
    for _ in range(TIMING['measured_repetitions']):
        start = time.perf_counter_ns()
        value = call()  # CPU synchronous work only; no CUDA events/transfers.
        times.append((time.perf_counter_ns()-start)/1e6)
    return value, dict(mean=statistics.mean(times), median=statistics.median(times),
                       p95=percentile(times, .95)), times


def existing_uniform(root, rec, block, expected):
    import torch
    items = [r for r in rec['reconstructed'] if r['block_id'] == block['block_id']
             and r['compression_mode'] == 'UNIFORM_INT8']
    if len(items) != 1:
        raise ValueError('Expected exactly one existing C1 UNIFORM_INT8 reconstruction')
    item = items[0]
    path = root/item['file']
    if not contained(path, root) or file_hash(path) != item['sha256']:
        raise ValueError('C1 reconstruction path/SHA256 mismatch')
    kv = torch.load(path, map_location='cpu', weights_only=True)
    if set(kv) != {'k', 'v'} or not all(kv[n].dtype == torch.float16 and torch.equal(kv[n], x)
                                      for n, x in zip('kv', expected)):
        raise ValueError('C1.5 quantization differs from existing C1 UNIFORM_INT8 reconstruction')
    return kv


def benchmark(args):
    with inputs(args) as (root, out, capture, rec, ids, selected, hashes):
        if (out/'benchmark_manifest.json').exists() or (out/'c15_block_raw.csv').exists():
            raise ValueError('Benchmark output exists; choose a new isolated run directory')
        profiles, pm = load_profiles(out, hashes, ids)
        frozen = {mode: model.to_bytes() for mode, model in profiles.items()}
        rows, entries, repeats = [], [], []
        for block in capture['blocks']:
            if block['partition'] != 'evaluation':
                continue
            assert block['block_id'] in ids['evaluation'] and block['block_id'] not in ids['calibration']
            qkv = load_fixture(root, block)
            quantized = quantize(qkv['k'], qkv['v'])
            baseline = Baseline('UNIFORM_INT8').decode(quantized)
            existing_uniform(root, rec, block, baseline)
            raw = raw_bytes(block['token_group_size'])
            for mode in ('FP16_RAW', 'UNIFORM_INT8', *MODES):
                row = dict(block_id=block['block_id'], dataset=block['dataset'],
                    token_group_size=block['token_group_size'], compression_mode=mode, **raw,
                    encoded_payload_bytes=0, local_metadata_bytes=0, scale_metadata_bytes=0,
                    symbol_roundtrip_exact=True if mode in MODES else None,
                    uniform_reconstruction_exact=True if mode != 'FP16_RAW' else None,
                    provenance=('MEASURED' if mode == 'FP16_RAW' else 'MEASURED; REPRODUCTION_CHOICE'
                                if mode == 'UNIFORM_INT8' else 'MEASURED_RESEARCH_EXTENSION'),
                    timing_device='not_retimed' if mode not in MODES else TIMING['timing_device'],
                    timing_method='storage baseline only' if mode not in MODES else TIMING['timing_method'],
                    warmup=0 if mode not in MODES else 5, measured_repetitions=0 if mode not in MODES else 20,
                    encode_ms_mean=None, encode_ms_median=None, encode_ms_p95=None,
                    decode_ms_mean=None, decode_ms_median=None, decode_ms_p95=None)
                if mode == 'FP16_RAW':
                    row['encoded_payload_bytes'] = raw['raw_kv_bytes']
                elif mode == 'UNIFORM_INT8':
                    row['encoded_payload_bytes'], row['local_metadata_bytes'] = Baseline(mode).sizes(quantized)
                    row['scale_metadata_bytes'] = row['local_metadata_bytes']
                else:
                    model = profiles[mode]
                    prepared = prepare(model, quantized)
                    blob = encode(model, *prepared)
                    check_roundtrip(model, quantized, blob)
                    row.update(inspect_block(blob, model)[3])
                    measured_blob, et, er = measure(lambda: encode(model, *prepared))
                    decoded, dt, dr = measure(lambda: decode(blob, model))
                    if measured_blob != blob or decoded != prepared:
                        raise ValueError('Timed codec differs from validated deterministic roundtrip')
                    for name, stats in (('encode', et), ('decode', dt)):
                        row.update({name+'_ms_'+key: value for key, value in stats.items()})
                    relative = f'bitstreams/{mode}/{digest(block["block_id"])[:24]}.bin'
                    new_file(out/relative, blob)
                    entries.append(dict(block_id=block['block_id'], compression_mode=mode,
                                        file=relative, sha256=file_hash(out/relative)))
                    repeats.append(dict(block_id=block['block_id'], compression_mode=mode,
                                        encode_wall_ms=er, decode_wall_ms=dr))
                    if model.to_bytes() != frozen[mode]:
                        raise ValueError('Evaluation modified frozen CDF')
                rows.append(row)
            print(f'Validated {block["block_id"]}', flush=True)
        summary = []
        for mode in ('FP16_RAW', 'UNIFORM_INT8', *MODES):
            group = [r for r in rows if r['compression_mode'] == mode]
            size = len(frozen[mode]) if mode in MODES else 0
            summary.append(dict(compression_mode=mode, pool='all evaluation blocks; both datasets/T',
                **pool_accounting(group, size),
                logical_shared_profile_tensor_bytes=profiles[mode].logical_bytes if mode in MODES else 0,
                provenance=group[0]['provenance']))
        load_profiles(out, hashes, ids)  # Recheck on-disk freeze after evaluation.
        new_csv(out/'c15_block_raw.csv', rows)
        new_csv(out/'c15_summary.csv', summary)
        new_json(out/'timing_repeats.json', dict(**TIMING, repeats=repeats))
        new_json(out/'benchmark_manifest.json', dict(profile_manifest_sha256=file_hash(out/'profile_manifest.json'),
            evaluation_ids_sha256=digest(ids['evaluation']), bitstreams=entries,
            **TIMING, all_symbol_roundtrips_exact=True, all_uniform_reconstructions_exact=True,
            profile_unchanged_after_evaluation=True))
        update_state(out, benchmark_status='MEASURED', benchmark_manifest_sha256=file_hash(out/'benchmark_manifest.json'))
        record_environment(out, 'benchmark')


def quality(args):
    import torch
    from semcache.models.loader import load_model
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.system_cost.base_projection import base_projection_path
    from semcache.evaluation.logit_metrics import compare_logits
    from semcache.utils.seed import seed_everything
    with inputs(args) as (root, out, capture, rec, ids, selected, hashes):
        if (out/'c15_quality.csv').exists():
            raise ValueError('Quality output already exists')
        profiles, pm = load_profiles(out, hashes, ids)
        state = read_json(out/'manifest.json')
        if file_hash(out/'benchmark_manifest.json') != state.get('benchmark_manifest_sha256'):
            raise ValueError('Benchmark manifest SHA256 mismatch')
        bm = read_json(out/'benchmark_manifest.json')
        if bm['profile_manifest_sha256'] != file_hash(out/'profile_manifest.json'):
            raise ValueError('Benchmark used different profiles')
        if [b['block_id'] for b in selected] != pm['quality_fixture_ids']:
            raise ValueError('Quality fixture selection differs from frozen C1 selection')
        cfg = dict(capture['model_config'], device=args.device, local_files_only=True)
        cfg.update(revision=capture['model_metadata']['resolved_model_revision'],
                   tokenizer_revision=capture['model_metadata']['resolved_tokenizer_revision'])
        seed_everything(capture['seed'])
        model, _, meta = load_model(cfg)
        if (meta['resolved_model_revision'] != cfg['revision'] or
                meta['resolved_tokenizer_revision'] != cfg['tokenizer_revision']):
            raise ValueError('Quality model/tokenizer revision mismatch')
        adapter = OPTModelAdapter(model)
        rows = []
        for block in selected:
            original = load_fixture(root, block)
            quantized = quantize(original['k'], original['v'])
            uniform = existing_uniform(root, rec, block, Baseline('UNIFORM_INT8').decode(quantized))
            inputs_tensor = torch.tensor([block['query_token_ids']], device=args.device)
            length, t = len(block['query_token_ids']), block['token_group_size']
            def forward(qkv=None):
                hits = quality_hits(adapter, qkv, block) if qkv is not None else None
                with torch.inference_mode(), (base_projection_path(adapter, hits, length)
                                               if hits is not None else nullcontext()) as audit_record:
                    logits = model(input_ids=inputs_tensor, use_cache=False).logits.detach().cpu()
                if hits is not None and any(r['reused_projection_rows'] != t or r['native_projection_rows'] != length-t
                                            for r in audit_record.records.values()):
                    raise ValueError('Quality physical row accounting mismatch')
                return logits
            native, raw = forward(), forward(original)
            if not torch.equal(native, raw):
                raise ValueError('C1 FP16_RAW exact native quality control no longer holds')
            uniform_logits = forward(dict(q=original['q'], **uniform))
            outputs = {'NATIVE': native, 'FP16_RAW': raw, 'UNIFORM_INT8': uniform_logits}
            for mode, profile_model in profiles.items():
                items = [r for r in bm['bitstreams'] if r['block_id'] == block['block_id'] and r['compression_mode'] == mode]
                if len(items) != 1:
                    raise ValueError('Missing/duplicate quality bitstream')
                path = out/items[0]['file']
                if not contained(path, out) or file_hash(path) != items[0]['sha256']:
                    raise ValueError('Quality bitstream SHA256 mismatch')
                k, v = check_roundtrip(profile_model, quantized, path.read_bytes())
                candidate = forward(dict(q=original['q'], k=k, v=v))
                if not torch.equal(candidate, uniform_logits):
                    raise ValueError('Shared-CDF logits differ from UNIFORM_INT8; fail closed')
                outputs[mode] = candidate
            for mode, logits in outputs.items():
                metric = compare_logits(native, logits, block['start_position'])
                rows.append(dict(fixture_id=block['block_id'], dataset=block['dataset'], token_group_size=t,
                    compression_mode=mode, max_logit_abs_error=metric['max_abs_logit_diff'],
                    logit_rel_l2=metric['relative_l2_logit_diff'], kl_divergence=metric['affected_suffix_mean_kl'],
                    argmax_agreement=float((native.argmax(-1) == logits.argmax(-1)).float().mean()),
                    max_logit_difference_vs_uniform_int8=(logits.double()-uniform_logits.double()).abs().max().item(),
                    status='MEASURED', provenance='MEASURED_RESEARCH_EXTENSION' if mode in MODES else
                    'MEASURED; REPRODUCTION_CHOICE' if mode == 'UNIFORM_INT8' else 'MEASURED'))
        load_profiles(out, hashes, ids)
        new_csv(out/'c15_quality.csv', rows)
        update_state(out, quality_status='MEASURED', quality_fixture_ids=pm['quality_fixture_ids'],
                     shared_logits_exactly_equal_uniform=True, kl_direction='native || candidate; affected suffix mean',
                     argmax_scope='all query positions')
        record_environment(out, 'quality')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('profile', 'benchmark', 'quality'):
        p = sub.add_parser(name)
        p.add_argument('--capture-manifest', type=Path, default=Path('results/cachegen/c1/capture_manifest.json'))
        p.add_argument('--output-dir', type=Path, default=Path('results/cachegen/c1_5'))
        if name == 'profile':
            p.add_argument('--cachegen-repo', type=Path, default=Path('/data/khuss/repos/CacheGen'))
        if name == 'quality':
            p.add_argument('--device', choices=('cpu', 'cuda', 'cuda:0'), default='cuda')
    args = parser.parse_args(argv)
    {'profile': profile, 'benchmark': benchmark, 'quality': quality}[args.command](args)
