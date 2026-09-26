"""Calibration-only profiles and resumable, single-pass C1.5-B2 storage jobs."""
import argparse
import csv
from hashlib import sha256
import inspect as pyinspect
import json
from pathlib import Path
import struct
import subprocess
import time

from .. import b1, codecs, common
from ..common import digest, environment, file_hash, load_fixture, raw_bytes, read_json
from ..shared import core, tensors, device_contract, harness as shared_harness
from ..shared.full_storage import (atomic_bytes, atomic_json, atomic_csv, exclusive_run,
                                   pair_name, verify_tensor_files)
from . import format as fmt

EXPECTED = {'calibration': {'snips': 103, 'multiwoz': 138, 'ALL': 241},
            'evaluation': {'snips': 93, 'multiwoz': 143, 'ALL': 236}}
FULL_PAIRS = 708
SMOKE_PAIRS = 6

RULE = dict(provenance='REPRODUCTION_CHOICE', threshold_percent=3.0,
    primary_mode=fmt.MODES[1], control_mode=fmt.MODES[0], metric='encoded_kv_pool_bytes',
    success='All exactness checks pass; primary KV pool is smaller than fair control AND reduction >=3%.',
    dataset_direction='Report both datasets; not an additional primary success condition',
    threshold_tuned_on_evaluation=False)
CONFIG = dict(schema_version=1, modes=list(fmt.MODES), distributions_per_mode=4,
    stream_order=['K_ANCHOR_RAW', 'K_NONANCHOR_OR_RESIDUAL', 'V_ANCHOR_RAW', 'V_NONANCHOR_OR_RESIDUAL'],
    transform_config_sha256=digest(b1.CONFIG), transform_function='cachegen.b1.transform (unchanged)',
    alphabet=core.CONFIG['alphabet'], smoothing=core.CONFIG['smoothing'], total_frequency=core.TOTAL,
    normalization=core.CONFIG['normalization'], coder='cachegen.shared.core.arithmetic_encode/arithmetic_decode (unchanged)',
    serialization='C1.5-A little-endian PROFILE_HEADER/BLOCK_HEADER; four uint32 stream lengths; unchanged FP32 scales; SHA256 trailer; distinct B2 magic/mode namespace',
    grouping='T=10 only; token 0 anchor; K/V and token roles separate; layers/datasets pooled',
    official_cachegen_reproduction=False, no_new_lossy_quantization=True, no_model_inference=True,
    Q_storage='FP16', B2_primary='K+V residual', hybrid_variant='POST_HOC_EXPLORATORY', decision_rule=RULE)
TIMING = dict(classification='SINGLE_PASS_OPERATIONAL_TIMING', latency_result='NOT_PRIMARY_LATENCY_RESULT',
    warmup=0, encode_passes_per_pair=1, decode_passes_per_pair=1, comparable_to_c1_timing=False)
CORRECTNESS = ('symbol_roundtrip_exact', 'inverse_transform_exact', 'scales_preserved_exact',
               'saved_c1_reconstruction_exact')


def source_hashes():
    from .. import harness as c1_harness
    from ..shared import full_storage
    modules = (common, codecs, c1_harness, b1, core, tensors, device_contract, shared_harness,
               full_storage, fmt, pyinspect.getmodule(source_hashes))
    return {str(Path(pyinspect.getfile(m)).resolve()): file_hash(pyinspect.getfile(m)) for m in modules}


def assert_unchanged(hashes):
    if any(file_hash(path) != expected for path, expected in hashes.items()):
        raise ValueError('Frozen input/source changed during B2')


def safe_output(args):
    parent = args.output_dir.resolve()
    for frozen in (args.capture_manifest.resolve().parent, args.c15a_dir.resolve(), args.b1_dir.resolve()):
        shared_harness.protect_output(parent, frozen)
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def reference_inputs(aroot, evaluation):
    """Read exact matching A T=10 summaries and their per-block reconciliation."""
    path = aroot/'full_storage/c15_full_summary.csv'
    strata = {'T=10': 'ALL', 'SNIPS/T10': 'snips', 'MultiWOZ/T10': 'multiwoz'}
    modes = ('UNIFORM_INT8', *core.MODES)
    with path.open(newline='') as f:
        source_rows = [r for r in csv.DictReader(f) if r['stratum'] in strata and r['compression_mode'] in modes]
    by_pool = {}
    for row in source_rows:
        key = strata[row['stratum']], row['compression_mode']
        if key in by_pool:
            raise ValueError('Duplicate A T=10 summary reference')
        by_pool[key] = row
    if len(by_pool) != 9:
        raise ValueError('Missing exact A T=10 summary rows')
    selected = {b['block_id']: b for b in evaluation}
    blocks = {}
    raw_path = aroot/'full_storage/c15_full_block_raw.csv'
    with raw_path.open(newline='') as f:
        for row in csv.DictReader(f):
            key = row['block_id'], row['compression_mode']
            if key[0] not in selected or key[1] not in core.MODES:
                continue
            if key in blocks or row['status'] != 'COMPLETED' or any(
                    row[n] != 'True' for n in ('symbol_roundtrip_exact', 'saved_c1_reconstruction_exact')):
                raise ValueError('Invalid/duplicate A T=10 raw reference')
            b = selected[key[0]]
            raw = raw_bytes(10, hidden_dim=b.get('hidden_dim', 2560))
            if row['dataset'] != b['dataset'] or int(row['token_group_size']) != 10 or any(int(row[n]) != v for n, v in raw.items()):
                raise ValueError('A T=10 population/raw accounting mismatch')
            blocks[key] = dict(block_id=key[0], dataset=b['dataset'], mode=key[1], **raw,
                total_payload_bytes=int(row['encoded_payload_bytes']), local_metadata_bytes=int(row['local_metadata_bytes']))
    if len(blocks) != len(evaluation)*2:
        raise ValueError('A reference does not cover exactly the evaluation T=10 IDs')
    for b in evaluation:
        raw = raw_bytes(10, hidden_dim=b.get('hidden_dim', 2560))
        blocks[b['block_id'], 'UNIFORM_INT8'] = dict(block_id=b['block_id'], dataset=b['dataset'],
            mode='UNIFORM_INT8', **raw, total_payload_bytes=raw['raw_kv_bytes']//2, local_metadata_bytes=2560)
    for (dataset, mode), row in by_pool.items():
        group = [v for v in blocks.values() if v['mode'] == mode and (dataset == 'ALL' or v['dataset'] == dataset)]
        n = len(group)
        q = sum(r['raw_q_bytes'] for r in group)
        kv = sum(r['raw_kv_bytes'] for r in group)
        payload = sum(r['total_payload_bytes'] for r in group)
        local = sum(r['local_metadata_bytes'] for r in group)
        shared = 0 if mode == 'UNIFORM_INT8' else file_size(aroot/shared_harness.PROFILE_FILES[mode])
        expected = dict(block_count=n, raw_kv_pool_bytes=kv, raw_semcache_pool_bytes=q+kv,
            encoded_payload_pool_bytes=payload, local_metadata_pool_bytes=local, shared_profile_bytes=shared,
            encoded_kv_pool_bytes=payload+local+shared, encoded_semcache_pool_bytes=q+payload+local+shared)
        if any(int(row[k]) != v for k, v in expected.items()):
            raise ValueError('A T=10 summary bytes fail raw-row reconciliation')
    return dict(source_file=str(path), source_sha256=file_hash(path), source_rows=source_rows,
        raw_source_file=str(raw_path), raw_source_sha256=file_hash(raw_path), blocks=list(blocks.values()))


def file_size(path):
    return Path(path).stat().st_size


def bind(args):
    root, selected, populations, rec, contract, hashes = b1.bind_inputs(args)
    if populations != EXPECTED:
        raise ValueError('B2 requires frozen T=10 counts: calibration 241 (103/138), evaluation 236 (93/143)')
    calibration = [b for b in selected if b['partition'] == 'calibration']
    evaluation = [b for b in selected if b['partition'] == 'evaluation']
    bm = read_json(args.b1_dir/'manifest.json')
    if (bm['status'] != 'COMPLETED' or bm['diagnostic_result_eligible'] is not True or
            bm['config'] != b1.CONFIG or bm['config_sha256'] != digest(b1.CONFIG) or
            bm['populations'] != populations or bm['completed_blocks'] != len(selected) or
            bm['decision']['recommendation'] != 'GO_TO_B2'):
        raise ValueError('Matching completed B1 GO result required')
    for split, blocks in (('calibration', calibration), ('evaluation', evaluation)):
        if bm['selected_block_ids'][split] != [b['block_id'] for b in blocks]:
            raise ValueError('B1/C1 frozen split identity mismatch')
    device_contract.verify_contract(bm['device_contract'], contract)
    assert_unchanged(bm['input_file_sha256'])
    if bm['implementation_sha256'].get(str(Path(b1.__file__))) != file_hash(b1.__file__):
        raise ValueError('Frozen B1 transform source changed')
    for name in ('manifest.json', 'b1_entropy_summary.csv', 'b1_entropy_raw.csv',
                 'b1_locality_summary.csv', 'b1_roundtrip.json', 'environment.json'):
        p = args.b1_dir.resolve()/name
        hashes[str(p)] = file_hash(p)
    rt = read_json(args.b1_dir/'b1_roundtrip.json')
    if rt['status'] != 'COMPLETED' or rt['all_symbol_roundtrips_exact'] is not True or rt['all_scales_preserved'] is not True:
        raise ValueError('B1 exact roundtrip evidence required')
    references = reference_inputs(args.c15a_dir.resolve(), evaluation)
    runtime = environment()
    base = dict(config=CONFIG, config_sha256=digest(CONFIG), device_contract=contract,
        input_sha256=hashes, implementation_sha256=source_hashes(), populations=populations,
        calibration_block_ids=[b['block_id'] for b in calibration], evaluation_block_ids=[b['block_id'] for b in evaluation],
        calibration_evaluation_overlap=0, runtime=runtime,
        git_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], text=True, check=True, capture_output=True).stdout.strip(),
        C15A_reference_sha256=references['source_sha256'], B1_manifest_sha256=file_hash(args.b1_dir/'manifest.json'))
    return dict(root=root, calibration=calibration, evaluation=evaluation, rec=rec,
                contract=contract, base=base, references=references)


def quantized_fixture(root, block, contract):
    if block['token_group_size'] != 10:
        raise ValueError('B2 forbids T=3, padding and regrouping')
    qkv = load_fixture(root, block)
    encoded = tensors.quantize(qkv['k'], qkv['v'], device=contract['quantization_device_resolved'])
    domains, scales, shape = pack_encoded(encoded)
    return domains, scales, shape, encoded


def pack_encoded(encoded):
    import torch
    shape = tuple(encoded[0][0].shape)
    fmt.validate_shape(shape)
    if len(encoded) != 2 or any(tuple(s.shape) != shape or s.dtype != torch.int8 or
            c.dtype != torch.float32 or tuple(c.shape) != (shape[0], 10, 1) for s, c in encoded):
        raise ValueError('Unchanged C1 int8 symbols/FP32 scale shapes required')
    domains = tuple(bytes((s.to(torch.int16)+127).to(torch.uint8).flatten().tolist()) for s, _ in encoded)
    scales = struct.pack('<'+'f'*(2*shape[0]*10), *[v for _, c in encoded for v in c.flatten().tolist()])
    return domains, scales, shape


def restored_encoded(domains, scales, shape):
    import torch
    values = struct.unpack('<'+'f'*(2*shape[0]*10), scales)
    cs = torch.tensor(values, dtype=torch.float32).reshape(2, shape[0], 10, 1)
    return tuple(((torch.frombuffer(bytearray(data), dtype=torch.uint8).to(torch.int16)-127).to(torch.int8).reshape(shape), cs[i])
                 for i, data in enumerate(domains))


def fit_profiles(args, out):
    path = out/'b2_profile_manifest.json'
    if path.exists() or (out/'profiles').exists() or (out/'manifest.json').exists():
        raise ValueError('B2 profiles already initialized; frozen profiles cannot be refit/overwritten')
    state = dict(stage='C1.5-B2', profile_status='INCOMPLETE', primary_result_eligible=False, config=CONFIG)
    atomic_json(out/'manifest.json', state)
    try:
        ctx = bind(args)
        base = ctx['base']
        def loader(b):
            if b['partition'] != 'calibration':
                raise ValueError('Evaluation leakage into B2 profile fitting')
            return quantized_fixture(ctx['root'], b, ctx['contract'])[:3]
        profiles, counts, fitted = fmt.fit(ctx['calibration']+ctx['evaluation'], loader)
        if fitted != base['calibration_block_ids'] or set(fitted) & set(base['evaluation_block_ids']):
            raise ValueError('Calibration-only fit coverage failed')
        assert_unchanged(base['input_sha256'])
        assert_unchanged(base['implementation_sha256'])
        device_contract.verify_contract(ctx['contract'], device_contract.resolve_contract(ctx['root']))
        items = {}
        for mode, profile in profiles.items():
            relative = 'profiles/'+fmt.FILES[mode]
            atomic_bytes(out/relative, profile.to_bytes())
            items[mode] = dict(file=relative, sha256=file_hash(out/relative), serialized_bytes=file_size(out/relative),
                distribution_count=4, calibration_counts=counts[mode], counts_sha256=digest(counts[mode]), **fmt.labels(mode))
        pm = dict(status='FROZEN_BEFORE_EVALUATION', base_contract=base, base_contract_sha256=digest(base),
            profiles=items, fitted_block_ids=fitted, fitting_partition='calibration',
            C15A_references=ctx['references'], decision_rule=RULE, primary_result_eligible=False,
            no_new_lossy_quantization=True, official_cachegen_reproduction=False)
        atomic_json(path, pm)
        state.update(profile_status='FROZEN', profile_manifest_sha256=file_hash(path),
            smoke_status='NOT_RUN', full_storage_status='NOT_RUN')
        atomic_json(out/'manifest.json', state)
    except BaseException as exc:
        state.update(profile_status='INCOMPLETE' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'FAILED', error=str(exc))
        atomic_json(out/'manifest.json', state)
        raise


def load_profiles(out, ctx):
    state = read_json(out/'manifest.json')
    path = out/'b2_profile_manifest.json'
    if state['profile_status'] != 'FROZEN' or file_hash(path) != state['profile_manifest_sha256']:
        raise ValueError('B2 profile manifest not frozen/hash mismatch')
    pm = read_json(path)
    if (pm['status'] != 'FROZEN_BEFORE_EVALUATION' or pm['base_contract'] != ctx['base'] or
            pm['base_contract_sha256'] != digest(ctx['base']) or pm['decision_rule'] != RULE or
            pm['fitted_block_ids'] != ctx['base']['calibration_block_ids'] or pm['fitting_partition'] != 'calibration' or
            pm['C15A_references'] != ctx['references'] or set(pm['profiles']) != set(fmt.MODES)):
        raise ValueError('Incompatible frozen B2 profiles/input/config/source/runtime contract')
    profiles = {}
    for mode in fmt.MODES:
        item = pm['profiles'][mode]
        relative = 'profiles/'+fmt.FILES[mode]
        if item['file'] != relative or item['distribution_count'] != 4 or any(item[k] != v for k, v in fmt.labels(mode).items()):
            raise ValueError('Profile mode/granularity/provenance mismatch')
        model = fmt.Profile.from_bytes((out/relative).read_bytes(), item['sha256'])
        expected = fmt.Profile(mode, tuple(core.cdf_from_counts(c) for c in item['calibration_counts']))
        if model != expected or item['counts_sha256'] != digest(item['calibration_counts']) or item['serialized_bytes'] != file_size(out/relative):
            raise ValueError('Profile differs from frozen calibration counts/bytes')
        profiles[mode] = model
    if len({len(p.to_bytes()) for p in profiles.values()}) != 1:
        raise ValueError('Unequal B2 profile granularity')
    return profiles, pm


def evaluate_pair(ctx, block, profile, path):
    """Single encode/decode, from an actual persisted file, per block/mode."""
    import torch
    domains, scale_bytes, shape, original = quantized_fixture(ctx['root'], block, ctx['contract'])
    streams = fmt.streams_from_domains(profile.mode, domains, shape)
    blob = fmt.encode(profile, streams, scale_bytes, shape)
    atomic_bytes(path, blob)
    disk = path.read_bytes()
    if file_size(path) != len(disk) or disk != blob:
        raise ValueError('Produced bitstream file differs from encoder bytes')
    decoded, decoded_scales, decoded_shape = fmt.decode(disk, profile)
    if decoded != streams or decoded_shape != shape:
        raise ValueError('Arithmetic symbol roundtrip mismatch')
    inverse = fmt.domains_from_streams(profile.mode, decoded, shape)
    if inverse != domains:
        raise ValueError('B1 residual inverse changed original C1 symbols')
    if decoded_scales != scale_bytes:
        raise ValueError('Scale bytes changed')
    recovered = restored_encoded(inverse, decoded_scales, shape)
    for (s, c), (rs, rc) in zip(original, recovered):
        if not torch.equal(s, rs) or not torch.equal(c, rc):
            raise ValueError('Original C1 symbols/scales changed')
    actual = tensors.reconstruct(recovered, device=ctx['contract']['reconstruction_device'])
    shared_harness.existing_uniform(ctx['root'], ctx['rec'], block, actual)
    sizes = fmt.inspect(disk, profile)[3]
    if sizes['total_payload_bytes']+sizes['local_metadata_bytes'] != file_size(path):
        raise ValueError('Actual bitstream size accounting mismatch')
    return dict(block_id=block['block_id'], dataset=block['dataset'], query_id=block.get('query_id'),
        token_group_size=10, **raw_bytes(10, hidden_dim=shape[2]), **sizes, mode=profile.mode,
        profile_bytes_reference=len(profile.to_bytes()), symbol_count=sum(fmt.stream_counts(shape)),
        **{k: True for k in CORRECTNESS}, **fmt.labels(profile.mode), status='COMPLETED', Q_storage='FP16',
        bitstream_sha256=sha256(disk).hexdigest(), actual_bitstream_bytes=file_size(path))


def load_completed(out, run_hash, blocks, profiles):
    expected = {(b['block_id'], m): b for b in blocks for m in fmt.MODES}
    rows = {}
    for path in sorted((out/'checkpoints').glob('*.json')):
        record = read_json(path)
        row = record['row']
        key = row['block_id'], row['mode']
        if (key not in expected or key in rows or path.stem != pair_name(*key) or
                record['run_contract_sha256'] != run_hash or record['row_sha256'] != digest(row)):
            raise ValueError('Incompatible/duplicate B2 checkpoint')
        b, profile = expected[key], profiles[key[1]]
        relative = f'bitstreams/{key[1]}/{pair_name(*key)}.bin'
        if (row['status'] != 'COMPLETED' or any(row[k] is not True for k in CORRECTNESS) or
                any(row[k] != v for k, v in fmt.labels(key[1]).items()) or
                row['dataset'] != b['dataset'] or row['query_id'] != b.get('query_id') or row['token_group_size'] != 10 or
                row['Q_storage'] != 'FP16' or row['profile_bytes_reference'] != len(profile.to_bytes()) or
                row['bitstream_file'] != relative or file_hash(out/relative) != row['bitstream_sha256']):
            raise ValueError('Invalid B2 checkpoint correctness/provenance/bitstream')
        shape, _, _, sizes = fmt.inspect((out/relative).read_bytes(), profile)
        if (shape != (32, 10, b.get('hidden_dim', 2560)) or row['symbol_count'] != sum(fmt.stream_counts(shape)) or
                any(row[k] != v for k, v in sizes.items()) or file_size(out/relative) != row['actual_bitstream_bytes'] or
                any(row[k] != v for k, v in raw_bytes(10, hidden_dim=shape[2]).items())):
            raise ValueError('B2 checkpoint byte accounting/shape mismatch')
        rows[key] = row
    return rows


def accounting(rows, shared):
    q = sum(r['raw_q_bytes'] for r in rows)
    kv = sum(r['raw_kv_bytes'] for r in rows)
    payload = sum(r['total_payload_bytes'] for r in rows)
    local = sum(r['local_metadata_bytes'] for r in rows)
    symbols = kv//2
    stored = payload+local+shared
    return dict(block_count=len(rows), raw_kv_pool_bytes=kv, raw_semcache_pool_bytes=q+kv,
        total_payload_bytes=payload, local_metadata_bytes=local, shared_profile_bytes=shared,
        encoded_kv_pool_bytes=stored, encoded_semcache_pool_bytes=q+stored,
        pool_kv_compression_ratio=kv/stored, pool_semcache_compression_ratio=(q+kv)/(q+stored),
        payload_only_bits_per_symbol=8*payload/symbols,
        profile_amortized_bits_per_symbol=8*(payload+shared)/symbols,
        all_stored_KV_bits_per_symbol=8*stored/symbols)


def summaries(rows, references, profiles, *, smoke):
    result = []
    ids = {r['block_id'] for r in rows}
    for dataset in ('ALL', 'snips', 'multiwoz'):
        reference_modes = ('UNIFORM_INT8', *core.MODES)
        pools = {}
        for mode in reference_modes:
            group = [r for r in references['blocks'] if r['block_id'] in ids and r['mode'] == mode and
                     (dataset == 'ALL' or r['dataset'] == dataset)]
            source = next(r for r in references['source_rows'] if r['compression_mode'] == mode and
                          r['stratum'] == {'ALL': 'T=10', 'snips': 'SNIPS/T10', 'multiwoz': 'MultiWOZ/T10'}[dataset])
            pools[mode] = accounting(group, int(source['shared_profile_bytes']))
        for mode in fmt.MODES:
            group = [r for r in rows if r['mode'] == mode and (dataset == 'ALL' or r['dataset'] == dataset)]
            pools[mode] = accounting(group, len(profiles[mode].to_bytes()))
        for mode in (*fmt.MODES, *reference_modes):
            account = pools[mode]
            group = [r for r in rows if r['mode'] == mode and (dataset == 'ALL' or r['dataset'] == dataset)]
            is_b2 = mode in fmt.MODES
            def saving(control):
                return 100*(1-account['encoded_kv_pool_bytes']/pools[control]['encoded_kv_pool_bytes'])
            result.append(dict(dataset=dataset, token_group_size=10, mode=mode, **account,
                k_payload_bytes=sum(r['k_anchor_payload_bytes']+r['k_nonanchor_or_residual_payload_bytes'] for r in group) if is_b2 else None,
                v_payload_bytes=sum(r['v_anchor_payload_bytes']+r['v_nonanchor_or_residual_payload_bytes'] for r in group) if is_b2 else None,
                scale_metadata_bytes=sum(r['scale_metadata_bytes'] for r in group) if is_b2 else None,
                local_transform_metadata_bytes=sum(r['local_transform_metadata_bytes'] for r in group) if is_b2 else None,
                storage_reduction_vs_RAW_ROLE_SPLIT_percent=saving(fmt.MODES[0]),
                storage_reduction_vs_primary_KV_residual_percent=saving(fmt.MODES[1]),
                storage_reduction_vs_C15A_T10_GLOBAL_percent=saving(core.MODES[0]),
                storage_reduction_vs_C15A_T10_LAYERGROUP_percent=saving(core.MODES[1]),
                result_classification=fmt.CLASSIFICATIONS[mode] if is_b2 else 'C15A_REFERENCE',
                provenance='MEASURED_RESEARCH_EXTENSION' if is_b2 else 'EXISTING_C15A_T10_REFERENCE',
                predeclaration=fmt.labels(mode)['predeclaration'] if is_b2 else 'FROZEN_REFERENCE',
                primary_result_eligible=not smoke and mode == fmt.MODES[1],
                run_classification='DIAGNOSTIC_ONLY' if smoke else 'FULL_STORAGE',
                profile_accounting='one complete profile per independent pool; dataset profile charges are not additive',
                profile_amortized_bits_definition='8*(payload+shared profile)/symbols; scale/local framing excluded',
                source_reference_sha256=references['source_sha256']))
    return result


def primary_decision(summary):
    rows = {r['dataset']: r for r in summary if r['mode'] == fmt.MODES[1]}
    gain = rows['ALL']['storage_reduction_vs_RAW_ROLE_SPLIT_percent']
    positive = {d: rows[d]['storage_reduction_vs_RAW_ROLE_SPLIT_percent'] > 0 for d in ('snips', 'multiwoz')}
    return dict(rule=RULE, status='B2_PRIMARY_SUCCESS' if gain >= 3.0 and gain > 0 else 'B2_PRIMARY_NO_SUCCESS',
        relative_KV_byte_reduction_percent=gain, dataset_positive_reductions=positive,
        both_datasets_positive=all(positive.values()), hybrid_variant='POST_HOC_EXPLORATORY',
        primary_mode=fmt.MODES[1])


def smoke_selection(blocks):
    selected, seen = [], set()
    for b in blocks:
        if b['partition'] != 'evaluation' or b['token_group_size'] != 10:
            raise ValueError('B2 selection must contain only evaluation T=10 blocks')
        if b['dataset'] not in seen:
            selected.append(b)
            seen.add(b['dataset'])
    if seen != {'snips', 'multiwoz'}:
        raise ValueError('Smoke requires both datasets')
    return selected


def evaluate(args, parent):
    smoke = args.command == 'smoke'
    out = parent/('smoke' if smoke else 'full_storage')
    out.mkdir(exist_ok=True)
    started = time.monotonic()
    state, rows, active = None, {}, None
    def publish(status, **fields):
        state.update(status=status, completed_block_mode_pairs=len(rows),
            primary_result_eligible=status == 'COMPLETED' and not smoke, **fields)
        if status != 'COMPLETED':
            (out/'b2_summary.csv').unlink(missing_ok=True)
        if rows:
            atomic_csv(out/'b2_block_raw.csv', list(rows.values()))
        atomic_json(out/'progress.json', state)
        atomic_json(out/'manifest.json', state)  # Authoritative completion marker written last.
        top_path = parent/'manifest.json'
        if top_path.exists():
            top = read_json(top_path)
            top['smoke_status' if smoke else 'full_storage_status'] = status
            if not smoke or status == 'FAILED':
                top['primary_result_eligible'] = status == 'COMPLETED' and not smoke
            atomic_json(top_path, top)
    try:
        if not (out/'manifest.json').exists():
            state = dict(status='INCOMPLETE', primary_result_eligible=False, initialization_pending=True,
                expected_block_mode_pairs=SMOKE_PAIRS if smoke else FULL_PAIRS, run_contract_sha256=None,
                failed_pairs=0, config=CONFIG, decision_rule=RULE, decision=None,
                run_classification='DIAGNOSTIC_ONLY' if smoke else 'PRIMARY_B2_FULL_STORAGE')
            publish('INCOMPLETE')
        ctx = bind(args)
        profiles, pm = load_profiles(parent, ctx)
        evaluation = ctx['evaluation']
        blocks = smoke_selection(evaluation) if smoke else evaluation
        expected_pairs = len(blocks)*3
        if expected_pairs != (SMOKE_PAIRS if smoke else FULL_PAIRS):
            raise ValueError('B2 expected coverage must be 6 smoke or 708 full pairs')
        run_contract = dict(base=ctx['base'], profile_manifest_sha256=file_hash(parent/'b2_profile_manifest.json'),
            profiles={m: p.sha256 for m, p in profiles.items()}, selected_block_ids=[b['block_id'] for b in blocks],
            modes=list(fmt.MODES), smoke=smoke, timing=TIMING)
        run_hash = digest(run_contract)
        old = read_json(out/'manifest.json')
        initializing = old.get('initialization_pending') is True and old.get('run_contract_sha256') is None
        if initializing and any((out/'checkpoints').glob('*.json')):
            raise ValueError('Orphan B2 checkpoints without bound contract')
        if not initializing and old['run_contract_sha256'] != run_hash:
            raise ValueError('Incompatible B2 resume contract')
        if old['status'] == 'FAILED':
            raise ValueError('FAILED B2 run is sealed; preserve/archive it before replacement')
        if not smoke:
            # Verify completed smoke pairs without another arithmetic decode or encode.
            sm = read_json(parent/'smoke/manifest.json')
            if (sm['status'] != 'COMPLETED' or sm['completed_block_mode_pairs'] != SMOKE_PAIRS or sm['failed_pairs'] != 0 or
                    sm['primary_result_eligible'] is not False or sm['run_contract']['base'] != ctx['base'] or
                    sm['run_contract']['profile_manifest_sha256'] != run_contract['profile_manifest_sha256']):
                raise ValueError('Matching exact completed smoke required before full storage')
            smoke_blocks = smoke_selection(evaluation)
            if (sm['run_contract']['selected_block_ids'] != [b['block_id'] for b in smoke_blocks] or
                    digest(sm['run_contract']) != sm['run_contract_sha256'] or
                    len(load_completed(parent/'smoke', sm['run_contract_sha256'], smoke_blocks, profiles)) != 6):
                raise ValueError('Smoke integrity/coverage mismatch')
        state = dict(initialization_pending=False, run_contract=run_contract, run_contract_sha256=run_hash,
            expected_block_mode_pairs=expected_pairs, evaluation_block_count=len(blocks), expected_block_count=len(blocks),
            failed_pairs=0, decision=None, config=CONFIG, decision_rule=RULE,
            run_classification='DIAGNOSTIC_ONLY' if smoke else 'PRIMARY_B2_FULL_STORAGE')
        verify_tensor_files(ctx['root'], blocks, ctx['rec'])
        rows = load_completed(out, run_hash, blocks, profiles)
        rows = {(b['block_id'], m): rows[b['block_id'], m] for b in blocks for m in fmt.MODES if (b['block_id'], m) in rows}
        publish('INCOMPLETE')
        atomic_json(out/'environment.json', ctx['base']['runtime'])
        for block in blocks:
            for mode in fmt.MODES:
                active = block['block_id'], mode
                if active in rows:
                    continue
                relative = f'bitstreams/{mode}/{pair_name(*active)}.bin'
                row = evaluate_pair(ctx, block, profiles[mode], out/relative)
                row.update(bitstream_file=relative, run_classification=state['run_classification'])
                # Check all correctness/provenance/accounting gates before transaction commit.
                if any(row[k] is not True for k in CORRECTNESS) or any(row[k] != v for k, v in fmt.labels(mode).items()):
                    raise ValueError('B2 exactness/provenance gate failed')
                atomic_json(out/'checkpoints'/(pair_name(*active)+'.json'),
                    dict(run_contract_sha256=run_hash, row_sha256=digest(row), row=row))
                rows[active] = row
                publish('INCOMPLETE')
                print(f'B2 {len(rows)}/{expected_pairs}: {block["block_id"]} {mode}', flush=True)
        active = None
        assert_unchanged(ctx['base']['input_sha256'])
        assert_unchanged(ctx['base']['implementation_sha256'])
        load_profiles(parent, ctx)
        device_contract.verify_contract(ctx['contract'], device_contract.resolve_contract(ctx['root']))
        verify_tensor_files(ctx['root'], blocks, ctx['rec'])
        checked = load_completed(out, run_hash, blocks, profiles)
        if len(checked) != expected_pairs or set(checked) != {(b['block_id'], m) for b in blocks for m in fmt.MODES}:
            raise ValueError('Incomplete B2 pair coverage')
        rows = {(b['block_id'], m): checked[b['block_id'], m] for b in blocks for m in fmt.MODES}
        summary = summaries(list(rows.values()), ctx['references'], profiles, smoke=smoke)
        if not smoke:
            state['decision'] = primary_decision(summary)
        atomic_csv(out/'b2_summary.csv', summary)
        publish('COMPLETED', all_correctness_checks_exact=True)
    except BaseException as exc:
        if state is not None:
            interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
            publish('INCOMPLETE' if interrupted else 'FAILED', decision=None,
                    failed_pairs=0 if interrupted else int(active is not None), error=f'{type(exc).__name__}: {exc}')
            if not interrupted:
                with (out/'failures.jsonl').open('a') as f:
                    f.write(json.dumps(dict(pair=active, error=str(exc)))+'\n')
                    f.flush()
        raise
    finally:
        if state is not None:
            atomic_json(out/'run_diagnostics.json', dict(**TIMING,
                invocation_wall_seconds=time.monotonic()-started, status=state['status'], primary_latency_result=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('profile-fit', 'smoke', 'storage-full'):
        p = sub.add_parser(command)
        p.add_argument('--capture-manifest', type=Path, default=Path('results/cachegen/c1/capture_manifest.json'))
        p.add_argument('--c15a-dir', type=Path, default=Path('results/cachegen/c1_5'))
        p.add_argument('--b1-dir', type=Path, default=Path('results/cachegen/c1_5b/b1'))
        p.add_argument('--output-dir', type=Path, default=Path('results/cachegen/c1_5b/b2'))
    args = parser.parse_args(argv)
    out = safe_output(args)
    with exclusive_run(out):
        if args.command == 'profile-fit':
            fit_profiles(args, out)
        else:
            evaluate(args, out)
