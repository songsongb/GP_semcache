"""Resumable, single-pass PRIMARY C1.5-A storage evaluation (no inference/timing loop)."""
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from collections import Counter
from contextlib import contextmanager

from ..common import digest, environment, file_hash, load_fixture, raw_bytes, read_json
from . import harness as h
from .core import MODES, decode, encode, inspect_block, pool_accounting
from .tensors import prepare, quantize, reconstruct, restore

EXPECTED_BLOCKS = 1308
EXPECTED_T = {3: 1072, 10: 236}
CLASSIFICATION = 'PRIMARY_C1_5A_FULL_STORAGE_EVALUATION'
PROVENANCE = 'MEASURED_RESEARCH_EXTENSION'
TIMING = dict(classification='SINGLE_PASS_OPERATIONAL_TIMING',
              latency_result='NOT_PRIMARY_LATENCY_RESULT', warmup=0,
              encode_passes_per_pair=1, decode_passes_per_pair=1,
              comparable_to_c1_benchmark_timings=False)


def atomic_bytes(path, data):
    """Durable file replacement; a pair JSON is the transaction commit marker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())


def atomic_csv(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())


def selection(blocks):
    ids = h.partition_ids(blocks)
    chosen = [b for b in blocks if b['partition'] == 'evaluation']
    if (len(chosen) != EXPECTED_BLOCKS or
            Counter(b['token_group_size'] for b in chosen) != EXPECTED_T or
            {b['dataset'] for b in chosen} != {'snips', 'multiwoz'}):
        raise ValueError('Full storage requires all 1,308 frozen evaluation blocks: T3=1072, T10=236')
    if set(ids['calibration']) & set(ids['evaluation']):
        raise ValueError('Calibration/evaluation overlap')
    return chosen


def references(root, blocks):
    """Read measured C1 storage controls; never invoke their benchmarks."""
    by_id = {b['block_id']: b for b in blocks}
    result = {}
    with (root/'c1_block_raw.csv').open(newline='') as f:
        for source in csv.DictReader(f):
            key = source['block_id'], source['compression_mode']
            if key[0] not in by_id or key[1] not in ('FP16_RAW', 'UNIFORM_INT8'):
                continue
            b = by_id[key[0]]
            if key in result or source['status'] != 'MEASURED':
                raise ValueError('Duplicate/unmeasured C1 storage reference')
            raw = raw_bytes(b['token_group_size'])
            if (any(int(source[k]) != v for k, v in raw.items()) or
                    source['dataset'] != b['dataset'] or
                    int(source['token_group_size']) != b['token_group_size']):
                raise ValueError('C1 reference identity/raw byte mismatch')
            payload, metadata = (int(source[k]) for k in
                                 ('encoded_kv_payload_bytes', 'encoded_metadata_bytes'))
            expected = ((raw['raw_kv_bytes'], 0) if key[1] == 'FP16_RAW' else
                        (raw['raw_kv_bytes']//2, 2*32*b['token_group_size']*4))
            if ((payload, metadata) != expected or
                    int(source['encoded_kv_total_bytes']) != payload+metadata or
                    int(source['semcache_total_stored_bytes']) != raw['raw_q_bytes']+payload+metadata):
                raise ValueError('Invalid C1 reference byte accounting')
            result[key] = dict(block_id=key[0], compression_mode=key[1], dataset=b['dataset'],
                token_group_size=b['token_group_size'], **raw, encoded_payload_bytes=payload,
                local_metadata_bytes=metadata)
    if len(result) != len(blocks)*2:
        raise ValueError('Missing C1 storage references')
    return result


def verify_tensor_files(root, blocks, rec):
    """Also hash inputs of skipped pairs, not just newly evaluated fixtures."""
    entries = {}
    for item in rec['reconstructed']:
        if item['compression_mode'] == 'UNIFORM_INT8':
            if item['block_id'] in entries:
                raise ValueError('Duplicate saved C1 reconstruction')
            entries[item['block_id']] = item
    for b in blocks:
        if b['block_id'] not in entries:
            raise ValueError('Missing saved C1 reconstruction')
        for item in (b, entries[b['block_id']]):
            path = root/item['file']
            if not h.contained(path, root) or file_hash(path) != item['sha256']:
                raise ValueError('C1 fixture/reconstruction SHA256 mismatch')


def evaluate_pair(root, rec, block, model, contract):
    """Exactly one entropy encode and decode; equality checks have no tolerances."""
    import torch
    qkv = load_fixture(root, block)
    if qkv['q'].dtype != torch.float16:
        raise ValueError('Q must remain FP16')
    quantized = quantize(qkv['k'], qkv['v'], device=contract['quantization_device_resolved'])
    blob = encode(model, *prepare(model, quantized))
    recovered = restore(*decode(blob, model), model)
    for (symbols, scales), (actual_symbols, actual_scales) in zip(quantized, recovered):
        if (symbols.dtype != torch.int8 or scales.dtype != torch.float32 or
                not torch.equal(symbols, actual_symbols) or not torch.equal(scales, actual_scales)):
            raise ValueError('Entropy symbol/scale roundtrip is not exact')
    actual = reconstruct(recovered, device=contract['reconstruction_device'])
    h.existing_uniform(root, rec, block, actual)
    sizes = inspect_block(blob, model)[3]
    if len(blob) != sizes['encoded_payload_bytes']+sizes['local_metadata_bytes']:
        raise ValueError('Produced bitstream byte accounting mismatch')
    return blob, dict(block_id=block['block_id'], dataset=block['dataset'],
        query_id=block.get('query_id'), token_group_size=block['token_group_size'],
        compression_mode=model.mode, **raw_bytes(block['token_group_size']), **sizes,
        shared_profile_bytes_reference=len(model.to_bytes()),
        symbol_count=sum(s.numel() for s, _ in quantized), symbol_roundtrip_exact=True,
        saved_c1_reconstruction_exact=True, q_dtype='float16', status='COMPLETED', provenance=PROVENANCE)


def pair_name(block_id, mode):
    return digest([block_id, mode])


def load_completed(out, run_hash, blocks, profiles):
    expected = {(b['block_id'], m): b for b in blocks for m in MODES}
    rows = {}
    for path in sorted((out/'checkpoints').glob('*.json')):
        record = read_json(path)
        row = record['row']
        key = row['block_id'], row['compression_mode']
        if (record['run_contract_sha256'] != run_hash or record['row_sha256'] != digest(row) or
                key not in expected or key in rows or path.stem != pair_name(*key)):
            raise ValueError('Incompatible/duplicate checkpoint')
        b, model = expected[key], profiles[key[1]]
        raw = raw_bytes(b['token_group_size'])
        if (row['status'] != 'COMPLETED' or row['symbol_roundtrip_exact'] is not True or
                row['saved_c1_reconstruction_exact'] is not True or row['q_dtype'] != 'float16' or
                row['provenance'] != PROVENANCE or row['dataset'] != b['dataset'] or
                row['token_group_size'] != b['token_group_size'] or
                row['query_id'] != b.get('query_id') or
                any(row[k] != v for k, v in raw.items()) or
                row['symbol_count'] != raw['raw_kv_bytes']//2 or
                row['shared_profile_bytes_reference'] != len(model.to_bytes())):
            raise ValueError('Invalid completed checkpoint correctness/accounting')
        relative = f'bitstreams/{key[1]}/{pair_name(*key)}.bin'
        if row['bitstream_file'] != relative or file_hash(out/relative) != row['bitstream_sha256']:
            raise ValueError('Checkpoint bitstream SHA256 mismatch')
        shape, _, _, sizes = inspect_block((out/relative).read_bytes(), model)
        if shape != (32, b['token_group_size'], b.get('hidden_dim', 2560)):
            raise ValueError('Checkpoint bitstream shape mismatch')
        if any(row[k] != v for k, v in sizes.items()):
            raise ValueError('Checkpoint bitstream accounting mismatch')
        rows[key] = row
    return rows


def summaries(rows, refs, profiles):
    """Each stratum is an independent pool, charged one complete shared profile."""
    all_rows = list(rows)+list(refs.values())
    selectors = [('ALL', lambda r: True)]
    for dataset in ('snips', 'multiwoz'):
        selectors.append(({'snips': 'SNIPS', 'multiwoz': 'MultiWOZ'}[dataset],
                          lambda r, d=dataset: r['dataset'] == d))
    for t in (3, 10):
        selectors.append((f'T={t}', lambda r, t=t: r['token_group_size'] == t))
    for dataset in ('snips', 'multiwoz'):
        for t in (3, 10):
            selectors.append((f'{"SNIPS" if dataset == "snips" else "MultiWOZ"}/T{t}',
                lambda r, d=dataset, t=t: r['dataset'] == d and r['token_group_size'] == t))
    result = []
    for name, select in selectors:
        uniform = pool_accounting([r for r in refs.values()
                                   if r['compression_mode'] == 'UNIFORM_INT8' and select(r)], 0)
        for mode in ('FP16_RAW', 'UNIFORM_INT8', *MODES):
            group = [r for r in all_rows if r['compression_mode'] == mode and select(r)]
            size = len(profiles[mode].to_bytes()) if mode in MODES else 0
            account = pool_accounting(group, size)
            result.append(dict(stratum=name, compression_mode=mode, **account,
                payload_only_ratio=account['payload_only_kv_compression_ratio'],
                additional_KV_saving_vs_UNIFORM_INT8_percent=100*(1-account['encoded_kv_pool_bytes']/uniform['encoded_kv_pool_bytes']),
                additional_total_saving_vs_UNIFORM_INT8_percent=100*(1-account['encoded_semcache_pool_bytes']/uniform['encoded_semcache_pool_bytes']),
                profile_accounting='one full profile per independent pool; strata profile charges are not additive',
                provenance=PROVENANCE if mode in MODES else 'EXISTING_C1_STORAGE_REFERENCE',
                primary_result_eligible=True))
    return result


@contextmanager
def exclusive_run(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another storage-full writer holds this output lock') from None
        previous = signal.getsignal(signal.SIGTERM)
        def interrupted(signum, frame):
            raise KeyboardInterrupt('SIGTERM: resume with the same storage-full command')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            yield
        finally:
            signal.signal(signal.SIGTERM, previous)


def storage_full(args):
    profile_out = args.output_dir.resolve()
    root = args.capture_manifest.resolve().parent
    out = h.protect_output(profile_out/'full_storage', root)
    # Guard the parent too: never follow a full_storage symlink into frozen outputs.
    h.protect_output(profile_out, root)
    if out != profile_out/'full_storage':
        raise ValueError('full_storage must not be a symlink')
    with exclusive_run(out):
        _run(args, out)


def _run(args, out):
    started = time.monotonic()
    rows, state, active_pair = {}, None, None
    def publish(status, **extra):
        if status != 'COMPLETED':
            (out/'c15_full_summary.csv').unlink(missing_ok=True)
        state.update(status=status, primary_result_eligible=status == 'COMPLETED',
                     completed_block_mode_pairs=len(rows), **extra)
        for name in ('progress.json', 'storage_manifest.json', 'manifest.json'):
            atomic_json(out/name, state)
        if rows:
            atomic_csv(out/'c15_full_block_raw.csv', list(rows.values()))
        else:
            (out/'c15_full_block_raw.csv').unlink(missing_ok=True)
    try:
        if not (out/'manifest.json').exists():
            state = dict(result_classification=CLASSIFICATION, provenance=PROVENANCE,
                expected_block_count=EXPECTED_BLOCKS, expected_block_mode_pairs=EXPECTED_BLOCKS*2,
                failed_pairs=0, initialization_pending=True, run_contract_sha256=None)
            publish('INCOMPLETE')
        with h.inputs(args) as (root, profile_out, capture, rec, ids, selected, hashes):
            blocks = selection(capture['blocks'])
            contract = h.resolve_contract(root)
            if not contract['quantization_device_resolved'].startswith('cuda:'):
                raise ValueError('PRIMARY full storage requires the C1 CUDA device contract')
            profiles, pm = h.load_profiles(profile_out, hashes, ids, contract)
            refs = references(root, blocks)
            verify_tensor_files(root, blocks, rec)
            source_hashes = h.implementation_hashes()
            # Bind orchestration and validation as well as the codec implementation.
            import inspect
            from .. import common, harness, codecs
            for name, obj in (('full_storage', inspect.getmodule(storage_full)),
                              ('shared_harness', h), ('c1_harness', harness),
                              ('common', common), ('codecs', codecs)):
                source_hashes[name] = file_hash(inspect.getfile(obj))
            git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
            run_contract = dict(schema_version=1, classification=CLASSIFICATION,
                c1_file_sha256=hashes, profile_manifest_sha256=file_hash(profile_out/'profile_manifest.json'),
                profile_hashes={m: profiles[m].sha256 for m in MODES}, device_contract=contract,
                implementation_sha256=source_hashes, git_commit=git, runtime=environment(),
                block_ids=[b['block_id'] for b in blocks], modes=list(MODES), timing=TIMING,
                expected_block_count=EXPECTED_BLOCKS, expected_T=EXPECTED_T, preserve_bitstreams=True)
            run_hash = digest(run_contract)
            manifest = out/'manifest.json'
            if manifest.exists():
                old = read_json(manifest)
                initializing = (old.get('initialization_pending') is True and
                                old.get('run_contract_sha256') is None and
                                not any(out.glob('checkpoints/*.json')))
                if not initializing and old['run_contract_sha256'] != run_hash:
                    raise ValueError('Incompatible checkpoint run contract; outputs cannot be mixed')
                if old['status'] == 'FAILED':
                    raise ValueError('FAILED run is sealed; investigate and archive full_storage before retrying')
            elif any(out.glob('checkpoints/*.json')):
                raise ValueError('Orphan checkpoints without a run contract')
            state = dict(initialization_pending=False, run_contract=run_contract, run_contract_sha256=run_hash,
                result_classification=CLASSIFICATION, provenance=PROVENANCE,
                evaluation_block_count=len(blocks), expected_block_count=EXPECTED_BLOCKS,
                expected_block_mode_pairs=EXPECTED_BLOCKS*2, failed_pairs=0,
                calibration_evaluation_overlap=0, **contract,
                quantization_device=contract['quantization_device_resolved'],
                profile_hashes=run_contract['profile_hashes'],
                c1_capture_manifest_sha256=hashes['capture_manifest.json'],
                c1_reconstruction_manifest_sha256=hashes['reconstruction_manifest.json'],
                baseline_source_sha256=source_hashes['c1_baseline'], git_commit=git)
            rows = load_completed(out, run_hash, blocks, profiles)
            # Preserve deterministic manifest block/mode order across resume.
            rows = {(b['block_id'], m): rows[b['block_id'], m] for b in blocks for m in MODES
                    if (b['block_id'], m) in rows}
            publish('INCOMPLETE')
            atomic_json(out/'environment.json', run_contract['runtime'])
            for block in blocks:
                for mode in MODES:
                    active_pair = (block['block_id'], mode)
                    if active_pair in rows:
                        continue
                    blob, row = evaluate_pair(root, rec, block, profiles[mode], contract)
                    name = pair_name(*active_pair)
                    relative = f'bitstreams/{mode}/{name}.bin'
                    atomic_bytes(out/relative, blob)
                    row.update(bitstream_file=relative, bitstream_sha256=hashlib.sha256(blob).hexdigest())
                    atomic_json(out/'checkpoints'/f'{name}.json', dict(run_contract_sha256=run_hash,
                                row_sha256=digest(row), row=row))
                    rows[active_pair] = row
                    publish('INCOMPLETE')
                    print(f'COMPLETED {len(rows)}/{EXPECTED_BLOCKS*2}: {block["block_id"]} {mode}', flush=True)
            active_pair = None
            h.verify_contract(contract, h.resolve_contract(root))
            h.load_profiles(profile_out, hashes, ids, contract)
            verify_tensor_files(root, blocks, rec)
            checked = load_completed(out, run_hash, blocks, profiles)
            if len(checked) != EXPECTED_BLOCKS*2 or set(checked) != set(rows):
                raise ValueError('Incomplete expected block/mode coverage')
            rows = {(b['block_id'], m): checked[b['block_id'], m] for b in blocks for m in MODES}
            summary = summaries(list(rows.values()), refs, profiles)
        # inputs() final immutable-C1 check must pass before publishing primary results.
        atomic_csv(out/'c15_full_summary.csv', summary)
        publish('COMPLETED', all_symbol_roundtrips_exact=True, all_saved_c1_reconstructions_exact=True)
    except BaseException as exc:
        if state is not None:
            interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
            publish('INCOMPLETE' if interrupted else 'FAILED',
                    failed_pairs=0 if interrupted else int(active_pair is not None),
                    exception=f'{type(exc).__name__}: {exc}')
            if not interrupted:
                with (out/'failures.jsonl').open('a') as f:
                    f.write(json.dumps(dict(pair=active_pair, error=str(exc), type=type(exc).__name__))+'\n')
                    f.flush()
                    os.fsync(f.fileno())
        raise
    finally:
        if state is not None:
            atomic_json(out/'run_diagnostics.json', dict(**TIMING,
                invocation_wall_seconds=time.monotonic()-started, status=state['status'],
                primary_latency_result=False))
