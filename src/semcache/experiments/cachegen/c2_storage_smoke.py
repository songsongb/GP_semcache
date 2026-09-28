"""C2-STORAGE-SMOKE: bounded captured-QKV exercise of the actual GlobalCache."""
from collections import Counter
from pathlib import Path
import statistics

from semcache.cache.admission import AdmissionPolicy
from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from .c2.physical_storage import (FrozenK20V16Codec, MODE_COMPRESSED, POLICY,
                                  PROFILE_SHA256)
from .common import file_hash, load_fixture, percentile, read_json, write_csv, write_json
from .c15c.harness import ROOT, git_state
from .c15c.holdout import evaluation_blocks
from .shared.device_contract import resolve_contract

OUTPUT = ROOT/'results/cachegen/c2/storage_smoke'


def select_smoke_blocks(capture, per_dataset=4):
    """Four sorted evaluation w=3 blocks from each C1 dataset, by default."""
    if type(per_dataset) is not int or not 1 <= per_dataset <= 4:
        raise ValueError('C2 storage smoke is capped at four blocks per dataset')
    selected, counts = evaluation_blocks(capture)
    by_dataset = {name: [b for b in selected if b['dataset'] == name]
                  for name in ('multiwoz', 'snips')}
    if any(len(pool) < per_dataset for pool in by_dataset.values()):
        raise ValueError('Insufficient existing evaluation w=3 blocks in SNIPS/MultiWOZ')
    chosen = [b for name in ('multiwoz', 'snips') for b in by_dataset[name][:per_dataset]]
    return chosen, counts


def _timing_summary(rows, key):
    values = [r[key] for r in rows]
    if any(v is None for v in values):
        raise ValueError(f'Missing C2 timing: {key}')
    return dict(mean=statistics.mean(values), p50=percentile(values, .5), p95=percentile(values, .95))


def smoke(args):
    import torch
    capture_path = args.capture_manifest.resolve()
    capture = read_json(capture_path)
    blocks, counts = select_smoke_blocks(capture, args.per_dataset)
    root = capture_path.parent
    contract = resolve_contract(root)
    device = contract['quantization_device_resolved']
    if not device.startswith('cuda:'):
        raise ValueError('Real C2 smoke requires recorded C1 CUDA quantization device')
    profile_path = args.profile_path.resolve()
    codec = FrozenK20V16Codec(profile_path, quantization_device=device,
                              decode_device=device, instrument=True)
    cache = GlobalCache(64*1024*1024, admission=AdmissionPolicy(threshold=0),
        physical_storage_mode=MODE_COMPRESSED, physical_codec=codec, instrument_storage=True)
    if cache.physical_codec.profile.sha256 != PROFILE_SHA256:
        raise ValueError('Frozen C2 profile hash mismatch')
    out = args.output_dir.resolve()
    if not out.is_relative_to(OUTPUT.resolve()):
        raise ValueError('C2 smoke output must be under results/cachegen/c2/storage_smoke')
    if out.is_relative_to(root) or out.is_relative_to(profile_path.parent):
        raise ValueError('C2 output overlaps frozen inputs')
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(stage='C2-STORAGE-SMOKE', status='RUNNING', gp_semcache_git=git_state(),
        capture_manifest_path=str(capture_path), capture_manifest_sha256=file_hash(capture_path),
        selected_block_ids=[b['block_id'] for b in blocks],
        dataset_counts=dict(Counter(b['dataset'] for b in blocks)),
        selected_block_count=len(blocks), required_window_size=3, capture_counts=counts,
        codec=MODE_COMPRESSED, policy=POLICY.name, storage_mode='B2_ANCHOR_MOD_RESIDUAL_KV',
        profile_path=str(profile_path), profile_sha256=PROFILE_SHA256,
        profile_bytes_once=cache.physical_codec.profile_bytes,
        q_source='real C1 captured Q; no synthetic Q used', q_storage_device='cpu',
        compressed_payload_storage='CPU bytes', metadata_storage='inside CPU B2 frame',
        decode_device=device, dequantization_device=device, device_contract=contract,
        model=capture['model_config'], model_metadata=capture.get('model_metadata'),
        seed=args.seed, selection='first sorted block IDs per dataset; seed recorded only',
        no_model_inference=True, no_cdf_fitting=True, completed_blocks=0)
    write_json(out/'manifest.json', manifest)
    rows = []
    try:
        for index, block in enumerate(blocks):
            fixture = load_fixture(root, block)
            tensors = {layer: tuple(fixture[name][layer:layer+1] for name in 'qkv') for layer in range(32)}
            position = (block['start_position'], block['start_position']+3)
            key_tokens = tuple(block['token_ids'])
            raw_size = sum(t.numel()*t.element_size() for qkv in tensors.values() for t in qkv)
            candidate = CacheEntry(index, key_tokens, position, raw_size,
                qkv_metadata=dict(component_scope='total_qkv', source_block_id=block['block_id']))
            miss_before = cache.lookup(candidate.key, record_reuse=False) is None
            def materialize():
                physical = cache.make_entry(index, key_tokens, position, tensors, 'cpu')
                physical.qkv_metadata = candidate.qkv_metadata
                return physical
            admitted = cache.insert(candidate, materialize=materialize)
            resident = cache.entries.get(candidate.key)
            hit = cache.lookup(candidate.key)
            hit_correct = admitted and hit is not None and hit.key == candidate.key
            if not miss_before or not hit_correct or resident is None or resident.tensors is not None:
                raise ValueError('C2 logical hit/miss or raw K/V retention invariant failed')
            direct = {role: POLICY.quantize(fixture[role.lower()].to(device), role)
                      for role in ('K', 'V')}
            symbol_mismatch = sum(int((hit.kv_symbols[i] != direct[role].symbols).sum().item())
                                  for i, role in enumerate(('K', 'V')))
            reconstructed = {role: torch.cat([hit.tensors[layer]['qkv'.index(role.lower())]
                                             for layer in range(32)], dim=0)
                             for role in ('K', 'V')}
            reconstruction_failures = sum(not torch.equal(reconstructed[role], direct[role].reconstructed)
                                          for role in ('K', 'V'))
            q_equal = all(torch.equal(hit.tensors[layer][0].cpu(), fixture['q'][layer:layer+1])
                          for layer in range(32))
            if symbol_mismatch or reconstruction_failures or not q_equal:
                raise ValueError('C2 symbol, reconstruction or Q equality invariant failed')
            insert_timing = cache.storage_timing_records[-2]
            lookup_timing = cache.storage_timing_records[-1]
            account = resident.storage_accounting
            rows.append(dict(block_id=block['block_id'], dataset=block['dataset'],
                raw_kv_bytes=account['raw_kv_bytes'], compressed_kv_entry_bytes=account['compressed_kv_entry_bytes'],
                encoded_kv_payload_bytes=account['encoded_kv_payload_bytes'],
                kv_metadata_bytes=account['kv_metadata_bytes'],
                raw_q_bytes=account['raw_q_bytes'], raw_qkv_bytes=account['raw_qkv_bytes'],
                actual_qkv_entry_bytes=account['actual_qkv_entry_bytes'],
                kv_compression_ratio=account['kv_compression_ratio'],
                overall_entry_compression_ratio=account['overall_entry_compression_ratio'],
                quantize_ms=insert_timing['quantize_ms'], encode_ms=insert_timing['encode_ms'],
                insert_ms=insert_timing['insert_ms'], decode_ms=lookup_timing['decode_ms'],
                dequantize_ms=lookup_timing['dequantize_ms'], lookup_ms=lookup_timing['lookup_ms'],
                symbol_mismatches=symbol_mismatch, reconstruction_equality_failures=reconstruction_failures,
                hit_miss_mismatches=int(not miss_before or not hit_correct)))
            manifest['completed_blocks'] = index+1
            write_json(out/'manifest.json', manifest)
            print(f'[{index+1}/{len(blocks)}] {block["dataset"]} {block["block_id"]}', flush=True)
            del fixture, tensors, direct, reconstructed, hit
        raw_q = sum(r['raw_q_bytes'] for r in rows)
        raw_kv = sum(r['raw_kv_bytes'] for r in rows)
        encoded_kv = sum(r['compressed_kv_entry_bytes'] for r in rows)
        payload_bytes = sum(r['encoded_kv_payload_bytes'] for r in rows)
        metadata_bytes = sum(r['kv_metadata_bytes'] for r in rows)
        stored_qkv = sum(r['actual_qkv_entry_bytes'] for r in rows)
        summary = dict(block_count=len(rows), raw_kv_bytes=raw_kv, compressed_kv_entry_bytes=encoded_kv,
            encoded_kv_payload_bytes=payload_bytes, per_entry_kv_metadata_bytes_total=metadata_bytes,
            kv_compression_ratio=raw_kv/encoded_kv, raw_q_bytes=raw_q,
            total_raw_qkv_bytes=raw_q+raw_kv, total_stored_qkv_bytes=stored_qkv,
            overall_qkv_compression_ratio=(raw_q+raw_kv)/stored_qkv,
            frozen_global_profile_bytes_counted_once=cache.physical_codec.profile_bytes,
            physical_bytes_including_one_profile=stored_qkv+cache.physical_codec.profile_bytes,
            symbol_mismatches=sum(r['symbol_mismatches'] for r in rows),
            reconstruction_equality_failures=sum(r['reconstruction_equality_failures'] for r in rows),
            hit_miss_mismatches=sum(r['hit_miss_mismatches'] for r in rows),
            timing_ms={key: _timing_summary(rows, key) for key in (
                'quantize_ms', 'encode_ms', 'insert_ms', 'decode_ms', 'dequantize_ms', 'lookup_ms')})
        write_json(out/'storage_summary.json', summary)
        write_csv(out/'per_block.csv', rows, rows[0].keys())
        manifest.update(status='COMPLETED', output_sha256={name: file_hash(out/name)
            for name in ('storage_summary.json', 'per_block.csv')})
    except BaseException as exc:
        manifest.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(out/'manifest.json', manifest)
    return summary
