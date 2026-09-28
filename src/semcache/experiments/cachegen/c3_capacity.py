"""C3 model-free capacity replay with measured C1.5C holdout frame sizes.

No quantizer, entropy coder, tensor fixture, or model is called by this module.
The M9-B replay owns the logical policy; only GlobalCache's capacity charge varies.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

from semcache.cache.global_cache import GlobalCache
from semcache.experiments.cachegen.c15c.holdout import UNIFORM, evaluation_blocks
from semcache.experiments.cachegen.c15c.harness import git_state
from semcache.experiments.cachegen.common import file_hash, read_json, write_csv, write_json
from semcache.semantic.subsequence import SubsequenceExtractor
from semcache.simulation.multi_user import digest, read_workload, simulate

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT/'results/cachegen/c3/capacity_pressure'
DEFAULT_BUDGET_MIB = (64, 128, 256, 512)
RAW_Q_BYTES = 491_520
RAW_K_BYTES = RAW_V_BYTES = 491_520
RAW_ENTRY_BYTES = RAW_Q_BYTES + RAW_K_BYTES + RAW_V_BYTES
PROFILE_BYTES = 4_108
PROFILE_SHA256 = '8defa27a83e8ad16e9c7efa0f40e68c302aef90363e14a1451288a3bf433cb2c'
RAW = 'RAW_FP16_QKV'
COMPRESSED = 'COMPRESSED_KV_K20_V16'
MODES = (RAW, COMPRESSED)
MAPPING_MODE = 'EMPIRICAL_PHYSICAL_SIZE_SIMULATION'
EXPECTED_HOLDOUT_COUNT = 1072


def validate_budgets(mib):
    if not mib or any(type(v) is not int or v <= 0 for v in mib) or len(set(mib)) != len(mib):
        raise ValueError('Budgets must be unique positive integer MiB values')
    return tuple(v*1024**2 for v in mib)


def validate_size_index(capture, holdout_manifest, summary, *, expected_count=EXPECTED_HOLDOUT_COUNT):
    """Pair the writer's sorted block IDs with its in-order B2 frame-length list.

    holdout.evaluation_blocks sorts by block_id; holdout.evaluate iterates that
    list without reordering, appending each frame length to the overall group;
    rate_storage.physical_accounting zips those lengths in the same order.
    Per-dataset subsequences supply an independent pairing cross-check.
    """
    if holdout_manifest.get('status') != 'COMPLETED' or holdout_manifest.get('selected_policy') != UNIFORM:
        raise ValueError('Completed frozen UNIFORM_K20_V16 holdout required')
    blocks, counts = evaluation_blocks(capture)
    ids = [b['block_id'] for b in blocks]
    recorded = holdout_manifest.get('evaluation_block_ids')
    if (len(ids) != expected_count or len(set(ids)) != len(ids) or recorded != ids or
            counts['selected_holdout_count'] != expected_count or
            holdout_manifest.get('selected_holdout_count') != expected_count or
            holdout_manifest.get('required_window_size') != 3 or
            summary.get('evaluation_block_count') != expected_count or
            summary.get('required_window_size') != 3):
        raise ValueError('Holdout block identity, count, or w=3 ordering mismatch')
    overall = summary['overall'][UNIFORM]
    physical = overall['physical_storage']
    sizes = physical['per_block_bitstream_bytes']
    if (overall['block_count'] != expected_count or len(sizes) != expected_count or
            any(type(n) is not int or n <= 0 for n in sizes) or
            sum(sizes) != physical['bitstream_pool_bytes'] or
            physical['global_profile_bytes'] != PROFILE_BYTES or
            physical['total_physical_bytes'] != sum(sizes)+PROFILE_BYTES):
        raise ValueError('Incomplete or inconsistent measured B2 frame lengths')
    for dataset in sorted({b['dataset'] for b in blocks}):
        subset = [n for b, n in zip(blocks, sizes) if b['dataset'] == dataset]
        group = summary['datasets'][dataset][UNIFORM]
        if (group['block_count'] != len(subset) or
                group['physical_storage']['per_block_bitstream_bytes'] != subset):
            raise ValueError(f'Per-dataset holdout frame ordering mismatch: {dataset}')
    entries = [dict(block_id=b['block_id'], dataset=b['dataset'], token_group_size=3,
                    compressed_kv_frame_bytes=n, compressed_qkv_entry_bytes=RAW_Q_BYTES+n)
               for b, n in zip(blocks, sizes)]
    return dict(schema='c3_size_index_v1', mapping_mode=MAPPING_MODE, count=len(entries),
                pairing_contract='holdout.evaluation_blocks sorted block_id -> holdout.evaluate overall append order -> physical_accounting zip; per-dataset subsequences verified',
                entries=entries)


def load_measured_sizes(capture_path, holdout_dir, calibration_dir, *, expected_count=EXPECTED_HOLDOUT_COUNT):
    capture_path, holdout_dir, calibration_dir = Path(capture_path), Path(holdout_dir), Path(calibration_dir)
    manifest_path, summary_path = holdout_dir/'manifest.json', holdout_dir/'evaluation_summary.json'
    profile_path = calibration_dir/'profiles/matched_uniform_k20_v16.bin'
    capture, manifest, summary = (read_json(p) for p in (capture_path, manifest_path, summary_path))
    if manifest.get('capture_manifest_sha256') != file_hash(capture_path):
        raise ValueError('Holdout/capture manifest hash mismatch')
    if manifest.get('output_sha256', {}).get('evaluation_summary.json') != file_hash(summary_path):
        raise ValueError('Holdout summary hash mismatch')
    if not profile_path.is_file() or file_hash(profile_path) != PROFILE_SHA256 or profile_path.stat().st_size != PROFILE_BYTES:
        raise ValueError('Frozen matched-uniform CDF profile missing or SHA256/size mismatch')
    frozen = manifest.get('frozen_calibration', {}).get('profiles', {}).get(UNIFORM, {})
    if frozen.get('sha256') != PROFILE_SHA256 or frozen.get('bytes') != PROFILE_BYTES:
        raise ValueError('Holdout profile provenance mismatch')
    index = validate_size_index(capture, manifest, summary, expected_count=expected_count)
    index['source_sha256'] = {str(p.resolve()): file_hash(p) for p in
                              (capture_path, manifest_path, summary_path, profile_path)}
    index['frozen_profile_sha256'] = PROFILE_SHA256
    return index


def empirical_frame_bytes(dataset, key, size_index):
    """Stable, dataset-stratified key assignment, independent of request order/budget."""
    frames = [e['compressed_kv_frame_bytes'] for e in size_index['entries'] if e['dataset'] == dataset]
    if not frames:
        raise ValueError(f'No measured {dataset} w=3 frame sizes')
    identity = json.dumps((dataset, key), separators=(',', ':'), ensure_ascii=False)
    position = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big') % len(frames)
    return frames[position]


class EmpiricalSizeAssigner:
    """Memoize stable key charges across all budgets without touching cache policy."""
    def __init__(self, dataset, size_index):
        self.dataset = dataset
        self.frames = tuple(e['compressed_kv_frame_bytes'] for e in size_index['entries']
                            if e['dataset'] == dataset)
        if not self.frames:
            raise ValueError(f'No measured {dataset} w=3 frame sizes')
        self.assigned = {}

    def frame_bytes(self, key):
        if key not in self.assigned:
            identity = json.dumps((self.dataset, key), separators=(',', ':'), ensure_ascii=False)
            position = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big') % len(self.frames)
            self.assigned[key] = self.frames[position]
        return self.assigned[key]


def workload_facts(rows):
    extractor = SubsequenceExtractor(3)
    seen, access_counts, accesses, revisits = set(), Counter(), 0, 0
    h = hashlib.sha256()
    for row in rows:
        keys = [(row['cluster_id'], w.token_ids) for w in extractor.extract(row['token_ids'])]
        accesses += len(keys)
        # M9-B performs all lookups before admitting this query's windows. A
        # duplicate inside one query is repeated, but is not yet a revisit.
        revisits += sum(key in seen for key in keys)
        access_counts.update(keys)
        seen.update(keys)
        h.update(json.dumps(keys, separators=(',', ':')).encode()+b'\n')
    return dict(query_count=len(rows), access_count=accesses, unique_cache_keys=len(seen),
                reuse_events=accesses-len(seen), repeated_accesses=accesses-len(seen),
                keys_accessed_more_than_once=sum(count > 1 for count in access_counts.values()),
                total_revisit_events=revisits, max_accesses_for_one_key=max(access_counts.values(), default=0),
                reuse_opportunity=bool(revisits), query_order_hash=digest(rows),
                access_sequence_sha256=h.hexdigest())


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    p = (len(ordered)-1)*fraction
    lo = int(p)
    return ordered[lo]+(ordered[min(lo+1, len(ordered)-1)]-ordered[lo])*(p-lo)


def replay_cell(rows, dataset, budget, mode, size_index, *, seed=42, users=25):
    if mode not in MODES or type(budget) is not int or budget <= 0:
        raise ValueError('Invalid capacity experiment cell')
    if mode == COMPRESSED:
        assigner = EmpiricalSizeAssigner(dataset, size_index)
        charge = lambda e: RAW_Q_BYTES+assigner.frame_bytes(e.key)
        overhead = PROFILE_BYTES
    else:
        charge, overhead = lambda e: e.size_bytes, 0
    stats = dict(requests=0, accesses=0, hits=0, misses=0, admissions=0, evictions=0,
                 rejected_admissions=0, peak_entries=0, peak_bytes=overhead,
                 resident_entry_samples=0, resident_byte_samples=0)
    admitted_sizes = []
    sequence = hashlib.sha256()

    def observe(*, index, row, user, keys, hit_windows, admissions, rejections, evictions,
                admitted_keys, evicted_keys, cache):
        sequence.update(json.dumps(keys, separators=(',', ':')).encode()+b'\n')
        stats['requests'] += 1
        stats['accesses'] += len(keys)
        stats['hits'] += sum(hit_windows)
        stats['misses'] += len(keys)-sum(hit_windows)
        stats['admissions'] += admissions
        stats['evictions'] += evictions
        stats['rejected_admissions'] += rejections
        resident = len(cache.entries)
        charged = cache.charged_cache_bytes
        if charged > budget:
            raise AssertionError('Physical byte budget exceeded after eviction')
        stats['peak_entries'] = max(stats['peak_entries'], resident)
        stats['peak_bytes'] = max(stats['peak_bytes'], charged)
        stats['resident_entry_samples'] += resident
        stats['resident_byte_samples'] += charged
        observe.last_bytes = charged
        if (index+1) % 100 == 0:
            cache.assert_homogeneous_replay()
        if mode == COMPRESSED:
            admitted_sizes.extend(RAW_Q_BYTES+assigner.frame_bytes(key)
                                  for key in admitted_keys)

    result = simulate(rows, users, seed=seed, capacity=budget, dataset=dataset,
                      cache_factory=lambda b: GlobalCache(b, capacity_charge=charge,
                                                         shared_overhead_bytes=overhead,
                                                         homogeneous_logical_fastpath=True),
                      query_observer=observe, collect_trace=False)
    if stats['requests'] != len(rows) or stats['hits']+stats['misses'] != stats['accesses']:
        raise AssertionError('Replay counter mismatch')
    final_entries = (result['cache_final_bytes']//RAW_ENTRY_BYTES)
    # simulate() retains the original logical cache size; every C3 entry is w=3.
    if result['cache_final_bytes'] != final_entries*RAW_ENTRY_BYTES:
        raise AssertionError('M9-B logical entry size changed')
    # Last post-query occupancy is exposed by the observer without retaining a trace.
    final_bytes = observe.last_bytes
    raw_represented = final_entries*RAW_ENTRY_BYTES
    row = dict(dataset=dataset, budget_bytes=budget, budget_mib=budget//1024**2,
               storage_mode=mode, total_requests=stats['requests'], total_accesses=stats['accesses'],
               cache_hits=stats['hits'], cache_misses=stats['misses'],
               hit_rate=stats['hits']/stats['accesses'] if stats['accesses'] else 0.,
               admissions=stats['admissions'], evictions=stats['evictions'],
               rejected_admissions=stats['rejected_admissions'],
               peak_resident_entries=stats['peak_entries'], final_resident_entries=final_entries,
               peak_physical_resident_bytes=stats['peak_bytes'], final_physical_resident_bytes=final_bytes,
               mean_resident_entries=stats['resident_entry_samples']/len(rows),
               mean_physical_bytes=stats['resident_byte_samples']/len(rows),
               total_logical_raw_bytes_represented=raw_represented,
               effective_capacity_multiplier=raw_represented/final_bytes if final_bytes else None,
               theoretical_raw_entry_capacity=budget//RAW_ENTRY_BYTES,
               estimated_compressed_entry_capacity=None,
               mean_compressed_entry_bytes=mean(admitted_sizes) if admitted_sizes else None,
               p50_compressed_entry_bytes=percentile(admitted_sizes, .5),
               p95_compressed_entry_bytes=percentile(admitted_sizes, .95),
               min_compressed_entry_bytes=min(admitted_sizes) if admitted_sizes else None,
               max_compressed_entry_bytes=max(admitted_sizes) if admitted_sizes else None,
               access_sequence_sha256=sequence.hexdigest(),
               query_order_hash=result['query_order_hash'], user_assignment_hash=result['user_assignment_hash'])
    if mode == COMPRESSED:
        population = [RAW_Q_BYTES+e['compressed_kv_frame_bytes'] for e in size_index['entries']
                      if e['dataset'] == dataset]
        row['estimated_compressed_entry_capacity'] = math.floor((budget-PROFILE_BYTES)/mean(population))
    return row


def paired_delta(raw, compressed):
    if any(raw[k] != compressed[k] for k in ('dataset', 'budget_bytes', 'total_requests',
                                             'total_accesses', 'access_sequence_sha256',
                                             'query_order_hash', 'user_assignment_hash')):
        raise ValueError('Storage modes did not replay the same logical access trace')
    ratio = lambda field: (compressed[field]/raw[field] if raw[field] else None)
    return dict(dataset=raw['dataset'], budget_bytes=raw['budget_bytes'], budget_mib=raw['budget_mib'],
                compressed_minus_raw_hit_rate_percentage_points=100*(compressed['hit_rate']-raw['hit_rate']),
                compressed_minus_raw_hit_count=compressed['cache_hits']-raw['cache_hits'],
                compressed_minus_raw_miss_count=compressed['cache_misses']-raw['cache_misses'],
                compressed_minus_raw_eviction_count=compressed['evictions']-raw['evictions'],
                peak_resident_entry_multiplier=ratio('peak_resident_entries'),
                final_resident_entry_multiplier=ratio('final_resident_entries'))


def run(args):
    budgets = validate_budgets(args.budgets_mib)
    index = load_measured_sizes(args.capture_manifest, args.holdout_dir, args.rate_calibration_dir)
    paths = {'snips': Path(args.snips), 'multiwoz': Path(args.multiwoz)}
    workloads = {dataset: read_workload(path, dataset) for dataset, path in paths.items()}
    facts = {dataset: workload_facts(rows) for dataset, rows in workloads.items()}
    if any(not any(e['dataset'] == dataset for e in index['entries']) for dataset in workloads):
        raise ValueError('Measured size index lacks one of the workload datasets')
    print(f"workloads: " + ', '.join(f"{d}={f['query_count']} queries/{f['access_count']} accesses"
                                  for d, f in facts.items()))
    print(f"measured w=3 frame sizes={index['count']}; mapping={MAPPING_MODE}")
    print(f"budgets MiB={list(args.budgets_mib)}; expected experiment cells={len(paths)*len(budgets)*len(MODES)}")
    if args.dry_run:
        for dataset, item in facts.items():
            print(f"{dataset}: total_accesses={item['access_count']} "
                  f"unique_cache_keys={item['unique_cache_keys']} "
                  f"repeated_accesses={item['repeated_accesses']} "
                  f"keys_accessed_more_than_once={item['keys_accessed_more_than_once']} "
                  f"total_revisit_events={item['total_revisit_events']} "
                  f"max_accesses_for_one_key={item['max_accesses_for_one_key']} "
                  f"reuse_opportunity={str(item['reuse_opportunity']).lower()}")
        print('Dry run complete; no replay or codec execution.')
        return dict(dry_run=True, workload_facts=facts, size_count=index['count'])
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError('C3 output directory already exists; frozen artifacts will not be overwritten')
    output.mkdir(parents=True)
    write_json(output/'size_index.json', index)
    results, deltas = [], []
    for dataset, rows in workloads.items():
        for budget in budgets:
            pair = {mode: replay_cell(rows, dataset, budget, mode, index, seed=args.seed, users=args.users)
                    for mode in MODES}
            for mode in MODES:
                if pair[mode]['access_sequence_sha256'] != facts[dataset]['access_sequence_sha256']:
                    raise AssertionError('M9-B replay changed prepared access order')
                results.append(pair[mode])
            deltas.append(paired_delta(pair[RAW], pair[COMPRESSED]))
            print(f"[{dataset} {budget//1024**2} MiB] raw={pair[RAW]['hit_rate']:.4f} compressed={pair[COMPRESSED]['hit_rate']:.4f}", flush=True)
    manifest = dict(stage='C3-CAPACITY-PRESSURE', status='COMPLETED', git=git_state(),
        storage_policies=list(MODES), compressed_policy='UNIFORM_K20_V16',
        storage_representation='B2_ANCHOR_MOD_RESIDUAL_KV',
        profile=dict(sha256=PROFILE_SHA256, bytes=PROFILE_BYTES,
                     path=str((Path(args.rate_calibration_dir)/'profiles/matched_uniform_k20_v16.bin').resolve())),
        holdout_artifact_sha256=index['source_sha256'], physical_size_mapping_mode=MAPPING_MODE,
        physical_size_assignment='SHA256(dataset, cache key) modulo sorted measured dataset-specific w=3 frames',
        workload_provenance={d: dict(path=str(paths[d].resolve()), sha256=file_hash(paths[d]), **facts[d]) for d in paths},
        budgets_bytes=list(budgets), datasets=list(paths), users=args.users, seed=args.seed,
        raw_entry_bytes=RAW_ENTRY_BYTES, raw_q_bytes=RAW_Q_BYTES,
        accounting_rule='raw: 1,474,560 bytes/entry; compressed: 491,520 Q bytes + measured B2 K/V frame; shared 4,108-byte profile once per cache; logical admission/eviction score uses unchanged raw size',
        occupancy_sampling='post-query resident occupancy; includes one shared profile in compressed mode',
        effective_capacity_multiplier_definition='final resident logical raw QKV bytes / final charged physical resident bytes',
        no_model_inference=True, no_codec_execution=True,
        operation_calls=dict(quantization=0, arithmetic_encode=0, arithmetic_decode=0,
                             dequantization=0, model_inference=0))
    write_csv(output/'capacity_results.csv', results, results[0].keys())
    write_csv(output/'paired_deltas.csv', deltas, deltas[0].keys())
    write_json(output/'summary.json', dict(cell_count=len(results), paired_cell_count=len(deltas),
        measured_size_count=index['count'], mapping_mode=MAPPING_MODE,
        best_hit_rate_delta_percentage_points=max(d['compressed_minus_raw_hit_rate_percentage_points'] for d in deltas),
        paired_deltas=deltas))
    manifest['output_sha256'] = {name: file_hash(output/name) for name in
        ('size_index.json', 'capacity_results.csv', 'paired_deltas.csv', 'summary.json')}
    write_json(output/'manifest.json', manifest)
    return dict(dry_run=False, cells=len(results), output=str(output))
