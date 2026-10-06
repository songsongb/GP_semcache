"""CPU-only controlled two-phase physical-byte capacity replay; no codec/model calls."""
import argparse
from collections import OrderedDict
import csv
import json
from pathlib import Path

from . import c7b3_q_freeze as provenance

require = provenance.require
POLICIES = ('RAW_QKV', 'KV_COMP', 'Q24_KV_COMP')
BUDGETS = (2, 4, 8, 16)
RAW_ROLE_BYTES = 32*3*2560*2
RAW_ENTRY_BYTES = 3*RAW_ROLE_BYTES
PROTOCOL = 'CONTROLLED_TWO_PHASE_CAPACITY_REPLAY'
DUPLICATE_RULE = 'REJECT_RESIDENT_DUPLICATE_KEEP_EXISTING_NO_REFRESH; re-admit normally if the key was evicted'
MODE = {'KV_COMP': provenance.BASELINE, 'Q24_KV_COMP': provenance.Q24}


def logical_key(episode):
    require(type(episode['cluster']) is int and len(episode['token_ids']) == 3 and
        all(type(t) is int and t >= 0 for t in episode['token_ids']), 'Expected semantic cluster + exact w3 token key')
    return episode['cluster'], tuple(episode['token_ids'])


def serialized_key(key):
    return [key[0], list(key[1])]


def validate_account(account, compressed_q):
    keys = ('raw_q_bytes', 'raw_k_bytes', 'raw_v_bytes', 'raw_qkv_bytes', 'resident_raw_q_bytes',
        'compressed_q_bitstream_bytes', 'local_q_metadata_bytes', 'compressed_q_frame_bytes',
        'compressed_kv_frame_bytes', 'local_kv_metadata_bytes', 'total_resident_qkv_bytes',
        'shared_profile_bytes_charged_per_entry')
    require(all(type(account[k]) is int and account[k] >= 0 for k in keys), 'Invalid measured entry bytes')
    require(all(account[k] == RAW_ROLE_BYTES for k in ('raw_q_bytes', 'raw_k_bytes', 'raw_v_bytes')) and
            account['raw_qkv_bytes'] == RAW_ENTRY_BYTES, 'Raw FP16 QKV dimension accounting changed')
    q_frame, kv_frame = account['compressed_q_frame_bytes'], account['compressed_kv_frame_bytes']
    require(q_frame == account['compressed_q_bitstream_bytes']+account['local_q_metadata_bytes'], 'Q metadata accounting mismatch')
    require(0 <= account['local_kv_metadata_bytes'] < kv_frame, 'KV frame must include local metadata and payload')
    require(account['shared_profile_bytes_charged_per_entry'] == 0, 'Shared profiles must not be charged per entry')
    require(account['resident_raw_q_bytes'] == (0 if compressed_q else RAW_ROLE_BYTES) and
            account['raw_q_resident_after_insert'] is (not compressed_q), 'Q residency mismatch')
    require((q_frame > 0) == compressed_q and account['total_resident_qkv_bytes'] ==
            account['resident_raw_q_bytes']+q_frame+kv_frame, 'Resident frame accounting mismatch')
    require(type(account['incremental_resident_byte_reduction_vs_kv_baseline']) is int and
            account['incremental_resident_byte_reduction_vs_kv_baseline'] ==
            RAW_ROLE_BYTES+kv_frame-account['total_resident_qkv_bytes'], 'Incremental bytes mismatch')


def prepare(args):
    frozen = provenance.validate_freeze(args.freeze_decision)
    require(Path(frozen['b2_manifest_path']).resolve() == (args.c7b2_root/'manifest.json').resolve(), 'Freeze belongs to another B2 root')
    hashes = dict(frozen['input_hashes'])
    hashes[str(args.freeze_decision.resolve())] = provenance.sha(args.freeze_decision)
    b2 = provenance.read(args.c7b2_root/'manifest.json')
    plan = provenance.read(args.plan_dir/'manifest.json')
    plan_sha = provenance.sha(args.plan_dir/'manifest.json')
    require(b2['teacher_forced_continuation_provenance']['plan_manifest_sha256'] == plan_sha, 'B2 plan manifest mismatch')
    provenance.fields(plan, dict(selection_version='current_user_content_v2', history_k=4,
        physical_safety_contract=provenance.CONTRACT), 'B3 plan')
    hashes.update(provenance.output_hashes(args.plan_dir, plan, ('evaluation_selection.json',)))
    hashes[str((args.plan_dir/'manifest.json').resolve())] = plan_sha
    provenance.check_hash(args.plan_dir/'evaluation_selection.json', provenance.SELECTION_SHA)
    selection = provenance.read(args.plan_dir/'evaluation_selection.json')
    episodes = selection['episodes']
    require(len(episodes) == 32 and len(provenance.unique(episodes, 'episode_id')) == 32 and
        selection['selection_sha256'] == provenance.digest(episodes), 'Frozen32 selection identity/digest mismatch')
    rows = provenance.read_csv(args.c7b2_root/'per_case.csv')
    accounting = provenance.read(args.c7b2_root/'storage_accounting.json')
    sizes = {'RAW_QKV': {e['episode_id']: RAW_ENTRY_BYTES for e in episodes}}
    accounts = {}
    for policy, mode in MODE.items():
        matched = provenance.unique([r for r in rows if r['mode'] == mode], 'episode_id')
        require(set(matched) == {e['episode_id'] for e in episodes}, 'B2 per-entry source identities differ')
        sizes[policy] = {}; accounts[policy] = {}
        for index, e in enumerate(episodes):
            key = logical_key(e)
            require(e['cache_key'] == serialized_key(key) and e['selected_hit_count'] == 1, 'Logical w3 HIT contract changed')
            r = matched[e['episode_id']]
            for field, value in e.items():
                actual = json.loads(r[field]) if isinstance(value, (dict, list)) else r[field]
                expected = value if isinstance(value, (dict, list)) else ('' if value is None else str(value))
                require(actual == expected, 'B2 frozen episode metadata differs: '+field)
            require(int(r['episode_index']) == index and r['logical_event_hash'] == provenance.digest(e), 'Frozen event binding changed')
            account = json.loads(r['storage_accounting'])
            validate_account(account, policy == 'Q24_KV_COMP')
            accounts[policy][e['episode_id']] = account
            sizes[policy][e['episode_id']] = account['total_resident_qkv_bytes']
        saved = accounting['modes'][mode]
        for field, total in saved['totals'].items():
            require(sum(a[field] for a in accounts[policy].values()) == total, 'Aggregate B2 byte mismatch: '+field)
        total = sum(sizes[policy].values())
        require(saved['total_resident_bytes'] == total and saved['mean_resident_bytes'] == total/32 and
                saved['whole_qkv_compression_ratio'] == 32*RAW_ENTRY_BYTES/total and
                saved['shared_profile_bytes_charged_per_entry'] == 0, 'Aggregate resident accounting differs')
    for e in episodes:
        a, b = (accounts[p][e['episode_id']] for p in MODE)
        require(all(a[k] == b[k] for k in ('compressed_kv_frame_bytes', 'local_kv_metadata_bytes')),
                'Frozen KV physical sizes differ between policies')
    shared = accounting['shared_profile_bytes']
    require(shared['Q24'] == Path(frozen['q_profile_path']).stat().st_size and
            shared['KV'] == Path(frozen['kv_profile_path']).stat().st_size, 'Shared profile byte mismatch')
    provenance.verify_files(hashes)
    return episodes, sizes, dict(RAW_QKV=0, KV_COMP=shared['KV'], Q24_KV_COMP=shared['KV']+shared['Q24']), hashes


class ByteLRU:
    """Experiment-only entry-byte LRU. Duplicates match GlobalCache.insert rejection."""
    def __init__(self, budget):
        require(type(budget) is int and budget >= 0, 'Nonnegative physical byte budget required')
        self.budget = budget
        self.entries = OrderedDict()
        self.resident_bytes = self.peak_bytes = self.peak_entries = 0

    def admit(self, key, entry_bytes, episode_id):
        require(type(entry_bytes) is int and entry_bytes > 0, 'Positive integer entry bytes required')
        result = dict(entry_bytes=entry_bytes, resident_bytes_before=self.resident_bytes, evicted_keys=[],
            evicted_bytes=0, evictions=[], admitted=False, admission_rejected_oversize=False)
        if key in self.entries:
            result['event_type'] = 'DUPLICATE_KEY_REJECTED'
        elif entry_bytes > self.budget:
            result.update(event_type='OVERSIZE_REJECTED', admission_rejected_oversize=True)
        else:
            while self.resident_bytes+entry_bytes > self.budget:
                victim, entry = self.entries.popitem(last=False)
                before = self.resident_bytes
                self.resident_bytes -= entry['entry_bytes']
                result['evicted_keys'].append(serialized_key(victim))
                result['evicted_bytes'] += entry['entry_bytes']
                result['evictions'].append(dict(logical_key=serialized_key(victim), **entry,
                    resident_bytes_before=before, resident_bytes_after=self.resident_bytes))
            self.entries[key] = dict(source_episode_id=episode_id, entry_bytes=entry_bytes)
            self.resident_bytes += entry_bytes
            self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
            self.peak_entries = max(self.peak_entries, len(self.entries))
            result.update(event_type='ADMITTED', admitted=True)
        result['resident_bytes_after'] = self.resident_bytes
        result['resident_entry_count'] = len(self.entries)
        return result

    def lookup(self, key):
        hit = key in self.entries
        if hit:
            self.entries.move_to_end(key)
        return dict(hit=hit, event_type='HIT' if hit else 'MISS', resident_bytes=self.resident_bytes,
            resident_entry_count=len(self.entries), resident_source_episode_id=self.entries[key]['source_episode_id'] if hit else None)

    def snapshot(self):
        return [dict(logical_key=serialized_key(key), **value) for key, value in self.entries.items()]


def replay(episodes, sizes, policy, budget, equivalent=None):
    require(policy in POLICIES, 'Only RAW_QKV, KV_COMP, Q24_KV_COMP allowed')
    cache = ByteLRU(budget)
    events = []
    common = dict(policy=policy, budget='B'+str(equivalent) if equivalent is not None else budget,
                  budget_bytes=budget, budget_raw_entry_equivalent=equivalent)
    for index, e in enumerate(episodes):
        key = logical_key(e)
        event = cache.admit(key, sizes[e['episode_id']], e['episode_id'])
        events.append(dict(common, phase='ADMISSION', event_index=index, episode_id=e['episode_id'],
                           logical_key=serialized_key(key), **event))
    admissions = list(events)
    retained = sum(logical_key(e) in cache.entries for e in episodes)
    after_admission = cache.snapshot()
    for index, e in enumerate(episodes):
        key = logical_key(e)
        events.append(dict(common, phase='LOOKUP', event_index=len(episodes)+index, episode_id=e['episode_id'],
                          logical_key=serialized_key(key), **cache.lookup(key)))
    hits = sum(e['hit'] for e in events if e['phase'] == 'LOOKUP')
    summary = dict(common, source_admission_events=len(episodes), unique_logical_keys_seen=len({logical_key(e) for e in episodes}),
        duplicate_key_updates=0, duplicate_key_rejections=sum(e['event_type'] == 'DUPLICATE_KEY_REJECTED' for e in admissions),
        successful_admissions=sum(e['admitted'] for e in admissions),
        oversize_rejections=sum(e['admission_rejected_oversize'] for e in admissions),
        eviction_count=sum(len(e['evicted_keys']) for e in admissions), evicted_bytes=sum(e['evicted_bytes'] for e in admissions),
        peak_resident_bytes=cache.peak_bytes, final_resident_bytes=cache.resident_bytes,
        peak_resident_entries=cache.peak_entries, final_resident_entries=len(cache.entries),
        target_lookups=len(episodes), target_hits=hits, target_misses=len(episodes)-hits,
        hit_rate=hits/len(episodes), miss_rate=1-hits/len(episodes),
        retained_selected_source_count=retained, retained_unique_key_count=len(cache.entries))
    return summary, events, dict(common, after_admission=after_admission, after_lookup=cache.snapshot())


def comparisons(summaries):
    rows = []
    for equivalent in BUDGETS:
        group = {r['policy']: r for r in summaries if r['budget_raw_entry_equivalent'] == equivalent}
        for before, after in (('RAW_QKV', 'KV_COMP'), ('KV_COMP', 'Q24_KV_COMP'), ('RAW_QKV', 'Q24_KV_COMP')):
            rows.append(dict(budget_raw_entry_equivalent=equivalent, budget_bytes=equivalent*RAW_ENTRY_BYTES,
                comparison=after+' - '+before, **{k: group[after][k]-group[before][k]
                    for k in ('final_resident_entries', 'eviction_count', 'target_hits', 'hit_rate')}))
    return rows


def representative_budgets(summaries, deltas):
    # Smallest differentiating budget, then largest larger budget with observable
    # additional Q24 benefit. No quality data or weighted score enters this rule.
    different = [b for b in BUDGETS if len({(r['final_resident_entries'], r['eviction_count'], r['target_hits'])
        for r in summaries if r['budget_raw_entry_equivalent'] == b}) > 1]
    if not different:
        return []
    chosen = [different[0]]
    extra = [r['budget_raw_entry_equivalent'] for r in deltas if r['comparison'] == 'Q24_KV_COMP - KV_COMP'
        and r['budget_raw_entry_equivalent'] > chosen[0] and
        (r['final_resident_entries'] > 0 or r['target_hits'] > 0 or r['eviction_count'] < 0)]
    if extra:
        chosen.append(max(extra))
    return chosen


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, separators=(',', ':')) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def run(args):
    provenance.output_location(args.output_root)
    require(not args.output_root.exists(), 'New C8-A output root required; no overwrite')
    episodes, sizes, shared, hashes = prepare(args)
    summaries, events, snapshots = [], [], []
    for equivalent in BUDGETS:
        for policy in POLICIES:
            row, trace, residency = replay(episodes, sizes[policy], policy, equivalent*RAW_ENTRY_BYTES, equivalent)
            summaries.append(row); events.extend(trace); snapshots.append(residency)
    deltas = comparisons(summaries)
    recommended = representative_budgets(summaries, deltas)
    provenance.verify_files(hashes)
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    summary = dict(replay_protocol=PROTOCOL, eviction_policy='BYTE_AWARE_LRU', duplicate_key_semantics=DUPLICATE_RULE,
        policies=list(POLICIES), budgets_raw_entry_equivalent=list(BUDGETS), results=summaries,
        shared_profile_bytes=shared, shared_profile_bytes_charged_per_entry=0,
        raw_q_bytes=RAW_ROLE_BYTES, raw_k_bytes=RAW_ROLE_BYTES, raw_v_bytes=RAW_ROLE_BYTES, raw_qkv_bytes=RAW_ENTRY_BYTES,
        physical_byte_source='RAW: exact FP16 dimensions validated against B2; compressed: measured per-case B2 storage_accounting',
        metadata_accounting='Q and KV frame byte counts already include local metadata; shared profiles excluded',
        accounting_scope='resident tensor/frame bytes; excludes logical index, Python/allocator overhead, and shared profiles',
        retained_selected_source_count_definition='number of selected logical source keys available, counted per episode including duplicates',
        recommended_c8b_budgets_raw_entry_equivalent=recommended,
        recommendation_rule='smallest differentiating budget; largest larger budget with Q24-vs-KV capacity/hit benefit, at most two')
    provenance.write(root/'summary.json', summary)
    provenance.write(root/'causal_comparisons.json', deltas)
    provenance.write(root/'residency_trace.json', dict(order='LRU to MRU', runs=snapshots))
    write_csv(root/'summary.csv', summaries); write_csv(root/'per_event.csv', events)
    lines = ['# C8-A fixed-byte capacity replay', '', PROTOCOL+'. All 32 source admission attempts precede all 32 target lookups.',
        'This is a controlled pressure trace, not a natural online arrival trace. No target admissions occur.',
        'Storage compression reduces bytes per entry. Cache-behavior benefit is measured separately as retention, eviction, and exact-w3 hits at equal bytes.',
        'Resident duplicates retain the existing source and do not refresh LRU; duplicate_key_updates is therefore zero. Rejected duplicates have their own counter.',
        'Hits depend only on (semantic cluster, exact three token IDs) residency. The retained source episode is recorded, including when a duplicate key came from another selected source.', '',
        '| Budget | Policy | Retained entries | Evictions | Hits / 32 | Hit rate |', '|---|---|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"| {row['budget']} | {row['policy']} | {row['final_resident_entries']} | {row['eviction_count']} | {row['target_hits']} | {row['hit_rate']:.6f} |")
    lines += ['', 'Incremental effects (negative eviction delta means fewer evictions):', '',
        '| Budget | Comparison | Entry delta | Eviction delta | Hit delta | Hit-rate delta |', '|---|---|---:|---:|---:|---:|']
    for row in deltas:
        lines.append(f"| B{row['budget_raw_entry_equivalent']} | {row['comparison']} | {row['final_resident_entries']} | {row['eviction_count']} | {row['target_hits']} | {row['hit_rate']:.6f} |")
    useful = [r['budget_raw_entry_equivalent'] for r in deltas if r['comparison'] == 'Q24_KV_COMP - KV_COMP' and
        (r['final_resident_entries'] > 0 or r['target_hits'] > 0 or r['eviction_count'] < 0)]
    hit_gains = [r['budget_raw_entry_equivalent'] for r in deltas if r['comparison'] == 'Q24_KV_COMP - KV_COMP' and r['target_hits'] > 0]
    lines += ['', 'Q24 additional observable cache-behavior benefit at budgets: '+str(useful)+'.',
        'Q24 increases target hit rate over KV_COMP at budgets: '+str(hit_gains)+'.',
        'Representative budgets for later C8-B review: '+str(recommended)+'. Selected only by capacity/hit behavior; C8-B was not launched.',
        'No model inference, quality evaluation, transport compression, codec execution, or latency benchmarking was performed.', '']
    (root/'summary.md').write_text('\n'.join(lines))
    provenance.write(root/'manifest.json', dict(stage='C8-A', status='COMPLETE',
        research_scope='fixed-byte resident capacity / eviction / exact-w3 hit replay', replay_protocol=PROTOCOL,
        eviction_policy='BYTE_AWARE_LRU', duplicate_key_semantics=DUPLICATE_RULE, policies=list(POLICIES),
        budgets_raw_entry_equivalent=list(BUDGETS), budgets_bytes=[b*RAW_ENTRY_BYTES for b in BUDGETS], target_lookups=32,
        frozen32_selection_sha=provenance.SELECTION_SHA, c7b3_freeze_decision_sha256=provenance.sha(args.freeze_decision),
        q24_profile_sha256=provenance.Q_SHA, kv_profile_sha256=provenance.KV_SHA,
        physical_safety_contract=provenance.CONTRACT, cached_payload='TOTAL_QKV', model_inference_performed=False,
        quality_evaluated=False, latency_evaluated=False, transport_compression_enabled=False,
        runtime_q_profile_fit_count=0, runtime_storage_kv_cdf_fit_count=0, input_hashes=hashes,
        shared_profile_bytes=shared, recommended_c8b_budgets_raw_entry_equivalent=recommended,
        git=provenance.git(), output_hashes={p.name: provenance.sha(p) for p in root.iterdir() if p.is_file()}))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in dict(freeze_decision='results/cachegen/c7/q24_freeze/freeze_decision.json',
        plan_dir='results/cachegen/c6b3/multiwoz_plan_v2', c7b2_root='results/cachegen/c7/b2_q_quality_gate_frozen32',
        output_root='results/cachegen/c8/a_capacity_replay').items():
        parser.add_argument('--'+name.replace('_', '-'), type=Path, default=Path(default))
    args = parser.parse_args(argv)
    run(args)
    print('C8-A COMPLETE: '+str(args.output_root))
