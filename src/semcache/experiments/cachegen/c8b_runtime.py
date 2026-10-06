"""C8-B experiment-only byte LRU with actual resident TOTAL-QKV payloads."""
from pathlib import Path
from types import SimpleNamespace
import importlib.metadata
import json

from . import c8b_quality as quality
from . import c8a_capacity as capacity
from . import c7b3_q_freeze as provenance
from . import c7b2_runtime as previous

require = provenance.require


def representation_check(policy, entry, storage=None):
    import torch
    require(policy in quality.POLICIES, 'Unknown C8-B policy')
    tensors, raw_q, kv = entry.tensors, getattr(entry, 'q_tensors', None), getattr(entry, 'compressed_kv', None)
    if policy == 'RAW_QKV':
        require(tensors and raw_q is None and kv is None, 'RAW requires raw QKV only')
        for values in tensors.values():
            require(len(values) == 3 and all(t.dtype == torch.float16 and t.shape == (1, 3, 2560)
                and t.device.type == 'cpu' and torch.isfinite(t).all().item() for t in values), 'Invalid raw TOTAL QKV')
    else:
        require(tensors is None and kv is not None and kv.profile_sha256 == provenance.KV_SHA,
            'Compressed modes must not retain raw K/V')
        require(storage is not None, 'Compressed lookup needs the frozen codec')
        if policy == 'KV_COMP':
            require(raw_q and not hasattr(entry, 'compressed_q_frame'), 'KV_COMP must retain raw Q only')
            require(all(t.dtype == torch.float16 and t.device.type == 'cpu' and t.shape == (1, 3, 2560)
                for t in raw_q.values()), 'Invalid resident raw Q')
        else:
            require(raw_q is None and entry.compressed_q_frame, 'Q24 must not retain raw resident Q')
            require(entry.compressed_q_profile_sha256 == provenance.Q_SHA, 'Resident Q24 profile hash changed')
    require((len(tensors) if policy == 'RAW_QKV' else len(raw_q) if policy == 'KV_COMP' else storage.q.profile.metadata['layers']) == 32,
        'Resident payload must cover all32 layers')
    return dict(policy=policy, raw_q_resident_after_insert=policy != 'Q24_KV_COMP',
        raw_kv_resident_after_insert=policy == 'RAW_QKV', resident_bytes=entry.physical_tensor_bytes,
        temporary_source_working_copies_charged=False, shared_profile_bytes_charged_per_entry=0)


class ResidentCache:
    """Same A eviction transitions; owns resident entries, decodes temporary HIT views."""
    def __init__(self, budget, policy, storage):
        require(budget in quality.BUDGETS and policy in quality.POLICIES, 'Only fixed B2/B8 conditions allowed')
        self.lru = capacity.ByteLRU(budget*capacity.RAW_ENTRY_BYTES)
        self.policy, self.storage, self.payloads = policy, storage, {}

    def admit(self, episode, entry):
        key = capacity.logical_key(episode)
        require(entry.key == key, 'Source logical key changed')
        event = self.lru.admit(key, entry.physical_tensor_bytes, episode['episode_id'])
        for victim in event['evicted_keys']:
            del self.payloads[(victim[0], tuple(victim[1]))]
        if event['admitted']: self.payloads[key] = entry
        return event

    def lookup(self, episode):
        # No admission API is invoked during target lookup.
        event = self.lru.lookup(capacity.logical_key(episode))
        if not event['hit']: return None, event
        resident = self.payloads[capacity.logical_key(episode)]
        representation_check(self.policy, resident, self.storage)
        if self.storage is None:
            view = resident
        else:
            view = self.storage.decode_entry(resident)
            self.storage.validate_decoded(resident, view)
            require(view.resident is resident, 'Decoded view must reference the actual retained resident')
            representation_check(self.policy, resident, self.storage)
        return view, event


def validate_retained_hit(episode, retained, hit):
    """Selected target key/mask stays fixed; duplicate payload provenance follows A."""
    from .c6_quality import validate_hit
    require(capacity.logical_key(episode) == capacity.logical_key(retained), 'Different retained logical key')
    observed = dict(episode, **{k: retained[k] for k in ('source_start', 'source_user', 'source_id')})
    validate_hit(observed, hit)


class Backend(previous.Backend):
    # Inherit the exact C7-B2 mixed projection, incremental greedy, and teacher APIs.
    def __init__(self, args, prepared):
        import torch
        from .c6_runtime import Storage, make_c6_cache
        from .c6b_runtime import load_base
        from semcache.models.task_adapters import load_two_task_users
        from semcache.models.model_adapter import OPTModelAdapter
        require(args.device.startswith('cuda:') and torch.cuda.is_available(), 'Single-GPU SERAPH execution required')
        self.args, self.counts = args, previous.runtime_counts()
        kv = Storage(args)
        require(prepared.input_hashes.get(str(Path(kv.source).resolve())) == provenance.sha(kv.source),
            'Loaded C2 storage source is not bound to canonical C7-B2')
        q = previous.FrozenQ(Path(prepared.frozen['q_profile_path']), 24, provenance.Q_SHA,
            prepared.q_backend, self.counts)
        self.storages = dict(RAW_QKV=None,
            KV_COMP=previous.StorageExperiment(kv, None, self.counts),
            Q24_KV_COMP=previous.StorageExperiment(kv, q, self.counts))
        make_c6_cache('STORAGE_KV_COMP', self.storages['Q24_KV_COMP'])
        model, self.tokenizer, self.metadata = load_base(args)  # local_files_only=True
        self.model = load_two_task_users(model, args.adapter_root/'user_a', args.adapter_root/'user_b')
        self.adapter = OPTModelAdapter(self.model)
        self.rows, self.reuse_audits = prepared.rows, []
        self.storage_source = kv.source

    def forward(self, ids, user, hits, transport, **kwargs):
        require(transport is False, 'C8-B transport compression forbidden')
        if hits is None:
            from .c6b2_runtime import Backend as Native
            return Native.forward(self, ids, user, None, False, **kwargs)
        return super().forward(ids, user, hits, False, **kwargs)

    def build_source(self, episode):
        from semcache.cache.cache_entry import CacheEntry
        # Capture once per admission event, independently of both budgets/policies.
        output, captured = self.forward(self.rows[episode['source_index']]['token_ids'],
            episode['source_user'], [], False, capture=True)
        self.counts['source_forward_count'] += 1
        del output
        start = episode['source_start']
        tensors = {layer: tuple(values[role][:, start:start+3].detach().cpu().clone()
            for role in 'qkv') for layer, values in captured.items()}
        del captured
        raw = CacheEntry.from_tensors(episode['cluster'], episode['token_ids'], (start, start+3), tensors)
        raw.qkv_metadata = dict(component_scope='total_qkv', source_user=episode['source_user'], source_id=episode['source_id'])
        entries = dict(RAW_QKV=raw)
        for policy in ('KV_COMP', 'Q24_KV_COMP'):
            entries[policy], _ = self.storages[policy].encode(episode, tensors)
        require(entries['KV_COMP'].compressed_kv.bitstream == entries['Q24_KV_COMP'].compressed_kv.bitstream,
            'K20/V16 frames must be identical across policies')
        del tensors
        return entries

    def audit_workload(self, episodes):
        for episode in episodes:
            for side in ('source', 'target'):
                row = self.rows[episode[side+'_index']]
                require(self.tokenizer(row['query_text'], add_special_tokens=True, truncation=False)['input_ids'] == row['token_ids'],
                    'Runtime tokenizer differs from frozen prompt')
                extra = self.args.max_new_tokens if side == 'target' else 0
                require(len(row['token_ids'])+extra <= self.model.config.max_position_embeddings, 'Frozen prompt exceeds context')


def build_caches(backend, prepared, build_audit):
    caches = {quality.condition(b, p): ResidentCache(b, p, backend.storages[p])
              for b in quality.BUDGETS for p in quality.POLICIES}
    audits = {name: dict(condition=name, admissions=[], representation_checks=[]) for name in caches}
    # Preserve even partially built runs if encoding/admission fails.
    build_audit.extend(audits.values())
    for episode in prepared.episodes:
        entries = backend.build_source(episode)
        for name, cache in caches.items():
            entry = entries[cache.policy]
            rep = representation_check(cache.policy, entry, cache.storage)
            require(rep['resident_bytes'] == prepared.sizes[cache.policy][episode['episode_id']],
                'Actual source physical bytes disagree with C8-A: '+name+'/'+episode['episode_id'])
            actual = cache.admit(episode, entry)
            expected = prepared.expected[name]['admissions'][len(audits[name]['admissions'])]
            require(all(actual[k] == expected[k] for k in actual), 'GPU admission/eviction differs from C8-A: '+name)
            audits[name]['admissions'].append(dict(episode_id=episode['episode_id'], **actual))
            audits[name]['representation_checks'].append(dict(episode_id=episode['episode_id'], **rep))
        del entries  # Evicted/rejected physical objects need not remain in a working pool.
    for name, cache in caches.items():
        snapshot = cache.lru.snapshot()
        require(snapshot == prepared.expected[name]['residency']['after_admission'], 'Final resident set/source differs: '+name)
        require(cache.lru.resident_bytes == prepared.expected[name]['summary']['final_resident_bytes'], 'Final resident bytes differ')
        audits[name].update(final_residents=snapshot, resident_bytes=cache.lru.resident_bytes,
            resident_entry_count=len(cache.payloads), c8a_residency_exact_match=True)
    return caches


def evaluate_targets(backend, prepared, caches, cases, hit_audit):
    from semcache.semantic.hit_selection import CacheHit
    from semcache.semantic.subsequence import Subsequence
    by_id = provenance.unique(prepared.episodes, 'episode_id')
    for index, (episode, official) in enumerate(zip(prepared.episodes, prepared.official)):
        canonical = list(official['generated_token_ids'])
        require(0 < len(canonical) <= 160, 'Invalid canonical FULL continuation')
        require(backend.tokenizer.decode(canonical, skip_special_tokens=True) == official['generated_text'], 'Canonical FULL text/token mismatch')
        base_context = SimpleNamespace(hits=None, transport=False)
        full_logits = backend.teacher_logits(base_context, episode, canonical)
        common = dict(episode, episode_index=index, user=episode['target_user'],
            reference_text=prepared.rows[episode['target_index']]['reference_text'],
            teacher_forced_canonical_token_ids=canonical, teacher_forced_canonical_sha256=provenance.digest(canonical))
        cases.append(dict(common, condition=quality.REFERENCE, policy=quality.REFERENCE,
            budget_raw_entry_equivalent=None, budget_bytes=None, hit=False, retained_source_episode_id=None,
            generated_token_ids=canonical, generated_text=official['generated_text'],
            generation_source='imported_hash_bound_FULL_RECOMPUTE',
            generation_fidelity=quality.generation_check(canonical, official['generated_text'], official, False),
            teacher_forced=quality.teacher_metrics(full_logits, full_logits, False)))
        for name, cache in caches.items():
            before = dict(backend.counts)
            backend.reuse_audits.clear()
            view, event = cache.lookup(episode)
            expected = prepared.expected[name]['lookups'][index]
            require(all(event[k] == expected[k] for k in event), 'Target HIT vector/source differs from C8-A: '+name)
            hits = None
            if view is not None:
                window = Subsequence(tuple(episode['token_ids']), episode['target_start'], episode['target_start']+3)
                hits = [CacheHit(window, view)]
                retained = by_id[event['resident_source_episode_id']]
                validate_retained_hit(episode, retained, hits[0])
            context = SimpleNamespace(hits=hits, transport=False)
            tokens, text = backend.greedy(context, episode)
            generation = quality.generation_check(tokens, text, official, event['hit'])
            logits = backend.teacher_logits(context, episode, canonical)
            teacher = quality.teacher_metrics(full_logits, logits, event['hit'])
            if hits: validate_retained_hit(episode, retained, hits[0])
            require(len(backend.reuse_audits) == (2 if hits else 0), 'HIT must reuse QKV in greedy prefill and teacher; MISS must be native')
            delta = {k: backend.counts[k]-before[k] for k in before}
            require(all(delta[k] == 0 for k in quality.SAFETY), 'Runtime fitting/transport forbidden')
            require(delta['source_forward_count'] == delta['q_encode_count'] == 0, 'No target admission/encoding/source forward allowed')
            audit = dict(episode_id=episode['episode_id'], condition=name, **event,
                c8a_hit_exact_match=True, reused_projection_rows_per_role_per_layer=3 if hits else 0,
                native_rows_skipped_per_role_per_layer=3 if hits else 0, same_reuse_mask_qkv=True,
                reuse_audits=list(backend.reuse_audits), target_admission_performed=False)
            hit_audit.append(audit)
            cases.append(dict(common, condition=name, policy=cache.policy,
                budget_raw_entry_equivalent=expected['budget_raw_entry_equivalent'], budget_bytes=cache.lru.budget,
                hit=event['hit'], retained_source_episode_id=event['resident_source_episode_id'],
                generated_token_ids=tokens, generated_text=text, generation_fidelity=generation,
                teacher_forced=teacher, runtime_counters=delta, logical_event_hash=provenance.digest(episode),
                generation_source='executed_c8b_independent_greedy'))
            del logits, context, view, hits
        del full_logits
        print(f'Completed frozen32 {index+1}/32: {episode["episode_id"]}', flush=True)
    for name, cache in caches.items():
        require(cache.lru.snapshot() == prepared.expected[name]['residency']['after_lookup'], 'Final lookup residency/order differs')
        observed = sum(r['hit'] for r in hit_audit if r['condition'] == name)
        require(observed == prepared.expected[name]['summary']['target_hits'], 'Executed hit count differs')


def run(args, prepared, backend_factory=Backend):
    import torch
    from . import c7b_q_capture as b0
    from .c6_quality import BLEU
    from .c6b3_2_multiwoz import GENERATION
    root = args.output_root
    require(not root.exists(), 'New C8-B root required')
    root.mkdir(parents=True)
    manifest = dict(stage='C8-B', status='STARTING',
        research_scope='selection-based controlled fixed-byte capacity/hit/quality replay', evaluation_label=quality.LABEL,
        budgets_raw_entry_equivalent=list(quality.BUDGETS), policies=list(quality.POLICIES), reference_mode=quality.REFERENCE,
        replay_protocol=capacity.PROTOCOL, eviction_policy='BYTE_AWARE_LRU', duplicate_key_semantics=capacity.DUPLICATE_RULE,
        physical_safety_contract=provenance.CONTRACT, cached_payload='TOTAL_QKV',
        c7b3_freeze_decision_sha256=provenance.sha(args.freeze_decision),
        **{'c8a_'+name.replace('.json', '')+'_sha256': provenance.sha(args.c8a_root/name)
            for name in ('manifest.json', 'summary.json', 'residency_trace.json')},
        c8a_per_event_sha256=provenance.sha(args.c8a_root/'per_event.csv'),
        frozen32_selection_sha256=provenance.SELECTION_SHA, q24_profile_sha256=provenance.Q_SHA, kv_profile_sha256=provenance.KV_SHA,
        model=b0.MODEL, model_revision=b0.REVISION, tokenizer_revision=b0.REVISION, adapter_hashes=b0.WEIGHTS,
        dtype='float16', device=args.device, seed=args.seed, prompt_version=b0.VERSION,
        adapter_file_hashes=prepared.provenance['adapter_hashes'],
        adapter_freeze_decision_sha256=provenance.sha(args.adapter_freeze_decision),
        model_inference_performed=False, training_performed=False, transport_compression_enabled=False,
        latency_evaluated=False,
        frozen32_used_for_q_selection=True, unbiased_final_test=False, system_policy_frozen=False,
        new_profile_selection_performed=False,
        runtime_q_profile_fit_count=0, runtime_storage_kv_cdf_fit_count=0, transport_encode_calls=0, transport_decode_calls=0,
        expected_hits_from_c8a={name: saved['summary']['target_hits'] for name, saved in prepared.expected.items()},
        review_rule=quality.REVIEW_RULE, miss_logit_tolerances=dict(max_abs=quality.MISS_MAX_ABS, mean_abs=quality.MISS_MEAN_ABS),
        quality_threshold_claimed=False, paper_bleu_claimed=False, protocol_provenance='REPRODUCTION_CHOICE',
        bleu_protocol=BLEU, generation=GENERATION, canonical_full_provenance=prepared.provenance,
        teacher_forced_canonical_source=prepared.canonical_source, teacher_forced_policy='same hash-bound FULL continuation; prompt + canonical[:-1]',
        source_capture_policy='one forward per canonical source admission event; CPU working copies not resident representations',
        shared_profile_bytes=prepared.shared, resident_accounting='exact C8-A frame/tensor bytes; local metadata included; shared profiles excluded',
        input_hashes=prepared.input_hashes, git=provenance.git())
    provenance.write(root/'manifest.json', manifest)
    cases, build, hits = [], [], []
    backend = None
    stage = 'backend_initialization'
    try:
        backend = backend_factory(args, prepared)
        manifest.update(counters=backend.counts, model_tokenizer_provenance=backend.metadata,
            software={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'sacrebleu')})
        manifest['storage_source_sha256'] = provenance.sha(backend.storage_source)
        props = torch.cuda.get_device_properties(torch.device(args.device))
        manifest['actual_gpu'] = dict(name=props.name, total_memory_bytes=props.total_memory, cuda_runtime=torch.version.cuda)
        stage = 'source_capture_admission'
        backend.audit_workload(prepared.episodes)
        manifest['model_inference_performed'] = True
        caches = build_caches(backend, prepared, build)
        require(backend.counts['source_forward_count'] == 32, 'Exactly32 source captures required')
        stage = 'target_quality_replay'
        evaluate_targets(backend, prepared, caches, cases, hits)
        require(all(backend.counts[k] == 0 for k in quality.SAFETY), 'Runtime fitting/transport forbidden')
        require(backend.counts['teacher_forced_forward_count'] == 224 and backend.counts['q_encode_count'] == 32,
            'Exactly224 teacher forwards and32 source Q24 encodes required')
        q_hits = sum(saved['summary']['target_hits'] for name, saved in prepared.expected.items() if name.endswith('_Q24_KV_COMP'))
        storage_hits = sum(saved['summary']['target_hits'] for name, saved in prepared.expected.items() if not name.endswith('_RAW_QKV'))
        require(backend.counts['q_decode_count'] == q_hits and backend.counts['storage_decode_count'] == storage_hits,
            'Each compressed HIT must decode its resident payload exactly once')
        provenance.verify_files(prepared.input_hashes)
        stage = 'quality_aggregation'
        official = [dict(r, episode_id=e['episode_id']) for r, e in zip(prepared.official, prepared.episodes)]
        summary, deltas = quality.results(cases, official)
        manifest.update(status='COMPLETE', c8a_residency_and_hit_vectors_reproduced=True,
            miss_paths_match_full=True, recommendation=summary['recommendation'])
        manifest.update({k: backend.counts[k] for k in quality.SAFETY})
        quality.write_reports(root, cases, build, hits, manifest, summary, deltas)
    except Exception as exc:
        manifest.update(status='INVALID', failure_stage=stage, failure_type=type(exc).__name__, failure_message=str(exc),
            recommendation='C8_B_REQUIRES_FURTHER_QUALITY_REVIEW')
        if backend is not None:
            manifest.update({k: backend.counts[k] for k in quality.SAFETY})
        try:
            quality.write_reports(root, cases, build, hits, manifest)
        except Exception as reporting:
            manifest['reporting_failure'] = str(reporting)
            (root/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n')
        raise
