"""Synchronized C9-A prompt timing over C8-B's actual resident entry/cache APIs."""
from collections import OrderedDict
from contextlib import contextmanager
import importlib.metadata
import json
from pathlib import Path
import time
from types import SimpleNamespace

from . import c9a_latency as latency
from . import c8b_runtime as c8run
from . import c8b_quality as c8
from . import c8a_capacity as capacity
from . import c7b3_q_freeze as provenance

require = provenance.require


class Clock:
    """Injectable clock; no implicit CUDA operation for CPU-only regions."""
    def __init__(self, synchronize, now=time.perf_counter_ns):
        self.sync, self.now = synchronize, now

    def measure(self, fn, gpu=True):
        if gpu: self.sync()
        start = self.now()
        result = fn()
        if gpu: self.sync()
        return result, (self.now()-start)/1e6


@contextmanager
def native_projection_events(adapter, event_factory):
    """Passive native-module hooks: no second projection, no tensor alteration."""
    records, pairs, handles = {}, {}, []
    try:
        for layer in range(len(adapter.layers)):
            for role, module in adapter.projection_modules(layer).items():
                require(not module._forward_hooks and not module._forward_pre_hooks and 'forward' not in module.__dict__,
                    'Native timing requires uninstrumented projections')
                key = (layer, role)
                def before(module, inputs, key=key):
                    require(key not in pairs, 'Native projection executed twice')
                    pair = event_factory(), event_factory(); pairs[key] = pair
                    pair[0].record()
                def after(module, inputs, output, key=key):
                    pairs[key][1].record(); records[key] = True
                handles.append(module.register_forward_pre_hook(before))
                handles.append(module.register_forward_hook(after))
        yield SimpleNamespace(pairs=pairs, records=records)
        require(len(records) == 3*len(adapter.layers), 'Missing native QKV timing regions')
    finally:
        for handle in handles: handle.remove()


class Backend(c8run.Backend):
    def prefill(self, ids, user, hits):
        """Same C8 prompt forward, with passive nested events; no greedy/teacher pass."""
        import torch
        from semcache.models.task_adapters import activate_task_user
        from semcache.edgelora.mixed_projection import mixed_projection_path
        activate_task_user(self.model, user)
        require(not any(p.requires_grad for p in self.model.parameters()), 'Evaluation parameters must remain frozen')
        inputs = dict(input_ids=torch.tensor([ids], device=self.args.device), use_cache=True)
        with torch.cuda.device(torch.device(self.args.device)), torch.inference_mode():
            if hits is None:
                with native_projection_events(self.adapter, lambda: torch.cuda.Event(enable_timing=True)) as audit:
                    output = self.model(**inputs)
                pairs = list(audit.pairs.values())
            else:
                with mixed_projection_path(self.adapter, user, hits, len(ids), measure_cuda=True) as audit:
                    output = self.model(**inputs)
                pairs = audit.cuda_event_pairs
        self.counts['target_prompt_forward_count'] = self.counts.get('target_prompt_forward_count', 0)+1
        return output, audit, pairs

    def build_source(self, episode):
        """C8 source construction split into measured forward/capture/preparation regions."""
        from semcache.cache.cache_entry import CacheEntry
        def forward():
            return self.forward(self.rows[episode['source_index']]['token_ids'], episode['source_user'], [], False, capture=True)
        (output, projections), forward_ms = self.clock.measure(forward)
        self.counts['source_forward_count'] += 1
        del output
        start = episode['source_start']
        def capture():
            return {layer: tuple(values[role][:, start:start+3].detach().cpu().clone() for role in 'qkv')
                for layer, values in projections.items()}
        tensors, capture_ms = self.clock.measure(capture)
        del projections

        def raw_prepare():
            raw = CacheEntry.from_tensors(episode['cluster'], episode['token_ids'], (start, start+3), tensors)
            raw.qkv_metadata = dict(component_scope='total_qkv', source_user=episode['source_user'], source_id=episode['source_id'])
            return raw

        if not self.codec_warmed:
            # Warm existing codecs/storage allocation with the first captured REAL
            # block. No extra source forward, admission, profile fitting, or synthetic
            # resident payload. These calls are excluded from build timings.
            raw_prepare()
            for policy in ('KV_COMP', 'Q24_KV_COMP'):
                resident, _ = self.storages[policy].encode(episode, tensors)
                view = self.storages[policy].decode_entry(resident)
                self.storages[policy].validate_decoded(resident, view)
                del view, resident
            self.clock.sync(); self.codec_warmed = True

        entries, encode_times = {}, {}
        entries['RAW_QKV'], encode_times['RAW_QKV'] = self.clock.measure(raw_prepare, gpu=False)
        for policy in ('KV_COMP', 'Q24_KV_COMP'):
            (entries[policy], _), encode_times[policy] = self.clock.measure(lambda policy=policy: self.storages[policy].encode(episode, tensors))
        require(entries['KV_COMP'].compressed_kv.bitstream == entries['Q24_KV_COMP'].compressed_kv.bitstream,
            'Frozen KV frames changed across policies')
        self.source_times[episode['episode_id']] = dict(source_forward_ms=forward_ms, source_capture_ms=capture_ms,
            encode_ms=encode_times)
        del tensors
        return entries


class TimedCache(c8run.ResidentCache):
    """Identical C8 state transitions; separates logical lookup from temporary decode."""
    def __init__(self, budget, policy, storage, clock):
        super().__init__(budget, policy, storage)
        self.clock, self.admission_times = clock, {}

    def admit(self, episode, entry):
        result, ms = self.clock.measure(lambda: super(TimedCache, self).admit(episode, entry), gpu=False)
        self.admission_times[episode['episode_id']] = ms
        return result

    def logical_lookup(self, episode):
        event = self.lru.lookup(capacity.logical_key(episode))
        resident = self.payloads[capacity.logical_key(episode)] if event['hit'] else None
        return resident, event


def timed_request(backend, cache, episode, clock):
    """Close synchronized total before correctness-only checks/event resolution."""
    from semcache.semantic.hit_selection import CacheHit
    from semcache.semantic.subsequence import Subsequence
    clock.sync(); start = clock.now()
    resident = view = None
    lookup_ms = decode_ms = 0.
    event = dict(hit=False, resident_source_episode_id=None)
    if cache is not None:
        (resident, event), lookup_ms = clock.measure(lambda: cache.logical_lookup(episode), gpu=False)
        if resident is not None:
            if cache.storage is None:
                view = resident
            else:
                # The request-start sync already completed all prior GPU work;
                # lookup itself is CPU-only. No redundant pre-decode sync.
                decode_start = clock.now()
                view = cache.storage.decode_entry(resident)
                clock.sync(); decode_ms = (clock.now()-decode_start)/1e6
    hits = None if view is None else [CacheHit(Subsequence(tuple(episode['token_ids']),
        episode['target_start'], episode['target_start']+3), view)]
    model_start = clock.now()
    output, audit, pairs = backend.prefill(backend.rows[episode['target_index']]['token_ids'], episode['target_user'], hits)
    next_token_logits = output.logits[0, -1]
    clock.sync(); model_end = clock.now()
    end = clock.now()
    model_ms, total_ms = (model_end-model_start)/1e6, (end-start)/1e6
    values = dict(lookup_ms=lookup_ms, storage_decode_ms=decode_ms, model_forward_ms=model_ms,
        target_total_ms=total_ms, control_overhead_ms=total_ms-lookup_ms-decode_ms-model_ms)
    return values, SimpleNamespace(output=output, logits=next_token_logits, audit=audit, event=event,
        pairs=pairs, resident=resident, view=view, hits=hits)


def warmup(backend, prepared, clock):
    # Exactly5 target forwards, both users/native and mixed-native paths. Source
    # construction later warms the fixed encode/decode paths without extra forwards.
    for index in range(latency.WARMUP):
        episode = prepared.episodes[index]
        ids = backend.rows[episode['target_index']]['token_ids']
        clock.sync()
        output, _, _ = backend.prefill(ids, episode['target_user'], None if index%2 == 0 else [])
        clock.sync(); del output


def evaluate(backend, prepared, caches, clock, raw, resolve_events):
    from semcache.experiments.cachegen.c7b2_runtime import projection_audit
    by_id = provenance.unique(prepared.episodes, 'episode_id')
    for index, episode in enumerate(prepared.episodes):
        for name in latency.CONDITIONS:
            cache = None if name == latency.REFERENCE else caches[name]
            initial = OrderedDict(cache.lru.entries) if cache is not None else None
            for repeat in range(latency.REPEATS):
                if cache is not None:
                    # Restore identical pre-request LRU order OUTSIDE timing. The
                    # third repetition's post-lookup order advances the trace once.
                    cache.lru.entries = OrderedDict(initial)
                values, evidence = timed_request(backend, cache, episode, clock)
                event = evidence.event
                if cache is not None:
                    expected = prepared.expected[name]['lookups'][index]
                    require(all(event[k] == expected[k] for k in event), 'C9 HIT vector/source differs from C8: '+name)
                    if evidence.resident is not None:
                        c8run.representation_check(cache.policy, evidence.resident, cache.storage)
                        if cache.storage is not None: cache.storage.validate_decoded(evidence.resident, evidence.view)
                        c8run.validate_retained_hit(episode, by_id[event['resident_source_episode_id']], evidence.hits[0])
                if evidence.hits is not None:
                    proof = projection_audit(evidence.audit, len(backend.adapter.layers),
                        len(backend.rows[episode['target_index']]['token_ids']), evidence.hits)
                    require(proof['hit_positions'] == list(range(episode['target_start'], episode['target_start']+3)), 'C9 QKV mask changed')
                require(len(evidence.pairs) == 3*len(backend.adapter.layers), 'All QKV regions must be timed')
                values['projection_or_mixed_projection_ms'] = sum(resolve_events(evidence.pairs))
                row = dict(episode_id=episode['episode_id'], episode_index=index, condition=name, repeat=repeat,
                    hit=event['hit'], retained_source_episode_id=event['resident_source_episode_id'],
                    prompt_tokens=len(backend.rows[episode['target_index']]['token_ids']),
                    native_projection_rows_skipped_per_role_per_layer=3 if event['hit'] else 0,
                    timing_method=latency.TIMING_METHOD, projection_timing_scope='nested within model_forward_ms; CUDA event sum', **values)
                latency.validate_timing(row, prepared.tie_tolerance_ms)
                raw.append(row)
                # Finite next-token diagnostics are deliberately after closure.
                backend.validate_logits(evidence.logits)
                require(all(backend.counts[k] == 0 for k in c8.SAFETY), 'Forbidden fitting/transport')
                del evidence
        print(f'Completed prompt timing {index+1}/32: {episode["episode_id"]}', flush=True)
    for name, cache in caches.items():
        require(cache.lru.snapshot() == prepared.expected[name]['residency']['after_lookup'], 'C9 final LRU state differs from C8')


def run(args, prepared):
    import torch
    from . import c7b_q_capture as b0
    from semcache.utils.timing import resolve_cuda_event_pairs
    root = args.output_root
    require(not root.exists(), 'New C9-A output root required')
    root.mkdir(parents=True)
    prepared.tie_tolerance_ms = latency.tie_tolerance_ms()
    manifest = dict(stage='C9-A', status='STARTING',
        research_scope='measured local/GPU prompt-side latency under fixed-byte SemCache capacity states',
        primary_latency_scope='cache lookup through next-token logits', budgets_raw_entry_equivalent=list(latency.BUDGETS),
        policies=list(latency.POLICIES), reference_mode=latency.REFERENCE, conditions=list(latency.CONDITIONS),
        repeats_per_target=latency.REPEATS, warmup_forwards=latency.WARMUP,
        timing_method=latency.TIMING_METHOD, cuda_synchronization_policy='sync before/end request; sync after decode and prompt forward; event resolution after enclosing sync',
        timer_resolution_seconds=time.get_clock_info('perf_counter').resolution, tie_tolerance_ms=prepared.tie_tolerance_ms,
        tie_rule='10 nominal perf_counter resolution ticks; numeric tie only, not statistical equivalence',
        nested_components=['projection_or_mixed_projection_ms'], model_use_cache=True, autoregressive_decode_steps=0,
        physical_safety_contract=provenance.CONTRACT, cached_payload='TOTAL_QKV',
        c7b3_freeze_sha256=provenance.sha(args.freeze_decision), c8b_manifest_sha256=provenance.sha(args.c8b_root/'manifest.json'),
        c8a_artifact_hashes={name: provenance.sha(args.c8a_root/name) for name in ('manifest.json', 'summary.json', 'residency_trace.json', 'per_event.csv')},
        q24_profile_sha256=provenance.Q_SHA, kv_profile_sha256=provenance.KV_SHA,
        model=b0.MODEL, model_revision=b0.REVISION, tokenizer_revision=b0.REVISION, adapter_hashes=b0.WEIGHTS,
        dtype='float16', device=args.device, seed=args.seed, frozen32_selection_sha256=provenance.SELECTION_SHA,
        model_inference_performed=False, training_performed=False, transport_compression_enabled=False,
        network_latency_evaluated=False, quality_evaluated_in_c9=False, source_build_cost_in_primary_target_latency=False,
        system_policy_frozen=False, selection_based_controlled_trace=True, review_rule=latency.REVIEW_RULE,
        source_initialization_policy='5 untimed target forwards before source timing; untimed first-real-block raw preparation and frozen codec encode/decode warmup; no per-shape sweep',
        codec_warmup_counts=dict(q_encode_count=1, q_decode_count=1, kv_encode_calls=2, storage_decode_count=2),
        replay_protocol=capacity.PROTOCOL, eviction_policy='BYTE_AWARE_LRU', duplicate_key_semantics=capacity.DUPLICATE_RULE,
        expected_hits_from_c8={name: saved['summary']['target_hits'] for name, saved in prepared.expected.items()},
        expected_execution_counts=latency.expected_execution_counts(prepared.expected),
        instrumentation_policy='passive native QKV hooks or existing mixed measure_cuda=True; no re-projection; all validation outside primary timing',
        input_hashes=prepared.input_hashes, git=provenance.git(),
        **{k: 0 for k in c8.SAFETY})
    provenance.write(root/'manifest.json', manifest)
    raw, source, backend = [], [], None
    stage = 'backend_initialization'
    try:
        backend = Backend(args, prepared)
        backend.clock = Clock(lambda: torch.cuda.synchronize(torch.device(args.device)))
        backend.source_times, backend.codec_warmed = {}, False
        backend.validate_logits = lambda logits: require(torch.isfinite(logits).all().item(), 'Nonfinite next-token logits')
        props = torch.cuda.get_device_properties(torch.device(args.device))
        manifest.update(actual_gpu=dict(name=props.name, total_memory_bytes=props.total_memory, cuda_runtime=torch.version.cuda),
            counters=backend.counts, model_tokenizer_provenance=backend.metadata,
            software={p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft')})
        stage = 'warmup'; backend.audit_workload(prepared.episodes)
        manifest['model_inference_performed'] = True
        warmup(backend, prepared, backend.clock)
        stage = 'source_build'
        # Reuse C8's physical constructors and byte-LRU admission transitions;
        # only experiment-local timing boundaries are added, with no global patch.
        caches = {c8.condition(b, p): TimedCache(b, p, backend.storages[p], backend.clock)
                  for b in latency.BUDGETS for p in latency.POLICIES}
        for index, episode in enumerate(prepared.episodes):
            entries = backend.build_source(episode)
            for name, cache in caches.items():
                entry = entries[cache.policy]
                rep = c8run.representation_check(cache.policy, entry, cache.storage)
                require(rep['resident_bytes'] == prepared.sizes[cache.policy][episode['episode_id']], 'Actual source bytes differ from C8')
                result = cache.admit(episode, entry)
                expected = prepared.expected[name]['admissions'][index]
                require(all(result[k] == expected[k] for k in result), 'C9 admission/eviction differs from C8')
                measured = backend.source_times[episode['episode_id']]
                source.append(dict(episode_id=episode['episode_id'], episode_index=index, condition=name, policy=cache.policy,
                    source_forward_ms=measured['source_forward_ms'], source_capture_ms=measured['source_capture_ms'],
                    storage_encode_ms=measured['encode_ms'][cache.policy], cache_admission_ms=cache.admission_times[episode['episode_id']],
                    source_forward_shared_across_conditions=True, encoded_representation_shared_across_budgets=True,
                    admitted=result['admitted'], resident_bytes=result['resident_bytes_after'],
                    entry_bytes=rep['resident_bytes'], raw_q_resident_after_insert=rep['raw_q_resident_after_insert'],
                    raw_kv_resident_after_insert=rep['raw_kv_resident_after_insert'], shared_profile_bytes_charged_per_entry=0))
            del entries
        require(backend.counts['source_forward_count'] == 32, 'Exactly32 source captures required')
        for name, cache in caches.items():
            require(cache.lru.snapshot() == prepared.expected[name]['residency']['after_admission'], 'C9 resident sources differ from C8')
        stage = 'target_prefill_timing'
        evaluate(backend, prepared, caches, backend.clock, raw,
            lambda pairs: resolve_cuda_event_pairs(pairs, synchronize=False))
        require(all(backend.counts[k] == value for k, value in manifest['expected_execution_counts'].items()),
            'Unexpected source/prompt/decode count or forbidden fitting/transport/quality execution')
        provenance.verify_files(prepared.input_hashes)
        latency.per_case(raw, prepared.episodes, prepared.expected, prepared.tie_tolerance_ms)
        manifest.update(status='COMPLETE', c8_hit_vectors_reproduced=True, latency_decomposition_consistent=True,
            storage_representations_verified=True,
            recommendation='C9_A_READY_FOR_NETWORK_ACCOUNTING', **{k: backend.counts[k] for k in c8.SAFETY})
        latency.write_reports(root, raw, source, prepared, manifest)
    except Exception as exc:
        manifest.update(status='INVALID', failure_stage=stage, failure_type=type(exc).__name__, failure_message=str(exc),
            recommendation='C9_A_LOCAL_LATENCY_NEEDS_REVIEW')
        if backend is not None: manifest.update({k: backend.counts[k] for k in c8.SAFETY})
        try:
            latency.write_reports(root, raw, source, prepared, manifest)
        except Exception as reporting:
            manifest['reporting_failure'] = str(reporting)
            (root/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n')
        raise
