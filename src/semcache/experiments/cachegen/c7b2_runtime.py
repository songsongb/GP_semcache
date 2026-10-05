"""C7-B2 storage-only experiment wrapper; production C6/C2 remain unchanged."""
from contextlib import contextmanager
from types import SimpleNamespace

from . import c7b_q_profiles as qcodec
from .c7b_q_capture import require, sha
from .c6_runtime import Storage, make_c6_cache, RAW_ENTRY_BYTES
from .c6b2_runtime import Backend as C6Backend
from .c6b3_2_runtime import MultiwozBackend
from .c6_quality import validate_hit

MODES = ('STORAGE_KV_COMP_BASELINE', 'STORAGE_Q24_KV_COMP', 'STORAGE_Q32_KV_COMP')
CANDIDATES = ('Q24', 'Q32')


def mode_bins(mode):
    require(mode in MODES, 'C7-B2 accepts exactly baseline, Q24 and Q32 storage modes')
    return dict(zip(MODES, (None, 24, 32)))[mode]


def runtime_counts():
    return dict(runtime_q_profile_fit_count=0, runtime_storage_kv_cdf_fit_count=0,
                storage_decode_count=0, q_encode_count=0, q_decode_count=0,
                source_forward_count=0, target_greedy_forward_count=0,
                teacher_forced_forward_count=0, transport_encode_calls=0, transport_decode_calls=0)


@contextmanager
def forbid_fitting(counts, role):
    import sys
    require(role in ('q', 'kv'), 'Unknown storage role')
    previous = sys.getprofile()
    def guard(frame, event, arg):
        name = (getattr(arg, '__name__', '') if event == 'c_call' else frame.f_code.co_name).lower()
        if event in ('call', 'c_call') and (name in ('fit', 'fit_profile', 'fit_profiles', 'cdf_from_counts') or
                ('cdf' in name and any(w in name for w in ('fit', 'build', 'calculate', 'train')))):
            key = 'runtime_q_profile_fit_count' if role == 'q' else 'runtime_storage_kv_cdf_fit_count'
            counts[key] += 1
            raise RuntimeError('C7-B2 runtime fitting forbidden: '+name)
    sys.setprofile(guard)
    try:
        yield
    finally:
        sys.setprofile(previous)


class FrozenQ:
    """Read-only already-fitted profile. No fitting API is exposed."""
    def __init__(self, path, bins, expected_sha, backend, counts):
        require(bins in (24, 32), 'Only frozen Q24/Q32 profiles accepted')
        require(sha(path) == expected_sha, 'Q profile hash mismatch')
        self.backend, self.counts = backend, counts
        with forbid_fitting(counts, 'q'):
            self.profile = qcodec.QProfile.from_bytes(path.read_bytes(), backend)
        m = self.profile.metadata
        require((m['bins'], m['layers'], m['tokens'], m['hidden'], m['transform']) ==
                (bins, 32, 3, 2560, qcodec.TRANSFORM), 'Frozen Q profile contract mismatch')
        self.profile_bytes = path.stat().st_size
        self.sha256 = expected_sha

    def encode(self, q):
        with forbid_fitting(self.counts, 'q'):
            result = qcodec.encode(q, self.profile, self.backend)
        self.counts['q_encode_count'] += 1
        return result

    def decode(self, frame):
        with forbid_fitting(self.counts, 'q'):
            result = qcodec.decode(frame, self.profile, self.backend)
        self.counts['q_decode_count'] += 1
        return result


def compressed_entry_type(base):
    """Experiment-local subclass supplies correct physical accounting to C2."""
    class QCompressedEntry(base):
        @property
        def physical_tensor_bytes(self):
            return len(self.compressed_q_frame) + len(self.compressed_kv.bitstream)

        @property
        def raw_q_bytes(self):
            return self.size_bytes // 3

        @property
        def storage_accounting(self):
            # Preserve the C2 accounting API without dereferencing absent raw Q.
            return dict(raw_q_bytes=self.raw_q_bytes, raw_kv_bytes=self.size_bytes-self.raw_q_bytes,
                raw_qkv_bytes=self.size_bytes, stored_q_bytes=len(self.compressed_q_frame),
                compressed_kv_entry_bytes=len(self.compressed_kv.bitstream),
                encoded_kv_payload_bytes=self.compressed_kv.payload_bytes,
                kv_metadata_bytes=self.compressed_kv.local_metadata_bytes,
                actual_qkv_entry_bytes=self.physical_tensor_bytes, global_profile_bytes_charged_per_entry=0)
    return QCompressedEntry


class _TemporaryQInput:
    def __init__(self, resident, q):
        self.resident = resident
        self.q_tensors = {l: q[l:l+1] for l in range(32)}

    def __getattr__(self, name):
        return getattr(self.resident, name)


class StorageExperiment:
    """Injected C2 lookup codec; all K/V encoding/decoding calls stay in C6/C2."""
    def __init__(self, kv_storage, frozen_q, counts):
        self.kv = kv_storage
        self.q = frozen_q
        self.counts = counts
        self.mode = kv_storage.mode
        self.codec = self  # The actual GlobalCache lookup invokes decode_entry.

    def encode(self, episode, tensors):
        import torch
        with forbid_fitting(self.counts, 'kv'):
            resident, original = self.kv.encode(episode, tensors)
        raw = [sum(values[i].numel()*values[i].element_size() for values in tensors.values()) for i in range(3)]
        require(sum(raw) == RAW_ENTRY_BYTES and len(set(raw)) == 1, 'Frozen raw QKV size changed')
        sizes = dict(compressed_q_bitstream_bytes=0, compressed_q_local_metadata_bytes=0)
        if self.q is not None:
            q = torch.cat([resident.q_tensors[l] for l in range(32)], dim=0).cpu()
            frame, sizes = self.q.encode(q)
            resident.__class__ = compressed_entry_type(type(resident))
            resident.compressed_q_frame = frame
            resident.compressed_q_profile_sha256 = self.q.sha256
            resident.q_tensors = None  # Before admission: no raw resident Q.
            del q
        require(resident.tensors is None and resident.compressed_kv is not None, 'Raw resident KV forbidden')
        require((resident.q_tensors is not None) == (self.q is None), 'Resident Q ownership mismatch')
        kv_frame = len(resident.compressed_kv.bitstream)
        q_resident = raw[0] if self.q is None else len(resident.compressed_q_frame)
        total = q_resident + kv_frame
        baseline = raw[0] + kv_frame
        require(total == resident.physical_tensor_bytes, 'Resident allocation accounting mismatch')
        account = dict(raw_q_bytes=raw[0], raw_k_bytes=raw[1], raw_v_bytes=raw[2], raw_qkv_bytes=sum(raw),
            resident_raw_q_bytes=raw[0] if self.q is None else 0,
            compressed_q_bitstream_bytes=sizes['compressed_q_bitstream_bytes'],
            local_q_metadata_bytes=sizes['compressed_q_local_metadata_bytes'],
            compressed_q_frame_bytes=0 if self.q is None else q_resident,
            compressed_kv_frame_bytes=kv_frame, local_kv_metadata_bytes=resident.compressed_kv.local_metadata_bytes,
            total_resident_qkv_bytes=total, q_compression_ratio=raw[0]/q_resident,
            whole_qkv_compression_ratio=sum(raw)/total, whole_qkv_byte_reduction_percentage=100*(1-total/sum(raw)),
            incremental_resident_byte_reduction_vs_kv_baseline=baseline-total,
            incremental_resident_byte_reduction_percentage_vs_kv_baseline=100*(1-total/baseline),
            raw_q_resident_after_insert=self.q is None, shared_profile_bytes_charged_per_entry=0)
        require(original['resident_kv_frame_bytes'] == kv_frame, 'C6 KV accounting changed')
        return resident, account

    def decode_entry(self, resident):
        if self.q is None:
            with forbid_fitting(self.counts, 'kv'):
                view = self.kv.codec.decode_entry(resident)
        else:
            require(resident.q_tensors is None and resident.tensors is None,
                    'Compressed Q must not retain raw tensors')
            require(resident.compressed_q_profile_sha256 == self.q.sha256, 'Resident Q profile changed')
            temporary = _TemporaryQInput(resident, self.q.decode(resident.compressed_q_frame))
            with forbid_fitting(self.counts, 'kv'):
                view = self.kv.codec.decode_entry(temporary)
            # The view references the real resident, never the temporary Q input.
            view.resident = resident
        self.counts['storage_decode_count'] += 1
        return view

    def validate_decoded(self, resident, view):
        import torch
        if self.q is None:
            return self.kv.validate_decoded(resident, view)
        require(view is not resident and view.resident is resident and resident.q_tensors is None
                and resident.tensors is None and len(view.tensors) == 32, 'Invalid Q-compressed lookup view')
        for values in view.tensors.values():
            require(len(values) == 3 and all(t.shape == (1, 3, 2560) and t.dtype == torch.float16
                and torch.isfinite(t).all().item() for t in values), 'Invalid decoded TOTAL QKV')


def projection_audit(audit, layers, sequence_length, hits):
    selected = [p for h in hits for p in range(h.window.start, h.window.end)]
    expected = 3*len(hits)
    fresh = [p for p in range(sequence_length) if p not in selected]
    require(len(audit.records) == 3*layers and audit.reused_mask.nonzero().flatten().tolist() == selected,
            'QKV reuse mask changed')
    records = []
    for layer in range(layers):
        record = dict(layer=layer)
        for role in 'qkv':
            r = audit.records[layer, role]
            require(r['reused_projection_rows'] == expected and r['native_positions'] == fresh and
                    r['native_projection_rows'] == len(fresh) and r['reconstructed_projection_rows'] == 0,
                    'Native QKV hit-row skipping/transport contract changed')
            record[role+'_reused_projection_rows'] = expected
            record[role+'_native_projection_rows_skipped'] = sequence_length-r['native_projection_rows']
        records.append(record)
    return dict(same_reuse_mask_qkv=True, hit_positions=selected, per_layer=records)


class Backend(C6Backend):
    # Retain the exact established incremental greedy and canonical teacher protocol.
    greedy = C6Backend.greedy
    teacher_logits = MultiwozBackend.teacher_logits

    def __init__(self, args, profiles, q_backend):
        import torch
        from .c6b_runtime import load_base
        from semcache.models.task_adapters import load_two_task_users
        from semcache.models.model_adapter import OPTModelAdapter
        require(args.device.startswith('cuda:') and torch.cuda.is_available(), 'SERAPH single-GPU execution required')
        self.args, self.counts = args, runtime_counts()
        kv = Storage(args)
        self.storages = {MODES[0]: StorageExperiment(kv, None, self.counts)}
        for mode in MODES[1:]:
            bins = mode_bins(mode)
            path, expected_sha = profiles['Q'+str(bins)]
            frozen = FrozenQ(path, bins, expected_sha, q_backend, self.counts)
            self.storages[mode] = StorageExperiment(kv, frozen, self.counts)
        for storage in self.storages.values():
            make_c6_cache('STORAGE_KV_COMP', storage)
        model, self.tokenizer, self.metadata = load_base(args)
        self.model = load_two_task_users(model, args.adapter_root/'user_a', args.adapter_root/'user_b')
        self.adapter = OPTModelAdapter(self.model)
        self.candidates = {'unused': []}  # No classification scoring or transport path.
        self.reuse_audits = []
        self.storage_source = kv.source
        self.shared_profile_bytes = dict(KV=kv.codec.profile_bytes,
            **{c: self.storages[MODES[i+1]].q.profile_bytes for i, c in enumerate(CANDIDATES)})

    def forward(self, ids, user, hits, transport, *, use_cache=False, past=None, capture=False):
        import torch
        from semcache.models.task_adapters import activate_task_user
        from semcache.edgelora.mixed_projection import mixed_projection_path
        require(transport is False, 'C7-B2 transport compression forbidden')
        activate_task_user(self.model, user)
        require(not any(p.requires_grad for p in self.model.parameters()), 'Evaluation parameters must be frozen')
        inputs = dict(input_ids=torch.tensor([ids], device=self.args.device), use_cache=use_cache)
        if past is not None:
            inputs['past_key_values'] = past
        with torch.inference_mode():
            require(hits is not None, 'Every executed C7-B2 mode uses SemCache')
            with mixed_projection_path(self.adapter, user, hits, len(ids)) as audit:
                output = self.model(**inputs)
            evidence = projection_audit(audit, len(self.adapter.layers), len(ids), hits)
            if hits:
                self.reuse_audits.append(evidence)
        return output, audit.projections if capture else None

    def prepare(self, episode, mode):
        mode_bins(mode)
        self.storage = self.storages[mode]
        original = self.args.max_sequence_length
        self.args.max_sequence_length = self.model.config.max_position_embeddings
        try:
            context = super().prepare(episode, 'STORAGE_KV_COMP')
        finally:
            self.args.max_sequence_length = original
        context.mode = mode
        require(context.transport is False and context.accounting['raw_q_resident_after_insert'] == (mode == MODES[0]),
                'Physical residency/transport mode changed')
        return context

    def event_identity(self, context, episode):
        mode_bins(context.mode)
        cache, entry = context.cache, context.entry
        require(len(context.hits) == 1 and cache.entries.get(entry.key) is entry and len(cache.entries) == 1
                and entry.frequency == 1 and entry.size_bytes == RAW_ENTRY_BYTES
                and cache.charged_cache_bytes == cache.logical_cache_bytes == RAW_ENTRY_BYTES
                and cache.hits == 1 and cache.misses == 0, 'Frozen logical admission/HIT changed')
        require(entry.tensors is None and entry.compressed_kv is not None and
                (entry.q_tensors is not None) == (context.mode == MODES[0]), 'Resident payload changed')
        return validate_hit(episode, context.hits[0])
