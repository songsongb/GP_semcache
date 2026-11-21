"""Explicit CPU-only, process-local owner diagnostics; never a runtime candidate.

Temporary wrappers observe the unchanged Strict functions. Use in an isolated,
single-threaded CPU process only. No raw tensor, input, logits, or object addresses
are exported. Existing serialization and SHA results are observed, never cached
for authorization. Official receipts and production imports are untouched.
"""
from collections import Counter
from contextlib import contextmanager, ExitStack
from functools import wraps
import os
import threading
import time
import weakref
from unittest.mock import patch

from .communication import require
from .r5_subsequence_profile import Profiler


COMPONENTS = (
    'completion_impact_history', 'completion_input_proof',
    'completion_admission_decision', 'completion_materialization',
    'completion_admission_proof', 'completion_eviction_pbr',
    'completion_remaining',
)


class CPUOwnerDiagnostics(Profiler):
    """One diagnostic session per observer; normally one measured request.

    Wall bins partition the instrumented complete() wrapper, including observer
    work. Decision ends when the materializer starts; post-materialization insert
    bookkeeping, eviction and PBR belong to eviction_pbr. No policy is evaluated
    twice. A successful INSERT can subsequently evict itself.
    """
    def __init__(self):
        super().__init__(cuda_events=False)
        self.bins = {name: 0 for name in COMPONENTS}
        self.counts = Counter()
        self.candidates = []
        self.content = Counter()
        self.objects = {}
        self.pending_serialization = None
        self.stage = None
        self.completing = False
        self.used = False
        self.pending_input = Counter()
        self.completion_total_ns = 0

    def detail(self, record):
        # Base selection details contain token-bearing cache keys. Keep counts only.
        self.counts['detail_' + record.get('boundary', 'UNKNOWN') + ':' +
                    record.get('reason', 'OBSERVED')] += 1

    def _switch(self, stage):
        now = time.perf_counter_ns()
        if self.stage is not None:
            elapsed = now - self.tick
            self.bins[self.stage] += elapsed
            if self.stage == 'completion_input_proof':
                self.pending_input['wall_ns'] += elapsed
        self.stage, self.tick = stage, now

    @contextmanager
    def section(self, name):
        if not self.completing:
            yield
            return
        previous = self.stage
        self._switch(name)
        try:
            yield
        finally:
            self._switch(previous)

    @contextmanager
    def scope(self, name, **kwargs):
        # The decorator also covers old callables held by RequestValidation.
        category = ('completion_input_proof' if self.completing and
                    self.stage == 'completion_remaining' and name == 'tensor_proof'
                    else self.stage)
        with self.section(category), super().scope(name, **kwargs):
            yield

    @staticmethod
    def cpu(tensor):
        require(str(tensor.device) == 'cpu', 'owner diagnostics are CPU-only')

    def _wrap_section(self, fn, category):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with self.section(category):
                return fn(*args, **kwargs)
        return wrapped

    @contextmanager
    def observe(self, request_id):
        from . import r5_subsequence_physical as physical
        from .r5_subsequence_integrity import RequestValidation
        from .r5_mstage_bridge import BoundedCache
        from .r5_subsequence_generation import StreamingImpact
        from semcache.cache.metric_manager import CacheMetricManager, QueryImpactHistory
        require(not self.used and threading.active_count() == 1,
                'one diagnostic session in an isolated single-threaded process required')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
                os.environ.get('HIP_VISIBLE_DEVICES') == '', 'explicit CPU isolation required')
        self.used = True
        original_blob, original_hash = physical.blob, physical.digest_bytes
        original_gather = physical.gather
        original_complete, original_project = physical.PhysicalOwner.complete, physical.PhysicalOwner.project
        original_insert, original_full = BoundedCache.insert, RequestValidation.full

        def blob(tensor):
            self.cpu(tensor)
            self.counts['serialization_attempts'] += 1
            data = original_blob(tensor)
            self.counts['serialized_tensor_bytes'] += len(data)
            self.counts['serialization_successes'] += 1
            self.counts['cpu_serialization_calls'] += 1
            # Weak references disambiguate Python ID reuse without pinning tensors.
            self.pending_serialization = (weakref.ref(tensor), tuple(tensor.shape), str(tensor.dtype))
            if self.stage == 'completion_input_proof':
                self.pending_input['tensor_proofs'] += 1
                self.pending_input['serialized_bytes'] += len(data)
            return data

        def sha(data):
            result = original_hash(data)
            pending, self.pending_serialization = self.pending_serialization, None
            if pending is not None:
                ref, shape, dtype = pending
                signature = (shape, dtype, result)
                self.counts['same_content_fingerprint_repeats'] += int(self.content[signature] > 0)
                self.content[signature] += 1
                tensor = ref()
                if tensor is not None:
                    previous = self.objects.get(id(tensor))
                    self.counts['same_live_object_same_fingerprint_repeats'] += int(
                        previous is not None and previous[0]() is tensor and previous[1] == signature)
                    self.objects[id(tensor)] = (ref, signature)
                require(len(self.content) <= 500000 and len(self.objects) <= 500000,
                        'diagnostic fingerprint inventory exceeded')
            return result

        def gather(tensor, positions):
            self.cpu(tensor)
            category = ('completion_input_proof' if self.completing and
                        self.stage == 'completion_remaining' else self.stage)
            with self.section(category):
                self.counts['gather_calls'] += 1
                if self.completing:
                    self.counts['completion_gather_calls'] += 1
                return original_gather(tensor, positions)

        def project(owner, layer, role, x, native, session):
            self.cpu(x)
            return original_project(owner, layer, role, x, native, session)

        def complete(owner, **kwargs):
            require(not self.completing, 'nested completion forbidden')
            if owner.active is not None:
                for tensor in owner.active['inputs'].values():
                    self.cpu(tensor)
                for roles in owner.active['captures'].values():
                    for tensor in roles.values():
                        self.cpu(tensor)
            attentions = kwargs.get('attentions')
            if isinstance(attentions, (list, tuple)):
                for tensor in attentions:
                    if hasattr(tensor, 'device'):
                        self.cpu(tensor)
            self.completing = True
            start = time.perf_counter_ns()
            self._switch('completion_remaining')
            try:
                return original_complete(owner, **kwargs)
            finally:
                self._switch(None)
                self.completion_total_ns += self.tick - start
                self.completing = False

        def full(validation, entry, *, admission=False, fingerprints=None):
            self.counts['request_validation_full_calls'] += 1
            self.counts['admission_entry_full_calls' if admission else 'full_entry_full_calls'] += 1
            with self.section('completion_admission_proof' if admission else self.stage):
                return original_full(validation, entry, admission=admission, fingerprints=fingerprints)

        def insert(cache, entry, *args, materialize=None, on_event=None, **kwargs):
            if not self.completing:
                return original_insert(cache, entry, *args, materialize=materialize, on_event=on_event, **kwargs)
            self.counts['admission_candidates'] += 1
            row = dict(input_proof=dict(self.pending_input), inserted=False, evictions=0)
            self.pending_input.clear()
            prior = 'KEY_PRESENT' if entry.key in cache.entries else (
                'CAPACITY' if entry.size_bytes > cache.capacity_bytes else None)
            before = dict(self.bins)
            def materializer():
                with self.section('completion_materialization'):
                    self.counts['materialization_calls'] += 1
                    result = materialize()
                # GlobalCache has finished the admission decision at this barrier.
                self._switch('completion_eviction_pbr')
                return result
            def event(kind, resident, score):
                if kind == 'INSERT':
                    row['inserted'] = True
                    self.counts['admission_approved'] += 1
                elif kind == 'EVICT':
                    row['evictions'] += 1
                    self.counts['evictions'] += 1
                if on_event is not None:
                    return on_event(kind, resident, score)
            try:
                with self.section('completion_admission_decision'):
                    result = original_insert(cache, entry, *args,
                        materialize=materializer if materialize is not None else None, on_event=event, **kwargs)
                row['resident_after_insert'] = bool(result)
                row['outcome'] = ('RESIDENT' if result else 'APPROVED_THEN_EVICTED'
                                  if row['inserted'] else prior or 'POLICY')
                if not row['inserted']:
                    self.counts['admission_rejected'] += 1
                    self.counts['rejected_' + row['outcome']] += 1
                return result
            except BaseException:
                row['outcome'] = 'EXCEPTION'
                raise
            finally:
                row['stages_ms'] = {name: (self.bins[name] - before[name]) / 1e6
                                    for name in COMPONENTS if self.bins[name] != before[name]}
                require(len(self.candidates) < 4096, 'diagnostic candidate bound exceeded')
                self.candidates.append(row)

        with ExitStack() as stack:
            for target, name, replacement in (
                (physical, 'blob', blob), (physical, 'digest_bytes', sha),
                (physical, 'gather', gather), (physical.PhysicalOwner, 'project', project),
                (physical.PhysicalOwner, 'complete', complete), (BoundedCache, 'insert', insert),
                (RequestValidation, 'full', full),
                (physical, 'actual_attention_impact', self._wrap_section(physical.actual_attention_impact, 'completion_impact_history')),
                (StreamingImpact, 'values_for', self._wrap_section(StreamingImpact.values_for, 'completion_impact_history')),
                (QueryImpactHistory, 'append', self._wrap_section(QueryImpactHistory.append, 'completion_impact_history')),
                (CacheMetricManager, 'reused', self._wrap_section(CacheMetricManager.reused, 'completion_impact_history')),
                (CacheMetricManager, 'pbr', self._wrap_section(CacheMetricManager.pbr, 'completion_eviction_pbr')),
            ):
                stack.enter_context(patch.object(target, name, replacement))
            with self.activate(request_id):
                yield self

    def report(self):
        profile = self.finish()
        counts = dict(self.counts)
        for key in ('admission_candidates', 'admission_approved', 'admission_rejected',
                    'serialized_tensor_bytes', 'serialization_attempts', 'serialization_successes',
                    'materialization_calls', 'same_content_fingerprint_repeats',
                    'same_live_object_same_fingerprint_repeats', 'request_validation_full_calls',
                    'full_entry_full_calls', 'admission_entry_full_calls', 'gather_calls',
                    'completion_gather_calls', 'cpu_serialization_calls', 'evictions'):
            counts.setdefault(key, 0)
        for name in ('tensor_proof', 'entry_proof_all_layers'):
            counts[name + '_calls'] = sum(r['count'] for r in profile['components'] if r['component'] == name)
        counts['gpu_to_cpu_host_transfer_calls'] = 0  # CUDA rejected before serialization.
        return dict(schema='D5_L2D_CPU_OWNER_DIAGNOSTICS_V1', CPU_ONLY=True,
                    production_candidate=None, counts=counts, candidates=self.candidates,
                    incomplete_input_proof=dict(self.pending_input),
                    completion_ms={k: v / 1e6 for k, v in self.bins.items()},
                    completion_total_ms=self.completion_total_ns / 1e6,
                    boundary_start_unassigned_ms=(self.completion_total_ns - sum(self.bins.values())) / 1e6,
                    profile=profile,
                    limitations='temporary process-local wrappers; single CPU thread only; instrumented wall includes observer overhead; '
                    'fingerprint equality uses existing shape/dtype/SHA256, not independent byte comparison; '
                    'object identity uses live weak references; equal-content repeats do not establish redundant validation; '
                    'decision excludes post-materialization insertion bookkeeping, assigned to eviction_pbr; '
                    'impact_history covers reducer/value retrieval, history.append and reused; other glue is remaining')
