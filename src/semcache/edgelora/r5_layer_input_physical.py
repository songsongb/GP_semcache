"""Isolated opt-in verified non-prefix owner. Historical D5 modes are untouched.

The native contract consumes cached rows only after paid full-shape output-byte
verification. Sparse work skipping is available solely in scalar CPU fixtures.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import time

from .communication import require, digest, digest_bytes, canonical, Channel
from .r3a_tiny import f32
from .r3b_fixture import ArrayTensor
from .r5_i5_contracts import validate_request
from .r5_mstage_bridge import BoundedCache
from semcache.cache.metric_manager import CacheMetricManager
from .r5_sparse_qkv import blob, gather, scatter
from .r5_subsequence_physical import PhysicalOwner, entry_proof, tensor_proof
from .r5_subsequence_integrity import RequestValidation, SCOPED
from .r5_subsequence_profile import call
from .r5_layer_input_arithmetic import ScalarFixtureArithmetic, NativeReferenceArithmetic, finite

MODE = 'LAYER_INPUT_BYTES_VERIFIED_V1'
SCHEMA = 'D5_LAYER_INPUT_PHYSICAL_V1'


def category(entry, req, start, end):
    prefix = 0
    for a, b in zip(entry.qkv_metadata['source_token_ids'], req['token_ids']):
        if a != b: break
        prefix += 1
    if entry.positions == (start, end): return 'A' if end <= prefix else 'B'
    return 'D' if end <= prefix else 'C'


@dataclass(frozen=True)
class InputBytes:
    layer: int
    shape: tuple
    dtype: str
    data: bytes

    def proof(self):
        return dict(shape=list(self.shape), dtype=self.dtype, sha256=digest_bytes(self.data))


class VerifiedValidation(RequestValidation):
    def _base(self, entry):
        super()._base(entry)
        if entry.qkv_metadata.get('provenance_mode') == MODE:
            self.owner.validate_input_storage(entry)


class VerifiedCache(BoundedCache):
    """Same F/A/I/S and entry cap; account for new authenticated input storage."""
    def __init__(self, owner, capacity):
        super().__init__(capacity); self.owner = owner

    def insert(self, candidate, *args, materialize=None, on_event=None, **kwargs):
        owner = self.owner; a = owner.active
        require(a is not None and a['mode'] == MODE and owner.completion_ack is not None, 'verified admission before completion ACK')
        # Conservative root-only provenance: a request that consumed any cached
        # projection cannot create a new source, even for a different window.
        if a['consumed']:
            owner.trace(dict(kind='VERIFIED_ADMISSION_SKIPPED', reason='TRANSITIVE_SOURCE_FORBIDDEN', key=candidate.key))
            return False
        positions = list(range(*candidate.positions)); snapshots = tuple(
            InputBytes(l, (1, 3, owner.bridge.identity['hidden_size']), str(a['inputs'][l].dtype),
                       blob(gather(a['inputs'][l], positions))) for l in range(owner.bridge.identity['layers']))
        extra = dict(schema=SCHEMA, mode=MODE, arithmetic=owner.arithmetic.metadata(),
                     source_request_sha256=a['req']['request_sha256'], completion_sha256=owner.completion_ack,
                     source_root='COMPLETED_REQUEST_WITH_NO_CACHED_CONSUMPTION',
                     qkv_input_contract='ACTUAL_FULL_INPUT_BYTES_EQUAL_ALL_THREE_ROLES', transitive_reuse=False)
        candidate.qkv_metadata['layer_input_verified'] = extra
        overhead = sum(len(s.data) for s in snapshots)+len(canonical(extra))
        candidate.size_bytes += overhead
        def owned():
            entry = materialize()
            entry.size_bytes = candidate.size_bytes
            owner.input_storage[entry.key] = snapshots
            owner.input_bindings[entry.key] = (entry, snapshots)
            return entry
        def event(kind, entry, score):
            if kind == 'EVICT':
                owner.input_storage.pop(entry.key, None); owner.input_bindings.pop(entry.key, None)
            if on_event: on_event(kind, entry, score)
        return super().insert(candidate, *args, materialize=owned, on_event=event, **kwargs)


class CompletionChannel:
    """Bind actual authenticated UD completion READY to this owner's request.

    Tests may use a clearly labeled fixture channel only for fixture requests.
    The original finish_request() validates the same control digest again.
    """
    def __init__(self, channel, owner):
        require(isinstance(channel, Channel) or owner.active['req']['TEST_FIXTURE_ONLY'] and
                getattr(channel, 'TEST_FIXTURE_ONLY', False), 'authenticated Channel required')
        self.channel = channel; self.owner = owner; self.pending = None
        self.request_sha = owner.active['req']['request_sha256']

    def __getattr__(self, name): return getattr(self.channel, name)

    def send(self, h, payload=b''):
        try: return self._send(h, payload)
        except BaseException:
            self.owner.abort(); raise

    def _send(self, h, payload=b''):
        control = h.get('metadata', {}).get('sparse_complete')
        if control is not None:
            require(self.pending is None and self.owner.active is not None and not payload and
                    control['request'] == self.owner.active['req'] and control['request']['request_sha256'] == self.request_sha,
                    'completion request/epoch differs')
            from .r4c_workload import termination
            require(termination(control['request']['generation_policy'], control['generated_token_ids']) == control['termination'] and
                    control['termination'] is not None, 'unfinished verified source')
            self.pending = (h['request_id'], digest(control))
        return self.channel.send(h, payload)

    def recv(self, **expected):
        try: return self._recv(**expected)
        except BaseException:
            self.owner.abort(); raise

    def _recv(self, **expected):
        h, payload, receipt = self.channel.recv(**expected)
        if self.pending is not None:
            rid, proof = self.pending
            require(h['kind'] == 'READY' and h['request_id'] == rid and h['layer_id'] == -1 and not payload and
                    h['metadata'] == dict(completion_sha256=proof), 'verified completion ACK differs')
            require(self.owner.active is not None and self.owner.active['req']['request_sha256'] == self.request_sha,
                    'stale completion ACK')
            self.owner.completion_ack = proof; self.pending = None
        return h, payload, receipt


class LayerInputOwner(PhysicalOwner):
    def __init__(self, identity, registry, *, arithmetic, capacity=512*2**20, trace=lambda r: None, evidence=None):
        require(type(arithmetic) in (ScalarFixtureArithmetic, NativeReferenceArithmetic), 'unsupported verified arithmetic contract')
        require(arithmetic.identity == identity, 'arithmetic/physical identity differs')
        super().__init__(identity, registry, capacity=capacity, trace=trace, evidence=evidence, validation_mode=SCOPED)
        self.arithmetic = arithmetic; self.input_storage = {}; self.input_bindings = {}; self.completion_ack = None
        self.bridge.cache = VerifiedCache(self, capacity)
        self.bridge.metrics = CacheMetricManager(self.cache)
        self.validation = VerifiedValidation(self, SCOPED, entry_proof, tensor_proof)

    def begin(self, req, assignment, *, mode=None, allow_approximate=False, reuse=True):
        require(mode == MODE and not allow_approximate, 'explicit LAYER_INPUT_BYTES_VERIFIED_V1 required; APPROX forbidden')
        require(self.active is None and not self.validation.open, 'previous verified request active')
        report = self.bridge.begin(req, assignment); p = self.bridge.pending
        self.completion_ack = None
        self.active = dict(req=p['req'], cluster=p['cluster'], windows=p['windows'], selected=p['selected'] if reuse else [],
            mode=MODE, captures={}, inputs={}, records=[], consumed=set(), seen=set(), role_inputs={}, decisions=[])
        self.validation.start(p['req'], self.active['selected'])
        report['verified_mode'] = MODE
        report['authorization'] = 'RESIDENT_CANDIDATES_ONLY; no position or token match is a grant'
        self.active['selection'] = deepcopy(report)
        return report

    def completion_channel(self, channel):
        require(self.active is not None, 'no verified request for completion channel')
        return CompletionChannel(channel, self)

    def validate_input_storage(self, entry):
        binding = self.input_bindings.get(entry.key); snapshots = self.input_storage.get(entry.key)
        require(binding is not None and binding[0] is entry and binding[1] is snapshots and
                isinstance(snapshots, tuple) and len(snapshots) == self.bridge.identity['layers'], 'missing/replaced authenticated input storage')
        for layer, snapshot in enumerate(snapshots):
            require(type(snapshot) is InputBytes and type(snapshot.data) is bytes and snapshot.layer == layer and
                    snapshot.proof() == entry.qkv_metadata['input_proofs'][str(layer)], 'authenticated source input bytes/proof differ')

    def project(self, layer, role, x, native, session):
        try: return self._verified_project(layer, role, x, native, session)
        except BaseException:
            self.abort(); raise

    def _verified_project(self, layer, role, x, native, session):
        started = time.monotonic_ns(); a = self.active
        require(a is not None and role in 'qkv' and type(layer) is int and 0 <= layer < self.bridge.identity['layers'] and
                (layer, role) not in a['seen'], 'duplicate/invalid verified projection')
        expected = (len(a['seen'])//3, 'qkv'[len(a['seen']) % 3])
        require((layer, role) == expected, 'verified layer/role order differs')
        req = validate_request(a['req'])
        require(req == self.validation.request and req['physical_identity'] == self.bridge.identity,
                'current target request identity changed')
        self.arithmetic.check(req, layer, role, x, native, session)
        input_data = blob(x)
        if role == 'q':
            a['role_inputs'][layer] = input_data; a['inputs'][layer] = x.detach().clone()
        else:
            require(a['role_inputs'][layer] == input_data, 'Q/K/V actual input bytes differ; unsupported OPT contract')
        n = len(req['token_ids']); h = self.bridge.identity['hidden_size']; accepted = []; decisions = []
        for hit in a['selected']:
            entry = hit.entry; start, end = hit.window.start, hit.window.end
            # Authenticate even an ultimately ineligible provenance, so corruption
            # does not silently turn into ordinary fresh fallback.
            self.valid(entry); meta = entry.qkv_metadata; reason = None
            target = gather(x, list(range(start, end))); target_proof = tensor_proof(target)
            verified = meta.get('layer_input_verified'); equality = None
            if meta.get('provenance_mode') != MODE:
                reason = 'SOURCE_PROVENANCE_MODE_MISMATCH'
            elif (not verified or verified.get('schema') != SCHEMA or verified.get('transitive_reuse') is not False or
                    verified.get('source_root') != 'COMPLETED_REQUEST_WITH_NO_CACHED_CONSUMPTION'):
                reason = 'UNSUPPORTED_SOURCE_ROOT'
            elif verified['arithmetic'] != self.arithmetic.metadata():
                reason = 'SOURCE_ARITHMETIC_CONTRACT_MISMATCH'
            else:
                self.validate_input_storage(entry)
                snapshot = self.input_storage[entry.key][layer]
                equality = snapshot.proof() == target_proof and snapshot.data == blob(target)
                if not equality: reason = 'PER_LAYER_INPUT_BYTES_MISMATCH'
            row = dict(layer=layer, role=role, source_request_id=meta['source_request_id'], source_positions=list(entry.positions),
                target_positions=[start, end], category=category(entry, req, start, end), reason=reason,
                target_input_proof=target_proof, source_input_proof=meta.get('input_proofs', {}).get(str(layer)),
                arithmetic=self.arithmetic.metadata(), input_bytes_equal=equality, integrity='AUTHENTICATED')
            if reason is None:
                snapshot, integrity = self.validation.consume(entry, layer, role)
                accepted.append((hit, snapshot, integrity, row))
            decisions.append(row)
        guard = self.arithmetic.reference_guard
        candidate_used = {p for hit, _, _, _ in accepted for p in range(hit.window.start, hit.window.end)}
        compute = list(range(n)) if guard else [p for p in range(n) if p not in candidate_used]
        delta = call('verified_cpu_lora_inclusive', session.correction, layer, role, x, compute)
        base = call('verified_native_projection_inclusive', native, x if guard else gather(x, compute)) if compute else None
        if base is None: total = None
        elif isinstance(x, ArrayTensor):
            total = ArrayTensor([f32(v+d) for v, d in zip(base.values, delta.values)], base.shape)
        else: total = call('verified_add_cast', lambda: (base+delta).to(base.dtype))
        if total is not None: finite(total)
        self.arithmetic.check(req, layer, role, x, native, session)
        reused = []; consumed = []; used = set()
        for hit, snapshot, integrity, row in accepted:
            start, end = hit.window.start, hit.window.end
            require(list(snapshot.shape) == [1, 3, h] and str(snapshot.dtype) == str(x.dtype) and str(snapshot.device) == str(x.device),
                    'cached projection shape/dtype/device differs')
            if guard and blob(snapshot) != blob(gather(total, list(range(start, end)))):
                row['reason'] = 'FULL_SHAPE_OUTPUT_BYTES_MISMATCH'; row['output_bytes_equal'] = False
                continue
            self.valid(hit.entry)
            row['output_bytes_equal'] = True if guard else 'SCALAR_ROW_LOCAL_CONTRACT'
            row['reason'] = 'CONSUMED'; reused.append((start, snapshot)); used.update(range(start, end))
            a['consumed'].add((hit.entry.key, start, end))
            item = dict(key=hit.entry.key, source_request_id=hit.entry.qkv_metadata['source_request_id'], source_positions=list(hit.entry.positions),
                target_positions=[start, end], category=row['category'], tensor=tensor_proof(snapshot), integrity_validation=integrity,
                source_input_proof=row['source_input_proof'], target_input_proof=row['target_input_proof'])
            if self.evidence is not None: item['tensor'] = self.evidence.save(snapshot, req['request_id']+f'_l{layer}_{role}_consumed_{start}')
            consumed.append(item)
        output_fresh = [p for p in range(n) if p not in used]
        fresh = gather(total, output_fresh) if guard and output_fresh else total if not guard else None
        out = call('verified_output_scatter', scatter, fresh, output_fresh, reused, n, h, x)
        if guard: require(blob(out) == blob(total), 'assembled output differs from paid full-shape reference')
        a['captures'].setdefault(layer, {})[role] = out.detach().clone(); a['seen'].add((layer, role))
        record = dict(schema=SCHEMA, mode=MODE, layer=layer, role=role, request_id=req['request_id'],
            request_sha256=req['request_sha256'], fresh_positions=compute, output_fresh_positions=output_fresh,
            reused_positions=sorted(used), native_projection_rows=len(compute), cpu_lora_rows=len(compute),
            projection_rows_skipped=n-len(compute), reference_projection_rows=n if guard else 0,
            consumed=consumed, decisions=decisions, input=tensor_proof(x), output=tensor_proof(out),
            arithmetic=self.arithmetic.metadata(), numerical_equivalence='GPU_NUMERICAL_VALIDATION_NOT_RUN', past_kv_grant=False)
        if self.evidence is not None:
            tag = req['request_id']+f'_l{layer}_{role}'
            for name, tensor in (('input', x), ('output', out), ('fresh_base', base), ('fresh_delta', delta), ('full_shape_reference', total if guard else None)):
                record[name] = None if tensor is None else self.evidence.save(tensor, tag+'_'+name)
        record['verified_projection_wall_ms'] = (time.monotonic_ns()-started)/1e6
        a['records'].append(record); a['decisions'].extend(decisions)
        self.trace(dict(kind='LAYER_INPUT_VERIFIED_PROJECTION', **record))
        return out

    def complete(self, *, completed, attentions, valid_attention_mask=None):
        try:
            require(completed is True and self.active is not None and self.completion_ack is not None, 'verified source requires actual completion ACK')
            selection = self.active['selection']; records = self.active['records']
            result = super().complete(completed=True, attentions=attentions, valid_attention_mask=valid_attention_mask)
            counts = Counter({c: 0 for c in 'ABCD'})
            for row in records:
                for item in row['consumed']: counts[item['category']] += item['target_positions'][1]-item['target_positions'][0]
            result.update(schema=SCHEMA, arithmetic=self.arithmetic.metadata(), selection=selection,
                reused_projection_rows_by_category=dict(counts),
                native_projection_rows=sum(r['native_projection_rows'] for r in records),
                cpu_lora_rows=sum(r['cpu_lora_rows'] for r in records),
                projection_rows_skipped=sum(r['projection_rows_skipped'] for r in records),
                reference_projection_rows=sum(r['reference_projection_rows'] for r in records),
                resident_input_storage_bytes=sum(len(s.data) for snapshots in self.input_storage.values() for s in snapshots),
                GPU_numerical_validation='NOT_RUN', GPU_E2E='NOT_RUN')
            self.completion_ack = None
            return result
        except BaseException:
            self.abort(); raise

    def abort(self):
        super().abort(); self.completion_ack = None

    def dispose(self):
        super().dispose(); self.input_storage.clear(); self.input_bindings.clear(); self.completion_ack = None


def generate_verified(model, adapter, tokenizer, req, channel, owner, index, *, evidence=None, profiler=None):
    """Explicit new-mode adapter around the unchanged native generation lifecycle.

    No model load, runner or GPU opt-in is provided here. Caller must satisfy all
    existing native preparation/identity/resource/numerical evidence preflights.
    """
    require(type(owner) is LayerInputOwner, 'new verified owner required')
    # Generation calls begin itself. Bind the completion wrapper lazily after
    # that begin, while preserving the same authenticated Channel for projection.
    class LazyChannel:
        def __init__(self): self.bound = None
        def __getattr__(self, name): return getattr(channel, name)
        def send(self, *args, **kwargs):
            if self.bound is None: self.bound = owner.completion_channel(channel)
            return self.bound.send(*args, **kwargs)
        def recv(self, **kwargs): return self.bound.recv(**kwargs)
    from .r5_subsequence_generation import generate
    return generate(model, adapter, tokenizer, req, LazyChannel(), owner, index,
                    condition='SEMCACHE_SUBSEQUENCE', mode=MODE, allow_approximate=False,
                    evidence=evidence, profiler=profiler)
