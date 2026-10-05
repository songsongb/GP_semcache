"""C7-A: isolated, CPU-only resident TOTAL Q audit; no production mutations.

The fixture runs the installed CacheEntry/GlobalCache/exact matcher and mixed
projection implementation with tiny PEFT Linear modules, then causal attention.
It does not instantiate a language model or import the CUDA transport codec.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import importlib.metadata
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

ROLES = ('RESIDENT_TOTAL_Q', 'TRANSPORT_LORA_Q_DELTA', 'FRESH_TARGET_Q',
         'DIAGNOSTIC_Q', 'ACCOUNTING_ONLY_Q', 'UNKNOWN_Q_ROLE')
NEGATIVES = ('q_required_for_semantic_lookup', 'q_required_for_exact_w3',
             'q_required_for_cache_key', 'q_required_for_attention_reuse',
             'q_required_for_kv_reconstruction')
COLUMNS = ('sequence', 'q_role', 'operation', 'object_or_field', 'symbol', 'file',
           'phase', 'post_insert', 'hit_path', 'value_consumed', 'notes')


class Trace:
    """Reads are tuple accesses; consumers are observed tensor operations.

    Counts are per projection tensor/event, not scalar elements. Accounting
    access counts as a read but never as a semantic value consumer.
    """
    def __init__(self):
        self.rows = []
        self.phase = 'source'
        self.post_insert = False
        self.hit_path = False
        self.counts = {f'resident_{r}_{op}_count': 0 for r in 'qkv' for op in ('write', 'read')}
        self.counts.update(resident_q_hit_read_count=0, resident_q_value_consumer_count=0,
                           transport_q_delta_encode_count=0, transport_q_delta_decode_count=0)

    def event(self, role, operation, field, symbol, file=__file__, consumed=False, notes=''):
        if role not in ROLES:
            raise ValueError(role)
        self.rows.append(dict(sequence=len(self.rows)+1, q_role=role, operation=operation,
            object_or_field=field, symbol=symbol, file=str(file), phase=self.phase,
            post_insert=self.post_insert, hit_path=self.hit_path,
            value_consumed=consumed, notes=notes))

    def resident(self, component, operation, symbol, file=__file__, consumed=False):
        if operation in ('write', 'read'):
            self.counts[f'resident_{component}_{operation}_count'] += 1
        if component == 'q':
            if operation == 'read' and self.hit_path:
                self.counts['resident_q_hit_read_count'] += 1
            if consumed and self.post_insert:
                self.counts['resident_q_value_consumer_count'] += 1
            role = 'ACCOUNTING_ONLY_Q' if self.phase == 'accounting' else 'RESIDENT_TOTAL_Q'
            self.event(role, operation, f'entry.tensors[0][{ "qkv".index(component)}]',
                       symbol, file, consumed)
        # K/V counters have their own component names; never label them Q.

    def transport(self, operation):
        if operation not in ('encode', 'decode'):
            raise ValueError(operation)
        self.counts[f'transport_q_delta_{operation}_count'] += 1
        self.event('TRANSPORT_LORA_Q_DELTA', operation, 'delta', 'transport.'+operation)

    def wrap_transport(self, role, encode, decode):
        """Optional wrappers for owned callables; never import or fit a codec.

        The CPU fixture intentionally leaves these unused: CUDA transport is
        outside its scope. Only successful Q-delta calls increment Q counters.
        """
        if role not in 'qkv' or len(role) != 1:
            raise ValueError(role)
        def wrap(operation, function):
            def call(*args, **kwargs):
                result = function(*args, **kwargs)
                if role == 'q':
                    self.transport(operation)
                return result
            return call
        return wrap('encode', encode), wrap('decode', decode)


class ObservedQKV(tuple):
    def __new__(cls, values, trace):
        obj = super().__new__(cls, values)
        obj.trace = trace
        return obj

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        caller = inspect.currentframe().f_back
        self.trace.resident('qkv'[index], 'read', getattr(caller.f_code, 'co_qualname', caller.f_code.co_name),
                            caller.f_code.co_filename)
        return super().__getitem__(index)

    def __iter__(self):
        caller = inspect.currentframe().f_back
        for i in range(len(self)):
            self.trace.resident('qkv'[i], 'read',
                getattr(caller.f_code, 'co_qualname', caller.f_code.co_name), caller.f_code.co_filename)
            yield tuple.__getitem__(self, i)


def recommend(evidence, *, runtime_dependency=False, counterfactual_changed=False,
              contradictory=False):
    """Unknown negatives never authorize dropping Q; runtime wins over assumptions."""
    if contradictory:
        return 'INCONCLUSIVE'
    if runtime_dependency:
        if evidence.get('q_value_used_after_insert') is False:
            return 'INCONCLUSIVE'
        return 'COMPRESS_RESIDENT_Q' if counterfactual_changed else 'INCONCLUSIVE'
    if any(evidence.get(k) is True for k in (
            'q_value_used_after_insert', 'q_value_required', 'q_required_for_admission',
            'q_required_for_eviction', 'q_required_for_lora_recombination')):
        return 'INCONCLUSIVE'
    if (evidence.get('q_written_to_resident_cache') is True
            and (evidence.get('q_read_on_hit') is False or evidence.get('q_value_used_after_insert') is False)
            and all(evidence.get(k) is False for k in NEGATIVES)
            and not counterfactual_changed):
        return 'DROP_RESIDENT_Q'
    return 'INCONCLUSIVE'


def fixture(variant='normal'):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from peft.tuners.lora.layer import Linear
    from peft import LoraConfig
    from semcache.cache.cache_entry import CacheEntry
    from semcache.cache.global_cache import GlobalCache
    from semcache.semantic.matcher import ExactTokenMatcher
    from semcache.semantic.subsequence import Subsequence
    from semcache.semantic.hit_selection import CacheHit, select_nonoverlapping
    from semcache.edgelora.mixed_projection import mixed_projection_path
    from .c6_quality import validate_hit

    if variant not in ('normal', 'zero', 'none'):
        raise ValueError(variant)
    trace = Trace()
    # Deterministic literal weights; no training, model loading, or RNG dependency.
    modules = {}
    for i, role in enumerate('qkv'):
        config = {'config': LoraConfig(r=2, lora_alpha=2)} if 'config' in inspect.signature(Linear).parameters else {}
        module = Linear(torch.nn.Linear(4, 4, bias=False), 'audit', r=2, lora_alpha=2, **config)
        with torch.no_grad():
            module.get_base_layer().weight.copy_(torch.eye(4)*(i+1)/3)
            module.lora_A['audit'].weight.fill_(0.03)
            module.lora_B['audit'].weight.fill_(0.07)
        module.eval()
        modules[role] = module
    adapter = SimpleNamespace(layers=[None], projection_modules=lambda layer: modules)
    source = torch.tensor([[[1., 0., 2., -1.], [0., 2., -1., 1.],
                            [2., -1., 0., 1.], [1., 1., -1., 0.]]])
    with mixed_projection_path(adapter, 'audit', [], 4) as capture:
        for module in modules.values():
            module(source)
    trace.event('RESIDENT_TOTAL_Q', 'materialize', "capture.projections[0]['q']",
                'mixed_projection_path.forward', inspect.getfile(inspect.unwrap(mixed_projection_path)),
                notes='source TOTAL Q, before insertion')
    tensors = {0: tuple(capture.projections[0][r][:, :3].to(torch.float16) for r in 'qkv')}
    entry = CacheEntry.from_tensors(0, (10, 11, 12), (0, 3), tensors)
    entry.qkv_metadata = dict(component_scope='total_qkv', source_user='audit', source_id='synthetic')
    original = entry.tensors[0]
    cache = GlobalCache(entry.size_bytes)
    assert cache.insert(entry)
    for r in 'qkv':
        trace.resident(r, 'write', 'CacheEntry.from_tensors / GlobalCache.insert',
                       inspect.getfile(CacheEntry.from_tensors))
    trace.post_insert = True
    trace.phase = 'counterfactual'
    changed_q = original[0] if variant == 'normal' else (torch.zeros_like(original[0]) if variant == 'zero' else None)
    entry.tensors[0] = ObservedQKV((changed_q, original[1], original[2]), trace)
    trace.event('RESIDENT_TOTAL_Q', 'counterfactual', 'entry.tensors[0][0]', 'fixture',
                notes=variant+'; audit-owned entry only, logical size preserved')
    # Observe genuine accounting reads separately; None intentionally fails schema.
    trace.phase = 'accounting'
    accounting_error = None
    try:
        physical = entry.physical_tensor_bytes
    except (AttributeError, TypeError) as exc:
        physical = None
        accounting_error = type(exc).__name__+': '+str(exc)
    trace.phase = 'lookup'
    trace.hit_path = True
    window = Subsequence((10, 11, 12), 1, 4)
    view = cache.lookup(ExactTokenMatcher().key(0, window), record_reuse=False)
    hit = CacheHit(window, view)
    episode = dict(cluster=0, token_ids=[10, 11, 12], source_start=0,
                   target_start=1, source_user='audit', source_id='synthetic')
    identity = validate_hit(episode, hit)
    hits, mask = select_nonoverlapping([hit], 5)
    cache.record_reuse(entry)
    result = dict(variant=variant, lookup_success=view is entry, selected_cache_entry=list(entry.key),
        logical_safety_assertion_passed=True, logical_event_hash=identity,
        logical_size_bytes=entry.size_bytes, physical_bytes=physical, accounting_error=accounting_error,
        schema_field_required=None, reuse_success=False, error=None)
    result['raw_resident_bytes'] = {r: t.numel()*t.element_size() for r, t in zip('qkv', original)}
    trace.phase = 'reuse'

    class Consumers(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            # Direct identity, not tensor equality. Observe operations reading the
            # resident object; propagated output is subsequently tested causally.
            def contains(value, target):
                if value is target:
                    return True
                if isinstance(value, (tuple, list)):
                    return any(contains(v, target) for v in value)
                if isinstance(value, dict):
                    return any(contains(v, target) for v in value.values())
                return False
            if changed_q is not None and contains((args, kwargs), changed_q):
                trace.resident('q', 'consume', 'mixed_projection_path.forward / '+str(func),
                               inspect.getfile(inspect.unwrap(mixed_projection_path)), consumed=True)
            return func(*args, **(kwargs or {}))

    target = torch.cat((source[:, 3:], source), dim=1)
    outputs = {}
    # Fresh Q callback uses native projection; transport codec deliberately absent.
    def fresh(layer, role, module, hidden, native):
        out = native(hidden)
        if role == 'q':
            trace.event('FRESH_TARGET_Q', 'project', 'fresh target rows', 'fixture.fresh', consumed=True)
        return out
    try:
        with Consumers(), mixed_projection_path(adapter, 'audit', hits, 5, fresh_projection=fresh):
            # K/V first permits their independent comparison even if Q is None.
            for role in 'kvq':
                outputs[role] = modules[role](target)
        q, k, v = (outputs[r] for r in 'qkv')
        scores = (q @ k.transpose(-1, -2)) / 2
        scores = scores.masked_fill(torch.ones(5, 5, dtype=torch.bool).triu(1), float('-inf'))
        attention = scores.softmax(-1)
        output = attention @ v
        result.update(reuse_success=True, attention=attention.tolist(), target_reuse_output=output.tolist(),
                      q_projection=q.tolist(), reused_q_matches_resident=torch.equal(q[:, 1:4], changed_q.float()))
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
    result['kv'] = {r: outputs[r].tolist() for r in 'kv' if r in outputs}
    result['counters'] = trace.counts
    return result, trace


def counterfactuals():
    results, traces = {}, {}
    for variant in ('normal', 'zero', 'none'):
        results[variant], traces[variant] = fixture(variant)
    normal = results['normal']
    for variant, row in results.items():
        row['changes_vs_normal'] = {
            'lookup_success': row['lookup_success'] != normal['lookup_success'],
            'selected_cache_entry': row['selected_cache_entry'] != normal['selected_cache_entry'],
            'reconstructed_kv': row['kv'] != normal['kv'],
            'target_reuse_output': (row.get('target_reuse_output') != normal.get('target_reuse_output')) if row['reuse_success'] else None,
            'logical_safety_assertion': row['logical_safety_assertion_passed'] != normal['logical_safety_assertion_passed'],
            'reuse_execution_failed': not row['reuse_success']}
        row['schema_field_required'] = ("'NoneType' object has no attribute 'shape'" in (row['error'] or '')) if variant == 'none' else None
        def max_delta(a, b):
            if isinstance(a, list):
                return max(max_delta(x, y) for x, y in zip(a, b))
            return abs(a-b)
        row['max_abs_output_delta'] = max_delta(row['target_reuse_output'], normal['target_reuse_output']) if row['reuse_success'] else None
    return dict(status='COMPLETE', variants=results,
        scope='tiny real raw cache and mixed_projection_path; single-head causal attention; no model inference',
        field_presence_vs_value='None failure is schema evidence; zero-Q output difference is independent value-dependency evidence'), traces


# Explicit reviewed symbols, not grep-derived claims. Missing external implementation
# remains unknown; caller contracts do not substitute for inspecting the codec.
STATIC = [
 ('semcache_engine.py', 'SemCacheEngine.query', 'capture/slice/insert', 'RESIDENT_TOTAL_Q', False,
  'Mixed source totals captured; blocks passed to CacheEntry.from_tensors after admission. Target calls the same mixed path.'),
 ('cache/cache_entry.py', 'CacheEntry.from_tensors', 'own', 'RESIDENT_TOTAL_Q', True,
  'Detached compact Q/K/V copies; Q is entry.tensors[layer][0]; all three validated and charged.'),
 ('cache/cache_entry.py', 'CacheEntry.physical_tensor_bytes', 'account', 'ACCOUNTING_ONLY_Q', False,
  'Enumerates Q storage identity/bytes, not numerical values. None breaks storage schema.'),
 ('cache/global_cache.py', 'GlobalCache.lookup', 'retrieve', 'RESIDENT_TOTAL_Q', False,
  'Returns whole entry by key; does not access its Q field.'),
 ('cache/global_cache.py', 'GlobalCache.insert', 'policy', 'ACCOUNTING_ONLY_Q', False,
  'Admission/eviction use size_bytes, frequency, impact and age; no direct Q values. Raw Q bytes contribute to size.'),
 ('semantic/matcher.py', 'ExactTokenMatcher.key', 'key', 'RESIDENT_TOTAL_Q', False,
  'Cluster and exact token tuple only, no Q dependency.'),
 ('semantic/hit_selection.py', 'select_nonoverlapping', 'select', 'RESIDENT_TOTAL_Q', False,
  'Window, utility, entry.key and token identity; no Q values.'),
 ('edgelora/mixed_projection.py', 'mixed_projection_path', 'copy into target projection', 'RESIDENT_TOTAL_Q', True,
  "forward reads hit.entry.tensors[layer]['qkv'.index(name)]; source.shape then source.to(output) copied into target rows INCLUDING q."),
 ('edgelora/mixed_projection.py', 'mixed_projection_path', 'fresh subset', 'FRESH_TARGET_Q', True,
  'Native/callback projection on unmatched rows; hits bypass fresh callback. Output is passed to attention.'),
 ('experiments/cachegen/c6_runtime.py', 'Storage.encode', 'store/account', 'RESIDENT_TOTAL_Q', True,
  'External C2 make_entry receives total QKV. Factory passes q_tensors and compressed_kv; Q counted separately, KV bitstream includes metadata.'),
 ('experiments/cachegen/c6_runtime.py', 'Storage.validate_decoded', 'equality assertion', 'DIAGNOSTIC_Q', True,
  'Reads resident.q_tensors and view.tensors[layer][0]; checks FP16 and torch.equal. Q values affect this post-lookup diagnostic assertion.'),
 ('experiments/cachegen/c6_runtime.py', 'insert_and_lookup_c6', 'exact hit', 'RESIDENT_TOTAL_Q', False,
  'Lookup owns external decode, validates decoded Q if storage enabled, then CacheHit and validate_hit.'),
 ('experiments/cachegen/c6_runtime.py', 'execute', 'source and target forward', 'RESIDENT_TOTAL_Q', True,
  'Both source capture and target HIT use mixed_projection_path; source spans contain q,k,v.'),
 ('experiments/cachegen/c6_quality.py', 'validate_hit', 'exact-w3 assertion', 'RESIDENT_TOTAL_Q', False,
  'Entry key/positions/window/source provenance only; exact-w3 is not a numerical safety guarantee.'),
 ('experiments/cachegen/c6b2_runtime.py', 'Backend.prepare', 'source insertion', 'RESIDENT_TOTAL_Q', True,
  'Captured qkv slices go to Storage.encode or CacheEntry.from_tensors; same C6 insert/lookup.'),
 ('experiments/cachegen/c6b2_runtime.py', 'Backend.forward', 'target reuse', 'RESIDENT_TOTAL_Q', True,
  'Uses mixed_projection_path for candidate, greedy prefill and teacher-forced hit forwards.'),
 ('experiments/cachegen/c6b2_runtime.py', 'Backend.transported', 'encode/decode/recombine', 'TRANSPORT_LORA_Q_DELTA', True,
  'projection_parts produces base and delta; encode/decode delta; base+decoded reconstructs fresh total. Cached totals bypass callback.'),
 ('experiments/cachegen/c6b2_snips.py', 'byte_accounting', 'account', 'ACCOUNTING_ONLY_Q', False,
  'Q bytes retained separately; payload=q+frame; local metadata already inside frame, never add twice.'),
 ('experiments/cachegen/c6b3_2_runtime.py', 'MultiwozBackend', 'inherit reuse', 'RESIDENT_TOTAL_Q', True,
  'Inherits B2 Backend, delegates forward/prepare, teacher_logits passes context.hits. No alternate KV-only reuse.'),
 ('experiments/cachegen/c6b3_2_runtime.py', 'MultiwozBackend.transported', 'traffic roles', 'TRANSPORT_LORA_Q_DELTA', True,
  'Records transport role then delegates; transport byte counts are not resident reads.'),
 ('experiments/cachegen/c6b3_2_multiwoz.py', 'module', 'policy', 'ACCOUNTING_ONLY_Q', False,
  'Declares TOTAL Q FP16 policy; imports byte_accounting from B2.'),
 ('edgelora/cachegen_projection.py', 'cachegen_reconstructed_projection_path', 'recombine', 'TRANSPORT_LORA_Q_DELTA', True,
  'Fresh base + decoded delta; not resident-source Q.'),
 ('edgelora/cachegen_codec.py', 'encode_lora_delta / decode_lora_delta', 'transport codec', 'TRANSPORT_LORA_Q_DELTA', True,
  'Codec takes a projection delta; CUDA-only implementation is inspected but not imported/executed by C7-A.'),
 ('edgelora/lora_projection.py', 'edge_qkv_projection / reconstructed_projection_path', 'recombine', 'TRANSPORT_LORA_Q_DELTA', True,
  'Fresh projection decomposition and communication accounting; no resident entry access.'),
 ('models/lora_decomposition.py', 'projection_parts', 'base plus delta', 'TRANSPORT_LORA_Q_DELTA', True,
  'Computes fresh base and LoRA delta; no resident Q involved.'),
 ('models/model_adapter.py', 'OPTModelAdapter.projection_modules', 'attention module boundary', 'RESIDENT_TOTAL_Q', True,
  'Maps q to layer.self_attn.q_proj. Mixed replacement returns source Q rows to the actual attention module, before scaling/head splitting.'),
 ('cache/metric_manager.py', 'actual_attention_impact / CacheMetricManager', 'indirect policy feedback', 'RESIDENT_TOTAL_Q', True,
  'No direct Q read; target attention depends on reused Q. Attention impact updates can indirectly affect later admission/eviction in SemCacheEngine.'),
 ('experiments/cachegen/c2/physical_storage.py', 'FrozenK20V16Codec', 'external storage', 'UNKNOWN_Q_ROLE', None,
  'Source absent in this checkout; no claim of having inspected or executed encode/decode internals.'),
]


def static_findings(root):
    findings = []
    for file, symbol, operation, role, consumed, notes in STATIC:
        path = root/'src/semcache'/file
        lines = path.read_text().splitlines() if path.is_file() else []
        needle = symbol.split('.')[-1].split(' / ')[0]
        line = next((i for i, s in enumerate(lines, 1) if ('def '+needle+'(' in s or 'class '+needle in s)), None)
        findings.append(dict(file=str(path.relative_to(root)), symbol=symbol, approximate_line=line,
            operation=operation, q_role=role, q_values_consumed=consumed, notes=notes,
            source_available=path.is_file()))
    return findings


def evidence_from(counter, traces):
    normal = counter['variants']['normal']
    changed = counter['variants']['zero']['changes_vs_normal']['target_reuse_output']
    counts = normal['counters']
    consumed = counts['resident_q_value_consumer_count'] > 0
    result = dict(q_materialized=True, q_written_to_resident_cache=counts['resident_q_write_count'] > 0,
        q_read_after_insert=counts['resident_q_read_count'] > 0,
        q_read_on_hit=counts['resident_q_hit_read_count'] > 0,
        q_value_used_after_insert=consumed, q_required_for_semantic_lookup=False,
        q_required_for_exact_w3=False, q_required_for_cache_key=False,
        q_required_for_admission=None, q_required_for_eviction=None,
        q_required_for_capacity_accounting=True, q_required_for_attention_reuse=bool(consumed and changed) if changed is not None else None,
        q_required_for_kv_reconstruction=False, q_required_for_lora_recombination=False,
        q_schema_field_required=counter['variants']['none']['schema_field_required'],
        q_value_required=bool(consumed and changed) if changed is not None else None, fresh_target_q_still_required=True,
        transport_lora_q_delta_still_required=True)
    contradictory = not normal['reuse_success'] or not consumed
    result.update(resident_q_role_summary='Source TOTAL Q is stored and copied into matched target projection rows; its values affect causal attention/output.',
        recommended_c7_path=recommend(result, runtime_dependency=consumed,
            counterfactual_changed=changed, contradictory=contradictory),
        counters=counts, runtime_contradiction=contradictory,
        could_zero_q_preserve_hit_output=not changed if changed is not None else None,
        could_q_be_absent_for_logical_lookup=counter['variants']['none']['lookup_success'],
        could_q_be_absent_for_physical_reuse=counter['variants']['none']['reuse_success'],
        q_required_for_target_projection_assembly=bool(consumed and changed) if changed is not None else None,
        raw_resident_byte_context=normal['raw_resident_bytes'],
        counter_scope='normal fixture only; other variant counters in q_counterfactual.json; trace includes all variants',
        resident_q_read_call_sites=sorted({r['symbol'] for r in traces['normal'].rows if r['operation']=='read'}),
        resident_q_consumer_call_sites=sorted({r['symbol'] for r in traces['normal'].rows if r['operation']=='consume'}),
        field_semantics=dict(q_required_for_capacity_accounting='Q storage size/schema contributes; numeric Q values are not needed for byte counting.',
            admission_eviction='Direct policy dependency on Q values is false; broader indirect attention-impact feedback exists, so necessity is null.',
            kv_reconstruction='Same-layer K/V projection reuse is independent of Q; deeper-layer fresh K/V may indirectly change through attention.',
            transport='Required by configured transport reconstruction; statically inspected, not executed on CPU.'),
        limitations=['External C2 physical_storage.py and extended GlobalCache API unavailable here; compressed storage not executed.',
            'CPU transport counters are zero because the real CUDA codec is not executed; no substitute codec.',
            'Tiny raw fixture proves the shared consumer requires Q, not model quality or compression tolerance.'])
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def storage_context(path, recommendation):
    context = dict(policy='TOTAL Q FP16 uncompressed; TOTAL K/V optionally storage-compressed',
        c6_summary_path=str(path) if path else None, c6_summary_sha256=None,
        status='NOT_SUPPLIED', modes={}, analytical_what_if=None)
    if path is None:
        return context
    data = json.loads(path.read_text())
    context.update(status='READ', c6_summary_sha256=sha(path))
    accounting = data.get('storage_accounting', {})
    for mode in ('STORAGE_KV_COMP', 'FULL_PIPELINE'):
        row = accounting.get(mode)
        if row is None:
            context['modes'][mode] = dict(status='NOT_FOUND')
            continue
        fields = ('resident_q_bytes', 'resident_kv_frame_bytes', 'resident_payload_bytes',
                  'resident_local_metadata_bytes', 'raw_qkv_bytes', 'raw_kv_bytes')
        context['modes'][mode] = {k: row.get(k) for k in fields}
        if recommendation == 'DROP_RESIDENT_Q' and all(row.get(k) is not None for k in fields[:3]):
            q, frame, payload = (row[k] for k in fields[:3])
            if payload != q+frame or payload <= 0:
                raise ValueError('C6 payload must equal Q + frame (metadata already included)')
            context['modes'][mode]['what_if'] = dict(label='ANALYTICAL_Q_REMOVAL_ACCOUNTING',
                physically_benchmarked=False, current_compressed_payload=payload,
                kv_only_payload=frame, bytes_removed_if_q_dropped=q,
                percent_reduction_vs_current_compressed_payload=100*q/payload,
                ratio_vs_original_raw_qkv=frame/row['raw_qkv_bytes'] if row.get('raw_qkv_bytes') else None,
                ratio_vs_original_raw_kv=frame/row['raw_kv_bytes'] if row.get('raw_kv_bytes') else None,
                ratio_definition='remaining payload / original raw bytes')
    return context


def run(output_root, c6_summary=None):
    root = Path(__file__).resolve().parents[4]
    output_root = Path(output_root)
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError('Refusing non-empty output root: '+str(output_root))
    if c6_summary is not None and not Path(c6_summary).is_file():
        raise FileNotFoundError(c6_summary)
    findings = static_findings(root)
    try:
        counter, traces = counterfactuals()
        audit = evidence_from(counter, traces)
    except Exception as exc:
        counter = dict(status='NOT_RUN', reason=type(exc).__name__+': '+str(exc))
        traces = {}
        keys = ['q_materialized', 'q_written_to_resident_cache', 'q_read_after_insert', 'q_read_on_hit',
                'q_value_used_after_insert', *NEGATIVES, 'q_required_for_admission', 'q_required_for_eviction',
                'q_required_for_capacity_accounting', 'q_required_for_lora_recombination', 'q_schema_field_required',
                'q_value_required', 'fresh_target_q_still_required', 'transport_lora_q_delta_still_required']
        audit = dict.fromkeys(keys)
        audit.update(recommended_c7_path='INCONCLUSIVE', resident_q_role_summary='Runtime incomplete; inspect static findings.', limitations=[counter['reason']])
    audit['static_findings'] = findings
    audit['storage_context'] = storage_context(Path(c6_summary) if c6_summary else None, audit['recommended_c7_path'])
    output_root.mkdir(parents=True, exist_ok=True)
    def write(name, value):
        (output_root/name).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    write('q_residency_audit.json', audit)
    write('q_counterfactual.json', counter)
    with (output_root/'q_residency_trace.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        sequence = 0
        for variant, trace in traces.items():
            for row in trace.rows:
                sequence += 1
                writer.writerow(dict(row, sequence=sequence, phase=variant+':'+row['phase']))
    summary = [
        '# C7-A resident TOTAL Q audit', '',
        'Question: is resident source TOTAL Q consumed after insertion on the target HIT path?', '',
        'Recommendation: **'+audit['recommended_c7_path']+'**.', '',
        'Static dataflow: source mixed projections → CacheEntry.tensors[layer][0] → GlobalCache.lookup → '
        'CacheHit → mixed_projection_path.forward → source.to(output) copied into matched target Q rows → attention. '
        'Compressed C6 contracts use q_tensors plus a temporary decoded QKV view; external C2 source is missing here.', '',
        'Exact cluster/token-w3 identity and selected entry do not inspect Q values. Byte accounting reads Q storage; '
        'the mixed projection consumer reads its numerical values. Same-layer K/V reuse and fresh LoRA recombination do not need resident Q. '
        'Attention-dependent impact updates can indirectly feed later policy decisions.', '',
        'Runtime counters (normal fixture): '+json.dumps(audit.get('counters', {}), sort_keys=True), '',
        'Counterfactual status: '+counter['status']+'. '+counter.get('reason', ''),
    ]
    if counter['status'] == 'COMPLETE':
        summary += ['', 'Zero-Q changes: '+json.dumps(counter['variants']['zero']['changes_vs_normal'], sort_keys=True),
                    'Zero-Q maximum absolute attention-output delta: '+str(counter['variants']['zero']['max_abs_output_delta']),
                    'None-Q changes: '+json.dumps(counter['variants']['none']['changes_vs_normal'], sort_keys=True),
                    'None-Q error: '+str(counter['variants']['none']['error']),
                    'Field-presence dependency != value dependency: None exposes schema assumptions; zero-Q independently tests numerical necessity.']
    summary += ['', 'Transported LoRA Q-delta activity does not by itself justify keeping resident TOTAL Q.',
        'Fresh target Q is a separate runtime object and is not evidence that cached resident source Q is required.', '',
        'Storage implication: Q remains raw FP16 in the C6 policy; compressed K/V frame metadata is already included in the frame. '
        'No Q removal benchmark or byte-savings-based recommendation is made. Optional artifact context is in the audit JSON.', '',
        'Scope: tiny CPU PEFT projection modules and causal attention, no language-model inference, no CUDA/transport codec execution, '
        'no training, no compression implementation, no production cache changes. The shared raw HIT consumer is tested; '
        'external compressed storage internals remain unverified.', '']
    (output_root/'summary.md').write_text('\n'.join(summary))
    inspected = {root/'src/semcache'/f for f, *_ in STATIC}
    inspected.update((root/'src/semcache/cache').glob('*.py'))
    inspected.update((root/'src/semcache/edgelora').glob('*.py'))
    inspected.add(Path(__file__))
    inspected.update((root/'scripts/58_audit_cachegen_c7_q_residency.py',
                      root/'tests/test_cachegen_c7_q_residency.py'))
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    manifest = dict(stage='C7-A', status='COMPLETE' if counter['status']=='COMPLETE' else 'INCOMPLETE',
        audit_scope='resident TOTAL Q lifecycle and HIT-path necessity', model_inference_performed=False,
        gpu_required=False, training_performed=False, q_compression_implemented=False, resident_q_removed=False,
        runtime_fixture_used='CacheEntry + GlobalCache + ExactTokenMatcher + validate_hit + select_nonoverlapping + mixed_projection_path; CPU PEFT Linear; literal causal attention',
        counterfactual_status=counter['status'], recommendation=audit['recommended_c7_path'],
        inspected_files={str(p.relative_to(root)): sha(p) if p.is_file() else None for p in sorted(inspected)},
        c6_summary_path=str(c6_summary) if c6_summary else None,
        c6_summary_sha256=sha(c6_summary) if c6_summary else None,
        software={name: importlib.metadata.version(name) for name in ('torch', 'peft', 'transformers')
                  if importlib.util.find_spec(name) is not None},
        git=dict(branch=git('branch', '--show-current'), commit=git('rev-parse', 'HEAD'), status=git('status', '--porcelain')),
        output_hashes={p.name: sha(p) for p in sorted(output_root.iterdir()) if p.is_file()})
    write('manifest.json', manifest)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('results/cachegen/c7/a_q_residency_audit'))
    parser.add_argument('--c6-summary', type=Path, help='Optional existing C6 summary; never required or implicitly downloaded')
    args = parser.parse_args(argv)
    audit = run(args.output_root, args.c6_summary)
    print(audit['recommended_c7_path'])
    return 0 if audit['recommended_c7_path'] != 'INCONCLUSIVE' else 2
