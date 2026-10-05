"""C7-A2: audit the subsequence HIT's TOTAL_QKV payload, CPU only.

No production changes, codec fitting/execution, model loading, or downloads.
The decoded-view probe tests the C6 consumer contract only; it never stands in
for evidence of the unavailable C2 compressed GlobalCache lookup.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

CONTRACT = dict(hit_unit='w3_token_subsequence',
    physical_safety_contract='PHYSICAL_EXACT_W3_WITHIN_SEMANTIC_CLUSTER',
    cached_payload_contract='TOTAL_QKV')
ROLE_FIELDS = ('source_projection_captured', 'source_w3_rows_extracted',
    'stored_in_cache_entry', 'present_after_lookup', 'consumed_on_hit',
    'injected_into_target_hit_rows', 'fresh_compute_skipped_on_hit_rows')
NEXT = dict(SEMCACHE_FAITHFUL_QKV_REUSE_CONFIRMED='C7-B_Q_COMPRESSION_CALIBRATION',
            Q_REUSE_IMPLEMENTATION_GAP='REPAIR_Q_REUSE_BEFORE_C7_B',
            INCONCLUSIVE='FURTHER_AUDIT_ONLY')


def decision(audit):
    """Fail closed on unknown storage or conflicting evidence; never infer Q from KV."""
    if audit.get('contradictory_evidence') is not False:
        return 'INCONCLUSIVE'
    if any(audit.get(k) != v for k, v in CONTRACT.items()):
        return 'INCONCLUSIVE'
    roles = audit.get('roles', {})
    q = roles.get('q', {})
    kv_reused = all(roles.get(r, {}).get('injected_into_target_hit_rows') is True for r in 'kv')
    q_recomputed = (q.get('stored_in_cache_entry') is True and
                    q.get('fresh_compute_skipped_on_hit_rows') is False)
    if q_recomputed or (kv_reused and q.get('injected_into_target_hit_rows') is False):
        return 'Q_REUSE_IMPLEMENTATION_GAP'
    required = ('same_reuse_mask_qkv', 'raw_semcache_qkv_reuse_confirmed',
                'storage_kv_comp_qkv_reuse_confirmed', 'transport_hit_qkv_bypass_confirmed',
                'full_pipeline_hit_qkv_reuse_confirmed')
    if (all(roles.get(r, {}).get(k) is True for r in 'qkv' for k in ROLE_FIELDS)
            and all(audit.get(k) is True for k in required)):
        return 'SEMCACHE_FAITHFUL_QKV_REUSE_CONFIRMED'
    return 'INCONCLUSIVE'


def q_object(role, phase):
    if role != 'q':
        return 'TOTAL_'+role.upper() if phase != 'delta' else 'LORA_'+role.upper()+'_DELTA'
    return {'cached': 'RESIDENT_CACHED_TOTAL_Q', 'fresh': 'FRESH_TARGET_Q',
            'delta': 'TRANSPORT_LORA_Q_DELTA'}[phase]


def decoded_view_contract(entry):
    """Explicit fabricated view INPUT to the real C6 validator, not a codec.

    This probe cannot confirm compressed-storage lookup, compression fidelity,
    resident byte ownership, or the external codec's view construction.
    """
    from .c6_runtime import Storage
    resident = SimpleNamespace(tensors=None, compressed_kv=object(),
        q_tensors={layer: values[0] for layer, values in entry.tensors.items()})
    view = SimpleNamespace(resident=resident, tensors={
        layer: (resident.q_tensors[layer], values[1].clone(), values[2].clone())
        for layer, values in entry.tensors.items()}, key=entry.key,
        token_ids=entry.token_ids, positions=entry.positions, qkv_metadata=entry.qkv_metadata)
    Storage.validate_decoded(resident, view)
    return resident, view


def fixture(*, layers=3, callback=False, view_probe=False, corrupt=None):
    """Run the actual source capture/cache/mixed path with tiny PEFT projections.

    Every layer has distinct literal weights and independent Q/K/V. Nested
    base/LoRA-A hooks observe actual input rows, not just common audit counts.
    The optional callback uses uncompressed base+delta, never a fake codec.
    """
    import torch
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear
    from semcache.cache.cache_entry import CacheEntry
    from semcache.cache.global_cache import GlobalCache
    from semcache.edgelora.mixed_projection import mixed_projection_path
    from semcache.models.lora_decomposition import projection_parts
    from semcache.semantic.matcher import ExactTokenMatcher
    from semcache.semantic.subsequence import Subsequence
    from semcache.semantic.hit_selection import CacheHit, select_nonoverlapping
    from .c6_quality import validate_hit

    if layers < 1 or corrupt not in (None, 'q', 'k', 'v'):
        raise ValueError('Invalid fixture configuration')
    modules = {}
    for layer in range(layers):
        modules[layer] = {}
        for index, role in enumerate('qkv'):
            config = {'config': LoraConfig(r=2, lora_alpha=2)} if 'config' in inspect.signature(Linear).parameters else {}
            module = Linear(torch.nn.Linear(4, 4, bias=False, device='cpu'), 'audit',
                            r=2, lora_alpha=2, **config)
            with torch.no_grad():
                module.get_base_layer().weight.copy_(torch.eye(4)*(1+layer+index/4))
                module.lora_A['audit'].weight.fill_(0.03125)
                module.lora_B['audit'].weight.fill_(0.0625)
            module.eval()
            modules[layer][role] = module
    adapter = SimpleNamespace(layers=list(range(layers)), projection_modules=modules.__getitem__)
    source = torch.arange(20, dtype=torch.float32, device='cpu').reshape(1, 5, 4)/8
    target = 10+torch.arange(28, dtype=torch.float32, device='cpu').reshape(1, 7, 4)/8
    hit_positions, fresh_positions = [2, 3, 4], [0, 1, 5, 6]
    trace = []
    def event(layer, role, operation, phase, rows, verified, symbol, file, notes=''):
        trace.append(dict(layer=layer, role=role, object_role=q_object(role, phase),
            operation=operation, positions=json.dumps(rows), verified=verified,
            symbol=symbol, file=file, notes=notes))
    mixed_file = inspect.getfile(inspect.unwrap(mixed_projection_path))
    with mixed_projection_path(adapter, 'audit', [], 5) as capture:
        for layer in modules:
            for module in modules[layer].values():
                module(source)
    tensors = {layer: tuple(capture.projections[layer][r][:, 1:4].to(torch.float16) for r in 'qkv')
               for layer in modules}
    entry = CacheEntry.from_tensors(7, (10, 11, 12), (1, 4), tensors)
    entry.qkv_metadata = dict(component_scope='total_qkv', source_user='audit', source_id='synthetic')
    cache = GlobalCache(entry.size_bytes)
    if not cache.insert(entry):
        raise AssertionError('Synthetic source admission failed')
    if corrupt:
        for values in entry.tensors.values():
            values['qkv'.index(corrupt)].add_(8)
    window = Subsequence((10, 11, 12), 2, 5)
    assert (99, 98, 10, 11, 12, 97, 96)[window.start:window.end] == window.token_ids
    matcher = ExactTokenMatcher()
    view = cache.lookup(matcher.key(7, window), record_reuse=False)
    assert view is entry
    # Prove exact token and cluster boundaries without touching selected-hit counters.
    isolation = (cache.entries.get(matcher.key(8, window)) is None and
                 cache.entries.get(matcher.key(7, Subsequence((10, 11, 13), 2, 5))) is None)
    decoded_q_preserved = None
    if view_probe:
        resident, view = decoded_view_contract(entry)
        decoded_q_preserved = all(view.tensors[layer][0] is resident.q_tensors[layer] for layer in modules)
    hit = CacheHit(window, view)
    identity = validate_hit(dict(cluster=7, token_ids=[10, 11, 12], source_start=1,
        target_start=2, source_user='audit', source_id='synthetic'), hit)
    hits, selected_mask = select_nonoverlapping([hit], 7)
    cache.record_reuse(entry)
    expected, reads, calls, callback_inputs = {}, {}, {}, {}
    for layer in modules:
        for role, module in modules[layer].items():
            with torch.inference_mode():
                expected[layer, role] = module(target[:, fresh_positions]).detach()
            reads[layer, role] = 0
            calls[layer, role] = {'base': [], 'lora_A': []}
    class Observed(tuple):
        def __new__(cls, values, layer):
            obj = super().__new__(cls, values)
            obj.layer = layer
            return obj
        def __getitem__(self, index):
            role = 'qkv'[index]
            reads[self.layer, role] += 1
            event(self.layer, role, 'read_hit_payload', 'cached', hit_positions, True,
                  'mixed_projection_path.forward', mixed_file)
            return tuple.__getitem__(self, index)
    # Keep reference expectations outside instrumentation, so reads mean reuse reads.
    cached = {layer: tuple(values) for layer, values in view.tensors.items()}
    view.tensors = {layer: Observed(values, layer) for layer, values in cached.items()}
    handles = []
    for layer in modules:
        for role, module in modules[layer].items():
            for name, component in (('base', module.get_base_layer()), ('lora_A', module.lora_A['audit'])):
                def hook(module, args, key=(layer, role), component=name):
                    calls[key][component].append(args[0].detach().clone())
                handles.append(component.register_forward_pre_hook(hook))
    def fresh(layer, role, module, hidden, native):
        callback_inputs[layer, role] = hidden.detach().clone()
        base, delta, _ = projection_parts(module, hidden, 'audit')
        event(layer, role, 'fresh_delta_boundary_probe', 'delta', fresh_positions, True,
              'projection_parts', inspect.getfile(projection_parts),
              'Uncompressed delta; NO transport encode/decode or codec fitting')
        return (base+delta).to(base.dtype)
    output = {}
    try:
        with mixed_projection_path(adapter, 'audit', hits, 7,
                fresh_projection=fresh if callback else None) as mixed:
            for layer in modules:
                for role, module in modules[layer].items():
                    output[layer, role] = module(target)
    finally:
        for handle in handles:
            handle.remove()
    records = []
    for layer in modules:
        for index, role in enumerate('qkv'):
            key = layer, role
            record = mixed.records[key]
            observed = calls[key]
            computation_fresh_only = all(len(values) == 1 and torch.equal(values[0], target[:, fresh_positions])
                                         for values in observed.values())
            hit_equal = torch.equal(output[key][:, hit_positions], cached[layer][index].to(output[key]))
            fresh_equal = torch.equal(output[key][:, fresh_positions], expected[key])
            # Compare the hit payload to hypothetical native hit computation outside hooks.
            with torch.inference_mode():
                distinct = not torch.equal(cached[layer][index].float(), modules[layer][role](target[:, hit_positions]))
            row = dict(layer=layer, role=role,
                source_projection_captured=role in capture.projections[layer],
                source_w3_rows_extracted=tensors[layer][index].shape[1] == 3,
                stored_in_cache_entry=torch.equal(tensors[layer][index], cached[layer][index]) if not corrupt else True,
                present_after_lookup=cached[layer][index].shape == (1, 3, 4),
                consumed_on_hit=reads[key] > 0 and hit_equal,
                injected_into_target_hit_rows=hit_equal,
                fresh_compute_skipped_on_hit_rows=computation_fresh_only,
                fresh_rows_match=fresh_equal, cached_and_fresh_values_distinct=distinct,
                reuse_mask=[i in hit_positions for i in range(7)] if hit_equal else None,
                reused_projection_rows=record['reused_projection_rows'],
                native_projection_rows_skipped=3 if computation_fresh_only else 0,
                native_projection_rows=record['native_projection_rows'],
                fresh_projection_rows_observed=sum(x.shape[1] for x in observed['base']),
                fresh_rows_callback_reconstructed=record['reconstructed_projection_rows'],
                fresh_rows_transport_reconstructed=None if callback else 0,
                hit_rows_from_cache=3 if hit_equal else 0,
                callback_received_only_fresh_rows=(torch.equal(callback_inputs[key], target[:, fresh_positions]) if callback else None))
            records.append(row)
            for operation, value in ((k, row[k]) for k in ROLE_FIELDS):
                positions = [1, 2, 3] if operation in ('source_projection_captured', 'source_w3_rows_extracted', 'stored_in_cache_entry') else hit_positions
                symbol, file = {
                    'source_projection_captured': ('mixed_projection_path.forward', mixed_file),
                    'source_w3_rows_extracted': ('fixture.source_w3_slice', __file__),
                    'stored_in_cache_entry': ('CacheEntry.from_tensors', inspect.getfile(CacheEntry)),
                    'present_after_lookup': ('GlobalCache.lookup' if not view_probe else 'decoded_view_contract',
                                             inspect.getfile(GlobalCache) if not view_probe else __file__),
                }.get(operation, ('mixed_projection_path.forward', mixed_file))
                event(layer, role, operation, 'cached', positions, value, symbol, file)
            event(layer, role, 'fresh_projection', 'fresh', fresh_positions, fresh_equal,
                  'mixed_projection_path.forward', mixed_file)
    return dict(status='COMPLETE', **CONTRACT, layers=layers, selected_hit_count=len(hits),
        source_positions=[1, 2, 3], target_hit_positions=hit_positions, fresh_positions=fresh_positions,
        selected_mask=selected_mask, mixed_mask=mixed.reused_mask.tolist(), logical_event_hash=identity,
        cluster_and_exact_token_isolation=isolation, records=records,
        decoded_view_q_preserved_contract_probe=decoded_q_preserved,
        real_compressed_lookup_executed=False, transport_codec_executed=False,
        resident_dtype='float16', fresh_compute_dtype='float32',
        fixture_kind='DECODED_VIEW_CONTRACT_ONLY' if view_probe else 'FRESH_CALLBACK_BOUNDARY' if callback else 'RAW_SEMCACHE',
        outputs={f'{layer}:{role}': value.tolist() for (layer, role), value in output.items()}), trace


def counterfactual(normal, layers):
    cases = {}
    for changed in 'qkv':
        altered, _ = fixture(layers=layers, corrupt=changed)
        changed_roles = {r: any(altered['outputs'][f'{l}:{r}'] != normal['outputs'][f'{l}:{r}']
                                for l in range(layers)) for r in 'qkv'}
        fresh_unchanged = all(altered['outputs'][key][0][i] == value[0][i]
            for key, value in normal['outputs'].items() for i in normal['fresh_positions'])
        cases[changed] = dict(changed_roles=changed_roles, fresh_rows_unchanged=fresh_unchanged,
            logical_hit_unchanged=altered['logical_event_hash'] == normal['logical_event_hash'],
            only_corrupted_payload_role_changed=all(v == (r == changed) for r, v in changed_roles.items()))
    return dict(status='COMPLETE', purpose='Prove each cached payload role is consumed; no removal question', cases=cases)


# Reviewed implementation links. Each role is expanded independently below;
# runtime per-role evidence complements this shared control-flow inspection.
STATIC = [
 ('semcache_engine.py', 'query', 'source capture/extract/insert and target reuse',
  "audit.projections[layer][role][:, window.start:window.end] -> CacheEntry.from_tensors; selected CacheHit list -> mixed_projection_path"),
 ('cache/cache_entry.py', 'from_tensors', 'own TOTAL projections',
  'entry.tensors[layer][0/1/2] owns compact detached Q/K/V; no Q-specific lookup key'),
 ('semantic/matcher.py', 'key', 'subsequence key', 'key=(cluster_id, tuple(token_ids)); no projection role in key'),
 ('semantic/hit_selection.py', 'select_nonoverlapping', 'select subsequence',
  'CacheHit(window,entry); checks exact tokens, resolves overlap, common target mask'),
 ('cache/global_cache.py', 'lookup', 'return hit entry',
  'Installed raw API returns indexed entry; extended compressed API must be checked separately'),
 ('edgelora/mixed_projection.py', 'forward', 'inject selected payload',
  "hit.entry.tensors[layer]['qkv'.index(name)] -> output[:,w.start:w.end]; fresh_inputs index_select excludes mask"),
 ('models/model_adapter.py', 'projection_modules', 'all layer roles',
  'OPTModelAdapter maps q/k/v to self_attn.q_proj/k_proj/v_proj for each decoder layer'),
 ('experiments/cachegen/c6_runtime.py', 'execute', 'C6 source and target',
  "Source capture includes qkv; w=3 slices; one selected hit; every record requires reused_projection_rows=3*len(hits)"),
 ('experiments/cachegen/c6_runtime.py', 'encode', 'compressed resident contract',
  'Storage factory receives raw q_tensors plus compressed_kv; resident Q counted separately; external make_entry implementation unavailable here'),
 ('experiments/cachegen/c6_runtime.py', 'insert_and_lookup_c6', 'compressed lookup owner',
  'GlobalCache.lookup owns decode; calls Storage.validate_decoded; returns CacheHit(window,view)'),
 ('experiments/cachegen/c6_runtime.py', 'validate_decoded', 'decoded-view Q validation',
  'Checks resident.tensors is None, compressed_kv exists, view.resident identity, resident Q FP16 and exact view Q equality'),
 ('experiments/cachegen/c6_quality.py', 'validate_hit', 'exact-w3 provenance',
  'Checks cluster/token key, source and target w=3 spans and source provenance; does not select projection-specific hits'),
 ('experiments/cachegen/c6b2_runtime.py', 'prepare', 'B2 capture and lookup',
  "Extracts source values[role] for role in qkv; uses same Storage/CacheEntry and insert_and_lookup_c6"),
 ('experiments/cachegen/c6b2_runtime.py', 'forward', 'B2 mixed target',
  'Passes hits and optional transported callback to mixed_projection_path; validates reused rows for every record'),
 ('experiments/cachegen/c6b2_runtime.py', 'transported', 'fresh transport delta',
  'projection_parts -> encode(delta) -> decode -> base+decoded; only called on fresh_inputs by mixed path'),
 ('experiments/cachegen/c6b2_snips.py', 'byte_accounting', 'C6 storage policy',
  'Resident TOTAL Q raw FP16; compressed KV frame includes local metadata; logical raw size unchanged'),
 ('experiments/cachegen/c6b3_2_runtime.py', 'forward', 'B3 delegates to B2',
  'MultiwozBackend inherits B2 Backend; delegates forward/prepare, teacher_logits uses context.hits'),
 ('experiments/cachegen/c6b3_2_multiwoz.py', 'module', 'frozen mode protocol',
  'Imports C6 modes and B2 byte_accounting; declares resident TOTAL Q FP16 policy'),
 ('experiments/cachegen/c2/physical_storage.py', 'FrozenK20V16Codec', 'actual decoded-view construction',
  'Required external implementation missing in this checkout; caller/contract probes do not establish actual compressed lookup behavior'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def static_audit(root):
    rows = []
    for relative, symbol, operation, description in STATIC:
        path = root/'src/semcache'/relative
        lines = path.read_text().splitlines() if path.is_file() else []
        line = next((i for i, text in enumerate(lines, 1) if f'def {symbol}(' in text or f'class {symbol}' in text), None)
        for role in 'qkv':
            rows.append(dict(file=str(path.relative_to(root)), symbol=symbol, approximate_line=line,
                role=role, operation=operation, finding=description, source_available=path.is_file(),
                field=f'tensors[layer][{"qkv".index(role)}]',
                resident_storage_field='q_tensors[layer]' if role=='q' else 'compressed_kv'))
    return rows


def storage_availability(root):
    from semcache.cache.global_cache import GlobalCache
    source = root/'src/semcache/experiments/cachegen/c2/physical_storage.py'
    return dict(source_available=source.is_file(),
        extended_global_cache_available='physical_codec' in inspect.signature(GlobalCache).parameters,
        status='NOT_RUN', actual_compressed_lookup_confirmed=None,
        reason='No frozen codec/profile execution in this audit. '+
            ('C2 source is missing. ' if not source.is_file() else 'C2 source needs further codec-path audit. ')+
            ('Installed GlobalCache lacks physical_codec. ' if 'physical_codec' not in inspect.signature(GlobalCache).parameters else '')+
            'Decoded-view contract probe is not a real compressed-storage lookup.')


def aggregate(raw, callback, view, counter, storage):
    roles = {r: {k: all(row[k] is True for row in raw['records'] if row['role']==r)
                 for k in ROLE_FIELDS} for r in 'qkv'}
    # Runtime mismatch of read/projection behavior contradicts the reviewed shared path.
    coverage = all(len(result['records']) == 3*result['layers'] and
        {(row['layer'], row['role']) for row in result['records']} ==
        {(layer, role) for layer in range(result['layers']) for role in 'qkv'}
        for result in (raw, callback, view))
    contradictory = not coverage or any(not all(row[k] is True for k in (*ROLE_FIELDS, 'fresh_rows_match', 'cached_and_fresh_values_distinct'))
        or row['reused_projection_rows'] != 3 or row['native_projection_rows_skipped'] != 3
        for result in (raw, callback, view) for row in result['records'])
    contradictory |= any(not case['only_corrupted_payload_role_changed'] or not case['fresh_rows_unchanged']
                         or not case['logical_hit_unchanged'] for case in counter['cases'].values())
    same_mask = all(result['selected_hit_count']==1 and sum(result['selected_mask'])==3 and
        result['mixed_mask']==result['selected_mask'] and
        all(row['reuse_mask']==result['selected_mask'] for row in result['records'])
        for result in (raw, callback, view))
    bypass = all(row['callback_received_only_fresh_rows'] is True and row['hit_rows_from_cache']==3
                 for row in callback['records'])
    audit = dict(**CONTRACT, roles=roles, same_reuse_mask_qkv=same_mask,
        raw_semcache_qkv_reuse_confirmed=all(all(values.values()) for values in roles.values()) and same_mask,
        storage_kv_comp_qkv_reuse_confirmed=storage['actual_compressed_lookup_confirmed'],
        transport_hit_qkv_bypass_confirmed=bypass,
        full_pipeline_hit_qkv_reuse_confirmed=None,
        fresh_target_q_separate_from_cached_q=True, transport_lora_q_delta_separate_from_cached_q=True,
        contradictory_evidence=contradictory, storage_runtime=storage,
        all_synthetic_projection_layers_covered=coverage,
        count_scope='Per role per layer; skipped counts describe the three HIT rows only. Callback fresh rows also bypass native PEFT forward but execute base/LoRA parts.',
        per_layer_projection_records=raw['records'], fresh_callback_records=callback['records'],
        decoded_view_contract_records=view['records'],
        layer_scope=f"All {raw['layers']} synthetic adapter layers independently executed; static loop covers every adapter.layers entry; no real OPT model executed",
        decoded_view_contract_probe=dict(q_preserved=view['decoded_view_q_preserved_contract_probe'],
            all_qkv_reused=all(row['injected_into_target_hit_rows'] for row in view['records']),
            real_compressed_lookup_executed=False),
        modes={
            'RAW_SEMCACHE': dict(status='CONFIRMED_SYNTHETIC', evidence='Real raw cache, exact matcher, mixed projections'),
            'STORAGE_KV_COMP': dict(status='NOT_RUN', evidence=storage['reason']),
            'TRANSPORT_QKV_COMP': dict(status='HIT_BYPASS_CONFIRMED_AT_CALLBACK_BOUNDARY',
                evidence='Real mixed callback receives only fresh rows; uncompressed projection_parts probe; CUDA codec not executed'),
            'FULL_PIPELINE': dict(status='NOT_RUN', evidence='Fresh callback isolation established; actual compressed lookup unverified')})
    for r in 'qkv':
        for field in ('reused_projection_rows', 'native_projection_rows_skipped', 'hit_rows_from_cache'):
            name = f'hit_{r}_rows_from_cache' if field=='hit_rows_from_cache' else f'{r}_{field}'
            audit[name] = {str(row['layer']): row[field] for row in raw['records'] if row['role']==r}
        audit[f'fresh_{r}_rows_transport_reconstructed'] = None
        audit[f'fresh_{r}_rows_callback_reconstructed'] = {str(row['layer']): row['fresh_rows_callback_reconstructed']
            for row in callback['records'] if row['role']==r}
    audit['recommendation'] = decision(audit)
    audit['next_stage'] = NEXT[audit['recommendation']]
    return audit


def protected_hashes(root):
    paths = list((root/'results/cachegen/c7/a_q_residency_audit').glob('*'))
    paths += [root/'src/semcache/experiments/cachegen/c7_q_residency.py',
              root/'scripts/58_audit_cachegen_c7_q_residency.py', root/'tests/test_cachegen_c7_q_residency.py']
    paths += list((root/'src/semcache/experiments/cachegen').glob('c6*.py'))
    return {str(p.relative_to(root)): sha(p) for p in paths if p.is_file()}


def run(output_root, *, layers=3):
    root = Path(__file__).resolve().parents[4]
    output_root = Path(output_root).resolve()
    historical = (root/'results/cachegen/c7/a_q_residency_audit').resolve()
    if output_root == historical or historical in output_root.parents or output_root in historical.parents:
        raise ValueError('Previous C7-A artifact tree is protected')
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError('Refusing nonempty output root: '+str(output_root))
    before = protected_hashes(root)
    traces = []
    storage = storage_availability(root)
    try:
        raw, rt = fixture(layers=layers)
        callback, ct = fixture(layers=layers, callback=True)
        view, vt = fixture(layers=layers, view_probe=True)
        for name, rows in (('RAW_SEMCACHE', rt), ('FRESH_CALLBACK_BOUNDARY', ct), ('DECODED_VIEW_CONTRACT_ONLY', vt)):
            traces.extend(dict(row, probe=name) for row in rows)
        counter = counterfactual(raw, layers)
        audit = aggregate(raw, callback, view, counter, storage)
        runtime_status = 'COMPLETE'
    except Exception as exc:
        runtime_status = 'INCOMPLETE'
        counter = dict(status='NOT_RUN', reason=f'{type(exc).__name__}: {exc}')
        audit = dict(**CONTRACT, roles={r: dict.fromkeys(ROLE_FIELDS) for r in 'qkv'},
            **dict.fromkeys(('same_reuse_mask_qkv', 'raw_semcache_qkv_reuse_confirmed',
                'storage_kv_comp_qkv_reuse_confirmed', 'transport_hit_qkv_bypass_confirmed',
                'full_pipeline_hit_qkv_reuse_confirmed', 'fresh_target_q_separate_from_cached_q',
                'transport_lora_q_delta_separate_from_cached_q')),
            recommendation='INCONCLUSIVE', next_stage=NEXT['INCONCLUSIVE'], runtime_error=counter['reason'], storage_runtime=storage)
    audit['static_findings'] = static_audit(root)
    output_root.mkdir(parents=True, exist_ok=True)
    def write(name, value):
        (output_root/name).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    write('qkv_reuse_audit.json', audit)
    write('qkv_counterfactual.json', counter)
    columns = ('sequence', 'probe', 'layer', 'role', 'object_role', 'operation', 'positions', 'verified', 'symbol', 'file', 'notes')
    with (output_root/'qkv_reuse_trace.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(dict(row, sequence=i) for i, row in enumerate(traces, 1))
    summary = '\n'.join([
        '# C7-A2 SemCache Q/K/V reuse audit', '',
        'The cache hit is a w=3 token-subsequence hit, not a Q/K/V-specific hit.',
        'Q/K/V are the cached payload associated with the selected hit.',
        'SemCache intentionally reuses Q as well as K and V because its target is projection reuse in EdgeLoRA, not merely standard autoregressive KV-cache reuse.',
        'Transport LoRA Q delta and cached resident TOTAL Q are distinct objects.',
        'Fresh target Q is separately computed on unmatched positions.', '',
        'Flow: source mixed projection capture -> source w=3 slices -> CacheEntry.tensors[layer][q/k/v index] -> '
        '(cluster, exact token IDs) lookup -> CacheHit -> common nonoverlap mask -> mixed_projection_path -> target hit rows.', '',
        'Raw Q/K/V reuse confirmed: '+str(audit['raw_semcache_qkv_reuse_confirmed'])+'. '
        'Same reuse mask: '+str(audit['same_reuse_mask_qkv'])+'.',
        f'CPU fixture: {layers} layers, one selected three-token hit at positions [2,3,4], four fresh rows. '
        'Per-role base and LoRA-A hooks check actual input rows. Per-layer reuse/skipped counts and value comparisons are in the JSON.',
        'Transport boundary bypass confirmed: '+str(audit['transport_hit_qkv_bypass_confirmed'])+'. '
        'Only an uncompressed base+delta callback probe ran; actual transport-reconstructed row counts are null.',
        'Compressed-storage / FULL_PIPELINE confirmation: unknown. '+storage['reason'],
        'The constructed decoded-view input passes the real C6 validator and exercises the real mixed consumer; '
        'this proves the consumer contract, not external codec view construction or compressed lookup.',
        'C6 raw/shared mixed execution retains Q as an active SemCache reuse payload. '
        'The broader claim that compressed C6 preserved it end-to-end remains unverified here.', '',
        'Counterfactual status: '+counter['status']+'. Each independent Q/K/V corruption tests payload consumption only.',
        'Previous C7-A framing is superseded; its implementation and artifacts remain unchanged and are not removal evidence.', '',
        'Recommendation: **'+audit['recommendation']+'**.',
        'Next stage: **'+audit['next_stage']+'**. Inspect and validate the actual frozen C2 codec and extended GlobalCache path before C7-B.',
        audit.get('runtime_error', ''), ''])
    (output_root/'summary.md').write_text(summary)
    inspected = {root/'src/semcache'/p for p, *_ in STATIC}
    inspected.update((root/'src/semcache/cache').glob('*.py'))
    inspected.update(root/p for p in ('src/semcache/models/lora_decomposition.py',
        'src/semcache/edgelora/cachegen_codec.py', 'src/semcache/edgelora/cachegen_projection.py',
        'src/semcache/experiments/cachegen/c7a2_qkv_reuse_audit.py',
        'scripts/59_audit_cachegen_c7a2_semcache_qkv_reuse.py', 'tests/test_cachegen_c7a2_qkv_reuse_audit.py'))
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    after = protected_hashes(root)
    if before != after:
        raise RuntimeError('Protected C6/C7-A files changed during audit')
    versions = {}
    for name in ('torch', 'peft', 'transformers'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = dict(stage='C7-A2', status='COMPLETE_WITH_LIMITATIONS' if runtime_status=='COMPLETE' else 'INCOMPLETE',
        runtime_fixture_status=runtime_status,
        audit_scope='SemCache-faithful cached TOTAL Q/K/V reuse on selected w=3 hit',
        original_semcache_hit_unit='token subsequence',
        current_physical_hit_unit='exact w=3 token subsequence within semantic cluster', cached_payload='TOTAL_QKV',
        q_compression_implemented=False, resident_q_removed=False, c6_modified=False,
        model_inference_performed=False, training_performed=False, gpu_required=False,
        downloads_performed=False, previous_c7a_superseded=True, supersedes_previous_c7a_framing=True,
        superseded_question='whether resident TOTAL Q could be dropped based on standard KV-cache reasoning',
        corrected_question='whether current implementation faithfully reuses cached TOTAL Q/K/V for the selected SemCache w=3 hit',
        recommendation=audit['recommendation'], next_stage=audit['next_stage'],
        inspected_files={str(p.relative_to(root)): sha(p) if p.is_file() else None for p in sorted(inspected)},
        preserved_historical_and_c6_hashes=before, protected_files_unchanged=True, software=versions,
        git=dict(branch=git('branch', '--show-current'), commit=git('rev-parse', 'HEAD'), status=git('status', '--porcelain')),
        output_hashes={p.name: sha(p) for p in sorted(output_root.iterdir()) if p.is_file()})
    write('manifest.json', manifest)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('results/cachegen/c7/a2_semcache_qkv_reuse_audit'))
    parser.add_argument('--layers', type=int, default=3, help='Tiny CPU synthetic layers; never an OPT model')
    args = parser.parse_args(argv)
    if args.layers < 1:
        parser.error('--layers must be positive')
    audit = run(args.output_root, layers=args.layers)
    print(audit['recommendation'])
    return 2 if audit['recommendation']=='INCONCLUSIVE' else 0
