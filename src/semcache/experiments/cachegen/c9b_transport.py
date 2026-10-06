"""Read-only C6 raw LoRA-delta accounting; no torch, codec, or model import."""
import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace

from . import c7b3_q_freeze as provenance
from . import c9a_latency as local

require = provenance.require
ROLES = 'qkv'
CONTRACT = 'raw active-user LoRA Q/K/V deltas [1,fresh_rows,out_features]; cached TOTAL-QKV HIT rows bypass fresh delta transport'


def bound_path(hashes, expected_sha, suffix=None):
    paths = [Path(p) for p, h in hashes.items() if h == expected_sha and (suffix is None or p.endswith(suffix))]
    require(len(paths) == 1, 'Missing/ambiguous hash-bound artifact: '+str(expected_sha)+' '+str(suffix))
    provenance.check_hash(paths[0], expected_sha)
    return paths[0]


class C6Accounting:
    """Reuse the audited C6 expressions and traffic_summary without importing CUDA.

    AST checks fail if the recorded accounting contract changes. Only these
    expressions / the pure summary function execute; no runtime module executes.
    """
    def __init__(self):
        root = Path(__file__).resolve().parents[2]
        paths = [Path(__file__).with_name(name) for name in ('c6b3_2_runtime.py', 'c6b2_runtime.py', 'c6b_runtime.py')]
        paths += [root/'edgelora'/name for name in ('mixed_projection.py', 'cachegen_codec.py')]
        paths += [root/'models'/name for name in ('task_adapters.py', 'lora_decomposition.py')]
        self.source_hashes = {str(p.resolve()): provenance.sha(p) for p in paths}
        tree = ast.parse(paths[0].read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiwozBackend')
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        assignments = {n.targets[0].id: n.value for n in ast.walk(forward)
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        expressions = dict(fresh='len(ids)-sum(h.window.end-h.window.start for h in hits)',
            n='fresh*module.out_features*module.lora_B[user].weight.element_size()')
        for name, expression in expressions.items():
            require(name in assignments and ast.dump(assignments[name]) == ast.dump(ast.parse(expression, mode='eval').body),
                'C6 transport accounting expression changed: '+name)
        self.fresh = compile(ast.Expression(assignments['fresh']), str(paths[0]), 'eval')
        self.size = compile(ast.Expression(assignments['n']), str(paths[0]), 'eval')
        summary = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'traffic_summary')
        namespace = {}
        exec(compile(ast.Module(body=[summary], type_ignores=[]), str(paths[0]), 'exec'), namespace)
        self.summary = namespace['traffic_summary']
        # Bind dtype and row-scope assumptions to the real reconstruction and
        # mixed execution expressions, not just to a byte-counter formula.
        guards = {
            paths[1]: ('Backend.transported', (
                'projection_parts(module, hidden, self.active_user)', 'self.encode(delta)')),
            root/'models'/'lora_decomposition.py': ('projection_parts', (
                'module._cast_input_dtype(hidden_states, a.weight.dtype)',
                'b(a(module.lora_dropout[adapter_name](x))) * module.scaling[adapter_name]',
                '(base + delta).to(base.dtype)')),
            root/'edgelora'/'mixed_projection.py': ('mixed_projection_path', (
                '(~mask).nonzero().flatten()', 'hidden_states.index_select(1, index)',
                'fresh_projection(layer, name, module, fresh_inputs, native)',
                "hit.entry.tensors[layer]['qkv'.index(name)]")),
        }
        for path, (symbol, guard_expressions) in guards.items():
            node = ast.parse(path.read_text())
            for part in symbol.split('.'):
                node = next(n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part)
            nodes = {ast.dump(n) for n in ast.walk(node)}
            for expression in guard_expressions:
                require(ast.dump(ast.parse(expression, mode='eval').body) in nodes,
                    'C6 projection/transport contract changed: '+symbol+' '+expression)
        self.provenance = dict(source_hashes=self.source_hashes,
            symbols=['MultiwozBackend.forward', 'traffic_summary', 'Backend.transported',
                'mixed_projection_path', 'projection_parts', 'load_two_task_users'],
            fresh_rows_expression=expressions['fresh'], raw_bytes_expression=expressions['n'],
            payload_scope='one target prompt forward only; excludes source build and autoregressive continuation',
            full_counter_caveat='C6 FULL native hits=None is not instrumented by its traffic counter; FULL payload is derived from the same fresh projection tensor contract, not its saved zero counter',
            network_transaction_count=None, control_round_trip_count=None,
            transaction_evidence='local callbacks/tensor payloads are not a measured network protocol; no RTT or transaction batching assumed')

    def bytes(self, prompt_rows, hit_rows, projections, user):
        require(type(prompt_rows) is int and type(hit_rows) is int and 0 <= hit_rows <= prompt_rows,
            'Invalid prompt/HIT row counts')
        hits = [SimpleNamespace(window=SimpleNamespace(start=0, end=hit_rows))] if hit_rows else []
        fresh = eval(self.fresh, {'len': len, 'sum': sum}, dict(ids=range(prompt_rows), hits=hits))
        counts = {f'raw_{role}_delta_bytes': 0 for role in ROLES}
        for projection in projections:
            weight = SimpleNamespace(element_size=lambda p=projection: p['element_size'])
            module = SimpleNamespace(out_features=projection['out_features'], lora_B={user: SimpleNamespace(weight=weight)})
            counts[f'raw_{projection["role"]}_delta_bytes'] += eval(self.size, {}, dict(fresh=fresh, module=module, user=user))
        result = self.summary(counts, False)
        return {role: result[f'raw_{role}_delta_bytes'] for role in ROLES}, result['transmitted_total_including_cdf_bytes'], fresh


def adapter_header(path):
    """Read safetensors metadata only; no tensor allocation or adapter/model load."""
    with path.open('rb') as stream:
        size = int.from_bytes(stream.read(8), 'little')
        require(0 < size <= 16*1024*1024 and size+8 <= path.stat().st_size, 'Invalid safetensors header: '+str(path))
        header = json.loads(stream.read(size))
    require(isinstance(header, dict), 'Malformed adapter header')
    pattern = re.compile(r'(?:^|\.)layers\.(\d+)\.self_attn\.([qkv])_proj\.lora_([AB])(?:\.[^.]+)?\.weight$')
    matrices = {}
    for name, tensor in header.items():
        if name == '__metadata__': continue
        match = pattern.search(name)
        require(match is not None and isinstance(tensor, dict), 'Unsupported frozen adapter tensor: '+name)
        layer, role, matrix = match.groups(); key = int(layer), role, matrix
        require(key not in matrices, 'Duplicate adapter matrix: '+repr(key))
        # The frozen task-training path saves FP32 A/B. PEFT keeps/promotes these
        # weights to FP32 and projection_parts casts fresh inputs to A's dtype.
        require(tensor.get('dtype') == 'F32', 'Unproven runtime LoRA delta dtype; frozen FP32 adapter required: '+name)
        expected = [8, 2560] if matrix == 'A' else [2560, 8]
        require(tensor.get('shape') == expected, 'Frozen adapter projection shape mismatch: '+name)
        offsets = tensor.get('data_offsets')
        require(isinstance(offsets, list) and len(offsets) == 2 and all(type(i) is int for i in offsets)
            and 0 <= offsets[0] < offsets[1] <= path.stat().st_size-8-size
            and offsets[1]-offsets[0] == 8*2560*4, 'Invalid adapter tensor byte extent: '+name)
        matrices[key] = tensor
    require(set(matrices) == {(l, r, m) for l in range(32) for r in ROLES for m in 'AB'}, 'Adapter must cover all32 QKV layers')
    return [dict(layer=l, role=r, out_features=matrices[l, r, 'B']['shape'][0],
        dtype=matrices[l, r, 'B']['dtype'], element_size=4) for l in range(32) for r in ROLES]


def audit_transport(prepared, raw, local_cases):
    accounting = C6Accounting()
    headers, header_paths = {}, {}
    for user, digest in prepared.c9_manifest['adapter_hashes'].items():
        path = bound_path(prepared.input_hashes, digest, 'adapter_model.safetensors')
        require(path.parent.name == user, 'Adapter/user provenance mismatch: '+str(path))
        headers[user], header_paths[user] = adapter_header(path), str(path)
    require(set(headers) == {'user_a', 'user_b'}, 'Two frozen user adapters required')
    sem_sha = prepared.c8b_manifest['canonical_full_provenance']['semantic_workload_sha256']
    semantic = bound_path(prepared.input_hashes, sem_sha, '.jsonl')
    rows = [json.loads(line) for line in semantic.read_text().splitlines() if line.strip()]
    by_source = provenance.unique(rows, 'source_id')
    lengths = {}
    for episode in prepared.episodes:
        row = by_source.get(episode['target_id'])
        require(row is not None and row.get('prompt_version') == 'c6b3_multiwoz_history_v1', 'Missing/version-changed frozen prompt')
        ids = row.get('token_ids'); start = episode['target_start']
        require(isinstance(ids, list) and all(type(i) is int for i in ids) and ids[start:start+3] == episode['token_ids'],
            'Target exact-w3 prompt differs: '+episode['episode_id'])
        if 'target_index' in episode: require(rows[episode['target_index']]['source_id'] == episode['target_id'], 'Target workload order changed')
        lengths[episode['episode_id']] = len(ids)
    for row in raw:
        require(row['prompt_tokens'] == lengths[row['episode_id']], 'C9 measured prompt length differs from frozen token IDs')
        require(row['native_projection_rows_skipped_per_role_per_layer'] == (3 if row['hit'] else 0), 'C9 measured QKV native skipping differs')
    by_episode = provenance.unique(prepared.episodes, 'episode_id')
    results = []
    for row in local_cases:
        episode = by_episode[row['episode_id']]
        hit_rows = 3 if row['hit'] else 0
        roles, total, fresh = accounting.bytes(lengths[row['episode_id']], hit_rows, headers[episode['target_user']], episode['target_user'])
        budget, policy = (None, local.REFERENCE) if row['condition'] == local.REFERENCE else local.parse_target_condition(row['condition'])
        results.append(dict(episode_id=row['episode_id'], condition=row['condition'], budget=budget, policy=policy,
            hit=row['hit'], resident_source_episode_id=row['retained_source_episode_id'], hit_rows=hit_rows,
            prompt_rows=lengths[row['episode_id']], fresh_rows=fresh,
            **{role+'_transport_bytes': roles[role] for role in ROLES}, total_transport_bytes=total,
            logical_delta_payload_count=len(headers[episode['target_user']]) if fresh else 0,
            transport_compression_enabled=False, payload_object='TRANSPORT_LORA_QKV_DELTA',
            accounting='EXACT_SHAPE_DTYPE_ACCOUNTING; transport not physically executed'))
    full = {r['episode_id']: r for r in results if r['condition'] == local.REFERENCE}
    indexed = {(r['condition'], r['episode_id']): r for r in results}
    for row in results:
        row['bytes_saved_vs_FULL'] = full[row['episode_id']]['total_transport_bytes']-row['total_transport_bytes']
        raw_name = row['budget']+'_RAW_QKV' if row['budget'] else None
        row['bytes_saved_vs_RAW_QKV'] = None if raw_name is None else indexed[raw_name, row['episode_id']]['total_transport_bytes']-row['total_transport_bytes']
    cross_checks = cross_check_c6(prepared, lengths, headers, accounting)
    prepared.input_hashes.update(accounting.source_hashes)
    summary = dict(transport_accounting_contract=CONTRACT, transport_accounting_source=accounting.provenance,
        adapter_header_paths=header_paths, per_layer_shapes=headers, transport_delta_dtype='float32',
        resident_total_dtype='float16', resident_compressed_frames_are_transport=False,
        canonical_semantic_workload_path=str(semantic), cross_checks=cross_checks,
        conditions=[dict(condition=name, cases=32, target_hits=sum(r['hit'] for r in results if r['condition'] == name),
            **{field: sum(r[field] for r in results if r['condition'] == name) for field in
                ('q_transport_bytes', 'k_transport_bytes', 'v_transport_bytes', 'total_transport_bytes', 'bytes_saved_vs_FULL')},
            mean_transport_bytes=sum(r['total_transport_bytes'] for r in results if r['condition'] == name)/32)
            for name in local.CONDITIONS])
    return results, summary


def cross_check_c6(prepared, lengths, headers, accounting):
    """Only inspect manifest paths already hash-bound to C9-A; no directory scan."""
    manifests = []
    for path in prepared.input_hashes:
        if Path(path).name == 'manifest.json':
            value = provenance.read(path)
            if value.get('stage') == 'C6-B3-2': manifests.append((Path(path), value))
    if not manifests: return dict(status='NOT_AVAILABLE', reason='No C6-B3-2 transport artifact in the authoritative C9-A hash chain')
    require(len(manifests) == 1, 'Ambiguous canonical C6-B3-2 transport provenance')
    path, manifest = manifests[0]
    provenance.fields(manifest, dict(status='COMPLETE', revision=prepared.c9_manifest['model_revision'],
        evaluation_selection_sha256=provenance.SELECTION_SHA,
        plan_manifest_sha256=provenance.sha(prepared.args.plan_dir/'manifest.json'),
        semantic_workload_sha256=prepared.c8b_manifest['canonical_full_provenance']['semantic_workload_sha256']), 'C6 transport provenance')
    require(manifest['adapter_hashes'] == prepared.c8b_manifest['canonical_full_provenance']['adapter_hashes'], 'C6 frozen adapters differ')
    cases_path = path.parent/'per_case.csv'
    require(prepared.input_hashes.get(str(cases_path.resolve())) == manifest['output_hashes']['per_case.csv'], 'Unbound C6 transport cases')
    source = manifest.get('codec_source_hashes', {}).get('transport')
    if source:
        provenance.check_hash(source['path'], source['sha256'])
        prepared.input_hashes[str(Path(source['path']).resolve())] = source['sha256']
    canonical = prepared.c8b_manifest['teacher_forced_canonical_source']
    canonical_path = Path(canonical['per_case_path'])
    require(prepared.input_hashes.get(str(canonical_path.resolve())) == canonical['per_case_sha256'], 'Unbound canonical continuation artifact')
    # capability_per_case.csv is keyed by target source_id, NOT episode_id;
    # C6-B3-2 import_canonical_cases validates that same source/user binding.
    official = provenance.unique(provenance.read_csv(canonical_path), 'source_id')
    require(set(official) == {e['target_id'] for e in prepared.episodes}, 'Canonical continuation target identities differ')
    cases = provenance.read_csv(cases_path)
    checks = []
    for mode in ('RAW_SEMCACHE', 'STORAGE_KV_COMP'):
        selected = provenance.unique([r for r in cases if r['mode'] == mode], 'episode_id')
        require(set(selected) == set(lengths), 'C6 transport cohort mismatch: '+mode)
        for episode in prepared.episodes:
            episode_id = episode['episode_id']; row = selected[episode_id]
            require(row['logical_event_hash'] == provenance.digest(episode), 'C6 transport logical HIT differs')
            full = official[episode['target_id']]
            require(full['user'] == episode['target_user'] and int(full['history_depth']) == episode['history_depth'],
                'Canonical continuation user/history differs')
            continuation = json.loads(full['generated_token_ids'])
            require(isinstance(continuation, list) and 0 < len(continuation) <= 160
                and all(type(i) is int and i >= 0 for i in continuation), 'Invalid canonical continuation token shape')
            # C6 diagnostic phase is ONE teacher forward: prompt+canonical[:-1].
            roles, total, _ = accounting.bytes(lengths[episode_id]+len(continuation)-1, 3,
                headers[episode['target_user']], episode['target_user'])
            recorded = json.loads(row['diagnostic_transport_accounting'])
            require(all(recorded.get('raw_'+role+'_delta_bytes') == roles[role] for role in ROLES)
                and recorded.get('transmitted_total_including_cdf_bytes') == total,
                'C6 raw transport accounting disagrees: '+mode+'/'+episode_id)
            checks.append(dict(mode=mode, episode_id=episode_id, matched=True))
    return dict(status='VERIFIED', manifest_path=str(path), cases_path=str(cases_path),
        scope='raw uncompressed single teacher-forward delta payload; task source+greedy totals intentionally not compared to prompt-only C9', checks=checks)
