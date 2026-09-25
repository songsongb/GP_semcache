"""C0/C1 commands. Explicit execution only; models are loaded only by capture/quality."""
import argparse
import importlib.util
import json
import statistics
import shutil
import uuid
from pathlib import Path
from .common import (DIMENSIONS, MODES, SUMMARY_FIELDS, digest, environment, errors,
    file_hash, load_fixture, raw_bytes, read_json, sample_partitions, storage,
    summarize, validate_qkv, write_csv, write_json)
from .codecs import Baseline, Official, Unavailable, audit, stage_status, timing

C0_FIELDS = '''run_id repeat codec source layers tokens hidden_dim heads head_dim dtype
raw_kv_bytes encoded_payload_bytes metadata_bytes encoded_total_bytes compression_ratio
encode_wall_ms encode_cuda_ms decode_wall_ms decode_cuda_ms symbol_roundtrip_exact
k_max_abs k_rel_l2 k_cosine v_max_abs v_rel_l2 v_cosine device gpu_name cachegen_revision
provenance timing_protocol padding_tokens'''.split()
BLOCK_FIELDS = '''dataset query_id block_id start_position token_group_size layers hidden_dim heads
head_dim dtype compression_mode raw_q_bytes raw_kv_bytes raw_qkv_bytes encoded_kv_payload_bytes
encoded_metadata_bytes encoded_kv_total_bytes kv_compression_ratio semcache_total_stored_bytes
semcache_total_compression_ratio encode_ms decode_ms k_max_abs k_rel_l2 k_cosine v_max_abs
v_rel_l2 v_cosine anchor_count delta_token_count seed model_revision tokenizer_revision
cachegen_revision provenance status unavailable_reason padding_bytes symbol_roundtrip_exact
encode_cuda_ms decode_cuda_ms timing_protocol'''.split()
QUALITY_FIELDS = '''fixture_id compression_mode reused_tokens fresh_tokens kv_compression_ratio
semcache_compression_ratio max_logit_abs_error logit_rel_l2 kl_divergence argmax_agreement
uncompressed_reference_max_abs compression_only_logit_delta encode_ms decode_ms status
unavailable_reason provenance'''.split()
TIMING = 'separate wall/no-events and CUDA-event passes; synchronized; 5 warmup, 20 measured; quality outside timing'


def runtime_environment():
    env = environment()
    spec = importlib.util.find_spec('torchac_cuda')
    env.update(torchac_cuda_origin=spec.origin if spec else None,
               torchac_cuda_import_available=False if spec is None else None, torchac_cuda_import_error=None,
               cuda_compiler_path=shutil.which('nvcc'),
               torchac_cuda_binary_available=bool(spec and spec.origin and spec.origin.endswith(('.so', '.pyd'))),
               torchac_cuda_build_attempted=False,
               torchac_cuda_build_availability='NOT_TESTED (build forbidden during audit)')
    if spec and spec.origin and spec.origin.endswith(('.so', '.pyd')):
        try:
            __import__('torchac_cuda')
            env['torchac_cuda_import_available'] = True
        except (ImportError, OSError, RuntimeError) as exc:
            env['torchac_cuda_import_available'] = False
            env['torchac_cuda_import_error'] = str(exc)
    return env


def smoke(args):
    out = args.output_dir
    report = audit(args.cachegen_repo)
    env = runtime_environment()
    env.update(cachegen_revision=report['cachegen_revision'], serializer_import=report['serializer_import'],
               deserializer_import=report['deserializer_import'])
    write_json(out/'compatibility.json', report)
    write_json(out/'environment.json', env)
    manifest = dict(stage='C0', status='AUDIT_ONLY', compatibility_sha256=digest(report),
                    warmup=5, measured=20, timing_protocol=TIMING,
                    source='deterministic analytic C0 synthetic only; never C1', cases=[])
    rows = []
    write_csv(out/'c0_codec_smoke.csv', rows, C0_FIELDS)
    write_json(out/'manifest.json', manifest)
    if args.audit_only:
        print(json.dumps(report, indent=2))
        return
    try:
        import torch
        for tokens in args.tokens:
            codec = Official(report, args.codec_model, tokens)
            report.setdefault('runtime_cases', []).append(dict(model=args.codec_model, tokens=tokens,
                config=vars(codec.serializer.cachegen_config), constructor_accepted=True))
            # Analytic synthetic signal, solely for independent C0 codec validation.
            x = torch.arange(args.layers*tokens*args.hidden_dim, dtype=torch.float32,
                             device='cuda').reshape(args.layers, tokens, args.hidden_dim)
            k, v = (torch.sin(x*.013).half(), torch.cos(x*.017).half())
            encode = lambda: codec.encode(k, v, args.heads)
            for _ in range(5):
                codec.decode(encode())
            wall_rows = []
            for repeat in range(20):
                encoded, ew, _ = timing(encode, 'cuda')
                (dk, dv), dw, _ = timing(lambda: codec.decode(encoded), 'cuda')
                payload, metadata = codec.sizes(encoded)
                exact = codec.symbols_exact(k, v, encoded)
                if exact is False:
                    raise AssertionError('Arithmetic-code symbol roundtrip failed')
                row = dict(run_id=manifest.setdefault('run_id', str(uuid.uuid4())), repeat=repeat,
                    codec='CACHEGEN_FULL', source='C0_ANALYTIC_SYNTHETIC', layers=args.layers,
                    tokens=tokens, hidden_dim=args.hidden_dim, heads=args.heads,
                    head_dim=args.hidden_dim//args.heads, dtype='float16',
                    raw_kv_bytes=2*k.numel()*k.element_size(), encoded_payload_bytes=payload,
                    metadata_bytes=metadata, encoded_total_bytes=len(encoded),
                    compression_ratio=2*k.numel()*k.element_size()/len(encoded),
                    encode_wall_ms=ew, decode_wall_ms=dw, symbol_roundtrip_exact=exact,
                    **errors(k, v, dk, dv), device='cuda:0', gpu_name=env['gpu'],
                    cachegen_revision=report['cachegen_revision'], provenance='OFFICIAL_ARTIFACT; MEASURED',
                    timing_protocol=TIMING, padding_tokens=0)
                wall_rows.append(row)
            for _ in range(5):
                encoded, _, _ = timing(encode, 'cuda', True)
                timing(lambda: codec.decode(encoded), 'cuda', True)
            for row in wall_rows:
                encoded, _, ec = timing(encode, 'cuda', True)
                _, _, dc = timing(lambda: codec.decode(encoded), 'cuda', True)
                row.update(encode_cuda_ms=ec, decode_cuda_ms=dc)
            rows.extend(wall_rows)
            (out/f'c0_T{tokens}.bin').write_bytes(encoded)
            manifest['cases'].append(dict(tokens=tokens, model_metadata=args.codec_model,
                encoded_file=f'c0_T{tokens}.bin', encoded_sha256=file_hash(out/f'c0_T{tokens}.bin')))
        manifest['status'] = 'MEASURED'
        env['measurement_status'] = 'MEASURED'
    except Exception as exc:
        manifest.update(status='BLOCKED', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['compatibility_sha256'] = digest(report)
        write_csv(out/'c0_codec_smoke.csv', rows, C0_FIELDS)
        write_json(out/'manifest.json', manifest)
        write_json(out/'compatibility.json', report)
        write_json(out/'environment.json', env)


def model_config(args):
    return dict(name='facebook/opt-2.7b', tokenizer='facebook/opt-2.7b',
        revision=args.revision, tokenizer_revision=args.tokenizer_revision or args.revision,
        dtype='float16', device=args.device, local_files_only=True, attention_implementation='eager')


def capture(args):
    from semcache.experiments.dataset_adapters import load_source, normalize
    # Validate and persist sampling BEFORE loading any model.
    plan, sources = {}, {}
    for dataset in ('snips', 'multiwoz'):
        records, source = load_source(dataset, getattr(args, dataset), split=getattr(args, dataset+'_split'))
        normalized, dropped = normalize(dataset, records, split=getattr(args, dataset+'_split'))
        plan[dataset] = sample_partitions(normalized, args.queries_per_split, args.seed)
        sources[dataset] = dict(source=source, dropped=dropped)
    out = args.output_dir
    manifest = dict(stage='C1_CAPTURE', status='PLANNED', schema_version=1, seed=args.seed,
        sampling=plan, sampling_sha256=digest(plan), sources=sources, blocks=[],
        token_groups=args.tokens, max_tokens=32, model_config=model_config(args),
        scope='base_raw_unscaled_linear_projection; no LoRA adapter',
        provenance='REPRODUCTION_CHOICE', skipped=[])
    write_json(out/'capture_manifest.json', manifest)
    write_csv(out/'c1_quality.csv', [], QUALITY_FIELDS)
    write_csv(out/'c1_block_raw.csv', [], BLOCK_FIELDS)
    write_csv(out/'c1_summary.csv', [], SUMMARY_FIELDS)
    write_json(out/'manifest.json', dict(stage='C1', capture_status='PLANNED', benchmark_status='NOT_RUN', quality_status='NOT_RUN'))
    write_json(out/'environment.json', {'capture': environment()})
    if args.plan_only:
        return
    import torch
    from semcache.models.loader import load_model
    from semcache.models.capture import qkv_capture
    from semcache.utils.seed import seed_everything
    seed_everything(args.seed)
    model, tokenizer, meta = load_model(model_config(args))
    if (model.config.num_hidden_layers, model.config.hidden_size, model.config.num_attention_heads) != (32, 2560, 32):
        raise ValueError('Expected OPT-2.7B dimensions')
    if not meta.get('resolved_model_revision') or not meta.get('resolved_tokenizer_revision'):
        raise ValueError('Resolved model/tokenizer commits required')
    manifest['model_metadata'] = meta
    for dataset, partitions in plan.items():
        for partition in ('calibration', 'evaluation'):
            for query in partitions[partition]:
                ids = tokenizer(query['query_text'], truncation=True, max_length=32)['input_ids']
                with torch.inference_mode(), qkv_capture(model, storage_device='cpu', validate=False) as captured:
                    model(input_ids=torch.tensor([ids], device=args.device), use_cache=False)
                for tokens in args.tokens:
                    if len(ids) < tokens:
                        manifest['skipped'].append(dict(dataset=dataset, partition=partition,
                            query_id=query['source_id'], tokens=tokens, reason='query shorter than group; no padding'))
                    # Independent nonoverlapping physical blocks for each group size; discard tail.
                    for start in range(0, len(ids)-tokens+1, tokens):
                        block_id = digest([dataset, partition, query['source_id'], tokens, start])[:24]
                        qkv = {name: torch.stack([captured.records[l][name][0, start:start+tokens]
                                                 for l in range(32)]).contiguous() for name in 'qkv'}
                        validate_qkv(qkv, tokens)
                        relative = f'fixtures/{block_id}.pt'
                        path = out/relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(qkv, path)
                        manifest['blocks'].append(dict(dataset=dataset, partition=partition,
                            query_id=query['source_id'], source_id=query['source_id'],
                            source_group_id=query.get('conversation_id') or query['source_id'],
                            block_id=block_id, file=relative, sha256=file_hash(path), start_position=start,
                            absolute_positions=list(range(start, start+tokens)), token_ids=ids[start:start+tokens],
                            query_token_ids=ids, token_group_size=tokens, **DIMENSIONS, seed=args.seed,
                            model_revision=meta['resolved_model_revision'],
                            tokenizer_revision=meta['resolved_tokenizer_revision'],
                            provenance='REPRODUCTION_CHOICE; '+('RESEARCH_EXTENSION' if tokens == 3
                                else 'PAPER_REFERENCE_GRANULARITY'), padding_tokens=0))
    manifest['status'] = 'CAPTURED'
    write_json(out/'capture_manifest.json', manifest)
    write_json(out/'environment.json', {'capture': dict(environment(), measurement_status='CAPTURED')})
    write_json(out/'manifest.json', dict(stage='C1', capture_status='CAPTURED',
        capture_manifest_sha256=digest(manifest), benchmark_status='NOT_RUN', quality_status='NOT_RUN'))


def verify_capture(manifest):
    if manifest['status'] != 'CAPTURED':
        raise ValueError('Capture has not completed')
    if digest(manifest['sampling']) != manifest['sampling_sha256']:
        raise ValueError('Sampling manifest hash mismatch')
    for dataset, plan in manifest['sampling'].items():
        groups = []
        for partition in ('calibration', 'evaluation'):
            if digest(plan[partition]) != plan['hashes'][partition]:
                raise ValueError('Partition hash mismatch')
            groups.append({r.get('conversation_id') or r['source_id'] for r in plan[partition]})
        if groups[0] & groups[1]:
            raise ValueError('Calibration/evaluation source leakage')
    identities = set()
    metadata = manifest.get('model_metadata', {})
    for block in manifest['blocks']:
        if block['block_id'] in identities:
            raise ValueError('Duplicate block identity')
        identities.add(block['block_id'])
        queries = {r['source_id']: r for r in manifest['sampling'][block['dataset']][block['partition']]}
        t, s = block['token_group_size'], block['start_position']
        if (t not in (3, 10) or s < 0 or len(block['query_token_ids']) > 32
                or any(block.get(k) != v for k, v in DIMENSIONS.items())
                or block['model_revision'] != metadata.get('resolved_model_revision')
                or block['tokenizer_revision'] != metadata.get('resolved_tokenizer_revision')
                or block['query_id'] not in queries or block['absolute_positions'] != list(range(s, s+t))
                or block['token_ids'] != block['query_token_ids'][s:s+t] or len(block['token_ids']) != t):
            raise ValueError('Block query/position provenance mismatch')
        query = queries[block['query_id']]
        if block['source_group_id'] != (query.get('conversation_id') or query['source_id']):
            raise ValueError('Block source group mismatch')


def measured_baseline(codec, k, v, device):
    for _ in range(5):
        codec.decode(codec.encode(k, v))
    wall = []
    for _ in range(20):
        encoded, ew, _ = timing(lambda: codec.encode(k, v), device)
        decoded, dw, _ = timing(lambda: codec.decode(encoded), device)
        wall.append((ew, dw))
    cuda = []
    if str(device).startswith('cuda'):
        for repeat in range(25):
            encoded, _, ec = timing(lambda: codec.encode(k, v), device, True)
            decoded, _, dc = timing(lambda: codec.decode(encoded), device, True)
            if repeat >= 5:
                cuda.append((ec, dc))
    return encoded, decoded, dict(encode_ms=statistics.mean(x[0] for x in wall),
        decode_ms=statistics.mean(x[1] for x in wall),
        encode_cuda_ms=statistics.mean(x[0] for x in cuda) if cuda else None,
        decode_cuda_ms=statistics.mean(x[1] for x in cuda) if cuda else None), wall, cuda


def benchmark(args):
    import torch
    capture_manifest = read_json(args.capture_manifest)
    verify_capture(capture_manifest)
    root, out = args.capture_manifest.parent, args.output_dir
    report = audit(args.cachegen_repo)
    # Persist compatibility BEFORE any C1 codec execution; cannot bypass split policy.
    write_json(out/'compatibility.json', report)
    rows, reconstruction, repetitions = [], [], []
    for block in capture_manifest['blocks']:
        if block['partition'] != 'evaluation':
            continue
        qkv = load_fixture(root, block)
        k, v = (qkv[n].to(args.device) for n in 'kv')
        for mode in args.modes:
            row = {name: block[name] for name in ('dataset', 'query_id', 'block_id', 'start_position',
                'token_group_size', 'layers', 'hidden_dim', 'heads', 'head_dim', 'dtype', 'seed',
                'model_revision', 'tokenizer_revision', 'provenance')}
            row.update(compression_mode=mode, cachegen_revision=report['cachegen_revision'],
                **raw_bytes(block['token_group_size']), status='UNAVAILABLE', unavailable_reason=None,
                padding_bytes=0, anchor_count=None, delta_token_count=None, symbol_roundtrip_exact=None,
                timing_protocol=TIMING)
            available, reason = stage_status(mode, report)
            if not available:
                row['unavailable_reason'] = reason
                rows.append(row)
                continue
            codec = Baseline(mode)
            encoded, (dk, dv), times, wall, cuda = measured_baseline(codec, k, v, args.device)
            payload, metadata = codec.sizes(encoded)
            row.update(**storage(block['token_group_size'], payload, metadata), **times,
                       **errors(k, v, dk, dv), status='MEASURED')
            row['provenance'] += '; MEASURED; '+('REPRODUCTION_BASELINE' if mode == 'UNIFORM_INT8' else 'FP16_REFERENCE')
            rows.append(row)
            for repeat, (ew, dw) in enumerate(wall):
                repetitions.append(dict(block_id=block['block_id'], compression_mode=mode, repeat=repeat,
                    encode_wall_ms=ew, decode_wall_ms=dw,
                    encode_cuda_ms=cuda[repeat][0] if cuda else None,
                    decode_cuda_ms=cuda[repeat][1] if cuda else None))
            relative = f'reconstructed/{block["block_id"]}_{mode}.pt'
            path = out/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            # Q comes from the untouched original fixture; only K/V cross this boundary.
            torch.save(dict(k=dk.cpu(), v=dv.cpu()), path)
            reconstruction.append(dict(block_id=block['block_id'], compression_mode=mode,
                file=relative, sha256=file_hash(path), accounting=row))
    if not rows:
        raise ValueError('No evaluation blocks available')
    write_csv(out/'c1_block_raw.csv', rows, BLOCK_FIELDS)
    write_csv(out/'c1_summary.csv', summarize(rows), SUMMARY_FIELDS)
    write_csv(out/'c1_quality.csv', [], QUALITY_FIELDS)
    write_json(out/'reconstruction_manifest.json', dict(capture_manifest_sha256=digest(capture_manifest),
        reconstructed=reconstruction, modes=args.modes, compatibility=report))
    write_json(out/'timing_repeats.json', dict(timing_protocol=TIMING, repeats=repetitions))
    env = read_json(out/'environment.json') if (out/'environment.json').exists() else {}
    env['benchmark'] = dict(runtime_environment(), measurement_status='MEASURED', device=args.device)
    write_json(out/'environment.json', env)
    write_json(out/'manifest.json', dict(stage='C1', capture_manifest=str(args.capture_manifest.resolve()),
        capture_manifest_sha256=digest(capture_manifest), benchmark_status='COMPLETED', quality_status='NOT_RUN',
        compatibility_sha256=digest(report), modes={m: dict(zip(('available', 'reason'), stage_status(m, report))) for m in args.modes},
        warmup=5, measured=20, timing_protocol=TIMING,
        byte_scope='logical in-memory codec representation; excludes .pt transport envelope and shared experiment JSON',
        uniform_int8='Per [layer,token] vector: float32 scale=max(abs(x))/127 (zero=>1); ties-to-even round, clamp [-127,127], int8; float32 scales stored for K and V; FP16 reconstruction. Shape/dtype fixed by shared manifest.',
        padding_bytes=0, anchor_delta='Not applicable to baselines; official anchor/delta stage unavailable'))


def quality(args):
    import torch
    from contextlib import nullcontext
    from types import SimpleNamespace
    from semcache.models.loader import load_model
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.system_cost.base_projection import base_projection_path
    from semcache.system_cost.base_runtime import materialize_hits
    from semcache.metrics.m8 import exact_parity_passed, EXACT_PARITY_TOLERANCES
    from semcache.evaluation.logit_metrics import compare_logits
    from semcache.utils.seed import seed_everything
    manifest = read_json(args.capture_manifest)
    verify_capture(manifest)
    rec = read_json(args.reconstruction_manifest)
    if rec['capture_manifest_sha256'] != digest(manifest):
        raise ValueError('Reconstructions belong to a different capture')
    seed_everything(manifest['seed'])
    cfg = dict(manifest['model_config'], device=args.device)
    cfg.update(revision=manifest['model_metadata']['resolved_model_revision'],
               tokenizer_revision=manifest['model_metadata']['resolved_tokenizer_revision'])
    model, _, meta = load_model(cfg)
    if (meta['resolved_model_revision'] != cfg['revision'] or
            meta['resolved_tokenizer_revision'] != cfg['tokenizer_revision']):
        raise ValueError('Quality model/tokenizer commit mismatch')
    adapter = OPTModelAdapter(model)
    # Same query, same positions, same model: strict M8/M9-A mechanism isolation.
    # Choose one evaluation block per dataset/T by default (four total).
    selected, groups = [], set()
    for b in manifest['blocks']:
        group = (b['dataset'], b['token_group_size'])
        if b['partition'] == 'evaluation' and (args.all_blocks or group not in groups):
            selected.append(b)
            groups.add(group)
    if not selected:
        raise ValueError('No evaluation blocks available for quality')
    rows = []
    for block in selected:
        original = load_fixture(args.capture_manifest.parent, block)
        ids = torch.tensor([block['query_token_ids']], device=args.device)
        t, start = block['token_group_size'], block['start_position']
        def forward(qkv=None):
            hits = None
            if qkv is not None:
                captured = SimpleNamespace(projections={l: {n: qkv[n][l].unsqueeze(0) for n in 'qkv'}
                                                        for l in range(32)})
                plan = [dict(start=start, end=start+t, source_start=0, source_end=t,
                             token_ids=block['token_ids'], cache_key=(0, tuple(block['token_ids'])))]
                hits = materialize_hits(adapter, captured, plan, 'cpu')
            with torch.inference_mode(), (base_projection_path(adapter, hits, len(block['query_token_ids']))
                                          if hits is not None else nullcontext()) as projection_audit:
                logits = model(input_ids=ids, use_cache=False).logits.detach().cpu()
            if hits is not None and any(r['reused_projection_rows'] != t for r in projection_audit.records.values()):
                raise AssertionError('Physical reused token count differs from fixture')
            return logits
        native, raw = forward(), forward(original)
        reference = compare_logits(native, raw, start)
        reference_passed = exact_parity_passed(dict(reference,
            last_position_kl=reference['last_position_kl_baseline_to_injected'],
            argmax_agreement=reference['last_argmax_agreement']))
        def result(mode, logits, accounting, reused):
            metric = compare_logits(native, logits, start)
            return dict(fixture_id=block['block_id'], compression_mode=mode,
                reused_tokens=reused, fresh_tokens=len(block['query_token_ids'])-reused,
                kv_compression_ratio=accounting.get('kv_compression_ratio'),
                semcache_compression_ratio=accounting.get('semcache_total_compression_ratio'),
                max_logit_abs_error=metric['max_abs_logit_diff'], logit_rel_l2=metric['relative_l2_logit_diff'],
                kl_divergence=metric['affected_suffix_mean_kl'],
                argmax_agreement=float((native.argmax(-1) == logits.argmax(-1)).float().mean()),
                uncompressed_reference_max_abs=reference['max_abs_logit_diff'],
                compression_only_logit_delta=(logits.double()-raw.double()).abs().max().item() if reused else None,
                encode_ms=accounting.get('encode_ms'), decode_ms=accounting.get('decode_ms'), status='MEASURED',
                unavailable_reason=None, provenance='MEASURED; REPRODUCTION_CHOICE; same-query same-position M9-A bare-OPT physical reuse control')
        rows.append(result('NATIVE', native, {}, 0))
        raw_item = next((x for x in rec['reconstructed'] if x['block_id'] == block['block_id'] and x['compression_mode'] == 'FP16_RAW'), None)
        rows.append(result('FP16_RAW', raw, raw_item['accounting'] if raw_item else dict(
            kv_compression_ratio=1., semcache_total_compression_ratio=1.), t))
        if not reference_passed:
            rows[-1]['status'] = 'REFERENCE_FAILED'
            rows[-1]['unavailable_reason'] = 'Uncompressed reuse failed existing M8 strict parity thresholds'
        for mode in rec['modes']:
            if mode == 'FP16_RAW':
                continue
            available, reason = stage_status(mode, rec['compatibility'])
            if available and not reference_passed:
                available, reason = False, 'Uncompressed reference failed strict parity; compression quality not attributed'
            if not available:
                rows.append(dict(fixture_id=block['block_id'], compression_mode=mode, status='UNAVAILABLE', unavailable_reason=reason))
                continue
            item = next(x for x in rec['reconstructed'] if x['block_id'] == block['block_id'] and x['compression_mode'] == mode)
            path = args.reconstruction_manifest.parent/item['file']
            if file_hash(path) != item['sha256']:
                raise ValueError('Reconstruction SHA256 mismatch')
            kv = torch.load(path, map_location='cpu', weights_only=True)
            qkv = dict(q=original['q'], **kv)
            validate_qkv(qkv, t)
            rows.append(result(mode, forward(qkv), item['accounting'], t))
    write_csv(args.output_dir/'c1_quality.csv', rows, QUALITY_FIELDS)
    out_manifest = read_json(args.output_dir/'manifest.json') if (args.output_dir/'manifest.json').exists() else {}
    out_manifest.update(quality_status='REFERENCE_FAILED' if any(r['status'] == 'REFERENCE_FAILED' for r in rows) else 'MEASURED', quality_fixture_count=len(selected),
        quality_scope='existing M9-A base_projection_path; actual same-position projection skipping; bare OPT only, no personalized output claim',
        kl_direction='native || candidate; mean over affected suffix',
        argmax_scope='all input positions', reference_tolerances=EXACT_PARITY_TOLERANCES,
        compression_only_logit_delta='max abs(candidate logits - uncompressed reuse logits)')
    write_json(args.output_dir/'manifest.json', out_manifest)
    env = read_json(args.output_dir/'environment.json') if (args.output_dir/'environment.json').exists() else {}
    env['quality'] = dict(environment(), measurement_status='MEASURED', model_metadata=meta)
    write_json(args.output_dir/'environment.json', env)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    c0 = commands.add_parser('c0')
    c0.add_argument('--cachegen-repo', type=Path)
    c0.add_argument('--audit-only', action='store_true')
    c0.add_argument('--codec-model', default='lmsys/longchat-7b-16k')
    c0.add_argument('--tokens', type=int, nargs='+', default=[3, 10])
    c0.add_argument('--layers', type=int, default=32)
    c0.add_argument('--hidden-dim', type=int, default=4096)
    c0.add_argument('--heads', type=int, default=32)
    c0.add_argument('--output-dir', type=Path, default=Path('results/cachegen/c0'))
    cap = commands.add_parser('capture')
    for dataset in ('snips', 'multiwoz'):
        cap.add_argument('--'+dataset, type=Path, required=True)
        cap.add_argument('--'+dataset+'-split', default='train_full' if dataset == 'snips' else 'train')
    cap.add_argument('--revision', required=True)
    cap.add_argument('--tokenizer-revision')
    cap.add_argument('--queries-per-split', type=int, default=128)
    cap.add_argument('--seed', type=int, default=42)
    cap.add_argument('--tokens', type=int, nargs='+', choices=[3, 10], default=[3, 10])
    cap.add_argument('--plan-only', action='store_true')
    cap.add_argument('--device', default='cuda')
    cap.add_argument('--output-dir', type=Path, default=Path('results/cachegen/c1'))
    bench = commands.add_parser('benchmark')
    bench.add_argument('--capture-manifest', type=Path, required=True)
    bench.add_argument('--cachegen-repo', type=Path)
    bench.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    bench.add_argument('--device', default='cpu')
    bench.add_argument('--output-dir', type=Path, default=Path('results/cachegen/c1'))
    qual = commands.add_parser('quality')
    qual.add_argument('--capture-manifest', type=Path, required=True)
    qual.add_argument('--reconstruction-manifest', type=Path, required=True)
    qual.add_argument('--device', default='cuda')
    qual.add_argument('--all-blocks', action='store_true')
    qual.add_argument('--output-dir', type=Path, default=Path('results/cachegen/c1'))
    args = p.parse_args(argv)
    if args.command == 'c0' and (min(args.tokens) <= 0 or args.layers <= 0 or args.heads <= 0
                               or args.hidden_dim <= 0 or args.hidden_dim % args.heads):
        p.error('Invalid codec dimensions')
    if hasattr(args, 'tokens') and len(args.tokens) != len(set(args.tokens)):
        p.error('Duplicate token-group sizes')
    if hasattr(args, 'modes') and len(args.modes) != len(set(args.modes)):
        p.error('Duplicate compression modes')
    if hasattr(args, 'device') and args.device not in ('cpu', 'cuda', 'cuda:0'):
        p.error('Use cpu or cuda:0; select physical GPU with CUDA_VISIBLE_DEVICES')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    {'c0': smoke, 'capture': capture, 'benchmark': benchmark, 'quality': quality}[args.command](args)
