"""C4 controlled bare-OPT quality validation for one frozen exact-repeat hit.

Model imports and inference occur only in run_quality(), never in select/dry-run.
The existing base_projection_path is the validated bare-OPT SemCache reuse path.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import time

from semcache.cache.global_cache import GlobalCache
from semcache.experiments.cachegen.b2 import format as fmt
from semcache.experiments.cachegen.c2.physical_storage import (
    FrozenK20V16Codec, MODE_COMPRESSED, POLICY, PROFILE_SHA256,
)
from semcache.experiments.cachegen.common import file_hash, write_csv, write_json
from semcache.experiments.cachegen.c15c.harness import ROOT, git_state
from semcache.experiments.m9b_semantic_workload import MODEL_ID, MODEL_REVISION
from semcache.semantic.matcher import ExactTokenMatcher
from semcache.semantic.subsequence import Subsequence, SubsequenceExtractor
from semcache.simulation.multi_user import digest, read_workload, safety_eligible

OUTPUT = ROOT/'results/cachegen/c4/quality_validation'
CONDITIONS = ('FULL_RECOMPUTE', 'RAW_REUSE', 'COMPRESSED_REUSE')
PAIRS = (('RAW_REUSE', 'COMPRESSED_REUSE'),
         ('FULL_RECOMPUTE', 'RAW_REUSE'),
         ('FULL_RECOMPUTE', 'COMPRESSED_REUSE'))
WINDOW = 3
MAX_PROMPT_TOKENS = 32
LOGICAL_CAPACITY_BYTES = 64*1024**2


def verify_profile(path):
    path = Path(path)
    if not path.is_file() or file_hash(path) != PROFILE_SHA256:
        raise ValueError('Frozen K20/V16 profile missing or SHA256 mismatch')
    return dict(path=str(path.resolve()), sha256=PROFILE_SHA256, bytes=path.stat().st_size)


def validate_case(case):
    """Full-prompt same-owner exact repeat; one position-aligned w=3 hit."""
    ids = case['target_token_ids']
    source_ids = case['source_token_ids']
    start = case['reuse_start']
    block = tuple(case['reused_token_ids'])
    if (case['source_id'] == case['target_id'] or ids != source_ids or
            type(start) is not int or start != 0 or len(block) != WINDOW or
            len(ids) < WINDOW+2 or len(ids) > MAX_PROMPT_TOKENS or
            tuple(ids[start:start+WINDOW]) != block or
            tuple(source_ids[start:start+WINDOW]) != block or
            case['cache_key'] != (case['cluster_id'], block)):
        raise ValueError('C4 requires one same-position exact-repeat w=3 block and post-block logits')
    owner = case['logical_user']
    prompt_hash = digest(ids)
    source = dict(user_id=owner, adapter_id='base_opt', prompt_hash=prompt_hash, start=start)
    target = dict(user_id=owner, adapter_id='base_opt', prompt_hash=prompt_hash)
    evidence = {(owner, 'base_opt', prompt_hash, start)}
    if not safety_eligible(source, target, start, evidence):
        raise ValueError('Exact-token safety evidence rejected')
    windows = SubsequenceExtractor(WINDOW).extract(ids)
    if not windows or ExactTokenMatcher().key(case['cluster_id'], windows[0]) != case['cache_key']:
        raise ValueError('SemCache matcher differs from frozen C4 key')
    return True


def select_cases(rows, dataset, limit):
    if type(limit) is not int or not 1 <= limit <= 16:
        raise ValueError('C4 is limited to 1..16 cases per dataset')
    selected, skipped, seen_ids = [], defaultdict(int), set()
    for row in rows:
        if len(selected) == limit:
            break
        if row.get('dataset') != dataset or row.get('model_id') != MODEL_ID:
            raise ValueError('Prepared workload namespace mismatch')
        if row.get('model_revision') not in (None, MODEL_REVISION):
            raise ValueError('Prepared workload model revision mismatch')
        ids, source_id = row['token_ids'], str(row['source_id'])
        if source_id in seen_ids:
            skipped['duplicate_source_id'] += 1
            continue
        if not WINDOW+2 <= len(ids) <= MAX_PROMPT_TOKENS:
            skipped['prompt_length_outside_5_to_32'] += 1
            continue
        if not isinstance(row.get('query_text'), str) or not row['query_text'].strip():
            skipped['missing_prepared_text'] += 1
            continue
        seen_ids.add(source_id)
        block = tuple(ids[:WINDOW])
        case = dict(case_id=f'{dataset}:{source_id}:exact_repeat', dataset=dataset,
            source_id=source_id, target_id=f'{source_id}::c4_exact_repeat',
            cluster_id=row['cluster_id'], cache_key=(row['cluster_id'], block),
            source_token_ids=list(ids), target_token_ids=list(ids), reused_token_ids=list(block),
            reuse_start=0, reuse_end=WINDOW, logical_user=f'c4_{dataset}_base',
            query_text=row['query_text'], source_workload_index=row.get('global_query_index'),
            query_text_sha256=hashlib.sha256(row['query_text'].encode('utf-8')).hexdigest(),
            selection_kind='same_prepared_query_cold_then_exact_repeat')
        validate_case(case)
        selected.append(case)
    return selected, dict(skipped)


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered)-1)*fraction
    low = int(position)
    return ordered[low]+(ordered[min(low+1, len(ordered)-1)]-ordered[low])*(position-low)


def logit_comparison(reference, candidate, start):
    """Per-position stable float32 distributions; no full probability cube."""
    import torch
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[0] != 1:
        raise ValueError('Paired [1, prompt, vocab] logits required')
    if not 0 <= start < reference.shape[1]:
        raise ValueError('At least one post-reuse logit position required')
    kls, cosines, top1, top5, abs_sum, elements, max_abs = [], [], 0, 0., 0., 0, 0.
    for position in range(start, reference.shape[1]):
        a, b = reference[0, position].float(), candidate[0, position].float()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Nonfinite logits')
        la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
        kl = torch.sum(la.exp()*(la-lb)).item()
        kls.append(max(0., kl))  # roundoff may make a tiny negative KL
        top1 += int(a.argmax().item() == b.argmax().item())
        k = min(5, a.numel())
        top5 += len(set(a.topk(k).indices.tolist()) & set(b.topk(k).indices.tolist()))/k
        an, bn = torch.linalg.vector_norm(a).item(), torch.linalg.vector_norm(b).item()
        cosines.append(torch.dot(a, b).item()/(an*bn) if an*bn else (1. if an == bn == 0 else 0.))
        delta = (a-b).abs()
        abs_sum += delta.sum().item()
        elements += delta.numel()
        max_abs = max(max_abs, delta.max().item())
    return dict(position_count=len(kls), mean_kl=statistics.mean(kls), median_kl=statistics.median(kls),
        p95_kl=percentile(kls, .95), top1_agreement=top1/len(kls),
        top5_set_overlap=top5/len(kls), logit_cosine=statistics.mean(cosines),
        mean_absolute_logit_difference=abs_sum/elements, max_absolute_logit_difference=max_abs,
        _kl_values=kls)


def observed_suffix_nll(logits, token_ids, start):
    """NLL of observed prompt suffix tokens with predictors after the reused block."""
    import torch
    if logits.shape[1] != len(token_ids):
        raise ValueError('Logits/token sequence length mismatch')
    values = []
    for position in range(start, len(token_ids)-1):
        logp = torch.log_softmax(logits[0, position].float(), -1)
        values.append(-logp[token_ids[position+1]].item())
    return statistics.mean(values) if values else None


def generation_comparison(reference, candidate):
    a, b = list(reference), list(candidate)
    maximum = max(len(a), len(b))
    if maximum == 0:
        raise ValueError('Nonempty generation required')
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j]+1, current[j-1]+1, previous[j-1]+(x != y)))
        previous = current
    return dict(exact_sequence_match=a == b, prefix_agreement_length=prefix,
        token_position_agreement=sum(x == y for x, y in zip(a, b))/maximum,
        normalized_token_edit_distance=previous[-1]/maximum)


def dry_run(args):
    profile = verify_profile(args.profile_path)
    if args.coder_backend != fmt.FAST_CODER:
        raise ValueError('C4 requires frozen FAST_PY_BITEXACT backend')
    paths = {'snips': Path(args.snips), 'multiwoz': Path(args.multiwoz)}
    rows = {dataset: read_workload(path, dataset) for dataset, path in paths.items()}
    cases, skipped = {}, {}
    for dataset, workload in rows.items():
        cases[dataset], skipped[dataset] = select_cases(workload, dataset, args.per_dataset)
        print(f"{dataset}: selected={len(cases[dataset])}/{args.per_dataset} "
              f"one_exact_w3_hit=true skipped={skipped[dataset]}")
        if not cases[dataset]:
            raise ValueError(f'No valid exact-repeat prepared cases for {dataset}')
    selected = [case for dataset in ('snips', 'multiwoz') for case in cases[dataset]]
    for case in selected:
        validate_case(case)
    print(f'profile_sha256={profile["sha256"]}; coder_backend={args.coder_backend}')
    print(f'expected controlled teacher-forced inference evaluations={3*len(selected)}')
    print(f'expected greedy decode steps at most={3*len(selected)*args.max_new_tokens}')
    if len(selected) < 2*args.per_dataset:
        print('Fewer than requested valid prepared cases; selected deterministic maximum.')
    return dict(cases=selected, skipped=skipped, profile=profile,
        workload_sha256={dataset: file_hash(path) for dataset, path in paths.items()},
        workload_paths={dataset: str(path.resolve()) for dataset, path in paths.items()},
        workload_counts={dataset: len(rows[dataset]) for dataset in rows})


def _sync(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


def _elapsed(start, device):
    _sync(device)
    return 1000*(time.perf_counter()-start)


def _make_caches(case, blocks, codec):
    """Insert the same frozen key once in each real GlobalCache instance."""
    from semcache.experiments.cachegen.c2.physical_storage import MODE_RAW
    common = dict(component_scope='base_qkv_latency_only', source_query_id=case['source_id'],
                  source_user=case['logical_user'], source_adapter='base_opt',
                  source_query_token_ids=tuple(case['source_token_ids']))
    raw = GlobalCache(LOGICAL_CAPACITY_BYTES, physical_storage_mode=MODE_RAW)
    comp = GlobalCache(LOGICAL_CAPACITY_BYTES, physical_storage_mode=MODE_COMPRESSED,
                       physical_codec=codec, instrument_storage=True)
    records = {}
    for mode, cache in (('RAW_REUSE', raw), ('COMPRESSED_REUSE', comp)):
        start = time.perf_counter()
        physical = cache.make_entry(case['cluster_id'], tuple(case['reused_token_ids']),
            (case['reuse_start'], case['reuse_end']), blocks, 'cpu')
        physical.qkv_metadata = dict(common)
        inserted = cache.insert(physical, observed_frequency=1)
        if not inserted or set(cache.entries) != {case['cache_key']} or len(cache.entries) != 1:
            raise ValueError('C4 raw/compressed admission or logical state differs')
        records[mode] = (cache, physical, _elapsed(start, codec.quantization_device))
    if records['RAW_REUSE'][1].size_bytes != records['COMPRESSED_REUSE'][1].size_bytes:
        raise ValueError('Raw/compressed logical capacity differs')
    if (set(raw.entries) != set(comp.entries) or raw.hits != comp.hits or
            raw.misses != comp.misses or raw.logical_cache_bytes != comp.logical_cache_bytes):
        raise ValueError('Raw/compressed logical cache state differs')
    if records['COMPRESSED_REUSE'][1].tensors is not None or records['COMPRESSED_REUSE'][1].compressed_kv is None:
        raise ValueError('Compressed resident retains raw K/V or lacks B2 payload')
    return records


def _lookup_one(case, cache):
    from semcache.semantic.hit_selection import CacheHit, select_nonoverlapping
    key = case['cache_key']
    start, end = case['reuse_start'], case['reuse_end']
    entry = cache.lookup(key, record_reuse=False)
    if entry is None or entry.key != key:
        raise ValueError('Frozen C4 logical hit missing')
    hit = CacheHit(Subsequence(tuple(case['reused_token_ids']), start, end), entry)
    selected, mask = select_nonoverlapping([hit], len(case['target_token_ids']))
    if len(selected) != 1 or sum(mask) != WINDOW or selected[0].entry.key != key:
        raise ValueError('C4 must execute exactly one frozen w=3 hit')
    return selected[0]


def _verify_codec(case, blocks, raw_hit, compressed_hit, resident, device):
    import torch
    if resident.tensors is not None or resident.compressed_kv.profile_sha256 != PROFILE_SHA256:
        raise ValueError('Compressed resident is not frozen B2 storage')
    if raw_hit.window != compressed_hit.window or raw_hit.entry.key != compressed_hit.entry.key:
        raise ValueError('RAW and COMPRESSED hits differ')
    original = {role: torch.cat([blocks[layer]['qkv'.index(role.lower())] for layer in range(32)], 0).to(device)
                for role in ('K', 'V')}
    direct = {role: POLICY.quantize(original[role], role) for role in ('K', 'V')}
    for index, role in enumerate(('K', 'V')):
        if not torch.equal(compressed_hit.entry.kv_symbols[index], direct[role].symbols):
            raise ValueError(f'{role} decoded integer symbol mismatch')
        reconstructed = torch.cat([compressed_hit.entry.tensors[layer]['qkv'.index(role.lower())]
                                   for layer in range(32)], 0)
        if (reconstructed.shape != original[role].shape or reconstructed.dtype != torch.float16 or
                not torch.equal(reconstructed, direct[role].reconstructed) or
                not torch.isfinite(reconstructed).all()):
            raise ValueError(f'{role} reconstructed tensor differs from direct quantization')
    for layer in range(32):
        a, b = raw_hit.entry.tensors[layer][0], compressed_hit.entry.tensors[layer][0]
        if not torch.equal(a, b) or not torch.isfinite(a).all():
            raise ValueError('Q differs between RAW and COMPRESSED reuse')


def _forward_and_generate(model, adapter, ids, hit, max_new_tokens, device, *, capture_source=False):
    import torch
    from semcache.models.capture import qkv_capture
    from semcache.system_cost.base_projection import base_projection_path
    from semcache.semantic.hit_selection import CacheHit
    if hit is not None and (not isinstance(hit, CacheHit) or len(hit.window.token_ids) != WINDOW):
        raise ValueError('One w=3 CacheHit or FULL recompute required')
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    start = time.perf_counter()
    with torch.inference_mode(), (qkv_capture(model, storage_device='cpu', validate=False)
                                  if capture_source else nullcontext(None)) as captured, (
                                  base_projection_path(adapter, [hit], len(ids)) if hit is not None else nullcontext()) as audit:
        output = model(input_ids=tokens, use_cache=True)
    forward_ms = _elapsed(start, device)
    if hit is not None and (audit is None or len(audit.records) != 96 or
                            any(record['reused_projection_rows'] != WINDOW for record in audit.records.values())):
        raise ValueError('Projection reuse did not cover exactly one block in all OPT layers')
    logits = output.logits.detach().cpu()
    past, next_logits, generated = output.past_key_values, output.logits[:, -1], []
    start = time.perf_counter()
    with torch.inference_mode():
        for step in range(max_new_tokens):
            next_token = next_logits.argmax(-1, keepdim=True)
            token_id = int(next_token.item())
            generated.append(token_id)
            if token_id == model.config.eos_token_id:
                break
            if step+1 < max_new_tokens:
                output = model(input_ids=next_token, past_key_values=past, use_cache=True)
                past, next_logits = output.past_key_values, output.logits[:, -1]
    generation_ms = _elapsed(start, device)
    peak = dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device)) if str(device).startswith('cuda') else dict(
                peak_cuda_allocated_bytes=None, peak_cuda_reserved_bytes=None)
    return logits, generated, dict(model_forward_ms=forward_ms, generation_ms=generation_ms,
                                    **peak), captured


def run_case(case, model, tokenizer, adapter, codec, max_new_tokens, device):
    import torch
    validate_case(case)
    tokenized = list(tokenizer(case['query_text'], add_special_tokens=True, truncation=False)['input_ids'])
    if tokenized != case['target_token_ids']:
        raise ValueError('Live tokenizer differs from prepared exact-token workload')
    timings, logits, generated = {}, {}, {}
    start_total = time.perf_counter()
    logits['FULL_RECOMPUTE'], generated['FULL_RECOMPUTE'], timings['FULL_RECOMPUTE'], captured = (
        _forward_and_generate(model, adapter, tokenized, None, max_new_tokens, device, capture_source=True))
    timings['FULL_RECOMPUTE'].update(cache_insert_ms=0., cache_lookup_ms=0., quantize_ms=0.,
        encode_ms=0., decode_ms=0., dequantize_ms=0., total_request_ms=_elapsed(start_total, device))
    blocks = {layer: tuple(captured.records[layer][name][:, :WINDOW].contiguous() for name in 'qkv')
              for layer in range(32)}
    caches = _make_caches(case, blocks, codec)
    hits = {}
    for mode in ('RAW_REUSE', 'COMPRESSED_REUSE'):
        cache, resident, insert_ms = caches[mode]
        start = time.perf_counter()
        hit = _lookup_one(case, cache)
        lookup_ms = _elapsed(start, device)
        hits[mode] = hit
        if mode == 'COMPRESSED_REUSE':
            _verify_codec(case, blocks, hits['RAW_REUSE'], hit, resident, device)
        logits[mode], generated[mode], timings[mode], _ = _forward_and_generate(
            model, adapter, tokenized, hit, max_new_tokens, device)
        codec_timing = resident.storage_timings_ms if mode == 'COMPRESSED_REUSE' else {}
        view_timing = hit.entry.storage_timings_ms if mode == 'COMPRESSED_REUSE' else {}
        timings[mode].update(cache_insert_ms=insert_ms, cache_lookup_ms=lookup_ms,
            quantize_ms=codec_timing.get('quantize_ms', 0.) or 0.,
            encode_ms=codec_timing.get('encode_ms', 0.) or 0.,
            decode_ms=view_timing.get('decode_ms', 0.) or 0.,
            dequantize_ms=view_timing.get('dequantize_ms', 0.) or 0.,
            total_request_ms=insert_ms+lookup_ms+timings[mode]['model_forward_ms']+
                             timings[mode]['generation_ms'])
    if (caches['RAW_REUSE'][0].hits != caches['COMPRESSED_REUSE'][0].hits or
            caches['RAW_REUSE'][0].misses != caches['COMPRESSED_REUSE'][0].misses or
            set(caches['RAW_REUSE'][0].entries) != set(caches['COMPRESSED_REUSE'][0].entries)):
        raise ValueError('RAW and COMPRESSED cache hit/state mismatch')
    comparisons = {}
    for a, b in PAIRS:
        item = logit_comparison(logits[a], logits[b], WINDOW)
        kls = item.pop('_kl_values')
        comparisons[f'{a}_vs_{b}'] = dict(**item, generation=generation_comparison(generated[a], generated[b]),
                                         _kl_values=kls)
    nll = {mode: observed_suffix_nll(logits[mode], tokenized, WINDOW) for mode in CONDITIONS}
    nll.update(raw_minus_full=nll['RAW_REUSE']-nll['FULL_RECOMPUTE'],
               compressed_minus_raw=nll['COMPRESSED_REUSE']-nll['RAW_REUSE'],
               compressed_minus_full=nll['COMPRESSED_REUSE']-nll['FULL_RECOMPUTE'])
    del captured, blocks, caches, hits, logits
    return dict(case_id=case['case_id'], dataset=case['dataset'], source_id=case['source_id'],
        target_id=case['target_id'], reused_token_ids=case['reused_token_ids'],
        cache_key=case['cache_key'], reuse_position=case['reuse_start'], reuse_end=case['reuse_end'],
        source_workload_index=case['source_workload_index'], query_text_sha256=case['query_text_sha256'],
        prompt_tokens=len(tokenized), timings=timings, comparisons=comparisons,
        observed_prompt_suffix_nll=nll, observed_suffix_label_count=len(tokenized)-WINDOW-1,
        generated_token_ids=generated,
        symbol_mismatches=0, reconstruction_failures=0, logical_hit_mismatches=0)


def _flatten_case(row):
    flat = {k: row[k] for k in ('case_id', 'dataset', 'source_id', 'target_id', 'reused_token_ids',
                                'cache_key', 'reuse_position', 'reuse_end', 'source_workload_index',
                                'query_text_sha256', 'prompt_tokens', 'observed_suffix_label_count',
                                'symbol_mismatches',
                                'reconstruction_failures', 'logical_hit_mismatches')}
    flat['reused_token_ids'] = json.dumps(flat['reused_token_ids'])
    flat['cache_key'] = json.dumps(flat['cache_key'])
    for mode, values in row['timings'].items():
        flat.update({f'{mode}_{k}': v for k, v in values.items()})
    for pair, values in row['comparisons'].items():
        flat.update({f'{pair}_{k}': v for k, v in values.items() if k not in ('generation', '_kl_values')})
        flat.update({f'{pair}_generation_{k}': v for k, v in values['generation'].items()})
    flat.update({f'nll_{k}': v for k, v in row['observed_prompt_suffix_nll'].items()})
    return flat


def summarize(rows):
    result = {}
    for dataset in ('snips', 'multiwoz', 'overall'):
        group = [r for r in rows if dataset == 'overall' or r['dataset'] == dataset]
        if not group:
            continue
        comparisons = {}
        for pair in (f'{a}_vs_{b}' for a, b in PAIRS):
            items = [r['comparisons'][pair] for r in group]
            kls = [x for item in items for x in item['_kl_values']]
            positions = sum(i['position_count'] for i in items)
            pooled = lambda key: sum(i[key]*i['position_count'] for i in items)/positions
            comparisons[pair] = dict(position_count=len(kls), mean_kl=statistics.mean(kls),
                median_kl=statistics.median(kls), p95_kl=percentile(kls, .95),
                top1_agreement=pooled('top1_agreement'),
                top5_set_overlap=pooled('top5_set_overlap'),
                logit_cosine=pooled('logit_cosine'),
                mean_absolute_logit_difference=pooled('mean_absolute_logit_difference'),
                max_absolute_logit_difference=max(i['max_absolute_logit_difference'] for i in items),
                generation_exact_match_rate=statistics.mean(i['generation']['exact_sequence_match'] for i in items),
                generation_mean_prefix_agreement=statistics.mean(i['generation']['prefix_agreement_length'] for i in items),
                generation_token_position_agreement=statistics.mean(i['generation']['token_position_agreement'] for i in items),
                generation_normalized_edit_distance=statistics.mean(i['generation']['normalized_token_edit_distance'] for i in items))
        result[dataset] = dict(case_count=len(group), comparisons=comparisons,
            observed_prompt_suffix_nll={mode: sum(r['observed_prompt_suffix_nll'][mode]*r['observed_suffix_label_count']
                                                  for r in group)/sum(r['observed_suffix_label_count'] for r in group)
                                        for mode in (*CONDITIONS, 'raw_minus_full', 'compressed_minus_raw',
                                                     'compressed_minus_full')})
    return result


def run_quality(args):
    plan = dry_run(args)
    if args.dry_run:
        return plan
    import torch
    from semcache.models.loader import load_model
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.utils.seed import seed_everything
    output = Path(args.output_dir).resolve()
    if output != OUTPUT.resolve() and not output.is_relative_to(OUTPUT.resolve()):
        raise ValueError('C4 outputs must remain under results/cachegen/c4/quality_validation')
    if (output/'manifest.json').exists():
        raise ValueError('Existing C4 result would be overwritten')
    if not str(args.device).startswith('cuda:') or not torch.cuda.is_available():
        raise ValueError('Real C4 requires SERAPH CUDA')
    seed_everything(args.seed)
    config = dict(name=MODEL_ID, tokenizer=MODEL_ID, revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION, dtype='float16', device=args.device,
        local_files_only=True, attention_implementation='eager')
    model, tokenizer, model_metadata = load_model(config)
    if (model_metadata['resolved_model_revision'] != MODEL_REVISION or
            model_metadata['resolved_tokenizer_revision'] != MODEL_REVISION or
            model.config.num_hidden_layers != 32 or model.config.hidden_size != 2560):
        raise ValueError('Loaded OPT/tokenizer revision or dimensions differ from prepared workload')
    adapter = OPTModelAdapter(model)
    codec = FrozenK20V16Codec(args.profile_path, quantization_device=args.device,
        decode_device=args.device, instrument=True, coder_backend=args.coder_backend)
    if codec.profile.sha256 != PROFILE_SHA256:
        raise ValueError('Frozen profile changed')
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(stage='C4-QUALITY-VALIDATION', status='RUNNING', git=git_state(),
        model=MODEL_ID, model_revision=model_metadata['resolved_model_revision'],
        tokenizer=MODEL_ID, tokenizer_revision=model_metadata['resolved_tokenizer_revision'],
        compression_policy=POLICY.name, transform='B2_ANCHOR_MOD_RESIDUAL_KV',
        profile=plan['profile'], coder_backend=args.coder_backend,
        selected_case_ids=[c['case_id'] for c in plan['cases']],
        selected_cases=[{k: c[k] for k in ('case_id', 'dataset', 'source_id', 'target_id',
            'source_workload_index', 'query_text_sha256', 'reused_token_ids', 'cache_key',
            'reuse_start', 'reuse_end')} for c in plan['cases']],
        dataset_counts={d: sum(c['dataset'] == d for c in plan['cases']) for d in ('snips','multiwoz')},
        selection_rule='First prepared rows in source order, unique source IDs, 5..32 OPT tokens; cold then exact repeat of same prepared query; first w=3 span only',
        exact_token_safety='Same full prompt, owner, base adapter, token IDs and source/target position; explicit evidence validated by safety_eligible',
        one_reuse_block_constraint=True, workload_sha256=plan['workload_sha256'],
        workload_paths=plan['workload_paths'], workload_counts=plan['workload_counts'],
        inference_config=dict(teacher_forced_prompt=True, use_cache=True, dtype='float16',
                              attention_implementation='eager', logical_cache_capacity_bytes=LOGICAL_CAPACITY_BYTES,
                              logical_resident_entries_per_reuse_mode=1),
        generation_config=dict(strategy='greedy', do_sample=False, max_new_tokens=args.max_new_tokens),
        comparison_scope='Logits at prompt positions >= reuse_end; KL is first named condition || second named condition; aggregates pool positions',
        nll_scope='Observed prepared prompt suffix: next-token labels at positions after the reused block; not task-answer NLL',
        timing_scope='FULL forward includes QKV capture for cold source; RAW/COMP total includes lookup, forward, greedy generation and applicable codec work',
        device=args.device, software=dict(torch=torch.__version__, transformers=model_metadata['transformers_version']),
        profile_frozen=True, cdf_fit_calls=0, completed_cases=0)
    write_json(output/'manifest.json', manifest)
    rows = []
    try:
        for index, case in enumerate(plan['cases'], 1):
            row = run_case(case, model, tokenizer, adapter, codec, args.max_new_tokens, args.device)
            rows.append(row)
            manifest['completed_cases'] = index
            write_json(output/'manifest.json', manifest)
            print(f'[{index}/{len(plan["cases"])}] {case["case_id"]}', flush=True)
        flat = [_flatten_case(row) for row in rows]
        write_csv(output/'case_results.csv', flat, flat[0].keys())
        write_json(output/'quality_summary.json', summarize(rows))
        manifest.update(status='COMPLETED', output_sha256={name: file_hash(output/name)
            for name in ('case_results.csv', 'quality_summary.json')})
    except BaseException as exc:
        manifest.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output/'manifest.json', manifest)
    return summarize(rows)
