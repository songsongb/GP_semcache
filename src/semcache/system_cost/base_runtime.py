"""Opt-in ES latency profiling runtime. No torch/model work at import time."""
from contextlib import nullcontext
import platform
from .es_base_profile import assert_no_lora, validate_fresh_m85, MEASUREMENT_LABEL
from .profiles import pair_key


def span_plan(row, token_ids):
    """Replay actual physical hits; never infer reused rows from index HIT count."""
    from semcache.metrics.m8 import token_ids_sha256
    if len(token_ids) != row['prompt_tokens'] or token_ids_sha256(token_ids) != row['prompt_token_ids_sha256']:
        raise ValueError('Reconstructed prompt differs from M8 token IDs')
    used, plan = set(), []
    for item in row.get('reuse_block_provenance') or []:
        start, end = item['destination_start'], item['destination_end']
        src_start, src_end = item['source_start'], item['source_end']
        if (any(type(x) is not int for x in (start, end, src_start, src_end))
                or not 0 <= start < end <= len(token_ids)
                or not 0 <= src_start < src_end <= len(token_ids)
                or end-start != src_end-src_start or used.intersection(range(start, end))
                or list(token_ids[start:end]) != item['token_ids']
                or list(token_ids[src_start:src_end]) != item['token_ids']):
            raise ValueError('Invalid/overlapping M8 physical span provenance')
        if (item.get('source_query_id') != 'cold_miss'
                or item.get('source_user') != row['user_id']
                or item.get('source_adapter') != row['adapter_name']):
            raise ValueError('M9-A.1 replays cold -> same-user exact preconditions only')
        used.update(range(start, end))
        plan.append(dict(start=start, end=end, source_start=src_start, source_end=src_end,
                         token_ids=tuple(item['token_ids']), cache_key=item['cache_key']))
    if len(used) != row['reused_tokens']:
        raise ValueError('Physical span count differs from reused_tokens')
    return plan


def run_prefill(model, adapter, inputs, hits=None, *, capture=False):
    """Same host/CUDA completion clocks as M8; no control plane or tokenization."""
    import torch
    from semcache.metrics.timing import CPUWallTimer, CUDATimer
    from .base_projection import base_projection_path, mixed_qkv_elapsed_ms
    cuda = inputs['input_ids'].is_cuda
    timer = CUDATimer(inputs['input_ids'].device) if cuda else None
    audit = None
    wall = CPUWallTimer()
    with torch.inference_mode(), wall:
        with timer if timer else nullcontext():
            if hits is not None or capture:
                with base_projection_path(adapter, hits or [], inputs['input_ids'].shape[1],
                                          measure_cuda=cuda and not capture) as audit:
                    output = model(**inputs, output_attentions=True)
            else:
                output = model(**inputs, output_attentions=True)
        if timer:
            timer.resolve(synchronize=True)
    # Intentionally no logits comparison or host logit materialization.
    del output
    return dict(prefill_wall_ms=wall.elapsed_ms, prefill_gpu_ms=timer.elapsed_ms if timer else None,
        qkv_execution_ms=mixed_qkv_elapsed_ms(audit, enclosing_region_synchronized=True) if audit and cuda else None,
        audit=audit)


def materialize_hits(adapter, capture, plan, storage_device):
    from semcache.cache.cache_entry import CacheEntry
    from semcache.semantic.hit_selection import CacheHit
    from semcache.semantic.subsequence import Subsequence
    resident, hits = {}, []
    for item in plan:
        key = (tuple(item['token_ids']), item['source_start'], item['source_end'])
        if key not in resident:
            blocks = {layer: tuple(record[name][:, item['source_start']:item['source_end']] for name in 'qkv')
                      for layer, record in capture.projections.items()}
            entry = CacheEntry.from_tensors(item['cache_key'][0], item['token_ids'],
                (item['source_start'], item['source_end']), blocks, storage_device)
            entry.qkv_metadata['component_scope'] = 'base_qkv_latency_only'
            resident[key] = entry
        hits.append(CacheHit(Subsequence(item['token_ids'], item['start'], item['end']), resident[key]))
    # Copies are deliberately outside measured hit-prefill, as cold priming is in M8.
    for hit in hits:
        for layer in range(len(adapter.layers)):
            module = adapter.projection_module(layer, 'q')
            for tensor in hit.entry.tensors[layer]:
                if tensor.shape != (1, hit.window.end-hit.window.start, module.out_features):
                    raise ValueError('Cache tensor shape mismatch')
                if tensor.dtype != module.weight.dtype or str(tensor.device) != str(storage_device):
                    raise ValueError('Cache dtype or storage differs from requested M8 path')
    return hits


def profile_pairs(model, tokenizer, adapter, metadata, pairs, source_env, fresh_id, source_hashes):
    """Profile the cold -> exact-repeat precondition independently per repetition."""
    import torch
    from semcache.metrics.m8 import cache_transfer_path, token_ids_sha256
    from semcache.metrics.scaling_workload import controlled_length_trace
    validation = assert_no_lora(model, adapter)
    trace = controlled_length_trace(tokenizer, 32)
    text, token_ids = trace[1]['text'], trace[1]['token_ids']
    row0 = pairs[0][0]
    device = str(next(model.parameters()).device)
    for _, _, physical in pairs:
        validate_fresh_m85(physical)
        if physical['model_revision'] != metadata['resolved_model_revision'] or physical['tokenizer_revision'] != metadata['resolved_tokenizer_revision']:
            raise ValueError('Base model revision differs from freshly measured M8 model/tokenizer')
        if physical['dtype'] != str(next(model.parameters()).dtype):
            raise ValueError('Base dtype differs from M8')
        if (physical.get('hostname') != platform.node()
                or physical.get('gpu_name') != (torch.cuda.get_device_name() if device.startswith('cuda') else None)):
            raise ValueError('Base and M8 hardware identities differ')
        if torch.device(source_env['model']['device']) != torch.device(metadata['device']):
            raise ValueError('Base and M8 devices differ')
        if cache_transfer_path(physical['physical_cache_storage_device'], device) != physical['cache_transfer_path']:
            raise ValueError('Cache transfer path differs from M8')
        span_plan(physical, token_ids)
    inputs = dict(input_ids=torch.tensor([token_ids], device=device), use_cache=False)
    warmups, count = row0['warmup_runs'], row0['measured_runs']
    if len(pairs) != count or sorted(p[0]['repeat_index'] for p in pairs) != list(range(count)):
        raise ValueError('Expected one complete repetition series for same_user_exact')
    results = {}
    for mode in ('ES_BASE_NATIVE', 'ES_BASE_SEMCACHE_REUSE'):
        for phase, iterations in [('warmup', warmups), ('measured', count)]:
            for index in range(iterations):
                native, _, physical = pairs[index if phase == 'measured' else index % count]
                row = native if mode == 'ES_BASE_NATIVE' else physical
                # Cold state and cache payload are rebuilt for every discarded warmup
                # and every measured repetition, never inherited from previous runs.
                cold = run_prefill(model, adapter, inputs, capture=mode != 'ES_BASE_NATIVE')
                hits = (materialize_hits(adapter, cold['audit'], span_plan(physical, token_ids),
                                         physical['physical_cache_storage_device'])
                        if mode != 'ES_BASE_NATIVE' else None)
                del cold
                # Match M8's untimed reference prefill before the measured target.
                reference = run_prefill(model, adapter, inputs)
                del reference
                if device.startswith('cuda'):
                    torch.cuda.synchronize()  # M8's pre-request memory boundary; outside timing
                measured = run_prefill(model, adapter, inputs, hits)
                audit = measured.pop('audit')
                reused = physical['reused_tokens'] if hits is not None else 0
                if audit:
                    if (len(audit.records) != 3*len(adapter.layers)
                            or any(record['reused_projection_rows'] != reused for record in audit.records.values())):
                        raise ValueError('Physical base projection skipping differs from replay plan')
                if phase == 'measured':
                    results[(row['repeat_index'], mode)] = dict(**measured, **validation,
                        mode=mode, provenance='MEASURED', measurement_label=MEASUREMENT_LABEL,
                        fresh_m85_profile_id=fresh_id, source_pair_key=list(pair_key(row)),
                        source_m8_sha256=source_hashes['m8'], source_environment_sha256=source_hashes['environment'],
                        model_revision=metadata['resolved_model_revision'], tokenizer_revision=metadata['resolved_tokenizer_revision'],
                        dtype=metadata['dtype'], device=device, attention_implementation='eager',
                        prompt_token_ids=token_ids, prompt_token_ids_sha256=token_ids_sha256(token_ids),
                        warmup_runs=warmups, measured_runs=count,
                        warmup_state_semantics='fresh_discarded_trace',
                        repetition_state_semantics='reconstructed_cold_then_same_user_exact',
                        profile_trace_scope='cold + target same_user_exact; later cross-user/unrelated requests excluded',
                        executed_reused_tokens=reused, executed_fresh_tokens=len(token_ids)-reused,
                        cache_storage_device=physical['physical_cache_storage_device'],
                        cache_transfer_path=physical['cache_transfer_path'], cache_dtype=metadata['dtype'],
                        cache_payload='base-cold QKV; latency-only surrogate for M8 total-QKV values',
                        qkv_tensor_layout='[1, span_tokens, hidden_size], one tensor per Q/K/V/layer',
                        hidden_size=model.config.hidden_size, layers=len(adapter.layers),
                        personalized_output_quality_claimed=False,
                        control_plane_inside_es_compute=False,
                        compute_scope='full OPT base prefill, including local embedding/logits; no LoRA/control')
                del audit, measured, hits
    return results
