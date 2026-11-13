"""Light generation kernel using unchanged physical/protocol/impact primitives.

Native OPT integration is NOT_GPU_VERIFIED. No monkeypatch of Strict modules.
"""
from contextlib import nullcontext
import math
import time

from .communication import digest, require
from .r4c_workload import termination
from .r5_subsequence_profile import Profiler, call


class LightProfiler(Profiler):
    """Bounded wall components, no CUDA events, no per-frame/detail persistence."""
    def __init__(self):
        super().__init__(cuda_events=False)
        self.detail_counts = {}

    def detail(self, record):
        key = record.get('boundary', 'UNKNOWN') + ':' + record.get('reason', 'OBSERVED')
        require(key in self.detail_counts or len(self.detail_counts) < 128, 'detail summary bound exceeded')
        self.detail_counts[key] = self.detail_counts.get(key, 0) + 1

    def finish(self):
        result = super().finish()
        result.update(detail_counts=dict(self.detail_counts), eligibility=[],
                      observer_effect='LIGHT_WALL_SCOPES; Python observer cost included, not separately calibrated',
                      profiler_overhead_ms=None, gpu_kernel_ms=None,
                      timing_scope='exclusive bins partition observed scopes ONLY; inclusive scopes overlap; unobserved time unassigned')
        return result


def generate_loop(req, forward, complete, decode_text, *, clock_ns=time.monotonic_ns):
    """Same greedy/finite/termination policy; bounded token summaries, no logits.

    forward returns host-visible logits. There is no extra unconditional CUDA
    synchronize. Host vocabulary transfer remains necessary in this version.
    """
    begin = clock_ns()
    tokens, past = [], None
    prefill_end = decode_ns = None
    for step in range(req['generation_policy']['max_new_tokens']):
        tick = clock_ns()
        values, past = forward(step, list(tokens), past)
        require(values and all(math.isfinite(x) for x in values), 'empty/nonfinite full-vocabulary logits')
        tokens.append(max(range(len(values)), key=values.__getitem__))
        now = clock_ns()
        if step == 0:
            prefill_end, decode_ns = now, 0
        else:
            decode_ns += now - tick
        reason = termination(req['generation_policy'], tokens)
        if reason is not None:
            break
    require(tokens and reason is not None, 'unfinished light generation')
    text = decode_text(tokens)
    model_end = clock_ns()
    completion = complete(tokens)
    end = clock_ns()
    return dict(request_id=req['request_id'], status='COMPLETE', generated_token_ids=tokens,
                generated_token_ids_sha256=digest(tokens), generated_text=text,
                generated_text_sha256=__import__('hashlib').sha256(text.encode()).hexdigest(), termination=reason,
                arrival_ns=begin, first_available_ns=prefill_end, model_end_ns=model_end, completion_ns=end,
                application_e2e_ms=(end - begin) / 1e6, application_ttft_ms=(prefill_end - begin) / 1e6,
                model_execution_e2e_ms=(model_end - begin) / 1e6,
                model_ttft_ms=(prefill_end - begin) / 1e6, prefill_ms=(prefill_end - begin) / 1e6,
                decode_ms=decode_ns / 1e6, admission_completion_ms=(end - model_end) / 1e6,
                completion=completion, independent_full_vocabulary_audit='NOT_EVALUATED',
                cuda_explicit_sync_ms=None, correctness_qualified_speedup='NOT_ESTABLISHED')


def generate_native(model, adapter, tokenizer, req, channel, owner, index, *,
                    condition, mode, allow_approximate=False, on_tcp_complete=None):
    """Opt-in kernel, caller must enforce real model/device/source/session guards.

    This function is not a GPU launcher. The CLI deliberately blocks native run
    until a resource-guarded worker and native CPU approval exist.
    """
    import torch
    import inspect
    from .r5_sparse_qkv import SparseSession
    from .r5_subsequence_generation import StreamingImpact, fresh_path, complete_request
    from .r5_subsequence_physical import projection_path, EXACT, APPROX
    from .r4a_diagnostics import observe_opt_positions
    from .r4a_generation import past_summary
    require(condition in ('FULL_RECOMPUTE', 'SEMCACHE_D5_EXACT', 'SEMCACHE_D5_APPROX'), 'unknown Light condition')
    require(mode == (APPROX if condition == 'SEMCACHE_D5_APPROX' else EXACT) and
            (condition != 'SEMCACHE_D5_APPROX' or allow_approximate), 'condition/mode/opt-in mismatch')
    require(not torch.is_autocast_enabled() and not model.training and
            all(type(l.self_attn).__name__ == 'OPTAttention' for l in adapter.layers), 'eager eval OPT required')
    require(owner is None or owner.evidence is None, 'Light must not use Strict evidence sink')
    profiler = LightProfiler()
    arrival = time.monotonic_ns()
    row_summary = dict(fresh_qkv_rows=0, skipped_qkv_rows=0)
    try:
        with profiler.activate(req['request_id']):
            assignment = None if condition == 'FULL_RECOMPUTE' else call('semantic_assignment_inclusive', index.assign, req['input_text'])
            matching = None if assignment is None else owner.begin(req, assignment, mode=mode, allow_approximate=allow_approximate)
            impact = None if assignment is None else StreamingImpact(req, len(adapter.layers), owner.active['windows'], evidence=None)
            device = next(model.parameters()).device

            def forward(step, generated, past):
                session = SparseSession(channel, req, req['request_id'] + f'_s{step:03d}', generated=generated)
                inputs = req['token_ids'] if not step else generated[-1:]
                mask = [1] * (len(req['token_ids']) + step)
                old = len(req['token_ids']) + step - 1 if step else 0
                kwargs = dict(input_ids=torch.tensor([inputs], device=device), attention_mask=torch.tensor([mask], device=device),
                              past_key_values=past, use_cache=True, output_attentions=bool(not step and impact is not None), return_dict=True)
                parameters = inspect.signature(model.forward).parameters
                if 'position_ids' in parameters:
                    kwargs['position_ids'] = torch.tensor([list(range(old, old + len(inputs)))], device=device)
                if 'cache_position' in parameters:
                    kwargs['cache_position'] = torch.arange(old, old + len(inputs), device=device)
                if past is not None:
                    past_summary(past, old, len(adapter.layers), model.config.num_attention_heads,
                                 req['physical_identity']['hidden_size'], req['physical_identity']['dtype'], str(device))
                records = []
                path = projection_path(adapter, owner, session) if not step and assignment is not None else fresh_path(adapter, session, records, evidence=None)
                with torch.inference_mode(), (impact.hooks(adapter) if not step and impact is not None else nullcontext()), path, observe_opt_positions(model.model.decoder.embed_positions, mask, old):
                    with profiler.location(-1, '', 'PREFILL' if not step else 'DECODE'):
                        out = call('native_model_forward_inclusive', model, **kwargs)
                values = call('logit_host_transfer', lambda: out.logits[0, -1].float().cpu().tolist())
                past_summary(out.past_key_values, old + len(inputs), len(adapter.layers), model.config.num_attention_heads,
                             req['physical_identity']['hidden_size'], req['physical_identity']['dtype'], str(device))
                for r in session.receipts:
                    row_summary['fresh_qkv_rows'] += 3 * r['fresh_rows_per_role']
                    row_summary['skipped_qkv_rows'] += 3 * r['skipped_rows_per_role']
                return values, out.past_key_values

            def complete(tokens):
                control, physical = complete_request(channel, req, tokens, owner=None if assignment is None else owner,
                                                       attentions=impact, on_tcp_complete=on_tcp_complete)
                return dict(control=control, physical=physical)

            result = generate_loop(req, forward, complete,
                                   lambda ids: tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        result.update(condition=condition, context_mode=mode, row_summary=row_summary,
                      matching_candidate_count=None if matching is None else len(matching['candidates']),
                      components=profiler.finish())
        result['arrival_ns'] = arrival
        result['completion_ns'] = time.monotonic_ns()
        result['application_ttft_ms'] = (result['first_available_ns'] - arrival) / 1e6
        result['application_e2e_ms'] = (result['completion_ns'] - arrival) / 1e6
        return result
    except BaseException:
        if owner is not None:
            owner.abort()
        raise


class SessionLifecycle:
    """Condition cache/session boundary contract; no process or GPU side effects."""
    def __init__(self, tasks):
        require(tasks and len({t['request_id'] for t in tasks}) == len(tasks), 'invalid session request stream')
        self.tasks = tasks
        self.next = 0
        self.pending = None
        self.stopped = set()
        self.byes = set()
        self.final = {t['adapter_id']: t['request_id'] for t in tasks}

    def begin(self, task):
        require(self.pending is None and self.next < len(self.tasks) and task == self.tasks[self.next] and
                task['adapter_id'] not in self.stopped, 'late, duplicate or reordered request')
        self.pending = task

    def ack(self, request_id):
        require(self.pending is not None and self.pending['request_id'] == request_id, 'invalid completion ACK')
        task = self.pending
        self.next += 1
        self.pending = None
        return self.final[task['adapter_id']] == request_id

    def stop_after_ack(self, adapter_id):
        require(adapter_id in self.final and adapter_id not in self.stopped and self.pending is None and
                any(t['request_id'] == self.final[adapter_id] for t in self.tasks[:self.next]), 'premature/duplicate STOP')
        self.stopped.add(adapter_id)

    def bye(self, adapter_id):
        require(adapter_id in self.stopped and adapter_id not in self.byes, 'unexpected/duplicate BYE')
        self.byes.add(adapter_id)

    def finish(self):
        require(self.pending is None and self.next == len(self.tasks) and self.stopped == self.byes == set(self.final),
                'missing final ACK/STOP/BYE completion')
