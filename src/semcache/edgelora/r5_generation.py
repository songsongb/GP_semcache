"""R5 loop: automatic decision, unchanged C1 model forward, optional admission."""
import time
from contextlib import nullcontext

from .communication import digest_bytes, require
from .r3a_artifacts import append
from .r4c_session import VERSION
from .r4c_workload import termination


def measured(driver,name):
    profiler=getattr(driver,'perf',None)
    return nullcontext() if profiler is None else profiler.span(name)


def generate(driver,request,controller,discovery,found):
    owner=controller.owner;rid=request['request_id'];target=request['token_ids'];p=request['generation_policy']
    entry,grant,past=(None,None,None) if found is None else found
    mode='PARTIAL_PREFIX_HIT' if found else 'COLD_MISS' if controller.reuse else 'NO_REUSE'
    driver.sample_id=request['sample_id'];start=time.monotonic_ns();frames_start=len(driver.channel.records)
    generated=[];steps=[];logits=bytearray();source_blocks=proof=None
    for step in range(p['max_new_tokens']):
        profiler=getattr(driver,'perf',None)
        if profiler is not None:profiler.step=step;event_start=len(profiler.events)
        tick=time.monotonic_ns();inputs=target[(grant['prefix_length'] if grant else 0):] if not step else [generated[-1]]
        out,record,blocks,values,blob=driver.c_forward(rid,step,inputs,target,past,p,mode,owner.cohort,grant)
        returned=time.monotonic_ns()
        with measured(driver,'greedy_token_selection'):
            token=max(range(len(values)),key=values.__getitem__)
        with measured(driver,'output_state_and_past_handling'):
            generated.append(token);logits.extend(blob);past=out.past_key_values
        if not step:
            if found is None and controller.reuse:source_blocks=blocks;proof=record['raw_kv_proof']
            prefill_end=time.monotonic_ns()
        del out,blocks;driver.synchronize();now=time.monotonic_ns()
        record.update(request_id=rid,step=step,selected_token_id=token,logits_sha256=digest_bytes(blob),
            wall_ms=(now-tick)/1e6,start_monotonic_ns=tick,available_monotonic_ns=now)
        if profiler is not None:
            record['perf_post_forward_ms']=(now-returned)/1e6
            record['perf_before_internal_and_after_ms']=record['wall_ms']-record['host_forward_and_evidence_ms']
        with measured(driver,'step_json_serialization_append'):
            append(driver.output/'r5_tokens.jsonl',record)
        steps.append(record)
        if profiler is not None:
            append(driver.output/'r5_perf_steps.jsonl',dict(request_id=rid,request_index=profiler.request_index,step=step,
                exclusive_forward_ms=record['perf_forward']['exclusive_ms'],nested_and_generation_ms=profiler.totals(event_start),
                wall_to_availability_ms=record['wall_ms'],post_forward_ms=record['perf_post_forward_ms'],
                step_through_json_ms=(time.monotonic_ns()-tick)/1e6,evidence=driver.evidence.diagnostics()))
        reason=termination(p,generated)
        if reason is not None:break
    require(generated and reason is not None,'incomplete generation')
    with measured(driver,'request_completion_control'):
        owner.control(dict(version=VERSION,cohort_sha256=owner.cohort,action='COMPLETE_REQUEST',parent_request_id=rid,
            generated_token_ids=generated,termination=reason))
    text=driver.decode_text([t for t in generated if t not in {p['bos_token_id'],p['eos_token_id'],p['pad_token_id']}]) if driver.fixture else driver.decode_text(generated)
    model_end=time.monotonic_ns();admit_start=model_end;owner.cache_type=driver.cache_type
    with measured(driver,'cache_admission_and_eviction'):
        admitted=controller.admit(request,discovery['assignment'],source_blocks,proof,completed=True)
    if entry is not None:owner.reused(owner.entries[grant['entry_id']].context,entry)
    driver.synchronize();end=time.monotonic_ns()
    frames=driver.channel.records[frames_start:]
    fields=('hidden_payload_bytes','delta_payload_bytes','application_wire_bytes','es_to_ud_wire_bytes','ud_to_es_wire_bytes')
    totals=lambda rows:{k:sum(s[k] for s in rows) for k in fields}
    result=dict(request_id=rid,status='COMPLETE',mode=mode,source=grant,admission=admitted,
        generated_token_ids=generated,generated_text=text,generated_tokens=len(generated),termination=reason,
        logits_blob=bytes(logits),logits_shape=[1,len(generated),driver.vocab],logits_sha256=digest_bytes(logits),
        model_start_ns=start,model_end_ns=model_end,generation_end_ns=end,model_execution_e2e_ms=(model_end-start)/1e6,
        model_ttft_ms=(steps[0]['available_monotonic_ns']-start)/1e6,prefill_ms=(prefill_end-start)/1e6,
        decode_ms=sum(s['wall_ms'] for s in steps[1:]),cache_admission_eviction_ms=(end-admit_start)/1e6,
        generated_tokens_per_second=len(generated)/((model_end-start)/1e9),
        prefill=totals(steps[:1]),decode=totals(steps[1:]),
        control_wire_bytes=sum(f['wire_bytes'] for f in frames if f['header']['kind'] not in ('HIDDEN','DELTAS')),
        total_request_wire_bytes=sum(f['wire_bytes'] for f in frames),
        reused_prefix_tokens=0 if grant is None else grant['prefix_length'],
        skipped_prefix_projection_token_rows=steps[0]['prefix_projection_token_rows_skipped'],
        executed_prefill_projection_token_rows=steps[0]['executed_projection_token_rows'],
        cached_prefix_q_consumed=False,projection_kernels_entirely_skipped=0,
        first_token_available_ns=steps[0]['available_monotonic_ns'])
    return result,steps
