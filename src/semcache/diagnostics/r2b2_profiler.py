"""Separate component/profiler trials; never consumed by minimal statistics."""
from contextlib import contextmanager,ExitStack
from datetime import datetime,timezone
import json
import os
import subprocess
from unittest.mock import patch
from semcache.diagnostics.r2a_profiler import require_profiler,ProfilerUnavailable,operator_rows,trace_rows
from semcache.diagnostics.r2b15_cost import CostBook,invoke,prepare_only


def monitor(device):
    """Query all physical GPUs, match Torch UUID; never guess CUDA_VISIBLE_DEVICES mapping."""
    state=dict(timestamp=datetime.now(timezone.utc).isoformat(),status='GPU_MONITOR_UNVERIFIED',device=device)
    try:
        import torch
        if not device.startswith('cuda'): raise RuntimeError('CPU execution')
        prop=torch.cuda.get_device_properties(device)
        state.update(name=prop.name,torch_uuid=str(getattr(prop,'uuid',None)),
            allocated_bytes=torch.cuda.memory_allocated(device),reserved_bytes=torch.cuda.memory_reserved(device))
        uuid=state['torch_uuid']
        if not uuid or uuid=='None': raise RuntimeError('Torch UUID unavailable; no physical-index assumption')
        # Some Torch versions omit the GPU- prefix. Querying all avoids --id exit 6.
        canonical=lambda x:x.strip().lower().removeprefix('gpu-')
        command=['nvidia-smi','--query-gpu=uuid,index,name,driver_version,memory.used,clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv,noheader,nounits']
        response=subprocess.run(command,capture_output=True,text=True,check=True,timeout=5)
        matches=[x for x in response.stdout.splitlines() if canonical(x.split(',')[0])==canonical(uuid)]
        if len(matches)!=1: raise RuntimeError('no unique physical UUID match')
        state.update(status='GPU_MONITOR_VERIFIED',physical_uuid_matched=True,query=command,values=dict(zip(
            ('uuid','physical_index','name','driver','memory_mib','clock_mhz','temperature_c','power_w','utilization_percent'),
            [x.strip() for x in matches[0].split(',')])))
        unavailable=[]
        for name in ('memory_mib','clock_mhz','temperature_c','power_w','utilization_percent'):
            try: state['values'][name]=float(state['values'][name])
            except (ValueError,TypeError): state['values'][name]=None; unavailable.append(name)
        if unavailable:
            state.update(status='GPU_MONITOR_UNVERIFIED',unavailable_fields=unavailable,reason='partial GPU telemetry unavailable')
        try:
            p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
            state['other_processes']=[x for x in p.stdout.splitlines() if canonical(x.split(',')[0])==canonical(uuid) and x.split(',')[1].strip()!=str(os.getpid())]
        except Exception as exc: state.update(other_processes=None,process_monitor_reason=str(exc))
    except Exception as exc: state['reason']=str(exc)
    return state


class ProfileBook(CostBook):
    @contextmanager
    def span(self,name,gpu=False):
        import torch
        with torch.profiler.record_function('r2b2::'+name),super().span(name,gpu): yield


def cpu_costs(events):
    """Attribute exclusive CPU self-time to nearest actual B2 range, once per event."""
    groups={}
    for event in events:
        if str(getattr(event,'device_type','')).lower().endswith('cuda'): continue
        parent=event; component='unknown_or_unattributed'
        while parent is not None:
            if str(parent.name).startswith('r2b2::'):
                component=parent.name[len('r2b2::'):]; break
            parent=getattr(parent,'cpu_parent',None)
        for row in operator_rows([event]):
            row['component']=component
            key=(row['operator'],component,row['shapes'])
            if key not in groups: groups[key]=row
            else:
                for field in ('calls','cpu_self_ms','self_cpu_memory_bytes'): groups[key][field]+=row[field]
                for field in ('device_self_attributed_ms','self_device_memory_bytes'):
                    if row[field] is not None: groups[key][field]=(groups[key][field] or 0)+row[field]
    return list(groups.values())


@contextmanager
def module_ranges(runtime):
    import torch
    with ExitStack() as stack:
        for i,layer in enumerate(runtime.adapter.layers):
            for name,module in [('decoder_layer',layer),('attention_inclusive',layer.self_attn),
                ('attention_output',layer.self_attn.out_proj),('ffn_up',layer.fc1),('ffn_down',layer.fc2)]:
                original=module.forward
                def called(*args,_fn=original,_name=name,_i=i,**kwargs):
                    with torch.profiler.record_function(f'r2b2::{_name}/layer{_i}'): return _fn(*args,**kwargs)
                stack.enter_context(patch.object(module,'forward',called))
        yield


def collect(runtime,mode,path):
    import torch
    profiler,activities=require_profiler(runtime.device)
    from semcache.metrics.r1_runtime import synchronize
    if mode=='PREPARE_ONLY' and not runtime.p:
        return dict(status='NOT_APPLICABLE_NO_PREFIX'),[],[],[]
    reference=None
    if mode!='PREPARE_ONLY': reference,_=invoke(runtime,mode)
    book=ProfileBook(runtime.device)
    with profiler.profile(activities=activities,record_shapes=True,profile_memory=True,with_stack=False) as prof:
        with module_ranges(runtime),profiler.record_function('r2b2::invocation_including_outside_timer'):
            if mode=='PREPARE_ONLY': result=prepare_only(runtime,book); ticket=None
            else: result,ticket=invoke(runtime,mode,book=book)
        synchronize(runtime.device)
    if reference is not None:
        from semcache.diagnostics.r2b2_benchmark import exact_cache
        if not torch.equal(reference['logits'],result['logits']) or not exact_cache(reference['cache'],result['cache']):
            raise RuntimeError('PROFILE_PARITY_FAILURE: instrumented result differs from uninstrumented reference')
    prof.export_chrome_trace(str(path))
    trace=json.loads(path.read_text())
    # Reuse tested R2-A parser by adapting labels in memory, not the saved trace.
    adapted=dict(trace,traceEvents=[dict(e,name=e.get('name','').replace('r2b2::','r2a::',1)) for e in trace.get('traceEvents',[])])
    kernels,definition=trace_rows(adapted)
    cpu=cpu_costs(prof.events())
    detail=dict(status='PROFILE_COLLECTED',instrumentation='torch_profiler_and_component_events',
        mode_classification='IDEAL_DIAGNOSTIC_ONLY' if mode=='WARM_PREPARED_DIAGNOSTIC' else 'COMPONENT_ONLY' if mode=='PREPARE_ONLY' else 'NATIVE_OR_PRIMARY',
        instrumented_wall_ms=result['wall_ms'],reference_minimal_wall_ms=reference['wall_ms'] if reference else None,
        instrumentation_difference_ms=result['wall_ms']-reference['wall_ms'] if reference else None,
        prepared=ticket,trace_bytes=path.stat().st_size,**definition,
        limitations='one profiler trial; reference delta includes ordering noise; parent/child clocks not added; invocation contains outside-timer audits/CPU logit export and prepared prebuild',
        scope='component preparation and native_forward ranges exclude audits; LM head nested in native_forward; CPU operator self time separate from GPU activities')
    runtime.check()
    return detail,cpu,kernels,book.export()
