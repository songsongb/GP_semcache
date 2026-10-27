"""Opt-in supported Torch profiler. CPU attribution and raw device timelines stay separate."""
from contextlib import contextmanager,ExitStack
from collections import defaultdict
import json
from unittest.mock import patch
from semcache.diagnostics.r2a_analysis import union_us

class ProfilerUnavailable(RuntimeError): pass


def require_profiler(device):
    try:
        import torch
        p=torch.profiler
        required=('profile','record_function','ProfilerActivity')
        if any(not hasattr(p,n) for n in required): raise ProfilerUnavailable('torch.profiler API missing')
        activities=[p.ProfilerActivity.CPU]
        supported=p.supported_activities()
        if device.startswith('cuda'):
            if p.ProfilerActivity.CUDA not in supported: raise ProfilerUnavailable('CUDA Kineto activity unavailable')
            activities.append(p.ProfilerActivity.CUDA)
        return p,activities
    except (ImportError,AttributeError) as exc: raise ProfilerUnavailable(str(exc)) from exc


@contextmanager
def ranges(runtime):
    """No parent QKV hooks (executor forbids them). Only children and scoped wrappers."""
    import torch
    import semcache.edgelora.batched_mixed_projection as bm
    from semcache.edgelora.batched_mapping_optimized import SourceBackingDescriptors,PreparedMappingCache
    from semcache.diagnostics.r1b3_fast_selection import FastOccurrenceIndex
    rf=torch.profiler.record_function
    with ExitStack() as stack:
        def wrap(owner,name,label):
            original=getattr(owner,name)
            def called(*args,**kwargs):
                with rf('r2a::'+label): return original(*args,**kwargs)
            stack.enter_context(patch.object(owner,name,called))
        for owner,name,label in [(type(runtime.encoder),'encode','semantic_encode'),
            (type(runtime.engine.clusterer),'observe_with_diagnostics','cluster_assignment'),
            (type(runtime.engine.extractor),'extract','window_extraction'),
            (type(runtime.engine.cache),'lookup','cache_lookup'),
            (FastOccurrenceIndex,'select','fast_selection'),(FastOccurrenceIndex,'guard','index_snapshot_validation'),
            (SourceBackingDescriptors,'validate','snapshot_validation'),(PreparedMappingCache,'get','mapping_prepare_hit_miss')]:
            wrap(owner,name,label)
        from semcache.utils.timing import TimingRegistry
        import semcache.cache.impact_execution as impact
        wrap(impact,'compute_window_values','attention_impact')
        original_cpu=TimingRegistry.cpu
        @contextmanager
        def cpu(registry,name,**kwargs):
            with rf('r2a::engine/'+name),original_cpu(registry,name,**kwargs) as value: yield value
        stack.enter_context(patch.object(TimingRegistry,'cpu',cpu))
        original_span=bm.span
        @contextmanager
        def span(book,name):
            with rf('r2a::executor/'+name),original_span(book,name): yield
        stack.enter_context(patch.object(bm,'span',span))
        # Each module callback stack belongs to this serial profiling request only.
        active=[]
        def hook(module,label):
            contexts=[]
            def before(m,args):
                ctx=rf('r2a::'+label); ctx.__enter__(); contexts.append(ctx); active.append(ctx)
            def after(m,args,out):
                ctx=contexts.pop(); active.remove(ctx); ctx.__exit__(None,None,None)
            h1=module.register_forward_pre_hook(before); h2=module.register_forward_hook(after)
            stack.callback(h1.remove); stack.callback(h2.remove)
        base=runtime.model.get_base_model(); decoder=base.model.decoder
        for name in ('embed_tokens','embed_positions'): hook(getattr(decoder,name),'embedding/'+name)
        for name,module in runtime.model.named_modules():
            if isinstance(module,torch.nn.LayerNorm): hook(module,'layernorm')
        for layer in runtime.adapter.layers:
            hook(layer.self_attn,'attention_inclusive')
            hook(layer.self_attn.out_proj,'attention_output_projection')
            hook(layer.fc1,'ffn_up'); hook(layer.fc2,'ffn_down')
        hook(base.lm_head,'lm_head')
        for i in range(len(runtime.adapter.layers)):
            for role,module in runtime.adapter.projection_modules(i).items():
                hook(module.get_base_layer(),role+'_base')
                hook(module.lora_A['user_a'],role+'_lora_A'); hook(module.lora_B['user_a'],role+'_lora_B')
        try: yield
        finally:
            while active: active.pop().__exit__(None,None,None)


def category(event):
    node=event
    while node is not None:
        if str(node.name).startswith('r2a::'): return node.name[5:]
        node=getattr(node,'cpu_parent',None)
    return 'unknown_or_unattributed'


def operator_rows(events):
    """Exclusive CPU operator costs; device self attribution is NOT raw kernel busy time."""
    grouped={}
    for e in events:
        if str(getattr(e,'device_type','')).lower().endswith('cuda'): continue
        shapes=str(getattr(e,'input_shapes',None)); key=(e.name,category(e),shapes)
        r=grouped.setdefault(key,dict(operator=e.name,component=key[1],shapes=shapes,calls=0,cpu_self_ms=0.,
            device_self_attributed_ms=None,self_cpu_memory_bytes=0,self_device_memory_bytes=None))
        r['calls']+=1; r['cpu_self_ms']+=getattr(e,'self_cpu_time_total',0.)/1000
        device=getattr(e,'self_device_time_total',None)
        if device is not None: r['device_self_attributed_ms']=(r['device_self_attributed_ms'] or 0)+device/1000
        r['self_cpu_memory_bytes']+=getattr(e,'self_cpu_memory_usage',0)
        memory=getattr(e,'self_device_memory_usage',None)
        if memory is not None: r['self_device_memory_bytes']=(r['self_device_memory_bytes'] or 0)+memory
    return list(grouped.values())


def trace_rows(trace):
    """Assign activities only through explicit External id/correlation + enclosing ranges.

    No inference from kernel names. Each raw activity gets exactly one component.
    CPU/CUDA lanes are not added together. Unlinked GPU activity remains unknown.
    """
    events=[e for e in trace.get('traceEvents',[]) if e.get('ph')=='X' and 'dur' in e and 'ts' in e]
    lanes=defaultdict(list)
    for e in events:
        if e.get('cat') in ('cpu_op','user_annotation','cuda_runtime','cuda_driver'):
            lanes[(e.get('pid'),e.get('tid'))].append(e)
    external={}; correlations={}
    for lane in lanes.values():
        stack=[]
        for e in sorted(lane,key=lambda x:(x['ts'],-x['dur'])):
            while stack and stack[-1]['ts']+stack[-1]['dur']<=e['ts']: stack.pop()
            parent=next((x['name'][5:] for x in reversed(stack) if x.get('name','').startswith('r2a::') and not x['name'].startswith('r2a::request/')),None)
            component=e['name'][5:] if e.get('name','').startswith('r2a::') else parent
            if component and component.startswith('request/'): component=None
            args=e.get('args',{}); ext=args.get('External id'); corr=args.get('correlation')
            if component:
                if ext is not None: external[ext]=component
                if corr is not None: correlations[corr]=component
            if e.get('cat')=='user_annotation': stack.append(e)
    rows=[]; intervals=[]; kernel_us=known_us=qkv_us=0.
    for e in events:
        cat=str(e.get('cat','')).lower()
        if not any(c in cat for c in ('kernel','gpu_memcpy','gpu_memset','cuda_runtime','cuda_driver')): continue
        gpu=any(c in cat for c in ('kernel','gpu_memcpy','gpu_memset'))
        if gpu: intervals.append((e['ts'],e['ts']+e['dur']))
        args=e.get('args',{}); component=external.get(args.get('External id')) or correlations.get(args.get('correlation')) or 'unknown_or_unattributed'
        if 'kernel' in cat:
            kernel_us+=e['dur']
            if component!='unknown_or_unattributed': known_us+=e['dur']
            if component in [r+s for r in 'qkv' for s in ('_base','_lora_A','_lora_B')]: qkv_us+=e['dur']
        rows.append(dict(operator_or_kernel=e.get('name'),activity=cat,component=component,
            duration_ms=e['dur']/1000,stream=args.get('stream'),activity_count=1,
            memcpy_bytes=args.get('bytes'),gpu_activity=gpu,
            attribution='explicit External id/correlation to actual enclosing range; unknown if absent'))
    grouped={}
    for r in rows:
        key=(r['operator_or_kernel'],r['activity'],r['component'],r['stream'])
        if key not in grouped: grouped[key]=dict(r,duration_min_ms=r['duration_ms'],duration_max_ms=r['duration_ms'])
        else:
            g=grouped[key]; g['activity_count']+=1; g['duration_ms']+=r['duration_ms']
            g['duration_min_ms']=min(g['duration_min_ms'],r['duration_ms']); g['duration_max_ms']=max(g['duration_max_ms'],r['duration_ms'])
            g['memcpy_bytes']=g['memcpy_bytes']+r['memcpy_bytes'] if g['memcpy_bytes'] is not None and r['memcpy_bytes'] is not None else None
    return list(grouped.values()),dict(gpu_activity_union_ms=union_us(intervals)/1000 if intervals else None,
        gpu_activity_sum_ms=sum(b-a for a,b in intervals)/1000 if intervals else None,
        kernel_duration_sum_ms=kernel_us/1000 if kernel_us else None,
        kernel_attributed_fraction=known_us/kernel_us if kernel_us else None,
        kernel_unattributed_fraction=1-known_us/kernel_us if kernel_us else None,
        qkv_kernel_duration_ms=qkv_us/1000 if kernel_us else None,
        qkv_kernel_fraction=qkv_us/kernel_us if kernel_us else None,
        raw_gpu_attribution='explicit trace links only; kernel duration fractions are not wall-time speedup bounds; LoRA scaling may remain in parent',
        sum_definition='sum is activity-duration accounting across streams; union is busy interval coverage, neither equals end-to-end')


def profile_request(runtime,mode,path,max_trace_mib=64,with_stack=False):
    p,activities=require_profiler(runtime.a.device)
    from semcache.metrics.r1_runtime import synchronize,memory_snapshot
    with p.profile(activities=activities,record_shapes=True,profile_memory=True,with_stack=with_stack) as prof:
        with ranges(runtime),p.record_function('r2a::request/'+mode):
            logits,record=runtime.invoke(mode)
        synchronize(runtime.a.device)
    prof.export_chrome_trace(str(path))
    size=path.stat().st_size
    if size>max_trace_mib*1024**2:
        # The requested trace budget is a hard artifact limit, not silent success.
        path.unlink()
        raise RuntimeError(f'TRACE_BUDGET_EXCEEDED: {size} bytes; reduce trace scope')
    trace=json.loads(path.read_text()); kernels,definition=trace_rows(trace)
    return logits,record,operator_rows(prof.events()),kernels,dict(definition,trace_bytes=size,settings=dict(record_shapes=True,profile_memory=True,with_stack=with_stack),memory=memory_snapshot(runtime.a.device))
