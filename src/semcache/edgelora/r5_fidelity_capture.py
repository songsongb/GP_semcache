"""Opt-in passive observations around the unchanged production OPT/TCP path.

No tensor returned by a hook/trace is substituted into inference. Python locals
are observed only when present in the installed eager implementation.
"""
import ast
from contextlib import contextmanager
import inspect
from pathlib import Path
import sys
import textwrap

from .communication import Channel, digest_bytes, require
from .r3a_artifacts import pack
from .r3b_fixture import ArrayTensor
from .r4a_generation import cache_pairs
from .r4b2_kv import attention_kv
from .r4c_generation import QualityDriver
from .r5_controller import RawOwner
from .r5_fidelity import SELECTED, historical_id
from .r5_perf_timing import Timing
from .r5_perf_evidence import PerformanceEvidence


def selected(rid):
    return historical_id(rid.split('.s')[0]) in SELECTED


def descriptor(item, *, layout='BTH', token_axis=1, key_axis=None):
    return dict(item, layout=layout, token_axis=token_axis, key_axis=key_axis)


def context_tensor(tensor):
    """Masks/positions may be integer or contain -Inf; preserve them as JSON."""
    import math
    data = tensor.detach().cpu().contiguous()
    flat = data.reshape(-1).tolist()
    return dict(shape=list(data.shape), dtype=str(data.dtype),
                sha256=digest_bytes(data.view(__import__('torch').uint8).numpy().tobytes()),
                values=[v if not isinstance(v, float) or math.isfinite(v) else str(v) for v in flat])


def original_method(method):
    """Use the same original Python function for source analysis and tracing."""
    try:
        function=inspect.unwrap(getattr(method,'__func__',method))
    except (ValueError,TypeError) as exc:
        raise ValueError(f'FORENSIC_OPT_FRAME: cannot unwrap {method!r}: {exc}') from exc
    require(inspect.isfunction(function),f'FORENSIC_OPT_FRAME: expected Python method, got {function!r}')
    code=function.__code__
    require(code.co_argcount>0 and code.co_varnames[0]=='self',
            f'FORENSIC_OPT_FRAME: unwrapped {function.__qualname__} at {code.co_filename}:{code.co_firstlineno} '
            'has no positional self; decorator may not expose __wrapped__')
    return function


def local_assignments(method, attention):
    """Identify real assignment sites, without depending on an HF line number."""
    method=original_method(method)
    lines, start = inspect.getsourcelines(method)
    source = textwrap.dedent(''.join(lines)); tree = ast.parse(source); result = {}
    # Nested definitions have their own code objects and are not this frame.
    def assignments(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda)):continue
            if isinstance(child,ast.Assign):yield child
            yield from assignments(child)
    require(len(tree.body)==1 and isinstance(tree.body[0],(ast.FunctionDef,ast.AsyncFunctionDef)),
            f'FORENSIC_OPT_FRAME: cannot identify source function {method.__qualname__}')
    for node in assignments(tree.body[0]):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id; rhs = ast.get_source_segment(source, node.value) or ''; label = None
        if attention and name in ('attn_weights', 'attn_probs'):
            if 'bmm' in rhs: label = 'attention_scores'
            elif 'softmax' in rhs: label = 'attention_probabilities'
            elif 'mask' in rhs and '+' in rhs: label = 'attention_masked_scores'
        elif not attention and name == 'hidden_states' and 'residual' in rhs and '+' in rhs:
            label = 'residual'
        if label:
            result[start+node.lineno-1] = (start+node.end_lineno-1, name, label, rhs)
    return result


class PassiveCapture:
    def __init__(self, driver, fid, n):
        self.driver = driver; self.fid = fid; self.n = n; self.tensors = {}; self.context = {}; self.sites = {}

    def save(self, name, tensor, *, layout='BTH', token_axis=1, key_axis=None):
        if name in self.tensors: return
        if tensor.ndim == 2 and layout == 'BTH': tensor = tensor.reshape(1, self.n, -1)
        # Own a snapshot before subsequent in-place scaling, residuals or cache append.
        snapshot=tensor.detach().clone()
        if name.endswith('.attention_masked_scores') and not hasattr(snapshot,'fixture_bytes'):
            # An eager FP16 additive mask may briefly overflow to -Inf before
            # the original clamp. Preserve those actual bytes without forcing
            # a different attention computation or rejecting valid masked data.
            import torch
            require(snapshot.dtype in (torch.float16,torch.float32) and
                    bool((torch.isfinite(snapshot)|torch.isneginf(snapshot)).all()),'unexpected nonfinite attention score')
            data=snapshot.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            item=dict(file=self.driver.evidence.blob(data,'tensors/'+self.fid+'_F_'+name+'.bin'),shape=list(snapshot.shape),
                      dtype=str(snapshot.dtype).removeprefix('torch.'),device=str(snapshot.device),sha256=digest_bytes(data),allow_negative_infinity=True)
        else:
            item = self.driver.evidence.save(snapshot, self.fid+'_F_'+name, force=True)
        self.tensors[name] = descriptor(item, layout=layout, token_axis=token_axis, key_axis=key_axis)

    @contextmanager
    def hooks(self, adapter):
        handles = []; methods = {}; instances = {}; frames = {}; completed = set(); original_trace = sys.gettrace()
        require(original_trace is None, 'forensic capture requires no other Python tracer')
        def pre(name):
            def observe(module, args, kwargs):
                x = kwargs.get('hidden_states', args[0] if args else None)
                self.save(name, x)
            return observe
        def post(name):
            def observe(module, args, kwargs, out):
                self.save(name, out[0] if isinstance(out, tuple) else out)
            return observe
        def diagnostic(frame):
            code=frame.f_code
            return f'{code.co_qualname if hasattr(code,"co_qualname") else code.co_name} at {code.co_filename}:{frame.f_lineno}'
        def flush(frame,state):
            site = state.pop('pending',None)
            if site is None: return
            end, name, label, rhs = site;layer=state['layer']
            if label == 'residual':
                index=state['residual_count'];state['residual_count']=index+1
                require(index < 2, f'FORENSIC_OPT_FRAME: extra residual at L{layer}, {diagnostic(frame)}')
                label = ('attention_residual', 'ffn_residual')[index]
            value = frame.f_locals.get(name)
            require(hasattr(value,'detach'),f'FORENSIC_OPT_FRAME: missing tensor {name} for L{layer}.{label}, {diagnostic(frame)}')
            key = f'L{layer}.{label}'
            scores = label in ('attention_scores','attention_masked_scores','attention_probabilities')
            self.save(key, value, layout=('BHQK' if value.ndim == 4 else 'HQK') if scores else 'BTH',
                      token_axis=value.ndim-2 if scores else 1, key_axis=value.ndim-1 if scores else None)
            self.sites[key] = dict(line=end, expression=rhs, file=frame.f_code.co_filename)
        def trace(frame, event, arg):
            code = frame.f_code
            if code in methods:
                attention,assignments=methods[code];ident=id(frame)
                if event=='call':
                    module=frame.f_locals.get('self');binding=instances.get(id(module))
                    require(binding is not None and binding[0] is module and binding[2] is code and binding[3]==attention,
                            f'FORENSIC_OPT_FRAME: cannot identify original-method self in {diagnostic(frame)}; '
                            f'local names={sorted(frame.f_locals)}; expected registered OPT layer/attention instance')
                    layer=binding[1];key=(layer,attention)
                    require(key not in completed and all(s['key']!=key for s in frames.values()),
                            f'FORENSIC_OPT_FRAME: repeated original-method call for L{layer}, {diagnostic(frame)}')
                    frames[ident]=dict(layer=layer,key=key,residual_count=0)
                    return trace
                state=frames.get(ident)
                require(state is not None,f'FORENSIC_OPT_FRAME: {event} without verified original-method call, {diagnostic(frame)}')
                layer=state['layer']
                if event == 'line':
                    site = state.get('pending')
                    if site and (frame.f_lineno > site[0] or frame.f_lineno < site[0] - 12): flush(frame,state)
                    if frame.f_lineno in assignments:
                        state['pending']=assignments[frame.f_lineno]
                elif event == 'return':
                    try:
                        # An exception unwinding through the frame produces a
                        # return event with None; preserve the original error.
                        if arg is not None:
                            flush(frame,state)
                            if attention:
                                heads=self.driver.heads;d=self.driver.identity['hidden_size']//heads
                                for role,name in (('k','key_states'),('v','value_states')):
                                    value=frame.f_locals.get(name)
                                    require(hasattr(value,'detach'),f'FORENSIC_OPT_FRAME: missing required L{layer}.{name}, {diagnostic(frame)}')
                                    require((value.ndim==3 and value.shape[0]==heads and value.shape[-1]==d) or
                                            (value.ndim==4 and list(value.shape[:2])==[1,heads] and value.shape[-1]==d),
                                            f'FORENSIC_OPT_FRAME: uninspected Attention K/V layout at L{layer}, {diagnostic(frame)}')
                                    self.save(f'L{layer}.used_attention_kv_{role}',value.reshape(1,heads,-1,d),layout='BHTD',token_axis=2)
                            completed.add(state['key'])
                    finally:frames.pop(ident,None)
                return trace
            # Observe actual native Linear and transported delta locals from the
            # frozen physical_path closure. No independent ES projection runs.
            if code.co_name == 'miss' and Path(code.co_filename).name == 'r3b_runtime.py':
                if event == 'return' and arg is not None:
                    v = frame.f_locals; layer, role = v['layer'], v['role']
                    for kind in ('base','delta'):
                        self.save(f'L{layer}.{kind}_{role}', v[kind])
                return trace
            return None
        try:
            for i, layer in enumerate(adapter.layers):
                prefix = f'L{i}.'
                handles.append(layer.register_forward_pre_hook(pre(prefix+'layer_input'), with_kwargs=True))
                handles.append(layer.register_forward_hook(post(prefix+'layer_output'), with_kwargs=True))
                attn = layer.self_attn
                def attn_pre(module, args, kwargs, i=i):
                    bound = inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments
                    self.save(f'L{i}.projection_input', bound['hidden_states'])
                    self.context[str(i)] = {k: context_tensor(v) for k, v in bound.items()
                                           if k in ('attention_mask','position_ids','cache_position') and hasattr(v, 'detach')}
                handles.append(attn.register_forward_pre_hook(attn_pre, with_kwargs=True))
                handles.append(attn.register_forward_hook(post(prefix+'attention_output'), with_kwargs=True))
                handles.append(attn.out_proj.register_forward_pre_hook(pre(prefix+'attention_context'), with_kwargs=True))
                for role, module in adapter.projection_modules(i).items():
                    handles.append(module.register_forward_hook(post(prefix+'raw_'+role), with_kwargs=True))
                for name, module in (('norm1',layer.self_attn_layer_norm), ('norm2',layer.final_layer_norm),
                                     ('fc1',layer.fc1), ('ffn',layer.fc2)):
                    handles.append(module.register_forward_hook(post(prefix+name), with_kwargs=True))
                handles.append(layer.fc2.register_forward_pre_hook(pre(prefix+'activation'), with_kwargs=True))
                # Code-to-sites metadata is fixed before tracing. Layer identity
                # belongs to a verified original-method call frame, never code.
                for method, attention in ((attn.forward,True),(layer.forward,False)):
                    function=original_method(method);code=function.__code__;sites=local_assignments(function,attention)
                    require(code not in methods or methods[code]==(attention,sites),
                            f'FORENSIC_OPT_FRAME: ambiguous layer/attention code {function.__qualname__}')
                    methods[code]=(attention,sites)
                    module=attn if attention else layer
                    require(id(module) not in instances,'FORENSIC_OPT_FRAME: repeated layer/attention instance')
                    instances[id(module)]=(module,i,code,attention)
            def head(module,args,kwargs,out): self.save('final_hidden',args[0])
            handles.append(self.driver.model.lm_head.register_forward_hook(head,with_kwargs=True))
            sys.settrace(trace)
            yield
            expected={(binding[1],binding[3]) for binding in instances.values()}
            missing=sorted(expected-completed)
            require(not frames and not missing,
                    f'FORENSIC_OPT_FRAME: missing original OPT method calls/returns {missing}; '
                    'expected unwrapped layer and attention frames; verify decorator __wrapped__ chain')
            for module,i,code,attention in instances.values():
                sites=methods[code][1]
                required=({'used_attention_kv_k','used_attention_kv_v'} if attention else {'attention_residual','ffn_residual'})
                required|={site[2] for site in sites.values() if site[2]!='residual'}
                missing=sorted(op for op in required if f'L{i}.{op}' not in self.tensors)
                require(not missing,f'FORENSIC_OPT_FRAME: missing required tensor observations at L{i}: {missing}; '
                        f'original code {code.co_filename}:{code.co_firstlineno}')
        finally:
            sys.settrace(original_trace)
            try:
                for handle in reversed(handles): handle.remove()
            finally:frames.clear()

    @contextmanager
    def tiny_hooks(self):
        """Passive scalar-fixture locals, explicitly not an OPT observation."""
        method = self.driver.tiny.forward; code = method.__func__.__code__
        lines, start = inspect.getsourcelines(method)
        enter = next(start+i for i,line in enumerate(lines) if 'out={}' in line)
        leave = next(start+i for i,line in enumerate(lines) if 'x=attended;' in line)
        h = self.driver.identity['hidden_size']; original = sys.gettrace()
        require(original is None, 'fixture tracer conflict')
        def trace(frame,event,arg):
            if frame.f_code is not code: return None
            if event == 'line' and frame.f_lineno in (enter,leave):
                v=frame.f_locals; name='layer_input' if frame.f_lineno==enter else 'layer_output'
                self.save(f'L{v["layer"]}.{name}',ArrayTensor(v['x'] if name=='layer_input' else v['attended'],[1,self.n,h]))
            return trace
        try:
            sys.settrace(trace); yield
        finally: sys.settrace(original)


class FidelityDriver(QualityDriver):
    def configure(self,*a,**k):
        contract = super().configure(*a,**k); self.perf = Timing(); return contract

    def c_forward(self,parent,step,inputs,target,past,policy,mode,cohort,grant=None,**kwargs):
        from contextlib import nullcontext
        fid=f'{parent}.s{step:02d}'; capture=PassiveCapture(self,fid,len(inputs))
        active=selected(parent)
        if active and past is not None:
            for layer,pair in enumerate(cache_pairs(past)):
                for role,t in zip('kv',pair): capture.save(f'L{layer}.input_past_{role}',t,layout='BHTD',token_axis=2)
        fixture_context=capture.tiny_hooks() if active and self.fixture else nullcontext()
        if active and not self.fixture: kwargs['observer']=capture.hooks
        with fixture_context:
            out,record,blocks,values,blob=super().c_forward(parent,step,inputs,target,past,policy,mode,cohort,grant,**kwargs)
        if active:
            for layer,block in blocks.items():
                for role,t in zip('qkv',block):
                    # On real OPT these came from hooks. Reuse their bytes to
                    # cross-check against the production owned Raw-QKV capture.
                    if f'L{layer}.raw_{role}' in capture.tensors:
                        require(capture.tensors[f'L{layer}.raw_{role}']['sha256']==record['qkv'][f'{layer}.{role}']['sha256'],'passive Raw-QKV capture differs')
                    else: capture.save(f'L{layer}.raw_{role}',t)
            for layer,pair in enumerate(cache_pairs(out.past_key_values)):
                for role,t in zip('kv',pair):
                    capture.save(f'L{layer}.attention_kv_{role}',t,layout='BHTD',token_axis=2)
                    used=capture.tensors.get(f'L{layer}.used_attention_kv_{role}')
                    if used is not None:
                        require(used['sha256']==capture.tensors[f'L{layer}.attention_kv_{role}']['sha256'],
                                'actual used Attention K/V differs from returned cache')
            logits=ArrayTensor(values,[1,1,len(values)]) if self.fixture else out.logits[:,-1:,:]
            capture.save('logits',logits)
            self.evidence.writer.append(self.output/'fidelity_steps.jsonl',dict(request_id=parent,step=step,forward_id=fid,
                context=record['context'],position_observation=record['position_observation'],tensors=capture.tensors,
                actual_layer_context=capture.context,local_sites=capture.sites,TEST_FIXTURE_ONLY=self.fixture))
        # Existing runtime expects timing fields, but this suite makes no timing inference.
        record['perf_forward']=dict(exclusive_ms={},scope='forensic hooks; no performance inference')
        return out,record,blocks,values,blob


class FidelityOwner(RawOwner):
    writer=None
    def selected(self,key,target,request_id):
        entry,grant,past=super().selected(key,target,request_id)
        if selected(request_id):
            evidence=PerformanceEvidence(self.writer.root/'es',self.writer,profile='STRICT_AUDIT')
            pairs=attention_kv(entry.tensors,self.proofs[key]['heads'],self._pm['identity']['hidden_size'])
            tensors={}
            for layer in range(len(pairs)):
                for j,role in enumerate('kv'):
                    for kind,t in (('cached_raw',entry.tensors[layer][j+1]),('derived',pairs[layer][j]),('reconstructed',cache_pairs(past)[layer][j])):
                        name=f'L{layer}.{kind}_{role}'
                        tensors[name]=descriptor(evidence.save(t,request_id+'_RECON_'+name,force=True),
                            layout='BTH' if kind=='cached_raw' else 'BHTD',token_axis=1 if kind=='cached_raw' else 2)
            self.writer.append(self.writer.root/'es/fidelity_reconstruction.jsonl',dict(request_id=request_id,grant=grant,
                tensors=tensors,source_proof=self.proofs[key],proof_sha256=self.proof_seals[key]))
        return entry,grant,past


def observed_channel(writer, output, *, ud=False):
    """Save actual payload bytes on both endpoints, without changing framing."""
    class ObservedChannel(Channel):
        current=None
        def capture(self,h,b,operation):
            if h['kind'] not in ('HIDDEN','DELTAS') or not selected(h['request_id']): return
            name=f'wire/{"ud" if ud else "es"}_{operation}_{h["request_id"]}_L{h["layer_id"]}_{h["kind"]}.bin'
            writer.bytes(output/name,b,exclusive=True)
            item=descriptor(dict(file=name,shape=h['shape'],dtype=h['dtype'],device='cpu',sha256=digest_bytes(b)),
                layout='RBTH' if h['kind']=='DELTAS' else 'BTH',token_axis=2 if h['kind']=='DELTAS' else 1)
            writer.append(output/('fidelity_ud_wire.jsonl' if ud else 'fidelity_wire.jsonl'),
                dict(forward_id=h['request_id'],layer=h['layer_id'],kind=h['kind'],operation=operation,tensor=item))
        def send(self,h,payload=b''):
            result=super().send(h,payload); self.capture(result['header'],payload,'send'); return result
        def recv(self,**expected):
            h,b,r=super().recv(**expected);self.current=h;self.capture(h,b,'recv');return h,b,r
    return ObservedChannel


def serve_ud(plan, output, writer):
    from . import r4c_dynamic_ud as ud
    from .r4c_session import ForwardOrder
    old_channel,old_math,old_order=ud.Channel,ud.cpu_deltas,ud.ForwardOrder
    instances=[];channel_type=observed_channel(writer,output,ud=True)
    class TrackedChannel(channel_type):
        def __init__(self,*a,**k): super().__init__(*a,**k); instances.append(self)
    class ObservedOrder(ForwardOrder):
        def __init__(self,*a,**k):
            super().__init__(*a,**k)
            writer.write(output/'ud_order_initial.json',dict(grants=list(self.grants),parents=list(self.parents),finished=list(self.finished),layer=self.layer,current=self.current))
    evidence=PerformanceEvidence(output,writer,profile='STRICT_AUDIT')
    def arithmetic(values,matrices,layer,hidden,**kwargs):
        result=old_math(values,matrices,layer,hidden,**kwargs)
        h=instances[0].current;rid=h['request_id']
        if selected(rid):
            n=h['shape'][1]
            x=ArrayTensor(values,[1,n,hidden]) if plan['fixture'] else values
            y=ArrayTensor(result,[3,1,n,hidden]) if plan['fixture'] else result
            writer.append(output/'fidelity_ud_math.jsonl',dict(forward_id=rid,layer=layer,
                input=descriptor(evidence.save(x,rid+f'_UD_L{layer}_input',force=True)),
                output=descriptor(evidence.save(y,rid+f'_UD_L{layer}_output',force=True),layout='RBTH',token_axis=2)))
        return result
    try:
        ud.Channel=TrackedChannel;ud.cpu_deltas=arithmetic;ud.ForwardOrder=ObservedOrder
        ud.serve(plan,output)
    finally: ud.Channel=old_channel;ud.cpu_deltas=old_math;ud.ForwardOrder=old_order


def execute(plan, root, writer):
    """Reuse I3P orchestration, including automatic routing and final generation."""
    from . import r5_perf_runtime as runtime
    old_driver,old_owner,old_channel,old_spawn=runtime.ProfiledDriver,runtime.RawOwner,runtime.Channel,runtime.spawn
    def spawn(command,*a,**k):
        command=list(command)
        if '--worker' in command and command[command.index('--worker')+1]=='ud':
            command[2]=str(Path(__file__).resolve().parents[3]/'scripts/114_run_r5_fidelity_forensics.py')
        return old_spawn(command,*a,**k)
    try:
        runtime.ProfiledDriver=FidelityDriver;runtime.RawOwner=FidelityOwner;FidelityOwner.writer=writer
        runtime.Channel=observed_channel(writer,Path(root)/'es');runtime.spawn=spawn
        return runtime.execute(plan,root,writer)
    finally:
        runtime.ProfiledDriver=old_driver;runtime.RawOwner=old_owner;runtime.Channel=old_channel;runtime.spawn=old_spawn;FidelityOwner.writer=None
