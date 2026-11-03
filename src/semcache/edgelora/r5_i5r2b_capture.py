"""Opt-in passive Layer-0 observations; original arithmetic executes exactly once.

TorchFunctionMode observes actual Linear/add/mul outputs, including ephemeral
pre-cast sums. Python tracing identifies production projection call frames.
Nothing observed is returned as a replacement tensor. Diagnostic timing is invalid
for serving comparisons. No GPU dependency is imported at module import time.
"""
from contextlib import contextmanager,nullcontext
import importlib.metadata
import inspect
import math
import os
from pathlib import Path
import sys

from .communication import digest_bytes,require
from .r3a_artifacts import pack
from .r3b_fixture import ArrayTensor
from .r3b_cache import raw
from .r4a_generation import cache_pairs
from .r5_i5_driver import QualityDriver
from .r5_i5r2_tensors import slice_axis,DTYPES,UNKNOWN
from .r5_i5r2b_plan import VERSION


OBSERVED_FIELDS={'observed_stride','original_contiguous','storage_offset'}
ANNOTATION_FIELDS={'actual_operation','allow_negative_infinity','full_payload_sha256',
    'full_payload_shape','full_payload_bytes','scale'}


def capture_metadata(shape,observed,extra):
    """Only layout observations and named annotations may extend tensor identity."""
    require(observed is None or isinstance(observed,dict),'invalid observed capture metadata')
    observed={} if observed is None else observed
    require(set(observed)<=OBSERVED_FIELDS,'protected/unknown observed capture metadata')
    require(set(extra)<=ANNOTATION_FIELDS,'protected/unknown extra capture metadata')
    if 'observed_stride' in observed:
        stride=observed['observed_stride']
        require(isinstance(stride,list) and len(stride)==len(shape) and all(type(v) is int and v>=0 for v in stride),
            'invalid observed tensor stride')
    if 'original_contiguous' in observed:
        require(type(observed['original_contiguous']) is bool,'invalid observed tensor contiguity')
    if 'storage_offset' in observed:
        require(type(observed['storage_offset']) is int and observed['storage_offset']>=0,'invalid observed tensor storage offset')
    metadata=dict(observed_stride=UNKNOWN,original_contiguous=UNKNOWN)
    metadata.update(observed)
    metadata.update(extra)
    return metadata


def backend(fixture=False):
    if fixture:return dict(TEST_FIXTURE_ONLY=True,engine='stdlib scalar fixture; not Torch/GPU arithmetic')
    import torch
    get=lambda obj,name:getattr(obj,name,UNKNOWN)
    return dict(TEST_FIXTURE_ONLY=False,torch=torch.__version__,transformers=importlib.metadata.version('transformers'),
        cuda_compiled_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version() if torch.cuda.is_initialized() else 'NOT_QUERIED_CPU_UD',
        cuda_initialized=torch.cuda.is_initialized(),TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
        FP16_reduced_precision_reduction=get(torch.backends.cuda.matmul,'allow_fp16_reduced_precision_reduction'),
        float32_matmul_precision=torch.get_float32_matmul_precision(),cudnn_TF32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cpu_threads=torch.get_num_threads(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else UNKNOWN,
        autocast_enabled=torch.is_autocast_enabled(),cuBLAS_library_version='NOT_EXPOSED_BY_THIS_PUBLIC_API',
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        accumulation_precision='Internal GEMM accumulation not inferred from result dtype')


class Capture:
    def __init__(self,output,writer,config,endpoint):
        self.output=Path(output);self.writer=writer;self.config=config;self.endpoint=endpoint
        self.entries={};self.contexts={};self.current=None;self.bytes=0;self.files=0
    def select(self,rid,origin,n):
        if rid not in (self.config['source_request_id'],self.config['target_request_id']):return False
        self.current=dict(request_id=rid,origin=origin,input_length=n)
        self.contexts.setdefault(rid,{}).update(self.current,capture_interval=[0,64]);return True
    def omit(self,name,reason):
        if self.current is None:return
        key=self.current['request_id']+'/'+name
        require(key not in self.entries,'duplicate diagnostic boundary: '+key)
        self.entries[key]=dict(status='NOT_CAPTURED',reason=reason)
    def blob(self,name,data,shape,dtype,device,axis,*,origin=None,observed=None,**extra):
        require(self.current is not None,'diagnostic context missing')
        metadata=capture_metadata(shape,observed,extra)
        origin=self.current['origin'] if origin is None else origin;n=shape[axis]
        start=max(0,-origin);stop=min(n,64-origin)
        if stop<=start:self.omit(name,'NO_PREFIX_QUERY_ROWS: actual HIT computes suffix only');return None
        require(dtype in DTYPES and len(data)==math.prod(shape)*DTYPES[dtype][1],'diagnostic byte/dtype/shape differs')
        selected,retained=slice_axis(data,shape,axis,start,stop,DTYPES[dtype][1]);key=self.current['request_id']+'/'+name
        require(key not in self.entries,'duplicate diagnostic boundary: '+key)
        require(self.files+1<=self.config['max_files'] and self.bytes+len(selected)<=self.config['capture_bytes'],'I5R2B_CAPTURE_BUDGET_EXHAUSTED')
        path='tensors/i5r2b_'+self.current['request_id']+'_'+name+'.bin'
        require('/' not in name and '..' not in name,'unsafe diagnostic boundary name')
        item=dict(status='CAPTURED',file=path,shape=retained,dtype=dtype,device=device,sha256=digest_bytes(selected),token_axis=axis,
            semantic_positions=[origin+start,origin+stop],observed_shape=list(shape))
        item.update(metadata)
        self.writer.bytes(self.output/path,selected,exclusive=True);self.files+=1;self.bytes+=len(selected)
        self.entries[key]=item;return item
    def save(self,name,tensor,axis=1,*,origin=None,**extra):
        shape=list(tensor.shape);dtype=str(tensor.dtype).removeprefix('torch.');device=str(tensor.device)
        # Reject reserved kwargs before forwarding positional shape/dtype/device
        # to blob(), rather than raising a duplicate-keyword TypeError there.
        capture_metadata(shape,None,extra)
        observed={}
        if not hasattr(tensor,'fixture_bytes'):
            observed=dict(observed_stride=list(tensor.stride()),original_contiguous=tensor.is_contiguous(),storage_offset=tensor.storage_offset())
        # Serialization copies/synchronizes but never changes a live operand.
        return self.blob(name,raw(tensor),shape,dtype,device,axis,origin=origin,observed=observed,**extra)
    def finish(self,*,ok):
        self.writer.write(self.output/'i5r2b_capture.json',dict(version=VERSION,DIAGNOSTIC_ONLY=True,endpoint=self.endpoint,
            TEST_FIXTURE_ONLY=self.config['TEST_FIXTURE_ONLY'],status='COMPLETE' if ok else 'FAILED',bytes=self.bytes,files=self.files,
            byte_limit=self.config['capture_bytes'],file_limit=self.config['max_files'],entries=self.entries,contexts=self.contexts,
            interference='Synchronous copies and observers alter scheduling; no speedup inference'))


@contextmanager
def operation_mode(observe,*,enabled=True):
    if not enabled:yield;return
    from torch.overrides import TorchFunctionMode
    class Observed(TorchFunctionMode):
        def __torch_function__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}))
            observe(getattr(func,'__name__',str(func)),args,kwargs or {},result)
            return result
    with Observed():yield


def installed_observer_check():
    """Actual small CPU Torch API check, not GPU numerical qualification."""
    import torch
    import torch.nn.functional as F
    seen=[];x=torch.ones(1,2,4,dtype=torch.float16);w=torch.ones(4,4,dtype=torch.float16);delta=torch.ones(1,2,4,dtype=torch.float32)
    with torch.inference_mode():
        baseline=(F.linear(x,w)+delta).to(x.dtype)
        with operation_mode(lambda name,args,kwargs,out:seen.append((name,str(getattr(out,'dtype',None))))):
            result=(F.linear(x,w)+delta).to(x.dtype)
    require(torch.equal(result,baseline) and sum(n=='linear' for n,d in seen)==1 and any(n in ('add','__add__') and d=='torch.float32' for n,d in seen),
            'installed TorchFunctionMode cannot observe native Linear and promoted sum unchanged')
    return dict(status='CPU_API_PASS_NOT_GPU_FIDELITY',observed=seen)


class ProjectionObserver:
    def __init__(self,store,fixture,n):self.store=store;self.fixture=fixture;self.n=n;self.active=None
    def array(self,values):return ArrayTensor(list(values),[1,self.n,len(values)//self.n])
    def observe(self,name,args,kwargs,out):
        active=self.active
        if active is None:return
        role=active['role']
        if name=='linear':
            require('base' not in active,'unexpected second GPU base Linear in projection')
            active['base']=out;self.store.save('base_'+role,out)
        elif name in ('add','__add__') and len(args)>=2 and args[0] is active.get('base') and args[1] is active.get('delta'):
            self.store.save('sum_before_cast_'+role,out,actual_operation=name)
    @contextmanager
    def trace(self):
        previous=sys.gettrace();require(previous is None,'diagnostic tracer conflicts with another tracer')
        def trace(frame,event,arg):
            code=frame.f_code;file=Path(code.co_filename).name
            projection=(code.co_name=='miss' and file=='r3b_runtime.py') if not self.fixture else (code.co_name=='project' and file=='r5_i5_driver.py')
            if projection and frame.f_locals.get('layer')==0:
                role=frame.f_locals['role']
                if event=='call':
                    require(self.active is None,'nested diagnostic projection')
                    self.active=dict(role=role,frame=frame)
                    x=frame.f_locals['x'];self.store.save('projection_input_'+role,self.array(x) if self.fixture else x)
                elif event=='return':
                    if arg is not None:self.store.save('total_'+role,self.array(arg) if self.fixture else arg)
                    self.active=None
                return trace
            if self.active is not None:
                if code.co_name=='correction' and file=='edge_server.py' and event=='return' and arg is not None:
                    self.active['delta']=arg;self.store.save('delta_on_ES_'+self.active['role'],self.array(arg) if self.fixture else arg)
                if self.fixture and code.co_name=='native' and file=='r4a_fixture.py' and event=='return' and arg is not None:
                    self.store.save('base_'+self.active['role'],self.array(arg))
                return trace
            return None
        try:
            sys.settrace(trace)
            with operation_mode(self.observe,enabled=not self.fixture):yield
        finally:sys.settrace(previous);self.active=None


@contextmanager
def model_hooks(driver,store):
    if driver.fixture:yield;return
    handles=[];layer=driver.adapter.layers[0]
    def pre(name):
        def hook(module,args,kwargs):store.save(name,kwargs.get('hidden_states',args[0] if args else None))
        return hook
    def post(name):
        def hook(module,args,kwargs,out):
            value=out[0] if isinstance(out,tuple) else out
            store.save(name,value,axis=0 if value.ndim==2 else 1)
        return hook
    try:
        handles.append(layer.register_forward_pre_hook(pre('layer_input'),with_kwargs=True))
        handles.append(layer.self_attn_layer_norm.register_forward_pre_hook(pre('norm_input'),with_kwargs=True))
        handles.append(layer.self_attn_layer_norm.register_forward_hook(post('norm_output'),with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(post('attention_output'),with_kwargs=True))
        def mask(module,args,kwargs):
            bound=inspect.signature(module.forward).bind_partial(*args,**kwargs).arguments
            value=bound.get('attention_mask')
            if value is None:store.omit('actual_attention_mask','No explicit additive mask observed')
            else:store.save('actual_attention_mask',value,axis=value.ndim-2,allow_negative_infinity=True)
        handles.append(layer.self_attn.register_forward_pre_hook(mask,with_kwargs=True))
        decoder=driver.model.model.decoder
        handles.append(decoder.embed_tokens.register_forward_hook(post('token_embedding'),with_kwargs=True))
        handles.append(decoder.embed_positions.register_forward_hook(post('position_embedding'),with_kwargs=True))
        yield
    finally:
        for handle in reversed(handles):handle.remove()


class DiagnosticDriver(QualityDriver):
    diagnostic_plan=None;diagnostic_writer=None;stores=None
    def configure(self,output,**kwargs):
        contract=super().configure(output,**kwargs)
        self.capture=Capture(output,self.diagnostic_writer,self.diagnostic_plan['i5r2b'],'ES')
        self.stores.append(self.capture)
        self.diagnostic_writer.write(Path(output)/'i5r2b_backend.json',backend(self.fixture));return contract
    def c_forward(self,parent,step,inputs,target,past,policy,mode,cohort,grant=None,**kwargs):
        store=self.capture;cfg=store.config
        if step or parent not in (cfg['source_request_id'],cfg['target_request_id']) or self.identity['adapter']!='Adapter_A':
            return super().c_forward(parent,step,inputs,target,past,policy,mode,cohort,grant,**kwargs)
        origin=0 if past is None else grant['prefix_length'];store.select(parent,origin,len(inputs))
        store.contexts[parent].update(token_ids=list(inputs),target_token_ids=list(target),positions=list(range(origin,origin+len(inputs))),
            attention_mask=[1]*(origin+len(inputs)),mode=mode,backend_before=backend(self.fixture))
        try:
            if past is not None:
                for role,t in zip('kv',cache_pairs(past)[0]):store.save('consumed_prefix_'+role,t,axis=2,origin=0)
            observer=ProjectionObserver(store,self.fixture,len(inputs))
            old_observer=kwargs.pop('observer',None);require(old_observer is None,'only one diagnostic observer supported')
            if not self.fixture:kwargs['observer']=lambda adapter:model_hooks(self,store)
            with observer.trace():out,record,blocks,values,blob=super().c_forward(parent,step,inputs,target,past,policy,mode,cohort,grant,**kwargs)
            for role,t in zip('qkv',blocks[0]):store.save('raw_qkv_'+role,t)
            for role,t in zip('kv',cache_pairs(out.past_key_values)[0]):store.save('returned_prefix_'+role,t,axis=2,origin=0)
            store.contexts[parent].update(actual_context=record['context'],position_observation=record['position_observation'],backend_after=backend(self.fixture))
            return out,record,blocks,values,blob
        finally:store.current=None


def channel_type(original,store_for):
    class ObservedChannel(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.capture=store_for(self.adapter);self.last_diagnostic_header=None
        def observe(self,h,payload,direction):
            if h['kind'] not in ('HIDDEN','DELTAS') or h['layer_id']!=0:return
            if h['kind']=='HIDDEN':c=h['metadata']['r4c_forward']
            else:
                c=self.last_diagnostic_header['metadata']['r4c_forward'] if self.last_diagnostic_header else None
                if c is None:return
            if c['pass_index'] or not self.capture.select(c['parent_request_id'],c['past_length'],c['input_length']):return
            self.capture.blob('wire_'+direction+'_'+h['kind'],payload,h['shape'],h['dtype'],'cpu',1 if h['kind']=='HIDDEN' else 2,
                full_payload_sha256=h['payload_sha256'],full_payload_shape=h['shape'],full_payload_bytes=h['payload_bytes'])
        def send(self,h,payload=b''):
            result=super().send(h,payload)
            if h['kind']=='HIDDEN':self.last_diagnostic_header=result['header']
            self.observe(result['header'],payload,'send');return result
        def recv(self,**expected):
            h,b,r=super().recv(**expected)
            if h['kind']=='HIDDEN':self.last_diagnostic_header=h
            self.observe(h,b,'recv');return h,b,r
    return ObservedChannel


def execute(plan,root,writer):
    from . import r5_i5_runtime as runtime
    from .r5_multi_artifacts import validate
    _,_,specs,_=validate(plan['multi_prepared']);stores=[];lookup={}
    original_driver,original_channel=runtime.QualityDriver,runtime.Channel
    class Driver(DiagnosticDriver):
        diagnostic_plan=plan;diagnostic_writer=writer
        def configure(self,output,**kwargs):
            result=super().configure(output,**kwargs)
            lookup[specs[self.identity['adapter']]['matrix_sha256']]=self.capture;return result
    Driver.stores=stores
    try:
        runtime.QualityDriver=Driver;runtime.Channel=channel_type(original_channel,lambda fingerprint:lookup[fingerprint])
        runtime.execute(plan,root,writer);ok=True
    finally:
        runtime.QualityDriver=original_driver;runtime.Channel=original_channel
        for store in stores:store.finish(ok='ok' in locals())


def serve(plan,output,writer):
    from . import r5_i5_session as session
    store=Capture(output,writer,plan['i5r2b'],'UD');original_channel,original_math=session.Channel,session.cpu_deltas;instances=[]
    class Channel(channel_type(original_channel,lambda fingerprint:store)):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);instances.append(self)
    def arithmetic(values,matrices,layer,hidden,**kwargs):
        h=instances[0].last_diagnostic_header;c=h['metadata']['r4c_forward'] if h else None
        selected=layer==0 and c is not None and not c['pass_index'] and store.select(c['parent_request_id'],c['past_length'],c['input_length'])
        if not selected or c['past_length']>=64:return original_math(values,matrices,layer,hidden,**kwargs)
        n=c['input_length'];fixture=plan['fixture'];counter=0;last={}
        store.contexts[c['parent_request_id']].update(actual_context=c,backend_before=backend(fixture))
        store.save('CPU_reconstructed_hidden',ArrayTensor(values,[1,n,hidden]) if fixture else values)
        def observe(name,args,kw,out):
            nonlocal counter
            if name=='linear':
                role='qkv'[counter//2];phase='A' if counter%2==0 else 'B';counter+=1
                if phase=='A':store.save('CPU_input_'+role,args[0])
                else:store.save('CPU_B_input_'+role,args[0])
                store.save('CPU_'+phase+'_'+role,out)
                if phase=='B':last.update(role=role,tensor=out)
            elif name in ('mul','__mul__') and args and args[0] is last.get('tensor'):
                store.save('CPU_scaled_'+last['role'],out,scale=1.)
        old_trace=sys.getprofile()
        def scalar_trace(frame,event,arg):
            nonlocal counter
            if event=='return' and frame.f_code.co_name=='linear' and Path(frame.f_code.co_filename).name=='r3a_tiny.py':
                role='qkv'[counter//2];phase='A' if counter%2==0 else 'B';counter+=1
                if phase=='A':store.save('CPU_input_'+role,ArrayTensor(frame.f_locals['x'],[1,n,hidden]))
                else:store.save('CPU_B_input_'+role,ArrayTensor(frame.f_locals['x'],[1,n,8]))
                store.save('CPU_'+phase+'_'+role,ArrayTensor(arg,[1,n,8 if phase=='A' else hidden]))
                if phase=='B':store.save('CPU_scaled_'+role,ArrayTensor(arg,[1,n,hidden]),scale='fixture scale-one contract, no separate multiplication')
        try:
            if fixture:require(old_trace is None,'fixture profile conflict');sys.setprofile(scalar_trace)
            with operation_mode(observe,enabled=not fixture):result=original_math(values,matrices,layer,hidden,**kwargs)
            require(counter==6,'CPU LoRA observer did not see exactly six original Linear calls')
            store.save('CPU_returned_bundle',ArrayTensor(result,[3,1,n,hidden]) if fixture else result,axis=2)
            store.contexts[c['parent_request_id']]['backend_after']=backend(fixture);return result
        finally:
            if fixture:sys.setprofile(old_trace)
    try:
        session.Channel=Channel;session.cpu_deltas=arithmetic;session.serve(plan,output,writer)
        writer.write(Path(output)/'i5r2b_backend.json',backend(plan['fixture']));ok=True
    finally:
        session.Channel=original_channel;session.cpu_deltas=original_math;store.finish(ok='ok' in locals())
