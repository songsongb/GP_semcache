"""Opt-in bounded wall/exclusive-wall and optional CUDA stream-span receipts.

No integrity memoization, arithmetic changes, implicit enablement or normal-path
CUDA synchronization. Nested scopes are explicitly inclusive, never additive.
"""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
import time,math
from .communication import require,digest

CURRENT=ContextVar('r5_subsequence_profiler',default=None)
VERSION='SPARSE_PREFILL_DIAGNOSTICS_V1'

class Profiler:
    def __init__(self,*,cuda_events=False,device=None,max_calls=500000,max_details=4096):
        require(0<max_calls<=500000 and 0<max_details<=4096,'invalid profile bounds')
        self.cuda_events=cuda_events;self.device=device;self.max_calls=max_calls;self.max_details=max_details
        self.stats={};self.calls=0;self.details=[];self.events=[];self.stack=[];self.entries={}
        self.layer=-1;self.role='';self.phase='SETUP';self.request_id=None
    @contextmanager
    def activate(self,request_id):
        require(CURRENT.get() is None,'nested request profiling forbidden')
        self.request_id=request_id;token=CURRENT.set(self)
        try:yield self
        finally:CURRENT.reset(token)
    @contextmanager
    def location(self,layer,role,phase='PREFILL'):
        previous=self.layer,self.role,self.phase;self.layer,self.role,self.phase=layer,role,phase
        try:yield
        finally:self.layer,self.role,self.phase=previous
    @contextmanager
    def scope(self,name,*,byte_count=0,cuda=False):
        self.calls+=1;require(self.calls<=self.max_calls,'profile call budget exhausted')
        key=(self.phase,self.layer,self.role,name);require(key in self.stats or len(self.stats)<8192,'profile category budget exhausted')
        stat=self.stats.setdefault(key,dict(count=0,bytes=0,inclusive_wall_ms=0.,exclusive_wall_ms=0.,max_wall_ms=0.,cuda_stream_span_ms=None))
        events=None
        if cuda and self.cuda_events:
            import torch
            require(self.device is not None and str(self.device).startswith('cuda'),'CUDA events need actual CUDA device')
            require(len(self.events)<8192,'CUDA event budget exhausted')
            events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));events[0].record()
        frame=[time.perf_counter_ns(),0];self.stack.append(frame)
        try:yield
        finally:
            if events:events[1].record();self.events.append((key,events))
            elapsed=time.perf_counter_ns()-frame[0];self.stack.pop()
            if self.stack:self.stack[-1][1]+=elapsed
            stat['count']+=1;stat['bytes']+=byte_count;stat['inclusive_wall_ms']+=elapsed/1e6
            stat['exclusive_wall_ms']+=(elapsed-frame[1])/1e6;stat['max_wall_ms']=max(stat['max_wall_ms'],elapsed/1e6)
    def detail(self,record):
        require(len(self.details)<self.max_details,'eligibility detail budget exhausted')
        self.details.append(dict(request_id=self.request_id,**record))
    def entry(self,key):
        name=digest(key);require(name in self.entries or len(self.entries)<4096,'entry profile budget exhausted')
        self.entries[name]=self.entries.get(name,0)+1
    def finish(self):
        require(not self.stack,'unfinished profile scopes')
        sync_ms=0.
        if self.events:
            import torch
            start=time.perf_counter_ns();torch.cuda.synchronize(self.device);sync_ms=(time.perf_counter_ns()-start)/1e6
            for key,(a,b) in self.events:
                stat=self.stats[key];stat['cuda_stream_span_ms']=(stat['cuda_stream_span_ms'] or 0.)+a.elapsed_time(b)
        return dict(version=VERSION,request_id=self.request_id,calls=self.calls,
            components=[dict(phase=k[0],layer=k[1],role=k[2],component=k[3],**v) for k,v in sorted(self.stats.items())],
            entry_proof_executions_by_key=self.entries,eligibility=self.details,cuda_events=self.cuda_events,
            final_observer_synchronize_ms=sync_ms,observer_effect='DIAGNOSTIC_ONLY; CUDA events optionally add final synchronization; hashing already entails synchronous host transfers',
            timing_scope='exclusive wall categories partition observed scopes only; inclusive scopes and CUDA stream spans overlap; stream spans are not kernel-only time')

def call(name,fn,*args,byte_count=0,cuda=False,**kwargs):
    p=CURRENT.get()
    if p is None:return fn(*args,**kwargs)
    if name in ('native_sparse_linear','native_dense_linear'):
        x=args[0];p.detail(dict(boundary='native_projection_layout',layer=p.layer,role=p.role,phase=p.phase,
            shape=list(x.shape),dtype=str(x.dtype),device=str(x.device),
            stride=list(x.stride()) if hasattr(x,'stride') else 'NOT_OBSERVABLE',
            contiguous=x.is_contiguous() if hasattr(x,'is_contiguous') else 'NOT_OBSERVABLE',
            storage_offset=x.storage_offset() if hasattr(x,'storage_offset') else 'NOT_OBSERVABLE'))
    with p.scope(name,byte_count=byte_count,cuda=cuda):return fn(*args,**kwargs)

def profiled(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args,**kwargs):return call(name,fn,*args,**kwargs)
        return wrapped
    return decorate

def validate_receipt(receipt,request_id,physical):
    require(receipt['version']==VERSION and receipt['request_id']==request_id and 0<receipt['calls']<=500000,'profile identity/budget differs')
    stats=receipt['components'];require(len(stats)<=8192 and len(receipt['eligibility'])<=4096,'profile inventory exceeds bound')
    require(sum(r['count'] for r in stats)==receipt['calls'],'profile call count differs')
    for r in stats:
        require(type(r['count']) is int and r['count']>0 and r['bytes']>=0,'invalid component count')
        require(all(math.isfinite(r[k]) and r[k]>=0 for k in ('inclusive_wall_ms','exclusive_wall_ms','max_wall_ms')) and r['exclusive_wall_ms']<=r['inclusive_wall_ms']+1e-6 and r['max_wall_ms']<=r['inclusive_wall_ms']+1e-6,'invalid profile wall scope')
    by_name=lambda name:sum(r['count'] for r in stats if r['component']==name)
    require(by_name('entry_proof_all_layers')==sum(receipt['entry_proof_executions_by_key'].values()),'entry proof count differs')
    require(sum(r['bytes'] for r in stats if r['component']=='tensor_sha256')==sum(r['bytes'] for r in stats if r['component']=='tensor_serialization_host_transfer_inclusive'),'tensor proof serialization/hash bytes differ')
    if physical is not None:
        for layer in {r['layer'] for r in physical['records']}:
            details=[r for r in receipt['eligibility'] if r['boundary']=='layer_eligibility' and r['layer']==layer]
            accepted=sorted(i for r in details if r['reason']=='ACCEPTED' for i in range(*r['target_positions']))
            actual=next(r for r in physical['records'] if r['layer']==layer and r['role']=='q')['reused_positions']
            require(accepted==actual,'eligibility receipt differs from physical consumption')
            for r in details:
                if r['reason']=='ACCEPTED':require(r['ownership_integrity_verified'] and (physical.get('mode','EXACT_CONTEXT_VERIFIED')!='EXACT_CONTEXT_VERIFIED' or r['source_input_proof']==r['target_input_proof']),'exact accepted input proof differs')
                elif r['reason']=='PER_LAYER_INPUT_BYTES_MISMATCH':require(r['source_input_proof']!=r['target_input_proof'],'false input mismatch receipt')
