"""Opt-in monotonic attribution around the unchanged C1 forward implementation."""
from contextlib import contextmanager
import inspect
from pathlib import Path
import statistics
import sys
import tempfile
import time

from .communication import require
from .r4c_generation import QualityDriver


class Timing:
    def __init__(self):
        self.events=[];self.request_id=None;self.request_index=None;self.step=None;self.trace_overhead_ns=0

    @contextmanager
    def span(self,name,**fields):
        start=time.monotonic_ns()
        try:yield
        finally:self.events.append(dict(name=name,start_ns=start,end_ns=time.monotonic_ns(),
            request_id=self.request_id,request_index=self.request_index,step=self.step,**fields))

    def totals(self,start=0):
        result={}
        for e in self.events[start:]:result[e['name']]=result.get(e['name'],0.)+(e['end_ns']-e['start_ns'])/1e6
        return result


class ProfiledDriver(QualityDriver):
    """No duplicate model forward. Line events delimit only the frozen wrapper.

    Bins are exclusive within c_forward. Nested evidence/IO/model/sync spans
    overlap those bins and must never be summed with them. Trace overhead is
    retained; both arms use exactly this instrumentation.
    """
    def configure(self,*args,**kwargs):
        result=super().configure(*args,**kwargs);self.perf=Timing()
        source,start=inspect.getsourcelines(QualityDriver.c_forward);category='forward_preflight';self._bins={}
        for i,line in enumerate(source):
            if 'self.evidence.check(' in line:category='evidence_budget_check'
            elif line.lstrip().startswith('old='):category='forward_preflight'
            elif line.lstrip().startswith('if self.fixture:'):category='distributed_forward_and_internal_hooks'
            elif line.lstrip().startswith('summary=None'):category='output_past_inspection'
            elif 'values,blob=self.last_logits(' in line:category='logit_extraction'
            elif "require(len(observed)" in line:category='forward_internal_evidence_and_validation'
            elif "append(self.output/'forward_completed" in line:category='forward_record_append'
            self._bins[start+i]=category
        backend=self.tiny if self.fixture else self.model;original=backend.forward
        def actual(*a,**k):
            with self.perf.span('actual_opt_distributed_forward_nested'):return original(*a,**k)
        backend.forward=actual
        return result

    def synchronize(self):
        if not hasattr(self,'perf'):return super().synchronize()
        with self.perf.span('explicit_gpu_synchronization'):return super().synchronize()

    def last_logits(self,*args,**kwargs):
        with self.perf.span('logit_extraction_nested'):return super().last_logits(*args,**kwargs)

    def c_forward(self,*args,**kwargs):
        require(sys.gettrace() is None,'another trace profiler is active')
        code=QualityDriver.c_forward.__code__;previous=None;exclusive={};trace_ns=0;start_events=len(self.perf.events)
        def trace(frame,event,arg):
            nonlocal previous,trace_ns
            if frame.f_code is not code:return None
            entered=time.monotonic_ns()
            if previous is not None:
                name,stamp=previous;exclusive[name]=exclusive.get(name,0)+entered-stamp
            previous=(self._bins.get(frame.f_lineno,'forward_wrapper_other'),time.monotonic_ns()) if event!='return' else None
            trace_ns+=time.monotonic_ns()-entered
            return trace
        self.perf.step=args[1];started=time.monotonic_ns()
        sys.settrace(trace)
        try:result=super().c_forward(*args,**kwargs)
        finally:sys.settrace(None)
        ended=time.monotonic_ns();out,record,blocks,values,blob=result
        frames=self.channel.records
        relevant=[f for f in frames if f['header']['request_id']==record['forward_id']]
        record['perf_forward']=dict(exclusive_ms={k:v/1e6 for k,v in exclusive.items()},
            c_forward_ms=(ended-started)/1e6,trace_callback_ms=trace_ns/1e6,nested_ms=self.perf.totals(start_events),
            cpu_ud_ms=sum(f['header']['metadata'].get('cpu_lora_and_serialization_ms',0.) for f in relevant if f['header']['kind']=='DELTAS'),
            tcp_send_ms=sum((f['monotonic_end_ns']-f['monotonic_start_ns'])/1e6 for f in relevant if f['operation']=='send'),
            tcp_receive_ms=sum((f['monotonic_end_ns']-f['monotonic_start_ns'])/1e6 for f in relevant if f['operation']=='recv'),
            cuda_event_ms=record['cuda_event_elapsed_ms'],evidence=self.evidence.diagnostics())
        if not self.fixture:
            import torch
            record['perf_forward']['gpu_memory']=dict(allocated=torch.cuda.memory_allocated(self.device),reserved=torch.cuda.memory_reserved(self.device),peak=torch.cuda.max_memory_allocated(self.device))
        record['perf_forward']['profiler_finalize_ms']=(time.monotonic_ns()-ended)/1e6
        return out,record,blocks,values,blob


def scan_diagnostic(counts=(0,1000,8000,32000),repeats=3):
    """Actual filesystem measurements, not OPT or historical GPU attribution."""
    from .r4c_evidence import Evidence
    rows=[]
    with tempfile.TemporaryDirectory(prefix='r5_scan_attribution_') as d:
        p=Path(d);created=0
        for count in counts:
            require(count>=created,'scan counts must increase')
            for i in range(created,count):(p/f'{i:06d}.bin').write_bytes(b'x')
            created=count;checker=Evidence(p,budget_bytes=2**30);elapsed=[]
            for _ in range(repeats):
                t=time.monotonic_ns();actual=checker.check();elapsed.append((time.monotonic_ns()-t)/1e6)
            rows.append(dict(files=count,bytes=actual,check_ms=elapsed,median_ms=statistics.median(elapsed)))
    return dict(schema='R5_SCAN_ATTRIBUTION_V1',scope='CPU_FILESYSTEM_ONLY',rows=rows,
        historical_job_verified=False,historical_slowdown_reproduced=False,
        interpretation='measures the actual legacy checker; does not assign historical GPU step gaps to it')
