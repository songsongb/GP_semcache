"""Independent CPU-only dynamic LoRA service; original math and original framing."""
import os
from pathlib import Path
import sys
import time

from .communication import Channel,digest,header,listener,require
from .r3a_artifacts import append,load_matrices,pack,process_environment,tensor_bytes,tensor_from_bytes,unpack,validate_prepared,versions,write
from .user_device import configure_cpu,cpu_deltas
from .r4a_session import validate_context


class ForwardOrder:
    def __init__(self):self.layer=0;self.forward=None;self.current=None;self.parents={};self.seen=set();self.completed=0
    def accept(self,h,identity):
        c=validate_context(h['metadata']['r4a_forward']);shape=[1,c['input_length'],identity['hidden_size']]
        require(h['kind']=='HIDDEN' and h['layer_id']==self.layer and h['shape']==shape and h['dtype']==identity['hidden_dtype'],
            'dynamic UD layer/dtype/shape mismatch')
        if self.layer==0:
            require(h['request_id'] not in self.seen and len(self.seen)<2048,'duplicate/excess forward')
            parent=c['parent_request_id'];previous=self.parents.get(parent)
            if previous is None:
                require(c['pass_index']==(1 if c['prefill_reused'] else 0),'missing/repeated prefill')
                if self.current is not None:
                    old=self.parents[self.current]
                    require(old['natural_eos'] or old['pass_index']==old['total_passes']-1,'prior generation incomplete')
                require(len(self.parents)<256,'too many generation requests');self.current=parent
            else:
                require(parent==self.current and c['pass_index']==previous['pass_index']+1 and
                    c['past_length']==previous['past_length']+previous['input_length'] and
                    all(c[k]==previous[k] for k in ('total_passes','prefill_reused','triage','natural_eos')),'decode sequence/past mismatch')
            self.parents[parent]=c;self.forward=h['request_id'];self.seen.add(self.forward);self.context=c
        require(h['request_id']==self.forward and c==self.context,'forward identity changed inside layer sequence')
        return c,shape
    def advance(self,layers):
        self.layer=(self.layer+1)%layers
        if self.layer==0:self.completed+=1
    def finish(self):
        require(self.layer==0,'shutdown during partial forward')
        if self.current is not None:
            c=self.parents[self.current];require(c['natural_eos'] or c['pass_index']==c['total_passes']-1,'shutdown during generation')


def serve(plan,output):
    output=Path(output);pm,ref,am=validate_prepared(plan['base_prepared']);fixture=plan['fixture']
    require(pm.get('TEST_FIXTURE_ONLY',False)==fixture,'UD fixture/production scope mismatch')
    start=time.monotonic_ns();cpu=configure_cpu(not fixture);matrices=load_matrices(plan['base_prepared'],am,torch_backend=not fixture)
    identity=pm['identity'];sock=channel=None;order=ForwardOrder();records=[]
    result=dict(execution_status='RUNNING',pid=os.getpid(),cpu=cpu,environment=process_environment(),versions=versions(),
        user_device_cuda_used=False,full_model_loaded=False,adapter_matrix_sha256=am['matrix_sha256'],
        adapter_load_ms=(time.monotonic_ns()-start)/1e6,TEST_FIXTURE_ONLY=fixture)
    try:
        sock=listener();write(output/'r4a_ud_ready.json',dict(pid=os.getpid(),port=sock.getsockname()[1]))
        connection,address=sock.accept();require(address[0]=='127.0.0.1','nonlocal UD connection');sock.close();sock=None
        channel=Channel(connection,plan['session_id'],am['matrix_sha256'],timeout=120.,trace=lambda r:append(output/'ud_wire.jsonl',r))
        h,_,_=channel.recv(kind='HELLO',request_id='handshake',layer_id=-1)
        require(h['metadata']==dict(application='R4A_GENERATION_V1',identity_sha256=pm['identity_sha256']),'wrong application/identity')
        channel.send(header('READY','handshake',-1,application='R4A_GENERATION_V1',pid=os.getpid(),user_device_cuda_used=False,
            identity_sha256=pm['identity_sha256']))
        while True:
            h,payload,received=channel.recv()
            if h['kind']=='STOP':order.finish();channel.send(header('BYE','shutdown',-1));break
            c,shape=order.accept(h,identity);begin=time.monotonic_ns()
            values=unpack(payload,h['dtype']) if fixture else tensor_from_bytes(payload,h['dtype'],shape)
            decoded=time.monotonic_ns();precision={}
            delta=cpu_deltas(values,matrices,h['layer_id'],identity['hidden_size'],torch_backend=not fixture,precision_out=precision)
            computed=time.monotonic_ns();blob=pack(delta,identity['delta_dtype']) if fixture else tensor_bytes(delta)
            encoded=time.monotonic_ns()
            if not fixture:
                import torch
                require(not torch.cuda.is_initialized() and all(t.device.type=='cpu' for t in matrices.values()) and
                    not any(n=='transformers' or n.startswith('transformers.') for n in sys.modules),'UD CUDA/full-model isolation failure')
            metrics=dict(pid=os.getpid(),user_device_cuda_used=False,hidden_payload_sha256=h['payload_sha256'],r4a_forward_sha256=digest(c),
                input_deserialization_ms=(decoded-begin)/1e6,cpu_lora_ms=(computed-decoded)/1e6,output_serialization_ms=(encoded-computed)/1e6,
                actual_precision=precision,monotonic_start_ns=begin,monotonic_end_ns=encoded)
            sent=channel.send(header('DELTAS',h['request_id'],h['layer_id'],identity['delta_dtype'],[3]+shape,**metrics),blob)
            row=dict(forward_id=h['request_id'],context=c,layer=h['layer_id'],metrics=metrics,
                hidden_frame_sha256=received['frame_sha256'],delta_frame_sha256=sent['frame_sha256'])
            records.append(row);append(output/'ud_layers.jsonl',row);order.advance(identity['layers'])
        result.update(execution_status='COMPLETE',completed_forwards=order.completed,parents=len(order.parents))
    except BaseException as exc:result.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}');raise
    finally:
        for resource in (channel,sock):
            if resource is not None:resource.close()
        result.update(closed_monotonic_ns=time.monotonic_ns(),layers_recorded=len(records));write(output/'ud_result.json',result)
