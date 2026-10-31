"""R5 multi-adapter envelope over unchanged C1 generation and TCP arithmetic."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import time

from .communication import Channel,digest,header,listener,require
from .r3a_artifacts import pack,read,tensor_bytes,tensor_from_bytes,unpack,process_environment
from .r4c_session import ForwardOrder
from .r5_contracts import validate_request
from .r5_multi_artifacts import load,resolve,sharing,validate
from .user_device import configure_cpu,cpu_deltas

VERSION='R5_MULTI_ADAPTER_TCP_V1'


class SessionGuard:
    def __init__(self,registry,physical):
        self.registry=deepcopy(registry);self.physical=deepcopy(physical);self.requests={}
    def accept(self,route,context):
        require(set(route)=={'schema','registry_sha256','request'} and route['schema']==VERSION and
                route['registry_sha256']==self.registry['sha256'],'invalid multi-adapter route envelope')
        req=validate_request(route['request']);aid=resolve(self.registry,req['user_id'],req['adapter_id'])
        require(aid==self.physical['adapter_id'] and req['physical_identity']==self.physical and
                req['request_id']==context['parent_request_id'] and req['token_ids']==context['target_token_ids'] and
                req['generation_policy']==context['policy'],'route/request/adapter/physical identity mismatch')
        previous=self.requests.get(req['request_id'])
        require(previous is None or previous==req,'request mapping changed during generation')
        if context['source_request_id'] is not None:
            source=self.requests.get(context['source_request_id'])
            require(source is not None and sharing(self.registry,source['user_id'],req['user_id'],source['adapter_id'],aid) is None,
                    'unauthorized source/target adapter or sharing')
        self.requests[req['request_id']]=deepcopy(req);return req


class RoutedChannel:
    def __init__(self,channel,registry,physical):
        self.channel=channel;self.registry=registry;self.physical=physical;self.request=None
    @property
    def records(self):return self.channel.records
    def send(self,h,payload=b''):
        require(self.request is not None and self.request['physical_identity']==self.physical and
                resolve(self.registry,self.request['user_id'],self.request['adapter_id'])==self.physical['adapter_id'],'unbound adapter route')
        route=dict(schema=VERSION,registry_sha256=self.registry['sha256'],request=self.request)
        return self.channel.send(dict(h,metadata=dict(h['metadata'],r5_multi=route)),payload)
    def recv(self,**expected):
        h,b,r=self.channel.recv(**expected)
        route=dict(schema=VERSION,registry_sha256=self.registry['sha256'],request=self.request)
        require(h['metadata']['route_sha256']==digest(route) and h['metadata']['adapter_id']==self.physical['adapter_id'] and
                h['metadata']['matrix_sha256']==self.physical['adapter_matrix_fingerprint'],'wrong adapter correction receipt')
        return h,b,r


def serve(plan,output,writer):
    output=Path(output);m,pm,adapters,registry=validate(plan['multi_prepared']);spec=adapters[plan['adapter_id']]
    require(m['TEST_FIXTURE_ONLY']==plan['fixture'] and spec['matrix_sha256']==plan['physical_identity']['adapter_matrix_fingerprint'],
            'UD adapter preparation/identity mismatch')
    cpu=configure_cpu(not plan['fixture']);matrices=load(plan['multi_prepared'],spec,torch_backend=not plan['fixture'])
    identity=dict(pm['identity'],adapter=spec['adapter_id'],adapter_matrix_sha256=spec['matrix_sha256'],delta_dtype=spec['matrix_dtype'])
    order=ForwardOrder(plan['cohort_sha256'],identity['layers'],plan['context_limit']);guard=SessionGuard(registry,plan['physical_identity'])
    writer.write(output/'initial.json',dict(pid=os.getpid(),session_id=plan['session_id'],adapter_id=spec['adapter_id'],matrix_sha256=spec['matrix_sha256'],
        grants=[],parents=[],finished=[],layer=order.layer,current=order.current,cpu=cpu))
    result=dict(status='RUNNING',pid=os.getpid(),session_id=plan['session_id'],adapter_id=spec['adapter_id'],matrix_sha256=spec['matrix_sha256'],
        user_device_cuda_used=False,full_model_loaded=False,environment=process_environment(),TEST_FIXTURE_ONLY=plan['fixture'])
    sock=channel=None;count=0
    try:
        sock=listener();writer.write(output/'ready.json',dict(pid=os.getpid(),port=sock.getsockname()[1]))
        connection,address=sock.accept();require(address[0]=='127.0.0.1','nonlocal UD');sock.close();sock=None
        channel=Channel(connection,plan['session_id'],spec['matrix_sha256'],timeout=180.,trace=lambda f:writer.append(output/'wire.jsonl',f))
        h,_,_=channel.recv(kind='HELLO',request_id='handshake',layer_id=-1)
        hello=dict(application=VERSION,adapter_id=spec['adapter_id'],matrix_sha256=spec['matrix_sha256'],registry_sha256=registry['sha256'],
                   physical_identity=plan['physical_identity'],cohort_sha256=plan['cohort_sha256'])
        require(h['metadata']==hello,'multi-adapter handshake mismatch')
        channel.send(header('READY','handshake',-1,**hello,pid=os.getpid(),user_device_cuda_used=False))
        while True:
            h,payload,received=channel.recv()
            if h['kind']=='STOP':order.finish();channel.send(header('BYE','shutdown',-1));break
            if h['kind']=='HELLO':
                require(set(h['metadata'])=={'application','control'} and h['metadata']['application']==VERSION,'invalid multi-adapter control')
                control=h['metadata']['control'];order.control(control)
                writer.append(output/'controls.jsonl',dict(control,request_id=h['request_id']))
                channel.send(header('READY',h['request_id'],-1,control_sha256=digest(control),pid=os.getpid(),user_device_cuda_used=False));continue
            require(h['kind']=='HIDDEN' and set(h['metadata'])=={'r4c_forward','r5_multi'},'unexpected correction request')
            c=h['metadata']['r4c_forward'];req=guard.accept(h['metadata']['r5_multi'],c);c,shape=order.accept(h,identity)
            start=time.monotonic_ns();values=unpack(payload,h['dtype']) if plan['fixture'] else tensor_from_bytes(payload,h['dtype'],shape)
            precision={};math_start=time.monotonic_ns()
            delta=cpu_deltas(values,matrices,h['layer_id'],identity['hidden_size'],torch_backend=not plan['fixture'],precision_out=precision)
            math_end=time.monotonic_ns();blob=pack(delta,identity['delta_dtype']) if plan['fixture'] else tensor_bytes(delta);end=time.monotonic_ns()
            if not plan['fixture']:
                import torch
                require(not torch.cuda.is_initialized() and all(t.device.type=='cpu' for t in matrices.values()) and
                        not any(n=='transformers' or n.startswith('transformers.') for n in sys.modules),'UD CUDA/model isolation failed')
            metrics=dict(pid=os.getpid(),user_device_cuda_used=False,adapter_id=spec['adapter_id'],matrix_sha256=spec['matrix_sha256'],
                user_id=req['user_id'],route_sha256=digest(h['metadata']['r5_multi']),hidden_payload_sha256=h['payload_sha256'],
                r4c_forward_sha256=digest(c),cpu_lora_ms=(math_end-math_start)/1e6,
                cpu_lora_and_serialization_ms=(end-start)/1e6,actual_precision=precision)
            sent=channel.send(header('DELTAS',h['request_id'],h['layer_id'],identity['delta_dtype'],[3]+shape,**metrics),blob)
            writer.append(output/'layers.jsonl',dict(forward_id=h['request_id'],context=c,layer=h['layer_id'],metrics=metrics,
                hidden_frame_sha256=received['frame_sha256'],delta_frame_sha256=sent['frame_sha256']));order.advance();count+=1
        result.update(status='COMPLETE',parents=len(order.parents),forwards=order.completed,finished=order.finished)
    except BaseException as exc:result.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}');raise
    finally:
        for resource in (channel,sock):
            if resource is not None:resource.close()
        result.update(layers=count,grants=list(order.grants),closed_ns=time.monotonic_ns());writer.write(output/'result.json',result)
