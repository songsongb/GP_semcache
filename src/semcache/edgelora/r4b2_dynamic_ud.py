"""Independent B2 CPU LoRA endpoint, including completed-source prefix grants."""
import os
from pathlib import Path
import sys
import time

from .communication import Channel,digest,header,listener,require
from .r3a_artifacts import append,load_matrices,pack,process_environment,tensor_bytes,tensor_from_bytes,unpack,validate_prepared,write
from .user_device import configure_cpu,cpu_deltas
from .r4b2_session import ForwardOrder


def serve(plan,output):
    output=Path(output);pm,ref,am=validate_prepared(plan['base_prepared']);fixture=plan['fixture']
    require(pm.get('TEST_FIXTURE_ONLY',False)==fixture,'UD scope mismatch');cpu=configure_cpu(not fixture)
    matrices=load_matrices(plan['base_prepared'],am,torch_backend=not fixture);identity=pm['identity'];sock=channel=None
    order=ForwardOrder(plan['cohort_sha256'],identity['layers']);count=0
    result=dict(execution_status='RUNNING',pid=os.getpid(),cpu=cpu,environment=process_environment(),user_device_cuda_used=False,
        full_model_loaded=False,adapter_matrix_sha256=am['matrix_sha256'],cohort_sha256=plan['cohort_sha256'],TEST_FIXTURE_ONLY=fixture)
    try:
        sock=listener();write(output/'r4b2_ud_ready.json',dict(pid=os.getpid(),port=sock.getsockname()[1]))
        connection,address=sock.accept();require(address[0]=='127.0.0.1','nonlocal UD');sock.close();sock=None
        channel=Channel(connection,plan['session_id'],am['matrix_sha256'],timeout=120.,trace=lambda r:append(output/'ud_wire.jsonl',r))
        h,_,_=channel.recv(kind='HELLO',request_id='handshake',layer_id=-1)
        require(h['metadata']==dict(application='R4B2_PREFIX_V2',cohort_sha256=plan['cohort_sha256'],identity_sha256=pm['identity_sha256']),
            'B2 handshake identity mismatch')
        channel.send(header('READY','handshake',-1,pid=os.getpid(),application='R4B2_PREFIX_V2',cohort_sha256=plan['cohort_sha256'],user_device_cuda_used=False))
        while True:
            h,payload,received=channel.recv()
            if h['kind']=='STOP':order.finish();channel.send(header('BYE','shutdown',-1));break
            if h['kind']=='HELLO':
                order.control(h['metadata']);append(output/'ud_prefix_controls.jsonl',dict(h['metadata'],request_id=h['request_id']))
                channel.send(header('READY',h['request_id'],-1,control_sha256=digest(h['metadata']),pid=os.getpid(),user_device_cuda_used=False));continue
            c,shape=order.accept(h,identity);start=time.monotonic_ns()
            values=unpack(payload,h['dtype']) if fixture else tensor_from_bytes(payload,h['dtype'],shape);precision={}
            delta=cpu_deltas(values,matrices,h['layer_id'],identity['hidden_size'],torch_backend=not fixture,precision_out=precision)
            blob=pack(delta,identity['delta_dtype']) if fixture else tensor_bytes(delta);end=time.monotonic_ns()
            if not fixture:
                import torch
                require(not torch.cuda.is_initialized() and all(t.device.type=='cpu' for t in matrices.values()) and
                    not any(n=='transformers' or n.startswith('transformers.') for n in sys.modules),'UD CUDA/model isolation failed')
            metrics=dict(pid=os.getpid(),user_device_cuda_used=False,hidden_payload_sha256=h['payload_sha256'],
                r4b2_forward_sha256=digest(c),cpu_lora_and_serialization_ms=(end-start)/1e6,actual_precision=precision)
            sent=channel.send(header('DELTAS',h['request_id'],h['layer_id'],identity['delta_dtype'],[3]+shape,**metrics),blob)
            append(output/'ud_layers.jsonl',dict(forward_id=h['request_id'],context=c,layer=h['layer_id'],metrics=metrics,
                hidden_frame_sha256=received['frame_sha256'],delta_frame_sha256=sent['frame_sha256']));count+=1;order.advance()
        result.update(execution_status='COMPLETE',completed_forwards=order.completed,parents=len(order.parents))
    except BaseException as exc:result.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}');raise
    finally:
        for resource in (channel,sock):
            if resource is not None:resource.close()
        result.update(layers_recorded=count,registered_entries=list(order.grants),closed_monotonic_ns=time.monotonic_ns());write(output/'ud_result.json',result)
