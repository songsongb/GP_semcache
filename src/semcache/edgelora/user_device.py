"""R3-A independent CPU-only adapter service. No full model or PEFT import."""
import os
from pathlib import Path
import socket
import sys
import time

from .communication import Channel, header, listener, require
from .r3a_artifacts import append, flags, load_matrices, process_environment, tensor_bytes, tensor_from_bytes, unpack, pack, validate_prepared, versions, write


def configure_cpu(torch_backend):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'UD must start with CUDA hidden')
    before = sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    if before:
        os.sched_setaffinity(0,before[:4])
    effective = sorted(os.sched_getaffinity(0)) if before else None
    threads = None
    if torch_backend:
        import torch
        require(not torch.cuda.is_initialized(), 'UD CUDA context detected')
        torch.set_num_threads(min(4,len(effective)) if effective else 4)
        torch.set_num_interop_threads(1)
        threads = torch.get_num_threads()
    return dict(requested_threads=4, effective_threads=threads, affinity_before=before,
                effective_affinity=effective, cpu_limit_applied=bool(effective),
                backend='torch CPU' if torch_backend else 'stdlib scalar fixture (single Python thread)')


def cpu_deltas(values, matrices, layer, hidden, *, torch_backend, precision_out=None):
    if torch_backend:
        import torch
        import torch.nn.functional as F
        with torch.inference_mode():
            result = []
            for role in 'qkv':
                a,b = matrices[layer,role,'A'], matrices[layer,role,'B']
                x=values.to(a.dtype);intermediate=F.linear(x,a);out=F.linear(intermediate,b)*1.
                require(intermediate.dtype==a.dtype and out.dtype==b.dtype and out.device.type=='cpu', 'CPU precision/placement mismatch')
                if precision_out is not None:
                    precision_out[role]=dict(input_dtype=str(x.dtype),A_weight_dtype=str(a.dtype),A_output_dtype=str(intermediate.dtype),
                        B_weight_dtype=str(b.dtype),B_output_dtype=str(out.dtype),scale=1.)
                result.append(out)
            return torch.stack(result,dim=0)
    from .r3a_tiny import linear
    return [v for role in 'qkv' for v in linear(linear(values,matrices[layer,role,'A'],hidden,8),
                                               matrices[layer,role,'B'],8,hidden)]


def serve(prepared, output, session_id):
    output = Path(output)
    started = time.monotonic_ns()
    pm,ref,am = validate_prepared(prepared)
    fixture = pm.get('TEST_FIXTURE_ONLY') is True
    cpu = configure_cpu(not fixture)
    matrices = load_matrices(prepared,am,torch_backend=not fixture)
    identity = pm['identity']; shape = [1,identity['length'],identity['hidden_size']]
    loaded = time.monotonic_ns()
    records = []; sock = connection = channel = None
    result = dict(execution_status='RUNNING', pid=os.getpid(), parent_pid=os.getppid(),
        identity_sha256=pm['identity_sha256'], adapter_matrix_sha256=am['matrix_sha256'], cpu=cpu,
        process_start_monotonic_ns=started, adapter_ready_monotonic_ns=loaded,
        adapter_load_and_preflight_ms=(loaded-started)/1e6, full_model_loaded=False, versions=versions(),
        environment=process_environment(),
        **flags(fixture=fixture))
    try:
        sock = listener(); port = sock.getsockname()[1]
        write(output/'r3a_ud_ready.json',dict(port=port,pid=os.getpid(),cpu=cpu,identity_sha256=pm['identity_sha256']))
        connection,address = sock.accept()
        require(address[0] == '127.0.0.1', 'non-loopback client')
        connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        sock.close(); sock = None  # no second client/listener survives the handshake
        channel = Channel(connection,session_id,am['matrix_sha256'],trace=lambda r:append(output/'r3a_ud_wire_trace.jsonl',r))
        hello,_,_ = channel.recv(kind='HELLO',request_id='handshake',layer_id=-1)
        require(hello['metadata']['identity_sha256'] == pm['identity_sha256'], 'UD handshake identity mismatch')
        channel.send(header('READY','handshake',-1,pid=os.getpid(),identity_sha256=pm['identity_sha256'],
            user_device_cuda_used=False,cpu=cpu,versions=versions()))
        result['actual_network_socket_used'] = True
        layer,request,seen = 0,None,set()
        while True:
            h,payload,received = channel.recv()
            if h['kind'] == 'STOP':
                require(layer == 0 and h['request_id'] == 'shutdown', 'stop inside unfinished request')
                channel.send(header('BYE','shutdown',-1,pid=os.getpid(),completed_requests=len(seen)))
                break
            require(h['kind'] == 'HIDDEN' and h['layer_id'] == layer and h['shape'] == shape and
                    h['dtype'] == identity['hidden_dtype'], 'UD layer/dtype/shape mismatch')
            if layer == 0:
                require(h['request_id'] not in seen and len(seen) < 10000, 'duplicate/excess request')
                request = h['request_id']; seen.add(request)
            require(h['request_id'] == request, 'UD request ID changed mid-forward')
            begin = time.monotonic_ns()
            values = unpack(payload,h['dtype']) if fixture else tensor_from_bytes(payload,h['dtype'],shape)
            decoded = time.monotonic_ns()
            precision={}
            delta = cpu_deltas(values,matrices,layer,shape[-1],torch_backend=not fixture,precision_out=precision)
            computed = time.monotonic_ns()
            reply = pack(delta,identity['delta_dtype']) if fixture else tensor_bytes(delta)
            encoded = time.monotonic_ns()
            if not fixture:
                import torch
                require(not torch.cuda.is_initialized() and all(t.device.type == 'cpu' for t in matrices.values()), 'UD isolation violation')
                require(not any(n == 'transformers' or n.startswith('transformers.') for n in sys.modules), 'full model package imported on UD')
            metrics = dict(pid=os.getpid(),user_device_cuda_used=False,
                input_deserialization_ms=(decoded-begin)/1e6, cpu_lora_ms=(computed-decoded)/1e6,
                output_serialization_ms=(encoded-computed)/1e6, monotonic_start_ns=begin, monotonic_end_ns=encoded,
                hidden_payload_sha256=h['payload_sha256'], A_B_dtype=identity['delta_dtype'],
                intermediate_dtype=identity['delta_dtype'], scale_dtype=identity['delta_dtype'])
            metrics['actual_precision']=precision if not fixture else {'scope':'stdlib TEST_FIXTURE_ONLY scalar f32 rounded linear outputs'}
            sent = channel.send(header('DELTAS',request,layer,identity['delta_dtype'],[3]+shape,**metrics),reply)
            records.append(dict(request_id=request,layer_id=layer,metrics=metrics,
                hidden_frame_sha256=received['frame_sha256'],delta_frame_sha256=sent['frame_sha256']))
            layer = (layer+1)%identity['layers']
        result.update(execution_status='COMPLETE',completed_requests=len(seen),layers=records)
    except BaseException as exc:
        result.update(execution_status='FAILED',error=f'{type(exc).__name__}: {exc}',layers=records)
        raise
    finally:
        for resource in (channel,connection,sock):
            if resource is not None: resource.close()
        result['closed_monotonic_ns'] = time.monotonic_ns()
        write(output/'r3a_ud_result.json',result)
