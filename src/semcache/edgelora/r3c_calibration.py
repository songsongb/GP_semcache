"""Bounded real TCP probes through the same persistent relay used by the model."""
import socket
import statistics
import time
import uuid

from .communication import Channel,digest,header,require
from .r3a_artifacts import append,write
from .r3c_network_proxy import connect,control,loopback_listener

SAMPLES=3
PROBE_BYTES=1024*1024
RTT_ABS_TOL_MS=5.
RTT_REL_TOL=.35
BW_REL_TOL=.25


def payload(size,seed):
    import struct
    require(type(size) is int and 4<=size<=PROBE_BYTES and size%4==0,'bounded probe size required')
    return struct.pack('<f',.125+(seed%4)*.0625)*(size//4)


def echo_worker(output):
    """Independent bounded CPU network sink; never a model/UD inference endpoint."""
    server=loopback_listener();write(output/'echo_ready.json',dict(port=server.getsockname()[1],pid=__import__('os').getpid()))
    shutdown=False;channels=0
    try:
        deadline=time.monotonic()+240.
        while not shutdown:
            require(time.monotonic()<deadline,'calibration echo session deadline exceeded')
            try:sock,address=server.accept()
            except socket.timeout:continue
            require(address[0]=='127.0.0.1','nonlocal probe source');channels+=1
            # First source message identifies the independently-generated probe
            # session. Channel still checks all framing/digests on subsequent IO.
            # Source supplies session through a tiny local ready-plan token,
            # keeping the immutable original frame protocol unchanged.
            plan=__import__('json').loads((output/'echo_session.json').read_text())
            channel=Channel(sock,plan['session'],digest('R3C_NETWORK_PROBE_NOT_MODEL'),timeout=20.,
                trace=lambda r:append(output/'echo_wire.jsonl',r))
            try:
                while True:
                    h,data,_=channel.recv()
                    if h['kind']=='STOP':
                        channel.send(header('BYE','shutdown',-1));shutdown=h['metadata'].get('shutdown_service',False);break
                    require(h['kind']=='HIDDEN','unexpected probe kind')
                    meta=h['metadata'];require(data==payload(len(data),meta['seed']),'independent probe request bytes differ')
                    reply=payload(meta['reply_bytes'],meta['seed']+1)
                    channel.send(header('DELTAS',h['request_id'],h['layer_id'],'float32',[1,1,len(reply)//4]),reply)
            finally:channel.close()
        return dict(execution_status='COMPLETE',sessions=channels,model_inference_performed=False)
    finally:server.close()


def validate_calibration(records,profiles):
    base=next(p for p in profiles if p['rtt_ms']==0 and p['es_to_ud_mbps'] is None and p['ud_to_es_mbps'] is None)
    baseline=statistics.median(r['wall_ms'] for r in records if r['condition_id']==base['id'] and r['probe']=='round_trip')
    result=[]
    for p in profiles:
        block=[r for r in records if r['condition_id']==p['id']]
        require(len(block)==3*SAMPLES,'incomplete calibration samples')
        small=[r for r in block if r['probe']=='round_trip']
        observed=statistics.median(r['wall_ms'] for r in small)-baseline
        serialization=statistics.median(sum(f['wire_bytes_forwarded']*.008/p[f['direction']+'_mbps']
            if p[f['direction']+'_mbps'] else 0. for f in r['proxy_frames']) for r in small)
        residual=observed-p['rtt_ms']-serialization
        tolerance=max(RTT_ABS_TOL_MS,RTT_REL_TOL*p['rtt_ms'])
        directions={};valid=abs(residual)<=tolerance
        for direction,probe in (('es_to_ud','upload'),('ud_to_es','download')):
            frames=[f for r in block if r['probe']==probe for f in r['proxy_frames'] if f['direction']==direction]
            require(len(frames)==SAMPLES and all(f['complete'] for f in frames),'incomplete directional calibration')
            rates=[f['effective_mbps'] for f in frames];rate=statistics.median(rates);requested=p[direction+'_mbps']
            relative=None if requested is None else (rate-requested)/requested
            ok=requested is None or abs(relative)<=BW_REL_TOL
            valid=valid and ok
            directions[direction]=dict(requested_mbps=requested,median_effective_mbps=rate,samples_mbps=rates,
                relative_error=relative,relative_tolerance=BW_REL_TOL,valid=ok,
                timing_scope='relay whole framed bytes / forward span after one-way delay; includes reads/pacing/send')
        result.append(dict(condition=p,status='VALID' if valid else 'INVALID_IMPAIRMENT',samples_per_probe=SAMPLES,
            proxy_baseline_round_trip_ms=baseline,observed_round_trip_increment_ms=observed,
            expected_one_pair_delay_ms=p['rtt_ms'],small_frame_serialization_budget_ms=serialization,
            round_trip_residual_ms=residual,round_trip_tolerance_ms=tolerance,directions=directions,
            interpretation='true small-frame request/reply dependency; not TCP ACK RTT or model request latency'))
    return result


def calibrate(profiles,proxy,token,echo,output):
    """Direct control and proxy control are separately labeled CPU probes."""
    adapter=digest('R3C_NETWORK_PROBE_NOT_MODEL');records=[];direct=[]
    def session(port):
        sid=uuid.uuid4().hex;write(output/'echo_session.json',dict(session=sid))
        return Channel(connect(port),sid,adapter,timeout=20.,trace=lambda r:append(output/'calibration_source_wire.jsonl',r))
    def probe(channel,identifier,send_n,reply_n,seed):
        start=time.monotonic_ns();channel.send(header('HIDDEN',identifier,0,'float32',[1,1,send_n//4],seed=seed,reply_bytes=reply_n),payload(send_n,seed))
        _,reply,_=channel.recv(kind='DELTAS',request_id=identifier,layer_id=0)
        require(reply==payload(reply_n,seed+1),'independent probe reply bytes differ')
        return (time.monotonic_ns()-start)/1e6
    source=session(echo['port'])
    try:
        for i in range(SAMPLES):direct.append(probe(source,f'cal_direct_{i}',32,32,i))
        source.send(header('STOP','shutdown',-1,shutdown_service=False));source.recv(kind='BYE',request_id='shutdown',layer_id=-1)
    finally:source.close()
    control(proxy['control_port'],token,'TARGET',port=echo['port'])
    control(proxy['control_port'],token,'PROFILE',condition=profiles[0])
    source=session(proxy['data_port']);index=control(proxy['control_port'],token,'DRAIN')['next_index']
    try:
        for profile in profiles:
            expected=dict(es_to_ud=source.sent,ud_to_es=source.received)
            control(proxy['control_port'],token,'PROFILE',condition=profile,expected=expected)
            for kind,up,down in (('round_trip',32,32),('upload',PROBE_BYTES,4),('download',4,PROBE_BYTES)):
                for i in range(SAMPLES):
                    identifier=f'cal_{profile["id"]}_{kind}_{i}'
                    wall=probe(source,identifier,up,down,i)
                    snapshot=control(proxy['control_port'],token,'DRAIN',expected=dict(es_to_ud=source.sent,ud_to_es=source.received),since=index)
                    index=snapshot['next_index'];frames=snapshot['records']
                    require(len(frames)==2 and all(f['header']['request_id']==identifier and
                        f['condition_id']==profile['id'] and f['profile']==profile and
                        f['delayed_frames']==int(profile['rtt_ms']>0) for f in frames),'probe/relay frame/profile mismatch')
                    record=dict(condition_id=profile['id'],probe=kind,sample=i,wall_ms=wall,proxy_frames=frames,
                        model_inference_performed=False)
                    records.append(record);append(output/'calibration_probes.jsonl',record)
        source.send(header('STOP','shutdown',-1,shutdown_service=True));source.recv(kind='BYE',request_id='shutdown',layer_id=-1)
    finally:source.close()
    control(proxy['control_port'],token,'DRAIN',disconnected=True)
    from .r3a_evidence import jsonl
    source_frames=jsonl(output/'calibration_source_wire.jsonl');sink_frames=jsonl(output/'echo_wire.jsonl')
    key=lambda r:(r['header']['session_id'],r['header']['sequence'],r['operation'])
    sinks={(r['header']['session_id'],r['header']['sequence'],'send' if r['operation']=='recv' else 'recv'):r for r in sink_frames}
    require(len(source_frames)==len(sink_frames)==len(sinks),'probe endpoint frame counts differ')
    for frame in source_frames:
        peer=sinks[key(frame)]
        require(frame['complete'] and peer['complete'] and all(frame[k]==peer[k] for k in ('header','frame_sha256','wire_bytes')),
                'independent source/sink probe SHA/byte mismatch')
    proxy_rows=[f for r in records for f in r['proxy_frames']]
    sources={key(r):r for r in source_frames}
    for frame in proxy_rows:
        h=frame['header'];peer=sources[(h['session_id'],h['sequence'],'send' if frame['direction']=='es_to_ud' else 'recv')]
        require(frame['complete'] and frame['frame_sha256']==peer['frame_sha256'] and
            frame['wire_bytes_read']==frame['wire_bytes_forwarded']==peer['wire_bytes'],'probe source/relay byte mismatch')
    validity=validate_calibration(records,profiles)
    baseline_id=profiles[0]['id'];baseline=statistics.median(r['wall_ms'] for r in records if r['condition_id']==baseline_id and r['probe']=='round_trip')
    report=dict(execution_status='COMPLETE' if all(r['status']=='VALID' for r in validity) else 'FAILED',
        conditions=validity,raw_probes=records,direct_loopback_round_trip_ms=direct,
        proxy_zero_delay_overhead_ms=baseline-statistics.median(direct),proxy_pid=proxy['pid'],
        independent_source_relay_sink_sha_and_bytes_passed=True,
        probe_scope='CPU network calibration only; same persistent proxy process, separate synthetic session before GPU/model requests',
        model_inference_performed=False)
    write(output/'r3c_network_calibration.json',report)
    return report
