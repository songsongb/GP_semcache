"""Transparent, bounded application-frame TCP impairment on IPv4 loopback.

One latency wait/frame, independently paced byte clocks/direction.
Original prefix/header/payload bytes are forwarded, never re-encoded.
"""
import hashlib
import math
import os
import socket
import threading
import time

from .communication import MAGIC,PREFIX,MAX_HEADER,json_loads,validate,require
from .r3a_artifacts import append,write

CHUNK=16*1024
MAX_PAYLOAD=2*1024*1024
CONTROL_LIMIT=1024*1024
MAX_FRAMES=40000


def condition(value):
    require(set(value)=={'id','rtt_ms','es_to_ud_mbps','ud_to_es_mbps'},'invalid network profile fields')
    require(isinstance(value['id'],str) and 1<=len(value['id'])<=32 and value['id'].isascii(),'invalid profile ID')
    rtt=value['rtt_ms']
    require(type(rtt) in (int,float) and math.isfinite(rtt) and 0<=rtt<=100,'RTT must be finite 0..100 ms')
    for key in ('es_to_ud_mbps','ud_to_es_mbps'):
        b=value[key]
        require(b is None or type(b) in (int,float) and math.isfinite(b) and 10<=b<=10000,
                'bandwidth is unlimited/null or finite 10..10000 decimal Mbps')
    return dict(value)


def loopback_listener():
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1',0));sock.listen(2);sock.settimeout(.2);return sock
    except BaseException:sock.close();raise


def connect(port,timeout=10.):
    require(type(port) is int and 0<port<65536,'invalid local port')
    sock=socket.create_connection(('127.0.0.1',port),timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);return sock


def control(control_port,token,op,**values):
    import json
    with connect(control_port) as sock:
        packet=json.dumps(dict(token=token,op=op,**values),allow_nan=False).encode()+b'\n'
        require(len(packet)<16384,'oversized control request');sock.sendall(packet)
        data=bytearray()
        while not data.endswith(b'\n'):
            block=sock.recv(65536);require(block,'proxy control unexpectedly closed');data.extend(block)
            require(len(data)<=CONTROL_LIMIT,'oversized control response')
    reply=json_loads(data);require(reply['ok'],reply.get('reason','proxy control failed'))
    return reply['result']


class FrameProxy:
    def __init__(self,token,output,*,frame_timeout=20.,idle_timeout=120.):
        require(frame_timeout>0 and idle_timeout>0,'positive proxy deadlines required')
        self.token=token;self.output=output;self.frame_timeout=frame_timeout;self.idle_timeout=idle_timeout
        self.lock=threading.RLock();self.stop_event=threading.Event();self.active=0;self.connected=False
        self.target=None;self.profile=condition(dict(id='INITIAL',rtt_ms=0,es_to_ud_mbps=None,ud_to_es_mbps=None))
        self.counts=dict(es_to_ud=0,ud_to_es=0);self.records=[];self.errors=[];self.sockets=[];self.threads=[]
        self.data=loopback_listener();self.commands=loopback_listener()
        self.ready=dict(pid=os.getpid(),data_port=self.data.getsockname()[1],control_port=self.commands.getsockname()[1],
            chunk_bytes=CHUNK,max_payload_bytes=MAX_PAYLOAD,stream_buffer_bound_bytes_per_direction=CHUNK+MAX_HEADER+PREFIX.size,
            policy='one delay/frame then paced virtual byte clock; initially zero credit, at most one chunk catch-up credit/frame; independent full-duplex clocks')

    def start(self):
        for name,target in (('accept',self._accept),('control',self._control)):
            thread=threading.Thread(target=target,name='r3c_'+name,daemon=True);self.threads.append(thread);thread.start()
        return self.ready

    def _fatal(self,exc):
        with self.lock:self.errors.append(f'{type(exc).__name__}: {exc}');self.stop_event.set()
        for sock in list(self.sockets):
            try:sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass

    def _wait(self,until):
        start=time.monotonic()
        while time.monotonic()<until:
            require(not self.stop_event.is_set(),'proxy stopped during delivery')
            self.stop_event.wait(min(.02,until-time.monotonic()))
        return (time.monotonic()-start)*1000

    def _exact(self,sock,size,deadline,*,first=False):
        values=bytearray()
        while len(values)<size:
            require(not self.stop_event.is_set(),'proxy stopped')
            remaining=deadline-time.monotonic();require(remaining>0,'proxy frame timeout')
            sock.settimeout(min(.2,remaining))
            try:block=sock.recv(min(CHUNK,size-len(values)))
            except socket.timeout:continue
            if not block and first and not values:return None
            require(block,'proxy truncated application frame')
            values.extend(block)
        return bytes(values)

    def _send(self,sock,data,deadline):
        sent=0
        while sent<len(data):
            require(time.monotonic()<deadline and not self.stop_event.is_set(),'proxy send timeout/stopped')
            sock.settimeout(min(.2,max(.001,deadline-time.monotonic())))
            try:n=sock.send(memoryview(data)[sent:])
            except socket.timeout:continue
            require(n>0,'proxy destination closed');sent+=n
        return sent

    def _pump(self,source,destination,direction):
        try:
            while not self.stop_event.is_set():
                # Idle is not an active frame; profiles can change here only
                # after source Channel counts and relay completed counts agree.
                first=self._exact(source,1,time.monotonic()+self.idle_timeout,first=True)
                if first is None:
                    try:destination.shutdown(socket.SHUT_WR)
                    except OSError:pass
                    return
                start=time.monotonic_ns();deadline=time.monotonic()+self.frame_timeout
                with self.lock:
                    require(len(self.records)+self.active<MAX_FRAMES,'bounded proxy frame ledger exhausted')
                    self.active+=1;profile=dict(self.profile)
                record=dict(direction=direction,condition_id=profile['id'],profile=profile,pid=os.getpid(),
                    thread_id=threading.get_ident(),frame_start_ns=start,wire_bytes_read=1,wire_bytes_forwarded=0,
                    delayed_frames=0,pacing_wait_ms=0.,one_way_delay_wait_ms=0.,chunks=0,complete=False)
                try:
                    prefix=first+self._exact(source,PREFIX.size-1,deadline)
                    magic,header_size,payload_size=PREFIX.unpack(prefix)
                    require(magic==MAGIC and 0<header_size<=MAX_HEADER and payload_size<=MAX_PAYLOAD,
                            'BLOCKED_NETWORK_EMULATION: unsupported prefix/header/payload size')
                    encoded=self._exact(source,header_size,deadline);h=json_loads(encoded);validate(h,payload_size)
                    record.update(header=h,wire_bytes_read=PREFIX.size+header_size,header_ready_ns=time.monotonic_ns())
                    frame_hash=hashlib.sha256();payload_hash=hashlib.sha256()
                    delay=profile['rtt_ms']/2000.
                    # Measure from the SAME origin used for the deadline. The
                    # helper's own entry is later and can underreport an exactly
                    # realized delay by dispatch time; never use that shorter
                    # interval to classify the delivered frame.
                    record['one_way_delay_start_ns']=time.monotonic_ns()
                    if delay:self._wait(record['one_way_delay_start_ns']/1e9+delay)
                    record['one_way_delay_end_ns']=time.monotonic_ns()
                    record['one_way_delay_wait_ms']=(record['one_way_delay_end_ns']-record['one_way_delay_start_ns'])/1e6 if delay else 0.
                    record['delayed_frames']=int(delay>0);record['serialization_start_ns']=time.monotonic_ns()
                    rate=profile[direction+'_mbps'];bytes_per_second=None if rate is None else rate*1e6/8
                    # Bound timer/read/hash overshoot credit to one chunk. Unlike
                    # restarting each chunk's clock at now, this does not charge
                    # scheduler overshoot repeatedly as extra link serialization.
                    # Start with no saved idle credit on every frame.
                    clock=time.monotonic()
                    record.update(application_queue_wait_ms=0.,max_pacing_credit_bytes=CHUNK,
                        upstream_socket_queue_time='NOT_MEASURED; ordered bounded streaming, no application backlog queue')
                    def forward(block):
                        nonlocal clock
                        for offset in range(0,len(block),CHUNK):
                            chunk=block[offset:offset+CHUNK]
                            if bytes_per_second:
                                clock=max(clock,time.monotonic()-CHUNK/bytes_per_second)+len(chunk)/bytes_per_second
                                record['pacing_wait_ms']+=self._wait(clock)
                            if 'first_write_ns' not in record:record['first_write_ns']=time.monotonic_ns()
                            record['wire_bytes_forwarded']+=self._send(destination,chunk,deadline)
                            record['last_write_ns']=time.monotonic_ns();record['chunks']+=1;frame_hash.update(chunk)
                    forward(prefix+encoded)
                    remaining=payload_size
                    while remaining:
                        block=self._exact(source,min(CHUNK,remaining),deadline)
                        record['wire_bytes_read']+=len(block);payload_hash.update(block);forward(block);remaining-=len(block)
                    require(payload_hash.hexdigest()==h['payload_sha256'],'proxy payload checksum mismatch')
                    record.update(complete=True,frame_sha256=frame_hash.hexdigest(),payload_sha256=payload_hash.hexdigest(),
                        serialization_ms=(record['last_write_ns']-record['serialization_start_ns'])/1e6,
                        kernel_queue_time='NOT_MEASURED',frame_end_ns=time.monotonic_ns())
                    record['effective_mbps']=record['wire_bytes_forwarded']*.008/max(record['serialization_ms'],1e-9)
                except BaseException as exc:
                    record.update(error=f'{type(exc).__name__}: {exc}',frame_end_ns=time.monotonic_ns())
                    self._fatal(exc);raise
                finally:
                    with self.lock:
                        record['index']=len(self.records);self.records.append(record)
                        append(self.output/'proxy_frames.jsonl',record)
                        if record['complete']:self.counts[direction]+=1
                        self.active-=1
        except BaseException as exc:
            if not self.stop_event.is_set():self._fatal(exc)

    def _accept(self):
        try:
            while not self.stop_event.is_set():
                try:client,address=self.data.accept()
                except socket.timeout:continue
                require(address[0]=='127.0.0.1','non-loopback proxy client')
                with self.lock:
                    require(not self.connected and self.target is not None,'proxy connection already active or no target')
                    self.connected=True;self.counts=dict(es_to_ud=0,ud_to_es=0);target=self.target
                try:upstream=connect(target)
                except BaseException:
                    client.close()
                    with self.lock:self.connected=False
                    raise
                for sock in (client,upstream):
                    sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                    sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,65536);sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,65536)
                with self.lock:self.sockets=[client,upstream]
                workers=[]
                for source,dest,direction in ((client,upstream,'es_to_ud'),(upstream,client,'ud_to_es')):
                    t=threading.Thread(target=self._pump,args=(source,dest,direction),daemon=True)
                    workers.append(t);self.threads.append(t);t.start()
                for t in workers:
                    while t.is_alive() and not self.stop_event.is_set():t.join(.1)
                for sock in (client,upstream):sock.close()
                for t in workers:t.join(2.)
                require(not any(t.is_alive() for t in workers),'proxy forwarding worker did not exit')
                with self.lock:self.sockets=[];self.connected=False
        except BaseException as exc:
            if not self.stop_event.is_set():self._fatal(exc)

    def _snapshot(self,expected=None,since=None,disconnected=False):
        deadline=time.monotonic()+10.
        while time.monotonic()<deadline:
            with self.lock:
                require(not self.errors,'proxy fatal: '+str(self.errors))
                require(not self.stop_event.is_set(),'proxy stopped while waiting for boundary')
                if self.active==0 and (not disconnected or not self.connected) and (expected is None or self.counts==expected):
                    rows=[] if since is None else self.records[since:]
                    require(len(rows)<=256,'too many frames in one boundary snapshot')
                    return dict(counts=dict(self.counts),records=list(rows),next_index=len(self.records),connected=self.connected)
            time.sleep(.002)
        raise TimeoutError('proxy drain/count checkpoint not reached')

    def _command(self,message):
        require(message.get('token')==self.token,'proxy control token mismatch')
        op=message.get('op')
        if op=='DRAIN':return self._snapshot(message.get('expected'),message.get('since'),message.get('disconnected',False))
        if op=='PROFILE':
            snapshot=self._snapshot(message.get('expected'))
            with self.lock:self.profile=condition(message['condition'])
            return snapshot
        if op=='TARGET':
            snapshot=self._snapshot(disconnected=True)
            require(type(message['port']) is int and 0<message['port']<65536,'invalid target port')
            with self.lock:self.target=message['port'];self.counts=dict(es_to_ud=0,ud_to_es=0)
            return snapshot
        if op=='STOP':
            result=self._snapshot(disconnected=True);self.stop_event.set();return result
        raise ValueError('unknown proxy control operation')

    def _control(self):
        import json
        while not self.stop_event.is_set():
            try:connection,address=self.commands.accept()
            except socket.timeout:continue
            except OSError:return
            with connection:
                connection.settimeout(10.);data=bytearray()
                try:
                    require(address[0]=='127.0.0.1','non-local control connection')
                    while not data.endswith(b'\n'):
                        block=connection.recv(4096);require(block,'truncated control');data.extend(block)
                        require(len(data)<=16384,'oversized control')
                    result=self._command(json_loads(data));reply=dict(ok=True,result=result)
                except BaseException as exc:reply=dict(ok=False,reason=f'{type(exc).__name__}: {exc}')
                try:connection.sendall(json.dumps(reply,allow_nan=False).encode()+b'\n')
                except OSError:pass

    def close(self):
        self.stop_event.set()
        for sock in list(self.sockets):
            try:sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            sock.close()
        self.data.close();self.commands.close()
        for thread in self.threads:thread.join(3.)
        alive=[t.name for t in self.threads if t.is_alive()]
        report=dict(pid=os.getpid(),errors=self.errors,records=len(self.records),threads_stopped=not alive,alive_threads=alive)
        write(self.output/'proxy_lifecycle.json',report)
        require(not alive,'proxy thread cleanup failed')
        return report
