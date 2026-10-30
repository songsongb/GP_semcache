"""Actual loopback sockets; synthetic probe bytes, no model/GPU dependencies."""
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

from semcache.edgelora.communication import Channel,MAX_HEADER,MAGIC,PREFIX,digest,encode,header
from semcache.edgelora.r3c_network_proxy import FrameProxy,control,connect,loopback_listener,MAX_PAYLOAD
from semcache.edgelora.r3c_calibration import payload


class FragmentSocket:
    def __init__(self,sock,limit):self.sock=sock;self.limit=limit
    def __getattr__(self,name):return getattr(self.sock,name)
    def send(self,data):return self.sock.send(data[:self.limit])
    def recv(self,n):return self.sock.recv(min(n,self.limit))


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.proxy=FrameProxy('TEST_TOKEN',Path(self.tmp.name),frame_timeout=2.,idle_timeout=5.)
        self.info=self.proxy.start();self.addCleanup(self.proxy.close)
        self.session='TEST_FIXTURE_ONLY';self.adapter=digest('TEST_FIXTURE_ONLY')
        self.sink_records=[];self.sink_errors=[];self.threads=[];self.listeners=[]
        self.addCleanup(self.finish_sink)

    def finish_sink(self):
        for s in self.listeners:s.close()
        for t in self.threads:t.join(3.)
        self.assertFalse(any(t.is_alive() for t in self.threads),'probe sink did not exit')

    def sink(self,fragment=False,close=False):
        server=loopback_listener();server.settimeout(3.);self.listeners.append(server)
        def serve():
            channel=None
            try:
                sock,address=server.accept()
                if close:sock.close();return
                channel=Channel(FragmentSocket(sock,13) if fragment else sock,self.session,self.adapter,timeout=2.,trace=self.sink_records.append)
                while True:
                    h,data,_=channel.recv()
                    if h['kind']=='STOP':channel.send(header('BYE','shutdown',-1));return
                    channel.send(header('DELTAS',h['request_id'],0,'float32',h['shape']),data)
            except BaseException as exc:self.sink_errors.append(exc)
            finally:
                if channel is not None:channel.close()
                server.close()
        thread=threading.Thread(target=serve,daemon=True);self.threads.append(thread);thread.start()
        control(self.info['control_port'],'TEST_TOKEN','TARGET',port=server.getsockname()[1])

    def exchange(self,size,rtt=0,bandwidth=None,fragment=False):
        self.sink(fragment=fragment)
        up,down=bandwidth if isinstance(bandwidth,tuple) else (bandwidth,bandwidth)
        profile=dict(id='TEST',rtt_ms=rtt,es_to_ud_mbps=up,ud_to_es_mbps=down)
        control(self.info['control_port'],'TEST_TOKEN','PROFILE',condition=profile)
        sock=connect(self.info['data_port']);source=Channel(FragmentSocket(sock,7) if fragment else sock,self.session,self.adapter,timeout=5.)
        data=payload(size,1);start=time.monotonic()
        source.send(header('HIDDEN','test_0',0,'float32',[1,1,size//4]),data)
        _,reply,_=source.recv(kind='DELTAS',request_id='test_0',layer_id=0);wall=time.monotonic()-start
        self.assertEqual(reply,data)
        snapshot=control(self.info['control_port'],'TEST_TOKEN','DRAIN',expected=dict(es_to_ud=1,ud_to_es=1),since=0)
        frames=snapshot['records'];self.assertEqual(len(frames),2)
        source_frames={(r['header']['sequence'],r['operation']):r for r in source.records}
        for f in frames:
            s=source_frames[(f['header']['sequence'],'send' if f['direction']=='es_to_ud' else 'recv')]
            self.assertEqual(f['frame_sha256'],s['frame_sha256']);self.assertEqual(f['wire_bytes_forwarded'],s['wire_bytes'])
            self.assertEqual(f['wire_bytes_read'],s['wire_bytes'])
        source.send(header('STOP','shutdown',-1));source.recv(kind='BYE',request_id='shutdown',layer_id=-1);source.close()
        control(self.info['control_port'],'TEST_TOKEN','DRAIN',disconnected=True)
        self.assertFalse(self.sink_errors)
        return frames,wall

    def test_zero_unlimited_byte_identity_and_fragmented_reads_writes(self):
        frames,wall=self.exchange(2048,fragment=True)
        self.assertTrue(all(f['delayed_frames']==0 and f['pacing_wait_ms']==0 for f in frames))
        self.assertTrue(all(f['complete'] for f in frames))

    def test_delay_is_once_per_real_frame_and_actual_source_blocks(self):
        frames,wall=self.exchange(128*1024,rtt=20)
        self.assertGreaterEqual(wall,.018)
        self.assertEqual(sum(f['delayed_frames'] for f in frames),2)
        self.assertTrue(all(9.5<=f['one_way_delay_wait_ms']<40 for f in frames))
        self.assertTrue(all(f['chunks']>1 for f in frames))
        self.assertTrue(all(f['one_way_delay_wait_ms']==(f['one_way_delay_end_ns']-f['one_way_delay_start_ns'])/1e6 for f in frames))

    def test_delay_accounting_uses_deadline_origin_not_wait_helper_entry(self):
        original=self.proxy._wait
        def delayed_dispatch(until):
            time.sleep(.001);original(until);return 0.  # deliberately misleading helper-only accounting
        self.proxy._wait=delayed_dispatch
        frames,wall=self.exchange(2048,rtt=20)
        self.assertTrue(all(f['one_way_delay_wait_ms']>=10 for f in frames))
        self.assertTrue(all(f['serialization_start_ns']>=f['one_way_delay_end_ns'] for f in frames))

    def test_directional_bandwidth_and_combined_delay_shape_real_delivery(self):
        frames,wall=self.exchange(128*1024,rtt=10,bandwidth=20)
        self.assertGreaterEqual(wall,2*128*1024*8/20e6+.008)
        for frame in frames:
            self.assertLessEqual(frame['effective_mbps'],20.1)
            self.assertGreater(frame['pacing_wait_ms'],0.)
            self.assertEqual(frame['delayed_frames'],1)

    def test_large_payload_bounded_streaming_preserves_frame(self):
        frames,wall=self.exchange(1024*1024)
        self.assertTrue(all(f['chunks']>=64 for f in frames))
        self.assertLess(self.info['stream_buffer_bound_bytes_per_direction'],MAX_PAYLOAD)

    def test_asymmetric_directional_limits_have_separate_byte_clocks(self):
        frames,wall=self.exchange(128*1024,bandwidth=(20,10))
        self.assertGreaterEqual(wall,128*1024*8/20e6+128*1024*8/10e6-.004)
        for frame in frames:
            requested=20 if frame['direction']=='es_to_ud' else 10
            self.assertLessEqual(frame['effective_mbps'],requested*1.04)

    def test_source_closure_mid_frame_is_a_truncation_failure(self):
        self.sink();sock=connect(self.info['data_port']);sock.sendall(b'R3A');sock.close()
        self.assertTrue(self.proxy.stop_event.wait(2.))
        self.assertTrue(any('truncated' in error for error in self.proxy.errors))

    def test_corrupt_payload_rejected_without_hanging(self):
        self.sink();sock=connect(self.info['data_port'])
        h=header('HIDDEN','bad',0,'float32',[1,1,4]);h.update(session_id=self.session,adapter_sha256=self.adapter,sequence=0)
        packet,_=encode(h,payload(16,1));sock.sendall(packet[:-1]+bytes([packet[-1]^1]));sock.close()
        self.assertTrue(self.proxy.stop_event.wait(2.));self.assertTrue(self.proxy.errors)

    def test_truncated_or_oversized_prefix_fails_closed(self):
        self.sink();sock=connect(self.info['data_port'])
        sock.sendall(PREFIX.pack(MAGIC,MAX_HEADER+1,MAX_PAYLOAD+1));sock.close()
        self.assertTrue(self.proxy.stop_event.wait(2.));self.assertTrue(self.proxy.errors)

    def test_forced_timeout_and_worker_cleanup(self):
        self.proxy.frame_timeout=.1;self.sink();sock=connect(self.info['data_port']);sock.sendall(b'R')
        self.assertTrue(self.proxy.stop_event.wait(2.));sock.close()
        self.assertTrue(any('timeout' in error for error in self.proxy.errors))

    def test_destination_closure_propagates_and_no_listener_survives(self):
        self.sink(close=True);source=Channel(connect(self.info['data_port']),self.session,self.adapter,timeout=1.)
        with self.assertRaises((OSError,ValueError)):
            source.send(header('HIDDEN','closed',0,'float32',[1,1,1]),payload(4,0));source.recv()
        source.close();self.proxy.close()
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
            probe.settimeout(.2);self.assertNotEqual(probe.connect_ex(('127.0.0.1',self.info['data_port'])),0)


if __name__=='__main__':unittest.main()
