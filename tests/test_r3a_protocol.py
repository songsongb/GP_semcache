"""CPU tests using real IPv4 loopback TCP; no socketpair or tensor libraries."""
import socket
import threading
import time
import unittest

from semcache.edgelora.communication import (Channel,MAGIC,MAX_HEADER,MAX_PAYLOAD,PREFIX,ProtocolError,
    canonical,connect,digest_bytes,encode,header,listener,payload_size)
from semcache.edgelora.r3a_artifacts import pack,unpack


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        server=listener();self.addCleanup(server.close)
        client=connect(server.getsockname()[1]);self.addCleanup(client.close)
        accepted,_=server.accept();self.addCleanup(accepted.close)
        self.a=Channel(client,'session','a'*64,timeout=.1)
        self.b=Channel(accepted,'session','a'*64,timeout=.1)

    def raw(self,**changes):
        h=dict(header('HIDDEN','request',0,'float32',[1,1,2]),session_id='session',adapter_sha256='a'*64,sequence=0,
               payload_bytes=8,payload_sha256=digest_bytes(pack([1.,2.],'float32')))
        h.update(changes)
        raw=canonical(h);return PREFIX.pack(MAGIC,len(raw),8)+raw+pack([1.,2.],'float32')

    def test_roundtrip_version_shapes_digest_and_wire_accounting(self):
        sent=self.a.send(header('HIDDEN','req',0,'float32',[1,1,2]),pack([1,2],'float32'))
        h,p,r=self.b.recv(kind='HIDDEN',request_id='req',layer_id=0,dtype='float32',shape=[1,1,2])
        self.assertEqual(unpack(p,h['dtype']),[1.,2.])
        self.assertEqual(sent['wire_bytes'],r['wire_bytes'])
        self.assertEqual(sent['frame_sha256'],r['frame_sha256'])
        self.assertEqual(sent['wire_bytes'],sent['header_bytes']+8)
        self.assertGreater(sent['header_bytes'],PREFIX.size)
        self.b.send(header('DELTAS','req',0,'float32',[3,1,1,2]),pack(range(6),'float32'))
        self.assertEqual(len(self.a.recv(kind='DELTAS')[1]),24)

    def test_version_adapter_session_and_request_layer_mismatch(self):
        for field,value in (('version',2),('adapter_sha256','b'*64),('session_id','stale'),('request_id','wrong'),('layer_id',1)):
            with self.subTest(field=field):
                # Each violation permanently closes its own stream.
                server=listener()
                try:
                    client=connect(server.getsockname()[1]);accepted,_=server.accept()
                    channel=Channel(accepted,'session','a'*64,timeout=.1)
                    try:
                        client.sendall(self.raw(**{field:value}))
                        with self.assertRaises(ProtocolError): channel.recv(request_id='request',layer_id=0)
                        self.assertTrue(channel.failed)
                        with self.assertRaises(ProtocolError): channel.recv()
                    finally: client.close();channel.close()
                finally: server.close()

    def test_digest_corruption_is_rejected(self):
        packet=self.raw();self.a.sock.sendall(packet[:-1]+bytes([packet[-1]^1]))
        with self.assertRaisesRegex(ProtocolError,'SHA256'): self.b.recv()

    def test_truncated_prefix_and_payload_close_without_hang(self):
        for size in (5,len(self.raw())-1):
            with self.subTest(size=size):
                server=listener();client=connect(server.getsockname()[1]);accepted,_=server.accept()
                channel=Channel(accepted,'session','a'*64,timeout=.1)
                try:
                    client.sendall(self.raw()[:size]);client.shutdown(socket.SHUT_WR)
                    with self.assertRaisesRegex(ProtocolError,'truncated'): channel.recv()
                finally: server.close();client.close();channel.close()

    def test_oversized_prefix_is_rejected_before_body_allocation(self):
        self.a.sock.sendall(PREFIX.pack(MAGIC,1,MAX_PAYLOAD+1))
        with self.assertRaisesRegex(ProtocolError,'oversized'): self.b.recv()

    def test_dtype_shape_mismatch_and_invalid_shapes(self):
        self.a.sock.sendall(self.raw(dtype='float16'))
        with self.assertRaisesRegex(ProtocolError,'length mismatch'): self.b.recv()
        for shape in ([True],[-1],[0],[1,2,3,4,5],[1000000,1000000]):
            with self.assertRaises(ProtocolError): payload_size('float32',shape)
        with self.assertRaises(ProtocolError): payload_size('pickle',[1])

    def test_receiver_expected_shape_mismatch(self):
        self.a.sock.sendall(self.raw())
        with self.assertRaisesRegex(ProtocolError,'shape'): self.b.recv(shape=[1,2,1])

    def test_duplicate_sequence_and_unknown_kind(self):
        packet=self.raw();self.a.sock.sendall(packet+packet)
        self.b.recv()
        with self.assertRaisesRegex(ProtocolError,'sequence'): self.b.recv()
        h=dict(header('PICKLE','req',0),session_id='x',adapter_sha256='a'*64,sequence=0)
        with self.assertRaises(ProtocolError): encode(h,b'')

    def test_timeout_and_absolute_deadline_slow_prefix(self):
        started=time.monotonic()
        with self.assertRaises(TimeoutError): self.b.recv()
        self.assertLess(time.monotonic()-started,.5)
        server=listener();client=connect(server.getsockname()[1]);accepted,_=server.accept()
        channel=Channel(accepted,'session','a'*64,timeout=.05)
        try:
            def slow():
                try:
                    for byte in self.raw()[:20]: client.send(bytes([byte]));time.sleep(.02)
                except OSError: pass
            thread=threading.Thread(target=slow,daemon=True);thread.start()
            started=time.monotonic()
            with self.assertRaises(TimeoutError): channel.recv()
            self.assertLess(time.monotonic()-started,.2)
            thread.join(timeout=.5)
        finally: server.close();client.close();channel.close()

    def test_nonfinite_raw_values_and_json_and_header_limits(self):
        import struct
        with self.assertRaises(ProtocolError): unpack(struct.pack('<f',float('nan')),'float32')
        h=dict(header('HELLO','hello',-1,huge='x'*MAX_HEADER),session_id='s',adapter_sha256='a'*64,sequence=0)
        with self.assertRaises(ProtocolError): encode(h,b'')
        h=dict(header('HELLO','hello',-1,bad=float('nan')),session_id='s',adapter_sha256='a'*64,sequence=0)
        with self.assertRaises(ValueError): encode(h,b'')


if __name__=='__main__': unittest.main()
