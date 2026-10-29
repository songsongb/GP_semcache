"""R3-A bounded, non-pickle loopback TCP framing; tensor payloads are raw LE bytes.

Wire counts include application prefix/JSON metadata, not kernel TCP/IP headers
or retransmissions. Each frame has a single absolute I/O deadline.
"""
import hashlib
import json
import math
import socket
import struct
import time

VERSION = 1
MAGIC = b'R3ATCP01'
PREFIX = struct.Struct('!8sIQ')
MAX_HEADER = 65536
MAX_PAYLOAD = 64*1024*1024
WIDTHS = {'float16': 2, 'float32': 4, 'none': 0}
KINDS = {'HELLO', 'READY', 'HIDDEN', 'DELTAS', 'STOP', 'BYE'}


class ProtocolError(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise ProtocolError(reason)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return digest_bytes(canonical(value))


def _object(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, 'duplicate JSON field')
        result[k] = v
    return result


def json_loads(data):
    def invalid(v):
        raise ProtocolError('nonfinite JSON: '+v)
    value = json.loads(data, object_pairs_hook=_object, parse_constant=invalid)
    def finite(x):
        if isinstance(x, float):
            require(math.isfinite(x), 'nonfinite JSON number')
        elif isinstance(x, dict):
            for v in x.values(): finite(v)
        elif isinstance(x, list):
            for v in x: finite(v)
    finite(value)
    return value


def payload_size(dtype, shape):
    require(dtype in WIDTHS and isinstance(shape, list) and len(shape) <= 4, 'invalid dtype/shape')
    if dtype == 'none':
        require(shape == [], 'control frame shape must be empty')
        return 0
    require(1 <= len(shape) <= 4, 'tensor rank required')
    count = WIDTHS[dtype]
    for dim in shape:
        require(type(dim) is int and 0 < dim <= 1_000_000, 'invalid tensor dimension')
        count *= dim
        require(count <= MAX_PAYLOAD, 'oversized tensor shape')
    return count


def header(kind, request_id, layer_id, dtype='none', shape=None, **metadata):
    return dict(version=VERSION, kind=kind, request_id=request_id, layer_id=layer_id,
                dtype=dtype, shape=[] if shape is None else list(shape), metadata=metadata)


def validate(h, payload_length):
    require(isinstance(h, dict), 'header must be an object')
    require(set(h) == {'version','kind','request_id','layer_id','dtype','shape','metadata',
                      'session_id','adapter_sha256','sequence','payload_bytes','payload_sha256'}, 'header fields mismatch')
    require(type(h['version']) is int and h['version'] == VERSION, 'protocol version mismatch')
    require(h['kind'] in KINDS, 'unknown message kind')
    for key in ('request_id', 'session_id'):
        require(isinstance(h[key], str) and 1 <= len(h[key]) <= 128 and h[key].isascii(), 'invalid '+key)
    for key in ('adapter_sha256', 'payload_sha256'):
        require(isinstance(h[key], str) and len(h[key]) == 64 and all(c in '0123456789abcdef' for c in h[key]), 'invalid '+key)
    require(type(h['layer_id']) is int and -1 <= h['layer_id'] < 256, 'invalid layer ID')
    require(type(h['sequence']) is int and 0 <= h['sequence'] < 1_000_000, 'invalid sequence')
    require(isinstance(h['metadata'], dict), 'metadata object required')
    require(type(h['payload_bytes']) is int and h['payload_bytes'] == payload_length == payload_size(h['dtype'], h['shape']),
            'dtype/shape/payload length mismatch')
    if h['kind'] in ('HIDDEN', 'DELTAS'):
        require(h['layer_id'] >= 0 and h['dtype'] != 'none', 'data frame needs layer/tensor')
    else:
        require(h['layer_id'] == -1 and h['dtype'] == 'none' and payload_length == 0, 'invalid control frame')


def encode(h, payload):
    require(isinstance(payload, bytes) and len(payload) <= MAX_PAYLOAD, 'invalid payload bytes')
    h = dict(h, payload_bytes=len(payload), payload_sha256=digest_bytes(payload))
    validate(h, len(payload))
    raw = canonical(h)
    require(0 < len(raw) <= MAX_HEADER, 'oversized header')
    return PREFIX.pack(MAGIC, len(raw), len(payload))+raw+payload, h


def listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', 0))
        sock.listen(1)
        sock.settimeout(300.)  # bounded cold model-load wait, outside request clock
        return sock
    except BaseException:
        sock.close()
        raise


def connect(port, timeout=30.):
    require(type(port) is int and 0 < port < 65536, 'invalid loopback port')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(('127.0.0.1', port))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock
    except BaseException:
        sock.close()
        raise


class Channel:
    def __init__(self, sock, session_id, adapter_sha256, *, timeout=30., trace=None):
        require(isinstance(timeout, (int,float)) and not isinstance(timeout,bool) and math.isfinite(timeout) and timeout > 0,
                'positive finite timeout required')
        peer = sock.getpeername()
        require(peer[0] == '127.0.0.1' and sock.family == socket.AF_INET, 'only IPv4 loopback TCP permitted')
        self.sock, self.session, self.adapter = sock, session_id, adapter_sha256
        self.timeout, self.trace = timeout, trace
        self.sent, self.received = 0, 0
        self.records = []
        self.failed = False

    def _remaining(self, deadline):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('frame deadline exceeded')
        self.sock.settimeout(remaining)

    def _record(self, operation, h, start, wire, error=None, frame_sha256=None, **timings):
        record = dict(operation=operation, pid=__import__('os').getpid(),
            monotonic_start_ns=start, monotonic_end_ns=time.monotonic_ns(),
            wire_bytes=wire, header_bytes=wire-h.get('payload_bytes',0) if not error else None,
            frame_sha256=frame_sha256, header=h, complete=error is None, error=error)
        record.update(timings)
        self.records.append(record)
        if self.trace is not None: self.trace(record)
        return record

    def send(self, h, payload=b''):
        require(not self.failed, 'channel already failed closed')
        start, sent = time.monotonic_ns(), 0
        h = dict(h, session_id=self.session, adapter_sha256=self.adapter, sequence=self.sent)
        try:
            packet, h = encode(h, payload)
            encoded = time.monotonic_ns()
            deadline = time.monotonic()+self.timeout
            io_start = time.monotonic_ns()
            while sent < len(packet):
                self._remaining(deadline)
                n = self.sock.send(memoryview(packet)[sent:])
                require(n > 0, 'socket closed during send')
                sent += n
            self.sent += 1
            io_end = time.monotonic_ns()
            return self._record('send', h, start, sent, frame_sha256=digest_bytes(packet),
                framing_serialization_ms=(encoded-start)/1e6, socket_call_wall_ms=(io_end-io_start)/1e6)
        except BaseException as exc:
            self._record('send', h, start, sent, error=f'{type(exc).__name__}: {exc}')
            self.failed = True; self.sock.close()
            raise

    def recv(self, **expected):
        require(not self.failed, 'channel already failed closed')
        start, received, h, chunks = time.monotonic_ns(), 0, {}, []
        io_ns = 0
        deadline = time.monotonic()+self.timeout
        def exact(size):
            nonlocal received,io_ns
            result = bytearray()
            while len(result) < size:
                self._remaining(deadline)
                io_start = time.monotonic_ns()
                data = self.sock.recv(min(size-len(result), 1_048_576))
                io_ns += time.monotonic_ns()-io_start
                require(bool(data), 'unexpected close / truncated frame')
                result.extend(data)
                received += len(data)
            blob = bytes(result)
            chunks.append(blob)
            return blob
        try:
            magic, head_size, size = PREFIX.unpack(exact(PREFIX.size))
            require(magic == MAGIC and 0 < head_size <= MAX_HEADER and size <= MAX_PAYLOAD, 'invalid magic or oversized frame')
            raw_header = exact(head_size)
            decode_start = time.monotonic_ns()
            h = json_loads(raw_header)
            validate(h, size)
            decode_ns = time.monotonic_ns()-decode_start
            require(h['session_id'] == self.session and h['adapter_sha256'] == self.adapter, 'session/adapter identity mismatch')
            require(h['sequence'] == self.received, 'out-of-order frame sequence')
            for key, value in expected.items():
                require(h.get(key) == value, 'unexpected '+key)
            payload = exact(size)
            require(digest_bytes(payload) == h['payload_sha256'], 'payload SHA256 mismatch')
            self.received += 1
            record = self._record('recv', h, start, received, frame_sha256=digest_bytes(b''.join(chunks)),
                framing_deserialization_ms=decode_ns/1e6, socket_call_wall_ms=io_ns/1e6)
            return h, payload, record
        except BaseException as exc:
            self._record('recv', h, start, received, error=f'{type(exc).__name__}: {exc}')
            self.failed = True; self.sock.close()
            raise

    def close(self):
        self.sock.close()
