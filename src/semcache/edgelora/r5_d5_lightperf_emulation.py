"""Unprivileged application byte pacing and observable UD service emulation."""
import math
import time

from .communication import require
from .r5_d5_channel import Channel as StrictChannel
from .r5_d5_lightperf_workload import knobs


class BytePacer:
    """Single owner, per-channel sender. No busy spin or frame-sized sleep.

    Reserve each bounded chunk before send, refund unsent bytes afterwards.
    Deadline is inherited from the frozen transport's whole-frame deadline.
    """
    def __init__(self, bandwidth_mbps=None, *, clock=time.monotonic, sleep=time.sleep):
        knobs(bandwidth_mbps)
        self.rate = None if bandwidth_mbps is None else bandwidth_mbps * 1e6 / 8
        self.clock, self.sleep = clock, sleep
        self.next = clock()
        self.bytes, self.pacing_seconds = 0, 0.
        self.first = None
        self.last = None

    def reserve(self, size, deadline):
        require(type(size) is int and 0 <= size <= 65536, 'invalid pacing chunk')
        now = self.clock()
        require(math.isfinite(deadline) and now < deadline, 'pacing deadline exceeded')
        if not size or self.rate is None:
            return
        target = max(self.next, now) + size / self.rate
        if target >= deadline:
            raise TimeoutError('bandwidth service exceeds original frame deadline')
        started = now
        # Tolerate early wakeups without spinning; re-sleep to monotonic target.
        while now < target:
            self.sleep(target - now)
            observed = self.clock()
            require(observed > now, 'pacing clock did not advance')
            now = observed
            if now >= deadline:
                raise TimeoutError('bandwidth pacing exceeded frame deadline')
        self.next = target
        self.pacing_seconds += now - started

    def commit(self, requested, actual, started):
        require(0 < actual <= requested <= 65536, 'invalid socket progress')
        if self.rate is not None:
            self.next -= (requested - actual) / self.rate
        self.bytes += actual
        self.first = started if self.first is None else self.first
        self.last = self.clock()

    def report(self):
        span = None if self.first is None else self.last - self.first
        return dict(actual_wire_bytes=self.bytes, added_pacing_ms=self.pacing_seconds * 1000,
                    requested_bandwidth_mbps=None if self.rate is None else self.rate * 8 / 1e6,
                    active_span_ms=None if span is None else span * 1000,
                    achieved_mbps=None if not span or not self.bytes else self.bytes * 8 / span / 1e6,
                    achieved_scope='first send start to last send end, including inter-frame compute/wait gaps',
                    scope='PER_SENDING_CHANNEL; serialized frame bytes; not TCP/IP packet overhead')


class PacedSocket:
    """Proxy only send; peer's sending proxy shapes the reverse direction."""
    def __init__(self, sock, pacer):
        self.raw, self.pacer = sock, pacer
        self.deadline = None

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def settimeout(self, seconds):
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
                'finite positive frame timeout required')
        self.deadline = self.pacer.clock() + seconds
        self.raw.settimeout(seconds)

    def send(self, data):
        size = min(len(data), 65536)
        if not size:
            return 0
        require(self.deadline is not None, 'frame deadline missing')
        started = self.pacer.clock()
        self.pacer.reserve(size, self.deadline)
        remaining = self.deadline - self.pacer.clock()
        if remaining <= 0:
            raise TimeoutError('paced send deadline expired')
        self.raw.settimeout(remaining)
        actual = self.raw.send(data[:size])
        self.pacer.commit(size, actual, started)
        return actual


class LightChannel(StrictChannel):
    def __init__(self, sock, session_id, adapter_sha256, *, bandwidth_mbps=None,
                 timeout=30., trace=None):
        self.pacer = BytePacer(bandwidth_mbps)
        self.summary = {op: dict(frames=0, wire_bytes=0, payload_bytes=0, frame_overhead_bytes=0,
                                socket_call_wall_ms=0.) for op in ('send', 'recv')}
        super().__init__(PacedSocket(sock, self.pacer), session_id, adapter_sha256,
                         timeout=timeout, trace=trace)

    def _record(self, operation, h, start, wire, error=None, **kwargs):
        try:
            if operation == 'send':
                kwargs['added_pacing_ms'] = (self.pacer.pacing_seconds - self._send_pacing_before) * 1000
            record = super()._record(operation, h, start, wire, error=error, **kwargs)
            if record is not None and error is None:
                s = self.summary[operation]
                s['frames'] += 1
                s['wire_bytes'] += wire
                s['payload_bytes'] += h['payload_bytes']
                s['frame_overhead_bytes'] += wire - h['payload_bytes']
                s['socket_call_wall_ms'] += record.get('socket_call_wall_ms', 0.)
            return record
        finally:
            # SparseSession retains returned records for its current pass only.
            # Channel must not retain the entire session's verbose ledger.
            self.records.clear()

    def send(self, h, payload=b''):
        if hasattr(self, 'condition_deadline'):
            self.remaining_timeout(self.condition_deadline)
        self._send_pacing_before = self.pacer.pacing_seconds
        return super().send(h, payload)

    def remaining_timeout(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self.failed = True
            primary = TimeoutError('condition deadline exceeded before wire operation')
            try:
                self.sock.close()
            except BaseException as exc:
                if hasattr(primary, 'add_note'):
                    primary.add_note('deadline socket close: ' + repr(exc))
            raise primary
        self.timeout = min(self.timeout, remaining)

    def recv(self, **expected):
        if hasattr(self, 'condition_deadline'):
            self.remaining_timeout(self.condition_deadline)
        return super().recv(**expected)


class UDServiceEmulator:
    """EMULATED_UD_SERVICE_TIME, not a hardware CPU throughput controller.

    Measure the exact supplied LoRA operation on every call, perform it once,
    then target measured_cost/factor. Scheduler oversleep and timer resolution
    remain observable. Serialization, authorization, ES and TCP are outside.
    """
    def __init__(self, factor=1., *, clock=time.monotonic, sleep=time.sleep):
        knobs(None, factor)
        self.factor, self.clock, self.sleep = factor, clock, sleep
        self.compute, self.service, self.pacing, self.operations = 0., 0., 0., 0

    def run_lora(self, compute, *args, deadline, **kwargs):
        start = self.clock()
        require(math.isfinite(deadline) and start < deadline, 'UD deadline exceeded')
        result = compute(*args, **kwargs)
        computed = self.clock()
        raw = computed - start
        require(raw >= 0, 'nonmonotonic UD compute clock')
        target = start + raw / self.factor
        if max(target, computed) >= deadline:
            raise TimeoutError('UD emulated service exceeds deadline')
        now = computed
        while now < target:
            self.sleep(target - now)
            observed = self.clock()
            require(observed > now, 'UD pacing clock did not advance')
            now = observed
            if now >= deadline:
                raise TimeoutError('UD service deadline expired')
        self.compute += raw
        self.pacing += now - computed
        self.service += now - start
        self.operations += 1
        return result

    def report(self):
        return dict(scope='EMULATED_UD_SERVICE_TIME', requested_factor=self.factor,
                    calibration='MEASURED_EACH_EXACT_LORA_OPERATION_WALL_COST', operations=self.operations,
                    measured_compute_ms=self.compute * 1000, measured_service_ms=self.service * 1000,
                    pacing_ms=self.pacing * 1000,
                    achieved_service_factor=None if self.service <= 0 else self.compute / self.service,
                    hardware_equivalence='NOT_ESTABLISHED')


def serve_light_sparse(channel, identity, registry, matrices, *, service, deadline,
                       fixture=False, trace=lambda record: None, expected_requests=None):
    """Separate UD service, same SparseOrder, arithmetic and protocol checks.

    Derived from the frozen serve_sparse loop. Only LoRA computation receives
    measured service pacing. No raw evidence and no module monkeypatching.
    Caller owns CPU isolation, matrices/source checks and process deadline.
    """
    import sys
    from .communication import header, digest, digest_bytes
    from .r3a_artifacts import pack, unpack, tensor_bytes, tensor_from_bytes
    from .r5_sparse_qkv import SparseOrder, envelope
    from .r5_i5_contracts import validate_request
    from .r4c_workload import termination
    from .user_device import cpu_deltas
    if not fixture:
        import torch
        require(not torch.cuda.is_initialized() and all(t.device.type == 'cpu' for t in matrices.values()) and
                'transformers' not in sys.modules, 'CPU-only isolated UD required')
    order, finished = SparseOrder(identity, registry), set()
    expected = None if expected_requests is None else list(expected_requests)
    if expected is not None:
        require(expected and len(expected) <= 50 and
                len({r['request_id'] for r in expected}) == len(expected), 'invalid expected UD requests')
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError('UD condition deadline exceeded')
        h, b, _ = channel.recv()
        if h['kind'] == 'STOP':
            require(order.layer == 0 and not b and h['request_id'] == 'shutdown' and h['layer_id'] == -1 and h['shape'] == [],
                    'invalid stop/unfinished sparse pass')
            require(set(order.parents) == finished and (expected is None or len(finished) == len(expected)),
                    'STOP before final request ACK')
            channel.send(header('BYE', 'shutdown', -1))
            return
        if h['kind'] == 'HELLO' and set(h['metadata']) == {'sparse_complete'}:
            control = h['metadata']['sparse_complete']
            req, history = validate_request(control['request']), control['generated_token_ids']
            previous = order.parents.get(req['request_id'])
            require(not b and order.layer == 0 and req['request_id'] not in finished and previous is not None and
                    previous[0] == req and len(history) == len(previous[1]) + 1 and history[:-1] == previous[1] and
                    termination(req['generation_policy'], history) == control['termination'], 'invalid sparse generation completion')
            finished.add(req['request_id'])
            channel.send(header('READY', h['request_id'], -1, completion_sha256=digest(control)))
            trace(dict(kind='COMPLETE_REQUEST', request_id=req['request_id']))
            continue
        require(set(h['metadata']) == {'sparse'} and h['kind'] in ('HIDDEN', 'HELLO'), 'sparse projection frame required')
        c = h['metadata']['sparse']
        if expected is not None:
            require(len(finished) < len(expected) and c['request'] == expected[len(finished)],
                    'UD original request/order/routing differs')
        order.accept(c)
        n, width = len(c['fresh_positions']), identity['hidden_size']
        require(c['request']['request_id'] not in finished, 'projection after completed request')
        require(h['request_id'] == c['pass_id'] and h['layer_id'] == (c['layer'] if n else -1), 'frame/context differs')
        if not n:
            require(h['kind'] == 'HELLO' and not b and h['shape'] == [], 'empty projection must have no tensor')
            channel.send(header('READY', c['pass_id'], -1, sparse_sha256=digest(c), cpu_projection_rows=0))
            trace(dict(request_id=c['request']['request_id'], cpu_projection_rows=0))
            continue
        require(h['kind'] == 'HIDDEN' and h['shape'] == [1, n, width] and h['dtype'] == identity['dtype'], 'sparse tensor shape/dtype differs')
        decoded_start = time.monotonic_ns()
        x = unpack(b, h['dtype']) if fixture else tensor_from_bytes(b, h['dtype'], h['shape'])
        precision = {}
        started = time.monotonic_ns()
        before = service.report()
        delta = service.run_lora(cpu_deltas, x, matrices, c['layer'], width,
                                 torch_backend=not fixture, precision_out=precision, deadline=deadline)
        computed = time.monotonic_ns()
        data = pack(delta, 'float32') if fixture else tensor_bytes(delta)
        encoded = time.monotonic_ns()
        hashed_start = time.monotonic_ns()
        record = dict(sparse_sha256=digest(c), hidden_sha256=digest_bytes(b), delta_sha256=digest_bytes(data),
                      actual_precision=precision, cpu_projection_rows=3 * n,
                      cpu_lora_ms=(computed - started) / 1e6, TEST_FIXTURE_ONLY=fixture or c['request']['TEST_FIXTURE_ONLY'])
        hashed_end = time.monotonic_ns()
        channel.send(header('DELTAS', c['pass_id'], c['layer'], 'float32', [3, 1, n, width], **record), data)
        trace(dict(request_id=c['request']['request_id'], cpu_projection_rows=3 * n,
                   serialization_ms=(encoded - computed) / 1e6,
                   deserialization_ms=(started - decoded_start) / 1e6,
                   hash_ms=(hashed_end - hashed_start) / 1e6,
                   service_delta={k: service.report()[k] - before[k] for k in
                                  ('measured_compute_ms', 'measured_service_ms', 'pacing_ms', 'operations')},
                   service=service.report()))
