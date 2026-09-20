"""Reusable host and CUDA timers for measurement instrumentation."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from time import perf_counter_ns


@dataclass(frozen=True)
class TimingScope:
    timing_scope: str
    timing_parent: str | None = None
    inclusive_or_exclusive: str = "exclusive"
    clock: str = "cpu_perf_counter_ns"


class CPUWallTimer:
    clock = "cpu_perf_counter_ns"

    def __init__(self):
        self.elapsed_ms = None

    def __enter__(self):
        self._start_ns = perf_counter_ns()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (perf_counter_ns() - self._start_ns) / 1_000_000


class CUDATimer:
    """CUDA-event timer; synchronize only the end event when read."""
    clock = "cuda_event"

    def __init__(self, device=None):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA timing requested but CUDA is unavailable")
        self.device = device if device is not None else torch.cuda.current_device()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self._elapsed_ms = None

    def __enter__(self):
        import torch
        with torch.cuda.device(self.device):
            self.start.record()
        return self

    def __exit__(self, *_):
        import torch
        with torch.cuda.device(self.device):
            self.end.record()

    @property
    def elapsed_ms(self):
        return self.resolve(synchronize=True)

    def resolve(self, *, synchronize=True):
        if self._elapsed_ms is None:
            if synchronize:
                self.end.synchronize()
            self._elapsed_ms = float(self.start.elapsed_time(self.end))
        return self._elapsed_ms


def resolve_cuda_event_pairs(event_pairs, *, synchronize=True):
    """Resolve ordered event pairs with at most one synchronization.

    Synchronizing the final end event completes all earlier work on the same
    stream. Callers with an already-synchronized enclosing event pass False.
    """
    if not event_pairs:
        return []
    if synchronize:
        event_pairs[-1][1].synchronize()
    return [float(start.elapsed_time(end)) for start, end in event_pairs]


class TimingRegistry:
    """Values plus scope metadata for one measured request."""

    def __init__(self, cuda_device=None):
        self.cuda_device = cuda_device
        self.values, self.scopes = {}, {}

    @contextmanager
    def cpu(self, name, *, parent=None, inclusive="exclusive"):
        timer = CPUWallTimer()
        with timer:
            yield timer
        self.values[name] = timer.elapsed_ms
        self.scopes[name] = asdict(TimingScope(name, parent, inclusive))

    @contextmanager
    def cuda(self, name, *, parent=None, inclusive="exclusive"):
        timer = CUDATimer(self.cuda_device)
        with timer:
            yield timer
        self.values[name] = timer
        self.scopes[name] = asdict(TimingScope(name, parent, inclusive, "cuda_event"))

    def export(self):
        cuda_items = [(k, v) for k, v in self.values.items()
                      if isinstance(v, CUDATimer) and v._elapsed_ms is None]
        if cuda_items:
            cuda_items[-1][1].end.synchronize()
        values = {k: (v.resolve(synchronize=False) if isinstance(v, CUDATimer) else v)
                  for k, v in self.values.items()}
        return values, dict(self.scopes)


def close_request_timing(request_timer, registry, collect_timing):
    """Close execution timing before any correctness-only diagnostics."""
    values, scopes = registry.export() if collect_timing else ({}, {})
    request_timer.__exit__(None, None, None)
    if collect_timing:
        values["request_wall_ms"] = request_timer.elapsed_ms
        scopes["request_wall_ms"] = asdict(TimingScope(
            "request_wall_ms", None, "inclusive", "cpu_perf_counter_ns"))
    return values, scopes


@contextmanager
def timer(device=None):
    """Backward-compatible synchronized broad timer used by pre-M8 callers."""
    import torch
    result = {"metric_source": "measured"}
    if device is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    start = perf_counter_ns()
    try:
        yield result
    finally:
        if device is not None and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        result["seconds"] = (perf_counter_ns() - start) / 1_000_000_000
