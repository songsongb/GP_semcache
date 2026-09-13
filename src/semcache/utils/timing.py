from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timer(device=None):
    def sync():
        if device is not None and str(device).startswith("cuda"):
            import torch
            torch.cuda.synchronize(device)
    result = {"metric_source": "measured"}
    sync()
    start = perf_counter()
    try:
        yield result
    finally:
        sync()
        result["seconds"] = perf_counter() - start
