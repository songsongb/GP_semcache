"""Development tie-break: earliest start, descending utility, semantic key."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheHit:
    window: Any
    entry: Any
    utility: float = 0.0


def select_nonoverlapping(hits, sequence_length):
    used = [False] * sequence_length
    selected = []
    for hit in sorted(hits, key=lambda h: (h.window.start, -h.utility, h.entry.key)):
        w = hit.window
        if not 0 <= w.start < w.end <= sequence_length:
            raise ValueError('Hit outside target sequence')
        if w.token_ids != hit.entry.token_ids:
            raise ValueError('Hit tokens differ from cache key')
        if any(used[w.start:w.end]):
            continue
        used[w.start:w.end] = [True] * (w.end-w.start)
        selected.append(hit)
    return selected, used
