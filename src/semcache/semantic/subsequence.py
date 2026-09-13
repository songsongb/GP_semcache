from dataclasses import dataclass


@dataclass(frozen=True)
class Subsequence:
    token_ids: tuple[int, ...]
    start: int
    end: int  # exclusive


class SubsequenceExtractor:
    def __init__(self, window_size=3):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size

    def extract(self, token_ids):
        w = self.window_size
        return [Subsequence(tuple(token_ids[i:i+w]), i, i+w) for i in range(len(token_ids)-w+1)]
