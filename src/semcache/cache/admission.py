from dataclasses import dataclass
from .cache_metrics import validate_weights


@dataclass(frozen=True)
class AdmissionPolicy:
    alpha: float = 0.5
    beta: float = 0.3
    delta: float = 0.2
    threshold: float = 0.3

    def __post_init__(self):
        validate_weights((self.alpha, self.beta, self.delta))
        if not 0 <= self.threshold <= 1:
            raise ValueError("Invalid admission threshold")

    def score(self, m):
        return self.alpha*m.frequency + self.beta*m.impact + self.delta*(1-m.size)

    def admit(self, m):
        return self.score(m) > self.threshold
