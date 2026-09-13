from dataclasses import dataclass
from .cache_metrics import validate_weights


@dataclass(frozen=True)
class EvictionPolicy:
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.2
    delta: float = 0.1

    def __post_init__(self):
        validate_weights((self.alpha, self.beta, self.gamma, self.delta))

    def score(self, m):
        return self.alpha*(1-m.frequency) + self.beta*(1-m.impact) + self.gamma*m.age + self.delta*m.size
