from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CacheMetrics:
    frequency: float
    impact: float
    age: float
    size: float


def normalize(metrics, population, zero_max=0.0):
    """Paper max normalization; callers explicitly select the population.

    Including the candidate avoids undefined cold-start maxima and values > 1.
    """
    population = list(population)
    names = ("frequency", "impact", "age", "size")
    for m in [metrics, *population]:
        if any(not math.isfinite(getattr(m, n)) or getattr(m, n) < 0 for n in names):
            raise ValueError("Metrics must be finite and nonnegative")
    if zero_max != 0.0:
        raise ValueError("Only zero-max -> 0 is implemented")
    return CacheMetrics(*(getattr(metrics, n)/maximum if (maximum := max((getattr(m, n) for m in population), default=0)) else zero_max for n in names))


def validate_weights(weights):
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1):
        raise ValueError("Weights must be nonnegative and sum to 1")
