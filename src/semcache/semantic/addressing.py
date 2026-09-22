"""Design-only distinction between matching, pool addressing and provenance.

Not wired into GlobalCache or physical reuse. A pool range is not an absolute
position in an arbitrary user query and is not proof of numerical equivalence.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PoolBlockAddress:
    """Paper (c,p), with a zero-based half-open cluster-pool token range."""
    cluster_id: int
    start: int
    end: int

    def __post_init__(self):
        if self.cluster_id < 0 or not 0 <= self.start < self.end:
            raise ValueError("Invalid cluster-pool token range")


@dataclass(frozen=True)
class SourceOccurrence:
    query_id: str
    user_id: str
    adapter_id: str
    start: int
    end: int

    def __post_init__(self):
        if not 0 <= self.start < self.end:
            raise ValueError("Invalid source occurrence")


@dataclass(frozen=True)
class BlockDescriptor:
    """One match key may identify several separately addressed occurrences.

    A future catalog maps (cluster, token IDs) to descriptors, then an explicit
    correctness policy chooses an address. Allocation/reclamation and that
    policy are intentionally not implemented by this design-only value object.
    """
    address: PoolBlockAddress
    token_ids: tuple[int, ...]
    source: SourceOccurrence

    def __post_init__(self):
        if (len(self.token_ids) != self.address.end - self.address.start
                or len(self.token_ids) != self.source.end - self.source.start):
            raise ValueError("Address, content and occurrence lengths must agree")

    @property
    def match_key(self):
        return self.address.cluster_id, self.token_ids
