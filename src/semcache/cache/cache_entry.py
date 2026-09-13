from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    cluster_id: int
    token_ids: tuple[int, ...]
    positions: tuple[int, int]
    size_bytes: int
    qkv_metadata: dict = field(default_factory=dict)
    tensors: dict[int, tuple[Any, Any, Any]] | None = None
    frequency: int = 0  # actual reuse count (eviction F)
    impact: float | None = None
    created_at: int = 0
    last_access: int = 0
    updated_at: int = 0

    def __post_init__(self):
        self.token_ids = tuple(self.token_ids)
        if self.size_bytes <= 0 or not self.token_ids:
            raise ValueError("Entries need positive size and nonempty tokens")
        if self.positions[0] < 0 or self.positions[1] - self.positions[0] != len(self.token_ids):
            raise ValueError("Invalid source positions")

        if self.tensors is not None:
            self.tensors = {i: tuple(t.detach() for t in qkv) for i, qkv in self.tensors.items()}

    @property
    def key(self):
        return self.cluster_id, self.token_ids

    def age(self, now):
        return max(0, now-self.last_access)

    @property
    def physical_tensor_bytes(self):
        # Storage-level accounting handles views and aliasing within an entry.
        storages = {}
        for qkv in (self.tensors or {}).values():
            for t in qkv:
                s = t.untyped_storage()
                storages[(str(t.device), s.data_ptr())] = s.nbytes()
        return sum(storages.values())

    @classmethod
    def from_tensors(cls, cluster_id, token_ids, positions, tensors, storage_device='cpu'):
        """Own compact detached copies; never retain a full prompt through a view."""
        if storage_device != 'cpu' and not str(storage_device).startswith('cuda'):
            raise ValueError('Storage device must be cpu or cuda')
        if not tensors:
            raise ValueError('Physical entry needs projections')
        owned = {}
        for layer, qkv in tensors.items():
            if len(qkv) != 3 or any(t.ndim != 3 or t.shape[1] != len(token_ids) or t.shape != qkv[0].shape for t in qkv):
                raise ValueError('Expected Q/K/V [batch, window, hidden]')
            owned[layer] = tuple(t.detach().to(storage_device).clone().contiguous() for t in qkv)
        size = sum(t.numel()*t.element_size() for qkv in owned.values() for t in qkv)
        return cls(cluster_id, tuple(token_ids), positions, size, tensors=owned)

    @property
    def logical_size_bytes(self):
        return self.size_bytes

    @property
    def physical_size_bytes(self):
        return self.physical_tensor_bytes
