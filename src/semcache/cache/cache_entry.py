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
    q_tensors: dict[int, Any] | None = None
    compressed_kv: Any | None = None
    storage_timings_ms: dict = field(default_factory=dict)

    def __post_init__(self):
        self.token_ids = tuple(self.token_ids)
        if self.size_bytes <= 0 or not self.token_ids:
            raise ValueError("Entries need positive size and nonempty tokens")
        if self.positions[0] < 0 or self.positions[1] - self.positions[0] != len(self.token_ids):
            raise ValueError("Invalid source positions")

        if self.tensors is not None:
            self.tensors = {i: tuple(t.detach() for t in qkv) for i, qkv in self.tensors.items()}
        if self.q_tensors is not None:
            self.q_tensors = {i: t.detach() for i, t in self.q_tensors.items()}
        if self.tensors is not None and self.compressed_kv is not None:
            raise ValueError('Compressed entries cannot retain raw K/V tensors')

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
        for t in (self.q_tensors or {}).values():
            s = t.untyped_storage()
            storages[(str(t.device), s.data_ptr())] = s.nbytes()
        return sum(storages.values()) + (len(self.compressed_kv.bitstream) if self.compressed_kv else 0)

    @classmethod
    def from_tensors(cls, cluster_id, token_ids, positions, tensors, storage_device='cpu', *, codec=None):
        """Own compact detached copies; never retain a full prompt through a view."""
        if storage_device != 'cpu' and not str(storage_device).startswith('cuda'):
            raise ValueError('Storage device must be cpu or cuda')
        if not tensors:
            raise ValueError('Physical entry needs projections')
        if codec is not None:
            return codec.make_entry(cls, cluster_id, token_ids, positions, tensors, storage_device)
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

    @property
    def raw_q_bytes(self):
        if self.compressed_kv is not None:
            return sum(t.numel()*t.element_size() for t in self.q_tensors.values())
        return sum(qkv[0].numel()*qkv[0].element_size() for qkv in (self.tensors or {}).values())

    @property
    def raw_kv_bytes(self):
        return self.size_bytes-self.raw_q_bytes if self.tensors or self.compressed_kv else 0

    @property
    def compressed_kv_entry_bytes(self):
        return len(self.compressed_kv.bitstream) if self.compressed_kv is not None else self.raw_kv_bytes

    @property
    def storage_accounting(self):
        raw_q = self.raw_q_bytes
        raw_kv = self.raw_kv_bytes
        stored_q = sum(t.untyped_storage().nbytes() for t in (self.q_tensors or {}).values()) if self.compressed_kv else raw_q
        stored_kv = self.compressed_kv_entry_bytes
        payload = self.compressed_kv
        return dict(raw_q_bytes=raw_q, raw_kv_bytes=raw_kv, raw_qkv_bytes=raw_q+raw_kv,
            stored_q_bytes=stored_q, compressed_kv_entry_bytes=stored_kv,
            encoded_kv_payload_bytes=payload.payload_bytes if payload else None,
            kv_metadata_bytes=payload.local_metadata_bytes if payload else 0,
            role_payload_bytes=dict(zip(('k_anchor', 'k_residual', 'v_anchor', 'v_residual'),
                                        payload.role_payload_bytes)) if payload else None,
            actual_qkv_entry_bytes=stored_q+stored_kv,
            kv_compression_ratio=raw_kv/stored_kv if stored_kv else None,
            overall_entry_compression_ratio=(raw_q+raw_kv)/(stored_q+stored_kv) if stored_q+stored_kv else None,
            global_profile_bytes_charged_per_entry=0)
