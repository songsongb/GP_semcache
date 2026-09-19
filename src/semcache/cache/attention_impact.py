"""Actual attention-derived impact with explicit replaceable reduction."""
from abc import ABC, abstractmethod
import math


class AttentionImpactReducer(ABC):
    @abstractmethod
    def reduce(self, attentions, start, end, valid_attention_mask=None):
        pass

    @property
    @abstractmethod
    def metadata(self):
        pass


class MeanLayerHeadFrobeniusReducer(AttentionImpactReducer):
    """Mean of per-(layer, head) Frobenius norms over valid query/key cells."""
    name = "mean_layer_head_frobenius_v1"

    @property
    def metadata(self):
        return dict(name=self.name, layer_aggregation="mean", head_aggregation="mean",
                    query_position_scope="all_valid_query_positions",
                    key_position_scope="block_[start:end)", causal_mask_respected=True,
                    provenance="REPRODUCTION_CHOICE")

    def reduce(self, attentions, start, end, valid_attention_mask=None):
        import torch
        if not attentions or not 0 <= start < end:
            raise ValueError("Need attention tensors and a nonempty key span")
        norm_tensors = []
        for attention in attentions:
            if attention is None or attention.ndim != 4 or attention.shape[0] != 1:
                raise ValueError("Expected batch-one [batch, head, query, key] attention")
            if end > attention.shape[-1]:
                raise ValueError("Block exceeds attention key dimension")
            values = attention.detach().double()[0, :, :, start:end]
            q, k = values.shape[-2:]
            valid = torch.ones((q, k), dtype=torch.bool, device=values.device)
            # The model-produced probabilities already encode causal masking;
            # explicitly exclude future key cells to make reducer semantics stable.
            query_positions = torch.arange(q, device=values.device)[:, None]
            key_positions = torch.arange(start, end, device=values.device)[None, :]
            valid &= key_positions <= query_positions
            if valid_attention_mask is not None:
                mask = valid_attention_mask.to(device=values.device, dtype=torch.bool)
                if mask.ndim == 2:
                    mask = mask[0]
                if mask.ndim != 1 or len(mask) < max(q, end):
                    raise ValueError("Valid mask does not cover attention positions")
                valid &= mask[:q, None] & mask[start:end][None, :]
            masked = values * valid.unsqueeze(0)
            norm_tensors.append(masked.square().sum(dim=(-2, -1)).sqrt())
        # One host materialization after all layers, not one GPU/CPU barrier per
        # layer. Python summation preserves the established aggregation order.
        norms = torch.cat(norm_tensors).tolist()
        if not all(math.isfinite(value) for value in norms):
            raise ValueError("Nonfinite attention")
        value = sum(norms) / len(norms)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid attention impact")
        return value
