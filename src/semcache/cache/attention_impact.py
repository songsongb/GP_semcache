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
    name = "reproduction_frobenius_mean"

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


class PaperRowL2SumReducer(AttentionImpactReducer):
    """Sum layer/token row L2 norms; mean heads per row is explicit choice.

    A is the actual full-prefill attention, not a newly normalized block-only
    attention matrix. Select query rows [start,end), retaining all causal valid
    keys. This interpretation and the head reduction are reported in metadata.
    """
    name = "paper_row_l2_sum"

    @property
    def metadata(self):
        return dict(name=self.name, formula="sum_l sum_t mean_h sqrt(sum_k A[l,h,t,k]^2)",
                    layer_aggregation="sum", token_aggregation="sum",
                    head_aggregation="mean_REPRODUCTION_CHOICE",
                    query_position_scope="block_[start:end)",
                    key_position_scope="all_valid_causal_keys_REPRODUCTION_CHOICE",
                    causal_mask_respected=True, provenance="PAPER_DEFINED_row_l2_sum",
                    interpretation="actual full-prefill attention; no block renormalization")

    def reduce(self, attentions, start, end, valid_attention_mask=None):
        import torch
        if not attentions or not 0 <= start < end:
            raise ValueError("Need attention tensors and a nonempty query span")
        total = 0.0
        for attention in attentions:
            if (attention is None or attention.ndim != 4 or attention.shape[0] != 1
                    or attention.shape[1] < 1 or attention.shape[2] != attention.shape[3]
                    or end > attention.shape[2]):
                raise ValueError("Expected batch-one square prefill attention covering query span")
            values = attention.detach().double()[0, :, start:end, :]
            n = attention.shape[-1]
            valid = (torch.arange(n, device=values.device)[None, :]
                     <= torch.arange(start, end, device=values.device)[:, None])
            if valid_attention_mask is not None:
                mask = valid_attention_mask.to(device=values.device, dtype=torch.bool)
                if mask.ndim == 2 and mask.shape[0] == 1:
                    mask = mask[0]
                if mask.ndim != 1 or len(mask) != n:
                    raise ValueError("Valid mask must cover prefill positions")
                valid &= mask[start:end, None] & mask[None, :]
            values = values.masked_fill(~valid.unsqueeze(0), 0)
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite valid attention")
            total += values.norm(dim=-1).mean(dim=0).sum().item()
        if not math.isfinite(total):
            raise ValueError("Nonfinite attention impact")
        return total


IMPACT_MODES = (PaperRowL2SumReducer.name, MeanLayerHeadFrobeniusReducer.name)


def make_impact_reducer(mode="paper_row_l2_sum"):
    reducers = {PaperRowL2SumReducer.name: PaperRowL2SumReducer,
                MeanLayerHeadFrobeniusReducer.name: MeanLayerHeadFrobeniusReducer}
    if mode not in reducers:
        raise ValueError(f"Unsupported attention impact reducer: {mode}")
    return reducers[mode]()
