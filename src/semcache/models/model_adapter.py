from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    @abstractmethod
    def compute_base_qkv(self, hidden_states, layer_idx):
        """Project actual attention-input hidden states, including model biases."""

    def compute_lora_qkv(self, hidden_states, layer_idx, user_id):
        raise NotImplementedError("User adapter loading/execution is deferred")

    def run_attention_with_qkv(self, q, k, v, layer_idx, **kwargs):
        raise NotImplementedError("Attention replacement requires numerical parity validation")

    def extract_qkv_positions(self, q, k, v, positions):
        from .qkv_projection import extract_qkv_positions
        return extract_qkv_positions(q, k, v, positions)

    def merge_cached_and_new_qkv(self, blocks, sequence_length):
        from .qkv_projection import merge_cached_and_new_qkv
        return merge_cached_and_new_qkv(blocks, sequence_length)


class OPTModelAdapter(ModelAdapter):
    """OPT only; Q/K/V are unscaled, pre-head-split projection outputs."""
    def __init__(self, model):
        if getattr(model.config, "model_type", None) != "opt":
            raise ValueError("Milestone 1 supports OPT only; GPT2/LLaMA are unsupported")
        self.model = model.eval()
        try:
            self.layers = model.model.decoder.layers
            for layer in self.layers:
                for name in ("q_proj", "k_proj", "v_proj"):
                    if not callable(getattr(layer.self_attn, name, None)):
                        raise AttributeError(name)
        except AttributeError as exc:
            raise ValueError("Unsupported OPT module layout") from exc

    def compute_base_qkv(self, hidden_states, layer_idx):
        import torch
        if not 0 <= layer_idx < len(self.layers):
            raise IndexError(layer_idx)
        a = self.layers[layer_idx].self_attn
        with torch.inference_mode():
            return a.q_proj(hidden_states), a.k_proj(hidden_states), a.v_proj(hidden_states)

    def inspect(self, inputs, layer_idx):
        """Hook real forward inputs/outputs, then compare independent projections.

        This verifies instrumentation on the SAME hidden states, not cross-query
        QKV reuse. Hooks are removed even if forward raises.
        """
        import torch
        if not 0 <= layer_idx < len(self.layers):
            raise IndexError(layer_idx)
        a = self.layers[layer_idx].self_attn
        captured = {}
        handles = []
        def hook(name):
            def capture(module, args, output):
                captured[name] = output.detach().clone()
                if name == "q":
                    captured["hidden"] = args[0].detach().clone()
            return capture
        try:
            for name in ("q", "k", "v"):
                handles.append(getattr(a, name+"_proj").register_forward_hook(hook(name)))
            with torch.inference_mode():
                self.model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        actual = self.compute_base_qkv(captured["hidden"], layer_idx)
        for name, value in zip(("q", "k", "v"), actual):
            torch.testing.assert_close(value, captured[name])
        return actual

    def projection_metadata(self, layer_idx, projection):
        a = self.layers[layer_idx].self_attn
        hidden = a.q_proj.in_features
        heads = a.num_heads
        widths = [getattr(a, n+'_proj').out_features for n in ('q', 'k', 'v')]
        if widths != [hidden]*3 or hidden % heads or a.head_dim != hidden//heads:
            raise ValueError('Unsupported loaded OPT projection/head dimensions')
        raw = list(projection.shape)
        if raw[-1] != hidden or len(raw) != 3:
            raise ValueError('Unexpected raw projection shape')
        return {'layer_idx': layer_idx, 'hidden_size': hidden,
                'num_attention_heads': heads, 'num_kv_heads': heads,
                'head_dim': a.head_dim, 'raw_shapes': {n: raw for n in ('q','k','v')},
                'head_shapes': {n: [raw[0], heads, raw[1], a.head_dim] for n in ('q','k','v')},
                'dtype': str(projection.dtype), 'device': str(projection.device),
                'scope': 'base_raw_unscaled_linear_projection'}
