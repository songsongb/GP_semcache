import pytest

torch = pytest.importorskip('torch')
from semcache.models.qkv_projection import extract_qkv_positions, merge_cached_and_new_qkv


def test_scatter_merge_and_validation():
    q = torch.arange(24).reshape(1, 4, 6).float()
    tensors = (q, q+1, q+2)
    a = extract_qkv_positions(*tensors, [0, 2])
    b = extract_qkv_positions(*tensors, [3, 1])
    for expected, actual in zip(tensors, merge_cached_and_new_qkv([([0, 2], a), ([3, 1], b)], 4)):
        torch.testing.assert_close(expected, actual)
    with pytest.raises(ValueError):
        merge_cached_and_new_qkv([([0, 2], a), ([0, 1], b)], 4)


def test_tiny_opt_projection_hooks_and_cleanup():
    transformers = pytest.importorskip('transformers')
    from semcache.models.model_adapter import OPTModelAdapter
    torch.manual_seed(42)
    config = transformers.OPTConfig(vocab_size=31, hidden_size=16, word_embed_proj_dim=16,
                                    num_hidden_layers=2, num_attention_heads=2,
                                    ffn_dim=32, max_position_embeddings=32, dropout=0,
                                    attention_dropout=0)
    config._attn_implementation = 'eager'
    model = transformers.OPTForCausalLM(config).eval()
    adapter = OPTModelAdapter(model)
    qkv = adapter.inspect({'input_ids': torch.tensor([[2, 4, 6]])}, 1)
    assert all(t.shape == (1, 3, 16) for t in qkv)
    assert all(not t.requires_grad for t in qkv)
    assert not adapter.layers[1].self_attn.q_proj._forward_hooks
    with pytest.raises(NotImplementedError):
        adapter.compute_lora_qkv(None, 0, 0)
