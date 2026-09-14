import os
from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
from semcache.models.model_adapter import OPTModelAdapter
from semcache.models.qkv_injection import qkv_injection
from semcache.evaluation.logit_metrics import compare_logits


def adapter_fixture():
    attention = SimpleNamespace(**{n+'_proj': torch.nn.Linear(4, 4) for n in 'qkv'})
    model = SimpleNamespace(config=SimpleNamespace(model_type='opt'),
                            model=SimpleNamespace(decoder=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])))
    model.eval = lambda: model
    return OPTModelAdapter(model)


@pytest.mark.parametrize('mode', ['q', 'k', 'v', 'qkv'])
def test_injection_positions_modes_and_cleanup(mode):
    adapter = adapter_fixture()
    modules = adapter.projection_modules(0)
    hidden = torch.randn(1, 6, 4)
    fresh = {n: m(hidden).detach() for n, m in modules.items()}
    source = tuple(torch.full((1, 4, 4), i+20., dtype=torch.float64, requires_grad=True) for i in range(3))
    with qkv_injection(adapter, 0, source, (1, 3), (2, 4), mode) as audit:
        actual = {n: m(hidden) for n, m in modules.items()}
    for i, n in enumerate('qkv'):
        expected = fresh[n].clone()
        if n in mode:
            expected[:, 2:4] = source[i].detach()[:, 1:3].float()
        assert torch.equal(actual[n], expected)
        assert not actual[n].requires_grad
        assert not modules[n]._forward_hooks
    assert audit.injected_tensor_bytes == len(mode)*2*4*4


@pytest.mark.parametrize('source_window,target_window', [((0, 2), (1, 4)), ((-1, 1), (1, 3)), ((0, 5), (0, 5))])
def test_bad_windows(source_window, target_window):
    with pytest.raises(ValueError):
        with qkv_injection(adapter_fixture(), 0, (torch.zeros(1, 4, 4),)*3, source_window, target_window):
            pass


def test_invalid_layer_shapes_and_exception_cleanup():
    adapter = adapter_fixture()
    payload = (torch.zeros(1, 2, 4),)*3
    with pytest.raises(ValueError):
        with qkv_injection(adapter, -1, payload, (0, 2), (1, 3)):
            pass
    for payload in [(torch.zeros(2, 2, 4),)*3, (torch.zeros(2, 4),)*3]:
        with pytest.raises(ValueError):
            with qkv_injection(adapter, 0, payload, (0, 2), (1, 3)):
                pass
    payload = (torch.zeros(1, 2, 4),)*3
    with pytest.raises(RuntimeError, match='intentional'):
        with qkv_injection(adapter, 0, payload, (0, 2), (1, 3)):
            raise RuntimeError('intentional')
    with pytest.raises(ValueError, match='Destination'):
        with qkv_injection(adapter, 0, payload, (0, 2), (4, 6)):
            adapter.projection_modules(0)['q'](torch.zeros(1, 3, 4))
    with pytest.raises(RuntimeError, match='did not execute'):
        with qkv_injection(adapter, 0, payload, (0, 2), (1, 3)):
            pass
    assert all(not m._forward_hooks for m in adapter.projection_modules(0).values())


def test_logits_identity_perturbation_direction_and_prefix():
    a = torch.tensor([[[1., 0., -1.], [0., 1., 3.], [2., 0., 0.]]])
    identity = compare_logits(a, a, 1)
    for key in ('max_abs_logit_diff', 'mean_abs_logit_diff', 'relative_l2_logit_diff',
                'last_position_kl_baseline_to_injected', 'affected_suffix_mean_kl', 'prefix_max_abs_logit_diff'):
        assert identity[key] == 0
    assert identity['last_argmax_agreement']
    b = a.clone()
    b[:, 1:, 0] -= 4
    m = compare_logits(a, b, 1)
    assert m['max_abs_logit_diff'] == 4 and m['relative_l2_logit_diff'] > 0
    assert m['prefix_max_abs_logit_diff'] == 0 and not m['last_argmax_agreement']
    p, q = a.double().softmax(-1), b.double().softmax(-1)
    expected = (p*(p/q).log()).sum(-1)
    assert m['last_position_kl_baseline_to_injected'] == pytest.approx(expected[0, -1].item())
    assert m['affected_suffix_mean_kl'] == pytest.approx(expected[:, 1:].mean().item())
    assert m['last_position_kl_baseline_to_injected'] != pytest.approx(compare_logits(b, a, 1)['last_position_kl_baseline_to_injected'])
    b[0, 0, 0] += 0.5
    assert compare_logits(a, b, 1)['prefix_max_abs_logit_diff'] == .5
    assert compare_logits(a, b, 0)['prefix_max_abs_logit_diff'] == 0


def test_cuda_preflight(monkeypatch):
    from semcache.models.runtime import deterministic_cuda_preflight
    monkeypatch.setattr(torch, 'are_deterministic_algorithms_enabled', lambda: True)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    deterministic_cuda_preflight('cpu')
    with pytest.raises(RuntimeError, match='export CUBLAS'):
        deterministic_cuda_preflight('cuda')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    deterministic_cuda_preflight('cuda')


def test_random_opt_cache_backed_output_impact():
    transformers = pytest.importorskip('transformers')
    from test_probes import FixtureTokenizer
    from semcache.evaluation.output_impact import run_output_impact
    from semcache.utils.seed import seed_everything
    seed_everything(42)
    config = transformers.OPTConfig(vocab_size=256, hidden_size=16, word_embed_proj_dim=16,
        num_hidden_layers=2, num_attention_heads=2, ffn_dim=32, max_position_embeddings=256,
        dropout=0, attention_dropout=0)
    config._attn_implementation = 'eager'
    model = transformers.OPTForCausalLM(config).eval()
    rows = run_output_impact(model, FixtureTokenizer(),
        dict(model='random-tiny-OPT-unit-fixture', resolved_model_revision=None, dtype='float32'), layers=[0, 1])
    assert len(rows) == 32
    assert all(r['cache_hit'] and r['projection_integrity_passed'] and r['prefix_max_abs_logit_diff'] == 0 for r in rows)
    assert all(r['max_abs_logit_diff'] == 0 for r in rows if r['probe_case'] == 'A' or (r['probe_case'] == 'B' and r['layer'] == 0))
    assert any(r['max_abs_logit_diff'] > 0 for r in rows if r['probe_case'] == 'C')
    assert all(not r['valid_reuse_candidate'] for r in rows if r['probe_case'] == 'D')


def test_optional_local_opt125m_injection():
    if os.environ.get('SEMCACHE_OPT_INTEGRATION') != '1':
        pytest.skip('Set SEMCACHE_OPT_INTEGRATION=1; local files only')
    pytest.importorskip('transformers')
    from semcache.models.loader import load_model
    from semcache.evaluation.output_impact import run_output_impact
    from semcache.utils.seed import seed_everything
    seed_everything(42)
    revision = '27dcfa74d334bc871f3234de431e71c6eeba5dd6'
    try:
        model, tokenizer, metadata = load_model(dict(name='facebook/opt-125m', tokenizer='facebook/opt-125m',
            revision=revision, tokenizer_revision=revision, dtype='float32', device='cpu',
            local_files_only=True, attention_implementation='eager'))
    except OSError as exc:
        pytest.skip(f'Local OPT-125m weights/tokenizer unavailable: {exc}')
    rows = run_output_impact(model, tokenizer, metadata, layers=[0], modes=['qkv'], cases=['A'])
    assert len(rows) == 1
    row = rows[0]
    assert row['cache_hit'] and row['projection_integrity_passed']
    assert row['max_abs_logit_diff'] <= 1e-6 and row['prefix_max_abs_logit_diff'] <= 1e-6
