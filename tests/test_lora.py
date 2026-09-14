import os
import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
peft = pytest.importorskip('peft')
from semcache.models.model_adapter import OPTModelAdapter
from semcache.models.lora_fixtures import (create_controlled_users, base_weight_fingerprint,
    assert_frozen_base, initialize_fixture, load_external_adapter)
from semcache.models.lora_decomposition import decompose_projection, assert_decomposition
from semcache.edgelora.lora_projection import communication_accounting, reconstructed_projection_path
from semcache.evaluation.lora_probe import (capture_user, decomposition_rows, run_multiuser_probe,
    validate_forward, assert_return_identity)


def tiny_base():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        c = transformers.OPTConfig(vocab_size=256, hidden_size=16, word_embed_proj_dim=16,
            num_hidden_layers=2, num_attention_heads=2, ffn_dim=32, max_position_embeddings=256,
            dropout=0, attention_dropout=0)
        c._attn_implementation = 'eager'
        return transformers.OPTForCausalLM(c).eval()


@pytest.fixture
def users():
    return create_controlled_users(tiny_base())


def test_shared_base_targets_rng_and_fingerprints():
    base = tiny_base()
    adapter = OPTModelAdapter(base)
    fingerprint = base_weight_fingerprint(adapter)
    weights = {f'{i}.{n}': m.weight for i in range(2) for n, m in adapter.projection_modules(i).items()}
    rng = torch.random.get_rng_state().clone()
    model, metadata = create_controlled_users(base)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert model.get_base_model() is base
    assert fingerprint == base_weight_fingerprint(OPTModelAdapter(model))
    assert_frozen_base(model)
    for i in range(2):
        for n, m in adapter.projection_modules(i).items():
            assert type(m) is peft.tuners.lora.layer.Linear
            assert m.get_base_layer().weight is weights[f'{i}.{n}']
            assert m.r['user_a'] == m.r['user_b'] == 8
            assert not torch.equal(m.lora_A['user_a'].weight, m.lora_A['user_b'].weight)
            assert not torch.equal(m.lora_B['user_a'].weight, m.lora_B['user_b'].weight)
        assert not adapter.is_lora_projection(adapter.layers[i].self_attn.out_proj)
        assert not adapter.is_lora_projection(adapter.layers[i].fc1)
        assert not adapter.is_lora_projection(adapter.layers[i].fc2)
    assert all(not m['trained_adapter'] for m in metadata.values())
    assert any(p.requires_grad for n, p in model.named_parameters() if '.lora_' in n)


def test_decomposition_nonzero_scaling_bias_and_switch(users):
    model, _ = users
    adapter = OPTModelAdapter(model)
    before = base_weight_fingerprint(adapter)
    hidden = torch.arange(80).reshape(1, 5, 16).float()/80
    for layer in range(2):
        for module in adapter.projection_modules(layer).values():
            outputs = []
            for user in ('user_a', 'user_b', 'user_a'):
                model.set_adapter(user)
                model.eval()
                p = decompose_projection(module, hidden, user)
                metric = assert_decomposition(p)
                assert metric['max_abs_error'] == metric['relative_l2'] == 0
                assert torch.equal(p.base_output, module.base_layer(hidden))
                assert p.lora_delta.norm() > 0
                outputs.append(p)
            a, b, returned = outputs
            assert torch.equal(a.base_output, b.base_output)
            assert not torch.equal(a.lora_delta, b.lora_delta)
            assert torch.equal(a.lora_delta, returned.lora_delta)
            assert torch.equal(a.combined_output, returned.combined_output)
    assert before == base_weight_fingerprint(adapter)
    assert_frozen_base(model)


def test_repeatable_fixture_and_configured_alpha_dropout():
    c = dict(rank=8, alpha=16, dropout=0.4)
    first, _ = create_controlled_users(tiny_base(), c)
    torch.rand(99)
    second, _ = create_controlled_users(tiny_base(), c)
    for (n, p), (nn, pp) in zip(first.named_parameters(), second.named_parameters()):
        assert n == nn and torch.equal(p, pp)
    module = OPTModelAdapter(first).projection_module(0, 'Q')
    assert module.scaling['user_a'] == 2
    assert module.lora_dropout['user_a'].p == .4
    p = decompose_projection(module, torch.ones(1, 3, 16), 'user_a')
    assert_decomposition(p)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize('batch', [1, 2])
def test_communication(dtype, batch):
    h = torch.zeros(batch, 3, 16, dtype=dtype)
    m = communication_accounting(h, h, h, h)
    assert m['es_to_ud_elements'] == batch*3*16
    assert m['ud_to_es_elements'] == 3*batch*3*16
    assert m['total_comm_elements'] == batch*4*3*16
    assert m['paper_comm_elements'] == 4*3*16
    assert m['measured_tensor_bytes'] == batch*4*3*16*h.element_size()
    assert m['hidden_tensor_bytes'] == batch*3*16*h.element_size()
    assert m['q_lora_bytes'] == m['k_lora_bytes'] == m['v_lora_bytes'] == m['hidden_tensor_bytes']


def test_mixed_dtype_communication_and_rejections():
    h = torch.zeros(1, 3, 16, dtype=torch.float16)
    delta = h.float()
    assert communication_accounting(h, delta, delta, delta)['measured_tensor_bytes'] == 48*(2+3*4)
    with pytest.raises(ValueError):
        communication_accounting(h, delta[:, :2], delta, delta)


@pytest.mark.parametrize('state', ['merged', 'disabled', 'multiple', 'wrong_name', 'dora', 'variant', 'train', 'transpose'])
def test_unsupported_states_fail(users, state):
    model, _ = users
    module = OPTModelAdapter(model).projection_module(0, 'q')
    name = 'user_a'
    if state == 'merged':
        module.merge()
    elif state == 'disabled':
        module.enable_adapters(False)
    elif state == 'multiple':
        module.set_adapter(['user_a', 'user_b'])
    elif state == 'wrong_name':
        name = 'user_b'
    elif state == 'dora':
        module.use_dora[name] = True
    elif state == 'variant':
        module.lora_variant[name] = object()
    elif state == 'train':
        module.train()
    elif state == 'transpose':
        module.fan_in_fan_out = True
    with pytest.raises(ValueError):
        decompose_projection(module, torch.ones(1, 3, 16), name)


def test_unwrapped_and_zero_delta_fail(users):
    with pytest.raises(ValueError):
        decompose_projection(torch.nn.Linear(16, 16).eval(), torch.ones(1, 3, 16), 'user_a')
    model, _ = users
    m = OPTModelAdapter(model).projection_module(0, 'q')
    with torch.no_grad():
        m.lora_B['user_a'].weight.zero_()
    with pytest.raises(AssertionError, match='delta is zero'):
        assert_decomposition(decompose_projection(m, torch.ones(1, 3, 16), 'user_a'))


def test_full_forward_and_cleanup(users):
    model, _ = users
    metric = validate_forward(model, [2, 3, 4])
    assert metric['reconstructed_projections'] == 6
    assert metric['max_abs_logit_diff'] == metric['relative_l2_logit_diff'] == metric['affected_suffix_mean_kl'] == 0
    assert metric['last_argmax_agreement']
    adapter = OPTModelAdapter(model)
    for layers in ([0], None):
        assert validate_forward(model, [2, 3, 4], layers=layers)['max_abs_logit_diff'] == 0
    with pytest.raises(RuntimeError, match='intentional'):
        with reconstructed_projection_path(adapter, 'user_a'):
            raise RuntimeError('intentional')
    with pytest.raises(RuntimeError, match='did not execute'):
        with reconstructed_projection_path(adapter, 'user_a'):
            pass
    for i in range(2):
        assert all(not m._forward_hooks for m in adapter.projection_modules(i).values())


def test_probe_schema_and_two_modes(users):
    from test_probes import FixtureTokenizer
    model, fixtures = users
    metadata = dict(model='random-tiny-OPT-unit-fixture', resolved_model_revision=None, dtype='float32')
    records, _ = capture_user(model, [2, 3, 4], 'user_a', [0, 1])
    rows = decomposition_rows(model, records, metadata, fixtures['user_a'])
    assert len(rows) == 6
    assert all(r['decomposition_max_abs_error'] == 0 and r['lora_delta_norm'] > 0 for r in rows)
    assert all(r['component'] == 'total' and not r['safe_reuse_claimed'] for r in rows)
    rows = run_multiuser_probe(model, FixtureTokenizer(), metadata, fixtures, [0, 1])
    assert len(rows) == 144
    assert set(r['probe_mode'] for r in rows) == {'fixed_hidden_input', 'full_user_forward', 'cross_context_full_user_forward'}
    fixed = [r for r in rows if r['same_hidden_input']]
    assert all(r['cross_user_max_abs_error'] == 0 for r in fixed if r['component'] == 'base')
    assert all(r['cross_user_max_abs_error'] > 0 for r in fixed if r['component'] == 'lora_delta')
    assert any(r['cross_user_max_abs_error'] > 0 for r in rows
               if r['probe_mode'] == 'full_user_forward' and r['layer'] == 1 and r['component'] == 'base')
    assert all(not r['trained_adapter'] and not r['safe_reuse_claimed'] for r in rows)
    assert all(set(r) == set(rows[0]) for r in rows)


def test_external_local_roundtrip(users, tmp_path):
    model, _ = users
    model.save_pretrained(tmp_path, selected_adapters=['user_a'])
    restored = load_external_adapter(tiny_base(), tmp_path/'user_a')
    original = OPTModelAdapter(model).projection_module(0, 'q')
    module = OPTModelAdapter(restored).projection_module(0, 'q')
    hidden = torch.ones(1, 3, 16)
    assert_decomposition(decompose_projection(module, hidden, 'external'))
    assert torch.equal(module(hidden), original(hidden))
    with pytest.raises(ValueError):
        load_external_adapter(tiny_base(), tmp_path/'missing')


def test_optional_local_opt125m_lora():
    if os.environ.get('SEMCACHE_OPT_INTEGRATION') != '1':
        pytest.skip('Set SEMCACHE_OPT_INTEGRATION=1; local files only')
    from semcache.models.loader import load_model
    revision = '27dcfa74d334bc871f3234de431e71c6eeba5dd6'
    try:
        base, tokenizer, metadata = load_model(dict(name='facebook/opt-125m', tokenizer='facebook/opt-125m',
            revision=revision, tokenizer_revision=revision, dtype='float32', device='cpu',
            local_files_only=True, attention_implementation='eager'))
    except OSError as exc:
        pytest.skip(f'Local OPT-125m unavailable: {exc}')
    model, fixtures = create_controlled_users(base)
    adapter = OPTModelAdapter(model)
    before = base_weight_fingerprint(adapter)
    ids = tokenizer('I need a hotel in Cambridge.')['input_ids']
    first, logits = capture_user(model, ids, 'user_a', [0])
    rows = decomposition_rows(model, first, metadata, fixtures['user_a'])
    other, _ = capture_user(model, ids, 'user_b', [0])
    returned, logits_returned = capture_user(model, ids, 'user_a', [0])
    assert_return_identity(first, returned, logits, logits_returned)
    assert all(r['lora_delta_norm'] > 0 and r['decomposition_max_abs_error'] <= 1e-6 for r in rows)
    assert any(not torch.equal(first[0][n], other[0][n]) for n in 'qkv')
    assert before == base_weight_fingerprint(adapter)


def test_mixed_precision_matches_native_cast_order(users):
    model, _ = users
    module = OPTModelAdapter(model).projection_module(0, 'q')
    module.base_layer.to(torch.float16)
    hidden = torch.ones(1, 3, 16, dtype=torch.float16)
    parts = decompose_projection(module, hidden, 'user_a')
    assert parts.base_output.dtype == parts.combined_output.dtype == torch.float16
    assert parts.lora_delta.dtype == torch.float32
    assert assert_decomposition(parts)['max_abs_error'] == 0


def test_parity_failure_and_frozen_failure_are_loud(users):
    model, _ = users
    module = OPTModelAdapter(model).projection_module(0, 'q')
    parts = decompose_projection(module, torch.ones(1, 3, 16), 'user_a')
    parts.combined_output = parts.combined_output + .01
    with pytest.raises(AssertionError, match='mismatch'):
        assert_decomposition(parts)
    for tolerance in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            assert_decomposition(parts, tolerance)
    module.base_layer.weight.requires_grad_(True)
    with pytest.raises(AssertionError, match='trainable'):
        assert_frozen_base(model)


@pytest.mark.parametrize('script', ['12_validate_lora_decomposition', '13_probe_multiuser_lora', '14_validate_edgelora_projection_path'])
def test_script_cli_offline(monkeypatch, tmp_path, script):
    import csv
    import json
    import runpy
    import sys
    from pathlib import Path
    from test_probes import FixtureTokenizer
    from semcache.models import loader
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    monkeypatch.setattr(loader, 'load_model', lambda config: (tiny_base(), FixtureTokenizer(),
        dict(model='random-tiny-OPT-unit-fixture', dtype='float32', resolved_model_revision=None)))
    # _lora_common can persist after another runpy invocation.
    sys.modules.pop('_lora_common', None)
    output = tmp_path/(script+'.csv')
    monkeypatch.setattr(sys, 'argv', [script, '--config', str(root/'configs/development.yaml'),
                                     '--layers', '0', '1', '--output', str(output)])
    runpy.run_path(str(root/'scripts'/(script+'.py')), run_name='__main__')
    report = json.loads(output.with_suffix('.json').read_text())
    assert report['rows'] and report['metadata']['model'] == 'random-tiny-OPT-unit-fixture'
    assert all(r['metric_source'] == 'measured' and not r['safe_reuse_claimed'] for r in report['rows'])
    with output.open() as f:
        assert len(list(csv.DictReader(f))) == len(report['rows'])
