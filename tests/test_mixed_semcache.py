import os
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('peft')
from test_lora import tiny_base
from semcache.models.lora_fixtures import create_controlled_users
from semcache.models.model_adapter import OPTModelAdapter
from semcache.edgelora.mixed_projection import mixed_projection_path
from semcache.semantic.subsequence import Subsequence
from semcache.semantic.hit_selection import CacheHit
from semcache.semantic.encoder import ControlledEncoder
from semcache.semantic.intent_clusterer import IntentClusterer
from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.cache.admission import AdmissionPolicy
from semcache.cache.metric_manager import attention_impact
from semcache.semcache_engine import SemCacheEngine
from semcache.evaluation.mixed_control import validate_mixed_control


class Tokenizer:
    def __call__(self, text):
        return {'input_ids': [int(x) for x in text.split()]}


@pytest.fixture
def users():
    return create_controlled_users(tiny_base())[0]


def make_engine(model, capacity=100000, admission=None):
    vectors = {'2 3 4 5 6 7 8 9': [0., 0.], '10 2 3 4 5 6 7 8 9': [0., 0.],
               '2 3 4 5 6 7 8 10': [10., 10.]}
    clusterer = IntentClusterer(2)
    clusterer.initialize([[0., 0.], [10., 10.]], counts=[0, 0])
    return SemCacheEngine(model, Tokenizer(), OPTModelAdapter(model), ControlledEncoder(vectors),
                          clusterer, GlobalCache(capacity, admission=admission))


def test_attention_impact_hand_arithmetic():
    # Two heads: norms [1, .5], [0, 1]; head mean, token sum = 1.25.
    a = torch.tensor([[[[1., 0.], [.3, .4]], [[0., 0.], [0., 1.]]]])
    assert attention_impact([a], 0, 2) == pytest.approx(1.25)
    assert attention_impact([a, a], 1, 2) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        attention_impact([None], 0, 1)


def test_exact_control_tiny_opt(users):
    result = validate_mixed_control(users, Tokenizer(), OPTModelAdapter(users), '2 3 4 5 6 7 8 9')
    assert result['exact_control_passed'] and result['matched_windows'] == 2
    assert result['mixed_native_projected_rows'] == 2 and result['reused_rows'] == 6
    assert result['max_abs_logit_diff'] <= 1e-6 and not result['safe_reuse_claimed']


@pytest.mark.parametrize('all_cached', [False, True])
def test_native_base_and_lora_receive_only_unmatched_rows(users, all_cached):
    adapter = OPTModelAdapter(users)
    hidden = torch.arange(128).float().reshape(1, 8, 16)/128
    sources = {i: tuple(m(hidden).detach() for m in adapter.projection_modules(i).values()) for i in range(2)}
    spans = [(0, 3), (3, 6)] if not all_cached else [(0, 3), (3, 6), (6, 8)]
    hits = []
    for start, end in spans:
        ids = tuple(range(start, end))
        entry = CacheEntry.from_tensors(0, ids, (start, end), {i: tuple(t[:, start:end] for t in qkv) for i, qkv in sources.items()})
        entry.qkv_metadata['component_scope'] = 'total_qkv'
        hits.append(CacheHit(Subsequence(ids, start, end), entry))
    calls, handles = [], []
    # Observe inside native PEFT, at base and LoRA A: impossible to hide a full projection.
    for i in range(2):
        for name, module in adapter.projection_modules(i).items():
            for component in (module.base_layer, module.lora_A['user_a']):
                handles.append(component.register_forward_pre_hook(lambda m, args: calls.append(args[0].clone())))
    try:
        with mixed_projection_path(adapter, 'user_a', hits, 8) as audit:
            actual = {i: tuple(m(hidden) for m in adapter.projection_modules(i).values()) for i in range(2)}
    finally:
        for h in handles:
            h.remove()
    assert len(calls) == (0 if all_cached else 12)
    assert all(torch.equal(x, hidden[:, 6:]) for x in calls)
    for i in range(2):
        for a, b in zip(actual[i], sources[i]):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
        assert all('forward' not in m.__dict__ for m in adapter.projection_modules(i).values())
    assert all(r['native_projection_rows']+r['reused_projection_rows'] == 8 for r in audit.records.values())


@pytest.mark.parametrize('failure', ['body', 'shape', 'incomplete', 'hook'])
def test_mixed_cleanup_after_exception(users, failure):
    adapter = OPTModelAdapter(users)
    module = adapter.projection_module(0, 'q')
    handle = module.register_forward_hook(lambda *args: None) if failure == 'hook' else None
    try:
        with pytest.raises((RuntimeError, ValueError)):
            with mixed_projection_path(adapter, 'user_a', [], 8):
                if failure == 'body':
                    raise RuntimeError('intentional')
                if failure == 'shape':
                    module(torch.ones(1, 7, 16))
        assert all('forward' not in m.__dict__ for i in range(2) for m in adapter.projection_modules(i).values())
    finally:
        if handle:
            handle.remove()


def test_integrated_cross_user_cluster_isolation_chu_pbr(users):
    validate_mixed_control(users, Tokenizer(), OPTModelAdapter(users), '2 3 4 5 6 7 8 9')
    engine = make_engine(users)
    first = engine.query('2 3 4 5 6 7 8 9', 'user_a', 'source', True)
    assert first['summary']['block_hit_count'] == 0 and engine.cache.physical_tensor_bytes > 0
    assert all(e.qkv_metadata['component_scope'] == 'total_qkv' for e in engine.cache.entries.values())
    second = engine.query('10 2 3 4 5 6 7 8 9', 'user_b', 'target', True)
    assert second['summary']['accepted_nonoverlap_hits'] == 2
    assert second['summary']['reused_unique_token_count'] == 6
    assert second['summary']['max_abs_logit_diff'] > 0 and not second['summary']['safe_reuse_claimed']
    fetch = next(e for e in second['events'] if e['event_type'] == 'FETCH')
    assert fetch['source_start'] == 0 and fetch['target_start'] == 1
    chu = [e for e in second['events'] if e['event_type'] == 'CHU']
    assert len(chu) == 2 and any(e['I'] != e['old_impact'] for e in chu)
    assert sum(e.frequency for e in engine.cache.entries.values()) == 2
    assert any(u['I'] != u['old_impact'] for u in engine.pbr(0))
    third = engine.query('2 3 4 5 6 7 8 10', 'user_b', 'other_cluster', True)
    assert third['summary']['cluster_id'] == 1 and third['summary']['block_hit_count'] == 0
    assert not engine.lookup_latest_token(0, 2, 8)['hit']


def test_integrated_exact_identity(users):
    engine = make_engine(users)
    engine.query('2 3 4 5 6 7 8 9', 'user_a', 'source')
    row = engine.query('2 3 4 5 6 7 8 9', 'user_a', 'exact', True)['summary']
    assert row['reused_unique_token_count'] == 6 and row['recomputed_tokens'] == 2
    assert row['max_abs_logit_diff'] <= 1e-6 and row['last_argmax_agreement']


def test_denied_does_not_materialize(users, monkeypatch):
    engine = make_engine(users, admission=AdmissionPolicy(threshold=1.))
    def forbidden(*args, **kwargs):
        raise AssertionError('denied physical allocation')
    monkeypatch.setattr(CacheEntry, 'from_tensors', forbidden)
    result = engine.query('2 3 4 5 6 7 8 9', 'user_a', 'denied')
    assert engine.cache.physical_tensor_bytes == engine.cache.logical_cache_bytes == 0
    assert sum(e['event_type'] == 'DENY' for e in result['events']) == 6


def test_integrated_overflow_scores(users):
    engine = make_engine(users, capacity=2*3*3*16*4*2)
    # Check every actual victim against the current population at the eviction callback.
    original = engine.emit
    victims = []
    from semcache.cache.cache_metrics import normalize
    def checked(kind, **details):
        if kind == 'EVICT':
            cache = engine.cache
            metrics = {key: cache._metrics(e) for key, e in cache.entries.items()}
            scores = {key: cache.eviction.score(normalize(m, metrics.values())) for key, m in metrics.items()}
            expected = min(scores, key=lambda key: (-scores[key], key))
            assert details['cache_key'] == expected
            assert details['eviction_score'] == max(scores.values())
            victims.append(expected)
        original(kind, **details)
    engine.emit = checked
    engine.query('2 3 4 5 6 7 8 9', 'user_a', 'overflow')
    assert victims and len(engine.cache.entries) == 2
    assert engine.cache.logical_cache_bytes <= engine.cache.capacity_bytes


@pytest.mark.parametrize('script', ['17_run_semcache_controlled_trace', '18_validate_mixed_projection'])
def test_m5_scripts_offline(users, monkeypatch, tmp_path, script):
    import runpy
    import sys
    import json
    from pathlib import Path
    from test_probes import FixtureTokenizer
    from semcache.models import loader
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    monkeypatch.setattr(loader, 'load_model', lambda config: (tiny_base(), FixtureTokenizer(),
        dict(model='random-tiny-OPT-unit-fixture', dtype='float32', resolved_model_revision=None)))
    sys.modules.pop('_semcache_common', None)
    output = tmp_path/(script+'.csv')
    monkeypatch.setattr(sys, 'argv', [script, '--config', str(root/'configs/development.yaml'),
                                    '--output', str(output), '--compare-baseline'])
    runpy.run_path(str(root/'scripts'/(script+'.py')), run_name='__main__')
    report = json.loads(output.with_suffix('.json').read_text())
    assert report['metadata']['model'] == 'random-tiny-OPT-unit-fixture'
    if script.startswith('17'):
        assert all(not row['safe_reuse_claimed'] for row in report['rows'])
        assert (tmp_path/'semcache_events.jsonl').exists()
    else:
        assert report['control']['exact_control_passed']


def test_optional_pretrained_mixed_control():
    if os.environ.get('SEMCACHE_OPT_INTEGRATION') != '1':
        pytest.skip('Set SEMCACHE_OPT_INTEGRATION=1; local files only')
    from semcache.models.loader import load_model
    revision = '27dcfa74d334bc871f3234de431e71c6eeba5dd6'
    try:
        model, tokenizer, _ = load_model(dict(name='facebook/opt-125m', tokenizer='facebook/opt-125m',
            revision=revision, tokenizer_revision=revision, dtype='float32', device='cpu',
            local_files_only=True, attention_implementation='eager'))
    except OSError as exc:
        pytest.skip(f'Local OPT-125m unavailable: {exc}')
    model, _ = create_controlled_users(model)
    result = validate_mixed_control(model, tokenizer, OPTModelAdapter(model))
    assert result['exact_control_passed'] and result['physical_cache_bytes'] > 0


def test_exact_control_failure_aborts(users, monkeypatch):
    from semcache.evaluation import mixed_control
    original = mixed_control.compare_logits
    def damaged(*args, **kwargs):
        result = original(*args, **kwargs)
        result['max_abs_logit_diff'] = .01
        return result
    monkeypatch.setattr(mixed_control, 'compare_logits', damaged)
    with pytest.raises(AssertionError, match='STOP: exact-control'):
        validate_mixed_control(users, Tokenizer(), OPTModelAdapter(users), '2 3 4 5 6 7 8 9')


def test_overlap_and_wrong_scope_rejected_before_projection(users):
    adapter = OPTModelAdapter(users)
    entry = CacheEntry.from_tensors(0, (2, 3, 4), (0, 3),
        {i: (torch.ones(1, 3, 16),)*3 for i in range(2)})
    hit = CacheHit(Subsequence(entry.token_ids, 0, 3), entry)
    with pytest.raises(ValueError, match='non-total'):
        with mixed_projection_path(adapter, 'user_a', [hit], 8):
            pass
    entry.qkv_metadata['component_scope'] = 'total_qkv'
    with pytest.raises(ValueError, match='overlapping'):
        with mixed_projection_path(adapter, 'user_a', [hit, hit], 8):
            pass
    assert all('forward' not in m.__dict__ for i in range(2) for m in adapter.projection_modules(i).values())
