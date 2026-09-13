import pytest

torch = pytest.importorskip('torch')
from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.models.qkv_projection import extract_qkv_positions
from semcache.evaluation.qkv_metrics import similarity


def test_physical_owned_cpu_slices_and_logical_mode():
    source = torch.arange(40, dtype=torch.float32).reshape(1,10,4).requires_grad_()
    qkv = extract_qkv_positions(source, source+1, source+2, [3,4])
    entry = CacheEntry.from_tensors(0, (3,4), (3,5), {0:qkv})
    assert entry.physical_size_bytes == entry.logical_size_bytes == 3*2*4*4
    assert all(t.device.type == 'cpu' and not t.requires_grad for t in entry.tensors[0])
    torch.testing.assert_close(entry.tensors[0][0], source[:,3:5])
    assert entry.tensors[0][0].data_ptr() != qkv[0].data_ptr()
    cache = GlobalCache(1000)
    assert cache.lookup(entry.key) is None
    assert cache.insert(entry)
    assert cache.lookup(entry.key) is entry
    assert cache.physical_tensor_bytes == 96
    logical = CacheEntry(0,(9,),(0,1),96)
    assert logical.physical_size_bytes == 0


def test_metrics_known_values_and_layers():
    a = torch.tensor([1.,0.])
    m = similarity(a, torch.tensor([0.,1.]))
    assert m == pytest.approx(dict(cosine_similarity=0, relative_l2=2**.5, max_abs_error=1, mean_abs_error=1))
    assert similarity(a,a)['cosine_similarity'] == 1
    assert similarity(torch.zeros(2),a)['relative_l2'] is None
    with pytest.raises(ValueError):
        similarity(a, torch.ones(3))
    from semcache.probe import rows_for
    from semcache.semantic.probes import select_pair
    wa,wb = select_pair([1,2,3],[1,2,4],2,'A')
    pair = dict(probe_case='A', query_a='a',query_b='b',window_a=wa,window_b=wb,input_ids_a=[1,2,3],input_ids_b=[1,2,4])
    blocks = {0:(a,a,a),1:(a,-a,a)}
    rows = rows_for(pair, blocks, {0:(a,a,a),1:(a,a,a)},dict(model='fixture',resolved_model_revision=None,model_revision='fixture',dtype='float32'))
    assert len(rows) == 6
    assert rows[4]['cosine_similarity'] == -1


def test_random_opt_capture_and_failure_cleanup():
    transformers = pytest.importorskip('transformers')
    from semcache.models.capture import qkv_capture
    model = transformers.OPTForCausalLM(transformers.OPTConfig(vocab_size=32, hidden_size=16,
        word_embed_proj_dim=16, num_hidden_layers=2, num_attention_heads=2, ffn_dim=32,
        max_position_embeddings=32, dropout=0, attention_dropout=0)).eval()
    with qkv_capture(model) as capture:
        with torch.inference_mode():
            model(input_ids=torch.tensor([[2,3,4]]), use_cache=False)
    assert set(capture.records) == {0,1}
    for record in capture.records.values():
        assert record['q'].shape == (1,3,16)
        assert record['metadata']['head_shapes']['q'] == [1,2,3,8]
        assert all(m['max_abs_error'] == 0 for m in record['validation'].values())
    with pytest.raises(RuntimeError, match='intentional'):
        with qkv_capture(model):
            raise RuntimeError('intentional')
    assert not model.model.decoder.layers[0].self_attn.q_proj._forward_hooks


def test_optional_local_opt125m():
    import os
    if os.environ.get('SEMCACHE_OPT_INTEGRATION') != '1':
        pytest.skip('Set SEMCACHE_OPT_INTEGRATION=1 for local-only OPT-125m integration')
    pytest.importorskip('transformers')
    from semcache.models.loader import load_model
    from semcache.probe import run_probe
    config = dict(name='facebook/opt-125m', tokenizer='facebook/opt-125m', revision='main',
        tokenizer_revision='main', dtype='float32', device='cpu', local_files_only=True,
        attention_implementation='eager')
    try:
        model,tokenizer,metadata = load_model(config)
    except OSError as exc:
        pytest.skip(f'Local OPT-125m weights/tokenizer unavailable: {exc}')
    rows,report = run_probe(model,tokenizer,metadata,layers=[0,1])
    assert len(rows) == 24
    rows,report = run_probe(model,tokenizer,metadata,layers=[0],physical=True)
    assert report['physical_cache_hit'] and not report['safe_inference_reuse']
