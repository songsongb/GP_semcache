"""CPU synthetic C7-B tests; existing archived backend, no OPT inference/downloads."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from semcache.experiments.cachegen import c7b_q_capture as b0
from semcache.experiments.cachegen import c7b_q_profiles as b1


@pytest.fixture(scope='module')
def cohort():
    rows=[];pool={'users':{u:{'row_ids':[]} for u in b0.USERS}}
    caps=[];frozen=[]
    for user in b0.USERS:
        for c in range(108):
            cid=f'{user}:c{c}'
            if c<32:caps.append(cid)
            if 32<=c<40:frozen.append(cid)
            for depth in range(1,5):
                sid=f'{cid}:d{depth}';text=f'Dialogue:\nUser: payload {sid}\nAssistant:'
                rows.append(dict(source_id=sid,conversation_id=cid,history_depth=depth,
                    token_ids=[2,6,7,8,9,10,11,12,13,2],current_user_token_indices=[4,5,6,7,8],
                    current_user_char_span=[16,len(text)-11],prompt_text=text,prompt_version=b0.VERSION,
                    prompt_sha256=b0.digest(text),reference_sha256=b0.digest('reference'),source_split='train'))
                pool['users'][user]['row_ids'].append(sid)
    provenance=dict(frozen32_conversation_ids=frozen,capability64_conversation_ids=caps,
        all_excluded_conversation_ids=sorted(frozen+caps),input_hashes={},adapter_hashes=b0.WEIGHTS)
    result=b0.build_cohort(rows,pool,provenance)
    return rows,pool,result


def test_deterministic_balanced_cohort_and_split(cohort):
    rows,pool,c=cohort
    assert b0.build_cohort(list(reversed(rows)),pool,c['provenance'])==c
    assert len(c['blocks'])==len({b['conversation_id'] for b in c['blocks']})==128
    for user in b0.USERS:
        assert sum(b['user']==user for b in c['blocks'])==64
        for depth in range(1,5):
            values=[b for b in c['blocks'] if b['user']==user and b['history_depth']==depth]
            assert len(values)==16
            assert sum(b['split']=='profile_fit' for b in values)==12
            assert sum(b['split']=='candidate_select' for b in values)==4
    assert sum(b['split']=='profile_fit' for b in c['blocks'])==96
    assert sum(b['split']=='candidate_select' for b in c['blocks'])==32
    fit={b['conversation_id'] for b in c['blocks'] if b['split']=='profile_fit'}
    select={b['conversation_id'] for b in c['blocks'] if b['split']=='candidate_select'}
    assert not fit&select
    for key in ('frozen32_conversation_ids','capability64_conversation_ids','all_excluded_conversation_ids'):
        assert not (fit|select)&set(c['provenance'][key])
    by_id={r['source_id']:r for r in rows}
    for block in c['blocks']:
        p=block['w3_start'];row=by_id[block['source_id']]
        assert len(block['token_ids'])==3 and block['token_ids']==row['token_ids'][p:p+3]
        assert {p,p+1,p+2}<=set(row['current_user_token_indices'])
        assert not set(block['token_ids'])&set(block['special_token_ids'])


@pytest.mark.parametrize('kind',['overlap','balance','span','special','duplicate'])
def test_invalid_cohort_fails_closed(cohort,kind):
    c=copy.deepcopy(cohort[2]);b=c['blocks'][0]
    if kind=='overlap':c['provenance']['frozen32_conversation_ids'].append(b['conversation_id'])
    if kind=='balance':b['split']='candidate_select'
    if kind=='span':b['w3_start']=0
    if kind=='special':b['token_ids'][0]=2
    if kind=='duplicate':c['blocks'][1]['conversation_id']=b['conversation_id']
    with pytest.raises(ValueError):b0.validate_cohort(c)


def test_infeasible_cohort_rejected(cohort):
    rows,pool,c=cohort;pool=copy.deepcopy(pool)
    pool['users']['user_a']['row_ids']=pool['users']['user_a']['row_ids'][:4]
    with pytest.raises(ValueError,match='Insufficient'):b0.build_cohort(rows,pool,c['provenance'])


def test_hash_bound_extra_cohort_required(tmp_path):
    path=tmp_path/'evaluation_selection.json'
    episodes=[{'source_conversation_id':'source','target_conversation_id':'target'}]
    b0.write(path,dict(episodes=episodes,selection_sha256=b0.digest(episodes)))
    with pytest.raises(ValueError,match='Unbound'):b0.bound_cohort(path)
    b0.write(tmp_path/'manifest.json',dict(output_hashes={str(path.resolve()):b0.sha(path)}))
    cids,hashes=b0.bound_cohort(path)
    assert cids=={'source','target'} and str(path.resolve()) in hashes
    path.write_text('{}')
    with pytest.raises(ValueError):b0.bound_cohort(path)


def test_native_total_q_capture_only_owns_w3():
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear
    modules=[]
    import inspect
    for layer in range(32):
        config={'config':LoraConfig(r=2,lora_alpha=2)} if 'config' in inspect.signature(Linear).parameters else {}
        module=Linear(torch.nn.Linear(4,2560,bias=False),'user_a',r=2,lora_alpha=2,**config).eval()
        with torch.no_grad():
            module.get_base_layer().weight.fill_(0.25)
            module.lora_A['user_a'].weight.fill_(0.125)
            module.lora_B['user_a'].weight.fill_(0.125)
        modules.append(module)
    adapter=SimpleNamespace(layers=modules,projection_modules=lambda l:{'q':modules[l]})
    x=torch.ones(1,8,4)
    with torch.inference_mode(), b0.total_q_capture(adapter,2) as captured:
        for m in modules:m(x)
    assert set(captured)==set(range(32))
    for q in captured.values():
        assert q.shape==(1,3,2560) and q.dtype==torch.float16 and q.device.type=='cpu'
        assert q.untyped_storage().nbytes()==3*2560*2
        # 1.0 base + 0.125 LoRA contribution, not the delta alone.
        assert torch.equal(q,torch.full_like(q,1.125))
    assert not any(m._forward_hooks for m in modules)


def test_capture_shape_and_finite_guards():
    b0.validate_q_block(torch.ones(32,3,2560,dtype=torch.float16))
    for q in (torch.ones(31,3,2560,dtype=torch.float16),torch.ones(32,4,2560,dtype=torch.float16),
              torch.ones(32,3,2560),torch.full((32,3,2560),float('nan'),dtype=torch.float16)):
        with pytest.raises(ValueError):b0.validate_q_block(q)


@pytest.fixture(scope='module')
def backend(tmp_path_factory):
    root=tmp_path_factory.mktemp('existing-b2')
    revision='08b553c6851a96a7f593f3f31de3b1cbf85067be'
    names=subprocess.run(['git','ls-tree','-r','--name-only',revision,'src/semcache/experiments/cachegen'],capture_output=True,text=True)
    if names.returncode:
        pytest.skip('Existing historical C2 backend not present; no downloads permitted')
    for name in names.stdout.splitlines():
        if not name.endswith('.py'):continue
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(subprocess.check_output(['git','show',revision+':'+name]))
    return b1.load_backend(root/'src')


@pytest.fixture(scope='module')
def small_corpus(cohort):
    # Same role/layer/token shape, smaller hidden width only in CPU codec tests.
    q=torch.linspace(-2,2,32*3*4).reshape(32,3,4).half()
    return {b['calibration_id']:(q+(i%5)/16).half() for i,b in enumerate(cohort[2]['blocks'])}


def test_exact_candidates_and_role_generic_transform(backend):
    assert b1.CANDIDATES==(16,20,24,32)
    assert backend['provenance']['role_generic_verified']
    shape=(32,3,4);raw=bytes(i%255 for i in range(384))
    assert b1.inverse_streams(b1.streams(raw,shape,backend),shape,backend)==raw


@pytest.mark.parametrize('bins',[16,20,24,32])
def test_fit_only_roundtrip_and_no_runtime_fitting(cohort,small_corpus,backend,bins):
    fit=[b for b in cohort[2]['blocks'] if b['split']=='profile_fit'];select=[b for b in cohort[2]['blocks'] if b['split']=='candidate_select']
    seen=[]
    def loader(cid):seen.append(cid);return small_corpus[cid]
    profile=b1.fit_profile(fit,loader,bins,backend,b0.digest(fit))
    assert set(seen)=={b['calibration_id'] for b in fit}
    assert not set(seen)&{b['calibration_id'] for b in select}
    profile=b1.QProfile.from_bytes(profile.to_bytes(),backend)
    counter={'runtime_profile_fit_count':0}
    with b1.no_runtime_fitting(counter):
        for block in select:
            q=small_corpus[block['calibration_id']]
            frame,sizes=b1.encode(q,profile,backend);restored=b1.decode(frame,profile,backend)
            assert torch.isfinite(restored).all()
            assert sizes['compressed_q_total_resident_bytes']==len(frame)
            assert sizes['compressed_q_bitstream_bytes']+sizes['compressed_q_local_metadata_bytes']==len(frame)
            b1.metrics(q,restored)
    assert counter['runtime_profile_fit_count']==0
    with pytest.raises(ValueError,match='only96'):b1.fit_profile(select,loader,bins,backend,'bad')
    with pytest.raises(ValueError):b1.decode(frame[:-1]+b'X',profile,backend)


def test_select_fit_guard_detects_real_calls(backend):
    counter={'runtime_profile_fit_count':0}
    with pytest.raises(RuntimeError,match='forbidden'):
        with b1.no_runtime_fitting(counter):backend['core'].cdf_from_counts([0]*255)
    assert counter['runtime_profile_fit_count']==1


def test_zero_q_and_nonfinite_quantization(cohort,backend):
    fit=[b for b in cohort[2]['blocks'] if b['split']=='profile_fit'];zero=torch.zeros(32,3,4,dtype=torch.float16)
    profile=b1.fit_profile(fit,lambda cid:zero,16,backend,'zero')
    frame,_=b1.encode(zero,profile,backend)
    assert torch.equal(b1.decode(frame,profile,backend),zero)
    with pytest.raises(ValueError):b1.quantize(torch.full_like(zero,float('inf')),16)


def test_accounting_aggregation_and_pareto():
    a=b1.accounting(1000,200,50)
    assert a['q_payload_compression_ratio']==5
    assert a['q_resident_compression_ratio']==4
    assert a['q_resident_byte_reduction_percentage']==75
    rows=[dict(mse=i,relative_l2=i,cosine_similarity=1-i/100,max_absolute_error=i) for i in range(20)]
    aggregate=b1.aggregate_metrics(rows)
    assert aggregate['relative_l2']['p95']==pytest.approx(18.05)
    assert aggregate['mse']['median']==9.5 and aggregate['cosine_similarity']['min']==0.81
    candidates=[dict(candidate='Q16',q_resident_compression_ratio=4,mean_relative_l2=.2),
        dict(candidate='Q20',q_resident_compression_ratio=3,mean_relative_l2=.1),
        dict(candidate='Q24',q_resident_compression_ratio=2,mean_relative_l2=.2)]
    assert b1.pareto(candidates)==['Q16','Q20']
    rows[0]['mse']=float('nan')
    with pytest.raises(ValueError):b1.aggregate_metrics(rows)
    with pytest.raises(ValueError):b1.metrics(torch.tensor([float('inf')]),torch.ones(1))


def test_complete_synthetic_b1_no_freeze_no_kv_change(tmp_path,monkeypatch,cohort,small_corpus,backend):
    root=tmp_path/'run';(root/'capture').mkdir(parents=True)
    (root/'capture/q_blocks.pt').write_bytes(b'SYNTHETIC UNIT FIXTURE')
    b0.write(root/'cohort.json',cohort[2]);b0.write(root/'capture/capture_manifest.json',{})
    kv=tmp_path/'kv.bin';kv.write_bytes(b'IMMUTABLE UNIT FIXTURE')
    before=b0.sha(kv)
    monkeypatch.setattr(b1,'load_capture',lambda path:(cohort[2],{'input_hashes':{}},small_corpus))
    monkeypatch.setattr(b1,'load_backend',lambda path:backend)
    monkeypatch.setattr(b1,'PROFILE_SHA',before)
    b1.calibrate(SimpleNamespace(capture_root=root,output_root=root,storage_src=None,kv_profile=kv))
    assert b0.sha(kv)==before
    m=b0.read(root/'manifest.json')
    assert m['q_candidates']==[16,20,24,32] and m['fit_blocks']==96 and m['candidate_select_blocks']==32
    assert m['runtime_profile_fit_count_on_candidate_select']==0
    assert m['selected_q_candidate'] is None and not m['q_profile_frozen'] and not m['downstream_quality_evaluated']
    assert m['existing_kv_profile_modified'] is False
    assert {p.name for p in (root/'profiles').iterdir()}=={'q16.bin','q20.bin','q24.bin','q32.bin'}
    report=b0.read(root/'candidate_summary.json')
    for profile in report['profiles'].values():
        assert len(profile['per_layer_metric_aggregates'])==32
        assert all('p95' in values['relative_l2'] for values in profile['per_layer_metric_aggregates'].values())
    for name,h in m['output_hashes'].items():assert b0.sha(root/name)==h
    with pytest.raises(ValueError,match='Refusing'):b1.calibrate(SimpleNamespace(capture_root=root,output_root=root,storage_src=None,kv_profile=kv))


def test_capture_artifact_validation_and_tampering(tmp_path,cohort):
    root=tmp_path/'b0';capture=root/'capture';capture.mkdir(parents=True)
    b0.write(root/'cohort.json',cohort[2])
    q=torch.ones(32,3,2560,dtype=torch.float16)
    corpus={b['calibration_id']:q for b in cohort[2]['blocks']}
    torch.save(dict(schema='c7b_total_q_v1',q_object=b0.Q_OBJECT,
        cohort_sha256=b0.sha(root/'cohort.json'),blocks=corpus),capture/'q_blocks.pt')
    manifest=dict(stage='C7-B0',status='COMPLETE',model=b0.MODEL,model_revision=b0.REVISION,tokenizer_revision=b0.REVISION,
        prompt_version=b0.VERSION,adapter_hashes=b0.WEIGHTS,q_object=b0.Q_OBJECT,layers=32,hidden_dim=2560,dtype='float16',blocks=128,
        model_inference_performed=True,training_performed=False,**b0.validate_cohort(cohort[2]),
        cohort_sha256=b0.sha(root/'cohort.json'),input_hashes={},raw_q_bytes=128*q.numel()*2,
        output_hashes={'q_blocks.pt':b0.sha(capture/'q_blocks.pt'),'../cohort.json':b0.sha(root/'cohort.json')})
    b0.write(capture/'capture_manifest.json',manifest)
    _,loaded,blocks=b1.load_capture(root)
    assert len(blocks)==128 and loaded['raw_q_bytes']==128*32*3*2560*2
    (capture/'q_blocks.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):b1.load_capture(root)


def test_b0_provenance_replays_saved_training_and_excludes_other_cohorts(tmp_path,cohort,monkeypatch):
    """Known B3 schemas with integrity checks; no tokenizer/model execution."""
    rows,_,c=cohort
    by_id={r['source_id']:r for r in rows}
    pool={'profile':'full','epochs':2,'users':{}}
    # Full adapter training excludes capability and frozen conversations.
    for user in b0.USERS:
        ids=[r['source_id'] for r in rows if r['source_id'].startswith(user) and r['conversation_id'] not in c['provenance']['all_excluded_conversation_ids']]
        pool['users'][user]=dict(row_ids=ids,conversation_ids=sorted({by_id[s]['conversation_id'] for s in ids}))
    cap=dict(cohort='capability64',examples=[dict(conversation_id=cid) for cid in c['provenance']['capability64_conversation_ids']],
             conversation_ids=c['provenance']['capability64_conversation_ids'])
    frozen={'episodes':[dict(source_conversation_id=cid,target_conversation_id=cid) for cid in c['provenance']['frozen32_conversation_ids']]}
    frozen['selection_sha256']=b0.digest(frozen['episodes'])
    plan=tmp_path/'b3/plan';adapter=tmp_path/'b3/b1_full';plan.mkdir(parents=True);adapter.mkdir()
    b0.write(plan/'evaluation_selection.json',frozen)
    b0.write(plan/'manifest.json',dict(output_hashes={str((plan/'evaluation_selection.json').resolve()):b0.sha(plan/'evaluation_selection.json')}))
    for name in ('training_plan.json','current_user_spans.json'):b0.write(plan/name,{})
    b0.write(adapter/'capability_validation.json',cap)
    saved=copy.deepcopy(pool)
    for user in b0.USERS:saved['users'][user]['supervised_tokens']=123  # Live saved schema includes training-time counters.
    b0.write(adapter/'train_selection.json',saved)
    source=tmp_path/'source.jsonl';semantic=tmp_path/'semantic.jsonl';source.write_text('source');semantic.write_text('semantic')
    provenance={'source_sha256':b0.sha(source)}
    trained=dict(**b0.b3.SCOPE,**provenance,capability_validation_file_sha256=b0.sha(adapter/'capability_validation.json'),
        plan_sha256=b0.digest(saved),train_selection_file_sha256=b0.sha(adapter/'train_selection.json'))
    b0.write(adapter/'training_manifest.json',trained)
    extra=tmp_path/'b3/other_eval';extra.mkdir()
    extra_cid=pool['users']['user_a']['conversation_ids'][0]
    episodes=[dict(source_conversation_id=extra_cid,target_conversation_id=extra_cid)]
    b0.write(extra/'evaluation_selection.json',dict(episodes=episodes,selection_sha256=b0.digest(episodes)))
    b0.write(extra/'manifest.json',dict(output_hashes={str((extra/'evaluation_selection.json').resolve()):b0.sha(extra/'evaluation_selection.json')}))
    monkeypatch.setattr(b0.b3,'load_inputs',lambda *a:(rows,{},frozen,provenance))
    monkeypatch.setattr(b0.b3,'capability_cohort',lambda *a:cap)
    monkeypatch.setattr(b0.b3,'training_subset',lambda *a:pool)
    monkeypatch.setattr(b0,'verify_adapters',lambda *a:None)
    monkeypatch.setattr(b0,'SELECTION_SHA',b0.sha(plan/'evaluation_selection.json'))
    loaded,training,proof=b0.load_provenance(SimpleNamespace(plan_dir=plan,adapter_root=adapter,source=source,semantic=semantic))
    assert extra_cid in proof['all_excluded_conversation_ids']
    assert extra_cid not in proof['frozen32_conversation_ids']
    assert str((extra/'evaluation_selection.json').resolve()) in proof['input_hashes']
    assert proof['adapter_hashes']==b0.WEIGHTS


def test_seraph_home_cache_rejected(tmp_path,monkeypatch):
    monkeypatch.setenv('HF_HOME','/home/khuss/huggingface')
    with pytest.raises(ValueError,match='Forbidden'):b0.seraph_paths(tmp_path)


def test_existing_data_hf_home_cache_is_preserved(tmp_path,monkeypatch):
    monkeypatch.setenv('HF_HOME','/data/khuss/existing_hf')
    for key in ('HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE'):
        monkeypatch.delenv(key,raising=False)
    b0.seraph_paths(tmp_path)
    import os
    assert os.environ['HF_HUB_CACHE']=='/data/khuss/existing_hf/hub'
    assert os.environ['TRANSFORMERS_CACHE']==os.environ['HF_HUB_CACHE']
    assert os.environ['HF_HUB_OFFLINE']=='1'


def test_missing_existing_storage_source_fails_clearly(tmp_path):
    with pytest.raises(ValueError,match='Existing C6 storage src export required'):
        b1.load_backend(tmp_path/'missing')


def test_full_width_codec_is_cpu_only(cohort,small_corpus,backend,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('CPU calibration must not initialize CUDA or download a model')
    monkeypatch.setattr(torch.cuda,'init',forbidden)
    monkeypatch.setattr(torch.Tensor,'cuda',forbidden)
    from transformers import AutoModelForCausalLM
    monkeypatch.setattr(AutoModelForCausalLM,'from_pretrained',forbidden)
    fit=[b for b in cohort[2]['blocks'] if b['split']=='profile_fit']
    small=b1.fit_profile(fit,small_corpus.__getitem__,20,backend,b0.digest(fit))
    # Reuse a synthetic fit CDF to test framing/indexing at the real width.
    # This is a codec test, not a measured/fitted OPT calibration artifact.
    profile=b1.QProfile(dict(small.metadata,hidden=2560),small.cdfs,backend)
    q=torch.linspace(-2,2,32*3*2560).reshape(32,3,2560).half()
    counter={'runtime_profile_fit_count':0}
    with b1.no_runtime_fitting(counter):
        frame,sizes=b1.encode(q,profile,backend)
        restored=b1.decode(frame,profile,backend)
        b1.metrics(q,restored)
    assert restored.shape==q.shape and restored.device.type=='cpu'
    assert sizes['raw_q_bytes']==32*3*2560*2 and counter['runtime_profile_fit_count']==0
