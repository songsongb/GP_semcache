"""CPU synthetic C7-B2 checks; no OPT, network, or GPU execution.

Physical integration uses actual archived C2 classes and literal synthetic CDFs,
not a replacement cache/codec or a claim of using the absent SERAPH profiles.
"""
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from semcache.experiments.cachegen import c7b2_q_quality as gate
from semcache.experiments.cachegen import c7b2_runtime as runtime
from semcache.experiments.cachegen import c6_runtime as c6
from semcache.experiments.cachegen import c7b_q_profiles as qcodec
from semcache.experiments.cachegen import c7b_q_capture as b0
from test_cachegen_c7b_q_calibration import backend, cohort


def literal_cdf():
    return tuple(round(65536*i/255) for i in range(256))


@pytest.fixture
def physical(tmp_path, monkeypatch, backend):
    from semcache.experiments.cachegen.c2 import physical_storage as ps
    import semcache.cache.cache_entry as entry_module
    classes = {}
    for name in ('cache_entry','global_cache'):
        path = tmp_path/(name+'.py')
        path.write_bytes(subprocess.check_output(['git','show',
            '08b553c6851a96a7f593f3f31de3b1cbf85067be:src/semcache/cache/'+name+'.py']))
        alias = 'semcache.cache._c7b2_'+name
        spec = importlib.util.spec_from_file_location(alias,path)
        module = importlib.util.module_from_spec(spec);sys.modules[alias] = module;spec.loader.exec_module(module)
        classes[name] = module
    monkeypatch.setattr(entry_module,'CacheEntry',classes['cache_entry'].CacheEntry)
    monkeypatch.setattr(c6,'GlobalCache',classes['global_cache'].GlobalCache)
    codec = ps.FrozenK20V16Codec.__new__(ps.FrozenK20V16Codec)
    codec.profile = backend['fmt'].Profile(backend['fmt'].MODES[1],(literal_cdf(),)*4)
    codec.quantization_device = codec.decode_device = 'cpu'
    codec.expected_hidden = 2560;codec.instrument = False
    codec.coder_backend = backend['fmt'].FAST_CODER;codec.profile_bytes = len(codec.profile.to_bytes())
    kv = c6.Storage.__new__(c6.Storage);kv.codec = codec;kv.mode = ps.MODE_COMPRESSED
    counts = runtime.runtime_counts();storages = {runtime.MODES[0]:runtime.StorageExperiment(kv,None,counts)}
    paths = {}
    for mode,bins in zip(runtime.MODES[1:],(24,32)):
        path = tmp_path/f'q{bins}.bin'
        p = qcodec.QProfile(dict(bins=bins,layers=32,tokens=3,hidden=2560,transform=qcodec.TRANSFORM),
                            (literal_cdf(),)*2,backend)
        path.write_bytes(p.to_bytes());paths['Q'+str(bins)] = path
        frozen = runtime.FrozenQ(path,bins,b0.sha(path),backend,counts)
        storages[mode] = runtime.StorageExperiment(kv,frozen,counts)
    # Distinct known role values at every layer/token; nonzero maxima for C6.
    x = torch.linspace(-2,2,32*3*2560).reshape(32,3,2560).half()
    tensors = {l:tuple((x[l:l+1]+i/8).half() for i in range(3)) for l in range(32)}
    episode = dict(cluster=7, token_ids=[10,11,12], source_start=1, target_start=2,
                   source_user='audit', source_id='synthetic')
    return SimpleNamespace(storages=storages,counts=counts,tensors=tensors,episode=episode,paths=paths,backend=backend)


def test_exact_modes_and_candidates():
    assert runtime.MODES == ('STORAGE_KV_COMP_BASELINE','STORAGE_Q24_KV_COMP','STORAGE_Q32_KV_COMP')
    assert runtime.CANDIDATES == ('Q24','Q32')
    for mode in ('STORAGE_Q16_KV_COMP','STORAGE_Q20_KV_COMP','FULL_PIPELINE','RAW_SEMCACHE'):
        with pytest.raises(ValueError):runtime.mode_bins(mode)
    for bins in (16,20):
        with pytest.raises(ValueError,match='Only frozen'):runtime.FrozenQ(Path('unused'),bins,'',{}, {})
    assert gate.Q_SHAS == dict(Q24='ca82734cb720a7cc27ce21bcbf214415e296d2bf4a57182eb9b2b88caa075885',
        Q32='37071f85757bb4a38973eb917d807322de27da89e76a1774ad8dab726b80a788')


def contains_tensor(value):
    if isinstance(value,torch.Tensor):return True
    if isinstance(value,dict):return any(contains_tensor(v) for v in value.values())
    if isinstance(value,(list,tuple)):return any(contains_tensor(v) for v in value)
    return False


def test_real_c2_lookup_qkv_residency_and_accounting(physical):
    from semcache.semantic.subsequence import Subsequence
    p = physical;frames = [];views = [];accounts = []
    for mode,storage in p.storages.items():
        resident,account = storage.encode(p.episode,p.tensors)
        before_dict = dict(resident.__dict__)
        cache = c6.make_c6_cache('STORAGE_KV_COMP',storage)
        hit = c6.insert_and_lookup_c6(cache,resident,p.episode,Subsequence((10,11,12),2,5),storage)
        assert cache.entries[resident.key] is resident and hit.entry.resident is resident
        assert cache.logical_cache_bytes == cache.charged_cache_bytes == c6.RAW_ENTRY_BYTES
        assert cache.physical_tensor_bytes == account['total_resident_qkv_bytes']
        assert resident.tensors is None
        assert account['raw_q_resident_after_insert'] == (mode == runtime.MODES[0])
        if mode == runtime.MODES[0]:
            assert resident.q_tensors and not hasattr(resident,'compressed_q_frame')
            for l in range(32):assert torch.equal(hit.entry.tensors[l][0],p.tensors[l][0])
        else:
            assert resident.q_tensors is None and not contains_tensor(resident.__dict__)
            expected_q = storage.q.decode(resident.compressed_q_frame)
            for l in range(32):assert torch.equal(hit.entry.tensors[l][0],expected_q[l:l+1])
            assert resident.storage_accounting['stored_q_bytes'] == len(resident.compressed_q_frame)
            assert resident.compressed_q_frame == before_dict['compressed_q_frame']
        assert account['shared_profile_bytes_charged_per_entry'] == 0
        assert account['compressed_q_frame_bytes'] == account['compressed_q_bitstream_bytes']+account['local_q_metadata_bytes']
        assert account['total_resident_qkv_bytes'] == account['resident_raw_q_bytes']+account['compressed_q_frame_bytes']+account['compressed_kv_frame_bytes']
        assert account['whole_qkv_compression_ratio'] == c6.RAW_ENTRY_BYTES/account['total_resident_qkv_bytes']
        assert account['whole_qkv_byte_reduction_percentage'] == pytest.approx(100*(1-account['total_resident_qkv_bytes']/c6.RAW_ENTRY_BYTES))
        frames.append(resident.compressed_kv.bitstream);views.append(hit.entry);accounts.append(account)
    assert frames[0] == frames[1] == frames[2]
    assert all(s.kv.codec is p.storages[runtime.MODES[0]].kv.codec for s in p.storages.values())
    for view in views[1:]:
        for l in range(32):
            for i in (1,2):assert torch.equal(view.tensors[l][i],views[0].tensors[l][i])
    for account in accounts[1:]:
        assert account['incremental_resident_byte_reduction_vs_kv_baseline'] == accounts[0]['total_resident_qkv_bytes']-account['total_resident_qkv_bytes']
    assert p.counts['storage_decode_count'] == 3
    assert p.counts['runtime_q_profile_fit_count'] == p.counts['runtime_storage_kv_cdf_fit_count'] == 0


@pytest.mark.parametrize('mode',runtime.MODES)
def test_lookup_payload_mixed_reuse_same_mask_and_native_skipping(physical,mode):
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear
    from semcache.edgelora.mixed_projection import mixed_projection_path
    from semcache.semantic.hit_selection import CacheHit
    from semcache.semantic.subsequence import Subsequence
    p = physical;storage = p.storages[mode]
    resident,_ = storage.encode(p.episode,p.tensors);view = storage.decode_entry(resident)
    modules = {};observed = []
    for l in range(32):
        modules[l] = {}
        for i,role in enumerate('qkv'):
            kwargs = dict(config=LoraConfig(r=2,lora_alpha=2)) if 'config' in inspect.signature(Linear).parameters else {}
            m = Linear(torch.nn.Linear(4,2560,bias=False),'audit',r=2,lora_alpha=2,**kwargs).eval()
            with torch.no_grad():
                m.get_base_layer().weight.fill_(1+i/8)
                m.lora_A['audit'].weight.zero_();m.lora_B['audit'].weight.zero_()
            m.get_base_layer().register_forward_pre_hook(lambda mod,args:observed.append(args[0].clone()))
            modules[l][role] = m
    adapter = SimpleNamespace(layers=list(range(32)),projection_modules=modules.__getitem__)
    target = torch.arange(28,dtype=torch.float32).reshape(1,7,4)+10
    hit = CacheHit(Subsequence((10,11,12),2,5),view)
    with mixed_projection_path(adapter,'audit',[hit],7) as audit:
        for l in range(32):
            for i,role in enumerate('qkv'):
                output = modules[l][role](target)
                assert torch.equal(output[:,2:5],view.tensors[l][i].to(output))
                expected = target[:,[0,1,5,6]].sum(-1,keepdim=True).expand(1,4,2560)*(1+i/8)
                assert torch.equal(output[:,[0,1,5,6]],expected)
    evidence = runtime.projection_audit(audit,32,7,[hit])
    assert evidence['same_reuse_mask_qkv'] and evidence['hit_positions'] == [2,3,4]
    assert len(observed) == 96 and all(torch.equal(x,target[:,[0,1,5,6]]) for x in observed)
    for row in evidence['per_layer']:
        assert all(row[r+'_reused_projection_rows'] == row[r+'_native_projection_rows_skipped'] == 3 for r in 'qkv')


@pytest.mark.parametrize('role',['q','kv'])
def test_runtime_fitting_hard_failure(backend,role):
    counts = runtime.runtime_counts()
    with pytest.raises(RuntimeError,match='fitting forbidden'):
        with runtime.forbid_fitting(counts,role):backend['core'].cdf_from_counts([0]*255)
    assert counts['runtime_q_profile_fit_count' if role=='q' else 'runtime_storage_kv_cdf_fit_count'] == 1


def test_profile_hash_mismatch_and_runtime_transport_rejected(physical):
    path = physical.paths['Q24']
    with pytest.raises(ValueError,match='hash mismatch'):
        runtime.FrozenQ(path,24,'different',physical.backend,physical.counts)
    obj = runtime.Backend.__new__(runtime.Backend)
    with pytest.raises(ValueError,match='transport compression forbidden'):obj.forward([1],'audit',[],True)


def test_identical_canonical_teacher_continuation():
    obj = runtime.Backend.__new__(runtime.Backend);obj.rows = [{'token_ids':[1,2,3,4]}]
    obj.counts = runtime.runtime_counts();seen = []
    def forward(ids,user,hits,transport):
        seen.append((list(ids),transport))
        return SimpleNamespace(logits=torch.ones(1,len(ids),8)),None
    obj.forward = forward
    for mode in runtime.MODES:
        logits = obj.teacher_logits(SimpleNamespace(mode=mode,hits=[],transport=False),
            dict(target_index=0,target_user='user_a'),[7,6,5])
        assert logits.shape == (3,8)
    assert seen == [([1,2,3,4,7,6],False)]*3


def test_baseline_reproduction_rejects_text_or_tokens():
    original = dict(generated_text='hello',generated_token_ids=[1,2])
    assert all(gate.baseline_case_check(original,original).values())
    for bad in (dict(original,generated_text='different'),dict(original,generated_token_ids=[1,3])):
        with pytest.raises(ValueError,match='baseline generation mismatch'):gate.baseline_case_check(bad,original)
    from semcache.evaluation.bleu import compute_bleu
    saved = compute_bleu(['a b c d'],['a b c d'],**gate.BLEU)
    assert saved['tokenizer'] == '13a' and saved['smoothing'] == 'exp'
    assert saved['effective_order'] is False and saved['lowercase'] is False
    assert 'nrefs:1' in saved['signature'] and not saved['paper_bleu_claimed']
    with pytest.raises(ValueError,match='BLEU mismatch'):gate.verify_score(dict(saved,value=99),saved)


@pytest.fixture
def calibration_files(tmp_path,cohort,backend,monkeypatch):
    root = tmp_path/'b1';(root/'profiles').mkdir(parents=True);(root/'capture').mkdir()
    b0.write(root/'cohort.json',cohort[2]);(root/'capture/q_blocks.pt').write_bytes(b'SYNTHETIC_CAPTURE_HASH_FIXTURE')
    monkeypatch.setattr(gate,'COHORT_SHA',b0.sha(root/'cohort.json'))
    monkeypatch.setattr(gate,'CAPTURE_SHA',b0.sha(root/'capture/q_blocks.pt'))
    capture = dict(stage='C7-B0',status='COMPLETE',model=b0.MODEL,model_revision=b0.REVISION,
        tokenizer_revision=b0.REVISION,prompt_version=b0.VERSION,adapter_hashes=b0.WEIGHTS,cohort_sha256=gate.COHORT_SHA,
        layers=32,hidden_dim=2560,dtype='float16',blocks=128,**b0.validate_cohort(cohort[2]),input_hashes={},
        output_hashes={'q_blocks.pt':gate.CAPTURE_SHA})
    b0.write(root/'capture/capture_manifest.json',capture)
    fit = [r for r in cohort[2]['blocks'] if r['split']=='profile_fit'];hashes = {}
    for bins in (24,32):
        p = qcodec.QProfile(dict(bins=bins,layers=32,tokens=3,hidden=2560,transform=qcodec.TRANSFORM,
            fit_cohort_sha256=b0.digest(fit),fit_block_ids=[r['calibration_id'] for r in fit]),(literal_cdf(),)*2,backend)
        path = root/'profiles'/f'q{bins}.bin';path.write_bytes(p.to_bytes());hashes['Q'+str(bins)] = b0.sha(path)
    monkeypatch.setattr(gate,'Q_SHAS',hashes)
    m = dict(stage='C7-B1',status='COMPLETE',q_candidates=[16,20,24,32],fit_blocks=96,candidate_select_blocks=32,
        **b0.validate_cohort(cohort[2]),runtime_profile_fit_count_on_candidate_select=0,downstream_quality_evaluated=False,
        q_profile_frozen=False,selected_q_candidate=None,existing_kv_profile_modified=False,model=b0.MODEL,
        model_revision=b0.REVISION,tokenizer_revision=b0.REVISION,adapter_hashes=b0.WEIGHTS,
        capture_artifact_sha256=gate.CAPTURE_SHA,cohort_sha256=gate.COHORT_SHA,
        capture_manifest_sha256=b0.sha(root/'capture/capture_manifest.json'),transform_provenance=backend['provenance'],
        existing_kv_profile={'sha256':gate.PROFILE_SHA},profile_hashes=hashes,
        output_hashes={str(p.relative_to(root)):b0.sha(p) for p in root.rglob('*') if p.is_file()})
    b0.write(root/'manifest.json',m)
    return root,m


def test_b1_provenance_profiles_read_only(calibration_files):
    root,_ = calibration_files
    before = {p:b0.sha(p) for p in root.rglob('*') if p.is_file()}
    _,_,profiles,_,fit = gate.verify_b1(root)
    assert tuple(profiles) == ('Q24','Q32') and len(fit) == 96
    assert all(b0.sha(p)==h for p,h in before.items())
    (root/'profiles/q24.bin').write_bytes(b'bad')
    with pytest.raises(ValueError,match='hash mismatch'):gate.verify_b1(root)


@pytest.mark.parametrize('field,value',[('status','FAILED'),('runtime_profile_fit_count_on_candidate_select',1),
    ('selected_q_candidate','Q24'),('q_profile_frozen',True),('q_candidates',[24,32]),('frozen32_overlap_count',1)])
def test_b1_binding_failure_closed(calibration_files,field,value):
    root,m = calibration_files;m = copy.deepcopy(m);m[field] = value
    (root/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):gate.verify_b1(root)


def test_authoritative_pins_fail_before_any_model_or_c6_execution(tmp_path,monkeypatch):
    decision = tmp_path/'freeze.json';decision.write_text('{}')
    def forbidden(*args,**kwargs):
        raise AssertionError('Mismatched authority must fail before model/runtime preparation')
    monkeypatch.setattr(gate.b3,'prepare',forbidden)
    args = SimpleNamespace(freeze_decision=decision,source=tmp_path/'source',semantic=tmp_path/'semantic',
        plan_dir=tmp_path/'plan',kv_profile=tmp_path/'kv.bin')
    with pytest.raises(ValueError,match='Artifact hash mismatch/missing'):gate.prepare(args)


def test_synthetic_quality_orchestration_no_freeze(tmp_path,monkeypatch):
    episodes = [dict(episode_id=str(i),source_index=0,target_index=i,target_user='user_a' if i<16 else 'user_b',
        history_depth=1+i//8) for i in range(32)]
    rows = [dict(token_ids=[1,2,3],reference_text='a b c d') for _ in range(32)]
    original = [dict(generated_token_ids=[4,5,6,7],generated_text='a b c d') for _ in episodes]
    official = [dict(generated_token_ids=[8,9,10],generated_text='canonical teacher only') for _ in episodes]
    sample = dict(user='user_a',history_depth=1,generated_length=4,generated_text='a b c d',reference_text='a b c d',
        normalized_edit_distance=0.,position_agreement=1.)
    saved = dict(corpus_bleu=gate.aggregate([sample])['corpus_bleu'])
    monkeypatch.setattr(gate,'CANONICAL_BLEU',saved['corpus_bleu']['value'])
    seen = []
    class SyntheticBackend:
        def __init__(self,*args):
            self.counts = runtime.runtime_counts();self.reuse_audits = [];self.metadata = {};self.tokenizer = object()
            self.shared_profile_bytes = dict(KV=4000,Q24=6000,Q32=6000)
        def prepare(self,e,mode):
            for k in ('source_forward_count','storage_decode_count'):self.counts[k] += 1
            if mode != runtime.MODES[0]:
                for k in ('q_encode_count','q_decode_count'):self.counts[k] += 1
            qraw = 100 if mode == runtime.MODES[0] else 0;qframe = 0 if qraw else 25
            a = dict(raw_q_bytes=100,raw_k_bytes=100,raw_v_bytes=100,raw_qkv_bytes=300,resident_raw_q_bytes=qraw,
                compressed_q_bitstream_bytes=20 if qframe else 0,local_q_metadata_bytes=5 if qframe else 0,
                compressed_q_frame_bytes=qframe,compressed_kv_frame_bytes=50,local_kv_metadata_bytes=10,
                total_resident_qkv_bytes=qraw+qframe+50,incremental_resident_byte_reduction_vs_kv_baseline=100-qraw-qframe,
                raw_q_resident_after_insert=bool(qraw))
            return SimpleNamespace(mode=mode,accounting=a)
        def event_identity(self,context,e):return b0.digest(e)
        def greedy(self,context,e):
            self.reuse_audits.append({'same_reuse_mask_qkv':True});seen.append(('greedy',context.mode))
            return [4,5,6,7],'a b c d'
        def teacher_logits(self,context,e,canonical):
            self.reuse_audits.append({'same_reuse_mask_qkv':True})
            seen.append(('teacher',context.mode,list(canonical)))
            self.counts['teacher_forced_forward_count'] += 1
            return torch.arange(24,dtype=torch.float32).reshape(3,8)
    monkeypatch.setattr(gate,'Backend',SyntheticBackend)
    monkeypatch.setattr(gate.b3.d,'encode_example',lambda *a:dict(input_ids=[1,2,3,4,5,6,7],prompt_length=3))
    prepared = SimpleNamespace(profiles={},q_backend={'provenance':{'synthetic':True}},rows=rows,episodes=episodes,official=official,
        baseline=original,input_hashes={},baseline_summary=saved,imported_context=[{'mode':'RAW_SEMCACHE'}])
    root = tmp_path/'quality';root.mkdir();manifest = dict(stage='C7-B2',q_profile_frozen=False,selected_q_candidate=None)
    gate.run(SimpleNamespace(output_root=root),prepared,manifest)
    summary = b0.read(root/'summary.json');m = b0.read(root/'manifest.json')
    assert summary['executed_c7b2_modes'] == list(runtime.MODES)
    assert summary['selected_q_candidate'] is None and not summary['q_profile_frozen'] and summary['manual_freeze_required']
    assert len([s for s in seen if s[0]=='greedy']) == 96
    assert all(s[2] == [8,9,10] for s in seen if s[0]=='teacher')
    assert len(summary['paired_quality']) == 2 and all(p['exact_generation_match_count']==32 for p in summary['paired_quality'])
    assert summary['storage_accounting'][runtime.MODES[1]]['whole_qkv_compression_ratio'] == 4
    assert summary['storage_accounting'][runtime.MODES[1]]['total_resident_bytes'] == 32*75
    assert m['baseline_matches_c6b3_2'] and m['runtime_q_profile_fit_count'] == m['runtime_storage_kv_cdf_fit_count'] == 0
    assert not (root/'freeze_decision.json').exists()
    assert {p.name for p in root.iterdir()} == {'summary.json','summary.csv','per_case.csv','paired_quality.json',
        'causal_comparisons.json','storage_accounting.json','manifest.json','summary.md'}
    for name,h in m['output_hashes'].items():assert b0.sha(root/name)==h


def test_c6_canonical_import_requires_hash_bound_episodes_and_bleu(tmp_path,monkeypatch):
    episodes = [dict(episode_id=str(i),source_index=0,target_index=i,source_user='user_a',target_user='user_a' if i<16 else 'user_b',
        history_depth=1+i//8,token_ids=[10,11,12],cache_key=[7,[10,11,12]],optional=None,
        nested={'z':1,'a':2}) for i in range(32)]
    rows = [dict(reference_text='a b c d') for _ in episodes]
    cases = [];official = []
    for e in episodes:
        shared = dict(e,user=e['target_user'],reference_text='a b c d',generated_text='a b c d',generated_token_ids=[4,5,6,7],
            generated_length=4,normalized_edit_distance=0.,position_agreement=1.,logical_event_hash=b0.digest(e))
        cases.extend([dict(shared,mode='FULL_RECOMPUTE'),dict(shared,mode='STORAGE_KV_COMP')])
        official.append(dict(generated_text='a b c d',generated_token_ids=[4,5,6,7]))
    saved = dict(mode='STORAGE_KV_COMP',corpus_bleu=gate.aggregate(cases)['corpus_bleu'])
    monkeypatch.setattr(gate,'CANONICAL_BLEU',saved['corpus_bleu']['value'])
    b0.write(tmp_path/'summary.json',{'modes':[saved]});gate.write_csv(tmp_path/'per_case.csv',cases)
    provenance = {k:'bound-current' for k in ('training_manifest_sha256','freeze_decision_sha256','evaluation_selection_sha256',
        'semantic_workload_sha256','plan_manifest_sha256','adapter_hashes')}
    m = dict(stage='C6-B3-2',status='COMPLETE',physical_safety_contract=gate.CONTRACT,model=b0.MODEL,revision=b0.REVISION,
        prompt_version=b0.VERSION,seed=42,dtype='float16',storage_profile_sha256=gate.PROFILE_SHA,generation=gate.b3.GENERATION,
        bleu_protocol=gate.BLEU,**provenance,model_tokenizer_provenance=dict(resolved_model_revision=b0.REVISION,
        resolved_tokenizer_revision=b0.REVISION),output_hashes={name:b0.sha(tmp_path/name) for name in ('summary.json','per_case.csv')})
    b0.write(tmp_path/'manifest.json',m)
    baseline,_,_,_ = gate.read_c6_baseline(tmp_path,rows,episodes,provenance,official)
    assert len(baseline)==32 and all(c['generated_token_ids']==[4,5,6,7] for c in baseline)
    # A correct self-hash cannot authorize a different frozen episode/HIT.
    bad = copy.deepcopy(cases);bad[1]['token_ids']=[10,11,13]
    gate.write_csv(tmp_path/'per_case.csv',bad);m['output_hashes']['per_case.csv']=b0.sha(tmp_path/'per_case.csv')
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='frozen episode changed'):gate.read_c6_baseline(tmp_path,rows,episodes,provenance,official)
    del m['output_hashes']['per_case.csv'];(tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='required output hash'):gate.read_c6_baseline(tmp_path,rows,episodes,provenance,official)
