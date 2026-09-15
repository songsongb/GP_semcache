import json
from pathlib import Path
import pytest
from semcache.experiments.dataset_adapters import normalize, load_source
from semcache.experiments.workload import build_workload, save_workload, read_workload, serialize
from semcache.experiments.user_assignment import assign_users
from semcache.experiments.audit import static_reuse_opportunity, validate_workload
from semcache.experiments.manifest import write_manifest

FIXTURES = {
 'multiwoz':[{'dialogue_id':'d1','services':['hotel'],'turns':{'turn_id':['0','1'],'speaker':[0,1],'utterance':['find a hotel','Which area?']}}],
 'coqa':[{'id':'c1','source':'fiction','story':'A girl lives in Seoul.','questions':[{'turn_id':1,'input_text':'Where does she live?'}],'answers':[{'turn_id':1,'input_text':'Seoul'}]}],
 'snips':[{'id':'s1','utterance':'play some music','intent':'PlayMusic'}]}

@pytest.mark.parametrize('dataset',FIXTURES)
@pytest.mark.parametrize('transformation',['raw_query','paper_reproduction_v1'])
def test_adapter_schema_deterministic(dataset,transformation):
    a,dropped=normalize(dataset,FIXTURES[dataset],transformation=transformation)
    assert (a,dropped)==normalize(dataset,FIXTURES[dataset],transformation=transformation)
    assert not dropped
    assert set(a[0])=={'dataset','source_split','source_id','query_text','reference_text','domain_or_intent','conversation_id','metadata'}
    assert a[0]['reference_text']
    if dataset=='coqa':
        assert ('Story:' in a[0]['query_text'])==(transformation=='paper_reproduction_v1')
        assert a[0]['metadata']['story']=='A girl lives in Seoul.'


def test_native_local_formats(tmp_path):
    cases={'multiwoz':{'d.json':{'goal':{'hotel':{}},'log':[{'text':'hotel please'},{'text':'yes'}]}},
           'coqa':{'version':'1.0','data':FIXTURES['coqa']},
           'snips':{'intents':{'PlayMusic':{'utterances':[{'data':[{'text':'play '},{'text':'jazz'}]}]}}}}
    for dataset, data in cases.items():
        path=tmp_path/f'{dataset}.json'; path.write_text(json.dumps(data))
        examples,source=load_source(dataset,path)
        rows,_=normalize(dataset,examples)
        assert rows and source['source_sha256']
        assert source['source_format'] == 'json'
    with pytest.raises(FileNotFoundError,match='No dataset supplied'):
        load_source('coqa')


def build(count=101,**kwargs):
    examples=[dict(id=str(i),utterance=f'play some music {i}',intent='music') for i in range(count)]
    return build_workload('snips',examples,{'source_split':'train'},**kwargs)


def test_users_order_hash_and_manifest(tmp_path):
    rows,m=build()
    assert len(set(r['user_id'] for r in rows))==50
    assert [r['source_id'] for r in rows]==list(map(str,range(101)))
    a,ma=build(order='seeded_shuffle',seed=12)
    b,mb=build(order='seeded_shuffle',seed=12)
    assert a==b and ma['sha256']==mb['sha256']
    assert a!=rows
    c,mc=build(order='seeded_shuffle',seed=13)
    assert mc['sha256']!=ma['sha256']
    assert {r['source_id']:r['user_id'] for r in a}=={r['source_id']:r['user_id'] for r in build(seed=12)[0]}
    assert assign_users(rows,mode='deterministic_hash')==assign_users(rows,mode='deterministic_hash')
    path=tmp_path/'data.jsonl'; save_workload(path,a,ma)
    assert read_workload(path)==(a,ma)
    assert path.read_bytes()==serialize(a)
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='SHA256'):
        read_workload(path)


def test_limits_drops_and_secrets(tmp_path):
    rows,m=build(max_queries=5)
    assert len(rows)==5 and m['limited_query_count']==96
    with pytest.raises(ValueError): build(max_queries=0)
    with pytest.raises(ValueError): assign_users(rows,user_count=0)
    with pytest.raises(ValueError,match='Secret'):
        write_manifest(tmp_path/'bad.json',{'nested':{'hf_token':'never write'}})
    assert not (tmp_path/'bad.json').exists()
    with pytest.raises(ValueError): normalize('snips',[{'utterance':'x','intent':'y','id':'1'}]*2)


def test_static_opportunities_and_coverage():
    rows=[dict(query_text='a b c d',user_id=u,domain_or_intent='x',reference_text='') for u in ['a','a','b']]
    out=static_reuse_opportunity(rows,str.split)
    assert out['total_windows']==6 and out['unique_windows']==2 and out['repeated_windows']==4
    assert out['repeated_within_user']==2 and out['repeated_across_users']==2
    assert out['potential_cross_user_reused_token_coverage']==4
    report=validate_workload(rows,{'user_count':50})
    assert report['queries_per_user']['min']==0
    assert report['duplicate_exact_queries']==2
    assert report['reported_hit_rate_definition'] is None


@pytest.mark.parametrize('dataset',FIXTURES)
def test_optional_real_dataset(dataset):
    import os
    if os.environ.get('SEMCACHE_DATASET_INTEGRATION')!='1':
        pytest.skip('Real dataset integration disabled')
    path=os.environ.get('SEMCACHE_'+dataset.upper()+'_PATH')
    if not path or not Path(path).exists():
        pytest.skip(f'No explicit local {dataset} dataset path')
    examples,source=load_source(dataset,path)
    rows,manifest=build_workload(dataset,examples[:10],source,max_queries=10)
    assert rows and validate_workload(rows,manifest)['query_count']>0


def test_hf_coqa_strings_and_jsonl(tmp_path):
    ex=dict(FIXTURES['coqa'][0],questions=['Where does she live?'],answers={'input_text':['Seoul'],'turn_id':[1]})
    rows,_=normalize('coqa',[ex])
    assert rows[0]['reference_text']=='Seoul'
    path=tmp_path/'snips.jsonl'
    path.write_text(json.dumps(FIXTURES['snips'][0])+'\n')
    examples,source=load_source('snips',path)
    assert examples==FIXTURES['snips']
    assert source['source_format'] == 'json'  # One JSONL record is valid full JSON.


@pytest.mark.parametrize('suffix,layout,detected', [
    ('.json', 'array', 'json'),
    ('.jsonl', 'lines', 'jsonl'),
    ('.json', 'lines', 'jsonl'),
    ('.json', 'pretty', 'json'),
    ('.jsonl', 'pretty', 'json'),
])
def test_local_json_content_detection(tmp_path, suffix, layout, detected):
    examples = [dict(FIXTURES['snips'][0], id=str(i), utterance=f'play 음악 {i}')
                for i in range(2)]
    if layout == 'lines':
        text = '\n\n' + '\n \n'.join(json.dumps(ex, ensure_ascii=False) for ex in examples) + '\n'
    else:
        text = json.dumps(examples, ensure_ascii=False, indent=2 if layout == 'pretty' else None)
    path = tmp_path / ('train' + suffix)
    path.write_text(text, encoding='utf-8')
    loaded, source = load_source('snips', path)
    assert loaded == examples
    assert source['source_format'] == detected
    import hashlib
    assert source['source_sha256'] == hashlib.sha256(text.encode('utf-8')).hexdigest()
    rows, manifest = build_workload('snips', loaded, source)
    assert manifest['source']['source_format'] == detected
    assert [r['query_text'] for r in rows] == [ex['utterance'] for ex in examples]


@pytest.mark.parametrize('dataset', FIXTURES)
def test_single_record_jsonl_remains_supported(tmp_path, dataset):
    path = tmp_path / 'record.jsonl'
    path.write_text(json.dumps(FIXTURES[dataset][0]) + '\n', encoding='utf-8')
    examples, source = load_source(dataset, path)
    assert examples == FIXTURES[dataset]
    assert source['source_format'] == 'json'


def test_malformed_local_json_diagnostics(tmp_path):
    path = tmp_path / 'train.json'
    path.write_text('{"id": 1}\n\n{"id": invalid}\n', encoding='utf-8')
    with pytest.raises(ValueError) as exc:
        load_source('multiwoz', path)
    message = str(exc.value)
    for detail in (str(path), "suffix='.json'", 'full-JSON parse error: Extra data',
                   'JSONL line 3 error: Expecting value'):
        assert detail in message


def test_cli_prepare_validate_smoke(tmp_path):
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    source=tmp_path/'source.json'; source.write_text(json.dumps([dict(id=str(i),utterance='play some music please',intent='music') for i in range(50)]))
    workload=tmp_path/'trace.jsonl'
    config=str(root/'configs/paper/snips.yaml')
    def run(script,*args):
        return subprocess.run([sys.executable,str(root/'scripts'/script),*map(str,args)],cwd=root,capture_output=True,text=True,check=True)
    run('20_prepare_paper_workloads.py','--dataset','snips','--config',config,'--input-path',source,'--output',workload,'--seed',42)
    before=workload.read_bytes()
    a=json.loads(run('21_validate_paper_workloads.py',workload,'--config',config).stdout)
    assert a['report']['query_count']==50 and workload.read_bytes()==before
    run('23_run_workload_smoke.py','--workload',workload,'--config',config,'--max-queries',50,'--output-root',tmp_path/'results','--run-id','cli_test')
    manifest=json.loads((tmp_path/'results/manifests/cli_test.json').read_text())
    assert manifest['query_count']==50 and manifest['execution_mode']=='ANALYTICAL_SIMULATION'
    assert manifest['configuration_provenance']['system.qkv_precision_bits']=='REPRODUCTION_CHOICE'
    other=tmp_path/'other.jsonl'
    run('20_prepare_paper_workloads.py','--dataset','snips','--config',config,'--input-path',source,'--output',other,'--seed',42)
    assert other.read_bytes()==before
    assert read_workload(workload)[1]['manifest_sha256']==read_workload(other)[1]['manifest_sha256']


def multiwoz_groups(count=137):
    return [dict(dialogue_id=f'd{i}', turns=[
        dict(turn_id=str(j), speaker='USER', utterance=f'query {i} turn {j}')
        for j in range(1 + i % 19)]) for i in range(count)]


def test_conversation_preserving_assignment_and_balance():
    examples = multiwoz_groups()
    rows, manifest = build_workload('multiwoz', examples, {'source_split': 'train'})
    again, repeated = build_workload('multiwoz', examples, {'source_split': 'train'})
    assert rows == again
    assert manifest['sha256'] == repeated['sha256']
    assert manifest['manifest_sha256'] == repeated['manifest_sha256']
    assert manifest['user_assignment_rule'] == 'seeded_group_balanced'
    assert manifest['assignment_unit'] == 'conversation'
    assert manifest['grouping_field'] == 'conversation_id'
    assert manifest['provenance']['user_assignment_rule'] == 'REPRODUCTION_CHOICE'
    assert manifest['seed'] == 42 and manifest['user_count'] == 50
    assert manifest['cluster_target'] == 20
    normalized, _ = normalize('multiwoz', examples)
    assert [r['source_id'] for r in rows] == [r['source_id'] for r in normalized]
    assert [r['query_text'] for r in rows] == [r['query_text'] for r in normalized]
    report = validate_workload(rows, manifest)
    assert report['active_users'] == 50
    assert report['total_conversation_count'] == len(examples)
    assert report['conversations_assigned_to_multiple_users'] == 0
    assert report['conversation_split_ratio'] == 0
    assert report['users_per_conversation'] == dict(max=1, mean=1)
    counts = manifest['per_user_record_counts']
    assert max(counts.values()) - min(counts.values()) <= 19  # Largest indivisible group.
    assert sum(counts.values()) == len(rows)
    assert sum(manifest['per_user_conversation_counts'].values()) == len(examples)
    assert report['conversations_per_user']['min'] >= 1
    assert report['per_user_record_counts'] == counts
    assert report['per_user_conversation_counts'] == manifest['per_user_conversation_counts']
    # Assignment is independent of input order and query text, using identities and sizes only.
    reversed_rows = [dict(r, query_text='different text') for r in reversed(normalized)]
    assert assign_users(reversed_rows, mode='seeded_group_balanced') == [r['user_id'] for r in reversed(rows)]
    assert assign_users(normalized, seed=43, mode='seeded_group_balanced') != [r['user_id'] for r in rows]


def test_group_assignment_validation_and_legacy_comparison():
    from copy import deepcopy
    examples = multiwoz_groups(10)
    source = {'source_split': 'train'}
    rows, manifest = build_workload('multiwoz', examples, source)
    broken = deepcopy(rows)
    broken[2]['user_id'] = 'user_049' if broken[1]['user_id'] != 'user_049' else 'user_048'
    with pytest.raises(ValueError, match='conversations_assigned_to_multiple_users=1'):
        validate_workload(broken, manifest)
    broken[2]['conversation_id'] = None
    with pytest.raises(ValueError, match='requires conversation_id'):
        validate_workload(broken, manifest)
    with pytest.raises(ValueError, match='requires a nonempty conversation_id'):
        assign_users([dict(conversation_id=None)], mode='seeded_group_balanced')
    legacy, old_manifest = build_workload('multiwoz', examples, source, assignment='seeded_round_robin')
    assert old_manifest['assignment_unit'] == 'query'
    assert old_manifest['grouping_field'] is None
    assert [r['user_id'] for r in legacy] == assign_users(legacy, mode='seeded_round_robin')
    report = validate_workload(legacy, old_manifest)
    assert report['conversations_assigned_to_multiple_users'] == 9
    assert report['conversation_split_ratio'] == .9
    assert report['users_per_conversation'] == dict(max=10, mean=5.5)


def test_group_assignment_ordering_and_limited_counts():
    examples = multiwoz_groups()
    source = {'source_split': 'train'}
    rows, _ = build_workload('multiwoz', examples, source)
    limited, manifest = build_workload('multiwoz', examples, source, max_queries=17)
    assert limited == rows[:17]
    assert sum(manifest['per_user_record_counts'].values()) == 17
    assert sum(manifest['per_user_conversation_counts'].values()) == len({r['conversation_id'] for r in limited})
    assert validate_workload(limited, manifest)['queries_per_user']['min'] == 0
    shuffled, m = build_workload('multiwoz', examples, source, order='seeded_shuffle')
    assert [r['source_id'] for r in shuffled] != [r['source_id'] for r in rows]
    assert {r['source_id']: r['user_id'] for r in shuffled} == {r['source_id']: r['user_id'] for r in rows}
    assert validate_workload(shuffled, m)['conversations_assigned_to_multiple_users'] == 0


def test_multiwoz_assignment_config_and_cli(tmp_path):
    import subprocess
    import sys
    from semcache.experiments.config import load_paper_config
    root = Path(__file__).resolve().parents[1]
    config = root / 'configs/paper/multiwoz.yaml'
    assert load_paper_config(config)['user_assignment']['mode'] == 'seeded_group_balanced'
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(multiwoz_groups(51)), encoding='utf-8')
    for mode in (None, 'seeded_group_balanced', 'seeded_round_robin'):
        output = tmp_path / f'{mode}.jsonl'
        command = [sys.executable, str(root / 'scripts/20_prepare_paper_workloads.py'),
                   '--dataset', 'multiwoz', '--config', str(config),
                   '--input-path', str(source), '--output', str(output)]
        if mode:
            command += ['--user-assignment', mode]
        subprocess.run(command, check=True, capture_output=True, text=True)
        rows, manifest = read_workload(output)
        assert manifest['user_assignment_rule'] == (mode or 'seeded_group_balanced')
        report = validate_workload(rows, manifest)
        assert report['active_users'] == 50
        assert (report['conversations_assigned_to_multiple_users'] == 0) == (mode != 'seeded_round_robin')
