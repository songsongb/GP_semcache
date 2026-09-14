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
