from pathlib import Path
from copy import deepcopy
import ast
import json
import pytest
from semcache.simulation.cost_model import paper_latency, projection_savings, communication_seconds, PaperCostModel, mbps_to_bits_per_second
from semcache.simulation.memory_model import qkv_block_bytes
from semcache.experiments.config import load_paper_config, load_model_spec
from semcache.experiments.provenance import metric
from semcache.experiments.aggregation import aggregate, compare
from semcache.experiments.paper_reference import TABLE_II, FIGURE_6
from semcache.experiments.workload import build_workload
from semcache.experiments.runner import make_logical_engine, run_workload

ROOT=Path(__file__).resolve().parents[1]


def test_defaults_and_model_metadata():
    for dataset,c in [('multiwoz',20),('coqa',40),('snips',30)]:
        config=load_paper_config(ROOT/f'configs/paper/{dataset}.yaml')
        assert config['cluster_count']==c and config['num_users']==50
        assert config['lora_rank']==8 and config['subsequence_window']==3
        assert config['logical_cache_capacity_gb']==20
        assert config['admission']==dict(alpha=.5,beta=.3,delta=.2,threshold=.3)
        assert config['eviction']==dict(alpha=.4,beta=.3,gamma=.2,delta=.1)
        assert config['semantic_impact']['rho']==.8 and config['semantic_impact']['history_lambda']==100
        assert config['cluster_update_interval_queries']==100
        assert config['system']['bandwidth_mbps']==200
        assert config['semantic_encoder']['model_id'] is None
        assert all(v is None for v in config['generation'].values())
        assert config['system']['qkv_precision_bits'] is None
        assert load_model_spec(config,ROOT/'configs/paper')['hidden_size']==4096


def test_equations_and_factor_eight():
    assert communication_seconds(25_000_000,1,200)==1
    assert communication_seconds(25_000_000,2,200)==2
    assert communication_seconds(25_000_000,1,400)==.5
    assert projection_savings(212,4096,8)['base_flops_saved']==21340618752
    args=dict(n=512,d=4096,r=8,f_ES=1e12,f_UD=1e9,B=200e6,element_size_bytes=2)
    a=paper_latency(n_reused=0,**args); b=paper_latency(n_reused=212,**args)
    assert a['communication_s']/b['communication_s']==pytest.approx(512/300)
    assert a['remaining_es_s']==b['remaining_es_s']
    assert a['latency_s']/b['latency_s'] != pytest.approx(512/300)
    full=paper_latency(n_reused=512,**args)
    assert full['latency_s']==full['remaining_es_s']
    with pytest.raises(ValueError): paper_latency(n_reused=513,**args)
    for bad in [0,-1,float('nan'),float('inf'),True]:
        with pytest.raises(ValueError): mbps_to_bits_per_second(bad)
    assert qkv_block_bytes(3,32,4096,4096,16)==2359296
    assert qkv_block_bytes(3,32,4096,1024,16)==1179648
    assert qkv_block_bytes(1,1,1,1,4)==2


def test_references_immutable_and_isolated():
    assert TABLE_II['OPT']['SEMCACHE']['latency_s']==6.07
    assert TABLE_II['OPT']['UD_ONLY'] is None
    with pytest.raises(TypeError): TABLE_II['OPT']['SEMCACHE']['latency_s']=0
    with pytest.raises(TypeError): FIGURE_6['cache_metrics']['F']['latency_s']=0
    paths=list((ROOT/'src/semcache/simulation').glob('*.py'))+[ROOT/'src/semcache/experiments/runner.py',ROOT/'src/semcache/experiments/config.py']
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom): assert 'paper_reference' not in (node.module or '')
            if isinstance(node,ast.Import): assert all('paper_reference' not in n.name for n in node.names)
    with pytest.raises((KeyError,ValueError,TypeError)):
        PaperCostModel().estimate(tokens=512,reused_tokens=212,model_config=TABLE_II['OPT']['SEMCACHE'],system_config={})


def test_aggregation_provenance_and_comparability():
    rows=[{'latency_s':metric(v,'SIMULATED','prefill','seconds'),
           'block_hit_count':metric(h,'SIMULATED','cache','count'),
           'block_lookup_count':metric(n,'SIMULATED','cache','count')} for v,h,n in [(1,1,1),(3,1,9)]]
    out=aggregate(rows)
    assert out['avg_latency_s']['value']==2
    assert out['p95_latency_s']['value']==pytest.approx(2.9)
    assert out['block_hit_ratio']['value']==.2
    rows[0]['latency_s']['metric_source']='MEASURED'
    with pytest.raises(ValueError): aggregate(rows)
    with pytest.raises(ValueError): metric(1,'measured','','s')
    context=dict(model='opt67b',precision='FP16',dataset='coqa',transformation='validated',hardware='A100',execution_mode='ANALYTICAL_SIMULATION',execution_scope='full_generation',metric_scope='system',unit='seconds')
    assert compare(6,6,our_context=context,paper_context=context)['comparability']=='DIRECT'
    different=dict(context,model='opt125m',hardware='A2000',execution_mode='MEASURED_MODEL')
    result=compare(1,6,our_context=different,paper_context=context,approximate_reason='cannot override wrong model')
    assert result['comparability']=='NOT_COMPARABLE' and result['relative_difference'] is None
    assert compare(1,6,our_context={},paper_context={})['comparability']=='NOT_COMPARABLE'


def test_m5_engine_logical_workload_smoke(tmp_path):
    config=load_paper_config(ROOT/'configs/paper/snips.yaml')
    examples=[dict(id=str(i),utterance='play some music please',intent='PlayMusic') for i in range(60)]
    rows,manifest=build_workload('snips',examples,{'source_split':'train'})
    config['system']['qkv_precision_bits']=16
    engine=make_logical_engine(rows,config)
    model=load_model_spec(config,ROOT/'configs/paper')
    result,run=run_workload(rows,config,model,engine=engine,workload_manifest=manifest,run_id='test',output_root=tmp_path,max_queries=60)
    assert engine.cache.physical_tensor_bytes==0
    assert result['block_hit_ratio']['value']>0
    assert result['avg_analytical_latency_s']['value'] is None
    assert run['execution_scope']=='prefill_only' and run['query_count']==60
    assert (tmp_path/'manifests/test.json').exists()
    saved=json.loads((tmp_path/'raw/test.json').read_text())
    assert saved[0]['summary']['impact_available'] is False
    assert all(r['summary']['physical_cache_tensor_bytes']==0 for r in saved)
    with pytest.raises(NotImplementedError):
        run_workload(rows,config,model,engine=engine,workload_manifest=manifest,run_id='bad',output_root=tmp_path,max_queries=2,scope='full_generation')
    config['execution_mode']='MEASURED_MODEL'
    with pytest.raises(ValueError):
        run_workload(rows,config,model,engine=engine,workload_manifest=manifest,run_id='bad',output_root=tmp_path,max_queries=2)


def test_bleu_protocol():
    from semcache.evaluation.bleu import compute_bleu
    with pytest.raises(ValueError):
        compute_bleu(['a'],['a'],implementation=None,tokenizer=None,smoothing=None,scope=None,effective_order=None,lowercase=None)
    pytest.importorskip('sacrebleu')
    out=compute_bleu(['one two three four'],['one two three four'],implementation='sacrebleu',tokenizer='none',smoothing='none',scope='corpus',effective_order=False,lowercase=False)
    assert out['value']==pytest.approx(100)
    assert not out['paper_bleu_claimed']


def test_invalid_config_and_override_provenance():
    from semcache.experiments.config import validate_config
    from semcache.experiments.provenance import configuration_provenance
    c=load_paper_config(ROOT/'configs/paper/coqa.yaml')
    assert configuration_provenance(c)['cluster_count']=='PAPER_DEFINED'
    c['cluster_count']=10
    assert configuration_provenance(c)['cluster_count']=='REPRODUCTION_CHOICE'
    for key,bad in [('execution_mode',None),('execution_scope','query'),('cache_storage_mode','allocate20GB')]:
        other=deepcopy(c); other[key]=bad
        with pytest.raises(ValueError): validate_config(other)


def test_cost_replay():
    from semcache.simulation.simulator import EdgeLoRASimulator
    model=dict(hidden_size=16,layers=2,lora_rank=8)
    system=dict(es_tflops=1,ud_tflops=.1,bandwidth_mbps=200,communication_element_bytes=2)
    rows=EdgeLoRASimulator(PaperCostModel()).run([dict(query_token_count=512,reused_token_count=212)],model_config=model,system_config=system)
    assert rows[0]['metric_source']=='SIMULATED'
    assert rows[0]['base_flops_saved']==6*212*16*16*2
