"""M6 normalized records through the existing random tiny-OPT M5 physical engine."""
import pytest
pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('peft')
from test_mixed_semcache import make_engine
from test_lora import tiny_base
from semcache.models.lora_fixtures import create_controlled_users
from semcache.experiments.config import load_paper_config
from semcache.experiments.workload import build_workload
from semcache.experiments.runner import run_workload
from pathlib import Path


@pytest.mark.parametrize('mode',['MEASURED_MODEL','HYBRID'])
def test_prepared_records_physical_engine(tmp_path,mode):
    config=load_paper_config(Path(__file__).resolve().parents[1]/'configs/paper/snips.yaml')
    model=create_controlled_users(tiny_base())[0]
    engine=make_engine(model)
    rows,manifest=build_workload('snips',[{'id':str(i),'utterance':'2 3 4 5 6 7 8 9','intent':'fixture'} for i in range(2)],{'source_split':'test'})
    # Explicit test adapter mapping; not a paper personalization prescription.
    for r in rows: r['user_id']='user_a'
    from semcache.experiments.workload import serialize
    from semcache.experiments.manifest import sha256
    manifest['sha256']=sha256(serialize(rows))
    manifest['user_assignment_rule']='explicit test user_a mapping; REPRODUCTION_CHOICE'
    config.update(execution_mode=mode,cache_storage_mode='physical_cpu',cluster_count=2,logical_cache_capacity_gb=.0001)
    config['system'].update(es_tflops=1,ud_tflops=.1,communication_element_bytes=4)
    spec=dict(hidden_size=16,kv_dimension=16,layers=2,lora_rank=8,model_id='random-tiny-opt')
    result,run=run_workload(rows,config,spec,engine=engine,workload_manifest=manifest,run_id=mode,output_root=tmp_path,max_queries=2)
    assert result['avg_measured_prefill_latency_s']['value']>0
    assert result['avg_measured_prefill_latency_s']['metric_source']=='MEASURED'
    assert result['sum_base_flops_saved']['value']>0
    assert (result['avg_analytical_latency_s']['value'] is not None)==(mode=='HYBRID')
    assert engine.cache.physical_tensor_bytes>0
