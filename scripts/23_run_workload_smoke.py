"""Logical M5-engine workload consumption. This is not Table II."""
from copy import deepcopy
from datetime import datetime, timezone
from _paper_common import parser, load_paper_config, tokenizer_options, tokenizer_for
from semcache.experiments.config import load_model_spec
from semcache.experiments.workload import read_workload
from semcache.experiments.runner import make_logical_engine, run_workload
from semcache.experiments.manifest import canonical


def main():
    p=parser(__doc__)
    p.add_argument('--workload',required=True)
    p.add_argument('--max-queries',required=True,type=int)
    p.add_argument('--encoder',choices=['fixture','real'],default='fixture')
    p.add_argument('--output-root',default='results')
    p.add_argument('--run-id')
    p.add_argument('--qkv-precision-bits',type=int,default=None,help='Explicit smoke choice: FP16 logical activation payload; not inferred from weight precision')
    tokenizer_options(p)
    a=p.parse_args(); c=deepcopy(load_paper_config(a.config))
    if c['execution_mode']!='ANALYTICAL_SIMULATION' or c['cache_storage_mode']!='logical_only':
        p.error('CLI supports logical analytical smoke; measured/hybrid use injected M5 engine runner interface')
    if a.max_queries < 1: p.error('--max-queries must be positive')
    rows,manifest=read_workload(a.workload)
    if manifest['user_count'] != c['num_users']:
        p.error('Workload/config user counts differ')
    c['system']['qkv_precision_bits']=(a.qkv_precision_bits if a.qkv_precision_bits is not None else c['system']['qkv_precision_bits'] or 16)
    c['provenance']['smoke_qkv_precision_bits']='REPRODUCTION_CHOICE'
    model=load_model_spec(c,a.config.parent)
    engine=make_logical_engine(rows[:a.max_queries],c,a.encoder,a.allow_download,tokenizer_for(a))
    run_id=a.run_id or 'm6a_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    result,manifest=run_workload(rows,c,model,engine=engine,workload_manifest=manifest,
        run_id=run_id,output_root=a.output_root,max_queries=a.max_queries)
    print(canonical(dict(run_id=run_id,aggregate=result,execution_mode=manifest['execution_mode'])))


if __name__=='__main__': main()
