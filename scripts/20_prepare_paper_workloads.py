"""Prepare one explicitly selected real source; no automatic download/substitute."""
from _paper_common import parser, load_paper_config
from semcache.experiments.dataset_adapters import load_source
from semcache.experiments.workload import build_workload, save_workload
from semcache.experiments.manifest import canonical


def main():
    p=parser(__doc__)
    p.add_argument('--dataset',required=True,choices=['multiwoz','coqa','snips'])
    p.add_argument('--input-path')
    p.add_argument('--split')
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int)
    p.add_argument('--max-queries',type=int)
    p.add_argument('--order',choices=['source_order','seeded_shuffle'])
    p.add_argument('--user-count',type=int)
    p.add_argument('--transformation',choices=['raw_query','paper_reproduction_v1'])
    p.add_argument('--user-assignment',choices=['seeded_round_robin','deterministic_hash'])
    p.add_argument('--hf-id')
    p.add_argument('--hf-config')
    p.add_argument('--revision')
    p.add_argument('--allow-download',action='store_true')
    a=p.parse_args(); c=load_paper_config(a.config)
    if c.get('dataset',a.dataset) != a.dataset:
        p.error('Dataset/config mismatch')
    s=c['dataset_source']
    try:
        examples,source=load_source(a.dataset,a.input_path or s['input_path'],a.split or s['split'],
            hf_id=a.hf_id or s['hf_id'],hf_config=a.hf_config or s['hf_config'],revision=a.revision or s['revision'],allow_download=a.allow_download)
        rows,manifest=build_workload(a.dataset,examples,source,seed=a.seed if a.seed is not None else c['seed'],
            user_count=a.user_count if a.user_count is not None else c['num_users'],
            assignment=a.user_assignment or c['user_assignment']['mode'],order=a.order or c['workload']['order'],
            transformation=a.transformation or c['workload']['transformation'],
            max_queries=a.max_queries if a.max_queries is not None else c['workload']['max_queries'])
        save_workload(a.output,rows,manifest)
    except (ValueError,FileNotFoundError,RuntimeError,KeyError) as exc:
        p.error(str(exc))
    print(canonical(manifest))


if __name__=='__main__': main()
