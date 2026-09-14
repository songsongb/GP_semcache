"""Read-only workload integrity, distribution and static sharing audit."""
from _paper_common import parser, load_paper_config, tokenizer_options, tokenizer_for
from semcache.experiments.workload import read_workload
from semcache.experiments.audit import validate_workload
from semcache.experiments.manifest import canonical


def main():
    p=parser(__doc__); p.add_argument('workloads',nargs='+'); tokenizer_options(p)
    a=p.parse_args(); c=load_paper_config(a.config); tokenizer=tokenizer_for(a)
    for path in a.workloads:
        rows,manifest=read_workload(path)
        if c.get('dataset') and c['dataset'] != manifest['dataset']:
            p.error('Dataset/config mismatch')
        print(canonical(dict(workload=path,report=validate_workload(rows,manifest,tokenizer,c['subsequence_window']))))


if __name__=='__main__': main()
