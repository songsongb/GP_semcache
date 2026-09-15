"""M6B-1 bounded CPU comparison; no generation or model downloads."""
from datetime import datetime, timezone
from _paper_common import parser, load_paper_config
from semcache.experiments.config import load_model_spec
from semcache.experiments.workload import read_workload
from semcache.experiments.baselines import BaselineKind, ALL_BASELINES
from semcache.experiments.baseline_runner import run_comparison, comparison_summary
from semcache.experiments.manifest import write_manifest


def main():
    p = parser(__doc__)
    p.add_argument('--workload', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--baseline', choices=[b.value for b in BaselineKind])
    group.add_argument('--all-baselines', action='store_true')
    p.add_argument('--max-queries', type=int, required=True)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--output', required=True)
    p.add_argument('--compact', action='store_true')
    p.add_argument('--qkv-precision-bits', type=int, default=None)
    a = p.parse_args()
    c = load_paper_config(a.config)
    c['system']['qkv_precision_bits'] = a.qkv_precision_bits or c['system']['qkv_precision_bits'] or 16
    rows, manifest = read_workload(a.workload)
    run_id = 'm6b1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    results = run_comparison(rows,c,load_model_spec(c,a.config.parent),workload_manifest=manifest,
        baselines=ALL_BASELINES if a.all_baselines else [a.baseline],
        max_queries=a.max_queries,seed=a.seed,run_id=run_id,compact=a.compact)
    write_manifest(a.output,dict(schema_version='semcache.comparison.v1',run_id=run_id,
        config_path=str(a.config.resolve()),results=results,summary=comparison_summary(results)))
    print(f'{run_id}: wrote {len(results)} baselines to {a.output}')


if __name__ == '__main__':
    main()
