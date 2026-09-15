"""Run qualitative logical SemCache sensitivity sweeps (no model downloads)."""
from _paper_common import parser, load_paper_config
from semcache.experiments.config import load_model_spec
from semcache.experiments.workload import read_workload
from semcache.experiments.manifest import write_manifest
from semcache.experiments.sweeps import TARGETS, run_sweep


def main():
    p = parser(__doc__)
    p.add_argument('--workload', required=True)
    p.add_argument('--sweep', choices=TARGETS, required=True)
    p.add_argument('--values', type=float, nargs='+', required=True)
    p.add_argument('--max-queries', type=int, required=True)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--output', required=True)
    p.add_argument('--compact', action='store_true')
    a = p.parse_args()
    c = load_paper_config(a.config)
    rows, manifest = read_workload(a.workload)
    result = run_sweep(rows, c, load_model_spec(c, a.config.parent), workload_manifest=manifest,
                       sweep=a.sweep, values=a.values, max_queries=a.max_queries, seed=a.seed, compact=a.compact)
    write_manifest(a.output, result)
    print(f'Wrote {len(result["points"])} qualitative sweep points to {a.output}')


if __name__ == '__main__':
    main()
