"""Development output effects; no universal reuse threshold inferred."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--input', type=Path, default=root/'results/raw/qkv_reuse_output_impact.csv')
    parser.add_argument('--output', type=Path, default=root/'results/figures/qkv_reuse_output_impact.png')
    args = parser.parse_args()
    with args.input.open() as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r['probe_case'] in ('B', 'C') and r['injection_mode'] == 'qkv']
    if not rows or any(r['metric_source'] != 'measured' for r in rows):
        raise ValueError('Need measured B/C qkv rows')
    if len({(r['model_id'], r['resolved_model_revision'], r['dtype'], r['seed']) for r in rows}) != 1:
        raise ValueError('Plot one model/revision/dtype/seed at a time')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for case in ('B', 'C'):
        selected = sorted((r for r in rows if r['probe_case'] == case), key=lambda r: int(r['layer']))
        for ax, metric, label in zip(axes, ('affected_suffix_mean_kl', 'relative_l2_logit_diff'),
                                    ('Suffix mean KL (baseline || injected)', 'Relative L2 logit difference')):
            ax.plot([int(r['layer']) for r in selected], [float(r[metric]) for r in selected], marker='o', label=case)
            ax.set(xlabel='Transformer layer', ylabel=label)
            ax.legend(title='Case')
    fig.suptitle(f"{rows[0]['model_id']}: controlled development QKV injection; no safety claim")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
