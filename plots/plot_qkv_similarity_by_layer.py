"""Plot raw per-case layer curves, preserving the full cosine range."""
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=Path('results/raw/qkv_cross_query_probe.csv'))
parser.add_argument('--output-dir', type=Path, default=Path('results/figures'))
args = parser.parse_args()
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit('Missing dependency: matplotlib. Install: python3 -m pip install matplotlib')
with args.input.open() as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit('Input contains no measurements')
args.output_dir.mkdir(parents=True, exist_ok=True)
for tensor in ('Q','K','V'):
    fig, ax = plt.subplots()
    groups = sorted({(r['model_id'], r['model_revision'], r['dtype'], r['query_pair'], r['probe_case']) for r in rows if r['tensor_type'] == tensor})
    for group in groups:
        selected = sorted([r for r in rows if r['tensor_type'] == tensor and tuple(r[k] for k in ('model_id','model_revision','dtype','query_pair','probe_case')) == group], key=lambda r:int(r['layer']))
        ax.plot([int(r['layer']) for r in selected], [float(r['cosine_similarity']) for r in selected], marker='.', label=f'{group[0]} {group[2]} case {group[4]}')
    ax.set(xlabel='Transformer layer', ylabel='Cosine similarity', ylim=(-1.05,1.05), title=f'Base OPT {tensor} cross-query similarity')
    ax.legend(fontsize='small')
    fig.tight_layout()
    fig.savefig(args.output_dir / f'{tensor.lower()}_cosine_similarity_by_layer.png', dpi=160)
    plt.close(fig)
