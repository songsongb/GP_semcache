"""Plot measured JSON aggregates from qualitative logical SemCache sweeps."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_sweep(data, output):
    sweep = data['sweep']
    points = sorted(data['points'], key=lambda p: p['value'])
    fields = {'cache_size': [('token_reuse_ratio', 'Token reuse ratio'), ('eviction_count', 'Evictions'), ('final_cache_bytes', 'Final cache (GB)')],
              'admission_threshold': [('token_reuse_ratio', 'Token reuse ratio'), ('admission_rate', 'Admission rate'), ('final_cache_bytes', 'Final cache (GB)')],
              'bandwidth': [('communication_time_s', 'Analytical communication time (s)')] if all(p['communication_time_available'] for p in points) else [('communication_elements', 'Communication volume (elements)')]}[sweep]
    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 4), squeeze=False)
    for ax, (key, label) in zip(axes[0], fields):
        ax.plot([p['value'] for p in points], [None if p[key] is None else p[key] / (1e9 if key == 'final_cache_bytes' else 1) for p in points], marker='o')
        ax.set_xlabel({'cache_size': 'Cache capacity (GB)', 'admission_threshold': 'Admission threshold θ', 'bandwidth': 'Bandwidth (Mbps)'}[sweep])
        ax.set_ylabel(label)
        ax.grid(alpha=.3)
    fig.suptitle('Qualitative logical SemCache simulation — reproduction choices')
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    plot_sweep(json.loads(a.input.read_text()), a.output)


if __name__ == '__main__':
    main()
