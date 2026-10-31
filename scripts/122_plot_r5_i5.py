#!/usr/bin/env python3
"""Eight thesis plot families from verified measured I5 consolidation; no synthetic points."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.r5_i5_evidence import verify,fresh,seal
from semcache.edgelora.r3a_artifacts import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    rows=[]
    for path in a.inputs:
        verify(path);rows.extend(csv.DictReader((path/'i5_plot_data.csv').open()))
    out=fresh(a.output_dir,a.inputs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs=[('prefix_ttft','prefix_length_target','application_ttft_ms'),('prefix_e2e','prefix_length_target','application_e2e_ms'),
        ('scale_latency','scale','application_e2e_ms'),('scale_hit_rate','scale','hit'),('dataset_quality','dataset','task_score'),
        ('dataset_wire','dataset','total_request_wire_bytes'),('hit_miss_latency','hit','application_e2e_ms'),
        ('quality_vs_reuse','reused_input_fraction','quality_delta_from_A')]
    artifacts=[]
    for name,x,y in specs:
        # Separate dataset/track/task metric; never pool incompatible scores or tracks.
        partitions=defaultdict(list)
        for r in rows:
            if name.startswith('prefix_') and r['track']!='CONTROLLED':continue
            partitions[(r['dataset'],r['track'],r['task_metric'],r['TEST_FIXTURE_ONLY']=='True')].append(r)
        for (dataset,track,metric,fixture),subset in partitions.items():
            fig,ax=plt.subplots(figsize=(7,4));plotted=False
            for condition in sorted({r['condition'] for r in subset}):
                rs=[r for r in subset if r['condition']==condition]
                if name=='quality_vs_reuse':
                    ax.scatter([float(r[x]) for r in rs],[float(r[y]) for r in rs],label=condition,alpha=.6);plotted=True;continue
                groups=defaultdict(list)
                for r in rs:
                    value=float(r[y]) if y!='hit' else float(r[y]=='True')
                    key={'SMOKE':10,'SMALL':100,'MEDIUM':500,'LARGE':1000}.get(r[x],r[x]) if x=='scale' else r[x]
                    if x in ('prefix_length_target','reused_input_fraction'):key=float(key)
                    groups[key].append(value)
                xs=sorted(groups);ax.scatter(xs,[statistics.mean(groups[k]) for k in xs],label=condition);plotted=True
            if not plotted:plt.close(fig);continue
            scope='CPU_FIXTURE_ONLY' if fixture else 'GPU_MEASUREMENTS'
            ax.set(xlabel=x,ylabel=metric if y=='task_score' else y,title=f'{dataset} / {track} / {scope}')
            ax.legend(fontsize=6);fig.tight_layout();file=f'{name}_{dataset}_{track}_{scope}.png';fig.savefig(out/file,dpi=180);plt.close(fig);artifacts.append(file)
    write(out/'i5_plot_manifest.json',dict(sources=[str(p.resolve()) for p in a.inputs],artifacts=artifacts,interpolation=False));seal(out)


if __name__=='__main__':main()
