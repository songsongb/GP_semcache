#!/usr/bin/env python3
"""R3-A isolated native preparation and physical localhost ES/UD no-reuse prefill."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.collaborative_inference import prepare,trial,analyze


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('prepare','pilot','latency','analyze'),required=True)
    p.add_argument('--model',choices=('facebook/opt-2.7b','facebook/opt-125m'))
    p.add_argument('--dtype',choices=('float16',))
    p.add_argument('--prompt-lengths')
    p.add_argument('--device')
    p.add_argument('--adapter',choices=('user_a',))
    p.add_argument('--seed',type=int)
    p.add_argument('--prepared',type=Path)
    p.add_argument('--pilot-inputs',type=Path)
    p.add_argument('--inputs',type=Path)
    p.add_argument('--warmup-runs',type=int)
    p.add_argument('--measured-runs',type=int)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    if a.phase=='prepare':
        if any(v is not None for v in (a.prepared,a.pilot_inputs,a.inputs,a.warmup_runs,a.measured_runs)):
            p.error('prepare does not consume trial artifacts/counts')
        a.model=a.model or 'facebook/opt-2.7b';a.dtype=a.dtype or 'float16';a.prompt_lengths=a.prompt_lengths or '32'
        a.adapter=a.adapter or 'user_a';a.seed=42 if a.seed is None else a.seed;a.device=a.device or 'cuda:0'
        if a.prompt_lengths!='32' or not a.device.startswith('cuda'): p.error('prepare requires CUDA, batch-one length-32 FP16 prefill')
    else:
        if not a.prepared: p.error('--prepared required')
        if any(v is not None for v in (a.model,a.dtype,a.prompt_lengths,a.adapter,a.seed)):
            p.error('model/dtype/input/adapter identity comes from --prepared')
        if a.phase in ('latency','analyze') and not a.pilot_inputs: p.error('matching --pilot-inputs required')
        if a.phase=='analyze':
            if not a.inputs or any(v is not None for v in (a.device,a.warmup_runs,a.measured_runs)):
                p.error('analyze requires --inputs and no runtime/count flags')
        else:
            if a.inputs or a.phase=='pilot' and a.pilot_inputs: p.error('unexpected phase inputs')
            a.device=a.device or 'cuda:0'
            a.warmup_runs=a.warmup_runs if a.warmup_runs is not None else (1 if a.phase=='pilot' else 3)
            a.measured_runs=a.measured_runs if a.measured_runs is not None else (3 if a.phase=='pilot' else 10)
            if not (1 if a.phase=='pilot' else 3)<=a.warmup_runs<=1000 or not (3 if a.phase=='pilot' else 10)<=a.measured_runs<=1000:
                p.error('pilot needs >=1 warmup/3 measured; latency needs >=3 warmup/10 measured; counts <=1000')
    return a


def main(argv=None):
    a=parse_args(argv);output=a.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        print('R3-A requires a new or empty output directory: '+str(output),file=sys.stderr);return 1
    if output==ROOT or any(p is not None and output==p.resolve() for p in (a.prepared,a.pilot_inputs,a.inputs)):
        print('output cannot replace repository/input directory',file=sys.stderr);return 1
    output.mkdir(parents=True,exist_ok=True)
    plain=SimpleNamespace(**{k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()})
    try:
        if a.phase=='prepare': prepare(plain,output)
        elif a.phase=='analyze': analyze(plain,output)
        else: trial(plain,output)
    except BaseException as exc:
        print(f'R3-A FAILED: {type(exc).__name__}: {exc}; evidence: {output}',file=sys.stderr)
        return 1
    print(f'R3-A {a.phase} complete: {output}',flush=True)
    return 0


if __name__=='__main__': sys.exit(main())
