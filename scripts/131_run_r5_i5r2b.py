#!/usr/bin/env python3
"""One manually opted-in diagnostic block through the existing I5 ES/UD pipeline."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import require
from semcache.edgelora.r5_i5r2b_plan import validate_plan

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--multi-prepared',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--runtime-budget-seconds',type=int,default=3000)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--enable-gpu-experiment',action='store_true');g.add_argument('--test-fixture-only',action='store_true')
    a=p.parse_args();plan=validate_plan(a.experiment,a.multi_prepared)
    require(a.source_commit==plan['i5r2b']['source']['commit'] and plan['TEST_FIXTURE_ONLY']==a.test_fixture_only,'diagnostic source/scope differs')
    cmd=[sys.executable,str(ROOT/'scripts/119_run_r5_i5.py'),'--experiment',str(a.experiment),'--multi-prepared',str(a.multi_prepared),
        '--output-dir',str(a.output_dir),'--source-commit',a.source_commit,'--runtime-budget-seconds',str(a.runtime_budget_seconds),
        '--enable-arithmetic-diagnostic','--test-fixture-only' if a.test_fixture_only else '--enable-gpu-experiment']
    code=subprocess.run(cmd,cwd=ROOT).returncode
    if code:return code
    from semcache.edgelora.r5_i5r2b_analysis import analyze
    report=analyze(a.experiment,a.output_dir,a.output_dir.with_name(a.output_dir.name+'_arithmetic_audit'))
    return 1 if report['GPU_EVIDENCE_COMPLETE']=='FAIL' else 0

if __name__=='__main__':sys.exit(main())
