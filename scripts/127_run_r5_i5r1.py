#!/usr/bin/env python3
"""One pinned R1 block through the unchanged I5 runner; no automatic submission."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import require
from semcache.edgelora.r5_i5r1_plan import validate_plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--multi-prepared',type=Path,required=True)
    p.add_argument('--block-index',type=int,default=0);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--runtime-budget-seconds',type=int,default=9000)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--enable-gpu-experiment',action='store_true');g.add_argument('--test-fixture-only',action='store_true')
    a=p.parse_args();plan=validate_plan(a.experiment,a.multi_prepared)
    require(a.source_commit==plan['i5r1']['source']['commit'],'runner commit differs from prepared protocol')
    require(plan['TEST_FIXTURE_ONLY']==a.test_fixture_only,'real/fixture scope differs')
    require(0<=a.block_index<len(plan['blocks']),'invalid block index')
    require(not a.output_dir.exists() and not a.output_dir.with_name(a.output_dir.name+'_i5r1_audit').exists(),'fresh execution/audit directories required')
    cmd=[sys.executable,str(ROOT/'scripts/119_run_r5_i5.py'),'--experiment',str(a.experiment),'--multi-prepared',str(a.multi_prepared),
        '--block-index',str(a.block_index),'--source-commit',a.source_commit,'--runtime-budget-seconds',str(a.runtime_budget_seconds),
        '--output-dir',str(a.output_dir),'--test-fixture-only' if a.test_fixture_only else '--enable-gpu-experiment']
    result=subprocess.run(cmd,cwd=ROOT)
    if result.returncode:return result.returncode
    from semcache.edgelora.r5_i5r1_analysis import summarize
    report=summarize(a.experiment,[a.output_dir],a.output_dir.with_name(a.output_dir.name+'_i5r1_audit'),block_only=True)
    # A scientific numerical failure remains in the report; it is not an evidence error.
    return 0 if report['functional_assertions']=='PASS' else 2


if __name__=='__main__':sys.exit(main())
