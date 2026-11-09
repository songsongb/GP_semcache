#!/usr/bin/env python3
"""Phase 2.5A CPU safety tests and sealed Phase 2 static extraction. No GPU action."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))

REQUIRED=('test_r5_layer_input_verified.Verified','test_r5_layer_input_audit.Audit')
REGRESSION=('test_r5_subsequence_natural','test_r5_sparse_physical','test_r5_subsequence_generation',
            'test_r5_subsequence_validation','test_r5_subsequence_audit','test_r5_mstage_bridge',
            'test_r5_semantic','test_r5_continuous_natural','test_r5_continuous_natural_phase2')


def flatten(suite):
    for test in suite:
        if isinstance(test,unittest.TestSuite):yield from flatten(test)
        else:yield test


def run_tests(out,name,modules):
    tests=list(flatten(unittest.defaultTestLoader.loadTestsFromNames(modules)))
    # The historical job-shell guard test executes a Slurm job shell before
    # rejecting the absent allocation. This task authorizes no job execution.
    excluded=[t.id() for t in tests if 'test_shell_requires_manual_allocation_before_execution' in t.id()]
    tests=[t for t in tests if t.id() not in excluded]
    with (out/(name+'.log')).open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.TestSuite(tests))
    return dict(tests=result.testsRun,passed=result.testsRun-len(result.errors)-len(result.failures)-len(result.skipped),
                failed=len(result.errors)+len(result.failures),skipped=[dict(test=t.id(),reason=r) for t,r in result.skipped],
                excluded_job_execution_tests=excluded,selection=list(modules))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('cpu-validate',))
    p.add_argument('--phase2-plan',default=str(ROOT/'results/r5/i5r/d5_natural/continuous_phase2_seraph_plan_02'))
    p.add_argument('--phase2-audit',default=str(ROOT/'results/r5/i5r/d5_natural/continuous_phase2_seraph_cpu_02'))
    p.add_argument('--output-dir',required=True);p.add_argument('--require-real-data',action='store_true')
    args=p.parse_args(argv)
    from semcache.edgelora.communication import require
    from semcache.edgelora.r5_subsequence_natural_io import output,put,seal,verify
    from semcache.edgelora.r5_i5r2_tensors import file_sha
    from semcache.edgelora.r5_layer_input_audit import static_feasibility
    from semcache.edgelora.r5_layer_input_physical import MODE
    try:
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for CPU validation')
        protected=[args.phase2_plan,args.phase2_audit]
        plan_path=Path(args.phase2_plan)/'phase2_plan.json'
        if plan_path.is_file():
            require(plan_path.stat().st_size<=16*2**20,'oversized source plan')
            source=json.loads(plan_path.read_bytes())
            protected += [source[k] for k in ('original_experiment','multi','phase1_audit')]
            protected += list(source['dataset_preparations'].values())
        out=output(args.output_dir,protected)
        required=run_tests(out,'new_operator_tests',REQUIRED)
        regressions=run_tests(out,'regressions',REGRESSION)
        native=run_tests(out,'native_cpu_contract',('test_r5_layer_input_native.Native',))
        ok=required['failed']==0 and not required['skipped'] and regressions['failed']==native['failed']==0
        record=dict(schema='D5_PHASE25A_CPU_V1',mode=MODE,
            implementation='IMPLEMENTATION_PASS' if ok else 'FAIL',cpu_safety='CPU_SAFETY_TESTS_PASS' if ok else 'FAIL',
            tests=dict(required=required,regressions=regressions,native_cpu=native),
            native_cpu_contract='NOT_RUN_DEPENDENCIES_ABSENT' if native['skipped'] else 'PASS' if not native['failed'] else 'FAIL',
            gpu_numerical_validation='GPU_NUMERICAL_VALIDATION_NOT_RUN',gpu_e2e='GPU_E2E_NOT_RUN',
            gpu_pilot_readiness='SPECIFICATION_ONLY; no GPU approval; resource/evidence and native numerical validation required',
            native_projection_work_saving='ZERO_WITH_FULL_SHAPE_REFERENCE_GUARD',
            fixture_counts_are_not_natural_results=True)
        try:
            feasibility=static_feasibility(args.phase2_plan,args.phase2_audit)
        except (ValueError,KeyError,FileNotFoundError) as exc:
            feasibility=dict(status='FAIL',reason=str(exc))
        record['real_data_static_analysis']=feasibility['status']
        put(out/'nonprefix_eligibility.json',feasibility)
        files=['src/semcache/edgelora/'+n for n in ('r5_layer_input_physical.py','r5_layer_input_arithmetic.py','r5_layer_input_audit.py',
            'r5_subsequence_physical.py','r5_mstage_bridge.py','r5_subsequence_generation.py','r5_subsequence_integrity.py',
            'r5_sparse_qkv.py','r5_continuous_natural.py','r5_continuous_natural_phase2.py')]
        files += ['scripts/156_r5_layer_input_verified.py','tests/test_r5_layer_input_verified.py','tests/test_r5_layer_input_audit.py','tests/test_r5_layer_input_native.py']
        record['source_files']={f:file_sha(ROOT/f) for f in files}
        put(out/'phase25a_validation.json',record)
        lines=['# D5 Phase 2.5A CPU result','',record['implementation'],record['cpu_safety'],
               record['real_data_static_analysis'],record['gpu_numerical_validation'],record['gpu_e2e'],'',
               'Native CPU contract: '+record['native_cpu_contract'],
               'Native cached-row consumption uses paid full-shape output verification; projection work skipped = 0.',
               'Required portable tests: '+str(required['passed'])+' passed; '+str(required['failed'])+' failed.',
               'Existing regressions: '+str(regressions['passed'])+' passed; '+str(regressions['failed'])+' failed; '+str(len(regressions['skipped']))+' skipped (listed in JSON).',
               'Actual NATURAL residency, layer-input equality, QKV reuse and all GPU numerical/latency outcomes remain NOT_MEASURED.']
        if feasibility['status']=='BLOCKED':lines+=['','Real-data blocker: '+feasibility['reason']]
        for d in feasibility.get('datasets',[]):
            lines+=['','## '+d['dataset'],'','All values are static Phase 2 token candidates, not input-equality or physical grants.']
            for scope,counts in d['scopes'].items():lines+=['',scope+': '+json.dumps(counts,sort_keys=True)]
        (out/'phase25a_report.md').write_text('\n'.join(lines)+'\n')
        seal(out);verify(out)
        print(json.dumps({k:v for k,v in record.items() if k not in ('tests','source_files')},indent=2))
        return 0 if ok and feasibility['status']!='FAIL' and (not args.require_real_data or feasibility['status']=='REAL_DATA_STATIC_ANALYSIS_PASS') else 2
    except (ValueError,KeyError,FileNotFoundError) as exc:
        print(json.dumps(dict(status='FAIL',reason=str(exc))),file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
