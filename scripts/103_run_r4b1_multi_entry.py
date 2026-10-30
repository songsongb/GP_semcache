#!/usr/bin/env python3
"""R4-B1 bounded multi-entry exact full-prompt Full E2E diagnostic."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import require
from semcache.edgelora.r3a_artifacts import now,read,sha,source_identity,write
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.parity_forensics_audit import Evidence
from semcache.edgelora.r4a_evidence import load_prepared,verify_sources
from semcache.edgelora.r4b1_analysis import analyze,export
from semcache.edgelora.r4b1_evidence import audit_run,flags,load_run,schedule


def frozen_runner():
    spec=importlib.util.spec_from_file_location('r4b1_frozen_r4a_cli',ROOT/'scripts/102_run_r4a_opt_full_e2e.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def sources():
    result=frozen_runner().sources()
    paths=[Path(__file__).resolve(),ROOT/'jobs/r4b1_opt_multi_entry.sh',ROOT/'docs/R4B1_MULTI_ENTRY_FULL_E2E.md',
        *sorted((ROOT/'src/semcache/edgelora').glob('r4b1_*.py'))]
    result.update({str(p.relative_to(ROOT)):sha(p) for p in paths});return result


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('dry-run','fixture','prepare','run','analyze'));p.add_argument('--output-dir',type=Path)
    p.add_argument('--base-prepared',type=Path);p.add_argument('--prepared',type=Path);p.add_argument('--inputs',type=Path)
    p.add_argument('--gold-jsonl',type=Path);p.add_argument('--capacity-bytes',type=int,default=64*2**20)
    p.add_argument('--device',default='cuda:0');p.add_argument('--enable-gpu-experiment',action='store_true');p.add_argument('--test-fixture-only',action='store_true')
    p.add_argument('--worker',choices=('es','ud'),help=argparse.SUPPRESS);p.add_argument('--plan',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.worker:
        if not a.plan or a.enable_gpu_experiment==a.test_fixture_only:p.error('worker requires plan and explicit scope')
        return a
    if not a.phase or not a.output_dir:p.error('--phase and --output-dir required')
    if not 0<a.capacity_bytes<=2**30:p.error('--capacity-bytes must be in (0, 1 GiB]')
    if a.phase in ('prepare','run') and (not a.enable_gpu_experiment or a.test_fixture_only or a.device!='cuda:0'):p.error('GPU phases require explicit allocated cuda:0 opt-in')
    if a.phase=='fixture' and (not a.test_fixture_only or a.enable_gpu_experiment):p.error('fixture requires --test-fixture-only')
    if a.phase in ('dry-run','analyze') and (a.test_fixture_only or a.enable_gpu_experiment):p.error('read-only phases execute no model')
    if (a.base_prepared is not None)!=(a.phase=='prepare'):p.error('--base-prepared required only for prepare')
    if (a.prepared is not None)!=(a.phase=='run'):p.error('--prepared required only for run')
    if (a.inputs is not None)!=(a.phase=='analyze'):p.error('--inputs required only for analyze')
    if a.gold_jsonl and a.phase!='run':p.error('--gold-jsonl is run-only')
    return a


def output_dir(path,protected):
    path=path.resolve();base=ROOT/'results/r4/b1'
    require(path.is_relative_to(base) and path!=base,'new results/r4/b1 output required')
    require(all(p is None or not path.is_relative_to(p.resolve()) for p in protected),'output cannot alter inputs')
    require(not path.exists() or path.is_dir() and not any(path.iterdir()),'refusing nonempty output')
    path.mkdir(parents=True,exist_ok=True);return path


def gpu_preflight():
    require(source_identity()['git_branch']=='exp/raw-semcache-r4b','R4-B1 requires exp/raw-semcache-r4b')
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') and
        len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1,'one explicit Slurm GPU allocation required')
    missing=[n for n in ('torch','numpy','transformers','peft','sacrebleu') if importlib.util.find_spec(n) is None]
    require(not missing,'MISSING_DEPENDENCY: '+', '.join(missing)+'; no installation/fallback')
    require(os.environ.get('CUDA_LAUNCH_BLOCKING')!='1','CUDA_LAUNCH_BLOCKING invalidates timing')


def worker_main(a):
    plan=read(a.plan);out=Path(plan['output']);result={}
    try:
        verify_sources(plan['sources']);require(plan['fixture']==a.test_fixture_only,'worker scope differs')
        if a.worker=='ud':
            from semcache.edgelora.r4a_dynamic_ud import serve
            serve(plan,out);result=read(out/'ud_result.json')
        else:
            from semcache.edgelora.r4b1_runtime import execute
            result=execute(plan,out)
    except BaseException as exc:
        result=dict(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());traceback.print_exc()
    result.update(pid=os.getpid(),finished_at=now());result.update(flags(plan['fixture']));write(out/(a.worker+'_result.json'),result)
    return 0 if result['execution_status']=='COMPLETE' else 1


def main(argv=None):
    a=parse_args(argv)
    if a.worker:return worker_main(a)
    try:out=output_dir(a.output_dir,(a.base_prepared,a.prepared,a.inputs))
    except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);return 1
    fixture=a.phase=='fixture';report=dict(execution_status='RUNNING',phase=a.phase,created_at=now(),sources=sources(),
        production_source_identity=source_identity(),gpu_inference_performed=False,**flags(fixture))
    decision=dict(execution_status='NOT_RUN',R4B1_PHYSICAL_E2E_PASS=False,R4B1_MULTI_ENTRY_RESIDENCY_PASS=False,
        R4B1_RAW_QKV_E2E_FIDELITY_PASS=False,NATIVE_PEFT_STRICT_PARITY_PASS=False,native_equivalent=False,**flags(fixture))
    evidence=Evidence();quality=dict(status='NOT_RUN');signatures={}
    try:
        for name,h in report['sources'].items():evidence.file(ROOT/name,h)
        write(out/'r4b1_preflight.json',dict(branch=report['production_source_identity']['git_branch'],capacity_bytes=a.capacity_bytes,
            missing_dependencies=[n for n in ('torch','numpy','transformers','peft','sacrebleu') if importlib.util.find_spec(n) is None],
            native_parity_deferred=True,downloads_performed=False))
        if a.phase=='dry-run':pass
        elif a.phase=='analyze':
            previous,plan,worker=load_run(a.inputs);fixture=plan['fixture'];report.update(flags(fixture))
            evidence.file(a.inputs/'r4b1_results.json');rows=jsonl(a.inputs/'es/r4b1_requests.jsonl');tokens=jsonl(a.inputs/'es/r4b1_tokens.jsonl')
            gold=read(a.inputs/'r4b1_gold.json') if (a.inputs/'r4b1_gold.json').exists() else None
            decision,quality,signatures=analyze(a.inputs/'es',Path(plan['prepared']),rows,tokens,worker,fixture=fixture,gold=gold)
            report.update(analysis_only=True,input_directory=str(a.inputs.resolve()),audit=previous['audit'],gpu_inference_performed=False)
        else:
            legacy=frozen_runner();runner=legacy.legacy_runner().process_runner()
            if not fixture:gpu_preflight()
            if a.phase in ('prepare','fixture'):
                if fixture:
                    from semcache.edgelora.r3a_tiny import prepare_tiny
                    base=prepare_tiny(out/'base_prepare')
                else:base=a.base_prepared.resolve()
                prepare_plan=dict(base_prepared=str(base),fixture=fixture,device='cpu' if fixture else a.device,scope='pilot',policy='fixed',sources=report['sources'])
                legacy.prepare_artifacts(prepare_plan,out/'native',runner)
                report['prepared_directory']=str(out/'native')
            if a.phase in ('run','fixture'):
                prepared=(out/'native') if fixture else a.prepared.resolve();m,suite,pm,ref,am=load_prepared(prepared)
                require(m['TEST_FIXTURE_ONLY']==fixture and m['sources']==report['sources'],'prepared B1 scope/source identity differs')
                require(len(suite['cells'])==1 and suite['cells'][0]['length']==32,'B1 is a bounded L32 pilot only')
                plan=dict(prepared=str(prepared),base_prepared=m['base_prepared'],fixture=fixture,device='cpu' if fixture else a.device,
                    sources=report['sources'],capacity_bytes=a.capacity_bytes)
                es=out/'es';es.mkdir();write(es/'es_plan.json',dict(plan,output=str(es)))
                runner([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','es','--plan',str(es/'es_plan.json'),
                    '--test-fixture-only' if fixture else '--enable-gpu-experiment'],es,timeout=7500.,env=legacy.offline_env())
                worker=read(es/'es_result.json');report['audit']=audit_run(out,plan,worker);report['model_result']=worker
                rows=jsonl(es/'r4b1_requests.jsonl');tokens=jsonl(es/'r4b1_tokens.jsonl');gold=None
                if a.gold_jsonl:evidence.file(a.gold_jsonl);gold=jsonl(a.gold_jsonl);write(out/'r4b1_gold.json',gold)
                decision,quality,signatures=analyze(es,prepared,rows,tokens,worker,fixture=fixture,gold=gold)
                report['gpu_inference_performed']=not fixture
        evidence.unchanged();report['execution_status']='COMPLETE'
    except BaseException as exc:
        report.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());traceback.print_exc()
        if (out/'es/es_result.json').exists():report['failed_worker']=read(out/'es/es_result.json')
        decision=dict(execution_status='FAILED',reason=report['reason'],R4B1_PHYSICAL_E2E_PASS=False,
            R4B1_MULTI_ENTRY_RESIDENCY_PASS=False,R4B1_RAW_QKV_E2E_FIDELITY_PASS=False,
            NATIVE_PEFT_STRICT_PARITY_PASS=False,native_equivalent=False,**flags(fixture))
    finally:
        export(out,decision,quality,signatures)
        if (out/'es').exists():
            for name in ('r4b1_requests.jsonl','r4b1_tokens.jsonl','r4b1_generation_outputs.jsonl','r4b1_cache_events.jsonl','r4b1_parity_diagnostics.json'):
                if (out/'es'/name).exists():(out/name).write_bytes((out/'es'/name).read_bytes())
        report.update(finished_at=now(),decision=decision,input_hashes=list(evidence.files.values()))
        if any(read(p).get('gpu_inference_performed') is True for p in out.rglob('gpu_execution.json')):
            report['gpu_inference_performed']=True
        report['gpu_findings_status']=('INCOMPLETE_FAILED' if report['execution_status']=='FAILED' else 'MEASURED') \
            if report['gpu_inference_performed'] else 'NOT_RUN'
        report['output_hashes']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('r4b1_results.json','r4b1_summary.txt')}
        write(out/'r4b1_results.json',report)
        (out/'r4b1_summary.txt').write_text('\n'.join([report['classification'],'Execution: '+report['execution_status'],
            'Reason: '+report.get('reason','none'),'GPU execution: '+str(report['gpu_inference_performed']),
            'Native strict parity: '+str(decision['NATIVE_PEFT_STRICT_PARITY_PASS']),
            'PRIMARY_PERFORMANCE_ELIGIBLE=False; correctness_qualified_speedup=False; exact full prompts only.'])+'\n')
    print(json.dumps(dict(report=str(out/'r4b1_results.json'),execution_status=report['execution_status'])))
    return 0 if report['execution_status']=='COMPLETE' else 1


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum));sys.exit(main())
