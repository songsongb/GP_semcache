#!/usr/bin/env python3
"""R5-I3P: attributed CPU fixture or explicitly opted-in balanced RTX3090 blocks."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import digest_bytes,require
from semcache.edgelora.r3a_artifacts import read,write,sha,source_identity
from semcache.edgelora.r4a_evidence import verify_sources
from semcache.edgelora.r5_evidence import fresh,seal
from semcache.edgelora.r5_perf_schedule import schedule,TINYBERT_REVISION
from semcache.edgelora.r5_perf_evidence import Writer,accounted_modules,spawn,finish
from semcache.edgelora.r5_perf_analysis import flags


def sources():
    paths=[*sorted((ROOT/'src').rglob('*.py')),ROOT/'configs/r5_integration.json',
        *[ROOT/'scripts'/n for n in ('110_run_r5_semcache_e2e.py','112_run_r5_balanced_benchmark.py','113_analyze_r5_balanced_benchmark.py')],
        ROOT/'jobs/r5_opt_balanced_benchmark.sh']
    return {str(p.relative_to(ROOT)):digest_bytes(p.read_bytes()) for p in paths if p.is_file()}


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('dry-run','diagnose','fixture','run','analyze'));p.add_argument('--output-dir',type=Path)
    p.add_argument('--test-fixture-only',action='store_true');p.add_argument('--enable-gpu-experiment',action='store_true')
    p.add_argument('--inputs',type=Path);p.add_argument('--historical-inputs',type=Path)
    p.add_argument('--base-prepared',type=Path);p.add_argument('--source-commit');p.add_argument('--semantic-revision',default=TINYBERT_REVISION)
    p.add_argument('--repetitions',type=int,default=1);p.add_argument('--start-order',choices=('ABBA','BAAB'),default='ABBA');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--generation-budget',type=int,default=2);p.add_argument('--generation-mode',choices=('fixed','quality'),default='fixed')
    p.add_argument('--evidence-profile',choices=('STRICT_AUDIT','SCALABLE_PERFORMANCE'),default='SCALABLE_PERFORMANCE')
    p.add_argument('--block-budget-bytes',type=int,default=2**30);p.add_argument('--reconcile-every',type=int,default=16)
    p.add_argument('--warmup-requests',type=int,default=1);p.add_argument('--legacy-scan',action='store_true')
    p.add_argument('--worker',choices=('es','ud'),help=argparse.SUPPRESS);p.add_argument('--plan',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.worker:
        if not a.plan:p.error('--plan required')
        return a
    if not a.phase or not a.output_dir:p.error('--phase and --output-dir required')
    if a.test_fixture_only!=(a.phase=='fixture'):p.error('fixture requires explicit --test-fixture-only')
    if a.enable_gpu_experiment!=(a.phase=='run'):p.error('run requires --enable-gpu-experiment')
    if a.phase=='run' and not a.base_prepared:p.error('--base-prepared required')
    if (a.inputs is not None)!=(a.phase=='analyze'):p.error('--inputs belongs to analyze')
    if not 1<=a.generation_budget<=16 or not 0<=a.warmup_requests<=2 or not 1<=a.reconcile_every<=128 or not 96*2**20<=a.block_budget_bytes<=16*2**30:p.error('bounded budgets/interval/warmup required')
    schedule(a.repetitions,a.seed,a.start_order)
    return a


def preflight(a,order):
    require(os.environ.get('R5_ENABLE_GPU')=='1' and os.environ.get('R5P_ENABLE_GPU')=='1','explicit R5_ENABLE_GPU=1 and R5P_ENABLE_GPU=1 required')
    require(a.semantic_revision==TINYBERT_REVISION,'previously verified immutable TinyBERT revision required')
    # Reuse the validated R5 live allocation/identity/offline checks.
    spec=importlib.util.spec_from_file_location('r5_validated_launcher',ROOT/'scripts/110_run_r5_semcache_e2e.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    a.evidence_budget_bytes=a.block_budget_bytes;m.gpu_preflight(a)
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    require(set(sources()).issubset(tracked),'commit all R5 performance execution sources before GPU run')
    require(shutil.disk_usage(a.output_dir).free>len(order['blocks'])*a.block_budget_bytes+2**30,'insufficient storage for worst-case block budgets plus setup/reports')


def worker(a):
    plan=read(a.plan);root=Path(plan['account_root']);writer=Writer(root,plan['block_budget_bytes']);out=Path(plan['output'])
    try:
        verify_sources(plan['sources'])
        if not plan['fixture']:require(os.environ.get('R5P_ENABLE_GPU')=='1' and os.environ.get('R5_ENABLE_GPU')=='1','GPU worker opt-ins absent')
        if a.worker=='ud':
            from semcache.edgelora import r4c_dynamic_ud as ud
            original=ud.ForwardOrder
            class ObservedOrder(original):
                def __init__(self,*args,**kwargs):
                    super().__init__(*args,**kwargs)
                    writer.write(out/'ud_order_initial.json',dict(grants=list(self.grants),parents=list(self.parents),finished=list(self.finished),layer=self.layer,current=self.current))
            ud.ForwardOrder=ObservedOrder
            with accounted_modules(writer):ud.serve(plan,out)
            writer.write(out/'ud_accounting.json',dict(writer_io_ms=writer.io_ms,scope='all UD metadata/trace writes; CPU LoRA compute recorded separately'))
        else:
            from semcache.edgelora.r5_perf_runtime import execute
            with accounted_modules(writer):execute(plan,root,writer)
        verify_sources(plan['sources']);return 0
    except BaseException as exc:
        # The root launcher preserves stderr and failure stage even when a
        # failed budget ledger correctly refuses further evidence writes.
        traceback.print_exc();return 1


def historical(path):
    if path is None or not path.is_dir():return dict(status='HISTORICAL_GPU_EVIDENCE_UNAVAILABLE_LOCALLY',
        seraph_path='/data/khuss/repos/GP_semcache/results/r5/opt_152301_20261031T074321Z/',
        required=['es/r5_tokens.jsonl','r5_generation_outputs.jsonl','es/es_plan.json','r5_output_manifest.json',
            'r5_identity.json','r5_requests.jsonl','r5_latency_summary.json'],causal_attribution_confirmed=False)
    from semcache.edgelora.r3a_evidence import jsonl
    steps=jsonl(path/'es/r5_tokens.jsonl');by={}
    for s in steps:
        r=by.setdefault(s['request_id'],dict(host_ms=0.,wall_ms=0.,outside_host_ms=0.))
        r['host_ms']+=s['host_forward_and_evidence_ms'];r['wall_ms']+=s['wall_ms'];r['outside_host_ms']+=s['wall_ms']-s['host_forward_and_evidence_ms']
    return dict(status='HISTORICAL_LEDGER_ONLY',requests=by,causal_attribution_confirmed=False,
        limitation='original run did not directly time recursive checks; wall-host residual cannot identify a single cause')


def main(argv=None):
    a=parse(argv)
    if a.worker:return worker(a)
    try:
        candidate=a.output_dir.resolve()
        require(not candidate.is_relative_to(ROOT) or candidate.is_relative_to(ROOT/'results/r5/performance'),'performance outputs must be under results/r5/performance')
        out=fresh(a.output_dir,(a.inputs,a.base_prepared,a.historical_inputs))
    except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);return 1
    result=dict(execution_status='RUNNING',stage='PREFLIGHT',GPU_NOT_RUN=True,source_identity=source_identity());decision=flags()
    try:
        order=schedule(a.repetitions,a.seed,a.start_order)
        if a.phase=='analyze':
            from semcache.edgelora.r5_perf_analysis import analyze,export
            report=analyze(a.inputs);export(out,report);decision=report['decision'];result['analysis_only']=True
        elif a.phase in ('diagnose','dry-run'):
            write(out/'r5_perf_historical.json',historical(a.historical_inputs))
            write(out/'r5_perf_order.json',order)
            if a.phase=='diagnose':
                from semcache.edgelora.r5_perf_timing import scan_diagnostic
                write(out/'r5_perf_scan_attribution.json',scan_diagnostic())
            else:write(out/'r5_perf_preflight.json',dict(missing_dependencies=[n for n in ('torch','numpy','transformers','peft') if importlib.util.find_spec(n) is None],
                semantic_revision=TINYBERT_REVISION,blocks=len(order['blocks']),storage_reservation_bytes=len(order['blocks'])*a.block_budget_bytes+2**30,job_submitted=False))
        else:
            if a.phase=='fixture':
                from semcache.edgelora.r3a_tiny import prepare_tiny
                base=prepare_tiny(out/'base_prepare')
            else:preflight(a,order);base=a.base_prepared.resolve()
            config=dict(fixture=a.phase=='fixture',generation_budget=a.generation_budget,generation_mode=a.generation_mode,
                evidence_profile=a.evidence_profile,warmup_requests=a.warmup_requests,legacy_scan=a.legacy_scan,reconcile_every=a.reconcile_every,
                seed=a.seed,semantic_revision=a.semantic_revision,block_budget_bytes=a.block_budget_bytes)
            write(out/'r5_perf_config.json',config);write(out/'r5_perf_order.json',order);write(out/'r5_perf_sources.json',sources())
            write(out/'r5_perf_historical.json',historical(a.historical_inputs));executed=[]
            for block in order['blocks']:
                result.update(stage='BLOCK_EXECUTION',block_id=block['block_id']);write(out/'r5_perf_status.json',result)
                dest=out/'blocks'/block['block_id'];dest.mkdir(parents=True);writer=Writer(dest,a.block_budget_bytes,create=True)
                es=dest/'es';es.mkdir()
                plan=dict(config,**{k:v for k,v in block.items() if k!='seed'},device='cpu' if config['fixture'] else 'cuda:0',
                    base_prepared=str(base),base_manifest_sha256=sha(base/'r3a_prepare_manifest.json'),sources=sources(),
                    account_root=str(dest),output=str(es),evidence_budget_bytes=a.block_budget_bytes,lifecycle_capacity=True)
                writer.write(es/'es_plan.json',plan);started=time.monotonic_ns()
                env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                child=spawn([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','es','--plan',str(es/'es_plan.json')],writer,es/'stdout.log',es/'stderr.log',env=env)
                life=finish(child);writer.write(es/'lifecycle.json',life)
                require(life['returncode']==0 and not life['forced_termination'] and not life['pipe_errors'],'block failed; preserved '+str(es/'stderr.log'))
                writer.reconcile();writer.write(dest/'r5_final_reconciliation.json',dict(status='PASS',scans=writer.scans));writer.reconcile()
                executed.append(dict(block_id=block['block_id'],condition=block['condition'],start_ns=started,end_ns=time.monotonic_ns(),es_pid=child[0].pid))
                write(out/'r5_perf_execution_order.json',executed)
            result['stage']='INDEPENDENT_AUDIT'
            from semcache.edgelora.r5_perf_analysis import analyze,export
            report=analyze(out,sealed=False);export(out,report);decision=report['decision'];result['GPU_NOT_RUN']=config['fixture']
            verify_sources(read(out/'r5_perf_sources.json'))
            from semcache.edgelora.r4c_evidence import directory_bytes
            require(directory_bytes(out)<len(order['blocks'])*a.block_budget_bytes+2**30,'EVIDENCE_BUDGET_EXCEEDED: experiment setup/report allowance')
        result.update(execution_status='COMPLETE',stage='COMPLETE')
    except BaseException as exc:
        result.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());traceback.print_exc()
        write(out/'r5_perf_failure.json',result)
    finally:
        if result['execution_status']!='COMPLETE':decision=flags()
        if a.phase=='run':
            result['gpu_attempted']=(out/'blocks').exists()
            result['gpu_forward_evidence_present']=any(out.rglob('gpu_execution.json'))
            result['GPU_NOT_RUN']=not result['gpu_attempted']
            result['gpu_status']='GPU_COMPLETE_WITH_AUDIT' if decision['R5_I3P_GPU_EXECUTION_PASS'] else 'GPU_RUN_INCOMPLETE_OR_FAILED' if result['gpu_attempted'] else 'GPU_NOT_RUN'
        write(out/'r5_perf_decision.json',decision);write(out/'r5_perf_results.json',result);seal(out)
    print(json.dumps(dict(status=result['execution_status'],output=str(out))))
    return 0 if result['execution_status']=='COMPLETE' else 1


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda s,f:sys.exit(128+s));sys.exit(main())
