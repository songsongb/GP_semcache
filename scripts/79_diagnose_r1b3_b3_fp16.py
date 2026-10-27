#!/usr/bin/env python3
"""B3 correctness-only hooks and D1 same-hidden shape probes; never latency evidence."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r1b3_b3_runtime import Runtime,common_arguments,config
from semcache.diagnostics.r1b3_break_even import create_run,failure,ParityFailure
from semcache.diagnostics.r1b3_fp16_divergence import StreamingAudit,logit_metrics,tensor_metrics,first_divergences,shape_probe,hypotheses
from semcache.metrics.r1 import atomic_json,raw_only_guard,digest
from semcache.metrics.r1b2 import csv_write


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__); common_arguments(p)
    p.add_argument('--layers',default='all',help='all or comma-separated layer indices; first observed divergence only in captured layers')
    p.add_argument('--snapshot-limit-mib',type=int,default=1024)
    p.add_argument('--secondary-executors',action='store_true',help='Also compare V1/V2 to target Full')
    p.add_argument('--different-adapter-control',action='store_true')
    p.add_argument('--full-shape-control',action='store_true',help='D1 full projection + source injection; correctness only, zero skipped rows')
    a=p.parse_args(argv); a.lengths=[int(x) for x in a.prompt_lengths.split(',')]
    if min(a.lengths)<4 or len(a.lengths)!=len(set(a.lengths)) or a.snapshot_limit_mib<1 or a.capacity_mib<1: p.error('Invalid length/capacity')
    if a.cache_device=='cuda' and not a.device.startswith('cuda'): p.error('GPU cache requires CUDA model')
    return a


def run(a,manifest):
    runtime=Runtime(a); manifest['runtime']=runtime.provenance
    layers=list(range(len(runtime.adapter.layers))) if a.layers=='all' else [int(x) for x in a.layers.split(',')]
    if not layers or len(set(layers))!=len(layers) or any(i<0 or i>=len(runtime.adapter.layers) for i in layers): raise ValueError('Invalid capture layer selection')
    manifest['captured_layers']=layers; manifest['timing_scope']='CORRECTNESS_ONLY; CPU transfers/hooks enabled; no serving latency claim'
    layer_rows=[]; logits=[]; probes=[]; roots={}; precision=[]
    def write():
        csv_write(a.output_dir/'layerwise_qkv_comparison.csv',layer_rows+probes)
        csv_write(a.output_dir/'first_divergence.csv',first_divergences(layer_rows))
        csv_write(a.output_dir/'logit_correctness.csv',logits)
        csv_write(a.output_dir/'precision_control.csv',precision)
        atomic_json(a.output_dir/'root_cause_summary.json',roots)
        atomic_json(a.output_dir/'baseline_repeatability.json',[r for r in logits if r['comparison'] in ('FULL_A_vs_FULL_B','SOURCE_FULL_vs_TARGET_FULL')])
        atomic_json(a.output_dir/'manifest.json',manifest)
    for n in a.lengths:
        manifest['active_case']=dict(length=n,phase='source_capture')
        source_audit=StreamingAudit(runtime.adapter,'user_a',layers,'SOURCE_FULL',retain=True,limit_mib=a.snapshot_limit_mib)
        source_logits=runtime.source(n,source_audit.hooks)
        identity=dict(code_identity=manifest['code']['sha256'],independent_run_id=a.independent_run_id,model_revision=runtime.meta['resolved_model_revision'],
            namespace=runtime.namespace,weight_identity=runtime.weights,length=n,dtype=a.dtype,model=a.model,source_identity=runtime.snapshot,target_identity=digest(runtime.ids),
            adapter='user_a',instrumentation='correctness_only',seed=a.seed)
        for layer in layers:
            for role,tensor in zip('qkv',runtime.catalog.sources['source']['qkv'][layer]):
                metric=tensor_metrics(source_audit.snapshots[(layer,'total_qkv',role)],tensor)
                layer_rows.append(dict(identity,comparison='SOURCE_CAPTURE_vs_CACHE_BACKING',layer=layer,stage='cache_backing',role=role,**metric))
                if not metric['exact_equal']:
                    write(); raise ParityFailure('Raw source backing differs from actual source projection capture')
        manifest.setdefault('cases',{})[str(n)]=dict(**identity,token_ids=runtime.ids,memory_preflight=runtime.preflight,
            source_costs_diagnostic_only=runtime.costs,source_capture='actual engine source prefill, no reuse, full native projection shapes')
        target_audit=StreamingAudit(runtime.adapter,'user_a',layers,'SOURCE_FULL_vs_TARGET_FULL',source_audit.snapshots,True,a.snapshot_limit_mib)
        target=runtime.full(runtime.text,target_audit.hooks)['logits']; source_audit.snapshots.clear()
        layer_rows.extend(dict(identity,**r) for r in target_audit.rows)
        reference=target_audit.snapshots; comparisons={}
        def record(label,other,ref=target):
            value=logit_metrics(ref,other); comparisons[label]=value; logits.append(dict(identity,comparison=label,**value)); write(); return value
        record('SOURCE_FULL_vs_TARGET_FULL',target,source_logits)
        audit=StreamingAudit(runtime.adapter,'user_a',layers,'FULL_A_vs_FULL_B',reference,limit_mib=a.snapshot_limit_mib)
        full_b=runtime.full(runtime.text,audit.hooks)['logits']; layer_rows.extend(dict(identity,**r) for r in audit.rows)
        record('FULL_A_vs_FULL_B',full_b)
        legacy_reference=None; legacy_logits=None; contig_logits=None; reuse_records={}; secondary_parity={}
        for mode in ['lookup','legacy','contiguous']+(['v1','v2'] if a.secondary_executors else []):
            manifest['active_case']=dict(length=n,phase='candidate_capture',executor=mode)
            label=('LEGACY_vs_' if mode in ('v1','v2') else 'TARGET_FULL_vs_')+mode.upper()
            audit=StreamingAudit(runtime.adapter,'user_a',layers,label,legacy_reference if mode in ('v1','v2') else reference,retain=mode=='legacy',limit_mib=a.snapshot_limit_mib)
            actual,execution=runtime.request(runtime.text,runtime.ids,mode,'max',observer=audit.hooks)
            reuse_records[mode]=execution
            layer_rows.extend(dict(identity,**r) for r in audit.rows); record('TARGET_FULL_vs_'+mode.upper(),actual)
            if mode in ('v1','v2'):
                same=record(label,actual,legacy_logits)
                secondary_parity[mode]=bool(same['exact_equal'] and all(r['exact_equal'] for r in audit.rows if r['stage']=='total_qkv')
                    and reuse_records['legacy']['selection_hash']==execution['selection_hash']
                    and reuse_records['legacy']['projection_audit']==execution['projection_audit'])
            if mode=='legacy': legacy_reference=audit.snapshots; legacy_logits=actual
            if mode=='contiguous': contig_logits=actual
        paired=StreamingAudit(runtime.adapter,'user_a',layers,'LEGACY_vs_CONTIGUOUS',legacy_reference,limit_mib=a.snapshot_limit_mib)
        repeated,execution=runtime.request(runtime.text,runtime.ids,'contiguous','max',observer=paired.hooks)
        layer_rows.extend(dict(identity,**r) for r in paired.rows)
        parity=record('LEGACY_vs_CONTIGUOUS',contig_logits,legacy_logits)
        replay=record('CONTIGUOUS_REPEATABILITY',repeated,contig_logits)
        qkv_exact=all(r['exact_equal'] for r in paired.rows if r['stage']=='total_qkv')
        selection_equal=reuse_records['legacy']['selection_hash']==reuse_records['contiguous']['selection_hash']
        audit_equal=reuse_records['legacy']['projection_audit']==reuse_records['contiguous']['projection_audit']
        legacy_reference.clear()
        runtime.model.set_adapter('user_a')
        hits=runtime.selected_hits('max')
        case_probes=shape_probe(runtime.adapter,reference,hits,n,'user_a',layers)
        probes.extend(dict(identity,**r) for r in case_probes)
        control=None
        if a.full_shape_control:
            audit=StreamingAudit(runtime.adapter,'user_a',layers,'TARGET_FULL_vs_FULL_SHAPE_INJECTION',reference,limit_mib=a.snapshot_limit_mib)
            actual=runtime.injection_control(audit.hooks)
            layer_rows.extend(dict(identity,**r) for r in audit.rows)
            control=record('TARGET_FULL_vs_FULL_SHAPE_INJECTION',actual)
            manifest['cases'][str(n)]['full_shape_control']=dict(projection_rows_skipped=0,classification='CORRECTNESS_CONTROL_NOT_REUSE_SPEEDUP')
        if a.different_adapter_control:
            audit=StreamingAudit(runtime.adapter,'user_b',layers,'USER_A_FULL_vs_USER_B_FULL',reference,limit_mib=a.snapshot_limit_mib)
            other=runtime.full(runtime.text,audit.hooks,user='user_b')['logits']
            layer_rows.extend(dict(identity,adapter_identity_match=False,**r) for r in audit.rows)
            value=logit_metrics(target,other); logits.append(dict(identity,comparison='USER_A_FULL_vs_USER_B_FULL',adapter_identity_match=False,**value))
        roots[str(n)]=dict(hypotheses=hypotheses(comparisons,[r for r in layer_rows if r['length']==n],case_probes,control),
            first_observed_nonzero=next((r for r in layer_rows if r['length']==n and r['comparison']=='TARGET_FULL_vs_CONTIGUOUS' and not r['exact_equal']),None),
            first_observed_tolerance_failure=next((r for r in layer_rows if r['length']==n and r['comparison']=='TARGET_FULL_vs_CONTIGUOUS' and not r['allclose']),None),
            executor_parity=parity['exact_equal'] and replay['exact_equal'] and qkv_exact and selection_equal and audit_equal and all(secondary_parity.values()),
            secondary_executor_parity=secondary_parity,
            qkv_exact=qkv_exact,selection_equal=selection_equal,projection_audit_equal=audit_equal,
            reuse={k:{f:v[f] for f in ('reused_tokens','actual_coverage','projection_audit','selected_provenance')} for k,v in reuse_records.items()},
            snapshot_limit_mib_per_reference=a.snapshot_limit_mib,max_simultaneous_cpu_references=2)
        precision.append(dict(identity,strict_allclose= comparisons['TARGET_FULL_vs_CONTIGUOUS']['allclose'],
            max_abs_logit_diff=comparisons['TARGET_FULL_vs_CONTIGUOUS']['max_abs_error'],execution_status='COMPLETE'))
        write(); reference.clear()
        if not roots[str(n)]['executor_parity']: raise ParityFailure('Legacy/Contiguous QKV, selection, row or final-logit parity failed')
    manifest.update(execution_status='COMPLETE',correctness_status='STRICT_PASS' if all(r['strict_allclose'] for r in precision) else 'STRICT_FAIL')
    write()


def main(argv=None):
    a=parse_args(argv); manifest=create_run(ROOT,a.output_dir,config(a),'R1-B3-B3 correctness')
    for name in ('first_divergence.csv','layerwise_qkv_comparison.csv','precision_control.csv','logit_correctness.csv'): csv_write(a.output_dir/name,[])
    for name in ('baseline_repeatability.json','root_cause_summary.json'): atomic_json(a.output_dir/name,{})
    try:
        with raw_only_guard() as guard:
            run(a,manifest); manifest['raw_only']=guard.evidence()
        atomic_json(a.output_dir/'manifest.json',manifest); return 0
    except (Exception,KeyboardInterrupt) as exc: return failure(a.output_dir,manifest,exc)
if __name__=='__main__': raise SystemExit(main())
