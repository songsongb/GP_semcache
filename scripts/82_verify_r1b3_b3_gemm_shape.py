#!/usr/bin/env python3
"""C1 opt-in GEMM shape causal controls; ORACLE correctness diagnosis only."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r1b3_b3_runtime import Runtime,common_arguments,config
from semcache.diagnostics.r1b3_break_even import create_run,failure,MissingEvidence
from semcache.diagnostics.r1b3_fp16_divergence import StreamingAudit,tensor_metrics,logit_metrics,first_divergences
from semcache.diagnostics.r1b3_gemm_shape import (Layer0Inputs,ReplayAudit,NativeShapeIntervention,shape_controls,
    assembly_controls,prefix_layers,row_cost,parameter_stamps,assert_parameters,classify_h1,tensor_sha)
from semcache.metrics.r1 import atomic_json,raw_only_guard,digest
from semcache.metrics.r1b2 import csv_write,coverage_value


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__); common_arguments(p)
    p.add_argument('--coverage',default='max')
    p.add_argument('--replay-prefixes',default='1',help='Full-native prefix layer counts, e.g. 1,2,4,8,all; ordinary baseline always runs')
    p.add_argument('--skip-replays',action='store_true',help='Layer-0 shape/assembly controls and ordinary baseline only')
    p.add_argument('--audit-layers',default='all',help='all or layer indices; must include 0; partial capture limits causal conclusions')
    p.add_argument('--snapshot-limit-mib',type=int,default=1024)
    a=p.parse_args(argv)
    try:
        a.lengths=[int(x) for x in a.prompt_lengths.split(',')]; a.requested_coverage=coverage_value(a.coverage)
        if not a.lengths or min(a.lengths)<4 or len(set(a.lengths))!=len(a.lengths) or min(a.capacity_mib,a.snapshot_limit_mib)<1: raise ValueError('Invalid lengths/capacity')
        if a.cache_device=='cuda' and not a.device.startswith('cuda'): raise ValueError('CUDA storage requires CUDA model')
    except ValueError as exc: p.error(str(exc))
    return a


def first(rows,field): return next((r for r in rows if r.get(field) is False),None)


def signature(provenance):
    return digest([dict(source_query=p['query'],source=p['source'],target=p['target'],tokens=p['tokens']) for p in provenance])


def hit_signature(hits):
    return signature([dict(query=h.entry.qkv_metadata['source_query_id'],source=list(h.entry.positions),
        target=[h.window.start,h.window.end],tokens=list(h.window.token_ids)) for h in hits])


def run(a,manifest):
    import torch
    import importlib.metadata
    runtime=Runtime(a); manifest['runtime']=runtime.provenance
    manifest['external_packages']={name:importlib.metadata.version(name) for name in ('torch','peft','transformers')}
    manifest['numerical_backend']=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        allow_fp16_reduced_precision_reduction=getattr(torch.backends.cuda.matmul,'allow_fp16_reduced_precision_reduction',None),
        fp32_reference='highest float32 matmul precision, TF32 disabled only inside functional control; restored afterwards')
    count=len(runtime.adapter.layers)
    layers=list(range(count)) if a.audit_layers=='all' else [int(x) for x in a.audit_layers.split(',')]
    if 0 not in layers or len(set(layers))!=len(layers) or any(i<0 or i>=count for i in layers): raise ValueError('Invalid audit layers, or Layer 0 missing')
    prefixes=[] if a.skip_replays else [p for p in prefix_layers(a.replay_prefixes,count) if p]
    manifest.update(captured_layers=layers,replay_prefix_counts=prefixes,timing_scope='CORRECTNESS_ONLY_HOOKED; no serving/performance claims',
        historical_pilot_reproduced=False,kernel_evidence='NOT_COLLECTED',
        intervention='scoped class native-forward interception; existing Contiguous selection/mapping/assembly code reused')
    shape_rows=[]; layer_rows=[]; replay_rows=[]; logit_rows=[]; summaries={}; identities={}
    def write():
        csv_write(a.output_dir/'shape_control_comparison.csv',shape_rows)
        csv_write(a.output_dir/'base_lora_total_comparison.csv',shape_rows)
        csv_write(a.output_dir/'counterfactual_layer_replay.csv',replay_rows)
        csv_write(a.output_dir/'first_divergence_control.csv',[dict(identities[r['length']],**r) for r in first_divergences(layer_rows)])
        csv_write(a.output_dir/'logit_control_comparison.csv',logit_rows)
        atomic_json(a.output_dir/'h1_verification_summary.json',summaries)
        atomic_json(a.output_dir/'manifest.json',manifest)
    for length in a.lengths:
        manifest['active_case']=dict(length=length,phase='source_prefill')
        source=StreamingAudit(runtime.adapter,'user_a',layers,'SOURCE_FULL',retain=True,limit_mib=a.snapshot_limit_mib)
        source_logits=runtime.source(length,source.hooks)
        stamps=parameter_stamps(runtime.adapter)
        common=dict(length=length,model=a.model,model_revision=runtime.meta['resolved_model_revision'],code_identity=manifest['code']['sha256'],
            weight_identity=runtime.weights,adapter='user_a',namespace=runtime.namespace,source_identity=runtime.snapshot,
            target_identity=digest(runtime.ids),precision=a.dtype,gpu=runtime.provenance['gpu'],seed=a.seed,
            independent_run_id=a.independent_run_id,classification='ORACLE_DIAGNOSTIC')
        identities[length]=common
        backing=[]
        for i in layers:
            for r,tensor in zip('qkv',runtime.catalog.sources['source']['qkv'][i]):
                row=dict(layer=i,role=r,**tensor_metrics(source.snapshots[(i,'total_qkv',r)],tensor))
                backing.append(row)
        cache_exact=all(r['exact_equal'] for r in backing)
        if not cache_exact: summaries[str(length)]=dict(cache_backing=backing); write(); raise MissingEvidence('Actual source capture/cache backing mismatch')
        target=StreamingAudit(runtime.adapter,'user_a',layers,'SOURCE_vs_TARGET_FULL',source.snapshots,True,a.snapshot_limit_mib)
        layer0=Layer0Inputs(runtime.adapter)
        with layer0.hooks(): full_a=runtime.full(runtime.text,target.hooks)
        if full_a['ids']!=runtime.ids: raise MissingEvidence('Full target input IDs differ from source fixture')
        reference=target.snapshots; source.snapshots.clear()
        layer_rows.extend(dict(common,**row) for row in target.rows)
        source_qkv=[r for r in target.rows if r['stage']=='total_qkv']
        source_exact=len(source_qkv)==len(layers)*3 and all(r['exact_equal'] for r in source_qkv)
        source_logit=logit_metrics(source_logits,full_a['logits'])
        repeat_audit=StreamingAudit(runtime.adapter,'user_a',layers,'FULL_A_vs_FULL_B',reference,limit_mib=a.snapshot_limit_mib)
        full_b=runtime.full(runtime.text,repeat_audit.hooks)
        layer_rows.extend(dict(common,**row) for row in repeat_audit.rows)
        repeat=logit_metrics(full_a['logits'],full_b['logits'])
        logit_rows.extend([dict(common,control='SOURCE_vs_TARGET_FULL',**source_logit),dict(common,control='FULL_A_vs_FULL_B',**repeat)])
        hits=runtime.selected_hits(a.requested_coverage)
        used={p for h in hits for p in range(h.window.start,h.window.end)}; fresh=[p for p in range(length) if p not in used]
        if not fresh: raise MissingEvidence('No fresh rows: reduced-shape control is not applicable; choose a partial coverage or a length with a remainder')
        manifest.setdefault('cases',{})[str(length)]=dict(token_ids=runtime.ids,selected_windows=len(hits),fresh_positions=fresh,
            actual_coverage=len(used)/length,preflight=runtime.preflight,layer0_actual_inputs=layer0.info,
            retained_original_layer0_input_bytes=sum(t.numel()*t.element_size() for t in layer0.inputs.values()),
            original_input_storage_shared_between_roles=True if len({t.data_ptr() for t in layer0.inputs.values()})==1 else False)
        for role in 'qkv':
            if not tensor_metrics(reference[(0,'qkv_input',role)],layer0.inputs[role])['exact_equal']:
                raise MissingEvidence('Retained Layer-0 GPU input differs from actual target Full capture')
        manifest['active_case']=dict(length=length,phase='layer0_shape_controls')
        probes,full_values=shape_controls(runtime.adapter,layer0.inputs,fresh)
        for row in probes:
            payload=dict(row); payload['compute_precision']=payload.pop('precision')
            shape_rows.append(dict(common,**payload))
        assembly,assembly_report,subset_outputs=assembly_controls(runtime,layer0.inputs,fresh,full_values,hits)
        for row in assembly:
            payload=dict(row); payload['compute_precision']=payload.pop('precision')
            shape_rows.append(dict(common,**payload))
        # Clear target-device snapshots as soon as the same-input probes finish.
        for role in 'qkv':
            if tensor_sha(layer0.inputs[role])!=next(r['hidden_sha256'] for r in probes if r['role']==role):
                raise MissingEvidence('Original Layer-0 hidden input mutated during controls')
        layer0.inputs.clear(); full_values.clear()
        manifest['active_case']=dict(length=length,phase='ordinary_contiguous')
        ordinary_audit=ReplayAudit(runtime.adapter,reference,layers,'FULL_vs_ORDINARY_CONTIGUOUS')
        ordinary,execution=runtime.request(runtime.text,runtime.ids,'contiguous',a.requested_coverage,
            observer=lambda positions:ordinary_audit.hooks(positions,set()))
        expected_signature=hit_signature(hits)
        if signature(execution['selected_provenance'])!=expected_signature: raise MissingEvidence('Ordinary selected occurrences differ from probe mapping')
        base_mapping_hash=digest(execution['mapping'])
        layer_rows.extend(dict(common,**row) for row in ordinary_audit.rows)
        ordinary_logits=logit_metrics(full_a['logits'],ordinary)
        base_cost=row_cost(length,len(fresh),count,[])
        if execution['projection_audit']['skipped_projection_rows_all_layers_roles']!=base_cost['actual_skipped_projection_rows']:
            raise MissingEvidence('Ordinary physical projection row audit differs from selected spans')
        subset_same=all(tensor_metrics(ordinary_audit.layer0_totals[r],subset_outputs[r])['exact_equal'] for r in 'qkv')
        input_rows=[r for r in ordinary_audit.rows if r['layer']==0 and r['stage']=='qkv_input']
        controls=dict(cache_backing_exact=cache_exact,source_target_qkv_exact=source_exact,
            shape_weight_identity_unchanged=True,
            full_repeat_exact=repeat['exact_equal'] and all(r['exact_equal'] for r in repeat_audit.rows),
            layer0_input_exact=len(input_rows)==3 and all(r['exact_equal'] for r in input_rows),
            subset_assembly_matches_ordinary=subset_same,ordinary_strict_fail=not ordinary_logits['allclose'])
        costs=[dict(base_cost,logits=ordinary_logits,mapping_and_source_unchanged=True,weight_identity_unchanged=True)]
        replay_rows.append(dict(common,control='ORDINARY_CONTIGUOUS',**costs[0],
            first_nonzero=first(ordinary_audit.rows,'exact_equal'),first_tolerance_failure=first(ordinary_audit.rows,'allclose'),
            reused_tokens=len(used),selected_windows=len(hits),actual_gemm_calls=ordinary_audit.gemm_calls,
            diagnostic_request_wall_ms=execution['request_ms']))
        logit_rows.append(dict(common,control='FULL_vs_ORDINARY_CONTIGUOUS',**ordinary_logits))
        write()
        for prefix in prefixes:
            manifest['active_case']=dict(length=length,phase='counterfactual_replay',full_shape_prefix=prefix)
            audit=ReplayAudit(runtime.adapter,reference,layers,f'FULL_vs_FULL_SHAPE_PREFIX_{prefix}')
            with NativeShapeIntervention(runtime.adapter,range(prefix)) as intervention:
                actual,record=runtime.request(runtime.text,runtime.ids,'contiguous',a.requested_coverage,
                    observer=intervention.observer(audit))
                ledger=intervention.accounting(length,len(fresh))
            unchanged=(signature(record['selected_provenance'])==expected_signature and digest(record['mapping'])==base_mapping_hash
                and record['reused_tokens']==execution['reused_tokens'] and runtime.snapshot==common['source_identity'])
            if not unchanged: raise MissingEvidence('Counterfactual changed source/occurrence/mapping/coverage')
            assert_parameters(stamps); runtime.index.guard()
            quality=logit_metrics(full_a['logits'],actual)
            cost=dict(ledger,logits=quality,mapping_and_source_unchanged=unchanged,weight_identity_unchanged=True)
            costs.append(cost)
            layer_rows.extend(dict(common,**row) for row in audit.rows)
            replay_rows.append(dict(common,control=f'FULL_SHAPE_PREFIX_{prefix}',**cost,
                first_nonzero=first(audit.rows,'exact_equal'),first_tolerance_failure=first(audit.rows,'allclose'),
                reused_tokens=len(used),selected_windows=len(hits),diagnostic_request_wall_ms=record['request_ms'],
                actual_gemm_calls=audit.gemm_calls,unchanged_executor_wrapper_audit=record['projection_audit']))
            logit_rows.append(dict(common,control=f'FULL_vs_FULL_SHAPE_PREFIX_{prefix}',**quality))
            write()
        assert_parameters(stamps)
        case_shape=[r for r in shape_rows if r['length']==length]
        verdict=classify_h1(case_shape,controls,costs,count,set(layers)==set(range(count)))
        summaries[str(length)]=dict(verdict=verdict,controls=controls,cache_backing=backing,
            source_vs_target_full_logits=source_logit,full_repeatability=repeat,ordinary_contiguous=ordinary_logits,
            assembly_controls=assembly_report,actual_shape_records='shape_control_comparison.csv native_call.gemm_calls',
            fp32_scope='upcast identical stored projection weights and identical hidden values; not a loaded OPT-2.7B FP32 model',
            kernel_mechanism_identified=False,all_layer_replay_executed=count in prefixes)
        write(); reference.clear(); ordinary_audit.layer0_totals.clear(); subset_outputs.clear()
    manifest.update(execution_status='COMPLETE',correctness_status='STRICT_PASS' if all(s['ordinary_contiguous']['allclose'] for s in summaries.values()) else 'STRICT_FAIL')
    write()


def main(argv=None):
    a=parse_args(argv); manifest=create_run(ROOT,a.output_dir,config(a),'R1-B3-B3-C1 GEMM shape causal verification')
    manifest.update(kernel_mechanism_identified=False,kernel_evidence='NOT_COLLECTED',historical_pilot_reproduced=False)
    for name in ('shape_control_comparison.csv','base_lora_total_comparison.csv','counterfactual_layer_replay.csv','first_divergence_control.csv','logit_control_comparison.csv'):
        csv_write(a.output_dir/name,[])
    atomic_json(a.output_dir/'h1_verification_summary.json',{})
    try:
        with raw_only_guard() as guard:
            run(a,manifest); manifest['raw_only']=guard.evidence()
        atomic_json(a.output_dir/'manifest.json',manifest); return 0
    except (Exception,KeyboardInterrupt) as exc: return failure(a.output_dir,manifest,exc)
if __name__=='__main__': raise SystemExit(main())
