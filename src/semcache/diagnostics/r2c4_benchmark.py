"""Six-arm RAW cache-path experiment; frozen C3 Impact Replay is reused."""
from time import perf_counter_ns
from semcache.diagnostics.r2c3_benchmark import Runtime as C3Runtime,BOUNDARIES as C3_BOUNDARIES,compact
from semcache.diagnostics.r2c1_benchmark import OutputAudit
from semcache.diagnostics.r2c3_impact_replay import BASES
from semcache.diagnostics.r1b3_break_even import schedule,MissingEvidence,ParityFailure
from semcache.diagnostics.r2b3_benchmark import strict_metrics
from semcache.diagnostics.r2c4_cache_path import ARMS,RAW,SNAPSHOT,MAPPING,OracleLease,PathProbe,validate_plan
from semcache.metrics.r1 import digest

BOUNDARIES=dict(C3_BOUNDARIES,
    snapshot_oracle='skip only SourceBackingDescriptors.validate at timed batched_v2_projection_path entry; full validation/content check before and after each oracle, outside clock',
    mapping_oracle='retain actual FastOccurrenceIndex selection and RequestMapping.select; replace only PreparedMappingCache.get(validate=False) by checked prebound access',
    retained='selector index.guard, executor selected-sequence/geometry checks, identical QKV owners/indices, fresh native projections, contiguous/fallback assembly, exact Impact Replay and policy',
    external_costs='oracle preselection, full descriptor validation, Source content hashes, mapping binding/index readiness and seals, postvalidation, assembled-QKV fingerprints and Logit equality; never amortized',
    minimal_observation='thin method counters and exact request-timer entry/exit marker; no SetupBook/ExecutorBook/CUDA events',
    breakdown='same paths, separate nonminimal trial with CPU method spans and existing SetupBook/ExecutorBook; event intervals are asynchronous and inclusive/nested, not kernel self times',
    historical_residual='C3 user-reported 181.10ms RAW-minus-NoReuse is an observed request-path difference, not independent additive CPU/GPU components')


class Runtime(C3Runtime):
    def request(self,*args,**kwargs):
        if getattr(self,'_cache_path_breakdown',False):kwargs['instrumentation']='breakdown'
        return super().request(*args,**kwargs)
    def invoke_cache_arm(self,arm,detailed=False):
        if arm not in ARMS:raise ValueError('unknown C4 arm')
        self.guard()
        preparation_start=perf_counter_ns()
        try:lease=OracleLease(self,arm) if arm in RAW[1:] else None
        except BaseException as exc:
            exc.cache_path_evidence=dict(arm=arm,phase='outside_preparation',
                outside_failed_preparation_ms=(perf_counter_ns()-preparation_start)/1e6)
            raise
        probe=PathProbe(self,arm,lease,detailed)
        self._cache_path_breakdown=detailed
        try:
            with probe.scoped():
                logits,record=super().invoke_arm(arm if arm in ARMS[:2] else 'RAW_SEMCACHE_QKV_IMPACT_REPLAY')
            if lease:lease.finish(probe.flow)
            geometry_start=perf_counter_ns()
            if probe.flow is not None and probe.flow.plan is not None:
                validate_plan(self,probe.flow.plan,probe.flow.hits)
            geometry_ms=(perf_counter_ns()-geometry_start)/1e6
            start=perf_counter_ns(); hashes=probe.output_hashes()
            audit_ms=(perf_counter_ns()-start)/1e6
            record.update(cache_path=probe.audit(record),assembled_qkv_sha256=hashes,
                qkv_audit_outside_request_ms=audit_ms,cache_path_host_breakdown=probe.export(),cache_path_arm=arm,
                plan_geometry_audit_outside_request_ms=geometry_ms,
                instrumentation='breakdown' if detailed else 'minimal')
            if lease and lease.mapping_oracle:
                record['mapping_dispatch_api_boolean']=record.get('mapping_cache_hit')
                record['mapping_cache_hit']=None
            return logits,record
        except BaseException as exc:
            exc.cache_path_evidence=dict(arm=arm,probe_calls={str(k):v for k,v in probe.calls.items()},
                request_starts=probe.request_starts,request_ends=probe.request_ends,
                excluded_costs_ms=lease.costs if lease else {},scope='oracle failed; no valid performance evidence')
            raise
        finally:
            self._cache_path_breakdown=False
            probe.flow=None  # release borrowed passive GPU outputs, no Tensor dump


def equivalence(reference_output,output,reference,record):
    import torch
    for field in ('coverage_audit','projection_audit','selection_hash','selected_provenance','output_shape',
                  'mapping','policy','impact_values_digest','reference_digest','assembled_qkv_sha256'):
        if reference.get(field)!=record.get(field):
            error=ParityFailure('RAW baseline/oracle evidence mismatch: '+field)
            error.evidence=dict(field=field,baseline=reference.get(field),oracle=record.get(field));raise error
    if not record.get('assembled_qkv_sha256'):raise MissingEvidence('actual QKV assembly evidence missing')
    if not torch.equal(reference_output,output):
        error=ParityFailure('RAW baseline/oracle logits not exactly equal')
        error.evidence=strict_metrics(reference_output,output);raise error
    if record['impact_real_reduction_calls'] or record['impact_fallback_calls'] or record['impact_dispatch_calls']!=1:
        raise ParityFailure('cache-path arm executed real Attention Impact reduction')
    return dict(exact_logits=True,exact_qkv=True,
        qkv_equality_method='SHA256 of all actual passive Q/K/V outputs, same shapes/dtypes; no recomputation or dumps',
        qkv_digest=digest(record['assembled_qkv_sha256']),strict_metrics=strict_metrics(reference_output,output),
        policy_equal=True,impact_values_equal=True,source_and_rows_equal=True,impact_replay_reduction_calls=0)


def gate(runtime):
    outputs={};records={};audits={}
    for arm in ARMS:
        observer=OutputAudit(runtime)
        with observer.hooks():outputs[arm],records[arm]=runtime.invoke_cache_arm(arm)
        audits[arm]=observer.report(records[arm]['reused_tokens'])
    parity={arm:equivalence(outputs[ARMS[2]],outputs[arm],records[ARMS[2]],records[arm]) for arm in RAW}
    quality={arm:strict_metrics(outputs['FULL_NATIVE'],out) for arm,out in outputs.items()}
    runtime.guard(True)
    return dict(ablation_equivalence_pass=True,impact_replay_reduction_calls=0,output_audits=audits,
        raw_equivalence=parity,full_relative_quality=quality,
        path_evidence={a:r['cache_path'] for a,r in records.items()},
        raw_coverage=records[ARMS[2]]['coverage_audit'])


def compact_record(record):
    return dict(compact(record),cache_path=record['cache_path'],
        qkv_digest=digest(record['assembled_qkv_sha256']) if record['assembled_qkv_sha256'] else None,
        qkv_audit_outside_request_ms=record['qkv_audit_outside_request_ms'],
        plan_geometry_audit_outside_request_ms=record['plan_geometry_audit_outside_request_ms'])


def paired(runtime,args,identity,save):
    from semcache.diagnostics.r2c4_analysis import differences
    if any(m._forward_hooks or m._forward_pre_hooks for m in runtime.model.modules()):raise MissingEvidence('hooks in minimal latency')
    rows=[]
    for phase,count,offset in [('warmup',args.warmup_runs,0),('measured',args.measured_runs,100003)]:
        for i,order in enumerate(schedule(ARMS,count,args.order_seed+runtime.length+offset)):
            outputs={};records={}
            for arm in order:
                save(context=dict(case=identity['case_id'],phase=phase,pair_index=i,arm=arm))
                outputs[arm],records[arm]=runtime.invoke_cache_arm(arm)
            audit_start=perf_counter_ns()
            parity={a:equivalence(outputs[ARMS[2]],outputs[a],records[ARMS[2]],records[a]) for a in RAW}
            wall={a:r['request_ms'] for a,r in records.items()}
            quality={a:strict_metrics(outputs['FULL_NATIVE'],v) for a,v in outputs.items()}
            row=dict(identity,phase=phase,pair_index=i,benchmark_phase=args.phase,instrumentation='minimal',
                execution_order=list(order),wall_ms=wall,wrapper_wall_ms={a:r['wrapper_invoke_wall_ms'] for a,r in records.items()},
                differences=differences(wall),arms={a:compact_record(r) for a,r in records.items()},
                raw_equivalence=parity,full_relative_quality=quality,
                outside_equivalence_quality_wall_ms=(perf_counter_ns()-audit_start)/1e6,
                ablation_equivalence_pass=True,structural_valid=True,cache_integrity_valid=True,
                impact_replay_reduction_calls=0,correctness_qualified_speedup=False,production_speedup_claimed=False)
            rows.append(row);outputs.clear();save(rows=rows)
    return rows
