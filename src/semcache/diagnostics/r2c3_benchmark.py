"""Five-arm conditional-clock ablation over the unchanged C1 executor runtime."""
from time import perf_counter_ns
from semcache.diagnostics.r2c1_benchmark import Runtime as C1Runtime, OutputAudit, BOUNDARIES as C1_BOUNDARIES
from semcache.diagnostics.r1b3_break_even import schedule, MissingEvidence, ParityFailure
from semcache.diagnostics.r2b3_benchmark import strict_metrics
from semcache.metrics.r1 import digest
from semcache.diagnostics.r2c3_impact_replay import ARMS,BASES,ImpactReference,ImpactSession,canonical,window_sequence

BOUNDARIES=dict(C1_BOUNDARIES,
    replay_excluded='actual TOKEN_PRECOMPUTE attention reduction, finite checks and packed attention-to-host impact materialization ONLY; exact previously computed values supplied',
    replay_retained='normal attention tensors; candidate iteration/key grouping/means; history.append; CHU; Admission/Eviction/materialization/PBR; original masks and all Forward flags',
    additional_excluded='untimed reference generation; pre/post policy snapshots, provenance hashes, equality gates, reference serialization; patch installation/removal',
    additional_inside='ON/Replay thin dispatch counter and CPU attention metadata/mask observation; Replay exact candidate order/metadata validation and precomputed-value dispatch',
    interpretation='ORACLE_UPPER_BOUND cost exclusion, not an independently deployable predictor or production/E2E/service speedup; no kernel-sum subtraction')


class Runtime(C1Runtime):
    def binding(self,base):
        return dict(base_mode=base,token_ids=list(self.ids),length=self.length,requested_coverage=self.coverage,
            coverage_scope='RAW condition request; No-Reuse actual physical coverage is always zero',
            model_weight_identity=self.model_weight_identity,adapter_weight_identity=self.weights,
            model_namespace=self.namespace,source_snapshot=self.snapshot,source_hashes_digest=digest(self.source_hashes),
            adapter='user_a',dtype=self.a.dtype,device=self.a.device,storage=self.storage,capacity_mib=self.a.capacity_mib,
            use_cache=False,output_attentions=True,attention_backend='eager',impact_execution='token_precompute',
            implementation_sha256=self.execution_code_sha256)

    def invoke_arm(self,arm,reference_capture=False):
        if arm not in ARMS: raise ValueError('unknown C3 arm')
        if arm=='FULL_NATIVE':
            if reference_capture: raise MissingEvidence('native Full cannot supply impact reference')
            logits,record=super().invoke(arm)
            record['impact_real_reduction_calls']=0
            return logits,record
        base=next(b for b in BASES if arm.startswith(b+'_IMPACT_'))
        replay=arm.endswith('_REPLAY'); binding=self.binding(base)
        if reference_capture:
            if replay: raise MissingEvidence('cannot build reference during Replay')
            reference=None; expected=None
        else:
            if base not in self.references: raise MissingEvidence('missing untimed mode-specific impact reference')
            reference=self.references[base]; expected=self.reference_digests[base]
            reference.verify(binding,expected)
        session=ImpactSession(self,binding,reference if replay else None,expected if replay else None)
        with session.scoped(): logits,record=super().invoke(base)
        values_digest=digest(canonical((window_sequence(session.windows),tuple(session.values))))
        if reference is not None:
            if (window_sequence(session.windows)!=reference.windows or
                tuple(float(v).hex() for v in session.values)!=tuple(v.hex() for v in reference.values) or
                digest(session.policy)!=reference.policy_digest):
                error=ParityFailure('ON/Replay actual impact values or policy differ from untimed ON reference')
                error.evidence=dict(base_mode=base,expected_policy_digest=reference.policy_digest,
                    actual_policy= session.policy,actual_impact_values_digest=values_digest,
                    expected_reference_digest=reference.sha256)
                raise error
        record.update(impact_real_reduction_calls=session.real_reduction_calls,
            impact_dispatch_calls=session.dispatch_calls,impact_fallback_calls=session.fallback_calls,
            impact_values_digest=values_digest,policy=session.policy,impact_range_wall_ms=session.impact_wall_ms,
            reference_digest=expected,arm=arm)
        if reference_capture: record['_session']=session
        return logits,record


def capture_references(runtime):
    runtime.references={}; runtime.reference_digests={}; runtime.reference_policies={}; exported={}; outputs={}; records={}
    for base in BASES:
        runtime.guard(True); observer=OutputAudit(runtime)
        start=perf_counter_ns()
        with observer.hooks(): logits,record=runtime.invoke_arm(base+'_IMPACT_ON',reference_capture=True)
        wrapper_wall=(perf_counter_ns()-start)/1e6
        reference=ImpactReference.capture(runtime.binding(base),record.pop('_session'))
        runtime.references[base]=reference; runtime.reference_digests[base]=reference.sha256
        runtime.reference_policies[base]=record['policy']
        record['reference_digest']=reference.sha256
        exported[base]=dict(reference.export(runtime.binding(base)),
            reference_conditional_request_ms=record['request_ms'],reference_request_wrapper_wall_ms=wrapper_wall,
            actual_coverage=record['coverage_audit']['actual_coverage'],projection_audit=record['projection_audit'],
            output_audit=observer.report(record['reused_tokens']),policy=record['policy'],
            cost_scope='untimed ON reference with output audit; excluded from all paired clocks, never amortized')
        exported[base]['reference_generation_wall_ms']=(perf_counter_ns()-start)/1e6
        outputs[base]=logits; records[base]=record
    return exported,outputs,records


def check_equivalence(on_output,replay_output,on,replay):
    import torch
    for name in ('coverage_audit','projection_audit','selection_hash','selected_provenance','output_shape',
                 'policy','impact_values_digest','reference_digest'):
        if name not in on or on[name]!=replay.get(name):
            error=ParityFailure('ON/REPLAY mismatch: '+name)
            error.evidence=dict(field=name,on=on.get(name),replay=replay.get(name))
            raise error
    if (on['impact_real_reduction_calls']!=1 or replay['impact_real_reduction_calls']!=0 or
        on['impact_dispatch_calls']!=1 or replay['impact_dispatch_calls']!=1 or
        on['impact_fallback_calls'] or replay['impact_fallback_calls']):
        raise ParityFailure('incorrect ON/REPLAY reduction dispatch audit')
    if not torch.equal(on_output,replay_output):
        error=ParityFailure('ON/REPLAY logits not exactly equal')
        error.evidence=dict(exact_logits=False,strict_metrics=strict_metrics(on_output,replay_output))
        raise error
    return dict(exact_logits=True,strict_metrics=strict_metrics(on_output,replay_output),policy_equal=True,
        coverage_and_rows_equal=True,impact_values_equal=True,policy=on['policy'],
        reference_digest=on['reference_digest'],impact_replay_reduction_calls=0)


def validate_replay(runtime,outputs,records):
    proof={}
    for base in BASES:
        observer=OutputAudit(runtime)
        with observer.hooks(): replay,record=runtime.invoke_arm(base+'_IMPACT_REPLAY')
        proof[base]=dict(check_equivalence(outputs[base],replay,records[base],record),
            output_audit=observer.report(record['reused_tokens']))
    runtime.guard(True)
    return proof


def compact(record):
    coverage=record['coverage_audit']
    return dict(reused_tokens=coverage['reused_tokens'],actual_coverage=coverage['actual_coverage'],
        skipped_projection_rows=coverage['skipped_projection_rows'],native_projection_rows=coverage['native_projection_rows'],
        selection_hash=record.get('selection_hash'),selected_provenance_digest=digest(record.get('selected_provenance',[])),
        policy=record.get('policy'),impact_values_digest=record.get('impact_values_digest'),
        reference_digest=record.get('reference_digest'),
        real_reduction_calls=record['impact_real_reduction_calls'],dispatch_calls=record.get('impact_dispatch_calls',0),
        fallback_calls=record.get('impact_fallback_calls',0),impact_range_wall_ms=record.get('impact_range_wall_ms'),
        mapping_cache_hit=record.get('mapping_cache_hit'),mapping_ms=record.get('mapping_ms'),
        context_setup_ms=record.get('context_setup_ms'),selection_ms=record.get('selection_ms'))


def paired(runtime,args,identity,save):
    if any(m._forward_hooks or m._forward_pre_hooks for m in runtime.model.modules()):
        raise MissingEvidence('model hooks remain in minimal loop')
    from semcache.diagnostics.r2c3_analysis import differences
    rows=[]
    for phase,count,offset in [('warmup',args.warmup_runs,0),('measured',args.measured_runs,100003)]:
        for repeat,order in enumerate(schedule(ARMS,count,args.order_seed+runtime.length+offset)):
            outputs={}; records={}
            for arm in order:
                save(context=dict(case=identity['case_id'],phase=phase,pair_index=repeat,arm=arm))
                runtime.guard(); outputs[arm],records[arm]=runtime.invoke_arm(arm); runtime.guard()
            parity={base:check_equivalence(outputs[base+'_IMPACT_ON'],outputs[base+'_IMPACT_REPLAY'],
                records[base+'_IMPACT_ON'],records[base+'_IMPACT_REPLAY']) for base in BASES}
            quality={arm:strict_metrics(outputs['FULL_NATIVE'],outputs[arm]) for arm in ARMS}
            wall={arm:records[arm]['request_ms'] for arm in ARMS}
            row=dict(identity,phase=phase,pair_index=repeat,execution_order=list(order),instrumentation='minimal',
                benchmark_phase=args.phase,wall_ms=wall,wrapper_wall_ms={a:r['wrapper_invoke_wall_ms'] for a,r in records.items()},
                differences=differences(wall),arms={a:compact(r) for a,r in records.items()},
                full_relative_quality=quality,on_replay_parity=parity,ablation_equivalence_pass=True,
                impact_replay_reduction_calls=0,structural_valid=True,cache_integrity_valid=True,
                correctness_qualified_speedup=False,production_speedup_claimed=False)
            rows.append(row); outputs.clear(); save(rows=rows)
    return rows
