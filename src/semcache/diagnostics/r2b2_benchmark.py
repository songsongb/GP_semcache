"""B2-only native runtime: B1 request sequence preserved, output contracts explicit.

The forward body is deliberately derived from frozen B1 (no executor changes).
Differences: native endpoint row count and optional returned attention validation.
"""
from contextlib import nullcontext
from time import perf_counter_ns
from semcache.diagnostics.r2b1_prefix_runtime import PrefixRuntime
from semcache.diagnostics.r2b1_prefix_cache import PrefixError,cache_pairs,validate_pairs
from semcache.diagnostics.r2b1_prefix_audit import ForwardAudit,compare_suffix
from semcache.diagnostics.r1b3_fp16_divergence import logit_metrics
from semcache.diagnostics.r2b15_cost import invoke
from semcache.diagnostics.r2b2_attention_ablation import B2API,endpoint_policy,backend_audit,validate_attention_output

MODES=('FULL_USE_CACHE','WARM_NORMAL','WARM_PREPARED_DIAGNOSTIC')

class B2Runtime(PrefixRuntime):
    def gpu_state(self):
        from semcache.diagnostics.r2b2_profiler import monitor
        return monitor(self.device)

    def configure(self,arm,endpoint):
        self.arm=arm
        original=getattr(self.api,'original',self.api)
        self.endpoint=endpoint_policy(original.logit_options,endpoint)
        self.api=B2API(original,arm,self.endpoint)
        self.backend=backend_audit(self)
        return self.backend

    def forward(self,path,audit=False):
        import torch
        from semcache.metrics.r1_runtime import synchronize,memory_snapshot
        if path not in ('FULL_USE_CACHE','PREFIX_SOURCE_PREFILL','EXACT_PREFIX_WARM'): raise ValueError(path)
        is_source=path=='PREFIX_SOURCE_PREFILL'
        p=self.p if path=='EXACT_PREFIX_WARM' else 0
        ids=self.ids[:self.p] if is_source else self.ids[p:]
        total=len(ids)+p
        if not ids: raise PrefixError('REJECT_LENGTH','no empty forward')
        if p and self.snapshot is None: raise PrefixError('REJECT_MISSING_CACHE','Source Prefix must complete before Target')
        observer=ForwardAudit(self.model,self.adapter,self.decoder,len(ids),p,total,path,ids) if audit else None
        # Hooks exist only for an untimed diagnostic pass. Minimal passes have none.
        with observer.hooks() if observer else nullcontext():
            synchronize(self.device); start=perf_counter_ns()
            self.model.set_adapter(self.a.adapter); self.model.eval()
            kwargs=self.api.kwargs(ids,p,total,self.device)
            request_cache=None; prep_ms=None; prepared=None; input_before_lengths=[]
            if p:
                started=perf_counter_ns()
                request_cache=self.snapshot.materialize(self.identity(),self.ids,position_start=p,mask=[1]*total)
                kwargs['past_key_values']=request_cache
                input_before_lengths=[int(pair[0].shape[2]) for pair in cache_pairs(request_cache)[0]]
                prep_ms=(perf_counter_ns()-started)/1e6
                if audit: prepared=self.snapshot.inspect_request(request_cache)
            with torch.inference_mode():
                if is_source: self.source_forwards+=1
                out=self.model(**kwargs)
            synchronize(self.device); wall=(perf_counter_ns()-start)/1e6
        # All cache shape/content checks, logits copies, and quality are OUTSIDE timer.
        if tuple(out.logits.shape)!=(1,1 if self.endpoint['effective']=='native_last' else len(ids),self.model.config.vocab_size): raise PrefixError('REJECT_LOGIT_SHAPE','all requested positions must be returned')
        pairs,api,_=cache_pairs(out.past_key_values)
        rows=validate_pairs(pairs,layers=len(self.adapter.layers),length=total,heads=self.heads,head_dim=self.head_dim,
            dtype=str(next(self.model.parameters()).dtype),device=self.device)
        attention_output=validate_attention_output(out.attentions,self.arm,len(self.adapter.layers),self.heads,len(ids),total)
        past_check=self.snapshot.inspect_output(out.past_key_values,total) if p else None
        input_lengths=[int(pair[0].shape[2]) for pair in cache_pairs(request_cache)[0]] if request_cache is not None else []
        result=dict(logits=out.logits.detach().cpu(),cache=out.past_key_values,wall_ms=wall,cache_prepare_host_ms=prep_ms,
            cache_prepare_scope='host enqueue + container construction; async clone completion included in synchronized request wall',
            request_clone_bytes=self.snapshot.bytes if p else 0,query_tokens=len(ids),past_tokens=p,output_kv_length=total,
            cache_api=api,actual_cache_class=f'{type(out.past_key_values).__module__}.{type(out.past_key_values).__qualname__}',
            request_cache_class=f'{type(request_cache).__module__}.{type(request_cache).__qualname__}' if request_cache is not None else None,
            materialization_mode=self.snapshot.mode if p else 'NO_INCOMING_CACHE',cache_rows=rows,request_input_before_lengths=input_before_lengths,request_input_after_lengths=input_lengths,
            request_container_grew=bool(p and any(length!=p for length in input_lengths)),
            canonical_source_length=self.snapshot.length if self.snapshot else None,output_prefix_check=past_check,
            prepared_input_audit=prepared,audit=observer.report() if observer else None,memory=memory_snapshot(self.device),
            cache_hit=bool(p),instrumented=audit)
        result.update(logit_shape=list(out.logits.shape),attention_output=attention_output,output_fields=list(out.keys()),endpoint=self.endpoint)
        return result


def modes(runtime):
    # Prepared cache is undefined at P=0; never relabel an empty cache as a hit.
    return MODES if runtime.p else MODES[:2]


def quality(runtime,full,warm):
    if runtime.endpoint['effective']=='native_last':
        q=logit_metrics(full,warm)
        return dict(last_position=q,suffix=None,strict=q['allclose'],scope='next-token native last logits only')
    q=compare_suffix(full,warm,runtime.p)
    return dict(**q,strict=q['suffix']['allclose'],scope='all suffix logits and next-token')


def exact_cache(a,b):
    import torch
    aa=cache_pairs(a)[0]; bb=cache_pairs(b)[0]
    return len(aa)==len(bb) and all(torch.equal(x,y) for ap,bp in zip(aa,bb) for x,y in zip(ap,bp))


def parity(normal,prepared):
    import torch
    if not torch.equal(normal['logits'],prepared['logits']) or not exact_cache(normal['cache'],prepared['cache']):
        raise PrefixError('REJECT_PREPARED_PARITY','Normal/Prepared logits or final KV differ')
    if normal['query_tokens']!=prepared['query_tokens']: raise PrefixError('REJECT_FORWARD_LENGTH','prepared rows differ')
    return dict(logits_exact=True,final_kv_exact=True)


def structural_gate(runtime):
    import torch
    runtime.check(); backend_audit(runtime)
    full=runtime.forward('FULL_USE_CACHE',audit=True)
    repeat=runtime.forward('FULL_USE_CACHE')
    normal=runtime.forward('EXACT_PREFIX_WARM',audit=True)
    evidence=dict(STRUCTURAL_REUSE_VALID=True,CACHE_INTEGRITY_VALID=True,
        full_repeat=dict(logits_exact=torch.equal(full['logits'],repeat['logits']),kv_exact=exact_cache(full['cache'],repeat['cache'])),
        quality=quality(runtime,full['logits'],normal['logits']),
        full=full['audit'],normal=normal['audit'],cache_hit=bool(runtime.p),
        source_costs=runtime.source_report,negative_tests=runtime.negative_tests(),
        output=dict(full_shape=full['logit_shape'],normal_shape=normal['logit_shape'],
            full_attention=full['attention_output'],normal_attention=normal['attention_output'],
            full_fields=full['output_fields'],normal_fields=normal['output_fields']),
        endpoint=runtime.endpoint)
    if runtime.p:
        prepared,ticket=invoke(runtime,'WARM_PREPARED_DIAGNOSTIC',audit=True)
        evidence['NORMAL_PREPARED_PARITY']=parity(normal,prepared)
        evidence['prepared']=prepared['audit']; evidence['prepared_ticket']=ticket
        evidence['prepared_classification']='IDEAL_DIAGNOSTIC_ONLY'
    else: evidence['NORMAL_PREPARED_PARITY']='NOT_APPLICABLE_NO_PREFIX'
    # Native last-logit capability must agree with the unmodified native all-row head.
    if runtime.endpoint['effective']=='native_last':
        requested=runtime.endpoint['requested']
        try:
            runtime.configure(runtime.arm,'b1')
            all_full=runtime.forward('FULL_USE_CACHE'); all_warm=runtime.forward('EXACT_PREFIX_WARM')
            fq=logit_metrics(all_full['logits'][:,-1:,:],full['logits'])
            nq=logit_metrics(all_warm['logits'][:,-1:,:],normal['logits'])
            evidence['native_endpoint_parity']=dict(full=fq,normal=nq)
            evidence['all_suffix_control']=compare_suffix(all_full['logits'],all_warm['logits'],runtime.p)
            evidence['quality']['strict']=evidence['quality']['strict'] and evidence['all_suffix_control']['suffix']['allclose']
            if not fq['allclose'] or not nq['allclose']:
                raise PrefixError('REJECT_ENDPOINT_PARITY','native last-token output changes strict next-token result')
        finally: runtime.configure(runtime.arm,requested)
    else: evidence['native_endpoint_parity']='NOT_APPLICABLE_ALL_LOGITS'
    evidence['canonical']=runtime.check()
    evidence['FULL_RELATIVE_STRICT_CORRECTNESS']='STRICT_PASS' if evidence['quality']['strict'] else 'STRICT_FAIL'
    return evidence


def run_pairs(runtime,a,identity,save):
    from semcache.diagnostics.r1b3_break_even import schedule
    if any(m._forward_hooks or m._forward_pre_hooks for m in runtime.model.modules()):
        raise PrefixError('REJECT_MINIMAL_INSTRUMENTATION','hooks remain before minimal benchmark')
    def binding():
        snapshot=getattr(runtime,'snapshot',None)
        return (tuple(runtime.ids),runtime.n,runtime.p,id(snapshot),getattr(snapshot,'identity_hash',None))
    expected_request=binding()
    rows=[]
    for phase,count,offset in [('warmup',a.warmup_runs,0),('measured',a.measured_runs,100003)]:
        for index,order in enumerate(schedule(modes(runtime),count,a.order_seed+offset+runtime.n+runtime.p)):
            results={}; tickets={}
            for mode in order:
                save(context=dict(identity,phase=phase,pair_index=index,mode=mode))
                if binding()!=expected_request: raise PrefixError('REJECT_PAIRED_IDENTITY','target/source changed within paired session')
                runtime.check(); backend_audit(runtime)
                result,ticket=invoke(runtime,mode)
                if binding()!=expected_request: raise PrefixError('REJECT_PAIRED_IDENTITY','target/source changed during request')
                runtime.check(); results[mode]=result; tickets[mode]=ticket
            f=results['FULL_USE_CACHE']; n=results['WARM_NORMAL']; p=results.get('WARM_PREPARED_DIAGNOSTIC')
            if p: parity(n,p)
            q=quality(runtime,f['logits'],n['logits'])
            row=dict(identity,phase=phase,pair_index=index,execution_order=list(order),instrumentation='minimal',
                full_wall_ms=f['wall_ms'],normal_warm_wall_ms=n['wall_ms'],prepared_wall_ms=p['wall_ms'] if p else None,
                full_minus_normal_ms=f['wall_ms']-n['wall_ms'],
                full_minus_prepared_ms=f['wall_ms']-p['wall_ms'] if p else None,
                normal_minus_prepared_ms=n['wall_ms']-p['wall_ms'] if p else None,
                normal_cache_prepare_host_ms=n['cache_prepare_host_ms'],normal_clone_bytes=n['request_clone_bytes'],
                prepared_ticket=tickets.get('WARM_PREPARED_DIAGNOSTIC'),prepared_classification='IDEAL_DIAGNOSTIC_ONLY' if p else 'NOT_APPLICABLE_NO_PREFIX',
                correctness_status='STRICT_PASS' if q['strict'] and identity['structural_strict_pass'] else 'STRICT_FAIL',quality=q,
                normal_prepared_parity='EXACT' if p else 'NOT_APPLICABLE',
                full_logit_shape=f['logit_shape'],normal_logit_shape=n['logit_shape'],
                full_query_tokens=f['query_tokens'],normal_query_tokens=n['query_tokens'],
                final_kv_length=n['output_kv_length'],cache_hit=bool(runtime.p),
                prefix_layer_rows_skipped=runtime.p*len(runtime.adapter.layers),
                memory=n['memory'])
            rows.append(row); results.clear(); del result,f,n,p
            save(rows=rows)
    return rows
