"""Long-context adapter over unchanged R1-B3 / R2-A Engine and QKV executors."""
from contextlib import contextmanager
from time import perf_counter_ns
from unittest.mock import patch
from semcache.diagnostics.r2a_runtime import CostRuntime
from semcache.diagnostics.r1b3_break_even import schedule,MissingEvidence,ParityFailure
from semcache.diagnostics.r1b3_gemm_shape import tensor_sha
from semcache.diagnostics.r2b3_benchmark import strict_metrics
from semcache.diagnostics.r2b2_profiler import monitor
from semcache.diagnostics.r2c1_workloads import MODES,check_coverage,expected_tokens
from semcache.metrics.r1 import digest

BOUNDARIES=dict(
    primary='frozen synchronized R1 request_wall_ms; conditional built-source request, NOT complete harness/service latency',
    FULL_NATIVE='tokenization + adapter/eval + input tensor + full native PEFT Forward and all-token LM Head',
    SEMCACHE_NO_REUSE='tokenization + semantic embedding + cluster update + exact-w3 lookup/nonoverlap + native-only mixed wrapper + Forward + impact + policy/CHU/PBR as executed',
    RAW_SEMCACHE_QKV='same Engine frontend/policy + fast position-aware Catalog selection + snapshot/mapping + unchanged contiguous/fallback mixed executor + fresh-row projections + remaining full Transformer/LM Head',
    excluded='source prefill/capture/catalog/index/descriptors; per-request Engine deepcopy and Catalog policy reset; mapping controller allocation; post-request audit and CPU logit export; serialization',
    wrapper_wall='separate invoke wall includes clone/control/postchecks/CPU export; never substituted for primary request wall',
    raw_classification='ORACLE_DIAGNOSTIC: real Semantic/Cluster/lookup/policy, but occurrence eligibility bypasses production Admission',
    prefix_reference='historical R2-B3.5 is use_cache=True/output_attentions=False/one-token head/suffix-only entire decoder skip; not pooled or paired with C1')


class Runtime(CostRuntime):
    def gpu_state(self):
        state=monitor(self.a.device)
        return dict(state,gpu_uuid=state.get('torch_uuid'))

    def setup_identity(self):
        self.model_hashes={n:tensor_sha(p) for n,p in self.model.named_parameters()}
        self.parameter_stamps=tuple((n,id(p),p._version) for n,p in self.model.named_parameters())
        return digest(self.model_hashes)

    def model_guard(self,content=False):
        if tuple((n,id(p),p._version) for n,p in self.model.named_parameters())!=self.parameter_stamps:
            raise ParityFailure('model parameter identity/version changed')
        if self.model.training or list(self.model.active_adapters)!=['user_a'] or self.model.config._attn_implementation!='eager':
            raise ParityFailure('model eval/adapter/backend changed')
        if content and {n:tensor_sha(p) for n,p in self.model.named_parameters()}!=self.model_hashes:
            raise ParityFailure('model weight content changed')

    def canonical_hashes(self):
        return {f'{query}:{layer}:{role}':tensor_sha(t) for query,s in self.catalog.sources.items()
            for layer,values in s['qkv'].items() for role,t in zip('qkv',values)}

    def prepare(self,length):
        super().prepare(length)
        self.source_hashes=self.canonical_hashes()
        self.expected_binding=(tuple(self.ids),self.snapshot,self.namespace,self.weights)
        self.coverage='max'

    def guard(self,content=False):
        self.model_guard(content)
        if (tuple(self.ids),self.snapshot,self.namespace,self.weights)!=self.expected_binding: raise ParityFailure('source/target identity changed')
        self.index.guard()
        if content and self.canonical_hashes()!=self.source_hashes: raise ParityFailure('canonical source QKV content mutated')

    def invoke(self,mode,capture=False):
        start=perf_counter_ns()
        if mode=='RAW_SEMCACHE_QKV': logits,record=self.request(self.text,self.ids,'contiguous',self.coverage)
        elif mode=='SEMCACHE_NO_REUSE': logits,record=super().invoke('LOOKUP_NO_REUSE')
        elif mode=='FULL_NATIVE': logits,record=super().invoke(mode)
        else: raise ValueError(mode)
        record['wrapper_invoke_wall_ms']=(perf_counter_ns()-start)/1e6
        record['memory_accounting_note']='Torch allocated/reserved peaks are measured; inherited executor byte counters include logical slice/view row volumes and are not measured temporary allocator peaks'
        record['output_shape']=list(logits.shape)
        if list(logits.shape)!=[1,self.length,self.model.config.vocab_size]: raise MissingEvidence('all-token Logit policy mismatch')
        record['coverage_audit']=check_coverage(record,self.ids,len(self.adapter.layers))
        if mode!='RAW_SEMCACHE_QKV' and record['reused_tokens']: raise ParityFailure('control physically reused rows')
        return logits,record


class OutputAudit:
    """Untimed root/head and LoRA-child hooks, never forbidden QKV-parent hooks."""
    def __init__(self,runtime): self.r=runtime; self.roots=[]; self.heads=[]; self.lora=[]
    @contextmanager
    def hooks(self):
        handles=[]; inputs=[]
        def root(m,args,kwargs):
            x=kwargs.get('input_ids'); self.roots.append(dict(ids=x[0].detach().cpu().tolist(),
                use_cache=kwargs.get('use_cache'),output_attentions=kwargs.get('output_attentions')))
        def head_in(m,args): inputs.append(list(args[0].shape))
        def head_out(m,args,out): self.heads.append(dict(input=inputs.pop(),output=list(out.shape),dtype=str(out.dtype)))
        try:
            head=self.r.model.get_base_model().get_output_embeddings()
            handles.extend([self.r.model.register_forward_pre_hook(root,with_kwargs=True),head.register_forward_pre_hook(head_in),head.register_forward_hook(head_out)])
            for i in range(len(self.r.adapter.layers)):
                for role,module in self.r.adapter.projection_modules(i).items():
                    from semcache.models.lora_decomposition import validate_projection
                    validate_projection(module,'user_a')
                    def observed(m,args,out,i=i,role=role,owner=module):
                        self.lora.append(dict(layer=i,role=role,rows=int(out.shape[1]),scaling=owner.scaling['user_a'],adapter='user_a'))
                    handles.append(module.lora_B['user_a'].register_forward_hook(observed))
            yield self
        finally:
            for h in handles: h.remove()
    def report(self,reused):
        r=self.r
        if len(self.roots)!=1 or self.roots[0]!=dict(ids=r.ids,use_cache=False,output_attentions=True): raise ParityFailure('native call/flags/IDs differ')
        if len(self.heads)!=1 or self.heads[0]['input'][:2]!=[1,r.length] or self.heads[0]['output']!=[1,r.length,r.model.config.vocab_size]: raise ParityFailure('LM Head call/row policy mismatch')
        expected=r.length-reused
        roles={(i,q) for i in range(len(r.adapter.layers)) for q in 'qkv'} if expected else set()
        if len(self.lora)!=len(roles) or {(v['layer'],v['role']) for v in self.lora}!=roles or any(v['rows']!=expected for v in self.lora):
            raise ParityFailure('native LoRA projection rows do not match skipped QKV positions')
        return dict(root=self.roots[0],head=self.heads[0],lora=self.lora,output_policy='native_all_token_logits',output_matched_within_c1=True)


@contextmanager
def projection_observer(runtime,observed):
    """Hash one existing passive projection at a time; no disk/GPU tensor dump."""
    import torch
    cls=runtime.b.RequestMapping; original=cls.context
    @contextmanager
    def context(flow,*args,**kwargs):
        with original(flow,*args,**kwargs) as audit:
            yield audit
            observed['hashes']={}; observed['cached_rows_equal']=True
            hits=flow.hits or []
            queries={h.entry.qkv_metadata['source_query_id'] for h in hits}
            for layer,projections in audit.projections.items():
                for role,t in projections.items():
                    observed['hashes'][f'{layer}:{role}']=tensor_sha(t)
                    for query in queries:
                        selected=[h for h in hits if h.entry.qkv_metadata['source_query_id']==query]
                        source=[p for h in selected for p in range(*h.entry.positions)]
                        target=[p for h in selected for p in range(h.window.start,h.window.end)]
                        backing=runtime.catalog.sources[query]['qkv'][layer]['qkv'.index(role)]
                        a=t.index_select(1,torch.tensor(target,device=t.device))
                        b=backing.index_select(1,torch.tensor(source,device=backing.device)).to(a)
                        if not torch.equal(a,b): raise ParityFailure('assembled cached QKV differs from canonical source rows')
    with patch.object(cls,'context',context): yield


def validate(runtime):
    import torch
    runtime.guard(True); outputs={}; audits={}; records={}
    for mode in MODES:
        observer=OutputAudit(runtime); physical={}
        with observer.hooks(),projection_observer(runtime,physical): logits,record=runtime.invoke(mode)
        audits[mode]=dict(output=observer.report(record['reused_tokens']),projection=physical,coverage=record['coverage_audit'])
        outputs[mode]=logits; records[mode]=record
    # Existing legacy window executor is the reference, with the SAME selector.
    legacy_evidence={}
    with projection_observer(runtime,legacy_evidence):
        legacy,legacy_record=runtime.request(runtime.text,runtime.ids,'legacy',runtime.coverage)
    if (legacy_evidence['hashes']!=audits['RAW_SEMCACHE_QKV']['projection']['hashes'] or not torch.equal(legacy,outputs['RAW_SEMCACHE_QKV']) or
        legacy_record['projection_audit']!=records['RAW_SEMCACHE_QKV']['projection_audit'] or legacy_record['selection_hash']!=records['RAW_SEMCACHE_QKV']['selection_hash']):
        raise ParityFailure('Contiguous/Legacy QKV, Logit, selection or projection audit mismatch')
    q={mode:strict_metrics(outputs['FULL_NATIVE'],out) for mode,out in outputs.items()}
    runtime.guard(True)
    return dict(structural_valid=True,canonical_content_unchanged=True,legacy_contiguous_exact=True,
        output_policy_parity=True,audits=audits,quality=q,source_backing_copy=runtime.backing_check,
        theoretical_w3_upper_tokens=runtime.length//3*3,requested_target_tokens=expected_tokens(runtime.length,runtime.coverage),
        classification='ORACLE_DIAGNOSTIC',safe_reuse_claimed=False)


def paired(runtime,args,identity,save):
    if any(m._forward_hooks or m._forward_pre_hooks for m in runtime.model.modules()): raise MissingEvidence('hooks remain before minimal')
    rows=[]
    for phase,count,offset in [('warmup',args.warmup_runs,0),('measured',args.measured_runs,100003)]:
        for index,order in enumerate(schedule(MODES,count,args.order_seed+runtime.length+offset)):
            records={}; outputs={}
            for mode in order:
                save(context=dict(case=identity['case_id'],phase=phase,pair_index=index,mode=mode))
                runtime.guard(); outputs[mode],records[mode]=runtime.invoke(mode); runtime.guard()
            full,miss,raw=[records[m]['request_ms'] for m in MODES]
            q={mode:strict_metrics(outputs['FULL_NATIVE'],outputs[mode]) for mode in MODES}
            row=dict(identity,phase=phase,pair_index=index,instrumentation='minimal',benchmark_phase=args.phase,execution_order=list(order),
                wall_ms={m:r['request_ms'] for m,r in records.items()},wrapper_wall_ms={m:r['wrapper_invoke_wall_ms'] for m,r in records.items()},
                delta_full_raw_ms=full-raw,delta_miss_raw_ms=miss-raw,delta_full_miss_ms=full-miss,speedup_full_raw=full/raw,
                coverage=records['RAW_SEMCACHE_QKV']['coverage_audit'],quality=q,
                mapping={k:records['RAW_SEMCACHE_QKV'].get(k) for k in ('mapping_cache_hit','mapping_ms','context_setup_ms','selection_ms','cache_statistics')},
                assembly_memory=records['RAW_SEMCACHE_QKV'].get('memory'),structural_valid=True,cache_integrity_valid=True,
                assembly_memory_note=records['RAW_SEMCACHE_QKV'].get('memory_accounting_note'),
                strict_pass=all(v['allclose'] for v in q.values()),correctness_qualified_speedup=False)
            rows.append(row); outputs.clear(); save(rows=rows)
    return rows
