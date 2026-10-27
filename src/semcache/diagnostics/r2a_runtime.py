"""R2-A adapters around frozen R1 runtime; no executor algorithm changes."""
from contextlib import nullcontext
from time import perf_counter_ns
from semcache.diagnostics.r1b3_b3_runtime import Runtime
from semcache.diagnostics.r1b3_break_even import MissingEvidence,ParityFailure
from semcache.metrics.r1 import audit_projection_records,digest

SEMANTICS={
 'FULL_NATIVE':dict(path='r1_runtime.native_request',executes=['tokenize','set_adapter/eval','native PEFT Full prefill'],bypasses=['semantic','lookup','selection','impact','policy']),
 'LOOKUP_NO_REUSE':dict(path='Runtime.request(mode=lookup)',executes=['tokenize','semantic encode','cluster update','exact-w3 lookup','legacy nonoverlap selection','native-only Legacy wrapper','attention impact','admission/CHU/PBR as triggered'],bypasses=['Fast occurrence selector','physical QKV reuse','prepared mapping']),
 'CONTIGUOUS_WARM':dict(path='Runtime.request(mode=contiguous)',executes=['all Engine frontend/policy','Fast occurrence selector','warm mapping cache lookup','snapshot validation','Contiguous executor'],bypasses=['production admission eligibility for oracle source catalog']),
 'ORACLE_PRESELECTED':dict(path='native_request inside unchanged batched_v2_projection_path(contiguous=True)',executes=['tokenize','set_adapter/eval','executor snapshot validation','module patch setup/cleanup','fresh native PEFT','same cached QKV assembly'],bypasses=['semantic encoding','cluster update','exact-w3 lookup','nonoverlap/Fast selection','mapping cache lookup/build','attention impact','admission/CHU/PBR','Engine logging/control'],qualification='prevalidated identical-target diagnostic, not end-to-end SemCache service')}


class CostRuntime(Runtime):
    def prepare(self,length):
        from contextlib import contextmanager
        from semcache.diagnostics.r1b3_gemm_shape import tensor_sha
        hashes={}
        @contextmanager
        def observer():
            handles=[]
            try:
                for layer in range(len(self.adapter.layers)):
                    for role,module in self.adapter.projection_modules(layer).items():
                        def after(m,args,out,key=(layer,role)): hashes[key]=tensor_sha(out)
                        handles.append(module.register_forward_hook(after))
                yield
            finally:
                for h in handles: h.remove()
        self.source(length,observer)
        self.backing_check=[]
        for layer,values in self.catalog.sources['source']['qkv'].items():
            for role,t in zip('qkv',values):
                equal=hashes.get((layer,role))==tensor_sha(t)
                self.backing_check.append(dict(layer=layer,role=role,source_sha256=hashes.get((layer,role)),backing_equal=equal))
                if not equal: raise ParityFailure('Actual source capture/backing differ')
        self.costs['source_prefill_note']='includes R2 source-only hash/capture instrumentation; not minimal native source latency'
        import torch
        started=perf_counter_ns(); self.hits=tuple(self.selected_hits('max'))
        self.plan,_=self.caches['contiguous'].get(self.hits,self.ids,'user_a',self.namespace,self.weights,
            self.a.device,getattr(torch,self.a.dtype),mode='contiguous')
        self.oracle_prepare_ms=(perf_counter_ns()-started)/1e6
        self.costs['oracle_preselection_mapping_ms']=self.oracle_prepare_ms
        self.oracle_ids=tuple(self.ids); self.oracle_snapshot=self.snapshot
        self.index.guard()

    def invoke(self,mode,capture=False):
        import torch
        from semcache.metrics.r1_runtime import synchronize,memory_snapshot
        from semcache.edgelora.batched_mapping_optimized import batched_v2_projection_path
        if mode=='FULL_NATIVE':
            result=self.full(self.text)
            if result['ids']!=self.ids: raise MissingEvidence('FULL token mismatch')
            return result['logits'],dict(request_ms=result['timing']['request_wall_ms'],prefill_ms=result['timing']['prefill_wall_ms'],
                reused_tokens=0,projection_audit=dict(native_projection_rows_all_layers_roles=self.length*3*len(self.adapter.layers),skipped_projection_rows_all_layers_roles=0),memory=memory_snapshot(self.a.device))
        if mode in ('LOOKUP_NO_REUSE','CONTIGUOUS_WARM'):
            return self.request(self.text,self.ids,'lookup' if mode=='LOOKUP_NO_REUSE' else 'contiguous','max',capture_projections=capture)
        if mode!='ORACLE_PRESELECTED': raise ValueError(mode)
        if tuple(self.ids)!=self.oracle_ids or self.snapshot!=self.oracle_snapshot: raise MissingEvidence('Oracle target/snapshot changed; rebuild preselection')
        synchronize(self.a.device); start=perf_counter_ns()
        self.model.set_adapter('user_a'); self.model.eval()
        ids=list(self.tokenizer(self.text)['input_ids'])
        if ids!=self.ids: raise MissingEvidence('Oracle token mismatch')
        inputs=dict(input_ids=torch.tensor([ids],device=self.a.device),use_cache=False,output_attentions=True)
        with batched_v2_projection_path(self.adapter,'user_a',self.hits,self.length,mapping=self.plan,contiguous=True) as audit:
            prefill_start=perf_counter_ns()
            with torch.inference_mode(): output=self.model(**inputs)
            synchronize(self.a.device)
            prefill_ms=(perf_counter_ns()-prefill_start)/1e6
        synchronize(self.a.device); total=(perf_counter_ns()-start)/1e6
        # Same model call/flags as native_request; logits transfer is outside timer.
        logits=output.logits.detach().cpu()
        spans=[p for h in self.hits for p in range(h.window.start,h.window.end)]
        checked=audit_projection_records(list(audit.records.values()),self.length,len(self.adapter.layers),self.plan.reused,spans)
        provenance=[dict(query=h.entry.qkv_metadata['source_query_id'],source=list(h.entry.positions),target=[h.window.start,h.window.end],
            tokens=list(h.window.token_ids),prefix_token_ids_equal=tuple(self.ids[:h.window.end])==tuple(self.catalog.sources[h.entry.qkv_metadata['source_query_id']]['ids'][:h.entry.positions[1]]),
            adapter_identity_match=h.entry.qkv_metadata['source_adapter']=='user_a') for h in self.hits]
        record=dict(request_ms=total,prefill_ms=prefill_ms,selected_provenance=provenance,selection_hash=digest(provenance),
            timer_definition='tokenization/adapter/input + executor setup + native forward + cleanup; synchronized',reused_tokens=self.plan.reused,projection_audit=checked,
            mapping_cache_hit=None,mapping_status='PRESELECTED_NOT_A_CACHE_LOOKUP',memory=dict(executor=audit.memory,**memory_snapshot(self.a.device)))
        if capture: record['_qkv']={l:{r:t.detach().to('cpu',copy=True) for r,t in q.items()} for l,q in audit.projections.items()}
        return logits,record

    def validate_modes(self):
        import torch
        from semcache.edgelora.batched_mixed_projection import compare_projections
        from semcache.diagnostics.r1b3_fp16_divergence import logit_metrics
        from contextlib import contextmanager
        shapes={}
        @contextmanager
        def observer():
            handles=[]
            try:
                for layer in range(len(self.adapter.layers)):
                    for role,module in self.adapter.projection_modules(layer).items():
                        def before(m,args,key=(layer,role)): shapes[key]=list(args[0].shape)
                        handles.append(module.register_forward_pre_hook(before))
                yield
            finally:
                for h in handles: h.remove()
        full=self.full(self.text,observer=observer)['logits']; repeat,_=self.invoke('FULL_NATIVE')
        if len(shapes)!=3*len(self.adapter.layers) or any(x[0]!=1 or x[1]!=self.length for x in shapes.values()):
            raise MissingEvidence('Native Full projection shape/coverage mismatch')
        legacy,ref=self.request(self.text,self.ids,'legacy','max',capture_projections=True)
        qkv=ref.pop('_qkv'); mask=[bool(x) for x in self.plan.mask]; results={}
        for mode in ('LOOKUP_NO_REUSE','CONTIGUOUS_WARM','ORACLE_PRESELECTED'):
            logits,record=self.invoke(mode,capture=True)
            projections=record.pop('_qkv',None)
            parity=None
            if mode!='LOOKUP_NO_REUSE':
                parity=compare_projections(qkv,projections,mask)
                parity['logits_exact']=bool(torch.equal(legacy,logits))
                parity['rows_equal']=record['projection_audit']==ref['projection_audit']
                parity['selection_equal']=record['selection_hash']==ref['selection_hash']
                if not (parity['passed'] and parity['logits_exact'] and parity['rows_equal'] and parity['selection_equal']): raise ParityFailure(f'{mode} differs from Legacy')
            elif record['reused_tokens'] or record['projection_audit']['skipped_projection_rows_all_layers_roles']:
                raise ParityFailure('No-reuse physically skipped projection rows')
            results[mode]=dict(quality=logit_metrics(full,logits),executor_parity=parity,audit=record)
        return dict(full_native_projection_shapes=[dict(layer=k[0],role=k[1],shape=v) for k,v in shapes.items()],full_repeat=logit_metrics(full,repeat),modes=results,source_snapshot=self.snapshot,
            source_costs=self.costs,selected_sequence=self.plan.selected_sequence,cache_bytes=self.catalog.resident_bytes,
            backing_identity=[dict(layer=l,role=r,shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device),data_ptr=t.data_ptr())
                for source in self.catalog.sources.values() for l,ts in source['qkv'].items() for r,t in zip('qkv',ts)])


def microbench(runtime,repeats=10):
    """Actual Layer0 inputs/weights only; component times never summed into requests."""
    import torch
    from semcache.diagnostics.r1b3_gemm_shape import Layer0Inputs,metadata
    from semcache.metrics.r1_runtime import synchronize
    captured=Layer0Inputs(runtime.adapter)
    runtime.full(runtime.text,observer=captured.hooks)
    rows=[]
    def measure(name,fn,meta):
        for _ in range(3):
            with torch.inference_mode(): fn()
        for repeat in range(repeats):
            synchronize(runtime.a.device)
            events=None
            if runtime.a.device.startswith('cuda'):
                with torch.cuda.device(runtime.a.device):
                    events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            start=perf_counter_ns()
            if events:
                with torch.cuda.device(runtime.a.device): events[0].record()
            with torch.inference_mode(): value=fn()
            if events:
                with torch.cuda.device(runtime.a.device): events[1].record()
            synchronize(runtime.a.device)
            rows.append(dict(component=name,repeat=repeat,host_wall_ms=(perf_counter_ns()-start)/1e6,
                cuda_interval_ms=events[0].elapsed_time(events[1]) if events else None,
                scope='isolated component; CUDA interval includes launch gaps; not whole-request',**meta))
            del value
    for role,module in runtime.adapter.projection_modules(0).items():
        hidden=captured.inputs[role]
        for shape,x in [('full',hidden),('subset',hidden.index_select(1,runtime.plan.index_tensors[('fresh',0)]))]:
            meta=dict(role=role,shape_condition=shape,input=metadata(x),weight_identity=runtime.weights)
            measure('native_total',lambda:module(x),meta)
            measure('base',lambda:module.get_base_layer()(x),meta)
            def delta():
                a,b=module.lora_A['user_a'],module.lora_B['user_a']
                return b(a(module.lora_dropout['user_a'](module._cast_input_dtype(x,a.weight.dtype))))*module.scaling['user_a']
            measure('lora_delta',delta,meta)
        for group in runtime.plan.groups:
            source=group['backing'][0]['qkv'.index(role)]
            out=torch.empty((1,runtime.length,module.out_features),device=hidden.device,dtype=hidden.dtype)
            if group['is_contiguous']:
                s,t=group['source'][0],group['target'][0]; n=len(group['source'])
                def copy(): out[:,t:t+n]=source[:,s:s+n].to(out)
                measure('contiguous_cached_slice_conversion_copy',copy,dict(role=role,source=metadata(source),output=metadata(out),copied_bytes=n*module.out_features*source.element_size()))
    measure('snapshot_validation',lambda:runtime.descriptor.validate(runtime.hits),{})
    measure('warm_mapping_lookup',lambda:runtime.caches['contiguous'].get(runtime.hits,runtime.ids,'user_a',runtime.namespace,runtime.weights,runtime.a.device,getattr(torch,runtime.a.dtype),mode='contiguous'),{})
    captured.inputs.clear()
    return rows
