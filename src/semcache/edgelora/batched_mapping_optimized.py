"""R1-B3-B2.1 opt-in source descriptors and exact-target mapping cache."""
from collections import OrderedDict
from contextlib import contextmanager
from time import perf_counter_ns
from semcache.diagnostics.r1b3_fast_selection import tensor_stamp, prefix_limit, owned_python_bytes

class SetupBook:
    STAGES=('source_backing_descriptor_preparation','source_snapshot_validation','target_hit_to_position_mapping','source_group_construction','cpu_index_tensor_construction','device_index_preparation','module_forward_patch_setup','context_cleanup','batched_gather_scatter_execution','mapping_setup_total','warm_request')
    def __init__(self): self.values={}; self.calls={}; self.cuda_events={}
    @contextmanager
    def span(self,name):
        start=perf_counter_ns()
        try: yield
        finally:
            self.values[name]=self.values.get(name,0.)+(perf_counter_ns()-start)/1e6
            self.calls[name]=self.calls.get(name,0)+1
    def export(self):
        output={}
        for k in self.STAGES:
            events=self.cuda_events.get(k,[])
            output[k]=dict(host_ms=self.values.get(k),cuda_ms=sum(a.elapsed_time(b) for a,b in events) if events else None,
                calls=self.calls.get(k),status='MEASURED' if k in self.values else 'UNAVAILABLE',
                scope='CUDA events resolved after enclosing request sync; never summed with host time')
        return output

class SourceBackingDescriptors:
    """Source-only proofs; retain references but never clone source QKV."""
    def __init__(self,index,book=None):
        start=perf_counter_ns(); self.index=index; self.catalog=index.catalog
        self.namespace=index.catalog.namespace; self.weight_identity=index.catalog.weight_identity; self.storage=index.catalog.storage
        self.layout=tuple((k,tuple(v)) for k,v in sorted(index.layout.items())); self.snapshot_id=id(index.catalog)
        self.owner_stamps=tuple((t,tensor_stamp(t)) for t,_ in index.owners)
        self.entry_guards=tuple(index.entry_guards); self.source_guards=tuple(index.source_guards); self.bucket_guards=tuple(index.index_guard)
        self.view_stamps={id(t):stamp for t,stamp in index.view_guards}
        self.counts=(len(index.sources),len(index.entry_guards),len(index.index_guard))
        self.build_ms=(perf_counter_ns()-start)/1e6
        excluded={id(index),id(self.catalog),*(id(t) for t,_ in self.owner_stamps),*(id(t) for t,_ in index.view_guards)}
        excluded.update(id(x) for x in index.entry_guards); excluded.update(id(x) for x in index.source_guards)
        excluded.update(id(x) for x in index.index_guard); excluded.update(id(x) for x in index.owners)
        excluded.update(id(stamp) for _,stamp in index.view_guards)
        for source in self.catalog.sources.values():
            excluded.update((id(source),id(source.get('ids')),id(source.get('qkv'))))
            for qkv in source.get('qkv',{}).values():
                excluded.add(id(qkv)); excluded.update(id(t) for t in qkv)
        for entry in self.catalog.entries.values():
            excluded.update((id(entry),id(entry.tensors),id(entry.qkv_metadata),id(entry.token_ids),id(entry.positions)))
            excluded.update(id(qkv) for qkv in entry.tensors.values())
        self.python_bytes_estimate=owned_python_bytes(self.__dict__,excluded); self.additional_qkv_bytes=0
        if book:
            book.values['source_backing_descriptor_preparation']=self.build_ms; book.calls['source_backing_descriptor_preparation']=1
    def validate(self,hits,book=None):
        start=perf_counter_ns(); c=self.catalog; i=self.index
        if (id(c)!=self.snapshot_id or c.namespace!=self.namespace or c.weight_identity!=self.weight_identity or c.storage!=self.storage
            or tuple((k,tuple(v)) for k,v in sorted(i.layout.items()))!=self.layout
            or (len(c.sources),len(c.entries),len(c.index))!=self.counts): raise ValueError('Source snapshot changed; rebuild descriptors')
        for qid,source,ids,mapping,pairs in self.source_guards:
            if c.sources.get(qid) is not source or source.get('ids') is not ids or source.get('qkv') is not mapping or source.get('adapter')!=i.sources[qid][1] or len(mapping)!=len(pairs):
                raise ValueError('Source identity/metadata changed')
            for layer,qkv in pairs:
                if mapping.get(layer) is not qkv: raise ValueError('Source backing mapping changed')
        for tensor,stamp in self.owner_stamps:
            if tensor_stamp(tensor)!=stamp: raise ValueError('Source QKV mutated; rebuild descriptors')
        for identity,e,cluster,tokens,positions,metadata,fields,mapping,pairs in self.entry_guards:
            if (c.entries.get(identity) is not e or e.cluster_id!=cluster or e.token_ids is not tokens or e.positions is not positions
                or e.qkv_metadata is not metadata or e.tensors is not mapping or len(mapping)!=len(pairs)
                or any(metadata.get(k) is not v for k,v in fields.items())): raise ValueError('Occurrence structure changed')
            if any(mapping.get(layer) is not qkv for layer,qkv in pairs): raise ValueError('Occurrence QKV references changed')
        for key,bucket,contents in self.bucket_guards:
            if c.index.get(key) is not bucket or tuple(bucket)!=contents: raise ValueError('Occurrence bucket changed')
        for hit in hits:
            e=hit.entry; identity=e.qkv_metadata.get('occurrence_identity')
            if c.entries.get(identity) is not e: raise ValueError('Selected occurrence stale')
            for qkv in e.tensors.values():
                for tensor in qkv:
                    expected=self.view_stamps.get(id(tensor))
                    # FastOccurrenceIndex stores torch.device, tensor_stamp stores
                    # its string form. Normalize representation, retaining every
                    # pointer/shape/stride/dtype/device check.
                    if expected is None or tensor_stamp(tensor)[:5]!=(*expected[:4],str(expected[4])):
                        raise ValueError('Selected QKV view rebound or changed')
        elapsed=(perf_counter_ns()-start)/1e6
        if book:
            book.values['source_snapshot_validation']=book.values.get('source_snapshot_validation',0.)+elapsed
            book.calls['source_snapshot_validation']=book.calls.get('source_snapshot_validation',0)+1
        return elapsed

class PreparedMapping:
    """Exact target geometry and lazily built reusable source/device indices."""
    def __init__(self,descriptor,hits,target_ids,adapter,namespace,weights,device,dtype,book=None):
        started=perf_counter_ns(); self.descriptor=descriptor; self.index=descriptor.index; self.catalog=descriptor.catalog
        self.adapter=adapter; self.namespace=namespace; self.weights=weights; self.device=str(device); self.dtype=str(dtype)
        self.target_ids=tuple(target_ids); self.length=len(self.target_ids); self.hit_ids=tuple(id(h) for h in hits)
        if namespace!=descriptor.namespace or weights!=descriptor.weight_identity: raise ValueError('Model/adapter namespace mismatch')
        if not self.length: raise ValueError('Target sequence must be nonempty')
        prefix={q:prefix_limit(s['ids'],self.target_ids) for q,s in self.catalog.sources.items()}
        grouped={}; mask=bytearray(self.length); ordered=[]
        for hit in hits:
            e,w=hit.entry,hit.window; m=e.qkv_metadata; q=m.get('source_query_id'); source=self.catalog.sources.get(q); begin,end=e.positions
            if (source is None or self.catalog.entries.get(m.get('occurrence_identity')) is not e or (begin,end)!=(w.start,w.end)
                or end-begin!=3 or not 0<=w.start<w.end<=self.length or end>len(source['ids'])
                or tuple(w.token_ids)!=e.token_ids or self.target_ids[w.start:w.end]!=e.token_ids or end>prefix[q]
                or source['adapter']!=adapter or m.get('source_adapter')!=adapter or m.get('model_namespace')!=namespace
                or m.get('weight_identity')!=weights or m.get('component_scope')!='total_qkv'):
                raise ValueError('Selected hit failed exact position/prefix/identity validation')
            if any(mask[w.start:w.end]): raise ValueError('Selected target windows overlap')
            mask[w.start:w.end]=b'\x01'*3
            g=grouped.setdefault(q,dict(query=q,backing=source['qkv'],source=[],target=[]))
            g['source'].extend((begin,begin+1,begin+2)); g['target'].extend((w.start,w.start+1,w.start+2))
            ordered.append((q,begin,end,w.start,w.end,e.token_ids,id(e)))
        if book:
            book.values['target_hit_to_position_mapping']=book.values.get('target_hit_to_position_mapping',0.)+(perf_counter_ns()-started)/1e6
            book.calls['target_hit_to_position_mapping']=book.calls.get('target_hit_to_position_mapping',0)+1
        group_start=perf_counter_ns(); self.groups=[]; self.contiguous=[]
        for g in grouped.values():
            g['source']=tuple(g['source']); g['target']=tuple(g['target'])
            g['is_contiguous']=bool(g['source'] and all(b==a+1 for a,b in zip(g['source'],g['source'][1:]))
                                      and all(b==a+1 for a,b in zip(g['target'],g['target'][1:])))
            self.groups.append(g); self.contiguous.append(g['is_contiguous'])
        self.groups=tuple(self.groups); self.fresh=tuple(i for i,x in enumerate(mask) if not x)
        self.mask=bytes(mask); self.reused=self.length-len(self.fresh); self.selected_sequence=tuple(ordered)
        if book:
            book.values['source_group_construction']=book.values.get('source_group_construction',0.)+(perf_counter_ns()-group_start)/1e6
            book.calls['source_group_construction']=book.calls.get('source_group_construction',0)+1
        self.index_tensors={}; self.prepared_modes=set(); self.cpu_index_build_ms=0.; self.device_index_build_ms=0.; self.cpu_index_bytes=0; self.cache_hits=0
        self.build_ms=(perf_counter_ns()-started)/1e6
        excluded={id(self.catalog),id(self.index),id(descriptor),id(self.target_ids)}
        for group in self.groups:
            excluded.add(id(group['backing']))
            for qkv in group['backing'].values(): excluded.add(id(qkv)); excluded.update(id(t) for t in qkv)
        for occurrence in self.selected_sequence:
            excluded.update((id(occurrence[0]),id(occurrence[-2])))
        self.python_bytes_estimate=owned_python_bytes((self.target_ids,self.groups,self.fresh,self.mask,self.selected_sequence),excluded)
    def cache_identity(self,hits,target_ids,adapter,namespace,weights,device,dtype):
        owners=tuple((id(t),s[0],s[1],s[2],str(s[3]),s[5]) for t,s in self.descriptor.owner_stamps)
        seq=tuple((id(h.entry),h.entry.qkv_metadata['source_query_id'],tuple(h.entry.positions),h.window.start,h.window.end,tuple(h.window.token_ids)) for h in hits)
        return (self.descriptor.snapshot_id,owners,tuple(target_ids),seq,adapter,namespace,weights,str(device),str(dtype),len(target_ids))
    def prepare_indices(self,output_device,book=None,mode='gather'):
        if mode not in ('gather','contiguous'): raise ValueError('Invalid mapping assembly mode')
        if mode in self.prepared_modes: return False
        import torch
        start=perf_counter_ns(); values=[]
        for n,g in enumerate(self.groups):
            if mode=='gather' or not g['is_contiguous']:
                values.extend(((('source',n),g['source']),(('target',n),g['target'])))
        if not self.index_tensors: values.append((('fresh',0),self.fresh))
        cpu={k:torch.tensor(v,dtype=torch.long) for k,v in values if k not in self.index_tensors}
        self.cpu_index_bytes+=sum(t.numel()*t.element_size() for t in cpu.values())
        cpu_elapsed=(perf_counter_ns()-start)/1e6; self.cpu_index_build_ms+=cpu_elapsed
        if book:
            book.values['cpu_index_tensor_construction']=book.values.get('cpu_index_tensor_construction',0.)+cpu_elapsed
            book.calls['cpu_index_tensor_construction']=book.calls.get('cpu_index_tensor_construction',0)+1
        start=perf_counter_ns(); prepared={}
        for key,t in cpu.items():
            if key[0]=='source': device=next(iter(self.groups[key[1]]['backing'].values()))[0].device
            else: device=output_device
            pair=None
            if book and str(device).startswith('cuda'):
                with torch.cuda.device(device):
                    pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)); pair[0].record()
            prepared[key]=t.to(device=device)
            if pair:
                with torch.cuda.device(device): pair[1].record()
                book.cuda_events.setdefault('device_index_preparation',[]).append(pair)
        self.index_tensors.update(prepared); device_elapsed=(perf_counter_ns()-start)/1e6; self.device_index_build_ms+=device_elapsed
        self.prepared_modes.add(mode)
        if book:
            book.values['device_index_preparation']=book.values.get('device_index_preparation',0.)+device_elapsed
            book.calls['device_index_preparation']=book.calls.get('device_index_preparation',0)+1
        return True

class PreparedMappingCache:
    """Bounded LRU keyed by exact source snapshot, Target and hit sequence."""
    def __init__(self,descriptor,capacity=4):
        if capacity<1: raise ValueError('Positive mapping cache capacity required')
        self.descriptor=descriptor; self.capacity=capacity; self.items=OrderedDict(); self.hits=0; self.misses=0; self.evictions=0; self.estimated_bytes=0
    def get(self,hits,target_ids,adapter,namespace,weights,device,dtype,book=None,mode='gather',validate=True):
        start=perf_counter_ns()
        if validate: self.descriptor.validate(hits,book)
        # A tiny probe computes a collision-free tuple key without building mapping objects.
        probe=PreparedMapping.__new__(PreparedMapping); probe.descriptor=self.descriptor
        key=probe.cache_identity(hits,target_ids,adapter,namespace,weights,device,dtype)
        self._addtime(book,'target_hit_to_position_mapping',(perf_counter_ns()-start)/1e6)
        value=self.items.get(key)
        if value is not None:
            old_bytes=sum(t.numel()*t.element_size() for t in value.index_tensors.values())
            value.prepare_indices(device,book,mode)
            self.estimated_bytes+=sum(t.numel()*t.element_size() for t in value.index_tensors.values())-old_bytes; self.items.move_to_end(key); self.hits+=1; value.cache_hits+=1
            return value,True
        self.misses+=1; value=PreparedMapping(self.descriptor,hits,target_ids,adapter,namespace,weights,device,dtype,book)
        value.prepare_indices(device,book,mode); self.items[key]=value
        self._set_bytes(1,value)
        while len(self.items)>self.capacity:
            _,removed=self.items.popitem(last=False); self._set_bytes(-1,removed); self.evictions+=1
        return value,False
    def _set_bytes(self,multiplier,value):
        self.estimated_bytes+=multiplier*(value.python_bytes_estimate+sum(t.numel()*t.element_size() for t in value.index_tensors.values()))
    @staticmethod
    def _addtime(book,name,value):
        if book: book.values[name]=book.values.get(name,0.)+value; book.calls[name]=book.calls.get(name,0)+1
    def stats(self):
        total=self.hits+self.misses
        return dict(hits=self.hits,misses=self.misses,hit_rate=self.hits/total if total else None,entries=len(self.items),
                    estimated_memory_bytes=self.estimated_bytes,evictions=self.evictions,
                    invalidation='descriptor rejects changed snapshot; exact target/hit key prevents cross-target reuse')

@contextmanager
def batched_v2_projection_path(adapter,adapter_name,hits,sequence_length,*,mapping,setup_book=None,executor_book=None,contiguous=False):
    """B2.1 executor using a validated exact-target PreparedMapping.

    Gather/scatter matches B2 V1. Optional contiguous assignments use identical
    selected QKV and only when source and destination offsets are each contiguous.
    """
    import torch
    from semcache.edgelora.batched_mixed_projection import span as exec_span
    from semcache.models.lora_decomposition import validate_projection
    if adapter_name!=mapping.adapter or sequence_length!=mapping.length:
        raise ValueError('Executor adapter/sequence differs from prepared mapping')
    expected=tuple((h.entry.qkv_metadata['source_query_id'],*h.entry.positions,h.window.start,h.window.end,
                    h.entry.token_ids,id(h.entry)) for h in hits)
    if expected!=mapping.selected_sequence: raise ValueError('Executor hits differ from cached mapping key')
    if not mapping.index_tensors: raise ValueError('Mapping indices were not prepared')
    setup_started=perf_counter_ns()
    # The runner may defer this one validation from cache lookup to executor
    # setup. This guards direct callers and closes the lookup/use mutation gap.
    mapping.descriptor.validate(hits,setup_book)
    mask=torch.tensor(list(mapping.mask),dtype=torch.bool)
    fresh_index=mapping.index_tensors[('fresh',0)]
    audit=type('Audit',(),{})()
    audit.records={}; audit.projections={}; audit.reused_mask=mask; audit.cuda_event_pairs=[]
    audit.memory=dict(index_tensor_bytes=sum(t.numel()*t.element_size() for t in mapping.index_tensors.values())+mask.numel()*mask.element_size(),
        temp_tensor_bytes=0,gather_allocation_bytes_total=0,conversion_allocation_bytes_total=0,
        gather_calls=0,scatter_calls=0,conversion_calls=0,conversion_copy_calls=0,h2d_cache_bytes=0,
        contiguous_copy_calls=0,contiguous_copy_bytes=0,fallback_group_count=0,kernel_calls=None,
        kernel_count_status='NOT_PROFILED; PyTorch op count is not kernel count')
    if setup_book:
        setup_book.values['module_forward_patch_setup']=0.; setup_book.calls['module_forward_patch_setup']=0
    originals=[]

    def replacement(layer,name,module,native):
        def forward(hidden_states,*args,**kwargs):
            key=(layer,name)
            if key in audit.records: raise RuntimeError('Use one forward per mixed context')
            if args or kwargs or hidden_states.shape!=(1,sequence_length,module.in_features): raise ValueError('Batch-one prefill required')
            with torch.inference_mode(),exec_span(executor_book,'full_mixed_projection'):
                with exec_span(executor_book,'fresh_index_preparation'):
                    fresh_inputs=hidden_states.index_select(1,fresh_index)
                with exec_span(executor_book,'fresh_native_qkv'):
                    fresh=native(fresh_inputs) if fresh_index.numel() else None
                dtype=fresh.dtype if fresh is not None else module.get_base_layer().weight.dtype
                with exec_span(executor_book,'output_allocation'):
                    output=torch.empty((1,sequence_length,module.out_features),device=hidden_states.device,dtype=dtype)
                if fresh is not None:
                    with exec_span(executor_book,'fresh_assembly'): output.index_copy_(1,fresh_index,fresh)
                assembly_started=perf_counter_ns() if setup_book else None
                for number,group in enumerate(mapping.groups):
                    with exec_span(executor_book,'source_reference_resolution'):
                        source=group['backing'][layer]['qkv'.index(name)]
                        if source.shape!=(1,len(mapping.catalog.sources[group['query']]['ids']),module.out_features):
                            raise ValueError('Source backing shape mismatch')
                        if source.dtype!=mapping.index.layout[layer]['qkv'.index(name)][1]: raise ValueError('Source dtype mismatch')
                    if contiguous and group['is_contiguous']:
                        source_start,target_start=group['source'][0],group['target'][0]
                        size=len(group['source']); source_end,target_end=source_start+size,target_start+size
                        with exec_span(executor_book,'cached_gather'):
                            rows=source[:,source_start:source_end]
                        with exec_span(executor_book,'source_conversion'):
                            converted=rows.to(output)
                        with exec_span(executor_book,'cached_scatter'):
                            output[:,target_start:target_end]=converted
                        audit.memory['contiguous_copy_calls']+=1; audit.memory['contiguous_copy_bytes']+=rows.numel()*rows.element_size()
                    else:
                        if contiguous: audit.memory['fallback_group_count']+=1
                        src_index=mapping.index_tensors[('source',number)]; dst_index=mapping.index_tensors[('target',number)]
                        with exec_span(executor_book,'cached_gather'): rows=source.index_select(1,src_index)
                        with exec_span(executor_book,'source_conversion'): converted=rows.to(output)
                        with exec_span(executor_book,'cached_scatter'): output.index_copy_(1,dst_index,converted)
                        audit.memory['gather_calls']+=1; audit.memory['scatter_calls']+=1
                    copied=converted is not rows; size=rows.numel()*rows.element_size()
                    converted_size=converted.numel()*converted.element_size() if copied else 0
                    audit.memory['conversion_calls']+=1; audit.memory['conversion_copy_calls']+=int(copied)
                    audit.memory['gather_allocation_bytes_total']+=size
                    audit.memory['conversion_allocation_bytes_total']+=converted_size
                    audit.memory['temp_tensor_bytes']=max(audit.memory['temp_tensor_bytes'],size+converted_size)
                    if source.device.type=='cpu' and output.device.type=='cuda': audit.memory['h2d_cache_bytes']+=size
                    del rows,converted
                if setup_book and assembly_started is not None:
                    setup_book.values['batched_gather_scatter_execution']=setup_book.values.get('batched_gather_scatter_execution',0.)+(perf_counter_ns()-assembly_started)/1e6
                    setup_book.calls['batched_gather_scatter_execution']=setup_book.calls.get('batched_gather_scatter_execution',0)+1
                audit.records[key]=dict(layer=layer,tensor_type=name,full_sequence_length=sequence_length,
                    native_projection_rows=int(fresh_index.numel()),reconstructed_projection_rows=0,
                    reused_projection_rows=mapping.reused,expected_saved_projection_positions=mapping.reused,
                    native_call_count=int(fresh is not None),native_positions=list(mapping.fresh),component_scope='total_qkv')
                audit.projections.setdefault(layer,{})[name]=output.detach()
                return output
        return forward
    try:
        patch_started=perf_counter_ns()
        with setup_book.span('module_forward_patch_setup') if setup_book else _nullspan():
            for layer in range(len(adapter.layers)):
                for name,module in adapter.projection_modules(layer).items():
                    validate_projection(module,adapter_name)
                    if module._forward_hooks or module._forward_pre_hooks or 'forward' in module.__dict__:
                        raise ValueError('Mixed path requires uninstrumented projections')
                    if any(layer not in group['backing'] for group in mapping.groups): raise ValueError('Missing source layer')
                    originals.append(module); module.forward=replacement(layer,name,module,module.forward)
        audit.module_patch_setup_ms=(perf_counter_ns()-patch_started)/1e6
        audit.context_setup_before_forward_ms=(perf_counter_ns()-setup_started)/1e6
        yield audit
        if len(audit.records)!=3*len(adapter.layers): raise RuntimeError('Not all QKV projections executed')
        audit.memory['temp_tensor_bytes']+=audit.memory['index_tensor_bytes']
        audit.memory['temporary_accounting']='logical peak gathered rows/conversion + retained per-request mask/index tensors; excludes common native/output/capture buffers and allocator reservation'
    finally:
        cleanup=perf_counter_ns()
        for module in originals: del module.forward
        audit.context_cleanup_ms=(perf_counter_ns()-cleanup)/1e6
        if setup_book:
            setup_book.values['context_cleanup']=setup_book.values.get('context_cleanup',0.)+audit.context_cleanup_ms
            setup_book.calls['context_cleanup']=setup_book.calls.get('context_cleanup',0)+1


@contextmanager
def _nullspan():
    yield


def balanced_orders(modes,iterations,seed):
    """Seeded cyclic balance over all orders (AB/BA, and all 3/4-mode orders)."""
    import itertools, random
    modes=tuple(modes)
    if not modes or len(set(modes))!=len(modes) or iterations<1: raise ValueError('Invalid order schedule')
    orders=list(itertools.permutations(modes)); random.Random(seed).shuffle(orders)
    return [list(orders[i%len(orders)]) for i in range(iterations)]

def paired_statistics(baseline,optimized,seed=0,bootstrap_samples=5000):
    """Paired improvement is baseline minus optimized; positive favors optimized."""
    import random, statistics
    if len(baseline)!=len(optimized) or not baseline: raise ValueError('Paired samples required')
    diffs=[float(a)-float(b) for a,b in zip(baseline,optimized)]
    ordered=sorted(diffs)
    def quantile(q):
        pos=(len(ordered)-1)*q; lo=int(pos); hi=min(lo+1,len(ordered)-1)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)
    rng=random.Random(seed); means=[]
    for _ in range(bootstrap_samples): means.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs)/len(diffs))
    means.sort()
    def bq(q):
        pos=(len(means)-1)*q; lo=int(pos); hi=min(lo+1,len(means)-1)
        return means[lo]+(means[hi]-means[lo])*(pos-lo)
    return dict(count=len(diffs),mean=statistics.mean(diffs),median=statistics.median(diffs),
        std=statistics.stdev(diffs) if len(diffs)>1 else 0.,p95=quantile(.95),min=min(diffs),max=max(diffs),
        optimized_win_count=sum(d>0 for d in diffs),optimized_win_rate=sum(d>0 for d in diffs)/len(diffs),
        bootstrap_mean_ci95=[bq(.025),bq(.975)] if len(diffs)>1 else None,bootstrap_samples=bootstrap_samples if len(diffs)>1 else 0,
        bootstrap_seed=seed,independent_run_count=1,inference='within-run paired bootstrap; exploratory until independent runs are aggregated')
