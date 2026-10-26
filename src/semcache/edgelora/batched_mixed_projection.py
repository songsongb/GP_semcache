"""Opt-in ORACLE diagnostic executor. Production mixed_projection is unchanged.

Only assembly changes: native PEFT sees the same fresh input shape/order. Owners
are retained by the plan; no target FULL projections are accepted as backing.
"""
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from time import perf_counter_ns
from types import SimpleNamespace
import inspect
import sys

from semcache.diagnostics.r1b3_fast_selection import prefix_limit, owned_python_bytes
from semcache.models.lora_decomposition import validate_projection
from semcache.edgelora.mixed_projection import mixed_projection_path as ORIGINAL_MIXED

STAGES = ('full_mixed_projection', 'source_reference_resolution', 'source_conversion',
          'window_conversion_assignment', 'fresh_index_preparation', 'fresh_native_qkv',
          'output_allocation', 'fresh_assembly', 'cached_gather', 'cached_scatter')


class ExecutorBook:
    """Events resolved only AFTER the enclosing synchronized request. Nested."""
    def __init__(self, device):
        self.device = device
        self.host = defaultdict(float); self.events = defaultdict(list); self.calls = defaultdict(int)

    @contextmanager
    def span(self, name):
        import torch
        pair = None
        if str(self.device).startswith('cuda'):
            with torch.cuda.device(self.device):
                pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                pair[0].record()
        start = perf_counter_ns()
        try: yield
        finally:
            self.host[name] += (perf_counter_ns()-start)/1e6
            self.calls[name] += 1
            if pair:
                with torch.cuda.device(self.device): pair[1].record()
                self.events[name].append(pair)

    def export(self):
        return {s:dict(host_ms=self.host.get(s),
            cuda_ms=sum(a.elapsed_time(b) for a,b in self.events[s]) if self.events[s] else None,
            calls=self.calls.get(s),status='MEASURED' if s in self.host else 'UNAVAILABLE',
            scope='inclusive projection forwards; do not add to child stages' if s=='full_mixed_projection' else 'projection subregion')
            for s in STAGES}


def span(book, name): return book.span(name) if book else nullcontext()


class PositionMapping:
    """Build from selected hits only, never select or invent additional candidates.

    Fast index owns validated versioned source tensors. Its guard rechecks owner
    and occurrence views, including .data rebinding. Multiple queries are grouped
    in first-seen order, with disjoint target indices. Noncontiguous owners work.
    """
    def __init__(self, index, hits, target_ids, adapter_name, namespace, weight_identity):
        started = perf_counter_ns()
        index.guard()
        self.adapter_name=adapter_name
        self.index = index; self.hits = tuple(hits); self.target_ids = tuple(target_ids)
        self.length = len(self.target_ids); self.mask = [False]*self.length
        if not self.length: raise ValueError('Nonempty target required')
        c = index.catalog
        if namespace != c.namespace or weight_identity != c.weight_identity:
            raise ValueError('Target namespace/weights mismatch')
        prefixes = {q:prefix_limit(s['ids'],self.target_ids) for q,s in c.sources.items()}
        groups = {}
        for h in self.hits:
            e,w = h.entry,h.window; m=e.qkv_metadata
            identity=m.get('occurrence_identity'); q=m.get('source_query_id')
            if c.entries.get(identity) is not e or q not in c.sources:
                raise ValueError('Selected occurrence not in validated snapshot')
            source = c.sources[q]; s,t=e.positions
            if (not 0<=w.start<w.end<=self.length or (s,t)!=(w.start,w.end)
                or t>len(source['ids']) or t-s!=3 or any(self.mask[w.start:w.end])
                or tuple(w.token_ids)!=e.token_ids or e.token_ids!=self.target_ids[w.start:w.end]
                or t>prefixes[q] or adapter_name!=source['adapter']
                or m['component_scope']!='total_qkv'):
                raise ValueError('Invalid overlap/position/token/prefix/adapter mapping')
            self.mask[w.start:w.end]=[True]*(w.end-w.start)
            g=groups.setdefault(q,dict(query=q,backing=source['qkv'],source=[],target=[]))
            g['source'].extend(range(s,t)); g['target'].extend(range(w.start,w.end))
        for group in groups.values():
            group['source']=tuple(group['source']); group['target']=tuple(group['target'])
        self.groups=tuple(groups.values())
        self.fresh=tuple(i for i,x in enumerate(self.mask) if not x)
        self.reused=sum(self.mask)
        self.python_bytes_estimate=owned_python_bytes(
            [self.mask,self.fresh,[(g['query'],g['source'],g['target']) for g in self.groups]],
            {id(g['query']) for g in self.groups})
        self.build_ms=(perf_counter_ns()-started)/1e6

    def evidence(self):
        return dict(length=self.length,reused_tokens=self.reused,fresh_positions=list(self.fresh),
            groups=[dict(source_query=g['query'],source_positions=g['source'],target_positions=g['target']) for g in self.groups],
            python_bytes_estimate=self.python_bytes_estimate,build_ms=self.build_ms,
            classification='ORACLE_DIAGNOSTIC')


@contextmanager
def batched_mixed_projection_path(adapter, adapter_name, hits, sequence_length, *, mapping,
                                  book=None, measure_cuda=False, fresh_projection=None):
    """Grouped gather/scatter. No full-shape projection or LoRA reconstruction."""
    import torch
    if fresh_projection is not None: raise ValueError('B2 does not alter native projection')
    if measure_cuda: raise ValueError('Use ExecutorBook breakdown; do not nest legacy event protocol')
    if adapter_name!=mapping.adapter_name: raise ValueError('Mapping adapter differs from executor')
    if sequence_length!=mapping.length or len(hits)!=len(mapping.hits) or any(a is not b for a,b in zip(hits,mapping.hits)):
        raise ValueError('Executor hits differ from validated position mapping')
    # Mapping is built immediately before this context; validate again for callers
    # that deliberately retain plans between requests. Cost is inside warm timing.
    mapping.index.guard()
    mask=torch.tensor(mapping.mask,dtype=torch.bool)
    fresh_positions=torch.tensor(mapping.fresh,dtype=torch.long)
    audit=SimpleNamespace(records={},projections={},reused_mask=mask,cuda_event_pairs=[],
        memory=dict(index_tensor_bytes=mask.numel()*mask.element_size()+fresh_positions.numel()*fresh_positions.element_size(),gather_allocation_bytes_total=0,
            conversion_allocation_bytes_total=0,temp_tensor_bytes=0,h2d_cache_bytes=0,
            gather_calls=0,scatter_calls=0,conversion_calls=0,conversion_copy_calls=0,
            kernel_calls=None,kernel_count_status='NOT_PROFILED; operation counts are not kernel counts'))
    indices={}; originals=[]

    def device_index(kind, group, positions, device):
        key=(kind,group,str(device))
        if key not in indices:
            indices[key]=torch.tensor(positions,dtype=torch.long,device=device)
            audit.memory['index_tensor_bytes']+=indices[key].numel()*indices[key].element_size()
        return indices[key]

    def replacement(layer,name,module,native):
        def forward(hidden_states,*args,**kwargs):
            key=(layer,name)
            if key in audit.records: raise RuntimeError('Use one forward per mixed context')
            if args or kwargs or hidden_states.shape!=(1,sequence_length,module.in_features):
                raise ValueError('Mixed path requires batch-one prefill hidden states only')
            with span(book,'full_mixed_projection'), torch.inference_mode():
                with span(book,'fresh_index_preparation'):
                    idx=device_index('fresh',None,mapping.fresh,hidden_states.device)
                    fresh_inputs=hidden_states.index_select(1,idx)
                with span(book,'fresh_native_qkv'):
                    fresh=native(fresh_inputs) if len(idx) else None
                dtype=fresh.dtype if fresh is not None else module.get_base_layer().weight.dtype
                with span(book,'output_allocation'):
                    output=torch.empty((1,sequence_length,module.out_features),device=hidden_states.device,dtype=dtype)
                if fresh is not None:
                    with span(book,'fresh_assembly'): output.index_copy_(1,idx,fresh)
                for number,g in enumerate(mapping.groups):
                    with span(book,'source_reference_resolution'):
                        source=g['backing'][layer]['qkv'.index(name)]
                        if (source.shape!=(1,len(mapping.index.catalog.sources[g['query']]['ids']),module.out_features)
                            or source.dtype!=mapping.index.layout[layer]['qkv'.index(name)][1]):
                            raise ValueError('Backing shape/dtype mismatch')
                    with span(book,'fresh_index_preparation'):
                        si=device_index('source',number,g['source'],source.device)
                        ti=device_index('target',number,g['target'],output.device)
                    with span(book,'cached_gather'): rows=source.index_select(1,si)
                    with span(book,'source_conversion'): converted=rows.to(output)
                    with span(book,'cached_scatter'): output.index_copy_(1,ti,converted)
                    size=rows.numel()*rows.element_size(); copied=converted is not rows
                    converted_size=converted.numel()*converted.element_size() if copied else 0
                    audit.memory['gather_calls']+=1; audit.memory['scatter_calls']+=1
                    audit.memory['conversion_calls']+=1; audit.memory['conversion_copy_calls']+=int(copied)
                    audit.memory['gather_allocation_bytes_total']+=size
                    audit.memory['conversion_allocation_bytes_total']+=converted_size
                    audit.memory['temp_tensor_bytes']=max(audit.memory['temp_tensor_bytes'],size+converted_size)
                    if source.device.type=='cpu' and output.device.type=='cuda':
                        audit.memory['h2d_cache_bytes']+=size
                    del converted,rows
                audit.records[key]=dict(layer=layer,tensor_type=name,full_sequence_length=sequence_length,
                    native_projection_rows=len(idx),reconstructed_projection_rows=0,
                    reused_projection_rows=mapping.reused,expected_saved_projection_positions=mapping.reused,
                    native_call_count=int(fresh is not None),native_positions=fresh_positions.tolist(),component_scope='total_qkv')
                audit.projections.setdefault(layer,{})[name]=output.detach()
                return output
        return forward
    try:
        for l in range(len(adapter.layers)):
            for name,module in adapter.projection_modules(l).items():
                validate_projection(module,adapter_name)
                if module._forward_hooks or module._forward_pre_hooks or 'forward' in module.__dict__:
                    raise ValueError('Mixed path requires uninstrumented projections')
                for g in mapping.groups:
                    if l not in g['backing']: raise ValueError('Missing source layer')
                originals.append(module); module.forward=replacement(l,name,module,module.forward)
        yield audit
        if len(audit.records)!=3*len(adapter.layers): raise RuntimeError('Not all projections executed')
        audit.memory['gather_conversion_live_bytes_upper_bound']=audit.memory['temp_tensor_bytes']
        audit.memory['temp_tensor_bytes']+=audit.memory['index_tensor_bytes']
        audit.memory['temporary_accounting']='logical upper bound: peak live gather/conversion plus retained indices; excludes common output/fresh/passive captures and allocator reservation'
    finally:
        for module in originals: del module.forward


@contextmanager
def observe_legacy(book):
    """Line observer of ORIGINAL executor, breakdown only. No algorithm replica.

    .to(output) and slice assignment share a line: report combined, never invent
    an exclusive transfer time. Tracing is expensive and is never used in minimal.
    No CUDA synchronize inside forward. Prior tracing is rejected (not clobbered).
    """
    if not book:
        yield; return
    import semcache.edgelora.mixed_projection as module
    if sys.gettrace() is not None: raise RuntimeError('Legacy observer requires no existing Python tracer')
    source,base=inspect.getsourcelines(ORIGINAL_MIXED)
    marks={}; native_region=False
    for offset,line in enumerate(source):
        stage=None
        if 'fresh = ((native' in line: native_region=True
        if 'dtype = fresh.dtype' in line: native_region=False
        if native_region: stage='fresh_native_qkv'
        elif 'index = fresh_positions.to' in line or 'fresh_inputs = hidden_states.index_select' in line: stage='fresh_index_preparation'
        elif 'output = torch.empty' in line: stage='output_allocation'
        elif 'output.index_copy_' in line: stage='fresh_assembly'
        elif 'source = hit.entry.tensors' in line: stage='source_reference_resolution'
        elif 'output[:, w.start:w.end]' in line: stage='window_conversion_assignment'
        if stage: marks[base+offset]=stage
    active={}
    def trace(frame,event,arg):
        if frame.f_code.co_filename!=module.__file__ or frame.f_code.co_name!='forward': return None
        key=id(frame)
        if event=='call':
            whole=book.span('full_mixed_projection'); whole.__enter__(); active[key]=[whole,None]
        elif key in active and event in ('line','return','exception'):
            record=active[key]
            if record[1]: record[1].__exit__(None,None,None); record[1]=None
            if event=='line' and frame.f_lineno in marks:
                record[1]=book.span(marks[frame.f_lineno]); record[1].__enter__()
            if event=='return': record[0].__exit__(None,None,None); del active[key]
        return trace
    sys.settrace(trace)
    try: yield
    finally:
        sys.settrace(None)
        for whole,child in active.values():
            if child: child.__exit__(None,None,None)
            whole.__exit__(None,None,None)


def compare_projections(reference,candidate,mask):
    """Untimed exact tensor comparison; retain scalar evidence only."""
    import torch
    if set(reference)!=set(candidate): return dict(passed=False,reason='layer mismatch',layers=[])
    rows=[]
    for l in reference:
        for role in 'qkv':
            a,b=reference[l][role],candidate[l][role]
            compatible=a.shape==b.shape and a.dtype==b.dtype and a.device==b.device
            reused=torch.tensor(mask,device=a.device,dtype=torch.bool)
            rows.append(dict(layer=l,role=role,assembled_equal=compatible and torch.equal(a,b),
                reused_equal=compatible and torch.equal(a[:,reused],b[:,reused]),
                fresh_equal=compatible and torch.equal(a[:,~reused],b[:,~reused])))
    return dict(passed=all(r['assembled_equal'] and r['reused_equal'] and r['fresh_equal'] for r in rows),layers=rows)
