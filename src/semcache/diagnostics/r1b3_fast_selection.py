"""Opt-in validated occurrence index; no executor, policy or legacy edits.

Index construction verifies structural/tensor invariants once. Each request
checks snapshot structure and backing version counters before using those proofs.
Policy counters/impacts may change; source identity or tensor mutation invalidates
this index and fails closed. No persistent target-prefix cache or hash safety rule.
"""
from contextlib import contextmanager, ExitStack, nullcontext
from collections import defaultdict
import math
import sys
from time import perf_counter_ns
from unittest.mock import patch

from semcache.metrics.r1 import digest
from semcache.metrics.r1b2 import coverage_value
from semcache.semantic.subsequence import SubsequenceExtractor
from semcache.semantic.hit_selection import CacheHit

STAGES=('source_only_index_construction','target_window_extraction','candidate_lookup',
    'position_filtering','prefix_context_validation','model_adapter_identity_validation',
    'nonoverlap_selection','tensor_backing_validation','coverage_filtering',
    'selection_audit_provenance_construction','snapshot_structure_validation',
    'identity_position_prefix_combined','audit_digest_only')


class HostBook:
    def __init__(self): self.values=defaultdict(float); self.calls=defaultdict(int)
    @contextmanager
    def span(self,name):
        start=perf_counter_ns()
        try: yield
        finally:
            self.values[name]+=(perf_counter_ns()-start)/1e6; self.calls[name]+=1
    def export(self):
        return {n:dict(host_ms=self.values.get(n),calls=self.calls.get(n),
                       status='MEASURED' if n in self.values else 'UNAVAILABLE') for n in STAGES}


def span(book,name): return book.span(name) if book else nullcontext()


def prefix_limit(source,target):
    """Exact common prefix length, equivalent to end-index mismatch-prefix sums."""
    for i,(a,b) in enumerate(zip(source,target)):
        if a!=b: return i
    return min(len(source),len(target))


def tensor_stamp(t):
    # Inference tensors without version counters cannot support this guard.
    # Catalog.add_source clones outside inference_mode; reject unsupported inputs.
    try: version=t._version
    except RuntimeError as exc: raise ValueError('Versioned owned source tensors required') from exc
    return (t.data_ptr(),tuple(t.shape),tuple(t.stride()),t.dtype,str(t.device),
            version,t.untyped_storage().nbytes())


def owned_python_bytes(value, excluded):
    """Reachable Python metadata estimate, excluding pre-existing objects/tensors."""
    seen=set(excluded)
    def visit(v):
        if id(v) in seen: return 0
        seen.add(id(v)); total=sys.getsizeof(v)
        if isinstance(v,dict): total+=sum(visit(k)+visit(x) for k,x in v.items())
        elif isinstance(v,(tuple,list,set)): total+=sum(visit(x) for x in v)
        elif isinstance(v,CacheHit): total+=visit(v.__dict__)
        return total
    return visit(value)


class FastOccurrenceIndex:
    classification='ORACLE_DIAGNOSTIC'
    def __init__(self,catalog,layout):
        start=perf_counter_ns()
        self.catalog=catalog; self.layout={l:tuple(v) for l,v in layout.items()}
        self.catalog_identity=(catalog.namespace,catalog.weight_identity,catalog.storage)
        self.source_guards=[]; self.entry_guards=[]; self.owners=[]; self.view_guards=[]; self.buckets={}; self.sources={}
        excluded={id(catalog),id(layout)}
        for qid,source in catalog.sources.items():
            ids=source['ids']
            if not isinstance(ids,tuple): raise ValueError('Immutable source IDs required')
            pairs=tuple(source['qkv'].items())
            self.source_guards.append((qid,source,ids,source['qkv'],pairs))
            self.sources[qid]=(ids,source['adapter'])
            excluded.update((id(qid),id(source),id(ids),id(source['qkv'])))
            for l,qkv in pairs:
                for t in qkv:
                    self.owners.append((t,tensor_stamp(t))); excluded.add(id(t))
        for identity,entry in catalog.entries.items():
            m=entry.qkv_metadata; qid=m['source_query_id']; ids=self.sources[qid][0]
            if (not isinstance(entry.token_ids,tuple) or not isinstance(entry.positions,tuple)
                or not isinstance(m['source_query_token_ids'],tuple)):
                raise ValueError('Immutable occurrence identity required')
            start_pos,end=entry.positions
            if (end-start_pos!=3 or not 0<=start_pos<end<=len(ids) or entry.token_ids!=ids[start_pos:end]
                or tuple(m['source_query_token_ids'])!=ids or m['source_adapter']!=self.sources[qid][1]
                or m['model_namespace']!=catalog.namespace or m['weight_identity']!=catalog.weight_identity
                or identity!=(entry.cluster_id,entry.token_ids,qid,start_pos,end,m['source_adapter'],catalog.namespace)
                or m['occurrence_identity']!=identity): raise ValueError('Invalid source occurrence identity')
            if set(entry.tensors)!=set(layout): raise ValueError('Catalog layer mismatch')
            for l,qkv in entry.tensors.items():
                if len(qkv)!=3: raise ValueError('Catalog role mismatch')
                for role,t in enumerate(qkv):
                    width,dtype=layout[l][role]; backing=catalog.sources[qid]['qkv'][l][role][:,start_pos:end]
                    if (t.data_ptr()!=backing.data_ptr() or t.stride()!=backing.stride() or t.shape!=(1,3,width)
                        or t.dtype!=dtype or str(t.device)!=str(catalog.storage)
                        or t._version!=catalog.sources[qid]['qkv'][l][role]._version):
                        raise ValueError('Catalog tensor backing/shape/dtype/residency mismatch')
                    self.view_guards.append((t,(t.data_ptr(),t.shape,t.stride(),t.dtype,t.device)))
                    excluded.add(id(t))
            # Immutable token/source-prefix tuples are guarded by identity, avoiding
            # their O(N) comparison on each target window.
            fields={k:m[k] for k in ('source_query_id','source_adapter','source_user','model_namespace',
                'weight_identity','source_query_token_ids','occurrence_identity','component_scope')}
            pairs=tuple(entry.tensors.items())
            self.entry_guards.append((identity,entry,entry.cluster_id,entry.token_ids,entry.positions,m,fields,entry.tensors,pairs))
            excluded.update((id(identity),id(entry),id(entry.token_ids),id(entry.positions),id(m),id(entry.tensors)))
            excluded.update(id(v) for v in fields.values())
        # Follow original bucket insertion order, including multiple query IDs.
        # Legacy sort has equal (start, -0.0, key) for a bucket: stable first wins.
        for key,identities in catalog.index.items():
            records=[]
            for identity in identities:
                e=catalog.entries[identity]
                if key!=(e.cluster_id,e.token_ids,*e.positions): raise ValueError('Invalid source index bucket')
                w=__import__('semcache.semantic.subsequence',fromlist=['Subsequence']).Subsequence(e.token_ids,*e.positions)
                records.append((e.qkv_metadata['source_query_id'],CacheHit(w,e),digest(identity)))
            self.buckets[key]=tuple(records)
        if sorted(x for values in catalog.index.values() for x in values)!=sorted(catalog.entries):
            raise ValueError('Missing/duplicate catalog index occurrence')
        self.index_guard=tuple((k,v,tuple(v)) for k,v in catalog.index.items())
        self.build_ms=(perf_counter_ns()-start)/1e6
        self.additional_index_memory_bytes=owned_python_bytes(self.__dict__,excluded)
        self.additional_qkv_bytes=0
        self.build_ms=(perf_counter_ns()-start)/1e6

    def guard(self,book=None):
        c=self.catalog
        with span(book,'snapshot_structure_validation'):
            if (c.namespace,c.weight_identity,c.storage)!=self.catalog_identity or len(c.sources)!=len(self.source_guards) or len(c.entries)!=len(self.entry_guards) or len(c.index)!=len(self.index_guard):
                raise ValueError('Source snapshot changed; rebuild index')
            for qid,s,ids,mapping,pairs in self.source_guards:
                if c.sources.get(qid) is not s or s['ids'] is not ids or s['qkv'] is not mapping or s['adapter']!=self.sources[qid][1] or len(mapping)!=len(pairs):
                    raise ValueError('Source snapshot changed')
                for l,qkv in pairs:
                    if mapping.get(l) is not qkv: raise ValueError('Source QKV mapping changed')
            for identity,e,cluster,tokens,positions,m,fields,mapping,pairs in self.entry_guards:
                if (c.entries.get(identity) is not e or e.cluster_id!=cluster or e.token_ids is not tokens
                    or e.positions is not positions or e.qkv_metadata is not m or e.tensors is not mapping or len(mapping)!=len(pairs)):
                    raise ValueError('Occurrence structure changed')
                if any(m.get(k) is not v for k,v in fields.items()): raise ValueError('Occurrence metadata changed')
                for l,qkv in pairs:
                    if mapping.get(l) is not qkv: raise ValueError('Occurrence QKV mapping changed')
            for k,bucket,contents in self.index_guard:
                if c.index.get(k) is not bucket or tuple(bucket)!=contents: raise ValueError('Occurrence bucket changed')
        with span(book,'tensor_backing_validation'):
            for t,stamp in self.owners:
                if tensor_stamp(t)!=stamp: raise ValueError('Source tensor mutated; rebuild index')
            # Detect even .data rebinding/view-metadata changes without relying
            # solely on version counters. No fresh backing slices/device strings.
            for t,stamp in self.view_guards:
                if (t.data_ptr(),t.shape,t.stride(),t.dtype,t.device)!=stamp:
                    raise ValueError('Occurrence view changed; rebuild index')

    def select(self,ids,cluster,adapter,namespace,weight_identity,layout,coverage,book=None):
        if layout!=self.layout: raise ValueError('Target projection layout differs from source index')
        self.guard(book)
        with span(book,'target_window_extraction'):
            target=tuple(ids); windows=SubsequenceExtractor(3).extract(target)
        with span(book,'prefix_context_validation'):
            prefixes={q:prefix_limit(s[0],target) for q,s in self.sources.items()}
        with span(book,'model_adapter_identity_validation'):
            identity_ok={q:adapter==s[1] and namespace==self.catalog.namespace and weight_identity==self.catalog.weight_identity for q,s in self.sources.items()}
        rows=[]; feasible=[]; previous_end=0
        for w in windows:
            with span(book,'candidate_lookup'): matches=self.buckets.get((cluster,w.token_ids,w.start,w.end),())
            accepted=[]; causes=[]; checks=[]
            for qid,hit,occurrence_hash in matches:
                # The bucket key proves token/position equality. Guarded source
                # validation proves backing and metadata; prefix depends on target.
                prefix=w.end<=prefixes[qid]; checks.append(prefix)
                if not identity_ok[qid]: causes.append('ADAPTER_OR_MODEL_MISMATCH')
                if not prefix: causes.append('PREFIX_CONTEXT_MISMATCH')
                if identity_ok[qid] and prefix: accepted.append((hit,occurrence_hash))
            if not matches: causes.append('NO_MATCHING_SOURCE_OCCURRENCE')
            with span(book,'nonoverlap_selection'):
                if accepted and w.start>=previous_end:
                    feasible.append(accepted[0][0]); previous_end=w.end
            with span(book,'selection_audit_provenance_construction'):
                rows.append(dict(start=w.start,end=w.end,eligible_occurrences=len(accepted),
                    prefix_token_ids_equal=any(checks) if checks else None,reasons=sorted(set(causes)),
                    occurrence_ids=[h for _,h in accepted]))
        with span(book,'coverage_filtering'):
            fraction=coverage_value(coverage)
            count=len(feasible) if fraction=='max' else min(len(feasible),math.floor(len(target)*fraction/3))
            chosen=feasible[:count]; mask=[False]*len(target)
            for h in chosen: mask[h.window.start:h.window.end]=[True]*3
        with span(book,'selection_audit_provenance_construction'):
            selected_starts={h.window.start for h in chosen}; possible={h.window.start for h in feasible}
            for row in rows:
                row['selection_reason']=('SELECTED' if row['start'] in selected_starts else 'COVERAGE_FILTER'
                    if row['start'] in possible else 'NONOVERLAP_CONFLICT' if row['eligible_occurrences'] else 'INELIGIBLE')
        return chosen,mask,rows


@contextmanager
def profile_legacy(catalog,book):
    """Scoped observers around the UNCHANGED legacy selector.

    Combined identity/position/prefix cost and digest-only audit cost are labeled
    honestly; unsupported exclusive components stay null. Tensor mapping iterator
    measures the original per-layer backing loop, without replacing its body.
    """
    import semcache.diagnostics.r1b3_occurrences as module
    def wrap(fn,name):
        def call(*a,**kw):
            with book.span(name): return fn(*a,**kw)
        return call
    class Lookup:
        def __init__(self,original): self.original=original
        def get(self,*a):
            with book.span('candidate_lookup'): return self.original.get(*a)
    class TensorMapping:
        def __init__(self,original): self.original=original
        def __iter__(self): return iter(self.original)
        def __len__(self): return len(self.original)
        def __getitem__(self,k): return self.original[k]
        def items(self):
            for pair in self.original.items():
                with book.span('tensor_backing_validation'): yield pair
    with ExitStack() as stack:
        stack.enter_context(patch.object(catalog,'index',Lookup(catalog.index)))
        for entry in catalog.entries.values(): stack.enter_context(patch.object(entry,'tensors',TensorMapping(entry.tensors)))
        for name,stage in [('occurrence_reasons','identity_position_prefix_combined'),('digest','audit_digest_only'),
                           ('select_nonoverlapping','nonoverlap_selection'),('filter_coverage','coverage_filtering')]:
            stack.enter_context(patch.object(module,name,wrap(getattr(module,name),stage)))
        stack.enter_context(patch.object(SubsequenceExtractor,'extract',wrap(SubsequenceExtractor.extract,'target_window_extraction')))
        yield


def selection_evidence(result):
    hits,mask,rows=result
    return dict(occurrences=[dict(identity=h.entry.qkv_metadata['occurrence_identity'],
        source=list(h.entry.positions),destination=[h.window.start,h.window.end],tokens=list(h.window.token_ids),
        tensor_ids=[id(t) for _,qkv in sorted(h.entry.tensors.items()) for t in qkv]) for h in hits],
        mask=mask,windows=rows)


def compare_selection(reference,candidate):
    a,b=selection_evidence(reference),selection_evidence(candidate)
    def first_difference(x,y):
        for i in range(max(len(x),len(y))):
            left=x[i] if i<len(x) else None; right=y[i] if i<len(y) else None
            if left!=right: return dict(index=i,reference=left,candidate=right)
        return None
    return dict(passed=a==b,selected_sequence_equal=a['occurrences']==b['occurrences'],
        mask_equal=a['mask']==b['mask'],validation_rows_equal=a['windows']==b['windows'],
        qkv_identity_implies_same_content=True if a['occurrences']==b['occurrences'] else None,
        reference=digest(a),candidate=digest(b),
        first_difference=next((i for i,(x,y) in enumerate(zip(a['occurrences'],b['occurrences'])) if x!=y),None),
        reference_count=len(a['occurrences']),candidate_count=len(b['occurrences']),
        first_occurrence_difference=first_difference(a['occurrences'],b['occurrences']),
        first_validation_difference=first_difference(a['windows'],b['windows']))
