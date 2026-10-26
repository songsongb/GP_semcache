"""Opt-in occurrence audit/catalog. Production key, policy and executor unchanged."""
from collections import Counter
from contextlib import contextmanager
from time import perf_counter_ns
from unittest.mock import patch

from semcache.semantic.subsequence import SubsequenceExtractor
from semcache.semantic.hit_selection import CacheHit, select_nonoverlapping
from semcache.metrics.r1 import digest
from semcache.metrics.r1b2 import filter_coverage

PRIORITY = ('ADAPTER_OR_MODEL_MISMATCH', 'PREFIX_CONTEXT_MISMATCH',
            'NO_MATCHING_SOURCE_OCCURRENCE', 'CAPACITY_OR_EVICTION',
            'ADMISSION_REJECTED', 'DUPLICATE_KEY_OCCURRENCE_LOST',
            'POSITION_MISMATCH', 'LEGACY_FULL_PROMPT_MISMATCH')


def key(value):
    return value[0], tuple(value[1])


def source_evidence(engine, result, namespace, weight_identity):
    s=result['summary']
    return dict(source_ids=s['token_ids'],cluster=s['cluster_id'],query_id=s['query_id'],
        adapter=s['adapter_name'],namespace=namespace,weight_identity=weight_identity,
        capacity_bytes=engine.cache.capacity_bytes,
        admission=s['admission_audit'],deduplication=s['deduplication_audit'],
        evictions=[e for e in result['events'] if e['event_type']=='EVICT'],
        entries=[dict(key=e.key,positions=e.positions,metadata=e.qkv_metadata,
            qkv_resident=bool(e.tensors),layers=sorted(e.tensors or {})) for e in engine.cache.entries.values()])


def occurrence_loss_audit(evidence, target_ids, adapter, namespace, weight_identity, target_cluster=None):
    """Pure provenance audit, no tensor synthesis and no cache mutation.

    Causes are independent flags. Primary cause uses PRIORITY, not an exclusive
    attribution. Residency refers to the captured pre-target source cache.
    """
    source=tuple(evidence['source_ids']); target=tuple(target_ids)
    cluster=evidence['cluster'] if target_cluster is None else target_cluster
    windows=SubsequenceExtractor(3).extract(source)
    by_key={}
    for w in windows: by_key.setdefault((evidence['cluster'],w.token_ids),[]).append(w)
    entries={key(e['key']):e for e in evidence['entries']}
    admissions={key(a['cache_key']):a for a in evidence['admission']}
    dedup={(key(d['cache_key']),d['source_start']) for d in evidence['deduplication']}
    evicted={key(e['cache_key']) for e in evidence['evictions']}
    identity=(adapter==evidence['adapter'] and namespace==evidence['namespace'] and weight_identity==evidence['weight_identity'])
    rows=[]; eligible=[]
    for w in SubsequenceExtractor(3).extract(target):
        k=(cluster,w.token_ids); occurrences=by_key.get(k,[]); resident=entries.get(k); decision=admissions.get(k)
        aligned=any(x.start==w.start and x.end==w.end for x in occurrences)
        prefix=aligned and source[:w.end]==target[:w.end]
        resident_position=bool(resident and tuple(resident['positions'])==(w.start,w.end))
        resident_qkv=bool(resident_position and resident['qkv_resident'])
        causes=[]
        if not identity: causes.append('ADAPTER_OR_MODEL_MISMATCH')
        if aligned and not prefix: causes.append('PREFIX_CONTEXT_MISMATCH')
        if not aligned: causes.append('NO_MATCHING_SOURCE_OCCURRENCE')
        if k in evicted or decision and decision.get('rejection_reason')=='entry_exceeds_capacity': causes.append('CAPACITY_OR_EVICTION')
        if decision and not decision['admission_decision'] and decision.get('rejection_reason')!='entry_exceeds_capacity': causes.append('ADMISSION_REJECTED')
        if (k,w.start) in dedup and not resident_position: causes.append('DUPLICATE_KEY_OCCURRENCE_LOST')
        if resident and not resident_position: causes.append('POSITION_MISMATCH')
        if source!=target: causes.append('LEGACY_FULL_PROMPT_MISMATCH')
        reasons=[reason for reason in PRIORITY if reason in causes]
        legacy=identity and resident_qkv and prefix and source==target
        if legacy: eligible.append(w)
        rows.append(dict(start=w.start,end=w.end,token_ids=list(w.token_ids),
            source_positions=[x.start for x in occurrences],aligned_source_occurrence_exists=aligned,
            prefix_token_ids_equal=prefix,resident_aligned_qkv=resident_qkv,
            retained_position=resident['positions'] if resident else None,
            legacy_eligible=legacy,causes=reasons,primary_cause=reasons[0] if reasons else None))
    used=set(); selected=[]
    for w in eligible:
        if not used.intersection(range(w.start,w.end)):
            selected.append([w.start,w.end]); used.update(range(w.start,w.end))
    selected_starts={s for s,e in selected}
    for row in rows:
        row['legacy_max_selected']=row['start'] in selected_starts
        row['selection_reason']='SELECTED' if row['legacy_max_selected'] else 'NONOVERLAP_CONFLICT' if row['legacy_eligible'] else 'INELIGIBLE'
    return dict(evidence_kind='ACTUAL_SOURCE_PROVENANCE' if evidence.get('actual_execution') else 'PROVIDED_PROVENANCE_UNVERIFIED_OR_FIXTURE',
        historical_54_token_cause_verified=False,priority=PRIORITY,total_extracted_windows=len(windows),
        unique_token_window_keys=len(by_key),duplicate_occurrence_count=len(windows)-len(by_key),
        admitted_keys=sum(a['admission_decision'] for a in admissions.values()),
        rejected_keys=sum(not a['admission_decision'] for a in admissions.values()),
        deduplicated_occurrences=len(evidence['deduplication']),
        retained_source_positions=[dict(key=e['key'],positions=e['positions']) for e in evidence['entries']],
        omitted_by_deduplication=[dict(key=d['cache_key'],start=d['source_start'],end=d['source_end']) for d in evidence['deduplication']],
        capacity_or_eviction_keys=[list(k) for k in sorted(evicted | {k for k,a in admissions.items() if a.get('rejection_reason')=='entry_exceeds_capacity'})],
        target_windows=rows,predicted_legacy_max_spans=selected,predicted_legacy_max_reused_tokens=len(used),
        cause_counts=dict(Counter(c for r in rows for c in r['causes'])))


def physical_bytes(tensors):
    """Count distinct backing storages, not the sum of overlapping view sizes."""
    storages={}
    for t in tensors:
        s=t.untyped_storage(); storages[(str(t.device),s.data_ptr())]=s.nbytes()
    return sum(storages.values())


def occurrence_reasons(entry, window, target_ids, adapter, namespace, weight_identity):
    """Tensor-free necessary safety predicates; not a numerical parity proof."""
    m=entry.qkv_metadata
    reasons=[]
    if m.get('source_adapter')!=adapter or m.get('model_namespace')!=namespace or m.get('weight_identity')!=weight_identity:
        reasons.append('ADAPTER_OR_MODEL_MISMATCH')
    if tuple(entry.positions)!=(window.start,window.end): reasons.append('POSITION_MISMATCH')
    if tuple(entry.token_ids)!=tuple(window.token_ids): reasons.append('NO_MATCHING_SOURCE_OCCURRENCE')
    prefix=tuple(m.get('source_query_token_ids',())[:window.end])==tuple(target_ids[:window.end])
    if not prefix: reasons.append('PREFIX_CONTEXT_MISMATCH')
    return reasons,prefix


class OccurrenceCatalog:
    classification='ORACLE_DIAGNOSTIC'

    def __init__(self, namespace, weight_identity, storage, capacity_bytes):
        self.namespace=namespace; self.weight_identity=weight_identity
        self.storage=storage; self.capacity_bytes=capacity_bytes
        self.sources={}; self.entries={}; self.index={}
        self.copy_ms=0.; self.transfer_bytes=0

    @property
    def tensors(self):
        return [t for s in self.sources.values() for qkv in s['qkv'].values() for t in qkv]

    @property
    def resident_bytes(self): return physical_bytes(self.tensors)

    def add_source(self, query_id, ids, cluster, adapter, projections, layout, *, prefill_complete, impacts=None, source_clock=0):
        """Own full source QKV, expose position-specific views; never target QKV."""
        import torch
        from semcache.cache.cache_entry import CacheEntry
        from semcache.metrics.r1_runtime import synchronize
        if not prefill_complete or query_id in self.sources: raise ValueError('Source must be completed and query ID unique')
        if set(projections)!=set(layout): raise ValueError('Missing source QKV layer')
        required=0
        for layer,qkv in projections.items():
            if len(qkv)!=3: raise ValueError('Three roles required')
            for role,t in enumerate(qkv):
                width,dtype=layout[layer][role]
                if (t.shape!=(1,len(ids),width) or t.dtype!=dtype or t.is_quantized
                    or not t.is_floating_point() or not bool(torch.isfinite(t).all())):
                    raise ValueError('Source QKV shape/dtype/nonfinite mismatch')
                required+=t.numel()*t.element_size()
        if self.resident_bytes+required>self.capacity_bytes: raise MemoryError('CATALOG_CAPACITY_EXCEEDED')
        source_device=next(iter(projections.values()))[0].device
        synchronize(source_device); start=perf_counter_ns()
        owned={l:tuple(t.detach().to(self.storage).clone().contiguous() for t in qkv) for l,qkv in projections.items()}
        synchronize(source_device); synchronize(self.storage)
        self.copy_ms+=(perf_counter_ns()-start)/1e6
        self.transfer_bytes+=sum(t.numel()*t.element_size() for q in projections.values() for t in q if str(t.device)!=str(torch.device(self.storage)))
        self.sources[query_id]=dict(ids=tuple(ids),qkv=owned,layout=layout,adapter=adapter,cluster=cluster)
        for w in SubsequenceExtractor(3).extract(ids):
            identity=(cluster,w.token_ids,query_id,w.start,w.end,adapter,self.namespace)
            tensors={l:tuple(t[:,w.start:w.end] for t in q) for l,q in owned.items()}
            logical=sum(t.numel()*t.element_size() for q in tensors.values() for t in q)
            entry=CacheEntry(cluster,w.token_ids,(w.start,w.end),logical,tensors=tensors,
                impact=impacts[w.start] if impacts is not None else None,created_at=source_clock,updated_at=source_clock,last_access=source_clock,
                qkv_metadata=dict(component_scope='total_qkv',source_query_id=query_id,source_user=adapter,
                    source_adapter=adapter,source_query_token_ids=tuple(ids),model_namespace=self.namespace,
                    weight_identity=self.weight_identity,occurrence_identity=identity))
            self.entries[identity]=entry
            self.index.setdefault((cluster,w.token_ids,w.start,w.end),[]).append(identity)
        if self.resident_bytes>self.capacity_bytes: raise MemoryError('CATALOG_PHYSICAL_CAPACITY_EXCEEDED')

    def candidates(self, ids, cluster, adapter, namespace, weight_identity, layout):
        rows=[]; candidates=[]
        identity_ok=namespace==self.namespace and weight_identity==self.weight_identity
        for w in SubsequenceExtractor(3).extract(ids):
            matches=self.index.get((cluster,w.token_ids,w.start,w.end),[])
            accepted=[]; causes=[]; prefixes=[]
            for occurrence in matches:
                entry=self.entries[occurrence]; m=entry.qkv_metadata
                reasons,prefix=occurrence_reasons(entry,w,ids,adapter,namespace,weight_identity); prefixes.append(prefix)
                if not identity_ok: reasons.append('ADAPTER_OR_MODEL_MISMATCH')
                if reasons: causes.extend(reasons); continue
                if tuple(entry.positions)!=(w.start,w.end) or tuple(entry.token_ids)!=w.token_ids:
                    raise ValueError('Corrupt occurrence position/token identity')
                if tuple(m['source_query_token_ids'])!=self.sources[m['source_query_id']]['ids']:
                    raise ValueError('Corrupt source prefix provenance')
                if set(entry.tensors)!=set(layout): raise ValueError('Catalog layer mismatch')
                for l,qkv in entry.tensors.items():
                    if len(qkv)!=3: raise ValueError('Catalog role mismatch')
                    for role,t in enumerate(qkv):
                        width,dtype=layout[l][role]
                        backing=self.sources[m['source_query_id']]['qkv'][l][role][:,w.start:w.end]
                        if t.data_ptr()!=backing.data_ptr() or t.stride()!=backing.stride():
                            raise ValueError('Occurrence tensor does not reference its source position')
                        if t.shape!=(1,3,width) or t.dtype!=dtype or str(t.device)!=str(__import__('torch').device(self.storage)):
                            raise ValueError('Catalog tensor shape/dtype/residency mismatch')
                accepted.append(CacheHit(w,entry))
            if not matches: causes.append('NO_MATCHING_SOURCE_OCCURRENCE')
            candidates.extend(accepted)
            rows.append(dict(start=w.start,end=w.end,eligible_occurrences=len(accepted),
                prefix_token_ids_equal=any(prefixes) if prefixes else None,reasons=sorted(set(causes)),
                occurrence_ids=[digest(h.entry.qkv_metadata['occurrence_identity']) for h in accepted]))
        return candidates,rows

    def select(self, ids, cluster, adapter, namespace, weight_identity, layout, coverage):
        candidates,rows=self.candidates(ids,cluster,adapter,namespace,weight_identity,layout)
        nonoverlap,_=select_nonoverlapping(candidates,len(ids))
        selected,mask=filter_coverage(nonoverlap,len(ids),coverage)
        selected_starts={h.window.start for h in selected}; feasible={h.window.start for h in nonoverlap}
        for row in rows:
            row['selection_reason']=('SELECTED' if row['start'] in selected_starts else
                'COVERAGE_FILTER' if row['start'] in feasible else 'NONOVERLAP_CONFLICT' if row['eligible_occurrences'] else 'INELIGIBLE')
        return selected,mask,rows


@contextmanager
def capture_completed_source():
    """Retain existing detailed source executor outputs; no new layer hooks/copies."""
    import semcache.metrics.r1_runtime as runtime
    original=runtime.detailed_mixed; captured={}
    @contextmanager
    def capture(*args,**kwargs):
        with original(*args,**kwargs) as audit:
            yield audit
        captured.update({l:tuple(p[n] for n in 'qkv') for l,p in audit.projections.items()})
    with patch.object(runtime,'detailed_mixed',capture): yield captured


@contextmanager
def catalog_override(catalog, engine, namespace, weight_identity, layout, coverage):
    """Diagnostic selection only. Existing lookup/policy still execute.

    CHU touches selected sidecar entries, not legacy resident entries. Hence this
    path is an admission-bypassing oracle, NOT policy-equivalent production.
    """
    import semcache.semcache_engine as module
    evidence={}
    def select(hits,n,ids,user):
        cluster=next(e['cluster_id'] for e in reversed(engine.events) if e['event_type']=='CLUSTER_ASSIGN')
        start=perf_counter_ns()
        chosen,mask,rows=catalog.select(ids,cluster,user,namespace,weight_identity,layout,coverage)
        evidence.update(selection_host_ms=(perf_counter_ns()-start)/1e6,windows=rows,hits=chosen)
        return chosen,mask
    with patch.object(module,'select_position_aligned_diagnostic',select): yield evidence
