"""Metadata only. Discovery is deliberately weaker than physical eligibility."""
from copy import deepcopy
import time

from semcache.semantic.subsequence import SubsequenceExtractor
from semcache.semantic.matcher import ExactTokenMatcher
from semcache.semantic.intent_clusterer import IntentClusterer
from .communication import digest, require


class SemanticIndex:
    def __init__(self, encoder, *, clusters=1, update_mode='immediate_eq9', update_interval=100,
                 warmup_texts=(), limit=64, trace=lambda e:None):
        require(1<=limit<=64, 'bounded candidate limit required')
        self.encoder=encoder;self.clusterer=IntentClusterer(clusters,update_interval,update_mode=update_mode)
        self.extractor=SubsequenceExtractor(3);self.matcher=ExactTokenMatcher()
        self.entries={};self.postings={};self.limit=limit;self.trace=trace
        if warmup_texts:self.clusterer.initialize(encoder.encode(list(warmup_texts)))

    def state(self):
        c=self.clusterer
        return dict(centroids=deepcopy(c.centroids), counts=list(c.counts),pending=deepcopy(c.pending),queries=c.queries,
            num_clusters=c.num_clusters,update_mode=c.update_mode,update_interval=c.update_interval)

    def assign(self,text):
        start=time.monotonic_ns();vector=self.encoder.encode([text])[0];encoded=time.monotonic_ns()
        if not self.clusterer.centroids:
            require(self.clusterer.num_clusters==1,'multiple clusters require explicit semantic warmup')
            self.clusterer.initialize([vector])
        before=self.state();assignment=self.clusterer.observe_with_diagnostics(vector);after=self.state()
        return dict(**assignment,embedding=vector,encoder=self.encoder.metadata,cluster_before=before,cluster_after=after,
            semantic_encoding_ms=(encoded-start)/1e6,cluster_assignment_ms=(time.monotonic_ns()-encoded)/1e6)

    def add(self,key,metadata):
        require(key not in self.entries, 'duplicate candidate metadata')
        require(metadata['entry_id']==key and metadata['completed'] is True, 'uncompleted index source')
        self.entries[key]=deepcopy(metadata)
        for w in self.extractor.extract(metadata['token_ids']):
            self.postings.setdefault(self.matcher.key(metadata['cluster_id'],w),set()).add(key)
        self.trace(dict(kind='INDEX_INSERT',metadata=deepcopy(metadata)))

    def invalidate(self,key):
        if key not in self.entries:return
        metadata=self.entries.pop(key)
        for k in {self.matcher.key(metadata['cluster_id'],w) for w in self.extractor.extract(metadata['token_ids'])}:
            self.postings[k].discard(key)
            if not self.postings[k]:del self.postings[k]
        self.trace(dict(kind='INDEX_INVALIDATE',entry_id=key,source_request_id=metadata['source_request_id']))

    def discover(self,cluster,ids):
        matches={}
        for w in self.extractor.extract(ids):
            for key in self.postings.get(self.matcher.key(cluster,w),()):
                matches.setdefault(key,[]).append(dict(start=w.start,end=w.end,tokens=list(w.token_ids)))
        keys=sorted(matches,key=lambda k:(-len(self.entries[k]['token_ids']),k))[:self.limit]
        return [dict(metadata=deepcopy(self.entries[k]),matching_windows=matches[k]) for k in keys]


def eligibility(metadata,request,resident):
    if metadata['entry_id'] not in resident:return 'STALE_EVICTED_ENTRY'
    if metadata['physical_identity']!=request['physical_identity']:return 'INCOMPATIBLE_PHYSICAL_IDENTITY'
    if not metadata['completed']:return 'INCOMPLETE_SOURCE'
    ids=metadata['token_ids'];p=len(ids)
    if p>=len(request['token_ids']):return 'TARGET_NOT_LONGER_THAN_SOURCE'
    if request['token_ids'][:p]!=ids:return 'EXACT_PREFIX_TOKEN_MISMATCH'
    if metadata['positions']!=request['positions'][:p] or metadata['attention_mask']!=request['attention_mask'][:p]:return 'POSITION_OR_MASK_MISMATCH'
    return None
