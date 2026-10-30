"""Multi-entry full-prompt residency over frozen, sealed R3-B tensor owners."""
from collections import OrderedDict
from copy import deepcopy
import time

from .communication import digest,require
from .r3b_cache import ExactRawCache,spec
from .r4a_cache import PromptCache
from .r4a_session import positions


class BoundPrompt:
    """A context view, never an owner of evicted tensors."""
    def __init__(self,owner,context):
        self.owner=owner;self.context=deepcopy(context);self.context_digest=digest(context)
    @property
    def bytes(self):return self.owner.bytes
    @property
    def events(self):return self.owner.events
    def reset(self,reason):self.owner.remove(self.context_digest,'INVALIDATE',reason)
    def lookup(self,context):return self.owner.lookup(context)
    def insert(self,context,blocks,request_id,*,completed):
        require(context==self.context,'bound full-prompt insertion identity mismatch')
        return self.owner.insert(context,blocks,request_id,completed=completed)
    def reused(self,entry):self.owner.reused(self.context,entry)


class MultiPromptCache:
    """Single-threaded ES cache. Global budget counts owned compact QKV storage.

    R3-B owns/clones/seals and validates each entry. The manager evicts BEFORE
    cloning, and uses validated-lookup LRU across namespaces. No KV is retained.
    """
    def __init__(self,pm,prepared_sha,gpu_uuid,device,capacity_bytes,*,tokenizer,sources,fixture=False,synchronize=lambda:None,trace=None):
        require(type(capacity_bytes) is int and 0<capacity_bytes<=2**30,'cache byte budget must be in (0, 1 GiB]')
        require(tokenizer and sources,'actual tokenizer and source provenance required')
        self._pm=deepcopy(pm);self.prepared_sha=prepared_sha;self.gpu_uuid=gpu_uuid;self.device=str(device)
        self.tokenizer=deepcopy(tokenizer);self.sources=deepcopy(sources);self.capacity_bytes=capacity_bytes;self.fixture=fixture
        self.synchronize=synchronize;self.trace=trace;self.entries=OrderedDict();self.events=[];self.peak_bytes=0
        self.active=self;self.request_id=None
    @property
    def bytes(self):return sum(c.bytes for c in self.entries.values())
    def snapshot(self):
        return dict(occupancy_bytes=self.bytes,entry_count=len(self.entries),capacity_bytes=self.capacity_bytes,
            peak_occupancy_bytes=self.peak_bytes,lru_to_mru=list(self.entries),resident_entries=[dict(context_sha256=k,
                bytes=c.bytes,source_request_id=next(iter(c.pool.entries.values())).qkv_metadata['source_request_id']) for k,c in self.entries.items()])
    def event(self,kind,start=None,**fields):
        self.synchronize();self.peak_bytes=max(self.peak_bytes,self.bytes)
        require(self.bytes<=self.capacity_bytes,'global physical byte capacity exceeded')
        row=dict(sequence=len(self.events),kind=kind,request_id=self.request_id,monotonic_ns=time.monotonic_ns(),
            duration_ms=0. if start is None else (time.monotonic_ns()-start)/1e6,**self.snapshot(),**fields)
        self.events.append(row)
        if self.trace is not None:self.trace(row)
        return row
    def context_for(self,prompt):
        ids=list(prompt['token_ids']);mask=list(prompt['attention_mask'])
        require(len(ids)==len(mask) and 1<=len(ids)<=128 and all(type(t) is int and t>=0 for t in ids),'invalid actual full-prompt IDs/mask')
        pos=positions(mask)
        require(prompt.get('positions',pos)==pos,'supplied positions differ from actual full mask')
        require(prompt.get('prompt_sha256',digest(dict(tokens=ids,mask=mask)))==digest(dict(tokens=ids,mask=mask)),
            'prompt digest differs from actual IDs/mask')
        # Reuse the frozen R4-A/R3-B identity construction without storing data
        # in its single-active-prompt router.
        ctx=PromptCache(self._pm,self.prepared_sha,self.gpu_uuid,self.device,fixture=self.fixture).bind(prompt).context
        ctx=deepcopy(ctx);ctx.update(schema='R4B1_EXACT_FULL_PROMPT_V1',tokenizer=deepcopy(self.tokenizer),
            execution_sources=deepcopy(self.sources),storage_device=self.device)
        return ctx
    def bind(self,prompt):
        self.active=BoundPrompt(self,self.context_for(prompt))
        return self.active
    def remove(self,key,kind,reason):
        self.synchronize();start=time.monotonic_ns();cache=self.entries.pop(key,None)
        if cache is None:return
        size=cache.bytes;source=next(iter(cache.pool.entries.values())).qkv_metadata['source_request_id']
        for entry in cache.pool.entries.values():entry.tensors=None
        cache.reset(reason)  # Clears tensor owners and seals even if a view survives.
        self.event(kind,start,context_sha256=key,released_bytes=size,source_request_id=source,reason=reason)
    def lookup(self,context):
        self.synchronize();start=time.monotonic_ns();key=digest(context);cache=self.entries.get(key)
        if cache is None:
            self.event('MISS',start,context_sha256=key,reason='exact_full_prompt_identity_absent');return None
        entry=cache.lookup(context)
        if entry is None:
            self.remove(key,'REJECT','sealed entry failed full identity/content validation')
            self.event('MISS',start,context_sha256=key,reason='integrity_rejected');return None
        self.entries.move_to_end(key)
        self.event('HIT',start,context_sha256=key,source_request_id=entry.qkv_metadata['source_request_id'],validation_read_bytes=entry.size_bytes)
        return entry
    def insert(self,context,blocks,request_id,*,completed):
        require(completed is True,'only completed full generations may be admitted')
        require(context==self.context_for(dict(token_ids=context['token_ids'],attention_mask=context['attention_mask'])),
            'insertion model/tokenizer/adapter/source/position identity mismatch')
        require(request_id==self.request_id,'insertion source request mismatch')
        self.synchronize();start=time.monotonic_ns();key=digest(context);identity=context['identity'];size=0
        require(set(blocks)==set(range(identity['layers'])),'partial full-prompt layer set')
        for block in blocks.values():
            require(len(block)==3,'partial QKV block')
            for tensor in block:
                value=spec(tensor)
                require(value['shape']==[1,identity['length'],identity['hidden_size']] and
                    value['dtype']==identity['hidden_dtype'] and value['device']==self.device,'insertion tensor shape/dtype/device mismatch')
                size+=tensor.numel()*tensor.element_size()
        require(key not in self.entries,'duplicate admission requires explicit targeted invalidation')
        if size>self.capacity_bytes:
            self.event('ADMISSION',start,context_sha256=key,allowed=False,candidate_bytes=size,reason='oversize');return False
        while self.bytes+size>self.capacity_bytes:
            self.remove(next(iter(self.entries)),'EVICT','capacity_lru')
        cache=ExactRawCache(context,self.capacity_bytes,self.device,fixture=self.fixture)
        allowed=cache.insert(context,blocks,request_id,completed=completed)
        if allowed:self.entries[key]=cache
        self.event('ADMISSION',start,context_sha256=key,allowed=allowed,candidate_bytes=size,
            primitive_events=deepcopy(cache.events),reason='frozen_R3B_admission_policy')
        if allowed:self.event('INSERT',context_sha256=key,source_request_id=request_id,inserted_bytes=size)
        return allowed
    def reused(self,context,entry):
        key=digest(context);require(key in self.entries,'cannot claim reuse after eviction')
        self.entries[key].validate(entry,context);self.entries[key].reused(entry)
        self.event('PHYSICAL_REUSE',context_sha256=key,source_request_id=entry.qkv_metadata['source_request_id'],qkv_outputs=96)
