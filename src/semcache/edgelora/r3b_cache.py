"""Bounded exact-full-prompt total-QKV storage using real SemCache objects.

No semantic/window candidate can authorize reuse. Original R3-A stays frozen.
"""
from copy import deepcopy

from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.cache.cache_metrics import normalize
from semcache.semantic.matcher import ExactTokenMatcher
from semcache.semantic.subsequence import Subsequence
from .communication import digest, digest_bytes, require
from .r3a_artifacts import tensor_bytes


def raw(tensor):
    return tensor.fixture_bytes() if hasattr(tensor,'fixture_bytes') else tensor_bytes(tensor)


def spec(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype).removeprefix('torch.'),
        device=str(tensor.device), sha256=digest_bytes(raw(tensor)))


def request_context(pm, prepared_sha, gpu_uuid):
    identity=pm['identity'];n=identity['length']
    return dict(schema='R3B_EXACT_FULL_PROMPT_V1',identity=deepcopy(identity),prepared_manifest_sha256=prepared_sha,
        gpu_uuid=gpu_uuid,token_ids=list(identity['token_ids']),positions=list(range(n)),
        attention_mask=[1]*n,context=dict(prefix_token_ids=[],causal=True,past_key_values=None),
        endpoint=deepcopy(identity['endpoint']),adapter=identity['adapter'],
        adapter_matrix_sha256=identity['adapter_matrix_sha256'],component_scope='total_qkv',
        complete_layers=identity['layers'],projection_roles=list('qkv'),inference_training=False)


class ExactRawCache:
    """Owns compact tensor copies, checks full SHA integrity on every lookup.

    Validation transfers to CPU for hashing are deliberately INSIDE lookup and
    request wall. They are memory-validation bytes, never network bytes.
    """
    def __init__(self,context,capacity_bytes,device,*,fixture=False):
        require(type(capacity_bytes) is int and capacity_bytes>0,'positive bounded cache capacity required')
        require(str(device).startswith('cuda') or fixture and device=='cpu','production cache must be ES GPU resident')
        self.context=deepcopy(context);self.context_digest=digest(context);self.device=str(device);self.fixture=fixture
        self.pool=GlobalCache(capacity_bytes);self.matcher=ExactTokenMatcher();self.events=[];self.seals={}

    @property
    def bytes(self):return self.pool.physical_tensor_bytes

    def event(self,kind,**values):self.events.append(dict(kind=kind,**values))

    def key(self,context):
        return self.matcher.key(0,Subsequence(tuple(context['token_ids']),0,len(context['token_ids'])))

    def reset(self,reason):
        before=self.bytes;self.pool.entries.clear();self.seals.clear()
        self.event('RESET',reason=reason,released_bytes=before)

    def validate(self,entry,context):
        require(context==self.context and digest(context)==self.context_digest,'request model/adapter/tokens/positions/context mismatch')
        require(entry.key in self.seals and self.pool.entries.get(entry.key) is entry,'foreign or unsealed entry')
        seal=self.seals[entry.key];metadata=entry.qkv_metadata;identity=context['identity']
        require(metadata==seal['metadata'] and metadata['complete_request'] is True and
            metadata['context_sha256']==self.context_digest and metadata['component_scope']=='total_qkv',
            'incompatible metadata or uncompleted request provenance')
        require(tuple(entry.token_ids)==tuple(context['token_ids']) and entry.positions==(0,identity['length']),
            'entry token or position mismatch')
        require(entry.tensors is not None and set(entry.tensors)==set(range(identity['layers'])), 'partial layer set')
        expected_shape=[1,identity['length'],identity['hidden_size']]
        expected_dtype=identity['hidden_dtype']
        actual={}
        for layer in range(identity['layers']):
            block=entry.tensors[layer];require(len(block)==3,'partial QKV block')
            for role,tensor in zip('qkv',block):
                value=spec(tensor)
                require(value['shape']==expected_shape and value['dtype']==expected_dtype and value['device']==self.device,
                        'cached tensor shape/dtype/device mismatch')
                require(tensor.is_contiguous(),'noncompact cached tensor')
                actual[f'{layer}.{role}']=value
        require(actual==seal['specs'],'cached total QKV content/provenance integrity mismatch')
        require(entry.size_bytes==seal['size_bytes']==entry.physical_tensor_bytes and self.bytes<=self.pool.capacity_bytes,
                'physical cache size/capacity invariant failed')
        return actual

    def lookup(self,context):
        self.pool.advance(self.pool.now+1)
        candidate=self.pool.lookup(self.key(context),record_reuse=False)
        if candidate is None:
            self.event('MISS',reason='exact_full_prompt_key_absent');return None
        try:self.validate(candidate,context)
        except (ValueError,KeyError,TypeError,AttributeError) as exc:
            self.event('REJECT',reason=str(exc));return None
        self.event('HIT',source_request_id=candidate.qkv_metadata['source_request_id'])
        return candidate

    def insert(self,context,blocks,request_id,*,completed):
        require(completed is True and context==self.context,'only completed matching requests may be admitted')
        identity=context['identity'];require(set(blocks)==set(range(identity['layers'])),'incomplete insertion')
        size=0
        for block in blocks.values():
            require(len(block)==3,'incomplete QKV insertion')
            for t in block:
                value=spec(t)
                require(value['shape']==[1,identity['length'],identity['hidden_size']] and
                    value['dtype']==identity['hidden_dtype'] and value['device']==self.device,'wrong insertion tensor')
                size+=t.numel()*t.element_size()
        candidate=CacheEntry(0,tuple(context['token_ids']),(0,identity['length']),size)
        frequencies={candidate.key:1}
        metrics=self.pool.admission_metrics(candidate,1,frequencies)
        allowed=self.pool.admission.admit(metrics) and size<=self.pool.capacity_bytes
        self.event('ADMISSION',allowed=allowed,score=self.pool.admission.score(metrics),candidate_bytes=size,
                   impact_evidence='unavailable; GlobalCache maps None to zero for policy only')
        if not allowed:return False
        if candidate.key in self.pool.entries:
            self.event('DENY_DUPLICATE');return False
        # Evict before cloning to keep resident physical storage bounded even
        # transiently. Victim score/tie order matches GlobalCache's policy.
        while self.bytes+size>self.pool.capacity_bytes:
            population={k:self.pool._metrics(e) for k,e in self.pool.entries.items()}
            victim=min(population,key=lambda k:(-self.pool.eviction.score(normalize(population[k],population.values())),k))
            self.event('EVICT',key=list(victim[1]),bytes=self.pool.entries[victim].size_bytes)
            del self.pool.entries[victim];self.seals.pop(victim,None)
        def materialize():
            owned=CacheEntry.from_tensors(0,tuple(context['token_ids']),(0,identity['length']),blocks,self.device)
            owned.qkv_metadata=dict(component_scope='total_qkv',context_sha256=self.context_digest,
                context=deepcopy(context),source_request_id=request_id,complete_request=True,
                provenance='physical_ES_base_plus_original_CPU_UD_delta',adapter=context['adapter'])
            return owned
        admitted=self.pool.insert(candidate,1,frequencies,materialize=materialize)
        if admitted:
            entry=self.pool.entries[candidate.key]
            self.seals[entry.key]=dict(metadata=deepcopy(entry.qkv_metadata),size_bytes=entry.size_bytes,
                specs={f'{layer}.{role}':spec(t) for layer,block in entry.tensors.items() for role,t in zip('qkv',block)})
            self.validate(entry,context)
            self.event('INSERT',source_request_id=request_id,physical_bytes=self.bytes)
        return admitted

    def reused(self,entry):
        self.pool.record_reuse(entry)
        self.event('PHYSICAL_REUSE',qkv_outputs=self.context['identity']['layers']*3)


def invalid_guard_checks(cache):
    """Metadata/lookup rejection fixtures, not executions of another adapter."""
    context=deepcopy(cache.context);entry=cache.pool.entries[cache.key(context)];results=[]
    for name in ('tokens','adapter','positions','context','model','metadata','incomplete','dtype','shape'):
        changed=deepcopy(context);saved_metadata=entry.qkv_metadata;saved_tensors=entry.tensors
        try:
            if name=='tokens':changed['token_ids'][0]+=1
            if name=='adapter':changed['adapter']='CORRUPTED_METADATA_NOT_AN_EXECUTED_ADAPTER'
            if name=='positions':changed['positions'][0]+=1
            if name=='context':changed['context']['prefix_token_ids']=[99]
            if name=='model':changed['identity']['base_weight_sha256']='0'*64
            if name=='metadata':entry.qkv_metadata=dict(entry.qkv_metadata,complete_request=False)
            if name=='incomplete':entry.tensors={k:v for k,v in entry.tensors.items() if k!=0}
            if name in ('dtype','shape'):
                tensor=entry.tensors[0][0].clone()
                if hasattr(tensor,'fixture_bytes'):
                    if name=='dtype':tensor.dtype='WRONG_DTYPE'
                    else:tensor.shape=(tensor.shape[1],tensor.shape[0],tensor.shape[2])
                else:
                    tensor=tensor.float() if name=='dtype' else tensor.transpose(0,1).contiguous()
                entry.tensors=dict(entry.tensors);entry.tensors[0]=(tensor,*entry.tensors[0][1:])
            rejected=False;reason=None
            try:cache.validate(entry,changed)
            except (ValueError,KeyError,TypeError,AttributeError) as exc:rejected=True;reason=str(exc)
            require(rejected,'unsafe candidate accepted: '+name)
            results.append(dict(case=name,rejected=rejected,reason=reason,real_alternative_adapter_or_model_executed=False))
        finally:entry.qkv_metadata=saved_metadata;entry.tensors=saved_tensors
    cache.validate(entry,context)
    return results
