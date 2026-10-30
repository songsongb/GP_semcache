"""Standalone-prefix index over unchanged B1/R3-B owned Raw-QKV storage."""
from copy import deepcopy

from .communication import digest,require
from .r4b1_cache import MultiPromptCache
from .r4b2_kv import attention_kv,construct_past,spec
from .r4b2_session import PREFIX_LENGTHS


class PrefixCache(MultiPromptCache):
    def __init__(self,*args,contract,observed_cache_type,control=lambda m:None,**kwargs):
        self.contract=deepcopy(contract);self.cache_type=observed_cache_type;self.proofs={};self.proof_seals={};self.control=control
        super().__init__(*args,**kwargs)
        self.cohort=digest(dict(identity=self._pm['identity'],prepared=self.prepared_sha,tokenizer=self.tokenizer,sources=self.sources,
            device=self.device,gpu_uuid=self.gpu_uuid,contract=self.contract))
    def check_namespace(self):
        require(self.cohort==digest(dict(identity=self._pm['identity'],prepared=self.prepared_sha,tokenizer=self.tokenizer,sources=self.sources,
            device=self.device,gpu_uuid=self.gpu_uuid,contract=self.contract)),'model/adapter/tokenizer/weights/device/source namespace changed')
    def context_for(self,prompt):
        require(prompt['attention_mask']==[1]*len(prompt['token_ids']),'B2 prefix/target prompts require zero padding')
        c=super().context_for(prompt);c.update(schema='R4B2_STANDALONE_PREFIX_V2',prefix_scope='explicit standalone source; no slicing longer source',
            opt_contract=deepcopy(self.contract));return c
    def remove(self,key,kind,reason):
        existed=key in self.entries;super().remove(key,kind,reason)
        self.proofs.pop(key,None);self.proof_seals.pop(key,None)
        if existed:self.control(dict(action='EVICT_PREFIX',b2_version=2,cohort_sha256=self.cohort,entry_id=key))
    def admit(self,prompt,blocks,proof,request_id,*,completed):
        self.check_namespace()
        n=len(prompt['token_ids']);require(n in PREFIX_LENGTHS,'unsupported standalone prefix length')
        require(completed is True and proof['prefix_length']==n and self.cache_type is not None and
            len(proof['layers'])==self._pm['identity']['layers'],'incomplete source or incompatible Raw-KV proof')
        derived=attention_kv(blocks,proof['heads'],self._pm['identity']['hidden_size'])
        for layer,r in enumerate(proof['layers']):
            require(r['layer']==layer and r['bitwise_equal'] is True and r['derived_k']==r['actual_k']==spec(derived[layer][0]) and
                r['derived_v']==r['actual_v']==spec(derived[layer][1]) and r['raw_k']==spec(blocks[layer][1]) and r['raw_v']==spec(blocks[layer][2]),
                'source Raw K/V proof differs from actual admitted tensors')
        require(len(self.entries)<64,'bounded prefix index exhausted')
        self.request_id=request_id;view=self.bind(prompt);view.reset('explicit cold source replacement')
        allowed=view.insert(view.context,blocks,request_id,completed=True)
        if not allowed:return None
        key=view.context_digest;self.proofs[key]=deepcopy(proof);self.proof_seals[key]=digest(proof)
        grant=dict(action='REGISTER_PREFIX',b2_version=2,cohort_sha256=self.cohort,entry_id=key,source_request_id=request_id,
            prefix_length=n,prefix_token_ids=list(prompt['token_ids']),proof_sha256=self.proof_seals[key])
        self.control(grant);return grant
    def find(self,target,request_id,*,expected_length=None,cohort=None):
        self.request_id=request_id
        self.check_namespace()
        self.context_for(target)  # Validate actual full IDs, mask, positions and optional digest.
        require(cohort in (None,self.cohort),'model/adapter/tokenizer/device/source identity mismatch')
        require(expected_length is None or expected_length in PREFIX_LENGTHS,'unsupported requested prefix length')
        ids=target['token_ids'];candidates=sorted(self.entries,key=lambda k:len(self.entries[k].context['token_ids']),reverse=True)
        for key in candidates:
            c=self.entries[key].context;p=len(c['token_ids'])
            if p>=len(ids) or ids[:p]!=c['token_ids']:continue
            require(c==self.context_for(dict(token_ids=ids[:p],attention_mask=[1]*p)),'stored standalone prefix context incompatible with current target')
            require(expected_length is None or p==expected_length,'MISS_OR_REJECT: declared prefix differs from longest eligible exact prefix')
            require(key in self.proofs and digest(self.proofs[key])==self.proof_seals[key],'corrupted prefix proof digest')
            entry=self.lookup(c)
            if entry is None:raise ValueError('MISS_OR_REJECT: prefix tensor integrity rejected')
            proof=self.proofs[key];pairs=attention_kv(entry.tensors,proof['heads'],self._pm['identity']['hidden_size'])
            for i,pair in enumerate(pairs):
                require(spec(pair[0])==proof['layers'][i]['actual_k'] and spec(pair[1])==proof['layers'][i]['actual_v'],
                    'cached derived prefix K/V integrity failed')
            past=construct_past(pairs,self.cache_type,proof['cache_api'])
            grant=dict(entry_id=key,source_request_id=entry.qkv_metadata['source_request_id'],prefix_length=p,proof_sha256=self.proof_seals[key])
            return entry,grant,past
        self.event('MISS',context_sha256=None,reason='no exact standalone prefix matches target')
        raise ValueError('MISS_OR_REJECT: no validated exact prefix; no fallback')
    def reused(self,context,entry):
        key=digest(context);require(key in self.entries,'cannot claim prefix reuse after eviction')
        owner=self.entries[key];owner.validate(entry,context);owner.pool.record_reuse(entry)
        self.event('PHYSICAL_REUSE',context_sha256=key,source_request_id=entry.qkv_metadata['source_request_id'],
            raw_kv_projections_consumed=2*self._pm['identity']['layers'],cached_q_consumed=False,
            prefix_tokens=len(context['token_ids']))
