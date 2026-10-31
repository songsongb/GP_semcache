"""I5 length-aware owner over unchanged R3-B physical storage and I4A authorization."""
from copy import deepcopy
from .communication import digest,require
from .r4a_cache import PromptCache
from .r4b1_cache import MultiPromptCache
from .r4b2_kv import attention_kv,spec
from .r5_multi_controller import MultiOwner
from .r5_i5_protocol import VERSION


class I5Owner(MultiOwner):
    def context_for(self,prompt):
        ids=list(prompt['token_ids']); mask=list(prompt['attention_mask']); n=len(ids)
        require(0<n<=2048 and all(type(t) is int and t>=0 for t in ids),'invalid C1 actual token IDs')
        require(mask==[1]*n and prompt.get('positions',list(range(n)))==list(range(n)),'C1 prefix/target mask/positions mismatch')
        require(prompt.get('prompt_sha256',digest(dict(tokens=ids,mask=mask)))==digest(dict(tokens=ids,mask=mask)),'C1 prompt digest mismatch')
        c=deepcopy(PromptCache(self._pm,self.prepared_sha,self.gpu_uuid,self.device,fixture=self.fixture).bind(prompt).context)
        c.update(schema='I5_STANDALONE_PREFIX_V1',prefix_scope='explicit completed standalone source; no slicing longer source',
            opt_contract=deepcopy(self.contract),tokenizer=deepcopy(self.tokenizer),execution_sources=deepcopy(self.sources),storage_device=self.device)
        c['identity']['endpoint']['output_attentions']=self.contract['output_attentions']
        c['r5_multi']=dict(schema='I5_CACHE_NAMESPACE_V1',adapter_spec_sha256=digest(self.adapter_spec), registry_sha256=self.registry['sha256'],sharing_policy=self.registry['policy'],scale=self.adapter_spec['scale'],layout=self.adapter_spec['layout'],cache_format='UNCHANGED_R3B_TOTAL_RAW_QKV')
        return c
    def remove(self,key,kind,reason):
        self.index.invalidate(key)
        existed=key in self.entries;MultiPromptCache.remove(self,key,kind,reason)
        self.proofs.pop(key,None);self.proof_seals.pop(key,None)
        if existed:self.control(dict(action='EVICT_PREFIX',version=VERSION,cohort_sha256=self.cohort,entry_id=key))
    def admit(self,prompt,blocks,proof,request_id,*,completed):
        self.check_namespace()
        n=len(prompt['token_ids']);require(0<n<2048,'unsupported standalone prefix length')
        require(completed is True and proof['prefix_length']==n and self.cache_type is not None and
            len(proof['layers'])==self._pm['identity']['layers'],'incomplete source or incompatible Raw-KV proof')
        derived=attention_kv(blocks,proof['heads'],self._pm['identity']['hidden_size'])
        for layer,r in enumerate(proof['layers']):
            require(r['layer']==layer and r['bitwise_equal'] is True and r['derived_k']==r['actual_k']==spec(derived[layer][0]) and
                r['derived_v']==r['actual_v']==spec(derived[layer][1]) and r['raw_k']==spec(blocks[layer][1]) and r['raw_v']==spec(blocks[layer][2]),
                'source Raw K/V proof differs from actual admitted tensors')
        self.request_id=request_id;view=self.bind(prompt);view.reset('explicit cold source replacement')
        if len(self.entries)>=64:self.remove(next(iter(self.entries)),'EVICT','entry_limit_lru')
        allowed=view.insert(view.context,blocks,request_id,completed=True)
        if not allowed:return None
        key=view.context_digest;self.proofs[key]=deepcopy(proof);self.proof_seals[key]=digest(proof)
        grant=dict(action='REGISTER_PREFIX',version=VERSION,cohort_sha256=self.cohort,entry_id=key,source_request_id=request_id,
            prefix_length=n,prefix_token_ids=list(prompt['token_ids']),proof_sha256=self.proof_seals[key])
        self.control(grant);return grant
