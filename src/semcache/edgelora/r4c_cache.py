"""C1 variable-length standalone prefixes over frozen B1/R3-B owned storage."""
from copy import deepcopy
import time

from .communication import digest,require
from .r4b2_cache import PrefixCache as FrozenPrefixCache
from .r4b1_cache import MultiPromptCache
from .r4b2_kv import attention_kv,construct_past,spec
from .r4a_cache import PromptCache
from .r4c_session import VERSION


class PrefixCache(FrozenPrefixCache):
    def context_for(self,prompt):
        ids=list(prompt['token_ids']); mask=list(prompt['attention_mask']); n=len(ids)
        require(0<n<=512 and all(type(t) is int and t>=0 for t in ids),'invalid C1 actual token IDs')
        require(mask==[1]*n and prompt.get('positions',list(range(n)))==list(range(n)),'C1 prefix/target mask/positions mismatch')
        require(prompt.get('prompt_sha256',digest(dict(tokens=ids,mask=mask)))==digest(dict(tokens=ids,mask=mask)),'C1 prompt digest mismatch')
        c=deepcopy(PromptCache(self._pm,self.prepared_sha,self.gpu_uuid,self.device,fixture=self.fixture).bind(prompt).context)
        c.update(schema='R4C_STANDALONE_PREFIX_V1',prefix_scope='explicit completed standalone source; no slicing longer source',
            opt_contract=deepcopy(self.contract),tokenizer=deepcopy(self.tokenizer),execution_sources=deepcopy(self.sources),storage_device=self.device)
        c['identity']['endpoint']['output_attentions']=self.contract['output_attentions']
        return c
    def remove(self,key,kind,reason):
        # Use the shared byte owner directly: B2.remove emits a B2 control
        # message, which must never reach the independent C1 protocol endpoint.
        existed=key in self.entries;MultiPromptCache.remove(self,key,kind,reason)
        self.proofs.pop(key,None);self.proof_seals.pop(key,None)
        if existed:self.control(dict(action='EVICT_PREFIX',version=VERSION,cohort_sha256=self.cohort,entry_id=key))
    def admit(self,prompt,blocks,proof,request_id,*,completed):
        self.check_namespace()
        n=len(prompt['token_ids']);require(0<n<512,'unsupported standalone prefix length')
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
        grant=dict(action='REGISTER_PREFIX',version=VERSION,cohort_sha256=self.cohort,entry_id=key,source_request_id=request_id,
            prefix_length=n,prefix_token_ids=list(prompt['token_ids']),proof_sha256=self.proof_seals[key])
        self.control(grant);return grant
    def find(self,target,request_id,*,expected_length=None,cohort=None):
        started=time.monotonic_ns()
        self.request_id=request_id
        self.check_namespace()
        self.context_for(target)  # Validate actual full IDs, mask, positions and optional digest.
        require(cohort in (None,self.cohort),'model/adapter/tokenizer/device/source identity mismatch')
        require(expected_length is None or type(expected_length) is int and 0<expected_length<512,'unsupported requested prefix length')
        ids=target['token_ids'];candidates=sorted(self.entries,key=lambda k:len(self.entries[k].context['token_ids']),reverse=True)
        for key in candidates:
            c=self.entries[key].context;p=len(c['token_ids'])
            if p>=len(ids) or ids[:p]!=c['token_ids']:continue
            require(c==self.context_for(dict(token_ids=ids[:p],attention_mask=[1]*p)),'stored standalone prefix context incompatible with current target')
            require(expected_length is None or p==expected_length,'MISS_OR_REJECT: declared prefix differs from longest eligible exact prefix')
            require(key in self.proofs and digest(self.proofs[key])==self.proof_seals[key],'corrupted prefix proof digest')
            entry=self.lookup(c)
            if entry is None:raise ValueError('MISS_OR_REJECT: prefix tensor integrity rejected')
            self.synchronize();validated=time.monotonic_ns()
            proof=self.proofs[key];pairs=attention_kv(entry.tensors,proof['heads'],self._pm['identity']['hidden_size'])
            for i,pair in enumerate(pairs):
                require(spec(pair[0])==proof['layers'][i]['actual_k'] and spec(pair[1])==proof['layers'][i]['actual_v'],
                    'cached derived prefix K/V integrity failed')
            past=construct_past(pairs,self.cache_type,proof['cache_api'])
            self.synchronize();self.last_lookup_timings=dict(cache_lookup_validation_ms=(validated-started)/1e6,
                kv_construction_and_validation_ms=(time.monotonic_ns()-validated)/1e6)
            grant=dict(entry_id=key,source_request_id=entry.qkv_metadata['source_request_id'],prefix_length=p,proof_sha256=self.proof_seals[key])
            return entry,grant,past
        self.event('MISS',context_sha256=None,reason='no exact standalone prefix matches target')
        raise ValueError('MISS_OR_REJECT: no validated exact prefix; no fallback')
