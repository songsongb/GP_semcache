"""Single active prompt namespace over genuine R3-B SemCache GPU tensor storage."""
from copy import deepcopy
from .communication import digest,require
from .r3b_cache import ExactRawCache,request_context
from .r4a_session import positions


class PromptCache:
    def __init__(self,pm,prepared_sha,gpu_uuid,device,capacity_bytes=64*2**20,*,fixture=False):
        self.pm=pm;self.prepared_sha=prepared_sha;self.gpu_uuid=gpu_uuid;self.device=device
        self.capacity_bytes=capacity_bytes;self.fixture=fixture;self.active=None;self.events=[]
    def bind(self,prompt):
        ids,mask=prompt['token_ids'],prompt['attention_mask'];require(len(ids)==len(mask),'prompt/mask mismatch')
        identity=deepcopy(self.pm['identity']);identity.update(length=len(ids),token_ids=list(ids),token_ids_sha256=digest(ids))
        identity['endpoint']=dict(batch_size=1,prompt_tokens=len(ids),attention='eager',use_cache=True,
            output_attentions=True,output_head='native_all_token_logits')
        view=dict(self.pm,identity=identity)
        ctx=request_context(view,self.prepared_sha,self.gpu_uuid)
        ctx.update(schema='R4A_EXACT_FULL_PROMPT_V1',positions=positions(mask),attention_mask=list(mask),
            input_prefix_scope='all prompt tokens including declared left padding; no previous-request KV',
            adapter_source='controlled_untrained_fixture')
        if self.active is None or self.active.context!=ctx:
            if self.active is not None:
                self.events.append(dict(kind='NAMESPACE_SWITCH_EVICTION',released_bytes=self.active.bytes))
                self.active.reset('R4 single active full-prompt namespace')
            self.active=ExactRawCache(ctx,self.capacity_bytes,self.device,fixture=self.fixture)
            self.events.append(dict(kind='NAMESPACE_BIND',prompt_sha256=digest(dict(tokens=ids,mask=mask))))
        return self.active
