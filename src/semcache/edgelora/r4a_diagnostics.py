"""Bounded observations only; never repair or discard nonfinite model output."""
from contextlib import contextmanager
import inspect
import math

from .communication import digest,require
from .r4a_session import MAX_NEW,MAX_PROMPT,positions


def logit_counts(logits,shape,mask,*,torch_backend):
    """Reduce on device to at most 128 rows of three integer counts.

    Logits cover current queries only. On decode the full mask includes cached
    padding, but the single returned query is valid; do not label it as padding.
    """
    require(len(shape)==3 and shape[0]==1 and 1<=shape[1]<=MAX_PROMPT and shape[2]>0,
        'bounded batch-one logit shape required')
    positions(mask);n,vocab=shape[1:];require(n<=len(mask)<=MAX_PROMPT+MAX_NEW,'logit/mask length mismatch')
    query_mask=mask[-n:]
    if torch_backend:
        import torch
        require(list(logits.shape)==shape,'model logit shape changed')
        rows=torch.stack((torch.isnan(logits[0]).sum(-1),torch.isposinf(logits[0]).sum(-1),
            torch.isneginf(logits[0]).sum(-1)),dim=-1).detach().cpu().tolist()
    else:
        require(len(logits)==n*vocab,'fixture logit shape changed');rows=[]
        for i in range(n):
            values=logits[i*vocab:(i+1)*vocab]
            rows.append([sum(math.isnan(v) for v in values),sum(v==math.inf for v in values),sum(v==-math.inf for v in values)])
    def counts(indices):
        indices=list(indices);nan,pos,neg=(sum(rows[i][j] for i in indices) for j in range(3))
        total=len(indices)*vocab;bad=nan+pos+neg
        return dict(positions=len(indices),elements=total,finite=total-bad,nonfinite=bad,nan=nan,positive_inf=pos,negative_inf=neg)
    return dict(all_logits=counts(range(n)),last_position_logits=counts([n-1]),
        valid_position_logits=counts(i for i,m in enumerate(query_mask) if m),
        padded_position_logits=counts(i for i,m in enumerate(query_mask) if not m))


@contextmanager
def observe_opt_positions(embedding,mask,past_length):
    """Check the actual OPT position-module call and its learned lookup output.

    This hook returns None. It neither changes positions nor replaces outputs.
    Unknown APIs fail closed instead of inferring correctness from our ledger.
    """
    import torch
    signature=inspect.signature(embedding.forward);observations=[];expected=positions(mask)[past_length:]
    require('attention_mask' in signature.parameters and 'past_key_values_length' in signature.parameters and
        embedding.offset==2 and embedding.max_norm is None,'unsupported actual OPT positional embedding API')
    def observe(module,args,kwargs,out):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();actual=bound.arguments
        observed_mask=actual['attention_mask'].detach().cpu().tolist()
        require(observed_mask==[mask] and actual['past_key_values_length']==past_length,
            'actual OPT positional mask/past length mismatch')
        ids=actual.get('position_ids')
        if ids is not None:
            require(ids.detach().cpu().tolist()==[expected],'actual OPT position IDs mismatch')
        # Verify the actual returned positional embeddings, even on APIs which
        # derive IDs internally. The only extra computation is a bounded lookup.
        indices=torch.tensor([expected],dtype=torch.long,device=module.weight.device)+module.offset
        reference=torch.nn.functional.embedding(indices,module.weight,module.padding_idx,module.max_norm,
            module.norm_type,module.scale_grad_by_freq,module.sparse)
        require(torch.equal(out,reference),'actual OPT positional lookup output mismatch')
        observations.append(dict(source='ACTUAL_OPT_POSITION_MODULE',positions=expected,embedding_indices=[p+module.offset for p in expected],
            attention_mask_sha256=digest(mask),past_length=past_length,explicit_position_ids=ids is not None,lookup_output_equal=True))
        return None
    handle=embedding.register_forward_hook(observe,with_kwargs=True)
    try:
        yield observations
        require(len(observations)==1,'actual OPT positional module missing or called repeatedly')
    finally:handle.remove()
