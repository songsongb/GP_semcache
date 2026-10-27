"""Untimed physical Forward/causal-position proof and B1 numerical contracts."""
from contextlib import contextmanager
import inspect
from semcache.diagnostics.r2b1_prefix_cache import PrefixError
from semcache.diagnostics.r1b3_fp16_divergence import logit_metrics


def geometry(length,prefix,max_positions=None):
    if type(length) is not int or type(prefix) is not int or not 0<=prefix<length:
        raise PrefixError('REJECT_LENGTH','need L>=1, 0<=P<L, S>=1')
    if max_positions is not None and length>max_positions: raise PrefixError('REJECT_MAX_POSITION','input exceeds model positions')
    return dict(total_length=length,prefix_length=prefix,suffix_length=length-prefix,cache_hit=prefix>0)


def validate_forward_rows(rows,layers,query_length):
    expected={(i,role) for i in range(layers) for role in ('layer','q','k','v','fc1','fc2','attention')}
    keys=[(r['layer'],r['role']) for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise PrefixError('MISSING_FORWARD_EVIDENCE','every layer/QKV/FFN/attention must execute exactly once')
    if any(r['tokens']!=query_length or r['batch']!=1 for r in rows):
        raise PrefixError('REJECT_FORWARD_LENGTH','query/projection/FFN did not process expected S only')
    return dict(status='VERIFIED',query_tokens=query_length,layers=layers,observed_calls=len(rows),rows=rows)


class ForwardAudit:
    def __init__(self,model,adapter,decoder,query_length,past_length,total_length,path,expected_ids):
        self.model=model; self.adapter=adapter; self.decoder=decoder; self.n=query_length; self.p=past_length; self.total=total_length; self.path=path
        self.expected_ids=list(expected_ids); self.observed_ids=[]
        self.rows=[]; self.masks=[]; self.positions=[]; self.root_calls=0; self.embedding_inputs=[]

    @contextmanager
    def hooks(self):
        import torch
        handles=[]
        def before(layer,role):
            def hook(module,args,kwargs):
                x=args[0] if args else kwargs.get('hidden_states')
                if x is None: raise PrefixError('MISSING_FORWARD_EVIDENCE',role+' hidden input missing')
                if x.ndim==3: batch,tokens=int(x.shape[0]),int(x.shape[1])
                elif x.ndim==2 and role in ('fc1','fc2'): batch,tokens=1,int(x.shape[0])
                else: raise PrefixError('UNSUPPORTED_API',f'{role} input shape {x.shape}')
                self.rows.append(dict(layer=layer,role=role,batch=batch,tokens=tokens,shape=list(x.shape)))
                if role=='attention':
                    bound=inspect.signature(module.forward).bind_partial(*args,**kwargs).arguments
                    mask=bound.get('attention_mask')
                    if mask is None or tuple(mask.shape)!=(1,1,self.n,self.total):
                        raise PrefixError('UNVERIFIED_POSITION',f'causal eager mask not observed: {None if mask is None else mask.shape}')
                    mask=mask.detach().cpu()[0,0]
                    allowed=torch.arange(self.total).unsqueeze(0)<=torch.arange(self.p,self.p+self.n).unsqueeze(1)
                    if not torch.isfinite(mask[allowed]).all() or not (mask[allowed]==0).all():
                        raise PrefixError('REJECT_MASK_MISMATCH','past/current positions must remain unmasked')
                    if (~allowed).any() and not (mask[~allowed]<-1e4).all():
                        raise PrefixError('REJECT_MASK_MISMATCH','future suffix positions not causally masked')
                    self.masks.append(dict(layer=layer,shape=[1,1,self.n,self.total],absolute_queries=[self.p,self.p+self.n],
                        total_kv_context=self.total,causal_mask_verified=True,prefix_columns_visible=self.p))
            return hook
        def root(module,args,kwargs):
            self.root_calls+=1
            x=kwargs.get('input_ids',args[0] if args else None); mask=kwargs.get('attention_mask')
            if x is None or tuple(x.shape)!=(1,self.n) or mask is None or tuple(mask.shape)!=(1,self.total) or not (mask==1).all():
                raise PrefixError('REJECT_FORWARD_LENGTH','root must receive suffix-only input and full mask')
            actual=x[0].detach().cpu().tolist()
            if actual!=self.expected_ids: raise PrefixError('REJECT_PREFIX_TOKEN_MISMATCH','actual root token IDs differ from controlled input')
            self.observed_ids.append(actual)
        def embed(module,args):
            if tuple(args[0].shape)!=(1,self.n): raise PrefixError('REJECT_FORWARD_LENGTH','token embedding recomputed prefix')
            if args[0][0].detach().cpu().tolist()!=self.expected_ids:
                raise PrefixError('REJECT_PREFIX_TOKEN_MISMATCH','embedding input IDs differ from root suffix')
            self.embedding_inputs.append(list(args[0].shape))
        def positions(module,args,kwargs,out):
            indices=torch.arange(self.p,self.p+self.n,device=module.weight.device)+module.offset
            expected=module.weight.index_select(0,indices).unsqueeze(0)
            if out.shape!=expected.shape or not torch.equal(out,expected):
                raise PrefixError('UNSUPPORTED_POSITION_HANDLING','actual OPT position embedding differs from absolute suffix range')
            self.positions.append(dict(shape=list(out.shape),absolute_positions=list(range(self.p,self.p+self.n)),
                learned_embedding_offset=module.offset,embedding_exact=True))
        try:
            handles.append(self.model.register_forward_pre_hook(root,with_kwargs=True))
            handles.append(self.decoder.embed_tokens.register_forward_pre_hook(embed))
            handles.append(self.decoder.embed_positions.register_forward_hook(positions,with_kwargs=True))
            for i,layer in enumerate(self.adapter.layers):
                modules={'layer':layer,'attention':layer.self_attn,'fc1':layer.fc1,'fc2':layer.fc2,**self.adapter.projection_modules(i)}
                for role,module in modules.items(): handles.append(module.register_forward_pre_hook(before(i,role),with_kwargs=True))
            yield self
            validate_forward_rows(self.rows,len(self.adapter.layers),self.n)
            if self.root_calls!=1 or len(self.embedding_inputs)!=1 or len(self.positions)!=1 or len(self.masks)!=len(self.adapter.layers):
                raise PrefixError('MISSING_FORWARD_EVIDENCE','root/embedding/position/mask evidence incomplete')
        finally:
            for handle in handles: handle.remove()

    def report(self):
        return dict(path=self.path,**validate_forward_rows(self.rows,len(self.adapter.layers),self.n),
            root_forward_calls=self.root_calls,actual_input_ids=self.observed_ids,token_embedding_inputs=self.embedding_inputs,positions=self.positions,attention_masks=self.masks,
            prefix_source_forwards_inside_target=0 if self.path=='EXACT_PREFIX_WARM' and self.root_calls==1 else None,
            prefix_layer_token_rows_skipped=self.p*len(self.adapter.layers),
            suffix_attention_uses_total_kv=self.total)


def compare_suffix(full,cached,prefix):
    if full.ndim!=3 or cached.ndim!=3 or full.shape[0]!=1 or tuple(full[:,prefix:].shape)!=tuple(cached.shape):
        raise PrefixError('REJECT_LOGIT_SHAPE','require Full[:,P:,:] versus suffix [:,:,:]')
    return dict(alignment='FULL[:,P:L,:] versus WARM[:,0:S,:]; next-token logits at corresponding input positions',
        suffix=logit_metrics(full[:,prefix:,:],cached),last_position=logit_metrics(full[:,-1:,:],cached[:,-1:,:]))
