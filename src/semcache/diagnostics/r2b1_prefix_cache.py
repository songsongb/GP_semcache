"""Exact-prefix namespace and independent request KV snapshots. No model imports at import time."""
from dataclasses import asdict, dataclass
from contextlib import nullcontext
import inspect
from semcache.metrics.r1 import digest


class PrefixError(RuntimeError):
    def __init__(self, code, message):
        self.code=code
        super().__init__(f'{code}: {message}')


@dataclass(frozen=True)
class PrefixIdentity:
    model_id: str
    model_revision: str
    base_weights: str
    tokenizer_id: str
    tokenizer_revision: str
    adapter: str
    adapter_weights: str
    dtype: str
    device: str
    backend: str = 'eager'
    tokenization: str = 'tokenizer-default-BOS; exact token IDs; batch1; no padding'
    position_policy: str = 'OPT learned absolute positions from zero'
    mask_policy: str = 'all-ones 2D input; native causal mask'
    output_attentions: bool = True
    use_cache: bool = True
    eval_mode: bool = True
    fixture: str = 'synthetic PEFT LoRA rank8 alpha8 dropout0'
    adapter_configuration: str = ''


REJECTIONS=dict(model_id='REJECT_MODEL_MISMATCH',model_revision='REJECT_MODEL_REVISION_MISMATCH',
    base_weights='REJECT_BASE_WEIGHT_MISMATCH',tokenizer_id='REJECT_TOKENIZER_MISMATCH',
    tokenizer_revision='REJECT_TOKENIZER_REVISION_MISMATCH',adapter='REJECT_ADAPTER_MISMATCH',
    adapter_weights='REJECT_ADAPTER_WEIGHT_MISMATCH',adapter_configuration='REJECT_ADAPTER_CONFIG_MISMATCH',dtype='REJECT_DTYPE_MISMATCH',device='REJECT_DEVICE_MISMATCH',
    backend='REJECT_BACKEND_MISMATCH',position_policy='REJECT_POSITION_MISMATCH',mask_policy='REJECT_MASK_MISMATCH')


def validate_request(source, target, prefix_ids, target_ids, *, source_start=0, position_start=None, mask=None):
    if source_start!=0: raise PrefixError('REJECT_NOT_PREFIX','source must start at absolute position zero')
    if any(type(v) is not int or v<0 for v in (*prefix_ids,*target_ids)):
        raise PrefixError('REJECT_TOKEN_IDS','nonnegative integer token IDs required; bool/float are not exact IDs')
    p=len(prefix_ids); n=len(target_ids)
    if not 0<p<n: raise PrefixError('REJECT_LENGTH','require 0<P<L; P=0 is a separate native control')
    for name,value in asdict(source).items():
        if getattr(target,name)!=value: raise PrefixError(REJECTIONS.get(name,'REJECT_NAMESPACE_MISMATCH'),name)
    if tuple(target_ids[:p])!=tuple(prefix_ids): raise PrefixError('REJECT_PREFIX_TOKEN_MISMATCH','exact first P IDs differ')
    if position_start is not None and position_start!=p: raise PrefixError('REJECT_POSITION_MISMATCH','suffix position must start at P')
    if mask is not None and (len(mask)!=n or any(type(v) is not int or v!=1 for v in mask)):
        raise PrefixError('REJECT_MASK_MISMATCH','batch1 unpadded total-length all-ones mask required')
    return dict(prefix_length=p,suffix_length=n-p,total_length=n,cache_hit=True)


def cache_pairs(cache):
    """Read actual self-attention K/V; unknown Cache implementations fail closed."""
    if isinstance(cache,tuple):
        pairs=cache; mode='legacy_tuple'; factory=None
    else:
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError as exc: raise PrefixError('UNSUPPORTED_CACHE_API',str(exc)) from exc
        if type(cache) is not DynamicCache:
            raise PrefixError('UNSUPPORTED_CACHE_API',f'Only native tuple or exact DynamicCache supported, got {type(cache)}')
        if not callable(getattr(cache,'to_legacy_cache',None)) or not callable(getattr(DynamicCache,'from_legacy_cache',None)):
            raise PrefixError('UNSUPPORTED_CACHE_API','DynamicCache legacy conversion unavailable')
        if getattr(cache,'offloading',False): raise PrefixError('UNSUPPORTED_CACHE_API','offloaded cache outside B1')
        if any(getattr(layer,'is_sliding',False) for layer in getattr(cache,'layers',())):
            raise PrefixError('UNSUPPORTED_CACHE_API','sliding/truncated cache outside OPT baseline')
        pairs=cache.to_legacy_cache(); mode='dynamic_cache_from_legacy'; factory=DynamicCache.from_legacy_cache
    if not pairs or any(not isinstance(pair,(tuple,list)) or len(pair)!=2 for pair in pairs):
        raise PrefixError('UNSUPPORTED_CACHE_SHAPE','every layer must contain exactly K,V')
    if mode=='dynamic_cache_from_legacy':
        if not callable(getattr(cache,'get_seq_length',None)):
            raise PrefixError('UNSUPPORTED_CACHE_API','DynamicCache sequence-length API unavailable')
        for i,pair in enumerate(pairs):
            if int(cache.get_seq_length(i))!=int(pair[0].shape[2]):
                raise PrefixError('REJECT_CACHE_LENGTH','Cache metadata and tensor sequence length differ')
    return tuple(tuple(pair) for pair in pairs),mode,factory


def validate_pairs(pairs, *, layers,length,heads,head_dim,dtype,device):
    if len(pairs)!=layers: raise PrefixError('REJECT_CACHE_LAYER_COUNT',f'{len(pairs)} != {layers}')
    rows=[]
    for layer,pair in enumerate(pairs):
        for role,t in zip('kv',pair):
            shape=tuple(t.shape)
            if shape!=(1,heads,length,head_dim): raise PrefixError('REJECT_CACHE_SHAPE',f'layer {layer} {role}: {shape}')
            if str(t.dtype)!=str(dtype): raise PrefixError('REJECT_DTYPE_MISMATCH',f'cache layer {layer}')
            if str(t.device)!=str(device): raise PrefixError('REJECT_DEVICE_MISMATCH',f'cache layer {layer}')
            rows.append(dict(layer=layer,role=role,shape=list(shape),stride=list(t.stride()),dtype=str(t.dtype),
                device=str(t.device),bytes=t.numel()*t.element_size()))
    return rows


def tensor_hash(t):
    from semcache.diagnostics.r1b3_gemm_shape import tensor_sha
    return tensor_sha(t)


def tensor_stamp(t):
    from semcache.diagnostics.r1b3_fast_selection import tensor_stamp as stamp
    return stamp(t)


def clone_tensor(t):
    import torch
    # Canonical and request copies have version counters even if source used inference_mode.
    with torch.inference_mode(False): return t.detach().clone()


class PrefixSnapshot:
    """Actual independent source-prefill output, cloned once into a canonical snapshot.

    Every materialize clones again inside the request timer. Mutable cache objects
    never see canonical tensors, even when legacy tuple handling would be immutable.
    """
    def __init__(self,cache,identity,prefix_ids,*,layers,heads,head_dim,source_path):
        if source_path!='PREFIX_SOURCE_PREFILL': raise PrefixError('REJECT_SOURCE_PROVENANCE','independent source prefill required')
        self.identity=identity; self.prefix_ids=tuple(prefix_ids); self.length=len(prefix_ids)
        if self.length<1: raise PrefixError('REJECT_LENGTH','no empty prefix snapshot')
        if any(type(v) is not int or v<0 for v in self.prefix_ids): raise PrefixError('REJECT_TOKEN_IDS','integer source IDs required')
        original,self.mode,self.factory=cache_pairs(cache)
        self.layout=dict(layers=layers,length=self.length,heads=heads,head_dim=head_dim,dtype=identity.dtype,device=identity.device)
        validate_pairs(original,**self.layout)
        self.pairs=tuple(tuple(clone_tensor(t) for t in pair) for pair in original)
        self.rows=validate_pairs(self.pairs,**self.layout)
        if any(a.data_ptr()==b.data_ptr() for old,new in zip(original,self.pairs) for a,b in zip(old,new)):
            raise PrefixError('REJECT_CACHE_ALIAS','canonical snapshot aliases source output')
        self.stamps=self._stamps(); self.hashes=self._hashes(); self.bytes=sum(r['bytes'] for r in self.rows)
        self.source_hashes=[tensor_hash(t) for pair in original for t in pair]
        if self.source_hashes!=self.hashes: raise PrefixError('REJECT_SOURCE_COPY_MISMATCH','canonical clone differs from actual source KV')
        self.identity_hash=digest(dict(namespace=asdict(identity),prefix_ids=self.prefix_ids,hashes=self.hashes))
        self.sealed=(identity,self.prefix_ids,self.length,tuple(self.layout.items()),self.mode,self.factory)

    def _stamps(self): return tuple((id(t),tensor_stamp(t)) for pair in self.pairs for t in pair)
    def _hashes(self): return [tensor_hash(t) for pair in self.pairs for t in pair]

    def check(self):
        if (self.identity,self.prefix_ids,self.length,tuple(self.layout.items()),self.mode,self.factory)!=self.sealed:
            raise PrefixError('REJECT_SOURCE_CACHE_MUTATION','snapshot namespace/layout changed')
        if digest(dict(namespace=asdict(self.identity),prefix_ids=self.prefix_ids,hashes=self.hashes))!=self.identity_hash:
            raise PrefixError('REJECT_SOURCE_CACHE_MUTATION','snapshot identity/hash binding changed')
        validate_pairs(self.pairs,**self.layout)
        if self._stamps()!=self.stamps or self._hashes()!=self.hashes:
            raise PrefixError('REJECT_SOURCE_CACHE_MUTATION','canonical source content/metadata changed')
        return dict(status='UNCHANGED',prefix_length=self.length,identity=self.identity_hash,rows=self.rows,hashes=self.hashes,
            source_capture_hashes=self.source_hashes,capture_equals_snapshot=True,cache_api=self.mode)

    def materialize(self,identity,ids,*,position_start,mask):
        validate_request(self.identity,identity,self.prefix_ids,ids,position_start=position_start,mask=mask)
        # Full canonical integrity checks live outside timing. Namespace lookup and
        # the copies/conversion necessary to execute the request stay inside.
        pairs=tuple(tuple(clone_tensor(t) for t in pair) for pair in self.pairs)
        request=self.factory(pairs) if self.factory else pairs
        validate_pairs(cache_pairs(request)[0],**self.layout)
        return request

    def inspect_request(self,cache):
        pairs,mode,_=cache_pairs(cache)
        validate_pairs(pairs,**self.layout)
        import torch
        for original,copy in zip(self.pairs,pairs):
            for a,b in zip(original,copy):
                if a.data_ptr()==b.data_ptr() or not torch.equal(a,b):
                    raise PrefixError('REJECT_CACHE_ALIAS','request input must be equal independent tensors')
        return dict(status='INDEPENDENT_EQUAL',length=self.length,api=mode)

    def inspect_output(self,cache,total_length):
        import torch
        pairs,mode,_=cache_pairs(cache)
        rows=validate_pairs(pairs,**dict(self.layout,length=total_length))
        for original,out in zip(self.pairs,pairs):
            for a,b in zip(original,out):
                if not torch.equal(a,b[:,:,:self.length,:]):
                    raise PrefixError('REJECT_PAST_CONTENT_MUTATION','output past differs from captured source')
        return dict(length=total_length,api=mode,rows=rows,prefix_preserved_exact=True)


class OPTCacheAPI:
    def __init__(self,model,decoder):
        self.base=model.get_base_model() if hasattr(model,'peft_config') else model
        names=('input_ids','attention_mask','past_key_values','use_cache','return_dict','output_attentions')
        self.signatures={name:str(inspect.signature(obj.forward)) for name,obj in [('wrapper',model),('native',self.base),('decoder',decoder),('embed_positions',decoder.embed_positions)]}
        native=inspect.signature(self.base.forward).parameters; dec=inspect.signature(decoder.forward).parameters
        wrapper=inspect.signature(model.forward).parameters
        if any(n not in native or n not in dec for n in names): raise PrefixError('UNSUPPORTED_API','native OPT/cache keyword contract missing')
        if not any(p.kind==inspect.Parameter.VAR_KEYWORD for p in wrapper.values()) and any(n not in wrapper for n in names):
            raise PrefixError('UNSUPPORTED_API','PEFT wrapper does not accept required inputs')
        self.optional=[n for n in ('position_ids','cache_position') if n in native and n in dec]
        self.logit_options=[n for n in ('logits_to_keep','num_logits_to_keep') if n in native]
        if getattr(decoder.embed_positions,'offset',None) is None:
            raise PrefixError('UNSUPPORTED_POSITION_HANDLING','OPT learned positional embedding with offset required')
        self.metadata=dict(signatures=self.signatures,explicit_optional=self.optional,all_logits_options=self.logit_options,
            positions='absolute range; explicit supported kwargs only; actual embedding/mask audited',cache_api='DETECTED_FROM_ACTUAL_OUTPUT')

    def kwargs(self,ids,start,total,device):
        import torch
        out=dict(input_ids=torch.tensor([list(ids)],dtype=torch.long,device=device),
            attention_mask=torch.ones((1,total),dtype=torch.long,device=device),
            use_cache=True,return_dict=True,output_attentions=True)
        absolute=torch.arange(start,start+len(ids),device=device)
        if 'position_ids' in self.optional: out['position_ids']=absolute.unsqueeze(0)
        if 'cache_position' in self.optional: out['cache_position']=absolute
        for option in self.logit_options: out[option]=0
        return out
