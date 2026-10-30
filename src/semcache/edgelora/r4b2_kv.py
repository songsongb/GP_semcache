"""Raw K/V to attention K/V: observed layout and public cache conversion only."""
import inspect
from pathlib import Path

from .communication import digest_bytes,require
from .r3a_artifacts import versions
from .r3b_cache import raw,spec
from .r3b_fixture import ArrayTensor
from .r4a_generation import cache_pairs,past_summary


def slice_tokens(t,start,stop):
    require(t.ndim in (3,4) and 0<=start<=stop<=t.shape[-2],'invalid token slice')
    if not hasattr(t,'fixture_bytes'):return (t[:,start:stop,:] if t.ndim==3 else t[:,:,start:stop,:]).detach().contiguous()
    n,w=t.shape[-2:];heads=1 if t.ndim==3 else t.shape[1]
    values=[v for h in range(heads) for v in t.values[(h*n+start)*w:(h*n+stop)*w]]
    shape=[1,stop-start,w] if t.ndim==3 else [1,heads,stop-start,w]
    return ArrayTensor(values,shape)


def attention_kv(blocks,heads,hidden):
    require(hidden%heads==0 and blocks,'invalid head dimensions');d=hidden//heads;p=next(iter(blocks.values()))[0].shape[1];pairs=[]
    for layer in range(len(blocks)):
        pair=[]
        for t in blocks[layer][1:]:
            require(list(t.shape)==[1,p,hidden],'Raw K/V layout mismatch')
            if hasattr(t,'fixture_bytes'):
                pair.append(ArrayTensor([t.values[i*hidden+h*d+j] for h in range(heads) for i in range(p) for j in range(d)],[1,heads,p,d]))
            else:pair.append(t.detach().view(1,p,heads,d).transpose(1,2).contiguous().clone())
        pairs.append(tuple(pair))
    return tuple(pairs)


def tensor_equal(a,b):return spec(a)==spec(b)


def describe_api(past):
    if type(past) in (tuple,list):return dict(kind='legacy_tuple',class_name=type(past).__name__,module=type(past).__module__)
    cls=type(past)
    require(cls.__name__=='DynamicCache' and cls.__module__.startswith('transformers.') and
        callable(getattr(cls,'from_legacy_cache',None)) and callable(getattr(past,'to_legacy_cache',None)),
        'ARCHITECTURAL_BLOCKER: only observed legacy tuple or public DynamicCache.from_legacy_cache is supported')
    return dict(kind='dynamic_from_legacy',class_name=cls.__name__,module=cls.__module__,
        factory_signature=str(inspect.signature(cls.from_legacy_cache)),factory_source_sha256=digest_bytes(inspect.getsource(cls.from_legacy_cache).encode()))


def construct_past(pairs,observed_type,descriptor):
    # pairs are request-owned copies, never writable views of cache storage.
    if descriptor['kind']=='legacy_tuple':require(observed_type in (tuple,list),'cache API changed');past=observed_type(pairs)
    else:
        require(describe_api_type(observed_type)==descriptor,'loaded HF cache factory changed')
        past=observed_type.from_legacy_cache(tuple(pairs))
    require(len(cache_pairs(past))==len(pairs) and all(tensor_equal(a,b) for pair,ref in zip(cache_pairs(past),pairs) for a,b in zip(pair,ref)),
        'public HF cache constructor changed derived K/V contents')
    return past


def describe_api_type(cls):
    require(cls.__name__=='DynamicCache' and cls.__module__.startswith('transformers.'),'unsupported HF cache type')
    return dict(kind='dynamic_from_legacy',class_name=cls.__name__,module=cls.__module__,
        factory_signature=str(inspect.signature(cls.from_legacy_cache)),factory_source_sha256=digest_bytes(inspect.getsource(cls.from_legacy_cache).encode()))


def verify_raw_kv(blocks,past,heads,identity,device):
    p=next(iter(blocks.values()))[0].shape[1]
    summary=past_summary(past,p,identity['layers'],heads,identity['hidden_size'],identity['hidden_dtype'],device)
    pairs=attention_kv(blocks,heads,identity['hidden_size']);actual=cache_pairs(past);records=[]
    for layer,(derived,observed) in enumerate(zip(pairs,actual)):
        require(all(tensor_equal(a,b) for a,b in zip(derived,observed)),
            f'ARCHITECTURAL_BLOCKER: Raw K/V head transform differs from actual HF K/V at layer {layer}')
        records.append(dict(layer=layer,raw_k=spec(blocks[layer][1]),raw_v=spec(blocks[layer][2]),
            derived_k=spec(derived[0]),derived_v=spec(derived[1]),actual_k=spec(observed[0]),actual_v=spec(observed[1]),bitwise_equal=True))
    api=describe_api(past);construct_past(pairs,type(past),api)
    return dict(schema='r4b2_raw_kv_proof_v1',prefix_length=p,heads=heads,head_dim=identity['hidden_size']//heads,
        transform='view(1,P,heads,head_dim).transpose(1,2).contiguous().clone(); no K/V scaling or rotary transform',
        q_consumed=False,summary=summary,cache_api=api,layers=records)


def installed_contract(driver):
    if driver.fixture:return dict(scope='TEST_FIXTURE_ONLY_SCALAR_CAUSAL',heads=driver.heads,head_dim=driver.identity['hidden_size']//driver.heads,
        versions=versions(),cache_api='observed at source prefill')
    base=driver.model.get_base_model() if hasattr(driver.model,'peft_config') else driver.model
    decoder=base.model.decoder
    require(base.config.model_type=='opt' and base.config._attn_implementation=='eager','B2 supports eager OPT only')
    methods=dict(attention=decoder.layers[0].self_attn.forward,decoder=decoder.forward,
        positions=decoder.embed_positions.forward,causal_lm=base.forward)
    return dict(scope='ACTUAL_LOADED_OPT_IMPLEMENTATION',versions=versions(),heads=driver.heads,
        head_dim=driver.identity['hidden_size']//driver.heads,config=base.config.to_dict(),
        methods={k:dict(signature=str(inspect.signature(m)),source=inspect.getsource(m),
            source_sha256=digest_bytes(inspect.getsource(m).encode())) for k,m in methods.items()})


class TensorEvidence:
    def __init__(self,directory):self.directory=Path(directory);self.bytes=0
    def save(self,t,name):
        data=raw(t);self.bytes+=len(data);require(self.bytes<=4*2**30,'bounded B2 tensor evidence exceeds 4 GiB')
        path=self.directory/'tensors'/f'{name}.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        return dict(spec(t),file=str(path.relative_to(self.directory)))
