"""Experimental all-layer TOTAL-QKV projection owner and bare-Linear executor.

Attention/FFN/LM-head always execute full inputs. No arbitrary-span past-KV grants.
"""
from contextlib import contextmanager
from .communication import require,digest,digest_bytes
from .r5_sparse_qkv import gather,scatter,blob
from .r5_mstage_bridge import MStageShadow
from .r3b_fixture import ArrayTensor
from semcache.cache.cache_entry import CacheEntry
from semcache.cache.metric_manager import actual_attention_impact
from semcache.semantic.hit_selection import select_nonoverlapping

EXACT='EXACT_CONTEXT_VERIFIED'
APPROX='APPROXIMATE_SUBSEQUENCE_TOTAL_QKV'

def tensor_proof(t):return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=digest_bytes(blob(t)))
def entry_proof(entry):
    return digest(dict(metadata=entry.qkv_metadata,positions=entry.positions,tokens=entry.token_ids,size=entry.size_bytes,
                       tensors={str(i):[tensor_proof(t) for t in ts] for i,ts in entry.tensors.items()}))

class PhysicalOwner:
    def __init__(self,identity,registry,*,capacity=512*2**20,trace=lambda r:None):
        self.bridge=MStageShadow(identity,registry,capacity=capacity,trace=trace);self.trace=trace;self.active=None;self.proofs={}
    @property
    def cache(self):return self.bridge.cache
    def begin(self,req,assignment,*,mode=EXACT,allow_approximate=False,reuse=True):
        require(mode in (EXACT,APPROX) and (mode!=APPROX or allow_approximate),'approximate mode requires explicit opt-in')
        report=self.bridge.begin(req,assignment);p=self.bridge.pending
        selected=p['selected'] if reuse else []
        if mode==EXACT:selected=[h for h in selected if h.entry.positions==(h.window.start,h.window.end) and h.entry.qkv_metadata['source_token_ids'][:h.window.end]==req['token_ids'][:h.window.end]]
        self.active=dict(req=p['req'],cluster=p['cluster'],windows=p['windows'],selected=selected,mode=mode,captures={},inputs={},records=[],consumed=set(),seen=set())
        for candidate in report['candidates']:candidate['physical_rejection']=['PER_LAYER_INPUT_AND_TENSOR_VALIDATION_PENDING']
        return report
    def valid(self,entry):
        require(self.cache.entries.get(entry.key) is entry and entry.tensors is not None,'stale/missing tensor grant')
        require(entry.qkv_metadata['physical_identity']==self.bridge.identity and entry.qkv_metadata['completed'] and entry.qkv_metadata['component_scope']=='total_qkv','QKV ownership differs')
        require(self.proofs.get(entry.key)==entry_proof(entry),'cached tensor/metadata integrity failure; no fallback')
    def project(self,layer,role,x,native,session):
        a=self.active;require(a is not None and (layer,role) not in a['seen'],'duplicate/inactive projection');a['seen'].add((layer,role))
        n=len(a['req']['token_ids']);h=self.bridge.identity['hidden_size'];require(list(x.shape)==[1,n,h],'projection full shape differs')
        # Same layer span decisions across Q/K/V; OPT presents identical inputs.
        if role=='q':
            accepted=[]
            for hit in a['selected']:
                self.valid(hit.entry);require(layer in hit.entry.tensors,'missing layer tensor')
                if a['mode']==EXACT and hit.entry.qkv_metadata['provenance_mode']!=EXACT:continue
                target_input=tensor_proof(gather(x,list(range(hit.window.start,hit.window.end))))
                if a['mode']==EXACT and target_input!=hit.entry.qkv_metadata['input_proofs'][str(layer)]:continue
                accepted.append(hit)
            a['layer_hits']=accepted
        hits=a['layer_hits'];used={i for hit in hits for i in range(hit.window.start,hit.window.end)};fresh=[i for i in range(n) if i not in used]
        delta=session.correction(layer,role,x,fresh)
        base=native(gather(x,fresh)) if fresh else None
        if isinstance(x,ArrayTensor):
            from .r3a_tiny import f32
            total=None if base is None else ArrayTensor([f32(v+d) for v,d in zip(base.values,delta.values)],base.shape)
        else:total=None if base is None else (base+delta).to(base.dtype)
        reused=[]
        for hit in hits:
            self.valid(hit.entry);t=hit.entry.tensors[layer]['qkv'.index(role)]
            reused.append((hit.window.start,t));a['consumed'].add((hit.entry.key,hit.window.start,hit.window.end))
        out=scatter(total,fresh,reused,n,h,x)
        a['captures'].setdefault(layer,{})[role]=out.detach().clone()
        if role=='q':a['inputs'][layer]=x.detach().clone()
        a['records'].append(dict(layer=layer,role=role,fresh_positions=fresh,reused_positions=sorted(used),native_projection_rows=len(fresh),cpu_lora_rows=len(fresh),consumed=[dict(key=hit.entry.key,source_request_id=hit.entry.qkv_metadata['source_request_id'],source_positions=hit.entry.positions,target_positions=[hit.window.start,hit.window.end],tensor=tensor_proof(hit.entry.tensors[layer]['qkv'.index(role)])) for hit in hits],output=tensor_proof(out),mode=a['mode'],numerical_equivalence='NOT_ESTABLISHED',past_kv_grant=False))
        return out
    def complete(self,*,completed,attentions,valid_attention_mask=None):
        a=self.active;require(a is not None and completed is True,'complete real source required')
        layers=self.bridge.identity['layers'];require(len(a['records'])==3*layers and set(a['captures'])==set(range(layers)),'incomplete all-layer projections')
        require(attentions is not None and len(attentions)==layers,'actual all-layer attention required; no zero-impact fallback')
        metrics=self.bridge.metrics;values={};per_span={}
        for w in a['windows']:
            value=actual_attention_impact(attentions,w.start,w.end,valid_attention_mask);key=self.bridge.matcher.key(a['cluster'],w)
            values.setdefault(key,[]).append(value);per_span[w.start]=value
        metrics.history.append(a['cluster'],a['req']['request_id'],self.cache.now,{k:sum(v)/len(v) for k,v in values.items()})
        # Physical reuse frequency is committed only after all projections finish.
        for hit in a['selected']:
            if (hit.entry.key,hit.window.start,hit.window.end) in a['consumed']:
                metrics.reused(hit.entry,per_span[hit.window.start])
        def event(kind,entry,score):
            if kind=='EVICT':self.proofs.pop(entry.key,None)
            self.trace(dict(kind=kind,key=entry.key,score=score,component_scope='physical_total_qkv'))
        seen=set()
        for w in a['windows']:
            key=self.bridge.matcher.key(a['cluster'],w)
            if key in seen or key in self.cache.entries:continue
            seen.add(key);positions=list(range(w.start,w.end));size=len(positions)*3*layers*self.bridge.identity['hidden_size']*a['inputs'][0].element_size()
            meta=dict(component_scope='total_qkv',completed=True,provenance_mode=a['mode'],physical_identity=self.bridge.identity,source_user=a['req']['user_id'],source_adapter=a['req']['adapter_id'],source_request_id=a['req']['request_id'],source_token_ids=a['req']['token_ids'],input_proofs={str(i):tensor_proof(gather(a['inputs'][i],positions)) for i in range(layers)})
            candidate=CacheEntry(a['cluster'],w.token_ids,(w.start,w.end),size,qkv_metadata=meta,impact=per_span[w.start])
            def materialize():
                tensors={i:tuple(gather(a['captures'][i][r],positions) for r in 'qkv') for i in range(layers)}
                entry=CacheEntry.from_tensors(a['cluster'],w.token_ids,(w.start,w.end),tensors,storage_device=str(a['inputs'][0].device));entry.qkv_metadata=meta;entry.impact=candidate.impact
                return entry
            self.cache.insert(candidate,metrics.frequencies[key],metrics.frequencies,materialize=materialize,on_event=event)
            if key in self.cache.entries:
                e=self.cache.entries[key];self.proofs[key]=entry_proof(e)
        if self.cache.now%self.bridge.pbr_interval==0:metrics.pbr(a['cluster'])
        require(self.cache.logical_cache_bytes<=self.cache.capacity_bytes and self.cache.physical_tensor_bytes<=self.cache.capacity_bytes,'physical capacity exceeded')
        result=dict(mode=a['mode'],records=a['records'],physical_projection_rows_reused=sum(len(r['reused_positions']) for r in a['records']),physical_tensor_bytes=self.cache.physical_tensor_bytes,kv_prefix_reuse=False,correctness_qualified_speedup='NOT_ESTABLISHED')
        self.trace(dict(kind='PHYSICAL_SUBSEQUENCE_COMPLETE',**result));self.active=None;self.bridge.pending=None;return result
    def dispose(self):self.bridge.dispose();self.proofs.clear();self.active=None

@contextmanager
def projection_path(adapter,owner,session):
    """M mixed-projection algorithm adapted to bare R5 Linear and sparse UD.

    Only projection instance forwards change; all native model downstream ops
    see full-length outputs. Never combine with the Exact Prefix physical_path.
    """
    import torch
    originals=[]
    try:
        for layer in range(len(adapter.layers)):
            for role,module in adapter.projection_modules(layer).items():
                require(type(module) is torch.nn.Linear and 'forward' not in module.__dict__ and not module._forward_hooks and not module._forward_pre_hooks,'bare uninstrumented Linear required')
                originals.append(module);native=module.forward
                def forward(x,*args,_layer=layer,_role=role,_native=native,**kwargs):
                    require(not args and not kwargs,'unsupported projection arguments');return owner.project(_layer,_role,x,_native,session)
                module.forward=forward
        yield
        session.complete()
    finally:
        for m in originals:del m.forward

def prefill(model,adapter,owner,session,req,assignment,*,condition='SEMCACHE_SUBSEQUENCE',mode=EXACT,allow_approximate=False,max_attention_bytes=4*2**30):
    """Real opt-in native model prefill, not an autoregressive campaign runner.

    Caller must finish generation and then call owner.complete(completed=True,
    attentions=out.attentions). No source is admitted before that barrier.
    """
    import torch
    require(condition in ('SEMCACHE_SUBSEQUENCE','SEMANTIC_NO_REUSE'),'separate native FULL_RECOMPUTE runner required')
    require(not model.training and len(adapter.layers)==owner.bridge.identity['layers'],'eval/model architecture differs')
    require(all(type(l.self_attn).__name__=='OPTAttention' for l in adapter.layers),'actual eager OPTAttention required')
    n=len(req['token_ids']);heads=model.config.num_attention_heads
    require(0<max_attention_bytes<=4*2**30 and 4*len(adapter.layers)*heads*n*n<=max_attention_bytes,'bounded actual attention capture exceeds budget')
    owner.begin(req,assignment,mode=mode,allow_approximate=allow_approximate,reuse=condition=='SEMCACHE_SUBSEQUENCE')
    device=next(model.parameters()).device
    with torch.inference_mode(),projection_path(adapter,owner,session):
        out=model(input_ids=torch.tensor([req['token_ids']],device=device),attention_mask=torch.tensor([req['attention_mask']],device=device),use_cache=True,output_attentions=True)
    require(out.attentions is not None and len(out.attentions)==len(adapter.layers) and all(t is not None and tuple(t.shape)==(1,heads,n,n) for t in out.attentions),'native attention probabilities unavailable; admission forbidden')
    return out
