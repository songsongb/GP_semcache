"""I5R-4 opt-in causal policy. No future trace, tensor arithmetic or safety grants."""
from collections import OrderedDict
from copy import deepcopy
import math
import time

from .communication import digest,require
from .r5_i5_cache import I5Owner
from .r5_i5_controller import I5Controller
from .r4b1_cache import MultiPromptCache
from .r5_i5_contracts import validate_request
from .r5_multi_artifacts import resolve

POLICIES=('P0','P1','P2','P3','P4','P5')
CAPACITY=512*2**20
MAX_ENTRIES=64

def parameters(saving_ms,admission_ms,**overrides):
    p=dict(expected_reuses=2.,saving_ms=float(saving_ms),admission_ms=float(admission_ms),
           retention_ms_per_mib_request=1.,ttl_min=2,ttl_max=8,ttl_idle_requests=4,
           exploration_every=3,max_entries=MAX_ENTRIES)
    p.update(overrides)
    require(set(p)=={'expected_reuses','saving_ms','admission_ms','retention_ms_per_mib_request',
                    'ttl_min','ttl_max','ttl_idle_requests','exploration_every','max_entries'},'unknown policy parameter')
    require(all(type(p[k]) in (float,int) and math.isfinite(p[k]) for k in p),'invalid policy parameter')
    require(p['saving_ms']>=0 and p['admission_ms']>=0 and p['expected_reuses']>=0 and p['retention_ms_per_mib_request']>0,'negative/zero policy cost')
    require(all(type(p[k]) is int for k in ('ttl_min','ttl_max','ttl_idle_requests','exploration_every','max_entries')),'integer policy bounds required')
    require(1<=p['ttl_min']<=p['ttl_idle_requests']<=p['ttl_max']<=16 and 1<=p['exploration_every']<=16 and p['max_entries']==64,'policy bounds differ')
    return p

class Engine:
    """Only observed requests, actual admissions/hits and pinned priors enter state.

    Entry history is a prefix-token digest, never a dataset/context future-use label.
    Request index is per adapter; intervening requests in other namespaces do not
    age this namespace. Wall time is logged separately, never a policy operand.
    """
    def __init__(self,policy,params,capacity=CAPACITY):
        require(policy in POLICIES and 0<capacity<=2**30,'invalid policy/physical capacity')
        self.policy=policy;self.params=parameters(params['saving_ms'],params['admission_ms'],**{k:v for k,v in params.items() if k not in ('saving_ms','admission_ms')})
        self.capacity=capacity;self.entries=OrderedDict();self.history={};self.index=-1;self.seen=set();self.candidates=0
    def snapshot(self):
        return dict(index=self.index,candidates=self.candidates,entries=deepcopy(dict(self.entries)),history=deepcopy(self.history))
    def begin(self,rid):
        require(rid not in self.seen,'duplicate policy request');self.seen.add(rid);self.index+=1
        return [k for k,e in self.entries.items() if self.policy in ('P2','P4') and self.index-e['last']>self.ttl(e)]
    def expected(self,token_hash):
        # Optimistic cold start is a declared prior, not a fitted probability.
        return self.params['expected_reuses']+min(2,self.history.get(token_hash,0))*.5
    def value(self,token_hash,size):
        p=self.params;expected=self.expected(token_hash)
        retention=p['retention_ms_per_mib_request']*(size/2**20)*p['ttl_idle_requests']
        return dict(expected_reuses=expected,predicted_per_reuse_saving_ms=p['saving_ms'],
                    incremental_admission_ms=p['admission_ms'],estimated_retention_cost_ms=retention,
                    admission_net_ms=expected*p['saving_ms']-p['admission_ms']-retention,
                    future_value_per_byte=(expected*p['saving_ms']-retention)/size)
    def ttl(self,e):
        p=self.params;cost=p['retention_ms_per_mib_request']*(e['bytes']/2**20)
        economic=max(p['ttl_min'],min(p['ttl_max'],int(self.expected(e['token_hash'])*p['saving_ms']/cost)))
        return min(economic,min(p['ttl_max'],p['ttl_idle_requests']+e['hits']))
    def admission(self,token_hash,size):
        require(size>0,'empty cache candidate');self.candidates+=1
        v=self.value(token_hash,size);fits=size<=self.capacity
        explore=(self.candidates-1)%self.params['exploration_every']==0
        allowed=fits and (self.policy not in ('P1','P4') or v['admission_net_ms']>0 or explore)
        return dict(allowed=allowed,exploration=explore and fits and v['admission_net_ms']<=0,
                    reason='CAPACITY' if not fits else 'POSITIVE_VALUE_OR_EXPLORATION' if allowed else 'NONPOSITIVE_EXPECTED_VALUE',**v)
    def victims(self,size,replacing=None):
        remaining=[k for k in self.entries if k!=replacing];victims=[]
        if self.policy in ('P3','P4'):
            remaining.sort(key=lambda k:(self.value(self.entries[k]['token_hash'],self.entries[k]['bytes'])['future_value_per_byte'],self.entries[k]['last'],k))
        used=sum(self.entries[k]['bytes'] for k in remaining)
        while remaining and (used+size>self.capacity or len(remaining)>=self.params['max_entries']):
            k=remaining.pop(0);victims.append(k);used-=self.entries[k]['bytes']
        require(used+size<=self.capacity and len(remaining)<64,'candidate cannot fit physical bounds')
        return victims
    def remove(self,key):self.entries.pop(key,None)
    def hit(self,key):
        require(key in self.entries,'false/stale policy HIT')
        e=self.entries[key];e['last']=self.index;e['hits']+=1;self.history[e['token_hash']]=self.history.get(e['token_hash'],0)+1
        self.entries.move_to_end(key)
    def admitted(self,key,token_hash,size):
        require(key not in self.entries,'duplicate policy insertion')
        self.entries[key]=dict(token_hash=token_hash,bytes=size,last=self.index,born=self.index,hits=0)
        require(sum(e['bytes'] for e in self.entries.values())<=self.capacity and len(self.entries)<=64,'policy physical bounds exceeded')

class PolicyOwner(I5Owner):
    """P1..P4 remove victims through I5's proof/index/UD invalidation chain.

    I5Owner.admit validates ALL actual tensor proofs before calling bind().
    At that boundary we free capacity before the lower-level insert can evict.
    P0 never instantiates this class.
    """
    def bind(self,prompt):
        view=super().bind(prompt)
        if getattr(self,'_policy_admitting',False):
            key=view.context_digest;size=len(prompt['token_ids'])*3*self._pm['identity']['layers']*self._pm['identity']['hidden_size']*(4 if self.fixture else 2)
            if key in self.entries:self.remove(key,'INVALIDATE','explicit cold source replacement')
            for victim in self.i4_engine.victims(size):self.remove(victim,'EVICT','I4_PRESELECTED_'+self.i4_engine.policy)
        return view
    def remove(self,key,kind,reason):
        super().remove(key,kind,reason)
        if hasattr(self,'i4_engine'):self.i4_engine.remove(key)
        require(key not in self.entries and key not in self.index.entries and key not in self.proofs and key not in self.proof_seals,'incomplete I4 invalidation')
    def insert(self,context,blocks,request_id,*,completed):
        first=len(self.events)
        result=MultiPromptCache.insert(self,context,blocks,request_id,completed=completed)
        require(not any(e['kind']=='EVICT' for e in self.events[first:]),'hidden lower-level eviction forbidden')
        return result
    def admit(self,*args,**kwargs):
        self._policy_admitting=True
        try:return super().admit(*args,**kwargs)
        finally:self._policy_admitting=False

class PolicyController(I5Controller):
    def __init__(self,*args,policy,params,receipt,**kwargs):
        super().__init__(*args,**kwargs);self.engine=Engine(policy,params,self.owner.capacity_bytes)
        require(self.condition==('FULL_RECOMPUTE' if policy=='P5' else 'SEMCACHE_INTEGRATED'),'I4 policy condition mismatch')
        self.owner.i4_engine=self.engine;self.receipt=receipt
    def log(self,kind,**fields):
        self.receipt(dict(kind=kind,policy=self.engine.policy,adapter_id=self.identity['adapter_id'],request_id=self.owner.request_id,
                          state=self.engine.snapshot(),**fields))
    def sync(self):
        for key in list(self.engine.entries):
            if key not in self.owner.entries:self.engine.remove(key)
        require(set(self.engine.entries)==set(self.owner.entries),'physical/policy state mismatch')
    def decide(self,req):
        # Warmup is the unchanged original FULL_RECOMPUTE path, no policy aging.
        if req['request_id'].startswith('W_'):return super().decide(req)
        validate_request(req)
        require(resolve(self.registry,req['user_id'],req['adapter_id'])==self.identity['adapter_id'] and req['physical_identity']==self.identity,'UNAUTHORIZED_USER_OR_ADAPTER')
        self.owner.request_id=req['request_id'];start=time.monotonic_ns();expired=self.engine.begin(req['request_id'])
        self.log('BEGIN',expired=expired,engine_compute_ms=(time.monotonic_ns()-start)/1e6)
        for key in expired:self.owner.remove(key,'EVICT','I4_IDLE_TTL_EXPIRED')
        self.sync();self.log('RETENTION_DONE',expired=expired,incremental_policy_ms=(time.monotonic_ns()-start)/1e6)
        event,found=super().decide(req)
        tick=time.monotonic_ns()
        if found:self.engine.hit(found[1]['entry_id'])
        self.log('LOOKUP_DONE',selected=None if found is None else found[1]['entry_id'],engine_compute_ms=(time.monotonic_ns()-tick)/1e6)
        return event,found
    def admit(self,req,assignment,blocks,proof,*,completed):
        if req['request_id'].startswith('W_'):return super().admit(req,assignment,blocks,proof,completed=completed)
        require(completed is True,'uncompleted source');start=time.monotonic_ns();decision=None
        candidate=self.reuse and self.virtual_selected is None and len(req['token_ids'])<2048
        size=len(req['token_ids'])*3*self.identity['layers']*self.identity['hidden_size']*(4 if req['TEST_FIXTURE_ONLY'] else 2)
        if candidate:
            decision_start=time.monotonic_ns()
            decision=self.engine.admission(digest(req['token_ids']),size)
            self.log('ADMISSION_DECISION',decision=decision,candidate_bytes=size,token_hash=digest(req['token_ids']),decision_compute_ms=(time.monotonic_ns()-decision_start)/1e6)
            if self.engine.policy in ('P1','P4') and not decision['allowed']:
                self.owner.event('ADMISSION_DENIED',reason='I4_'+decision['reason']);grant=None
            else:grant=super().admit(req,assignment,blocks,proof,completed=completed)
        else:grant=super().admit(req,assignment,blocks,proof,completed=completed)
        # P0 evictions are exactly the original owner operations, merely observed.
        for key in list(self.engine.entries):
            if key not in self.owner.entries:self.engine.remove(key)
        if grant:
            self.engine.remove(grant['entry_id']) # P0 targeted replacement keeps its original owner untouched.
            self.engine.admitted(grant['entry_id'],digest(req['token_ids']),size)
        self.sync();self.log('COMPLETE',admitted=None if grant is None else grant['entry_id'],incremental_policy_ms=(time.monotonic_ns()-start)/1e6,
                            timing_scope='Includes original admission; policy decision overhead separately timed in receipts')
        return grant

def factories(policy,params,receipt):
    require(policy in POLICIES,'unknown policy')
    def controller(*args,**kwargs):return PolicyController(*args,policy=policy,params=params,receipt=receipt,**kwargs)
    return (I5Owner if policy in ('P0','P5') else PolicyOwner),controller
