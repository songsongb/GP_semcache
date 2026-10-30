"""B2 v2 semantic protocol over unchanged R3-A bounded TCP framing."""
from .communication import digest,require
from .edge_server import ProjectionSession

MODES=('NO_REUSE','COLD_MISS','PARTIAL_PREFIX_HIT')
PREFIX_LENGTHS=(32,64,96)


def context(parent,step,tokens,target,past,total,mode,cohort,grant=None):
    grant=grant or {};p=grant.get('prefix_length',0)
    c=dict(b2_version=2,parent_request_id=parent,pass_index=step,total_passes=total,mode=mode,cohort_sha256=cohort,
        phase='DECODE' if step else 'SUFFIX_PREFILL' if p else 'FULL_PREFILL',target_token_ids=list(target),target_length=len(target),
        prefix_length=p,suffix_length=len(target)-p,prefix_token_ids=list(target[:p]),
        source_entry_id=grant.get('entry_id'),source_request_id=grant.get('source_request_id'),proof_sha256=grant.get('proof_sha256'),
        input_token_ids=list(tokens),input_length=len(tokens),past_length=past,attention_mask=[1]*(past+len(tokens)),
        positions=list(range(past,past+len(tokens))),cache_positions=list(range(past,past+len(tokens))))
    c['attention_mask_sha256']=digest(c['attention_mask']);c['target_attention_mask_sha256']=digest([1]*len(target))
    validate_context(c);return c


def validate_context(c):
    fields={'b2_version','parent_request_id','pass_index','total_passes','mode','cohort_sha256','phase','target_token_ids','target_length',
        'prefix_length','suffix_length','prefix_token_ids','source_entry_id','source_request_id','proof_sha256','input_token_ids',
        'input_length','past_length','attention_mask','positions','cache_positions','attention_mask_sha256','target_attention_mask_sha256'}
    require(set(c)==fields and type(c['b2_version']) is int and c['b2_version']==2,'unsupported B2 execution schema')
    require(isinstance(c['parent_request_id'],str) and c['parent_request_id'].isascii() and 1<=len(c['parent_request_id'])<=80,'bad B2 parent')
    require(c['mode'] in MODES and all(type(c[k]) is int for k in ('pass_index','total_passes','target_length','prefix_length','suffix_length','input_length','past_length')),
        'B2 mode/size types invalid')
    step,total,n,p,old=c['pass_index'],c['total_passes'],c['target_length'],c['prefix_length'],c['past_length']
    require(0<=step<total<=16 and 1<=n<=128 and c['input_length']==len(c['input_token_ids']) and len(c['target_token_ids'])==n,
        'bounded B2 forward required')
    require(all(type(t) is int and t>=0 for t in c['target_token_ids']+c['input_token_ids']),'invalid B2 token IDs')
    require(c['suffix_length']==n-p and 0<=p<n and c['prefix_token_ids']==c['target_token_ids'][:p],'prefix boundary/token mismatch')
    if c['mode']=='PARTIAL_PREFIX_HIT':
        require(p in PREFIX_LENGTHS and all(isinstance(c[k],str) and c[k] for k in ('source_entry_id','source_request_id','proof_sha256')),
            'partial HIT needs an explicit verified prefix grant')
    else:require(p==0 and all(c[k] is None for k in ('source_entry_id','source_request_id','proof_sha256')),'unverified past in full prefill')
    if step==0:
        require(old==p and c['input_token_ids']==c['target_token_ids'][p:] and
            c['phase']==('SUFFIX_PREFILL' if p else 'FULL_PREFILL'),'first forward must consume only the declared suffix/full prompt')
    else:require(old==n+step-1 and c['input_length']==1 and c['phase']=='DECODE','B2 Prefill-to-Decode transition invalid')
    require(c['attention_mask']==[1]*(old+c['input_length']) and len(c['attention_mask'])==n+step and
        c['positions']==c['cache_positions']==list(range(old,old+c['input_length'])),'B2 mask/position/cache slots differ')
    require(c['attention_mask_sha256']==digest(c['attention_mask']) and c['target_attention_mask_sha256']==digest([1]*n),'B2 mask digest differs')
    for k in ('cohort_sha256','proof_sha256','source_entry_id'):
        if c[k] is not None:require(len(c[k])==64 and all(x in '0123456789abcdef' for x in c[k]),'invalid B2 identity digest')
    return c


class ForwardChannel:
    def __init__(self,channel,c):self.channel=channel;self.c=validate_context(c)
    def send(self,h,payload):return self.channel.send(dict(h,metadata=dict(r4b2_forward=self.c)),payload)
    def recv(self,**expected):
        h,data,record=self.channel.recv(**expected)
        require(h['metadata']['r4b2_forward_sha256']==digest(self.c),'B2 correction belongs to another forward')
        return h,data,record


class DynamicSession(ProjectionSession):
    def __init__(self,channel,identity,fid,c,*,torch_backend):
        super().__init__(ForwardChannel(channel,c),dict(identity,length=c['input_length']),fid,torch_backend=torch_backend)


class ForwardOrder:
    def __init__(self,cohort,layers):
        self.cohort=cohort;self.layers=layers;self.layer=0;self.current=None;self.parents={};self.seen=set();self.grants={};self.completed=0
    def control(self,m):
        require(type(m.get('b2_version')) is int and m.get('b2_version')==2 and m.get('cohort_sha256')==self.cohort and self.layer==0,'invalid B2 cache control')
        fields={'action','b2_version','cohort_sha256','entry_id'}
        if m.get('action')=='REGISTER_PREFIX':fields|={'source_request_id','prefix_length','prefix_token_ids','proof_sha256'}
        require(set(m)==fields,'B2 prefix-control metadata fields differ')
        key=m['entry_id'];require(isinstance(key,str) and len(key)==64 and all(c in '0123456789abcdef' for c in key),'invalid prefix registration identity')
        if m['action']=='EVICT_PREFIX':
            require(key in self.grants,'evicting unknown UD prefix grant');del self.grants[key];return
        require(m['action']=='REGISTER_PREFIX' and key not in self.grants and len(self.grants)<64,'invalid/duplicate prefix registration')
        source=self.parents.get(m['source_request_id'])
        require(source is not None and source['mode']=='COLD_MISS' and source['pass_index']==source['total_passes']-1 and
            source['target_token_ids']==m['prefix_token_ids'] and len(m['prefix_token_ids'])==m['prefix_length'] in PREFIX_LENGTHS,
            'prefix source has not completed a real standalone cold generation')
        require(isinstance(m['proof_sha256'],str) and len(m['proof_sha256'])==64 and all(c in '0123456789abcdef' for c in m['proof_sha256']),
            'missing Raw-KV / actual HF cache proof')
        self.grants[key]=dict(m)
    def accept(self,h,identity):
        c=validate_context(h['metadata']['r4b2_forward']);shape=[1,c['input_length'],identity['hidden_size']]
        require(c['cohort_sha256']==self.cohort and h['kind']=='HIDDEN' and h['layer_id']==self.layer and
            h['shape']==shape and h['dtype']==identity['hidden_dtype'],'B2 layer/cohort/dtype/shape mismatch')
        if c['mode']=='PARTIAL_PREFIX_HIT':
            g=self.grants.get(c['source_entry_id']);require(g is not None and all(c[k]==g[k] for k in
                ('source_request_id','proof_sha256','prefix_length','prefix_token_ids')),'unknown/evicted/incompatible prefix grant')
        if self.layer==0:
            require(h['request_id']==f'{c["parent_request_id"]}.s{c["pass_index"]:02d}' and h['request_id'] not in self.seen and len(self.seen)<1024,
                'duplicate or excess B2 forward')
            parent=c['parent_request_id'];previous=self.parents.get(parent)
            if previous is None:
                require(c['pass_index']==0 and len(self.parents)<128,'B2 parent must start with a verified prefill')
                if self.current is not None:
                    old=self.parents[self.current];require(old['pass_index']==old['total_passes']-1,'prior B2 generation incomplete')
            else:
                require(parent==self.current and c['pass_index']==previous['pass_index']+1 and all(c[k]==previous[k] for k in
                    ('mode','cohort_sha256','total_passes','target_token_ids','prefix_length','suffix_length','source_entry_id','source_request_id','proof_sha256')),
                    'B2 decode changed parent/source/policy/history')
            self.current=parent;self.parents[parent]=dict(c);self.forward=h['request_id'];self.c=dict(c);self.seen.add(self.forward)
        require(h['request_id']==self.forward and c==self.c,'B2 context changed within layer sequence')
        return c,shape
    def advance(self):
        self.layer=(self.layer+1)%self.layers
        if not self.layer:self.completed+=1
    def finish(self):
        require(self.layer==0 and (self.current is None or self.parents[self.current]['pass_index']==self.parents[self.current]['total_passes']-1),
            'B2 shutdown with incomplete suffix/decode')
