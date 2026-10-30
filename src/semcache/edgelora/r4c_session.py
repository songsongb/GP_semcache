"""C1 v1 suffix-prefill/EOS protocol over frozen bounded TCP framing."""
from .communication import digest, require
from .edge_server import ProjectionSession
from .r4c_workload import policy, termination

VERSION = 'R4C_OPT_PREFIX_V1'


def context(parent, step, tokens, target, past, p, mode, cohort, grant=None, *, context_limit=2048):
    grant=grant or {}; n=grant.get('prefix_length',0)
    c=dict(version=VERSION,parent_request_id=parent,pass_index=step,policy=p,mode=mode,cohort_sha256=cohort,
        phase='DECODE' if step else 'SUFFIX_PREFILL' if n else 'FULL_PREFILL',target_token_ids=list(target),target_length=len(target),
        prefix_length=n,suffix_length=len(target)-n,prefix_token_ids=list(target[:n]),source_entry_id=grant.get('entry_id'),
        source_request_id=grant.get('source_request_id'),proof_sha256=grant.get('proof_sha256'),input_token_ids=list(tokens),
        input_length=len(tokens),past_length=past,attention_mask=[1]*(past+len(tokens)),positions=list(range(past,past+len(tokens))),
        cache_positions=list(range(past,past+len(tokens))),context_limit=context_limit)
    c['target_attention_mask_sha256']=digest([1]*len(target)); c['attention_mask_sha256']=digest(c['attention_mask'])
    return validate_context(c)


def validate_context(c):
    expected={'version','parent_request_id','pass_index','policy','mode','cohort_sha256','phase','target_token_ids','target_length',
        'prefix_length','suffix_length','prefix_token_ids','source_entry_id','source_request_id','proof_sha256','input_token_ids',
        'input_length','past_length','attention_mask','positions','cache_positions','context_limit','target_attention_mask_sha256','attention_mask_sha256'}
    require(set(c)==expected and c['version']==VERSION,'unsupported C1 schema')
    require(isinstance(c['parent_request_id'],str) and c['parent_request_id'].isascii() and 1<=len(c['parent_request_id'])<=80,'invalid C1 parent')
    require(all(type(c[k]) is int for k in ('pass_index','target_length','prefix_length','suffix_length','input_length','past_length','context_limit')), 'invalid C1 integer fields')
    p=c['policy']; require(p==policy({k:p[k] for k in ('bos_token_id','eos_token_id','pad_token_id')},p['mode'],p['max_new_tokens']), 'invalid C1 greedy policy')
    step,n,pre,old=c['pass_index'],c['target_length'],c['prefix_length'],c['past_length']
    require(0<=step<p['max_new_tokens']<=128 and 0<n<=512 and n+p['max_new_tokens']<=c['context_limit']<=32768,'bounded C1 forward/context required')
    require(len(c['target_token_ids'])==n and c['input_length']==len(c['input_token_ids']) and
            all(type(t) is int and t>=0 for t in c['target_token_ids']+c['input_token_ids']), 'invalid C1 actual tokens')
    require(0<=pre<n and c['suffix_length']==n-pre and c['prefix_token_ids']==c['target_token_ids'][:pre], 'exact prefix boundary differs')
    require(c['mode'] in ('NO_REUSE','COLD_MISS','PARTIAL_PREFIX_HIT'), 'unsupported C1 mode')
    if c['mode']=='PARTIAL_PREFIX_HIT':
        require(pre>0 and all(isinstance(c[k],str) and c[k] for k in ('source_entry_id','source_request_id','proof_sha256')),'partial HIT needs completed source grant')
    else: require(pre==0 and all(c[k] is None for k in ('source_entry_id','source_request_id','proof_sha256')), 'unverified nonzero first past')
    if step==0:
        require(old==pre and c['input_token_ids']==c['target_token_ids'][pre:] and c['phase']==('SUFFIX_PREFILL' if pre else 'FULL_PREFILL'), 'hidden full-prefill recomputation / wrong suffix')
    else: require(old==n+step-1 and c['input_length']==1 and c['phase']=='DECODE', 'invalid C1 decode continuation')
    require(c['attention_mask']==[1]*(n+step) and c['positions']==c['cache_positions']==list(range(old,old+c['input_length'])), 'C1 mask/positions mismatch')
    require(c['attention_mask_sha256']==digest(c['attention_mask']) and c['target_attention_mask_sha256']==digest([1]*n), 'C1 mask digest differs')
    for k in ('cohort_sha256','source_entry_id','proof_sha256'):
        if c[k] is not None: require(len(c[k])==64 and all(x in '0123456789abcdef' for x in c[k]), 'invalid C1 identity digest')
    return c


class ForwardChannel:
    def __init__(self,channel,c): self.channel=channel; self.c=validate_context(c)
    def send(self,h,payload): return self.channel.send(dict(h,metadata=dict(r4c_forward=self.c)),payload)
    def recv(self,**expected):
        h,b,r=self.channel.recv(**expected); require(h['metadata']['r4c_forward_sha256']==digest(self.c),'C1 correction context mismatch'); return h,b,r


class DynamicSession(ProjectionSession):
    def __init__(self,channel,identity,fid,c,*,torch_backend):
        super().__init__(ForwardChannel(channel,c),dict(identity,length=c['input_length']),fid,torch_backend=torch_backend)


class ForwardOrder:
    def __init__(self,cohort,layers,context_limit):
        self.cohort=cohort; self.layers=layers; self.context_limit=context_limit; self.layer=0; self.current=None
        self.parents={}; self.finished={}; self.grants={}; self.seen=set(); self.completed=0

    def control(self,m):
        require(m.get('version')==VERSION and m.get('cohort_sha256')==self.cohort and self.layer==0,'invalid C1 control')
        action=m.get('action'); base={'version','cohort_sha256','action'}
        if action=='COMPLETE_REQUEST':
            require(set(m)==base|{'parent_request_id','generated_token_ids','termination'},'C1 completion schema differs')
            rid=m['parent_request_id']; c=self.parents.get(rid); ids=m['generated_token_ids']
            require(c is not None and rid==self.current and rid not in self.finished and len(ids)==c['pass_index']+1,'premature/duplicate C1 completion')
            require(all(type(t) is int and t>=0 for t in ids) and all(self.parents[rid]['consumed_generated'][i]==t for i,t in enumerate(ids[:-1])), 'completion disagrees with decode inputs')
            require(termination(c['policy'],ids)==m['termination'] and m['termination'] is not None,'invalid EOS/length completion')
            if c['policy']['stop_at_eos']: require(c['policy']['eos_token_id'] not in ids[:-1], 'generation continued past EOS')
            self.finished[rid]=dict(m); return
        require(action in ('REGISTER_PREFIX','EVICT_PREFIX'),'unknown C1 control')
        key=m['entry_id']; require(isinstance(key,str) and len(key)==64,'invalid cache entry ID')
        if action=='EVICT_PREFIX':
            require(set(m)==base|{'entry_id'} and key in self.grants,'unknown C1 eviction'); del self.grants[key]; return
        require(set(m)==base|{'entry_id','source_request_id','prefix_length','prefix_token_ids','proof_sha256'},'C1 registration schema differs')
        source=self.parents.get(m['source_request_id'])
        require(source is not None and m['source_request_id'] in self.finished and source['mode']=='COLD_MISS' and
            source['target_token_ids']==m['prefix_token_ids'] and len(m['prefix_token_ids'])==m['prefix_length'] and 0<m['prefix_length']<512,
            'source not a completed standalone prefix')
        require(key not in self.grants and len(self.grants)<64 and len(m['proof_sha256'])==64,'duplicate/excess/corrupt prefix grant')
        self.grants[key]=dict(m)

    def accept(self,h,identity):
        c=validate_context(h['metadata']['r4c_forward']); shape=[1,c['input_length'],identity['hidden_size']]
        require(c['cohort_sha256']==self.cohort and c['context_limit']==self.context_limit and h['kind']=='HIDDEN' and
            h['layer_id']==self.layer and h['shape']==shape and h['dtype']==identity['hidden_dtype'],'C1 layer/shape/dtype/identity differs')
        if c['mode']=='PARTIAL_PREFIX_HIT':
            g=self.grants.get(c['source_entry_id']); require(g is not None and all(c[k]==g[k] for k in ('source_request_id','prefix_length','prefix_token_ids','proof_sha256')), 'invalid/evicted C1 HIT')
        if self.layer==0:
            parent=c['parent_request_id']; previous=self.parents.get(parent)
            require(h['request_id']==f'{parent}.s{c["pass_index"]:02d}' and h['request_id'] not in self.seen and len(self.seen)<16384,'duplicate/excess C1 forward')
            if previous is None:
                require(c['pass_index']==0 and len(self.parents)<128 and (self.current is None or self.current in self.finished),'previous C1 request not completed')
                consumed=[]
            else:
                require(parent==self.current and parent not in self.finished and c['pass_index']==previous['pass_index']+1 and all(c[k]==previous[k] for k in
                    ('target_token_ids','mode','prefix_length','suffix_length','source_entry_id','source_request_id','proof_sha256','policy','cohort_sha256')), 'C1 continuation changed source/policy/history')
                consumed=previous['consumed_generated']+c['input_token_ids']
                require(not c['policy']['stop_at_eos'] or c['policy']['eos_token_id'] not in c['input_token_ids'],'decode after EOS')
            self.current=parent; self.parents[parent]=dict(c,consumed_generated=consumed); self.forward=h['request_id']; self.c=dict(c); self.seen.add(self.forward)
        require(h['request_id']==self.forward and c==self.c,'context changed within layer sequence'); return c,shape

    def advance(self):
        self.layer=(self.layer+1)%self.layers
        if not self.layer: self.completed+=1

    def finish(self):
        require(self.layer==0 and set(self.parents)==set(self.finished),'shutdown with incomplete C1 request/forward')
