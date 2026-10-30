"""Dynamic R4 metadata adapter over the unchanged R3-A framed Channel/session."""
from .communication import digest,require
from .edge_server import ProjectionSession

MAX_PROMPT=128
MAX_NEW=16


def positions(mask):
    require(mask and all(type(m) is int and m in (0,1) for m in mask) and mask[-1]==1,'valid left-padded attention mask required')
    seen=0;out=[]
    for m in mask:
        seen+=m;out.append(seen-1 if m else -1)
    require(mask==sorted(mask),'only left padding supported');return out


def context(parent,step,tokens,mask,past_length,total_passes,*,hit=False,triage=False,natural=False):
    require(type(step) is int and 0<=step<total_passes<=MAX_NEW,'bounded forward index required')
    require(tokens and all(type(t) is int and t>=0 for t in tokens),'valid input IDs required')
    require(len(mask)==past_length+len(tokens)<=MAX_PROMPT+MAX_NEW,'attention/past/input length mismatch')
    require((step==0 and past_length==0 and 1<=len(tokens)<=MAX_PROMPT) or
        (step>0 and len(tokens)==1 and past_length>=1),'decode must consume one token with a real past')
    value=dict(r4a_version=1,parent_request_id=parent,pass_index=step,total_passes=total_passes,
        input_length=len(tokens),past_length=past_length,input_ids_sha256=digest(tokens),attention_mask_sha256=digest(mask),
        positions_sha256=digest(positions(mask)[past_length:]),prefill_reused=hit,triage=triage,natural_eos=natural)
    validate_context(value);return value


def validate_context(c):
    require(set(c)=={'r4a_version','parent_request_id','pass_index','total_passes','input_length','past_length',
        'input_ids_sha256','attention_mask_sha256','positions_sha256','prefill_reused','triage','natural_eos'},'R4 forward metadata fields differ')
    require(c['r4a_version']==1 and isinstance(c['parent_request_id'],str) and c['parent_request_id'].isascii() and
        1<=len(c['parent_request_id'])<=80,'unsupported R4 application identity')
    require(all(type(c[k]) is int for k in ('pass_index','total_passes','input_length','past_length')),'integer forward sizes required')
    step=c['pass_index'];n=c['input_length'];past=c['past_length']
    require(0<=step<c['total_passes']<=MAX_NEW and 0<=past and n+past<=MAX_PROMPT+MAX_NEW,'bounded forward sizes required')
    require((step==0 and past==0 and 1<=n<=MAX_PROMPT) or (step>0 and n==1 and past>=step),'invalid dynamic prefill/decode shape')
    require(all(type(c[k]) is bool for k in ('prefill_reused','triage','natural_eos')),'boolean forward policy required')
    require(not c['triage'] or step==0 and c['total_passes']==1 and not c['prefill_reused'],'triage is one recompute prefill')
    for k in ('input_ids_sha256','attention_mask_sha256','positions_sha256'):
        require(isinstance(c[k],str) and len(c[k])==64 and all(x in '0123456789abcdef' for x in c[k]),'bad forward identity hash')
    return c


class ForwardChannel:
    def __init__(self,channel,metadata):self.channel=channel;self.metadata=validate_context(metadata)
    def send(self,h,payload):
        require(h['kind']=='HIDDEN','dynamic projection only sends HIDDEN')
        return self.channel.send(dict(h,metadata=dict(r4a_forward=self.metadata)),payload)
    def recv(self,**expected):
        h,data,record=self.channel.recv(**expected)
        require(h['metadata']['r4a_forward_sha256']==digest(self.metadata),'returned delta belongs to another forward')
        return h,data,record


class DynamicSession(ProjectionSession):
    def __init__(self,channel,identity,forward_id,metadata,*,torch_backend):
        self.metadata=validate_context(metadata)
        super().__init__(ForwardChannel(channel,metadata),dict(identity,length=metadata['input_length']),forward_id,torch_backend=torch_backend)
