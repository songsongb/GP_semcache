"""Separate sparse projection TCP V1. No changes to I5 prefix protocols."""
from copy import deepcopy
import time
from .communication import require,digest,digest_bytes,header
from .r3a_artifacts import pack,unpack,tensor_bytes,tensor_from_bytes
from .r3b_fixture import ArrayTensor
from .r5_i5_contracts import validate_request
from .r5_multi_artifacts import resolve
from .user_device import cpu_deltas

VERSION='R5_SPARSE_TOTAL_QKV_TCP_V1'

def blob(t):return t.fixture_bytes() if isinstance(t,ArrayTensor) else tensor_bytes(t)
def gather(t,positions):
    require(len(t.shape)==3 and t.shape[0]==1 and positions==sorted(set(positions)) and all(0<=p<t.shape[1] for p in positions),'invalid gather positions')
    if isinstance(t,ArrayTensor):return ArrayTensor([v for p in positions for v in t.values[p*t.shape[2]:(p+1)*t.shape[2]]],[1,len(positions),t.shape[2]])
    import torch
    return t.index_select(1,torch.tensor(positions,device=t.device,dtype=torch.long))

def scatter(fresh,positions,reused,n,h,like):
    """Owned output; input rows must cover the target exactly once."""
    used=set(positions);require(len(used)==len(positions),'duplicate fresh positions')
    for start,t in reused:
        rows=set(range(start,start+t.shape[1]));require(not used&rows and min(rows)>=0 and max(rows)<n,'overlap/outside scatter');used|=rows
    require(used==set(range(n)),'incomplete scatter coverage')
    if isinstance(like,ArrayTensor):
        vals=[None]*(n*h)
        if fresh is not None:
            require(fresh.shape==(1,len(positions),h),'fresh shape differs')
            for j,p in enumerate(positions):vals[p*h:(p+1)*h]=fresh.values[j*h:(j+1)*h]
        for start,t in reused:
            require(t.shape[2]==h,'cached hidden size differs');vals[start*h:(start+t.shape[1])*h]=t.values
        return ArrayTensor(vals,[1,n,h])
    import torch
    out=torch.empty((1,n,h),device=like.device,dtype=like.dtype)
    if fresh is not None:out[:,positions,:]=fresh
    for start,t in reused:out[:,start:start+t.shape[1],:]=t.to(out)
    return out

def envelope(req,layer,positions,pass_id):
    validate_request(req);n=len(req['token_ids'])
    require(type(layer) is int and 0<=layer<req['physical_identity']['layers'] and positions==sorted(set(positions)) and all(type(p) is int and 0<=p<n for p in positions),'invalid sparse coordinates')
    require(isinstance(pass_id,str) and pass_id,'unique prefill pass required')
    return dict(version=VERSION,request=req,pass_id=pass_id,layer=layer,fresh_positions=positions,roles=list('qkv'),
                full_length=n,absolute_positions=[req['positions'][p] for p in positions],mask_sha256=digest(req['attention_mask']))

class SparseOrder:
    def __init__(self,identity,registry):self.identity=identity;self.registry=registry;self.current=None;self.layer=0;self.seen=set()
    def accept(self,c):
        require(set(c)=={'version','request','pass_id','layer','fresh_positions','roles','full_length','absolute_positions','mask_sha256'},'sparse schema differs')
        req=c['request'];require(c==envelope(req,c['layer'],c['fresh_positions'],c['pass_id']),'sparse context differs')
        require(req['physical_identity']==self.identity and resolve(self.registry,req['user_id'],req['adapter_id'])==self.identity['adapter_id'],'sparse physical/authorization mismatch')
        if self.layer==0:
            require(c['pass_id'] not in self.seen,'duplicate sparse pass');self.seen.add(c['pass_id']);self.current=(c['pass_id'],deepcopy(req))
        require(self.current==(c['pass_id'],req) and c['layer']==self.layer,'sparse pass/layer order differs')
        self.layer=(self.layer+1)%self.identity['layers']
        if not self.layer:self.current=None

def serve_sparse(channel,identity,registry,matrices,*,fixture=False,trace=lambda r:None):
    """CPU UD endpoint. Matrices must be loaded/verified by existing multi.load.

    Production caller must run in a separate configure_cpu()-isolated process.
    Channel verifies session/matrix/payload hashes and unchanged MAX_HEADER.
    """
    if not fixture:
        import torch,sys
        require(not torch.cuda.is_initialized() and all(t.device.type=='cpu' for t in matrices.values()) and 'transformers' not in sys.modules,'CPU-only UD required')
    order=SparseOrder(identity,registry)
    while True:
        h,b,r=channel.recv()
        if h['kind']=='STOP':
            require(order.layer==0 and not b and h['request_id']=='shutdown' and h['layer_id']==-1 and h['shape']==[],'invalid stop/unfinished sparse pass');channel.send(header('BYE','shutdown',-1));return
        require(set(h['metadata'])=={'sparse'} and h['kind'] in ('HIDDEN','HELLO'),'sparse projection frame required')
        c=h['metadata']['sparse'];order.accept(c);n=len(c['fresh_positions']);width=identity['hidden_size']
        require(h['request_id']==c['pass_id'] and h['layer_id']==(c['layer'] if n else -1),'frame/context differs')
        if not n:
            require(h['kind']=='HELLO' and not b and h['shape']==[],'empty projection must have no tensor')
            channel.send(header('READY',c['pass_id'],-1,sparse_sha256=digest(c),cpu_projection_rows=0));continue
        require(h['kind']=='HIDDEN' and h['shape']==[1,n,width] and h['dtype']==identity['dtype'],'sparse tensor shape/dtype differs')
        x=unpack(b,h['dtype']) if fixture else tensor_from_bytes(b,h['dtype'],h['shape']);start=time.monotonic_ns()
        delta=cpu_deltas(x,matrices,c['layer'],width,torch_backend=not fixture)
        data=pack(delta,'float32') if fixture else tensor_bytes(delta)
        record=dict(sparse_sha256=digest(c),hidden_sha256=digest_bytes(b),cpu_projection_rows=3*n,cpu_lora_ms=(time.monotonic_ns()-start)/1e6,TEST_FIXTURE_ONLY=fixture)
        channel.send(header('DELTAS',c['pass_id'],c['layer'],'float32',[3,1,n,width],**record),data);trace(dict(record,context=c))

class SparseSession:
    def __init__(self,channel,req,pass_id,*,fixture=False):
        self.channel=channel;self.req=validate_request(req);self.pass_id=pass_id;self.fixture=fixture;self.layer=0;self.role=0;self.bundle=None;self.receipts=[]
    def correction(self,layer,role,x,positions):
        require(layer==self.layer and role=='qkv'[self.role],'sparse role/layer order differs')
        require(list(x.shape)==[1,len(self.req['token_ids']),self.req['physical_identity']['hidden_size']],'full projection input shape differs')
        c=envelope(self.req,layer,positions,self.pass_id);n=len(positions);h=x.shape[2]
        if self.role==0:
            self.context=c;self.input_sha=digest_bytes(blob(x));self.bundle=None
            if n:
                data=blob(gather(x,positions));dtype=self.req['physical_identity']['dtype']
                sent=self.channel.send(header('HIDDEN',self.pass_id,layer,dtype,[1,n,h],sparse=c),data)
                reply,b,received=self.channel.recv(kind='DELTAS',request_id=self.pass_id,layer_id=layer)
                require(reply['metadata']['sparse_sha256']==digest(c) and reply['metadata']['hidden_sha256']==digest_bytes(data) and reply['metadata']['cpu_projection_rows']==3*n and reply['shape']==[3,1,n,h] and reply['dtype']=='float32','sparse correction receipt differs')
                self.bundle=unpack(b,'float32') if self.fixture else tensor_from_bytes(b,'float32',[3,1,n,h]).to(x.device)
            else:
                sent=self.channel.send(header('HELLO',self.pass_id,-1,sparse=c));reply,b,received=self.channel.recv(kind='READY',request_id=self.pass_id,layer_id=-1)
                require(not b and reply['metadata']==dict(sparse_sha256=digest(c),cpu_projection_rows=0),'empty projection receipt differs')
            self.receipts.append(dict(context=c,input_sha256=self.input_sha,sent=sent,received=received,hidden_wire_bytes=sent['wire_bytes'],delta_wire_bytes=received['wire_bytes'],fresh_rows_per_role=n,skipped_rows_per_role=len(self.req['token_ids'])-n))
        else:require(c==self.context and digest_bytes(blob(x))==self.input_sha,'Q/K/V inputs or positions differ')
        result=None if not n else ArrayTensor(self.bundle[self.role*n*h:(self.role+1)*n*h],[1,n,h]) if self.fixture else self.bundle[self.role]
        self.role+=1
        if self.role==3:self.role=0;self.layer+=1
        return result
    def complete(self):require(self.layer==self.req['physical_identity']['layers'] and self.role==0,'incomplete sparse prefill')

def audit_sparse_receipts(projections,client,ud):
    """Independent count/coordinate/wire replay; NOT a GPU tensor-value audit."""
    require(len(projections)==3*len(client),'incomplete projection receipt coverage')
    require(len({(r['layer'],r['role']) for r in projections})==len(projections),'duplicate projection receipts')
    used=set();fresh_rows=reused_rows=wire=0
    for receipt in client:
        c=receipt['context'];n=len(c['fresh_positions']);rows=[r for r in projections if r['layer']==c['layer']]
        require({r['role'] for r in rows}==set('qkv'),'missing projection role')
        for row in rows:
            fresh=set(row['fresh_positions']);reused=set(row['reused_positions'])
            require(row['fresh_positions']==c['fresh_positions'] and len(fresh)==n and not fresh&reused and fresh|reused==set(range(c['full_length'])),'gather/scatter partition differs')
            require(row['native_projection_rows']==row['cpu_lora_rows']==n and row['past_kv_grant'] is False,'false computation/forward-skip claim')
            fresh_rows+=n;reused_rows+=len(reused)
        if n:
            matches=[(i,r) for i,r in enumerate(ud) if r['context']==c]
            require(len(matches)==1 and matches[0][0] not in used,'missing/duplicate UD computation receipt')
            i,record=matches[0];used.add(i);require(record['cpu_projection_rows']==3*n,'UD row accounting differs')
        sent=receipt['sent'];received=receipt['received']
        require(sent['complete'] and received['complete'] and sent['operation']=='send' and received['operation']=='recv','incomplete wire evidence')
        require(sent['header']['metadata']['sparse']==c and received['header']['metadata']['sparse_sha256']==digest(c),'wire context differs')
        if n:require(sent['header']['payload_bytes']==n*c['request']['physical_identity']['hidden_size']*(4 if sent['header']['dtype']=='float32' else 2) and received['header']['payload_bytes']==3*n*c['request']['physical_identity']['hidden_size']*4,'payload byte accounting differs')
        wire+=sent['wire_bytes']+received['wire_bytes']
    require(len(used)==len(ud),'extra UD execution receipts')
    return dict(status='CPU_RECEIPT_REPLAY_PASS',fresh_projection_rows=fresh_rows,reused_projection_rows=reused_rows,actual_wire_bytes=wire,tensor_equivalence='NOT_EVALUATED_BY_THIS_AUDIT')
