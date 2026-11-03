# I5 additive version of r4c_generation.py; frozen source remains unchanged. See I5 report.
"""C1 suffix-only physical OPT forward with variable policy and bounded evidence."""
from contextlib import contextmanager
import json
import time

from .communication import digest,digest_bytes,require
from .r3a_artifacts import append,pack
from .r3a_tiny import f32
from .r3b_cache import spec
from .r3b_fixture import ArrayTensor
from .r3b_runtime import physical_path,wire_totals
from .r4a_diagnostics import observe_opt_positions
from .r4a_generation import ForwardDriver,past_summary,cache_pairs
from .r4b2_kv import attention_kv,installed_contract,slice_tokens,verify_raw_kv
from .r5_i5_protocol import DynamicSession,context
from .r4c_evidence import Evidence
from .r4c_workload import termination
import math


class QualityDriver(ForwardDriver):
    def last_logits(self,out,*,sample_id,record,mask):
        shape=list(out.shape if self.fixture else out.logits.shape); c=record['context']; n=c['input_length']
        require(shape==[1,n,self.vocab], 'actual C1 logit shape differs')
        if self.fixture:
            all_values=list(out.logits); values=all_values[-self.vocab:]
            count=lambda xs:dict(total=len(xs),nonfinite=sum(not math.isfinite(v) for v in xs),nan=sum(math.isnan(v) for v in xs),positive_inf=sum(v==math.inf for v in xs),negative_inf=sum(v==-math.inf for v in xs))
            counts=dict(all_logits=count(all_values),valid_positions=count(all_values),padded_positions=count([]),last_position=count(values))
        else:
            import torch
            logits=out.logits.detach()
            def count(t):
                return dict(total=t.numel(),nonfinite=int((~torch.isfinite(t)).sum()),nan=int(torch.isnan(t).sum()),positive_inf=int(torch.isposinf(t).sum()),negative_inf=int(torch.isneginf(t).sum()))
            counts=dict(all_logits=count(logits),valid_positions=count(logits),padded_positions=count(logits[:,:0]),last_position=count(logits[:,-1:]))
            values=logits[0,-1].float().cpu().tolist()
        diagnostic=dict(sample_id=sample_id,request_id=c['parent_request_id'],generation_step=c['pass_index'],total_input_length=len(mask),
            nonpadding_tokens=sum(mask),padding_tokens=len(mask)-sum(mask),input_length=n,**counts)
        append(self.diagnostics_path,diagnostic);record['logit_diagnostic']=diagnostic
        require(counts['all_logits']['nonfinite']==0,'nonfinite actual model logits: '+json.dumps(diagnostic,sort_keys=True))
        return values,pack(values,'float32')
    def configure(self,output,*,budget_bytes=2**30):
        require(not self.native,'C1 target endpoint must be distributed')
        self.output=output;self.evidence=Evidence(output,budget_bytes=budget_bytes);self.diagnostics_path=output/'r4c_logit_diagnostics.jsonl'
        self.contract=installed_contract(self);self.cache_type=None
        self.context_limit=2048 if self.fixture else int(self.model.config.max_position_embeddings)
        self.contract.update(c1_version=1,output_attentions=False,context_limit=self.context_limit,vocabulary_size=self.vocab)
        if not self.fixture:
            require(all(type(layer.self_attn).__name__=='OPTAttention' for layer in self.adapter.layers),'C1 requires inspected eager OPTAttention; no backend substitution')
        if not self.fixture:
            import transformers.cache_utils as cu
            from pathlib import Path
            self.contract['cache_module_sha256']=digest_bytes(Path(cu.__file__).read_bytes())
        return self.contract

    @contextmanager
    def components(self,n,captured,tag):
        handles=[]
        def hook(key):
            def observe(module,args,kwargs,out):
                t=out[0] if isinstance(out,tuple) else out
                if t.ndim==2:t=t.reshape(1,n,-1)
                require(t.ndim==3 and list(t.shape[:2])==[1,n],'actual attention/FFN processed prefix rows on HIT')
                captured[key]=self.evidence.save(t,tag+'_'+key)
            return observe
        try:
            for i,layer in enumerate(self.adapter.layers):
                for name,module in (('attention',layer.self_attn),('ffn',layer.fc2),('layer',layer)):
                    handles.append(module.register_forward_hook(hook(f'{i}.{name}'),with_kwargs=True))
            def head(module,args,out):
                require(list(out.shape)==[1,n,self.vocab] and args[0].shape[1]==n,'LM head secretly processed prefix')
            handles.append(self.model.lm_head.register_forward_hook(head));yield
        finally:
            for handle in handles:handle.remove()

    def c_forward(self,parent,step,inputs,target,past,policy,mode,cohort,grant=None,*,use_cache=True,observer=None):
        self.evidence.check(16*2**20)  # Bounded JSON/wire ledger headroom before another forward.
        old=0 if past is None else len(target)+step-1 if step else grant['prefix_length']
        c=context(parent,step,inputs,target,old,policy,mode,cohort,grant,context_limit=self.context_limit);fid=f'{parent}.s{step:02d}'
        append(self.output/'forward_attempts.jsonl',dict(forward_id=fid,sample_id=self.sample_id,context=c))
        input_past=None if past is None else past_summary(past,old,self.identity['layers'],self.heads,self.identity['hidden_size'],self.identity['hidden_dtype'],self.device)
        input_kv=[] if past is None else [[spec(t) for t in pair] for pair in cache_pairs(past)]
        session=DynamicSession(self.channel,self.identity,fid,c,torch_backend=not self.fixture)
        transform=getattr(self,'experimental_native_projection',None)
        if transform is not None:session.experimental_native_projection=lambda layer,role,native,x:transform(c,layer,role,native,x)
        first=len(self.channel.records);captures={};observed=[];components={};start=time.monotonic_ns();gpu_ms=None
        if self.fixture:
            def project(layer,role,x,native):
                delta=session.correction(layer,role,x)
                if transform is None:base=native()
                else:
                    from .r3a_tiny import linear
                    h=self.identity['hidden_size']
                    fn=lambda padded:ArrayTensor(linear(padded.values,self.tiny.base['layers'][layer][role],h,h),list(padded.shape))
                    base=transform(c,layer,role,fn,ArrayTensor(x,[1,len(inputs),h])).values
                tensor=ArrayTensor([f32(a+b) for a,b in zip(base,delta)],[1,len(inputs),self.identity['hidden_size']])
                captures.setdefault(layer,{})[role]=tensor.clone();observed.append(dict(layer=layer,role=role,reused=False));return tensor.values
            out=self.tiny.forward(inputs,c['attention_mask'],past,project,use_cache);session.complete()
            position=dict(source='TEST_FIXTURE_ONLY_ACTUAL_TINY_POSITION_MATH',positions=c['positions'],past_length=old)
        else:
            import torch
            args=dict(input_ids=torch.tensor([inputs],device=self.device),attention_mask=torch.tensor([c['attention_mask']],device=self.device),
                past_key_values=past,use_cache=use_cache,output_attentions=False,return_dict=True)
            if 'position_ids' in self.parameters:args['position_ids']=torch.tensor([c['positions']],device=self.device)
            if 'cache_position' in self.parameters:args['cache_position']=torch.tensor(c['cache_positions'],device=self.device)
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            from contextlib import nullcontext
            with torch.inference_mode(),physical_path(self.adapter,session,None,captures,observed),self.components(len(inputs),components,fid), \
                    observe_opt_positions(self.position_embedding,c['attention_mask'],old) as positions, \
                    (observer(self.adapter) if observer is not None else nullcontext()):
                begin.record();out=self.model(**args);end.record();self.synchronize()
            gpu_ms=begin.elapsed_time(end);position=positions[0]
            from .r3a_artifacts import write
            write(self.output/'gpu_execution.json',dict(gpu_inference_performed=True))
        summary=None if not use_cache else past_summary(out.past_key_values,old+len(inputs),self.identity['layers'],self.heads,
            self.identity['hidden_size'],self.identity['hidden_dtype'],self.device)
        record=dict(forward_id=fid,context=c,input_past=input_past,past=summary,position_observation=position,cache_positions=c['cache_positions'],
            model_logits_dtype='float32' if self.fixture else str(out.logits.dtype).removeprefix('torch.'),cuda_event_elapsed_ms=gpu_ms)
        values,blob=self.last_logits(out,sample_id=self.sample_id,record=record,mask=c['attention_mask'])
        require(len(observed)==3*self.identity['layers'],'incomplete actual suffix QKV execution')
        blocks={l:tuple(b[r] for r in 'qkv') for l,b in captures.items()};record['qkv']={}
        for l,block in blocks.items():
            for r,t in zip('qkv',block):record['qkv'][f'{l}.{r}']=self.evidence.save(t,f'{fid}_L{l}_{r}')
        record['components']=components
        record['kv']=[];record['input_kv']=input_kv;record['raw_kv_proof']=None
        if use_cache:
            derived=attention_kv(blocks,self.heads,self.identity['hidden_size'])
            for l,pair in enumerate(cache_pairs(out.past_key_values)):
                item={}
                for j,(r,t) in enumerate(zip('kv',pair)):
                    require(spec(slice_tokens(t,old,old+len(inputs)))==spec(derived[l][j]),'actual attention K/V append differs from Raw K/V transform')
                    if old:require(spec(slice_tokens(t,0,old))==input_kv[l][j],'actual HF decode/suffix changed prior cached history')
                    item[r]=self.evidence.save(t,f'{fid}_L{l}_past_{r}') if step==0 else spec(t)
                record['kv'].append(item)
            if step==0 and not old:
                record['raw_kv_proof']=verify_raw_kv(blocks,out.past_key_values,self.heads,self.identity,self.device)
                self.cache_type=type(out.past_key_values)
        frames=self.channel.records[first:];record.update(wire_totals(frames),
            es_to_ud_wire_bytes=sum(f['wire_bytes'] for f in frames if f['operation']=='send'),
            ud_to_es_wire_bytes=sum(f['wire_bytes'] for f in frames if f['operation']=='recv'),
            executed_projection_calls=len(observed),executed_projection_token_rows=len(inputs)*len(observed),
            prefix_projection_token_rows_skipped=(grant['prefix_length']*3*self.identity['layers'] if grant is not None and step==0 else 0),
            prefix_q_consumed=False,host_forward_and_evidence_ms=(time.monotonic_ns()-start)/1e6)
        record['temporary_raw_kv_verification_bytes']=sum(t.numel()*t.element_size() for pair in derived for t in pair) if use_cache else 0
        append(self.output/'forward_completed.jsonl',dict(forward_id=fid,context=c,actual_kv=summary))
        return out,record,blocks,values,blob

