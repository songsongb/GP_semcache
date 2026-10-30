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
from .r4c_session import DynamicSession,context
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
        first=len(self.channel.records);captures={};observed=[];components={};start=time.monotonic_ns();gpu_ms=None
        if self.fixture:
            def project(layer,role,x,native):
                delta=session.correction(layer,role,x);tensor=ArrayTensor([f32(a+b) for a,b in zip(native(),delta)],[1,len(inputs),self.identity['hidden_size']])
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


def generate(driver,task,cache):
    prompt=task['prompt'];p=task['policy'];rid=task['request_id'];mode=task['mode'];target=list(prompt['token_ids'])
    driver.sample_id=task['sample_id'];driver.evidence.profile=task['evidence_profile'];cache.request_id=rid
    driver.evidence.check();driver.synchronize();start=time.monotonic_ns();before=cache.snapshot();first=len(driver.channel.records);event_start=len(cache.events)
    entry=grant=past=None;lookup_ms=insert_ms=0.;step=None;steps=[];generated=[];logits=bytearray();source_blocks=proof=None
    try:
        require(prompt['attention_mask']==[1]*len(target) and prompt.get('positions',list(range(len(target))))==list(range(len(target))), 'C1 requires unpadded aligned prompts')
        if mode=='PARTIAL_PREFIX_HIT':
            t=time.monotonic_ns();entry,grant,past=cache.find(prompt,rid,expected_length=task['prefix_length']);driver.synchronize();lookup_ms=(time.monotonic_ns()-t)/1e6
            require(grant['source_request_id']==task['source_request_id'],'HIT source differs from scheduled completed source')
        else:require(mode in ('NO_REUSE','COLD_MISS'),'unsupported mode; no fallback')
        initial_bytes=0 if past is None else sum(t.numel()*t.element_size() for pair in cache_pairs(past) for t in pair)
        for step in range(p['max_new_tokens']):
            tick=time.monotonic_ns();inputs=target[(grant['prefix_length'] if grant else 0):] if step==0 else [generated[-1]]
            out,record,blocks,values,blob=driver.c_forward(rid,step,inputs,target,past,p,mode,cache.cohort,grant)
            token=max(range(len(values)),key=values.__getitem__);generated.append(token);logits.extend(blob);past=out.past_key_values
            if step==0:
                source_blocks=blocks if mode=='COLD_MISS' else None;proof=record['raw_kv_proof'];prefill_end=time.monotonic_ns()
            del out,blocks;driver.synchronize();now=time.monotonic_ns()
            record.update(request_id=rid,step=step,selected_token_id=token,logits_sha256=digest_bytes(blob),wall_ms=(now-tick)/1e6,
                          start_monotonic_ns=tick,available_monotonic_ns=now)
            steps.append(record);append(driver.output/'r4c_tokens.jsonl',record);driver.evidence.check()
            if step==0:ttft=(now-start)/1e6
            reason=termination(p,generated)
            if reason is not None:break
        require(reason is not None,'generation terminated without EOS/length cap')
        # UD verifies all expected layer forwards and consumed generated IDs before
        # granting completion; early EOS is not an unverified shortened B2 run.
        from .r4c_session import VERSION
        cache.control(dict(version=VERSION,cohort_sha256=cache.cohort,action='COMPLETE_REQUEST',parent_request_id=rid,
                           generated_token_ids=generated,termination=reason))
        text=driver.decode_text([t for t in generated if t not in {p['bos_token_id'],p['eos_token_id'],p['pad_token_id']}]) if driver.fixture else driver.decode_text(generated)
        if mode=='COLD_MISS':
            cache.cache_type=driver.cache_type;t=time.monotonic_ns();admitted=cache.admit(prompt,source_blocks,proof,rid,completed=True)
            require(admitted is not None,'MISS_OR_REJECT: source admission denied');driver.synchronize();insert_ms=(time.monotonic_ns()-t)/1e6
        if entry is not None:cache.reused(cache.entries[grant['entry_id']].context,entry)
        driver.synchronize();end=time.monotonic_ns();elapsed=(end-start)/1e6
        fields=('hidden_frames','delta_frames','hidden_payload_bytes','delta_payload_bytes','application_wire_bytes','es_to_ud_wire_bytes','ud_to_es_wire_bytes')
        totals=lambda rows:{k:sum(s[k] for s in rows) for k in fields}
        frames=driver.channel.records[first:]
        row=dict(request_id=rid,sample_id=task['sample_id'],pair_id=task['pair_id'],dataset=task['dataset'],mode=mode,status='COMPLETE',policy=p,
            evidence_profile=task['evidence_profile'],workload_label=task['workload_label'],source=grant,generated_token_ids=generated,generated_text=text,
            generated_tokens=len(generated),termination=reason,prompt_sha256=prompt['prompt_sha256'],target_token_ids=target,
            target_prepared_sha256=prompt.get('prepared_record_sha256',prompt.get('target_prepared_sha256')),
            ttft_ms=ttft,e2e_ms=elapsed,prefill_ms=(prefill_end-start)/1e6,decode_ms=sum(s['wall_ms'] for s in steps[1:]),
            request_start_monotonic_ns=start,request_end_monotonic_ns=end,
            decode_tokens_per_second=None if len(steps)<2 else (len(steps)-1)/(sum(s['wall_ms'] for s in steps[1:])/1000),
            cache_lookup_validation_and_kv_construction_ms=lookup_ms,cache_insert_ms=insert_ms,
            cache_lookup_validation_ms=0. if grant is None else cache.last_lookup_timings['cache_lookup_validation_ms'],
            kv_construction_and_validation_ms=0. if grant is None else cache.last_lookup_timings['kv_construction_and_validation_ms'],
            prefill=totals(steps[:1]),decode=totals(steps[1:]),cache_before=before,cache_after=cache.snapshot(),cache_events=cache.events[event_start:],
            reused_prefix_tokens=0 if grant is None else grant['prefix_length'],newly_computed_prefill_tokens=steps[0]['context']['input_length'],
            skipped_prefix_projection_token_rows=steps[0]['prefix_projection_token_rows_skipped'],executed_suffix_projection_token_rows=steps[0]['executed_projection_token_rows'],
            projection_kernels_entirely_skipped=0,cached_prefix_q_consumed=False,cached_raw_qkv_bytes=cache.bytes,cached_derived_kv_bytes=0,
            temporary_derived_kv_bytes=initial_bytes,control_wire_bytes=sum(f['wire_bytes'] for f in frames if f['header']['kind'] not in ('HIDDEN','DELTAS')),
            total_request_wire_bytes=sum(f['wire_bytes'] for f in frames),steps=steps,logits_blob=bytes(logits),logits_shape=[1,len(generated),driver.vocab],logits_dtype='float32',
            timing_scope='synchronized instrumented localhost wall; evidence hashing/writes and completion receipt included; nested timers not additive')
        return row
    except BaseException as exc:
        append(driver.output/'r4c_failed_requests.jsonl',dict(request_id=rid,sample_id=task['sample_id'],mode=mode,generation_step=step,
            stage='CACHE_LOOKUP' if mode=='PARTIAL_PREFIX_HIT' and grant is None else 'GENERATION_OR_ADMISSION',source=grant,
            target_token_ids=target,partial_generated_ids=generated,reason=f'{type(exc).__name__}: {exc}',cache=cache.snapshot()))
        raise

