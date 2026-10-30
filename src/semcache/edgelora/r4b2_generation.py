"""Suffix-only B2 prefill and normal G-1 decode; no prefix forward on HIT."""
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
from .r4a_generation import ForwardDriver,past_summary,cache_pairs,generation_policy,validate_generated
from .r4b2_kv import TensorEvidence,attention_kv,installed_contract,slice_tokens,verify_raw_kv
from .r4b2_session import DynamicSession,context


class PrefixDriver(ForwardDriver):
    def configure(self,output):
        require(not self.native,'B2 target endpoint must be distributed')
        self.output=output;self.evidence=TensorEvidence(output);self.diagnostics_path=output/'r4b2_logit_diagnostics.jsonl'
        self.contract=installed_contract(self);self.cache_type=None
        if not self.fixture:
            import transformers.cache_utils as cu
            from pathlib import Path
            self.contract['cache_module_sha256']=digest_bytes(Path(cu.__file__).read_bytes())
        return self.contract

    @contextmanager
    def components(self,n,captured):
        handles=[]
        def hook(key):
            def observe(module,args,kwargs,out):
                t=out[0] if isinstance(out,tuple) else out
                if t.ndim==2:t=t.reshape(1,n,-1)
                require(t.ndim==3 and t.shape[0]==1 and t.shape[1]==n,'actual suffix operation processed an unexpected token count')
                captured[key]=t.detach().clone();return None
            return observe
        try:
            for i,layer in enumerate(self.adapter.layers):
                for name,module in (('attention',layer.self_attn),('ffn',layer.fc2),('layer',layer)):
                    handles.append(module.register_forward_hook(hook(f'{i}.{name}'),with_kwargs=True))
            def head(module,args,out):
                require(list(out.shape)==[1,n,self.vocab] and args[0].shape[1]==n,'LM head secretly processed prefix tokens')
            handles.append(self.model.lm_head.register_forward_hook(head))
            yield
        finally:
            for handle in handles:handle.remove()

    def b2_forward(self,parent,step,inputs,target,past,total,mode,cohort,grant=None,*,use_cache=True,observer=None):
        old=0 if past is None else len(target)+step-1 if step else grant['prefix_length']
        c=context(parent,step,inputs,target,old,total,mode,cohort,grant);fid=f'{parent}.s{step:02d}'
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
                past_key_values=past,use_cache=use_cache,output_attentions=True,return_dict=True)
            if 'position_ids' in self.parameters:args['position_ids']=torch.tensor([c['positions']],device=self.device)
            if 'cache_position' in self.parameters:args['cache_position']=torch.tensor(c['cache_positions'],device=self.device)
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            from contextlib import nullcontext
            with torch.inference_mode(),physical_path(self.adapter,session,None,captures,observed),self.components(len(inputs),components), \
                    observe_opt_positions(self.position_embedding,c['attention_mask'],old) as positions, \
                    (observer(self.adapter) if observer is not None else nullcontext()):
                begin.record();out=self.model(**args);end.record();self.synchronize()
            gpu_ms=begin.elapsed_time(end);position=positions[0]
        summary=None if not use_cache else past_summary(out.past_key_values,old+len(inputs),self.identity['layers'],self.heads,
            self.identity['hidden_size'],self.identity['hidden_dtype'],self.device)
        record=dict(forward_id=fid,context=c,input_past=input_past,past=summary,position_observation=position,cache_positions=c['cache_positions'],
            model_logits_dtype='float32' if self.fixture else str(out.logits.dtype).removeprefix('torch.'),cuda_event_elapsed_ms=gpu_ms)
        values,blob=self.last_logits(out,sample_id=self.sample_id,record=record,mask=c['attention_mask'])
        require(len(observed)==3*self.identity['layers'],'incomplete actual suffix QKV execution')
        blocks={l:tuple(b[r] for r in 'qkv') for l,b in captures.items()};record['qkv']={}
        for l,block in blocks.items():
            for r,t in zip('qkv',block):record['qkv'][f'{l}.{r}']=self.evidence.save(t,f'{fid}_L{l}_{r}')
        record['components']={k:self.evidence.save(t,f'{fid}_{k}') for k,t in components.items()} if step==0 else {k:spec(t) for k,t in components.items()}
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
        append(self.output/'forward_completed.jsonl',dict(forward_id=fid,context=c,actual_kv=summary))
        return out,record,blocks,values,blob


def generate(driver,prompt,policy,rid,cache,mode,*,expected_prefix=None,cohort=None):
    before=cache.snapshot();event_start=len(cache.events);driver.sample_id=prompt['sample_id'];cache.request_id=rid
    frame_start=len(driver.channel.records)
    driver.synchronize();start=time.monotonic_ns();lookup_ms=insert_ms=text_ms=0.;entry=grant=past=None
    target=list(prompt['token_ids']);current_step=None
    try:
        require(prompt['attention_mask']==[1]*len(target),'B2 requires unpadded actual tokens')
        require(policy==generation_policy({k:policy[k] for k in ('bos_token_id','eos_token_id','pad_token_id')},'fixed',policy['max_new_tokens']),
            'B2 pilot requires identical bounded fixed greedy policy')
        if mode=='PARTIAL_PREFIX_HIT':
            t=time.monotonic_ns();entry,grant,past=cache.find(prompt,rid,expected_length=expected_prefix,cohort=cohort);driver.synchronize();lookup_ms=(time.monotonic_ns()-t)/1e6
        else:require(mode in ('NO_REUSE','COLD_MISS'),'unsupported B2 mode; no full-prompt HIT substitution')
        initial_kv_bytes=0 if past is None else sum(t.numel()*t.element_size() for pair in cache_pairs(past) for t in pair)
        generated=[];steps=[];logits=bytearray();source_blocks=source_proof=None;available=[]
        for step in range(policy['max_new_tokens']):
            current_step=step
            inputs=target[(grant['prefix_length'] if grant else 0):] if step==0 else [generated[-1]]
            tick=time.monotonic_ns();out,record,blocks,values,blob=driver.b2_forward(rid,step,inputs,target,past,policy['max_new_tokens'],mode,cache.cohort,grant)
            token=max(range(len(values)),key=values.__getitem__);generated.append(token);logits.extend(blob);past=out.past_key_values
            if step==0:source_blocks=blocks;source_proof=record['raw_kv_proof'];prefill_end=time.monotonic_ns()
            del out;driver.synchronize();now=time.monotonic_ns();available.append(now)
            record.update(request_id=rid,step=step,selected_token_id=token,logits_sha256=digest_bytes(blob),start_monotonic_ns=tick,
                available_monotonic_ns=now,wall_ms=(now-tick)/1e6)
            steps.append(record);append(driver.output/'r4b2_tokens.jsonl',record)
        validate_generated(policy,generated);t=time.monotonic_ns();text=driver.decode_text(generated);text_ms=(time.monotonic_ns()-t)/1e6
        if mode=='COLD_MISS':
            cache.cache_type=driver.cache_type;t=time.monotonic_ns();registered=cache.admit(prompt,source_blocks,source_proof,rid,completed=True)
            require(registered is not None,'MISS_OR_REJECT: insufficient prefix capacity/admission denied');driver.synchronize();insert_ms=(time.monotonic_ns()-t)/1e6
        if entry is not None:
            c=cache.entries[grant['entry_id']].context;cache.reused(c,entry)
        driver.synchronize();end=time.monotonic_ns();events=cache.events[event_start:]
        fields=('hidden_frames','delta_frames','hidden_payload_bytes','delta_payload_bytes','application_wire_bytes','es_to_ud_wire_bytes','ud_to_es_wire_bytes')
        pre={k:steps[0][k] for k in fields};dec={k:sum(s[k] for s in steps[1:]) for k in fields}
        return dict(request_id=rid,sample_id=prompt['sample_id'],mode=mode,cache_status='PARTIAL_PREFIX_HIT' if grant else 'MISS' if mode=='COLD_MISS' else 'BYPASS',
            target_token_ids=target,prompt_sha256=prompt['prompt_sha256'],policy=policy,generated_token_ids=generated,generated_text=text,
            generated_tokens=len(generated),prompt_tokens=len(target),reused_prefix_tokens=0 if grant is None else grant['prefix_length'],
            newly_computed_prefill_tokens=len(steps[0]['context']['input_token_ids']),source=grant,prefill_accounting=pre,decode_accounting=dec,
            ttft_ms=(available[0]-start)/1e6,e2e_wall_ms=(end-start)/1e6,prefill_ms=(prefill_end-start)/1e6,
            decode_ms=(available[-1]-available[0])/1e6,cache_lookup_ms=lookup_ms,cache_insert_ms=insert_ms,text_postprocess_ms=text_ms,
            cache_eviction_ms=sum(e['duration_ms'] for e in events if e['kind'] in ('EVICT','INVALIDATE','REJECT')),
            request_start_monotonic_ns=start,request_end_monotonic_ns=end,first_token_available_monotonic_ns=available[0],
            final_decode_available_monotonic_ns=available[-1],decode_forward_count=len(generated)-1,
            prefix_projection_token_rows_skipped=steps[0]['prefix_projection_token_rows_skipped'],
            suffix_projection_token_rows_executed=steps[0]['executed_projection_token_rows'],prefix_q_consumed=False,
            cached_raw_qkv_bytes=cache.bytes,cached_derived_kv_bytes=0,request_initial_prefix_kv_bytes=initial_kv_bytes,
            cache_before=before,cache_after=cache.snapshot(),cache_events=events,
            cache_control_wire_bytes=sum(f['wire_bytes'] for f in driver.channel.records[frame_start:] if f['header']['kind'] not in ('HIDDEN','DELTAS')),
            request_total_wire_bytes=sum(f['wire_bytes'] for f in driver.channel.records[frame_start:]),
            logits_blob=bytes(logits),logits_sha256=digest_bytes(bytes(logits)),logits_shape=[1,len(generated),driver.vocab],logits_dtype='float32',
            timing_scope='synchronized instrumented wall includes cache validation/construction, full QKV/KV evidence hashing and diagnostic writes; nested timers not additive')
    except BaseException as exc:
        append(driver.output/'r4b2_failed_requests.jsonl',dict(request_id=rid,sample_id=prompt['sample_id'],mode=mode,status='MISS_OR_REJECT' if grant is None and mode=='PARTIAL_PREFIX_HIT' else 'FAILED',
            source=grant,generation_step=current_step,target_token_ids=target,attention_mask=list(prompt['attention_mask']),positions=prompt.get('positions'),
            prompt_sha256=prompt.get('prompt_sha256'),expected_prefix=expected_prefix,requested_cohort_sha256=cohort,cache_cohort_sha256=cache.cohort,
            reason=f'{type(exc).__name__}: {exc}',request_start_monotonic_ns=start,failed_monotonic_ns=time.monotonic_ns()))
        raise
