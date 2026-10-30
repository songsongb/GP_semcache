"""Direct OPT prefill + single-token KV-cached decode, shared by all modes."""
from contextlib import nullcontext
import inspect
import json
from pathlib import Path
import time

from .communication import digest,digest_bytes,require
from .r3a_artifacts import append,pack,tensor_bytes,unpack
from .r3b_cache import spec
from .r3b_fixture import ArrayTensor
from .r3b_runtime import physical_path,total_projection,wire_totals
from .r4a_session import DynamicSession,context,positions
from .r4a_diagnostics import logit_counts,observe_opt_positions


def cache_pairs(past):
    if isinstance(past,(list,tuple)):return list(past)
    if hasattr(past,'to_legacy_cache'):return list(past.to_legacy_cache())
    if hasattr(past,'layers'):return [(layer.keys,layer.values) for layer in past.layers]
    if hasattr(past,'key_cache') and hasattr(past,'value_cache'):return list(zip(past.key_cache,past.value_cache))
    raise ValueError('unsupported actual HF past_key_values type; no guessed KV layout')


def past_summary(past,expected_length,layers,heads,hidden,dtype,device):
    require(past is not None,'use_cache=True returned no actual KV cache')
    reported=None
    if hasattr(past,'get_seq_length'):
        reported=int(past.get_seq_length());require(reported==expected_length,'actual cache-reported sequence length mismatch')
    pairs=cache_pairs(past);require(len(pairs)==layers,'incomplete actual past-KV layers');records=[]
    for i,pair in enumerate(pairs):
        require(len(pair)==2,'unsupported cross-attention/static KV cache')
        for role,tensor in zip('kv',pair):
            require(list(tensor.shape)==[1,heads,expected_length,hidden//heads] and
                str(tensor.dtype).removeprefix('torch.')==dtype and str(tensor.device)==device,'KV shape/length/dtype/device mismatch')
        records.append(dict(layer=i,k_shape=list(pair[0].shape),v_shape=list(pair[1].shape),dtype=dtype,device=device))
    return dict(type=type(past).__name__,sequence_length=expected_length,reported_sequence_length=reported,layers=records)


class ForwardDriver:
    def __init__(self,identity,device,*,channel=None,model=None,adapter=None,tiny=None,native=False,diagnostics_path=None):
        self.identity=identity;self.device=device;self.channel=channel;self.model=model;self.adapter=adapter;self.tiny=tiny;self.native=native
        self.fixture=tiny is not None
        self.diagnostics_path=None if diagnostics_path is None else Path(diagnostics_path)
        self.diagnostic_records=0
        if not self.fixture:
            base=model.get_base_model() if hasattr(model,'peft_config') else model
            self.parameters=inspect.signature(base.forward).parameters
            require(all(k in self.parameters for k in ('input_ids','attention_mask','past_key_values','use_cache')),'unsupported OPT forward API')
            require(not model.training and not any(m.training for m in model.modules()),'eval/dropout contract violated')
            self.heads=model.config.num_attention_heads;self.vocab=model.config.vocab_size
            self.position_embedding=base.model.decoder.embed_positions
        else:self.heads=tiny.heads;self.vocab=tiny.vocab
    def synchronize(self):
        if not self.fixture:
            import torch
            if str(self.device).startswith('cuda'):torch.cuda.synchronize(self.device)
    def decode_text(self,tokens):
        return self.tiny.decode_text(tokens) if self.fixture else self.model_tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    def forward(self,parent,step,tokens,mask,past,total_passes,*,entry=None,prefill_hit=False,triage=False,use_cache=True,observer=None,natural=False):
        old=0 if step==0 else len(mask)-len(tokens)
        input_past=None
        if past is not None:input_past=past_summary(past,old,self.identity['layers'],self.heads,self.identity['hidden_size'],self.identity['hidden_dtype'],self.device)
        else:require(step==0,'decode missing actual in-request KV')
        c=context(parent,step,tokens,mask,old,total_passes,hit=prefill_hit,triage=triage,natural=natural)
        fid=f'{parent}.s{step:02d}';captures={};observed=[];first=len(self.channel.records) if self.channel is not None else 0
        session=None if self.native else DynamicSession(self.channel,self.identity,fid,c,torch_backend=not self.fixture)
        forward_start=time.monotonic_ns();gpu_ms=None
        position_observations=[dict(source='TEST_FIXTURE_ONLY_EXPECTED_POSITIONS',positions=positions(mask)[old:],past_length=old)]
        if self.fixture:
            from .r3a_tiny import f32
            def project(layer,role,x,base):
                def miss():return ArrayTensor([f32(a+b) for a,b in zip(base(),session.correction(layer,role,x))],[1,len(tokens),self.identity['hidden_size']])
                tensor=total_projection(entry,layer,role,miss);captures.setdefault(layer,{})[role]=tensor.clone();observed.append(role)
                return tensor.values
            out=self.tiny.forward(tokens,mask,past,None if self.native else project,use_cache)
            if not self.native and entry is None:session.complete()
        else:
            import torch
            args=dict(input_ids=torch.tensor([tokens],device=self.device),attention_mask=torch.tensor([mask],device=self.device),
                past_key_values=past,use_cache=use_cache,output_attentions=True,return_dict=True)
            if 'position_ids' in self.parameters:args['position_ids']=torch.tensor([positions(mask)[old:]],device=self.device)
            if 'cache_position' in self.parameters:args['cache_position']=torch.arange(old,old+len(tokens),device=self.device)
            path=nullcontext() if self.native else physical_path(self.adapter,session,entry,captures,observed)
            # Passive triage hooks are installed AFTER the bare-path hook checks
            # and removed BEFORE its finally restores the original projections.
            cuda=str(self.device).startswith('cuda')
            if cuda:begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            with torch.inference_mode(),path,(observer(self.adapter) if observer is not None else nullcontext()), \
                    observe_opt_positions(self.position_embedding,mask,old) as position_observations:
                if cuda:begin.record()
                out=self.model(**args)
                if cuda:end.record();self.synchronize()
            if cuda:gpu_ms=begin.elapsed_time(end)
        if not self.native:
            require(sum(len(v) for v in captures.values())==3*self.identity['layers'],'not all actual QKV outputs observed')
        require(use_cache or out.past_key_values is None,'use_cache=False endpoint returned an unexpected past')
        summary=None if not use_cache else past_summary(out.past_key_values,old+len(tokens),self.identity['layers'],self.heads,
            self.identity['hidden_size'],self.identity['hidden_dtype'],self.device)
        frames=[] if self.channel is None else self.channel.records[first:]
        totals=wire_totals(frames) if frames else dict(hidden_frames=0,delta_frames=0,hidden_payload_bytes=0,delta_payload_bytes=0,
            application_wire_bytes=0,send_blocking_ms=0.,receive_blocking_ms=0.,framing_serialization_ms=0.)
        qkv={f'{l}.{r}':spec(t)['sha256'] for l,b in captures.items() for r,t in b.items()}
        record=dict(forward_id=fid,context=c,input_past=input_past,past=summary,position_observation=position_observations[0],
            cache_positions=list(range(old,old+len(tokens))),qkv_sha256=qkv,reused_qkv_outputs=96 if entry is not None else 0,
            es_to_ud_wire_bytes=sum(f['wire_bytes'] for f in frames if f['operation']=='send'),
            ud_to_es_wire_bytes=sum(f['wire_bytes'] for f in frames if f['operation']=='recv'),
            model_logits_dtype='float32' if self.fixture else str(out.logits.dtype).removeprefix('torch.'),
            host_forward_and_evidence_ms=(time.monotonic_ns()-forward_start)/1e6,cuda_event_elapsed_ms=gpu_ms,
            cuda_scope='nested stream elapsed, includes host/socket idle; not kernel self time',
            ud_cpu_lora_ms=0. if session is None else sum(l['ud']['cpu_lora_ms'] for l in session.trace),
            es_copy_cast_ms=0. if session is None else sum(l['es_hidden_copy_cast_ms']+sum(l['es_delta_cpu_to_gpu_ms'].values()) for l in session.trace),
            precision_scope=self.identity['precision'],**totals)
        blocks={l:tuple(b[r] for r in 'qkv') for l,b in captures.items()}
        return out,record,blocks
    def last_logits(self,out,*,sample_id,record,mask):
        shape=list(out.shape if self.fixture else out.logits.shape)
        require(shape==[1,record['context']['input_length'],self.vocab],'actual model logit shape mismatch')
        counts=logit_counts(out.logits,shape,mask,torch_backend=not self.fixture)
        c=record['context']
        diagnostic=dict(schema='r4a_logit_diagnostic_v1',sample_id=sample_id,request_id=c['parent_request_id'],
            forward_id=record['forward_id'],generation_step=c['pass_index'],phase='prefill' if c['pass_index']==0 else 'decode',
            native=self.native,TEST_FIXTURE_ONLY=self.fixture,total_input_length=len(mask),input_length=c['input_length'],
            nonpadding_tokens=sum(mask),padding_tokens=len(mask)-sum(mask),query_nonpadding_tokens=sum(mask[-shape[1]:]),
            query_padding_tokens=shape[1]-sum(mask[-shape[1]:]),logits_shape=shape,model_logits_dtype=record['model_logits_dtype'],
            attention_mask=list(mask),positions=positions(mask)[c['past_length']:],cache_positions=record['cache_positions'],
            input_past=record['input_past'],output_past=record['past'],position_observation=record['position_observation'],**counts)
        record['logit_diagnostic']=diagnostic
        # Persist before the guard: failed requests never reach store_generation.
        # Hard budget covers 1296 serving + 192 Native-reference forwards.
        require(self.diagnostic_records<1500,'bounded R4 logit diagnostic budget exceeded')
        self.diagnostic_records+=1
        if self.diagnostics_path is not None:append(self.diagnostics_path,diagnostic)
        require(counts['all_logits']['nonfinite']==0,'nonfinite actual model logits: '+json.dumps(dict(
            sample_id=sample_id,request_id=c['parent_request_id'],generation_step=c['pass_index'],total_input_length=len(mask),
            nonpadding_tokens=sum(mask),padding_tokens=len(mask)-sum(mask),**counts),sort_keys=True))
        if self.fixture:values=list(out.logits[-self.vocab:])
        else:
            values=out.logits[0,-1,:].detach().float().cpu().tolist()
        require(len(values)==self.vocab,'last-token logit shape mismatch');blob=pack(values,'float32')
        return values,blob


def generation_policy(special,mode='fixed',max_new_tokens=8):
    require(mode in ('fixed','natural-eos') and type(max_new_tokens) is int and 1<=max_new_tokens<=16,'explicit bounded EOS/length policy required')
    require(set(special)=={'bos_token_id','eos_token_id','pad_token_id'} and
        all(type(v) is int and v>=0 for v in special.values()),'known BOS/EOS/pad IDs required')
    return dict(mode=mode,max_new_tokens=max_new_tokens,do_sample=False,temperature=None,top_k=None,top_p=None,
        ignore_eos=mode=='fixed',decode_skip_special_tokens=True,decode_clean_up_tokenization_spaces=False,**special)


def validate_generated(policy,tokens):
    require(tokens and len(tokens)<=policy['max_new_tokens'],'empty/excess generated output')
    if policy['mode']=='fixed':require(len(tokens)==policy['max_new_tokens'],'fixed generation ended early')
    else:
        eos=[i for i,t in enumerate(tokens) if t==policy['eos_token_id']]
        require(not eos or eos[0]==len(tokens)-1,'natural-EOS generation continued beyond EOS')
        require(len(tokens)==policy['max_new_tokens'] or tokens[-1]==policy['eos_token_id'],'natural generation stopped without EOS')


def generate(driver,prompt,policy,request_id,*,router=None,mode='NO_REUSE',on_token=None):
    """All arms use normal in-request KV. The final emitted token needs no decode."""
    require(mode in ('NO_REUSE','COLD_MISS','WARM_HIT'),'unknown serving mode')
    require(policy==generation_policy({k:policy[k] for k in ('bos_token_id','eos_token_id','pad_token_id')},policy['mode'],policy['max_new_tokens']),
        'decoding policy was altered')
    ids=list(prompt['token_ids']);mask=list(prompt['attention_mask']);positions(mask)
    cache_before=0 if router is None or router.active is None else router.active.bytes
    driver.synchronize();start=time.monotonic_ns();entry=cache=None;lookup_ms=insert_ms=0.;event_start=0
    if not driver.native:
        require(router is not None,'physical cache namespace required even for bypass accounting')
        if mode!='NO_REUSE':
            t=time.monotonic_ns();cache=router.bind(prompt);event_start=len(cache.events)
            if mode=='COLD_MISS':cache.reset('measured cold request reset inside request wall')
            entry=cache.lookup(cache.context);lookup_ms=(time.monotonic_ns()-t)/1e6
        require((entry is not None)==(mode=='WARM_HIT'),'invalid HIT/MISS; no fallback counted as HIT')
    past=None;generated=[];steps=[];logits=bytearray();prefix_blocks=None;availability=[]
    for step in range(policy['max_new_tokens']):
        inputs=ids if step==0 else [generated[-1]];t=time.monotonic_ns()
        out,record,blocks=driver.forward(request_id,step,inputs,mask,past,policy['max_new_tokens'],entry=entry if step==0 else None,
            prefill_hit=mode=='WARM_HIT',natural=policy['mode']=='natural-eos')
        if step==0:prefix_blocks=blocks;prefill_end=time.monotonic_ns()
        values,blob=driver.last_logits(out,sample_id=prompt.get('sample_id',request_id),record=record,mask=mask)
        selected=max(range(len(values)),key=values.__getitem__);generated.append(selected)
        logits.extend(blob);past=out.past_key_values;del out
        driver.synchronize();available=time.monotonic_ns();availability.append(available)
        record.update(step=step,selected_token_id=selected,logits_sha256=digest_bytes(blob),logits_shape=[1,1,driver.vocab],
            start_monotonic_ns=t,available_monotonic_ns=available,wall_ms=(available-t)/1e6,
            positions=positions(mask)[len(mask)-len(inputs):],input_token_ids=list(inputs),attention_mask_sha256=digest(mask))
        steps.append(record)
        if on_token is not None:on_token(record)
        if policy['mode']=='natural-eos' and selected==policy['eos_token_id']:break
        mask.append(1)
    validate_generated(policy,generated)
    # Publish a cold entry only after the ENTIRE generation succeeded. Final
    # response availability waits for admission. This costs E2E, not first token.
    if mode=='COLD_MISS':
        t=time.monotonic_ns();require(cache.insert(cache.context,prefix_blocks,request_id,completed=True),'cold prefix admission rejected')
        driver.synchronize();insert_ms=(time.monotonic_ns()-t)/1e6
    if entry is not None:cache.reused(entry)
    text_start=time.monotonic_ns();text=driver.decode_text(generated);text_ms=(time.monotonic_ns()-text_start)/1e6
    driver.synchronize();end=time.monotonic_ns()
    first=availability[0];final=end  # final response finalization includes cold admission
    gaps=[(b-a)/1e6 for a,b in zip(availability,availability[1:])]
    n=len(generated);wall=(end-start)/1e6
    fields=('hidden_frames','delta_frames','hidden_payload_bytes','delta_payload_bytes','application_wire_bytes',
        'es_to_ud_wire_bytes','ud_to_es_wire_bytes','ud_cpu_lora_ms','es_copy_cast_ms','framing_serialization_ms')
    prefill_accounting={k:steps[0][k] for k in fields};decode_accounting={k:sum(s[k] for s in steps[1:]) for k in fields}
    return dict(request_id=request_id,mode=mode,prompt_tokens=len(ids),nonpadding_prompt_tokens=sum(prompt['attention_mask']),generated_tokens=n,
        generated_token_ids=generated,generated_text=text,text_postprocess_ms=text_ms,policy=policy,prompt_sha256=prompt['prompt_sha256'],
        request_start_monotonic_ns=start,request_end_monotonic_ns=end,first_token_available_monotonic_ns=first,
        final_token_response_available_monotonic_ns=final,ttft_ms=(first-start)/1e6,e2e_wall_ms=wall,
        prefill_forward_ms=(prefill_end-start)/1e6,decode_and_intertoken_total_ms=(availability[-1]-first)/1e6,
        finalization_ms=(end-availability[-1])/1e6,intertoken_ms=gaps,generated_tokens_per_second=n*1000/wall,
        sequential_requests_per_second=1000/wall,cache_lookup_ms=lookup_ms,cold_insert_ms=insert_ms,
        cache_status='HIT' if entry is not None else 'MISS' if mode=='COLD_MISS' else 'BYPASS',
        prefill_reused_qkv_outputs=96 if entry is not None else 0,full_request_hit=int(entry is not None),
        reused_prompt_tokens=len(ids) if entry is not None else 0,cache_validation_read_bytes=entry.size_bytes if entry is not None else 0,
        reused_nonpadding_prompt_tokens=sum(prompt['attention_mask']) if entry is not None else 0,
        prefill_accounting=prefill_accounting,decode_accounting=decode_accounting,
        total_accounting={k:prefill_accounting[k]+decode_accounting[k] for k in fields},
        cache_bytes_before=cache_before,cache_bytes_after=0 if router is None or router.active is None else router.active.bytes,
        cache_capacity_bytes=0 if router is None else router.capacity_bytes,cache_admitted=mode=='COLD_MISS',
        cache_source_request_id=entry.qkv_metadata['source_request_id'] if entry is not None else request_id if mode=='COLD_MISS' else None,
        cache_context_sha256=None if cache is None else cache.context_digest,
        cache_events=[] if cache is None else list(cache.events[event_start:]),steps=steps,logits_blob=bytes(logits),
        logits_sha256=digest_bytes(bytes(logits)),logits_dtype='float32',logits_shape=[1,n,driver.vocab],
        timing_scope='synchronized instrumented serving wall; evidence copy/hash/selection inside; nested timers not additive',
        eos_ignored=policy['ignore_eos'],in_request_kv_enabled=True,decode_forward_count=n-1)
