"""Separate sparse R5 autoregressive lifecycle; old Exact Prefix driver untouched."""
from contextlib import contextmanager,nullcontext
import time
from .communication import require,digest,digest_bytes
from .r3a_artifacts import pack
from .r4c_workload import termination
from .r5_sparse_qkv import SparseSession,finish_request
from .r5_subsequence_physical import projection_path,EXACT
from semcache.cache.metric_manager import actual_attention_impact
from .r5_subsequence_profile import CURRENT,call

class StreamingImpact:
    """Original M reducer per span/layer; no all-layer probability retention.

    Matrix bytes can be streamed to evidence for independent CPU reduction.
    Removing returned telemetry never changes attn_output or native arithmetic.
    """
    def __init__(self,req,layers,windows,*,evidence=None,max_layer_bytes=128*2**20):
        require(0<max_layer_bytes<=128*2**20,'bounded layer probability budget required')
        self.req=req;self.layers=layers;self.windows=windows;self.evidence=evidence;self.max_layer_bytes=max_layer_bytes
        self.observed=[];self.values={w.start:0. for w in windows};self.records=[];self.elapsed_ms=0.;self.peak_bytes=0;self.peak_working_estimate=0
    def observe(self,layer,prob):
        require(layer==len(self.observed) and layer<self.layers,'attention layer order differs')
        n=len(self.req['token_ids']);require(prob is not None and len(prob.shape)==4 and prob.shape[0]==1 and tuple(prob.shape[2:])==(n,n),'actual square prefill probabilities required')
        size=prob.numel()*prob.element_size();require(size<=self.max_layer_bytes,'single-layer attention budget exceeded')
        working_bytes=size+prob.numel()*8+prob.shape[1]*3*n*8
        require(working_bytes<=2**30,'single-layer impact working set exceeds1GiB estimate')
        start=time.monotonic_ns();double=call('impact_fp64_conversion',lambda:prob.detach().double(),cuda=True)
        # Same M reducer and exact FP64 inputs; avoid recasting the entire matrix
        # once per window. No live native operand is modified or retained.
        values=call('actual_attention_impact_reduction',lambda:{w.start:actual_attention_impact([double],w.start,w.end) for w in self.windows},cuda=True)
        for k,v in values.items():self.values[k]+=v
        item=None if self.evidence is None else call('impact_evidence_save_inclusive',self.evidence.save,prob,self.req['request_id']+f'_impact_l{layer}')
        self.elapsed_ms+=(time.monotonic_ns()-start)/1e6;self.peak_bytes=max(self.peak_bytes,size)
        self.peak_working_estimate=max(self.peak_working_estimate,working_bytes)
        self.records.append(dict(layer=layer,probabilities=item,span_values=values));self.observed.append(layer)
    def values_for(self,active):
        require(active['req']==self.req and self.observed==list(range(self.layers)) and set(self.values)=={w.start for w in active['windows']},'incomplete/mismatched actual impact')
        return self.values
    @contextmanager
    def hooks(self,adapter):
        handles=[]
        try:
            for layer,module in enumerate(adapter.layers):
                def hook(m,args,out,layer=layer):
                    require(isinstance(out,tuple) and len(out)>=2,'unsupported OPTAttention return')
                    from contextlib import nullcontext
                    p=CURRENT.get()
                    with (p.location(layer,'','PREFILL_IMPACT') if p else nullcontext()):self.observe(layer,out[1])
                    return (out[0],None,*out[2:])
                handles.append(module.self_attn.register_forward_hook(hook))
            yield
        finally:
            for h in handles:h.remove()

@contextmanager
def fresh_path(adapter,session,records,evidence=None,tag='fresh'):
    """All-fresh prefill or single-row decode; CPU LoRA stays on UD."""
    import torch
    originals=[]
    try:
        for layer in range(len(adapter.layers)):
            for role,module in adapter.projection_modules(layer).items():
                require(type(module) is torch.nn.Linear and 'forward' not in module.__dict__ and not module._forward_hooks and not module._forward_pre_hooks,'bare untouched Linear required')
                native=module.forward;originals.append(module)
                def compute(x,*args,_l=layer,_r=role,_native=native,**kwargs):
                    require(not args and not kwargs,'unsupported projection call')
                    delta=call('sparse_tcp_correction_inclusive',session.correction,_l,_r,x,list(range(x.shape[1])));base=call('native_dense_linear',_native,x,cuda=True);out=call('delta_add_cast',lambda:(base+delta).to(base.dtype),cuda=True)
                    record=dict(layer=_l,role=_r,rows=x.shape[1])
                    if evidence is not None:
                        record.update(input=call('evidence_save_inclusive',evidence.save,x,tag+f'_l{_l}_{_r}_input'),output=call('evidence_save_inclusive',evidence.save,out,tag+f'_l{_l}_{_r}_output'),fresh_base=call('evidence_save_inclusive',evidence.save,base,tag+f'_l{_l}_{_r}_base'),fresh_delta=call('evidence_save_inclusive',evidence.save,delta,tag+f'_l{_l}_{_r}_delta'))
                    records.append(record);return out
                def forward(x,*args,_compute=compute,_l=layer,_r=role,**kwargs):
                    p=CURRENT.get()
                    with (p.location(_l,_r,'DECODE' if session.generated else 'PREFILL') if p else nullcontext()):return _compute(x,*args,**kwargs)
                module.forward=forward
        yield;session.complete()
    finally:
        for m in originals:del m.forward

def generate_loop(req,forward,complete,decode_text,*,synchronize=lambda:None):
    """Shared frozen greedy policy/termination; CPU fixtures inject forward."""
    begin=time.monotonic_ns();tokens=[];steps=[];blobs=[];past=None;prefill_end=None
    for step in range(req['generation_policy']['max_new_tokens']):
        tick=time.monotonic_ns();values,past,record=forward(step,list(tokens),past)
        require(values and all(__import__('math').isfinite(x) for x in values),'empty/nonfinite full-vocabulary logits')
        token=max(range(len(values)),key=values.__getitem__);tokens.append(token);blobs.append(pack(values,'float32'));synchronize();now=time.monotonic_ns()
        if step==0:prefill_end=now
        steps.append(dict(step=step,selected_token_id=token,logits_sha256=digest_bytes(blobs[-1]),available_ns=now,wall_ms=(now-tick)/1e6,record=record))
        reason=termination(req['generation_policy'],tokens)
        if reason is not None:break
    require(reason is not None and tokens,'generation did not complete')
    text=decode_text(tokens);model_end=time.monotonic_ns();completion=complete(tokens);synchronize();end=time.monotonic_ns()
    return dict(status='COMPLETE',request_id=req['request_id'],generated_token_ids=tokens,generated_text=text,termination=reason,
                logits_blob=b''.join(blobs),logits_shape=[1,len(tokens),len(values)],steps=steps,completion=completion,
                arrival_ns=begin,completion_ns=end,application_e2e_ms=(end-begin)/1e6,model_execution_e2e_ms=(model_end-begin)/1e6,
                application_ttft_ms=(steps[0]['available_ns']-begin)/1e6,model_ttft_ms=(steps[0]['available_ns']-begin)/1e6,prefill_ms=(prefill_end-begin)/1e6,decode_ms=sum(s['wall_ms'] for s in steps[1:]),admission_completion_ms=(end-model_end)/1e6)

def generate(model,adapter,tokenizer,req,channel,owner,index,*,condition,mode=EXACT,allow_approximate=False,evidence=None,profiler=None):
    if profiler is None:return _generate(model,adapter,tokenizer,req,channel,owner,index,condition=condition,mode=mode,allow_approximate=allow_approximate,evidence=evidence)
    try:
        with profiler.activate(req['request_id']):
            result=_generate(model,adapter,tokenizer,req,channel,owner,index,condition=condition,mode=mode,allow_approximate=allow_approximate,evidence=evidence)
    except BaseException as primary:
        if evidence is not None:
            try:
                import json
                record=dict(status='FAILED',reason=repr(primary),profile=profiler.finish());data=json.dumps(record,sort_keys=True).encode()
                require(len(data)<=8*2**20,'failed profile metadata exceeds8MiB')
                evidence.blob(data,req['request_id']+'_FAILED_profile.json')
            except BaseException as exc:
                if hasattr(primary,'add_note'):primary.add_note('failed profile export: '+repr(exc))
        raise
    result['diagnostics']=profiler.finish()
    end=time.monotonic_ns();result['completion_ns']=end;result['application_e2e_ms']=(end-result['arrival_ns'])/1e6
    return result

def _generate(model,adapter,tokenizer,req,channel,owner,index,*,condition,mode=EXACT,allow_approximate=False,evidence=None):
    """Native full prefill + native cached autoregressive decode over sparse UD.

    Requires caller's existing R5 model/adapter/GPU preflight and fresh sessions.
    Sparse reuse never supplies past KV; decode past is this target's actual output.
    """
    import torch
    require(condition in ('FULL_RECOMPUTE','SEMANTIC_NO_REUSE','SEMCACHE_SUBSEQUENCE'),'unknown sparse condition')
    require(not torch.is_autocast_enabled(),'undeclared autocast is unsupported')
    require(not model.training and all(type(l.self_attn).__name__=='OPTAttention' for l in adapter.layers),'inspected eager eval OPT required')
    device=next(model.parameters()).device;semantic_start=time.monotonic_ns()
    assignment=None if condition=='FULL_RECOMPUTE' else call('semantic_assignment_inclusive',index.assign,req['input_text'])
    semantic_ms=(time.monotonic_ns()-semantic_start)/1e6
    match_start=time.monotonic_ns();match_report=None
    if assignment is not None:match_report=owner.begin(req,assignment,mode=mode,allow_approximate=allow_approximate,reuse=condition=='SEMCACHE_SUBSEQUENCE')
    matching_policy_setup_ms=(time.monotonic_ns()-match_start)/1e6
    impact=None if assignment is None else StreamingImpact(req,len(adapter.layers),owner.active['windows'],evidence=evidence)
    receipts=[];prefill_records=[]
    def forward(step,generated,past):
        nonlocal prefill_records
        if evidence is not None:evidence.check()
        session=SparseSession(channel,req,req['request_id']+f'_s{step:03d}',generated=generated)
        inputs=req['token_ids'] if not step else generated[-1:];mask=[1]*(len(req['token_ids'])+step)
        from contextlib import nullcontext
        from .r4a_diagnostics import observe_opt_positions
        from .r4a_generation import past_summary
        import inspect
        old=len(req['token_ids'])+step-1 if step else 0
        kwargs=dict(input_ids=torch.tensor([inputs],device=device),attention_mask=torch.tensor([mask],device=device),past_key_values=past,use_cache=True,output_attentions=bool(not step and impact is not None),return_dict=True)
        parameters=inspect.signature(model.forward).parameters
        if 'position_ids' in parameters:kwargs['position_ids']=torch.tensor([list(range(old,old+len(inputs)))],device=device)
        if 'cache_position' in parameters:kwargs['cache_position']=torch.arange(old,old+len(inputs),device=device)
        before=None if past is None else past_summary(past,old,len(adapter.layers),model.config.num_attention_heads,req['physical_identity']['hidden_size'],req['physical_identity']['dtype'],str(device))
        pass_records=prefill_records if not step else []
        path=projection_path(adapter,owner,session) if not step and assignment is not None else fresh_path(adapter,session,pass_records,evidence,req['request_id']+f'_s{step}')
        with torch.inference_mode(),(impact.hooks(adapter) if not step and impact is not None else nullcontext()),path,observe_opt_positions(model.model.decoder.embed_positions,mask,old) as observed:
            p=CURRENT.get()
            with (p.location(-1,'','PREFILL' if not step else 'DECODE') if p else nullcontext()):out=call('native_model_forward_inclusive',model,**kwargs)
        receipts.extend(session.receipts)
        values=out.logits[0,-1].float().cpu().tolist()
        after=past_summary(out.past_key_values,old+len(inputs),len(adapter.layers),model.config.num_attention_heads,req['physical_identity']['hidden_size'],req['physical_identity']['dtype'],str(device))
        return values,out.past_key_values,dict(sparse=session.receipts,fresh_projection_records=pass_records,position_observation=observed[0],input_past=before,past=after)
    def complete(tokens):
        p=CURRENT.get()
        with (p.location(-1,'','COMPLETION') if p else nullcontext()):
            control=call('completion_tcp_acknowledgement',finish_request,channel,req,tokens)
            physical=None if assignment is None else owner.complete(completed=True,attentions=impact)
        if evidence is not None:evidence.check()
        return dict(control=control,physical=physical,impact=None if impact is None else dict(records=impact.records,elapsed_ms=impact.elapsed_ms,peak_probability_bytes=impact.peak_bytes,peak_working_estimate_bytes=impact.peak_working_estimate))
    result=generate_loop(req,forward,complete,lambda tokens:tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False),synchronize=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None)
    # Admission/semantic preparation are part of the application scope, not free.
    result['application_e2e_ms']=(result['completion_ns']-semantic_start)/1e6;result['application_ttft_ms']=(result['steps'][0]['available_ns']-semantic_start)/1e6;result['arrival_ns']=semantic_start
    result.update(condition=condition,context_mode=mode,assignment=assignment,semantic_policy_setup_ms=semantic_ms,matching_policy_setup_ms=matching_policy_setup_ms,matching=match_report,sparse_receipts=receipts,prefill_records=prefill_records,correctness_qualified_speedup='NOT_ESTABLISHED')
    physical=result['completion']['physical']
    result['cached_q_consumed']=False if physical is None else any(r['role']=='q' and r['reused_positions'] for r in physical['records'])
    result['generated_tokens_per_second']=len(result['generated_token_ids'])/(result['application_e2e_ms']/1000)
    return result
