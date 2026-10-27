"""Independent native OPT/PEFT exact-prefix K/V baseline, no SemCache Engine."""
from dataclasses import asdict,replace
from time import perf_counter_ns
from contextlib import nullcontext
import hashlib
import inspect
from pathlib import Path

from semcache.diagnostics.r2b1_prefix_cache import (PrefixError,PrefixIdentity,PrefixSnapshot,OPTCacheAPI,
    cache_pairs,validate_pairs,validate_request,tensor_hash)
from semcache.diagnostics.r2b1_prefix_audit import geometry,ForwardAudit
from semcache.metrics.r1 import digest
from semcache.diagnostics.r1b3_fp16_divergence import logit_metrics


class PrefixRuntime:
    def __init__(self,args):
        import torch,transformers,peft
        from semcache.models.loader import load_model
        from semcache.models.lora_fixtures import create_controlled_users
        from semcache.utils.seed import seed_everything
        seed_everything(args.seed)
        if args.device.startswith('cuda'):
            if not torch.cuda.is_available(): raise PrefixError('UNVERIFIED_CUDA','CUDA unavailable')
            free,total=torch.cuda.mem_get_info(args.device)
            params=2_700_000_000 if args.model.endswith('2.7b') else 125_000_000
            estimate=params*(2 if args.dtype=='float16' else 4)+3*1024**3
            if free<estimate: raise MemoryError(f'MEMORY_PREFLIGHT_FAILED: model screen required={estimate}, free={free}')
        started=perf_counter_ns()
        model,tokenizer,meta=load_model(dict(name=args.model,tokenizer=args.model,revision=args.revision,
            tokenizer_revision=args.tokenizer_revision or args.revision,dtype=args.dtype,device=args.device,
            attention_implementation='eager',local_files_only=not args.allow_download))
        if getattr(model,'is_quantized',False) or getattr(model.config,'quantization_config',None):
            raise PrefixError('UNSUPPORTED_API','unquantized native OPT required')
        model,fixture=create_controlled_users(model)
        from semcache.models.tokenizer_provenance import is_snapshot_commit,tokenizer_artifact_fields,validate_tokenizer_snapshot
        if not is_snapshot_commit(meta.get('resolved_model_revision')) or not is_snapshot_commit(meta.get('resolved_tokenizer_revision')):
            raise PrefixError('UNVERIFIED_IDENTITY','resolved model/tokenizer commit hashes required')
        validate_tokenizer_snapshot(dict(model_id=meta['model'],model_revision=meta['resolved_model_revision'],**tokenizer_artifact_fields(meta)))
        model.set_adapter(args.adapter); model.eval()
        self.initialize(model,tokenizer,meta,fixture,args)
        self.provenance['initialization_ms']=(perf_counter_ns()-started)/1e6

    def initialize(self,model,tokenizer,meta,fixture,args):
        """Dependency injection for small random OPT tests; no download or fixture substitution in CLI."""
        import torch
        from semcache.models.model_adapter import OPTModelAdapter
        self.model=model; self.tokenizer=tokenizer; self.meta=meta; self.fixture=fixture; self.a=args
        self.adapter=OPTModelAdapter(model); self.base=model.get_base_model() if hasattr(model,'peft_config') else model
        self.decoder=self.base.model.decoder
        self.device=str(next(model.parameters()).device)
        if args.cache_device is not None and str(torch.device(args.cache_device))!=self.device:
            raise PrefixError('REJECT_DEVICE_MISMATCH','B1 cache must remain on model device; CPU transfer deferred')
        if getattr(model.config,'_attn_implementation',None)!='eager': raise PrefixError('REJECT_BACKEND_MISMATCH','eager required')
        self.api=OPTCacheAPI(model,self.decoder)
        self.heads=model.config.num_attention_heads; self.head_dim=model.config.hidden_size//self.heads
        self.source_forwards=0; self.snapshot=None
        self.weight_hashes={name:tensor_hash(p) for name,p in model.named_parameters()}
        self.base_hash=digest({name:h for name,h in self.weight_hashes.items() if 'lora_' not in name})
        self.adapter_hashes={user:digest({name:h for name,h in self.weight_hashes.items() if 'lora_' in name and f'.{user}.' in name}) for user in ('user_a','user_b')}
        if self.adapter_hashes['user_a']==self.adapter_hashes['user_b']:
            raise PrefixError('UNVERIFIED_ADAPTER','controlled user adapters must have distinct actual weights')
        self.stamps=self.model_stamps()
        self.adapter_configs={user:self.adapter_configuration(user) for user in ('user_a','user_b')}
        import transformers,peft
        self.provenance=dict(model=meta,fixture=fixture,api=self.api.metadata,torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
            cuda=torch.version.cuda,device=self.device,gpu_name=torch.cuda.get_device_name(self.device) if self.device.startswith('cuda') else None,
            gpu=self.gpu_state(),base_weight_identity=self.base_hash,adapter_weights=self.adapter_hashes,weight_hashes=self.weight_hashes,
            source_and_target_adapter=args.adapter,training=model.training,dropout=[dict(name=n,p=m.p,training=m.training) for n,m in model.named_modules() if isinstance(m,torch.nn.Dropout)],
            attention='eager',use_cache=True,output_attentions=True,return_dict=True,position_embedding_class=type(self.decoder.embed_positions).__name__,
            float32_matmul_precision=torch.get_float32_matmul_precision(),tf32=torch.backends.cuda.matmul.allow_tf32,
            dtype=str(next(model.parameters()).dtype),tokenizer_default_special_ids=tokenizer.all_special_ids)
        for name,obj in [('OPTForward',self.base.forward),('decoder',self.decoder.forward),('positions',self.decoder.embed_positions.forward)]:
            path=inspect.getsourcefile(obj)
            self.provenance.setdefault('installed_api_source',{})[name]=dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None)

    def model_stamps(self):
        from semcache.diagnostics.r1b3_fast_selection import tensor_stamp
        return tuple((name,id(p),tensor_stamp(p)) for name,p in self.model.named_parameters())

    def adapter_configuration(self,user):
        return [dict(layer=i,role=role,rank=m.r[user],alpha=m.lora_alpha[user],scaling=m.scaling[user],
            dropout=getattr(m.lora_dropout[user],'p',0.),merged=m.merged,disabled=m.disable_adapters,
            use_dora=m.use_dora.get(user,False)) for i in range(len(self.adapter.layers)) for role,m in self.adapter.projection_modules(i).items()]

    def identity(self,user=None):
        user=user or self.a.adapter
        if list(self.model.active_adapters)!=[user]: raise PrefixError('REJECT_ADAPTER_MISMATCH','active PEFT adapter differs')
        return PrefixIdentity(self.meta['model'],self.meta['resolved_model_revision'],self.base_hash,
            self.meta['tokenizer'],self.meta['resolved_tokenizer_revision'],user,self.adapter_hashes[user],
            str(next(self.model.parameters()).dtype),self.device,adapter_configuration=digest(self.adapter_configuration(user)))

    def check(self):
        if self.model.training or any(m.training for m in self.model.modules()): raise PrefixError('REJECT_NAMESPACE_MISMATCH','eval required')
        if self.model_stamps()!=self.stamps: raise PrefixError('REJECT_MODEL_WEIGHT_MUTATION','model tensor changed')
        if getattr(self.model.config,'_attn_implementation',None)!='eager': raise PrefixError('REJECT_BACKEND_MISMATCH','backend changed')
        self.identity()
        if any(self.adapter_configuration(user)!=self.adapter_configs[user] for user in self.adapter_configs):
            raise PrefixError('REJECT_ADAPTER_CONFIG_MISMATCH','LoRA scale/rank/dropout changed')
        return self.snapshot.check() if self.snapshot else dict(status='NO_PREFIX_CONTROL')

    def gpu_state(self):
        import subprocess
        import torch
        state=dict(gpu_uuid=None,driver=None)
        try:
            if not self.device.startswith('cuda'): raise RuntimeError('CPU experiment')
            uuid=str(getattr(torch.cuda.get_device_properties(self.device),'uuid',None))
            if uuid=='None': raise RuntimeError('Torch UUID unavailable; no guessed physical GPU')
            r=subprocess.run(['nvidia-smi','--id='+uuid,'--query-gpu=uuid,driver_version,memory.used,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
            state.update(gpu_uuid=uuid,driver=r.stdout.strip())
        except Exception as exc: state['reason']=str(exc)
        return state

    def prepare(self,length,prefix):
        from semcache.metrics.scaling_workload import construct_natural_prompt
        trace=construct_natural_prompt(self.tokenizer,length)
        if len(trace['token_ids'])!=length or list(self.tokenizer(trace['text'])['input_ids'])!=trace['token_ids']:
            raise PrefixError('REJECT_TOKEN_LENGTH','tokenizer output must match requested length exactly')
        self.ids=list(trace['token_ids']); self.text=trace['text']; self.p=prefix; self.n=length
        if any(type(v) is not int or not 0<=v<self.model.config.vocab_size for v in self.ids):
            raise PrefixError('REJECT_TOKEN_IDS','tokenizer emitted invalid model vocabulary IDs')
        self.geometry=geometry(length,prefix,self.model.config.max_position_embeddings)
        self.snapshot=None; self.source_forwards=0; self.source_report=None
        from semcache.metrics.r1_runtime import memory_preflight
        # Full output cache + canonical P + request P + growing target L coexist.
        kv_bytes=2*len(self.adapter.layers)*length*self.model.config.hidden_size*next(self.model.parameters()).element_size()
        self.preflight=memory_preflight(self.model,length,4*kv_bytes,self.device,self.device)
        self.check()
        if prefix:
            source=self.forward('PREFIX_SOURCE_PREFILL',audit=False)
            from semcache.metrics.r1_runtime import synchronize
            synchronize(self.device); started=perf_counter_ns()
            self.snapshot=PrefixSnapshot(source['cache'],self.identity(),self.ids[:prefix],layers=len(self.adapter.layers),
                heads=self.heads,head_dim=self.head_dim,source_path='PREFIX_SOURCE_PREFILL')
            synchronize(self.device); snapshot_ms=(perf_counter_ns()-started)/1e6
            audit_source=self.forward('PREFIX_SOURCE_PREFILL',audit=True)
            repeat_cache=self.snapshot.inspect_output(audit_source['cache'],prefix)
            self.source_report=dict(prefix_prefill_ms=source['wall_ms'],diagnostic_prefix_prefill_ms=audit_source['wall_ms'],
                source_repeat_cache=repeat_cache,source_repeat_logits=logit_metrics(source['logits'],audit_source['logits']),
                snapshot_ms=snapshot_ms,cache_bytes=self.snapshot.bytes,
                source_identity=self.snapshot.identity_hash,source_input_ids=self.ids[:prefix],source_forward_audit=audit_source['audit'],
                snapshot_mode=self.snapshot.mode,native_cache_class=f'{type(source["cache"]).__module__}.{type(source["cache"]).__qualname__}',
                source_prefill_in_warm=False)
            self.source_logits=source['logits']; self.check()
            del source,audit_source
        else:
            self.source_logits=None; self.source_report=dict(status='NO_PREFIX_CONTROL',prefix_prefill_ms=None,snapshot_ms=None,cache_bytes=0)

    def forward(self,path,audit=False):
        import torch
        from semcache.metrics.r1_runtime import synchronize,memory_snapshot
        if path not in ('FULL_USE_CACHE','PREFIX_SOURCE_PREFILL','EXACT_PREFIX_WARM'): raise ValueError(path)
        is_source=path=='PREFIX_SOURCE_PREFILL'
        p=self.p if path=='EXACT_PREFIX_WARM' else 0
        ids=self.ids[:self.p] if is_source else self.ids[p:]
        total=len(ids)+p
        if not ids: raise PrefixError('REJECT_LENGTH','no empty forward')
        if p and self.snapshot is None: raise PrefixError('REJECT_MISSING_CACHE','Source Prefix must complete before Target')
        observer=ForwardAudit(self.model,self.adapter,self.decoder,len(ids),p,total,path,ids) if audit else None
        # Hooks exist only for an untimed diagnostic pass. Minimal passes have none.
        with observer.hooks() if observer else nullcontext():
            synchronize(self.device); start=perf_counter_ns()
            self.model.set_adapter(self.a.adapter); self.model.eval()
            kwargs=self.api.kwargs(ids,p,total,self.device)
            request_cache=None; prep_ms=None; prepared=None; input_before_lengths=[]
            if p:
                started=perf_counter_ns()
                request_cache=self.snapshot.materialize(self.identity(),self.ids,position_start=p,mask=[1]*total)
                kwargs['past_key_values']=request_cache
                input_before_lengths=[int(pair[0].shape[2]) for pair in cache_pairs(request_cache)[0]]
                prep_ms=(perf_counter_ns()-started)/1e6
                if audit: prepared=self.snapshot.inspect_request(request_cache)
            with torch.inference_mode():
                if is_source: self.source_forwards+=1
                out=self.model(**kwargs)
            synchronize(self.device); wall=(perf_counter_ns()-start)/1e6
        # All cache shape/content checks, logits copies, and quality are OUTSIDE timer.
        if tuple(out.logits.shape)!=(1,len(ids),self.model.config.vocab_size): raise PrefixError('REJECT_LOGIT_SHAPE','all requested positions must be returned')
        pairs,api,_=cache_pairs(out.past_key_values)
        rows=validate_pairs(pairs,layers=len(self.adapter.layers),length=total,heads=self.heads,head_dim=self.head_dim,
            dtype=str(next(self.model.parameters()).dtype),device=self.device)
        if out.attentions is None or len(out.attentions)!=len(self.adapter.layers) or any(tuple(a.shape)!=(1,self.heads,len(ids),total) for a in out.attentions):
            raise PrefixError('MISSING_FORWARD_EVIDENCE','actual suffix attention must see total cached+fresh KV context')
        past_check=self.snapshot.inspect_output(out.past_key_values,total) if p else None
        input_lengths=[int(pair[0].shape[2]) for pair in cache_pairs(request_cache)[0]] if request_cache is not None else []
        result=dict(logits=out.logits.detach().cpu(),cache=out.past_key_values,wall_ms=wall,cache_prepare_host_ms=prep_ms,
            cache_prepare_scope='host enqueue + container construction; async clone completion included in synchronized request wall',
            request_clone_bytes=self.snapshot.bytes if p else 0,query_tokens=len(ids),past_tokens=p,output_kv_length=total,
            cache_api=api,actual_cache_class=f'{type(out.past_key_values).__module__}.{type(out.past_key_values).__qualname__}',
            request_cache_class=f'{type(request_cache).__module__}.{type(request_cache).__qualname__}' if request_cache is not None else None,
            materialization_mode=self.snapshot.mode if p else 'NO_INCOMING_CACHE',cache_rows=rows,request_input_before_lengths=input_before_lengths,request_input_after_lengths=input_lengths,
            request_container_grew=bool(p and any(length!=p for length in input_lengths)),
            canonical_source_length=self.snapshot.length if self.snapshot else None,output_prefix_check=past_check,
            prepared_input_audit=prepared,audit=observer.report() if observer else None,memory=memory_snapshot(self.device),
            cache_hit=bool(p),instrumented=audit)
        return result

    def verify_weight_hashes(self):
        current={name:tensor_hash(p) for name,p in self.model.named_parameters()}
        if current!=self.weight_hashes: raise PrefixError('REJECT_MODEL_WEIGHT_MUTATION','final actual weight hashes changed')
        return dict(status='UNCHANGED',base_weights=self.base_hash,adapter_weights=self.adapter_hashes)

    def negative_tests(self):
        if not self.snapshot: return [dict(status='NOT_APPLICABLE',reason='P=0 native control')]
        identity=self.identity(); results=[]
        def attempt(name,fn,kind):
            try: fn()
            except PrefixError as exc: results.append(dict(test=name,kind=kind,status='REJECTED',code=exc.code)); return
            raise PrefixError('NEGATIVE_TEST_FAILED',name+' was accepted')
        changed=list(self.ids); changed[0]=(changed[0]+1)%self.model.config.vocab_size
        attempt('changed_prefix_token',lambda:validate_request(identity,identity,self.ids[:self.p],changed), 'ACTUAL_TOKEN_IDS_NO_FORWARD')
        self.model.set_adapter('user_b' if self.a.adapter=='user_a' else 'user_a')
        try:
            other=self.identity('user_b' if self.a.adapter=='user_a' else 'user_a')
            attempt('other_active_adapter',lambda:validate_request(identity,other,self.ids[:self.p],self.ids),'ACTUAL_ADAPTER_SWITCH_REJECTED_BEFORE_FORWARD')
        finally: self.model.set_adapter(self.a.adapter)
        for field,value in [('model_id','other'),('model_revision','other'),('base_weights','other'),('tokenizer_revision','other'),
            ('adapter_weights','other'),('adapter_configuration','other'),('dtype','other'),('device','other'),('backend','sdpa'),('position_policy','other'),('mask_policy','other')]:
            attempt(field,lambda f=field,v=value:validate_request(identity,replace(identity,**{f:v}),self.ids[:self.p],self.ids),'METADATA_ONLY_NEGATIVE')
        attempt('substring_source',lambda:validate_request(identity,identity,self.ids[:self.p],self.ids,source_start=1),'METADATA_ONLY_NEGATIVE')
        attempt('wrong_position',lambda:validate_request(identity,identity,self.ids[:self.p],self.ids,position_start=0),'TARGET_ARGUMENT_REJECTION')
        attempt('suffix_only_mask',lambda:validate_request(identity,identity,self.ids[:self.p],self.ids,mask=[1]*(self.n-self.p)),'TARGET_ARGUMENT_REJECTION')
        return results
