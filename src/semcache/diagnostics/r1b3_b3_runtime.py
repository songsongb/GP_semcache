"""B3 orchestration only; all selection/executor implementations are reused unchanged."""
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.util
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch
from semcache.metrics.r1 import digest, audit_projection_records
from semcache.diagnostics.r1b3_break_even import MissingEvidence, fallback_filter
ROOT=Path(__file__).resolve().parents[3]


def b21():
    spec=importlib.util.spec_from_file_location('b3_b21_reference',ROOT/'scripts/77_benchmark_r1b3_b21_setup.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def common_arguments(parser):
    parser.add_argument('--model',choices=('facebook/opt-125m','facebook/opt-2.7b'),default='facebook/opt-125m')
    parser.add_argument('--dtype',choices=('float16','float32'),default='float32')
    parser.add_argument('--device',default='cuda:0'); parser.add_argument('--cache-device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--revision'); parser.add_argument('--tokenizer-revision'); parser.add_argument('--encoder-revision',default='main')
    parser.add_argument('--seed',type=int,default=42); parser.add_argument('--allow-download',action='store_true')
    parser.add_argument('--prompt-lengths',default='32'); parser.add_argument('--capacity-mib',type=int,default=256)
    parser.add_argument('--independent-run-id',default='run01')
    parser.add_argument('--output-dir',type=Path,required=True)


def config(args): return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='output_dir'}


class Runtime:
    def __init__(self,args):
        import torch
        from semcache.models.loader import load_model
        from semcache.models.lora_fixtures import create_controlled_users
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.semantic.encoder import TinyBERTSemanticEncoder
        from semcache.utils.seed import seed_everything
        self.a=args; self.b=b21(); self.clone,self.a3=self.b.runtime_helpers()
        seed_everything(args.seed)
        if args.device.startswith('cuda'):
            if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable: model/GPU experiment NOT EXECUTED')
            free,_=torch.cuda.mem_get_info(args.device)
            params=2_700_000_000 if args.model.endswith('2.7b') else 125_000_000
            estimated=params*(4 if args.dtype=='float32' else 2)+3*1024**3
            if free<estimated: raise MemoryError(f'Pre-load model screen: {estimated} required, {free} free; estimate only')
        self.model,self.tokenizer,self.meta=load_model(dict(name=args.model,tokenizer=args.model,revision=args.revision,
            tokenizer_revision=args.tokenizer_revision or args.revision,dtype=args.dtype,device=args.device,
            local_files_only=not args.allow_download,attention_implementation='eager'))
        if getattr(self.model,'is_quantized',False) or getattr(self.model.config,'quantization_config',None): raise ValueError('RAW-only model required')
        self.model,self.lora=create_controlled_users(self.model); self.adapter=OPTModelAdapter(self.model)
        self.encoder=TinyBERTSemanticEncoder(device=args.device,dtype=args.dtype,revision=args.encoder_revision,local_files_only=not args.allow_download)
        self.namespace=digest({k:v for k,v in self.meta.items() if k not in ('device','dtype','torch_version','transformers_version')})
        hashes={n:hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for n,p in self.model.named_parameters() if 'lora_' in n}
        self.weights=digest(dict(revision=self.meta.get('resolved_model_revision'),adapter_weights=hashes))
        if not self.meta.get('resolved_model_revision'): raise MissingEvidence('Resolved model revision required')
        self.layout={i:tuple((m.out_features,m.get_base_layer().weight.dtype) for m in self.adapter.projection_modules(i).values()) for i in range(len(self.adapter.layers))}
        self.original=__import__('semcache.edgelora.mixed_projection',fromlist=['mixed_projection_path']).mixed_projection_path
        self.storage=str(next(self.model.parameters()).device) if args.cache_device=='cuda' else 'cpu'
        self.provenance=dict(model=self.meta,adapter_fixture=self.lora,adapter_weight_hashes=hashes,weight_identity=self.weights,
            namespace=self.namespace,torch=torch.__version__,cuda=torch.version.cuda,encoder=self.encoder.metadata,
            gpu=self.gpu_state(),eval=not self.model.training,
            dropout=[dict(name=n,p=m.p,training=m.training) for n,m in self.model.named_modules() if isinstance(m,torch.nn.Dropout)],
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)

    def gpu_state(self):
        """Query the actual Torch device UUID, never silently assume physical GPU 0."""
        import subprocess
        from datetime import datetime,timezone
        state=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),gpu_uuid=None,utilization_percent=None,
            memory_used_mib=None,sm_clock_mhz=None,temperature_c=None,power_w=None,other_processes=None)
        try:
            import torch
            if not self.a.device.startswith('cuda'): raise RuntimeError('CPU experiment; no GPU identity')
            properties=torch.cuda.get_device_properties(self.a.device)
            uuid=getattr(properties,'uuid',None)
            if uuid is None: raise RuntimeError('Torch does not expose device UUID; monitoring not assigned to guessed GPU')
            uuid=str(uuid); state['gpu_uuid']=uuid
            query='uuid,utilization.gpu,memory.used,clocks.sm,temperature.gpu,power.draw'
            result=subprocess.run(['nvidia-smi','--id='+uuid,'--query-gpu='+query,'--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
            fields=[x.strip() for x in result.stdout.strip().split(',')]
            if fields[0]!=uuid: raise RuntimeError('nvidia-smi UUID differs from Torch device')
            state.update(dict(zip(('gpu_uuid','utilization_percent','memory_used_mib','sm_clock_mhz','temperature_c','power_w'),fields)))
            processes=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
            state['other_processes']=[line for line in processes.stdout.splitlines() if line.startswith(uuid)]
            state['scope']='measurement block boundary; outside requests; process list includes own PID'
        except Exception as exc: state['reason']=str(exc)
        return state

    def source(self,length,observer=None):
        from semcache.metrics.scaling_workload import controlled_length_trace
        from semcache.semantic.intent_clusterer import IntentClusterer
        from semcache.cache.global_cache import GlobalCache
        from semcache.semcache_engine import SemCacheEngine
        from semcache.diagnostics.r1b3_occurrences import OccurrenceCatalog,capture_completed_source
        from semcache.diagnostics.r1b3_fast_selection import FastOccurrenceIndex
        from semcache.edgelora.batched_mapping_optimized import SourceBackingDescriptors,PreparedMappingCache
        from semcache.metrics import r1_runtime as rt
        trace=controlled_length_trace(self.tokenizer,length); fixture=trace[1]
        self.ids=list(fixture['token_ids']); self.text=fixture['text']; self.length=length
        if len(self.ids)!=length or list(self.tokenizer(self.text)['input_ids'])!=self.ids: raise MissingEvidence('Source tokenizer length/identity mismatch')
        self.preflight=rt.memory_preflight(self.model,length,2*self.a.capacity_mib*1024**2,self.storage,self.a.device)
        cluster=IntentClusterer(2,initialization='first_k',update_mode='buffered',update_interval=100)
        cluster.initialize(self.encoder.encode([trace[0]['text'],trace[-1]['text']]))
        self.engine=SemCacheEngine(self.model,self.tokenizer,self.adapter,self.encoder,cluster,GlobalCache(self.a.capacity_mib*1024**2),
            storage_device=self.storage,metadata=self.meta,seed=self.a.seed,impact_reducer='paper_row_l2_sum',impact_execution='token_precompute',
            capture_impact_values=True,rho=.8,history_lambda=100,pbr_interval_queries=100)
        with capture_completed_source() as captured:
            original=rt.detailed_mixed
            @contextmanager
            def source_context(*args,**kwargs):
                with original(*args,**kwargs) as audit:
                    with observer() if observer else nullcontext(): yield audit
            with patch.object(rt,'detailed_mixed',source_context):
                result=rt.engine_request(self.engine,self.text,'user_a','source','SEMCACHE_LOOKUP_NO_REUSE',True)
        self.cluster=result['summary']['cluster_id']; source_logits=result['logits']
        self.costs=dict(source_prefill_ms=result['summary']['timing']['request_wall_ms'],source_timing='detailed LOOKUP_NO_REUSE, capture enabled; not a minimal native source cost')
        start=perf_counter_ns(); self.catalog=OccurrenceCatalog(self.namespace,self.weights,self.storage,self.a.capacity_mib*1024**2)
        self.catalog.add_source('source',self.ids,self.cluster,'user_a',captured,self.layout,prefill_complete=True,
            impacts={w['start']:w['value'] for w in result['summary']['impact_windows']},source_clock=self.engine.cache.now)
        rt.synchronize(self.a.device); self.costs['catalog_build_ms']=(perf_counter_ns()-start)/1e6
        captured.clear(); del result
        start=perf_counter_ns(); self.index=FastOccurrenceIndex(self.catalog,self.layout); self.costs['fast_index_ms']=(perf_counter_ns()-start)/1e6
        self.descriptor=SourceBackingDescriptors(self.index); self.costs['descriptor_ms']=self.descriptor.build_ms
        self.costs['incremental_build_ms']=sum(self.costs[k] for k in ('catalog_build_ms','fast_index_ms','descriptor_ms'))
        self.costs.update(catalog_copy_ms_included_in_build=self.catalog.copy_ms,source_prefill_included_in_incremental=False,
            catalog_bytes=self.catalog.resident_bytes,descriptor_python_bytes=self.descriptor.python_bytes_estimate,
            fast_index_python_bytes=self.index.additional_index_memory_bytes,allocation_basis='source snapshot shared by targets in one case')
        self.caches={mode:PreparedMappingCache(self.descriptor,4) for mode in ('v2','contiguous')}
        self.policy_state={k:(e.frequency,e.impact,e.created_at,e.last_access,e.updated_at) for k,e in self.catalog.entries.items()}
        self.snapshot=digest(dict(ids=self.ids,namespace=self.namespace,weights=self.weights,adapter='user_a',storage=self.storage))
        return source_logits

    def selected_hits(self,coverage):
        for k,values in self.policy_state.items():
            e=self.catalog.entries[k]; e.frequency,e.impact,e.created_at,e.last_access,e.updated_at=values
        return self.index.select(self.ids,self.cluster,'user_a',self.namespace,self.weights,self.layout,coverage)[0]

    def accepts(self,text,ids):
        # Fixture screening only; the real encoder and cluster assignment still run inside every request.
        return (max(ids)<self.model.config.vocab_size and ids[:-1]==self.ids[:-1]
            and self.engine.clusterer.assign(self.encoder.encode([text])[0])==self.cluster)

    def full(self,text,observer=None,user='user_a',detailed=False):
        from semcache.metrics.r1_runtime import native_request
        with observer() if observer else nullcontext():
            return native_request(self.model,self.tokenizer,self.adapter,text,user,detailed)

    def request(self,text,ids,mode,coverage,workload='warm',observer=None,instrumentation='minimal',capture_projections=False):
        import torch
        import semcache.semcache_engine as em
        import semcache.edgelora.mixed_projection as mm
        from semcache.metrics.r1_runtime import engine_request,memory_snapshot
        from semcache.edgelora.batched_mixed_projection import ExecutorBook
        from semcache.edgelora.batched_mapping_optimized import SetupBook
        for k,values in self.policy_state.items():
            e=self.catalog.entries[k]; e.frequency,e.impact,e.created_at,e.last_access,e.updated_at=values
        target=self.clone(self.engine); selected={}; diag={}
        setup=SetupBook() if instrumentation=='breakdown' else None
        book=ExecutorBook(self.a.device) if instrumentation=='breakdown' else None
        effective='legacy' if mode=='lookup' else mode
        flow=self.b.RequestMapping(effective,self.index,self.caches.get(mode),self.namespace,self.weights,
            self.a.device,getattr(torch,self.a.dtype),diag,setup,book)
        def select(hits,n,target_ids,user):
            if list(target_ids)!=list(ids): raise MissingEvidence('Target tokenizer differs from paired Full IDs')
            cluster=next(e['cluster_id'] for e in reversed(target.events) if e['event_type']=='CLUSTER_ASSIGN')
            start=perf_counter_ns()
            answer=self.index.select(target_ids,cluster,user,self.namespace,self.weights,self.layout,coverage)
            if workload=='noncontiguous':
                chosen,mask=fallback_filter(answer[0],n); answer=(chosen,mask,answer[2])
            if mode=='lookup': raise MissingEvidence('Native LOOKUP_NO_REUSE must not use the oracle selector')
            selected.update(answer=answer,selection_ms=(perf_counter_ns()-start)/1e6,cluster=cluster)
            flow.select(answer[0],target_ids,user)
            return answer[:2]
        @contextmanager
        def context(ad,user,hits,n,**kwargs):
            if mode=='lookup':
                if hits: raise MissingEvidence('LOOKUP_NO_REUSE executed cached hits')
                selected.update(answer=([],[False]*n,[]),selection_ms=None)
                flow.select([],ids,user)
            with flow.context(self.original,ad,user,hits,n,**kwargs) as audit:
                fresh=[p for p in range(n) if not any(h.window.start<=p<h.window.end for h in hits)] if observer else None
                with observer(fresh) if observer else nullcontext(): yield audit
        if self.a.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(self.a.device)
        before=memory_snapshot(self.a.device)
        with patch.object(em,'select_position_aligned_diagnostic',select),patch.object(mm,'mixed_projection_path',context):
            result=engine_request(target,text,'user_a','target',
                'SEMCACHE_LOOKUP_NO_REUSE' if mode=='lookup' else 'SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC',False)
        if result['summary']['token_ids']!=list(ids): raise MissingEvidence('Executed target token IDs differ from paired Full')
        hits,mask,_=selected['answer']; spans=[p for h in hits for p in range(h.window.start,h.window.end)]
        audit=audit_projection_records(list(result['projection_audit'].values()),len(ids),len(self.adapter.layers),sum(mask),spans)
        evidence,mapping_memory=flow.report()
        provenance=[dict(query=h.entry.qkv_metadata['source_query_id'],source=list(h.entry.positions),target=[h.window.start,h.window.end],
            tokens=list(h.window.token_ids),prefix_token_ids_equal=tuple(ids[:h.window.end])==tuple(self.ids[:h.entry.positions[1]]),
            adapter_identity_match=h.entry.qkv_metadata['source_adapter']=='user_a') for h in hits]
        if any(not p['prefix_token_ids_equal'] or not p['adapter_identity_match'] or p['source']!=p['target'] for p in provenance): raise MissingEvidence('Invalid selected source/target provenance')
        record=dict(request_ms=result['summary']['timing']['request_wall_ms'],reused_tokens=sum(mask),actual_coverage=sum(mask)/len(ids),
            projection_audit=audit,selected_provenance=provenance,selection_hash=digest(provenance),mapping=evidence,
            mapping_cache_hit=flow.cache_hit,mapping_ms=flow.mapping_ms,context_setup_ms=flow.setup_ms,selection_ms=selected['selection_ms'],
            memory=dict(before=before,after=memory_snapshot(self.a.device),**mapping_memory,temporary=getattr(flow.audit,'memory',{})),
            setup_breakdown=setup.export() if setup else None,executor_breakdown=book.export() if book else None,
            cache_statistics=self.caches[mode].stats() if mode in self.caches else None)
        self.index.guard()
        if capture_projections:
            record['_qkv']={l:{role:t.detach().to('cpu',copy=True) for role,t in q.items()} for l,q in flow.audit.projections.items()}
        return result['logits'],record

    def injection_control(self,observer):
        import torch
        from semcache.diagnostics.r1b1_d1 import full_shape_injection
        hits=self.selected_hits('max')
        self.model.set_adapter('user_a'); self.model.eval()
        with torch.inference_mode(),full_shape_injection(self.adapter,hits),observer():
            out=self.model(input_ids=torch.tensor([self.ids],device=self.a.device),use_cache=False,output_attentions=True)
        return out.logits.detach().cpu()
