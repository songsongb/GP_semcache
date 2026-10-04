"""B2 trained-user backend using existing C6 physical paths and B1 task scoring."""
import inspect
from pathlib import Path
from types import SimpleNamespace

from . import c6b_snips as b1
from .c6b_runtime import load_base
from .c6_runtime import Storage, make_c6_cache, insert_and_lookup_c6, RAW_ENTRY_BYTES
from .c6_quality import MODES, validate_hit
from .c6b2_snips import COUNTERS, evaluate_case, paired_rows, summarize
from semcache.models.task_adapters import load_two_task_users, activate_task_user


class Backend:
    def __init__(self,args):
        import torch
        from semcache.edgelora.cachegen_codec import encode_lora_delta, decode_lora_delta
        from semcache.models.model_adapter import OPTModelAdapter
        if not args.device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Real five-mode execution requires CUDA; use --dry-run for CPU planning')
        self.args=args
        self.counts={key:0 for key in COUNTERS}
        self.storage=Storage(args)
        # Validate the installed physical API before loading OPT.
        for mode in MODES:
            if MODES[mode][0]: make_c6_cache(mode,self.storage)
        self.encode,self.decode=encode_lora_delta,decode_lora_delta
        original=self.storage.codec.decode_entry
        def counted_decode(resident):
            view=original(resident)
            self.counts['storage_decode_count']+=1
            return view
        # Instrument this owned codec instance; the SAME object is registered in GlobalCache.
        self.storage.codec.decode_entry=counted_decode
        model,self.tokenizer,self.metadata=load_base(args)
        self.model=load_two_task_users(model,args.adapter_root/'user_a',args.adapter_root/'user_b')
        self.adapter=OPTModelAdapter(self.model)
        self.candidates={label:b1.label_ids(self.tokenizer,label) for label in b1.LABELS}
        self.codec_sources={
            'transport':dict(path=inspect.getfile(self.encode),sha256=b1.file_hash(inspect.getfile(self.encode))),
            'storage':dict(path=self.storage.source,sha256=b1.file_hash(self.storage.source))}
        # Include transitive frozen storage sources so an external export is identifiable.
        storage_root=Path(self.storage.source).parents[1]
        self.storage_source_tree_hashes={str(p.relative_to(storage_root)):b1.file_hash(p)
                                        for p in sorted(storage_root.rglob('*.py'))}

    def transported(self,layer,role,module,hidden,native):
        import torch
        from semcache.models.lora_decomposition import projection_parts
        base,delta,_=projection_parts(module,hidden,self.active_user)
        with torch.cuda.device(torch.device(self.args.device)):
            packet=self.encode(delta)  # Existing cdf=None default: one fit per call.
            self.counts['transport_encode_calls']+=1
            self.counts['runtime_transport_cdf_fit_count']+=1
            decoded=self.decode(packet)
            self.counts['transport_decode_calls']+=1
        return (base+decoded.to(base)).to(base.dtype)

    def forward(self,ids,user,hits,transport,*,use_cache=False,past=None,capture=False):
        import torch
        from semcache.edgelora.mixed_projection import mixed_projection_path
        activate_task_user(self.model,user)
        if any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('All B2 evaluation parameters must remain frozen')
        self.active_user=user
        inputs=dict(input_ids=torch.tensor([ids],device=self.args.device),use_cache=use_cache)
        if past is not None: inputs['past_key_values']=past
        with torch.inference_mode():
            if hits is None:
                if transport: raise ValueError('Transport must use the explicit projection path')
                return self.model(**inputs),None
            with mixed_projection_path(self.adapter,user,hits,len(ids),
                    fresh_projection=self.transported if transport else None) as audit:
                output=self.model(**inputs)
            if any(r['reused_projection_rows']!=3*len(hits) for r in audit.records.values()):
                raise ValueError('Executed projection reuse differs from frozen event')
        return output,audit.projections if capture else None

    def prepare(self,episode,mode):
        from semcache.cache.cache_entry import CacheEntry
        from semcache.semantic.subsequence import Subsequence
        source=self.rows[episode['source_index']]; target=self.rows[episode['target_index']]
        for row in (source,target):
            ids=self.tokenizer(row['query_text'],add_special_tokens=True,truncation=False)['input_ids']
            if ids!=row['token_ids']:
                raise ValueError('Tokenizer changed frozen workload prompt IDs')
        bound=min(self.args.max_sequence_length,self.model.config.max_position_embeddings)
        if max(len(source['token_ids']),len(target['token_ids'])+max(self.args.max_new_tokens,max(map(len,self.candidates.values()))))>bound:
            raise ValueError('Frozen source/target exceeds context bound; no truncation or filtering')
        reuse,transport,store=MODES[mode]
        context=SimpleNamespace(mode=mode,transport=transport,cache=None,hits=None,entry=None,accounting={})
        if not reuse: return context
        output,projections=self.forward(source['token_ids'],episode['source_user'],[],transport,capture=True)
        self.counts['source_forward_count']+=1
        del output
        start=episode['source_start']
        tensors={layer:tuple(values[role][:,start:start+3].detach().clone() for role in 'qkv')
                 for layer,values in projections.items()}
        del projections
        if store:
            entry,accounting=self.storage.encode(episode,tensors)
        else:
            entry=CacheEntry.from_tensors(episode['cluster'],episode['token_ids'],(start,start+3),tensors)
            accounting=dict(resident_payload_bytes=sum(t.numel()*t.element_size() for values in tensors.values() for t in values),
                resident_q_bytes=sum(values[0].numel()*values[0].element_size() for values in tensors.values()))
        del tensors
        entry.qkv_metadata=dict(component_scope='total_qkv',source_user=episode['source_user'],source_id=episode['source_id'])
        window=Subsequence(tuple(episode['token_ids']),episode['target_start'],episode['target_start']+3)
        if tuple(target['token_ids'][window.start:window.end])!=entry.token_ids:
            raise ValueError('Target physical token span changed')
        cache=make_c6_cache(mode,self.storage)
        hit=insert_and_lookup_c6(cache,entry,episode,window,self.storage if store else None)
        context.cache,context.entry,context.hits,context.accounting=cache,entry,[hit],accounting
        return context

    def event_identity(self,context,episode):
        if not MODES[context.mode][0]:
            if context.cache is not None or context.hits is not None:
                raise ValueError('FULL_RECOMPUTE cannot contain cache state')
            return None
        cache,entry=context.cache,context.entry
        if (len(context.hits)!=1 or cache.entries.get(entry.key) is not entry or len(cache.entries)!=1
                or entry.frequency!=1 or entry.size_bytes!=RAW_ENTRY_BYTES
                or cache.charged_cache_bytes!=RAW_ENTRY_BYTES or cache.logical_cache_bytes!=RAW_ENTRY_BYTES
                or cache.hits!=1 or cache.misses!=0):
            raise ValueError('Frozen cache admission/reuse/charge event changed')
        if MODES[context.mode][2] and (entry.tensors is not None or entry.compressed_kv is None):
            raise ValueError('Compressed resident must retain Q plus KV bytes, not decoded KV')
        return validate_hit(episode,context.hits[0])

    def score(self,context,episode,label):
        prompt=self.rows[episode['target_index']]['token_ids']
        ids=self.candidates[label]
        output,_=self.forward(prompt+ids,episode['target_user'],context.hits,context.transport)
        self.counts['target_candidate_forward_count']+=1
        return b1.score_candidate_logits(output.logits[0],len(prompt),ids)

    def greedy(self,context,episode):
        prompt=self.rows[episode['target_index']]['token_ids']
        output,_=self.forward(prompt,episode['target_user'],context.hits,context.transport,use_cache=True)
        self.counts['target_greedy_forward_count']+=1
        tokens=[]
        for step in range(self.args.max_new_tokens):
            token=int(output.logits[0,-1].argmax())
            tokens.append(token)
            if token==self.tokenizer.eos_token_id or step+1==self.args.max_new_tokens: break
            output,_=self.forward([token],episode['target_user'],[] if context.hits is not None else None,
                context.transport,use_cache=True,past=output.past_key_values)
            self.counts['target_greedy_forward_count']+=1
        return tokens,self.tokenizer.decode(tokens,skip_special_tokens=True)


def run(args,rows,episodes,manifest):
    import torch
    backend=Backend(args)
    backend.rows=rows
    props=torch.cuda.get_device_properties(torch.device(args.device))
    manifest.update(actual_gpu=dict(name=props.name,total_memory_bytes=props.total_memory,
        compute_capability=[props.major,props.minor],cuda_runtime=torch.version.cuda),
        model_tokenizer_provenance=backend.metadata,resolved_revision=backend.metadata['resolved_model_revision'],
        resolved_tokenizer_revision=backend.metadata['resolved_tokenizer_revision'],
        codec_source_hashes=backend.codec_sources,storage_source_tree_hashes=backend.storage_source_tree_hashes,
        adapter_validation='B1 two-user loader: vanilla pinned QKV LoRA; base QKV unchanged; all parameters frozen',
        cdf_counter_scope='successful transport encode calls using cdf=None; storage fitting forbidden by C6 call guard',
        counters=backend.counts)
    cases=[]
    for episode in episodes:
        for mode in MODES:
            cases.append(evaluate_case(backend,episode,mode))
        print(f'Completed {episode["episode_id"]}: all five modes',flush=True)
    pairs=paired_rows(cases); summary=summarize(cases,pairs)
    summary.update(subset=manifest['subset'],frozen_cohort_size=32,executed_case_count=len(episodes))
    b1.write_csv(args.output_dir/'per_case.csv',cases)
    b1.write_csv(args.output_dir/'paired_task_quality.csv',pairs)
    b1.write_json(args.output_dir/'summary.json',summary)
    b1.write_csv(args.output_dir/'summary.csv',summary['modes'])
    manifest.update(status='COMPLETE',output_hashes={name:b1.file_hash(args.output_dir/name)
        for name in ('per_case.csv','paired_task_quality.csv','summary.json','summary.csv')})
    b1.write_json(args.output_dir/'manifest.json',manifest)
