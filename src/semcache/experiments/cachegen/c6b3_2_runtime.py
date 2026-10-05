"""B3 adapter over the unchanged B2 physical backend; no selection or training."""
import importlib.metadata
import json
from collections import defaultdict
from statistics import mean
from . import c6b3_2_multiwoz as b
from .c6b2_runtime import Backend
from .c6_quality import logit_metrics


class MultiwozBackend(Backend):
    def __init__(self,args):
        super().__init__(args)
        self.phase='task'
        self.traffic=defaultdict(lambda:defaultdict(int))
        original=self.encode
        def encode(delta):
            packet=original(delta)
            sizes=packet['sizes'];role=self.transport_role
            self.traffic[self.phase][f'raw_{role}_delta_bytes']+=sizes['raw_bytes']
            self.traffic[self.phase][f'compressed_{role}_delta_bytes']+=sizes['runtime_total_bytes']
            self.traffic[self.phase][f'{role}_cdf_bytes']+=sizes['cdf_bytes']
            return packet
        self.encode=encode

    def transported(self,layer,role,module,hidden,native):
        self.transport_role=role
        return super().transported(layer,role,module,hidden,native)

    def prepare(self,episode,mode):
        # B3-1's 384 applies to supervised examples; generated continuations use
        # actual OPT position capacity. No prompt/token alteration occurs here.
        original=self.args.max_sequence_length
        self.args.max_sequence_length=self.model.config.max_position_embeddings
        try:return super().prepare(episode,mode)
        finally:self.args.max_sequence_length=original

    def forward(self,ids,user,hits,transport,**kwargs):
        output=super().forward(ids,user,hits,transport,**kwargs)
        if hits is not None and not transport:
            fresh=len(ids)-sum(h.window.end-h.window.start for h in hits)
            for layer in range(len(self.adapter.layers)):
                for role,module in self.adapter.projection_modules(layer).items():
                    n=fresh*module.out_features*module.lora_B[user].weight.element_size()
                    self.traffic[self.phase][f'raw_{role}_delta_bytes']+=n
        return output

    def teacher_logits(self,context,episode,canonical):
        prompt=self.rows[episode['target_index']]['token_ids']
        output,_=self.forward(prompt+canonical[:-1],episode['target_user'],context.hits,context.transport)
        self.counts['teacher_forced_forward_count']=self.counts.get('teacher_forced_forward_count',0)+1
        return output.logits[0,len(prompt)-1:len(prompt)-1+len(canonical)].detach().float().cpu()


def traffic_summary(counts,compressed):
    out={k:counts.get(k,0) for role in 'qkv' for k in (f'raw_{role}_delta_bytes',f'compressed_{role}_delta_bytes',f'{role}_cdf_bytes')}
    raw=sum(out[f'raw_{r}_delta_bytes'] for r in 'qkv')
    packet=sum(out[f'compressed_{r}_delta_bytes'] for r in 'qkv')
    cdf=sum(out[f'{r}_cdf_bytes'] for r in 'qkv')
    actual=packet+cdf if compressed else raw
    out.update(raw_total_delta_bytes=raw,compressed_total_delta_bytes=packet,
        per_packet_cdf_bytes=cdf,transmitted_total_including_cdf_bytes=actual,
        transport_compression_ratio=raw/actual if actual else None,
        transport_byte_reduction_percentage=100*(1-actual/raw) if raw else None,
        scope='logical LoRA delta payload; compressed runtime packet plus per-call fitted CDF; no network latency')
    return out


def run(args,rows,episodes,canonical,canonical_summary,manifest):
    import torch
    imported=b.mode_summary('FULL_RECOMPUTE',canonical)
    for key in ('version','signature'):
        b.require(imported['corpus_bleu'][key]==canonical_summary['corpus_bleu'][key],'Canonical BLEU environment mismatch: '+key)
    b.require(abs(imported['corpus_bleu']['value']-canonical_summary['corpus_bleu']['value'])<1e-10,'Canonical BLEU differs from imported texts')
    backend=MultiwozBackend(args);backend.rows=rows
    # Recompute offsets only, never clusters or selection, with the loaded pinned tokenizer.
    from .c6b3_selection import annotate
    annotated=annotate(rows,backend.tokenizer)
    by_id={r['source_id']:r for r in annotated}
    for e in episodes:
        for side in ('source','target'):
            r=by_id[e[side+'_id']];p=e[side+'_start']
            b.require({p,p+1,p+2}<=set(r['current_user_token_indices']),'Runtime offset audit failed')
    props=torch.cuda.get_device_properties(torch.device(args.device))
    manifest.update(actual_gpu=dict(name=props.name,total_memory_bytes=props.total_memory,cuda_runtime=torch.version.cuda),
        software={name:importlib.metadata.version(name) for name in ('torch','transformers','peft','sacrebleu')},model_tokenizer_provenance=backend.metadata,
        codec_source_hashes=backend.codec_sources,storage_source_tree_hashes=backend.storage_source_tree_hashes,
        shared_storage_profile_bytes=args.profile_path.stat().st_size,
        teacher_forced_policy='single forward per mode on canonical FULL_RECOMPUTE generated continuation; prompt plus previous canonical tokens; logits discarded per episode',
        transport_accounting_policy='task source+greedy traffic separate from diagnostic forward traffic; ratios include runtime CDF bytes',
        canonical_generation_policy='import only; no second FULL_RECOMPUTE generation',counters=backend.counts)
    cases=[];paired=[]
    for index,(e,official) in enumerate(zip(episodes,canonical)):
        logits={};episode_cases={}
        for mode,(reuse,transport,storage) in b.MODES.items():
            before=dict(backend.counts);backend.traffic.clear();backend.phase='task'
            context=backend.prepare(e,mode)
            identity=backend.event_identity(context,e)
            b.require(identity==(b.d.b.digest(e) if reuse else None),'Logical event mismatch')
            if mode=='FULL_RECOMPUTE':
                tokens=official['generated_token_ids'];text=official['generated_text']
                b.require(backend.tokenizer.decode(tokens,skip_special_tokens=True)==text,'Canonical token/text mismatch')
            else:tokens,text=backend.greedy(context,e)
            # Use B3-1's reference tokenization and exact fidelity definitions.
            encoded=b.d.encode_example(backend.tokenizer,rows[e['target_index']])
            reference_ids=encoded['input_ids'][encoded['prompt_length']:]
            fidelity=b.generation_metrics(reference_ids,tokens)
            if mode=='FULL_RECOMPUTE':
                for key in ('normalized_edit_distance','position_agreement'):
                    b.require(abs(fidelity[key]-official[key])<1e-10,'Canonical reference diagnostic mismatch')
            backend.phase='teacher_forced'
            logits[mode]=backend.teacher_logits(context,e,official['generated_token_ids'])
            b.require(backend.event_identity(context,e)==identity,'Logical event changed during forwards')
            delta={k:backend.counts[k]-before.get(k,0) for k in backend.counts}
            b.require(delta['source_forward_count']==int(reuse) and delta['storage_decode_count']==int(storage),'Source build/decode count mismatch')
            b.require(delta['runtime_storage_cdf_fit_count']==0,'Forbidden storage CDF fit')
            b.require(delta['transport_encode_calls']==delta['transport_decode_calls']==delta['runtime_transport_cdf_fit_count'],'Transport counter mismatch')
            b.require(delta['transport_encode_calls']==(96*(int(reuse)+len(tokens)+1) if transport else 0),'Projection codec count mismatch')
            account=b.byte_accounting(mode,context.accounting)
            account.update(raw_resident_q_bytes=account['raw_qkv_bytes']//3,raw_resident_k_bytes=account['raw_qkv_bytes']//3,
                raw_resident_v_bytes=account['raw_qkv_bytes']//3,
                compressed_resident_kv_frame_bytes=account['resident_kv_frame_bytes'] if storage else 0)
            case=dict(e)
            case.update(**b.SCOPE,episode_index=index,mode=mode,user=e['target_user'],
                reference_text=rows[e['target_index']]['reference_text'],generated_text=text,generated_token_ids=tokens,
                **fidelity,logical_event_hash=identity,selected_mode_hit_count=int(reuse),storage_accounting=account,
                transport_accounting=traffic_summary(backend.traffic['task'],transport),
                diagnostic_transport_accounting=traffic_summary(backend.traffic['teacher_forced'],transport),runtime_counters=delta,
                generation_source='canonical_frozen32' if not reuse else 'measured_b3_2')
            cases.append(case);episode_cases[mode]=case
            del context
        for baseline,mode,effect in b.PAIRS:
            paired.append(dict(episode_id=e['episode_id'],effect=effect,baseline_mode=baseline,mode=mode,
                generation_fidelity=b.generation_metrics(episode_cases[baseline]['generated_token_ids'],episode_cases[mode]['generated_token_ids']),
                teacher_forced=logit_metrics(logits[baseline],logits[mode]),
                scope='paired mode fidelity, not task quality; same canonical FULL continuation'))
        del logits
        print(f'Completed {index+1}/32: {e["episode_id"]}',flush=True)
    modes=[imported if mode=='FULL_RECOMPUTE' else b.mode_summary(mode,[c for c in cases if c['mode']==mode]) for mode in b.MODES]
    full=modes[0]
    # Verify canonical corpus scores from imported aligned texts; no hardcoded score threshold.
    def verify_score(actual,saved):
        for key in ('version','signature','tokenizer','smoothing','effective_order','lowercase','scope','implementation'):
            b.require(actual[key]==saved[key],'Canonical SacreBLEU environment/protocol changed: '+key)
        b.require(abs(actual['value']-saved['value'])<1e-10,'Canonical BLEU verification mismatch')
    verify_score(full['corpus_bleu'],canonical_summary['corpus_bleu'])
    for user in b.d.USERS:verify_score(full[user+'_bleu'],canonical_summary['user_bleu'][user])
    for depth in ('1','2','3','4'):verify_score(full['history_depth_bleu'][depth],canonical_summary['history_depth_bleu_diagnostic'][depth])
    full['corpus_bleu']=canonical_summary['corpus_bleu']
    comparisons=b.causal_comparisons(modes)
    pair_summary=[]
    for _,_,effect in b.PAIRS:
        group=[p for p in paired if p['effect']==effect]
        pair_summary.append(dict(effect=effect,count=len(group),
            generation_fidelity={k:mean(p['generation_fidelity'][k] for p in group) for k in ('exact_generation','normalized_edit_distance','position_agreement','common_prefix_length')},
            teacher_forced={k:mean(p['teacher_forced'][k] for p in group) for k in group[0]['teacher_forced']},
            aggregation='unweighted mean of per-case metrics, including per-case median KL'))
    storage_summary={}
    for mode in b.MODES:
        group=[c['storage_accounting'] for c in cases if c['mode']==mode]
        storage_summary[mode]={key:mean(r[key] for r in group) if group[0][key] is not None else None
                               for key in group[0]}
    transport_summary={mode:traffic_summary({key:sum(c['transport_accounting'][key] for c in cases if c['mode']==mode) for role in 'qkv' for key in (f'raw_{role}_delta_bytes',f'compressed_{role}_delta_bytes',f'{role}_cdf_bytes')},b.MODES[mode][1]) for mode in b.MODES}
    summary=dict(**b.SCOPE,canonical_full_recompute_bleu=canonical_summary['corpus_bleu']['value'],modes=modes,
        causal_comparisons=comparisons,paired_quality=pair_summary,storage_accounting=storage_summary,
        transport_accounting=transport_summary,selection_audit=manifest['selection_audit'])
    b.write_csv(args.output_root/'per_case.csv',cases);b.write_csv(args.output_root/'summary.csv',modes)
    for name,value in [('summary.json',summary),('causal_comparisons.json',comparisons),('paired_quality.json',dict(per_case=paired,aggregates=pair_summary))]:b.d.b.write(args.output_root/name,value)
    manifest.update(status='COMPLETE',output_hashes={name:b.d.b.m9.file_hash(args.output_root/name) for name in ('per_case.csv','summary.csv','summary.json','causal_comparisons.json','paired_quality.json')})
    (args.output_root/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
