"""Lazy C6 execution backend; importing this module loads no models or codecs."""
import importlib.metadata
import inspect
import platform
import subprocess
import sys
from contextlib import contextmanager

from .c6_quality import (MODES, CONTRACT, BLEU, MODEL_ID, MODEL_REVISION, PROFILE_SHA,
    file_hash, digest, verify_profile, validate_hit, case_result, generation_metrics, logit_metrics, summarize, write_json, write_csv)


def manifest(args, selection):
    def git(*command):
        return subprocess.check_output(['git', *command], text=True).strip()
    versions = {}
    for package in ('torch', 'transformers', 'peft', 'sacrebleu'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(**CONTRACT, schema='c6_quality_v1', model=MODEL_ID, tokenizer=MODEL_ID,
        requested_revision=MODEL_REVISION, resolved_revision=None, dtype='float16', device=args.device,
        semantic_device=args.semantic_device, semantic_execution='frozen measured assignments; no encoder execution',
        workload_hashes={d:file_hash(getattr(args,d)) for d in ('snips','multiwoz')},
        selection_hash=file_hash(args.output_dir/'selection.json'),
        profile_path=str(args.profile_path), expected_profile_sha256=PROFILE_SHA,
        storage_src=str(args.storage_src) if args.storage_src else None,
        storage_codec_provenance='existing C2 FrozenK20V16Codec; FAST_PY_BITEXACT; external adapter if absent locally',
        transport_codec_provenance='semcache.edgelora.cachegen_codec encode_lora_delta/decode_lora_delta; num_bins=32; runtime transport CDFs',
        transport_scope='source prefill plus target fresh prefill/decode rows; LoRA deltas only',
        teacher_forced_scope='one mixed full-sequence forward per mode; RAW canonical previous tokens; fixed prompt hit only',
        teacher_forced_transport_packet_scope='all fresh rows in one packet per projection; differs from incremental generation packet grouping',
        generation=dict(do_sample=False, greedy=True, seed=args.seed, max_new_tokens=args.max_new_tokens, respect_eos=True),
        bleu_protocol=dict(**BLEU, protocol_provenance='REPRODUCTION_CHOICE', paper_bleu_claimed=False),
        controlled_fixture=dict(rank=8, users=['user_a','user_b'], seeds=[101,202], trained_adapter=False),
        git=dict(branch=git('branch','--show-current'), commit=git('rev-parse','HEAD'), status=git('status','--porcelain')),
        software=dict(python=platform.python_version(), **versions), output_hashes={},
        script48_discrepancy='script48 resident Q absent/current Q recomputed; C6 source TOTAL Q remains resident FP16')


@contextmanager
def forbid_storage_fitting():
    """Fail closed on Python/C CDF fitting calls during storage operations."""
    previous = sys.getprofile()
    def guard(frame, event, arg):
        name = (getattr(arg, '__name__', '') if event == 'c_call' else frame.f_code.co_name).lower()
        if event in ('call', 'c_call') and (name in ('fit', 'fit_profiles', 'cdf_from_counts') or
                ('cdf' in name and any(w in name for w in ('fit', 'build', 'calculate', 'train')))):
            raise RuntimeError(f'Runtime storage CDF fitting forbidden: {name}')
    sys.setprofile(guard)
    try:
        yield
    finally:
        sys.setprofile(previous)


class Storage:
    """Reuse C2's entry boundary; keep owned source Q alongside compressed KV."""
    def __init__(self, args):
        verify_profile(args.profile_path)
        if args.storage_src:
            import semcache.experiments.cachegen
            # Extend the C6 package to locate the existing external C2 implementation.
            external = args.storage_src.resolve()/'semcache'/'experiments'/'cachegen'
            semcache.experiments.cachegen.__path__.append(str(external))
        from semcache.experiments.cachegen.c2.physical_storage import FrozenK20V16Codec
        from semcache.experiments.cachegen.b2 import format as fmt
        self.source = inspect.getfile(FrozenK20V16Codec)
        with forbid_storage_fitting():
            self.codec = FrozenK20V16Codec(args.profile_path, quantization_device=args.device,
                                          decode_device=args.device, expected_hidden=2560, coder_backend=fmt.FAST_CODER)

    def encode(self, episode, tensors):
        from semcache.cache.cache_entry import CacheEntry
        def entry_factory(*args, q_tensors, compressed_kv, **kwargs):
            resident = CacheEntry(*args, qkv_metadata=dict(component_scope='total_qkv',
                source_user=episode['source_user'], source_id=episode['source_id']))
            resident.q_tensors = q_tensors
            resident.compressed_kv = compressed_kv
            return resident
        with forbid_storage_fitting():
            resident = self.codec.make_entry(entry_factory, episode['cluster'], tuple(episode['token_ids']),
                (episode['source_start'], episode['source_start']+3), tensors, 'cpu')
        if resident.compressed_kv.profile_sha256 != PROFILE_SHA:
            raise ValueError('Storage payload profile mismatch')
        q_bytes = sum(t.numel()*t.element_size() for t in resident.q_tensors.values())
        frame_bytes = len(resident.compressed_kv.bitstream)
        return resident, dict(resident_q_bytes=q_bytes, resident_kv_frame_bytes=frame_bytes,
            resident_payload_bytes=q_bytes+frame_bytes,
            resident_local_metadata_bytes=resident.compressed_kv.local_metadata_bytes)

    def decode(self, resident):
        import torch
        with forbid_storage_fitting():
            view = self.codec.decode_entry(resident)
        for layer, q in resident.q_tensors.items():
            if q.dtype != torch.float16 or not torch.equal(view.tensors[layer][0].cpu(), q.cpu()):
                raise ValueError('Storage codec changed resident Q')
        return view


def execute(args, workloads, selection, provenance):
    # Dependencies and profile checked before model loading; no substitute codec.
    storage = Storage(args) if any(MODES[m][2] for m in args.modes) else None
    transport = any(MODES[m][1] for m in args.modes)
    if transport:
        from semcache.edgelora.cachegen_codec import encode_lora_delta, decode_lora_delta
    import torch
    from semcache.models.loader import load_model
    from semcache.models.lora_fixtures import create_controlled_users
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.models.lora_decomposition import projection_parts
    from semcache.edgelora.mixed_projection import mixed_projection_path
    from semcache.cache.cache_entry import CacheEntry
    from semcache.cache.global_cache import GlobalCache
    from semcache.semantic.matcher import ExactTokenMatcher
    from semcache.semantic.hit_selection import CacheHit
    from semcache.semantic.subsequence import Subsequence
    if transport and (not args.device.startswith('cuda') or not torch.cuda.is_available()):
        raise RuntimeError('Real transport codec requires CUDA; use --dry-run for CPU validation')
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model, tokenizer, metadata = load_model(dict(name=MODEL_ID, tokenizer=MODEL_ID,
        revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION, dtype='float16', device=args.device,
        local_files_only=True, attention_implementation='eager'))
    if metadata['resolved_model_revision'] != MODEL_REVISION or metadata['resolved_tokenizer_revision'] != MODEL_REVISION:
        raise ValueError('Resolved model revision mismatch')
    model, fixtures = create_controlled_users(model)
    model.eval()
    adapter = OPTModelAdapter(model)
    counts = dict(storage_runtime_cdf_fits=0, transport_runtime_cdf_fits=0)

    def compressed_projection(layer, role, module, hidden, native):
        base, delta, _ = projection_parts(module, hidden, active_user[0])
        # Existing transport default explicitly fits a CDF per transmitted delta.
        with torch.cuda.device(torch.device(args.device)):
            packet = encode_lora_delta(delta)
            decoded = decode_lora_delta(packet)
        counts['transport_runtime_cdf_fits'] += 1
        return (base+decoded.to(base)).to(base.dtype)

    active_user = [None]
    def forward(ids, user, hits, compressed=False, past=None, capture=False):
        active_user[0] = user
        model.set_adapter(user)
        model.eval()
        inputs = dict(input_ids=torch.tensor([ids],device=args.device), use_cache=True)
        if past is not None:
            inputs['past_key_values'] = past
        with torch.inference_mode():
            if hits is None:
                return model(**inputs), None
            with mixed_projection_path(adapter,user,hits,len(ids),
                    fresh_projection=compressed_projection if compressed else None) as audit:
                output = model(**inputs)
            if any(r['reused_projection_rows'] != 3*len(hits) for r in audit.records.values()):
                raise ValueError('Matched logical reuse event changed')
        return output, audit.projections if capture else None

    def generate(ids, user, hits, compressed):
        generated = []
        output, _ = forward(ids,user,hits,compressed)
        for step in range(args.max_new_tokens):
            token = int(output.logits[0,-1].argmax())
            generated.append(token)
            if token == tokenizer.eos_token_id or step+1 == args.max_new_tokens:
                break
            output, _ = forward([token],user,[] if hits is not None else None,compressed,past=output.past_key_values)
        return generated

    cases, pairs = [], []
    for episode in selection['episodes']:
        rows = workloads[episode['dataset']]
        source, target = rows[episode['source_index']], rows[episode['target_index']]
        for row in (source,target):
            if tokenizer(row['query_text'],add_special_tokens=True,truncation=False)['input_ids'] != row['token_ids']:
                raise ValueError('Loaded tokenizer differs from frozen workload')
        max_length = max(len(source['token_ids']),len(target['token_ids'])+args.max_new_tokens)
        if max_length > model.config.max_position_embeddings:
            raise ValueError('Episode exceeds model context; never truncate selected workload')
        per_mode = {}
        for mode in args.modes:
            reuse, compressed, store = MODES[mode]
            hits, accounting = None, dict(resident_payload_bytes=0, resident_q_bytes=0)
            if reuse:
                _, projections = forward(source['token_ids'],episode['source_user'],[],compressed,capture=True)
                start = episode['source_start']
                tensors = {layer: tuple(values[r][:,start:start+3].detach().clone() for r in 'qkv') for layer,values in projections.items()}
                del projections
                if store:
                    entry, accounting = storage.encode(episode,tensors)
                else:
                    accounting = dict(resident_payload_bytes=sum(t.numel()*t.element_size() for v in tensors.values() for t in v),
                        resident_q_bytes=sum(v[0].numel()*v[0].element_size() for v in tensors.values()))
                    entry = CacheEntry.from_tensors(episode['cluster'],episode['token_ids'],(start,start+3),tensors)
                entry.qkv_metadata = dict(component_scope='total_qkv',source_user=episode['source_user'],source_id=episode['source_id'])
                window = Subsequence(tuple(episode['token_ids']),episode['target_start'],episode['target_start']+3)
                if tuple(target['token_ids'][window.start:window.end]) != entry.token_ids:
                    raise ValueError('Frozen physical event mismatch')
                cache = GlobalCache(3*3*2560*32*2)  # fixed raw logical size, identical in every mode
                if not cache.insert(entry):
                    raise ValueError('Frozen source admission failed')
                resident = cache.lookup(ExactTokenMatcher().key(episode['cluster'],window), record_reuse=False)
                if resident is not entry or len(cache.entries) != 1:
                    raise ValueError('Frozen cache lookup event mismatch')
                cache.record_reuse(resident)
                hits = [CacheHit(window,storage.decode(resident) if store else resident)]
                validate_hit(episode,hits[0])
                del tensors
            tokens = generate(target['token_ids'],episode['target_user'],hits,compressed)
            per_mode[mode] = dict(tokens=tokens,hits=hits,compressed=compressed)
            cases.append(case_result(episode,mode,tokens,tokenizer.decode(tokens,skip_special_tokens=True),accounting))
        canonical = per_mode['RAW_SEMCACHE']['tokens']
        for mode, data in per_mode.items():
            output, _ = forward(target['token_ids']+canonical[:-1],episode['target_user'],data['hits'],data['compressed'])
            start = len(target['token_ids'])-1
            data['logits'] = output.logits[0,start:start+len(canonical)].detach().cpu()
            del output
        comparisons = [('RAW_SEMCACHE',m) for m in args.modes]
        comparisons += [('TRANSPORT_QKV_COMP','FULL_PIPELINE'), ('FULL_RECOMPUTE','RAW_SEMCACHE')]
        for baseline, mode in comparisons:
            if baseline not in per_mode or mode not in per_mode:
                continue
            a,b = per_mode[baseline],per_mode[mode]
            pairs.append(dict(episode_id=episode['episode_id'],dataset=episode['dataset'],baseline=baseline,mode=mode,
                **CONTRACT, **generation_metrics(a['tokens'],b['tokens']), **logit_metrics(a['logits'],b['logits']),
                canonical_continuation='RAW_SEMCACHE', metric_scope='secondary generation/logit fidelity',
                effect='semantic reuse' if 'FULL_RECOMPUTE' in (baseline,mode) else 'compression' if baseline != mode else 'identity control'))
        print(f'Completed {episode["episode_id"]}',flush=True)
    summary = summarize(cases,pairs)
    causal = []
    for dataset in workloads:
        scores = {r['mode']:r['corpus_bleu'] for r in summary if r['dataset']==dataset}
        for baseline,mode,effect in [('RAW_SEMCACHE','STORAGE_KV_COMP','storage'),
            ('RAW_SEMCACHE','TRANSPORT_QKV_COMP','transport'),('TRANSPORT_QKV_COMP','FULL_PIPELINE','storage after transport'),
            ('RAW_SEMCACHE','FULL_PIPELINE','combined compression'),('FULL_RECOMPUTE','RAW_SEMCACHE','semantic reuse')]:
            if baseline in scores and mode in scores:
                causal.append(dict(dataset=dataset,baseline=baseline,mode=mode,effect=effect,delta_bleu=scores[mode]-scores[baseline]))
    write_csv(args.output_dir/'per_case.csv',cases)
    write_csv(args.output_dir/'paired_quality.csv',pairs)
    write_csv(args.output_dir/'summary.csv',summary)
    write_json(args.output_dir/'summary.json',dict(**CONTRACT,dataset_modes=summary,causal_comparisons=causal))
    provenance.update(status='COMPLETE',resolved_revision=metadata['resolved_model_revision'],
        model_metadata=metadata,controlled_fixture=fixtures, **counts,
        storage_profile_verified=storage is not None, storage_codec_file=storage.source if storage else None,
        storage_codec_sha256=file_hash(storage.source) if storage else None,
        transport_codec_sha256=file_hash(inspect.getfile(encode_lora_delta)) if transport else None,
        cdf_counter_scope='transport encode calls without shared CDF; storage fitting-call guard (fit, fit_profiles, cdf_from_counts and named CDF builders)',
        bleu_signatures=[r['bleu_protocol'] for r in summary])
    provenance['output_hashes'] = {p.name:file_hash(p) for p in args.output_dir.iterdir() if p.name != 'manifest.json'}
    write_json(args.output_dir/'manifest.json',provenance)
