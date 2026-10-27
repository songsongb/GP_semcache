"""R2-A5 boundary audit and scoped validation intervention; frozen paths untouched."""
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
import ast
import hashlib
import threading
from time import perf_counter_ns
from unittest.mock import patch

from semcache.metrics.r1 import digest
from semcache.diagnostics.r1b3_break_even import MissingEvidence, ParityFailure


def boundaries(root):
    """Source-linked measurement contracts; these are static evidence, not new timings."""
    records = [
        ('warm_mapping_lookup', 'src/semcache/diagnostics/r2a_runtime.py', 'microbench',
         'measure warm_mapping_lookup: synchronize/start/events -> PreparedMappingCache.get(default validate=True) -> end/events/synchronize',
         'INCLUDES snapshot_validation; independent trials, overlapping work',
         'SUPPORTED', 'Do not add mapping lookup and validation microbench means'),
        ('snapshot_validation', 'src/semcache/edgelora/batched_mapping_optimized.py', 'SourceBackingDescriptors.validate',
         'entry perf_counter -> catalog/source/owner/entry/bucket/selected-view checks -> elapsed return',
         'nested in default get and executor setup; separate from request clone',
         'SUPPORTED', 'CPU structure/version/pointer/shape checks; CUDA interval can include host idle'),
        ('normal_warm_mapping', 'scripts/77_benchmark_r1b3_b21_setup.py', 'RequestMapping.select',
         'mapping timer around cache.get(validate=False); separate descriptor.validate in executor entry',
         'mapping and executor validation independent call sites in actual request',
         'SUPPORTED', 'Microbench get default boundary differs from normal warm request'),
        ('selection', 'src/semcache/diagnostics/r1b3_fast_selection.py', 'FastOccurrenceIndex.select',
         'layout check -> guard -> windows/prefix/identity/candidates/nonoverlap/coverage/provenance -> return',
         'includes index.guard; B3 selection_ms closes before flow.select/mapping',
         'SUPPORTED', 'B3 selection_ms closes at selected.update before flow.select: mapping excluded; guard included'),
        ('attention_impact', 'src/semcache/semcache_engine.py', 'SemCacheEngine.query',
         'post-forward attention_impact_timer -> token precompute + window grouping + history.append -> timer exit',
         'sequential after model; r2a attention_impact range covers compute_window_values only',
         'SUPPORTED', 'Profiler inner function omits outer grouping/history overhead'),
        ('cache_policy', 'src/semcache/semcache_engine.py', 'SemCacheEngine.query',
         'CHU timer separately; policy_timer wraps dedup/admission/cache.insert/materialize/events; PBR timer separately',
         'materialization nested in policy; CHU/PBR independent sibling intervals',
         'SUPPORTED', 'Do not sum policy plus materialization'),
        ('executor_setup_cleanup', 'src/semcache/edgelora/batched_mapping_optimized.py', 'batched_v2_projection_path',
         'entry identity checks; setup_started -> descriptor.validate/mask/audit/module patch -> yield; finally delete forwards',
         'validation nested in setup; cleanup after forward; initial identity checks before setup_started',
         'SUPPORTED', 'context_setup_before_forward_ms includes validation; module_patch_setup_ms is narrower'),
        ('fresh_qkv', 'src/semcache/edgelora/batched_mapping_optimized.py', 'batched_v2_projection_path',
         'fresh_native_qkv span: native(fresh_inputs); fresh_index_preparation outside span',
         'Base/A/B nested inside native; subset indexing and assembly are siblings',
         'SUPPORTED', 'Do not sum inclusive fresh_native and its child Base/LoRA costs'),
        ('qkv_assembly', 'src/semcache/edgelora/batched_mapping_optimized.py', 'batched_v2_projection_path',
         'fresh_assembly index_copy_; cached source resolution/gather/conversion/scatter each separately scoped',
         'all nested in full_mixed_projection, which itself runs inside self_attn',
         'SUPPORTED', '.to may be no-op; cached_gather slice can be a view'),
        ('attention_tag', 'src/semcache/diagnostics/r2a_profiler.py', 'trace_rows',
         'self_attn hook range surrounds QKV + attention + out_proj; table assigns closest r2a range via links',
         'nearest-tag partition is not inclusive parent duration; child assignment changes across modes',
         'SUPPORTED', 'Lower attention_inclusive label duration is not evidence of attention skipping'),
        ('historical_attention_kernel_movement', 'src/semcache/diagnostics/r2a_profiler.py', 'trace_rows',
         'requires original FULL/Oracle Chrome events and explicit CPU-op/kernel correlations',
         'aggregated category totals cannot reconstruct parent/child movement',
         'UNRESOLVED', 'Original historical traces unavailable locally; offline trace audit can supply evidence'),
        ('repeated_validation_causal_cost', 'src/semcache/diagnostics/r2a_runtime.py', 'CostRuntime.invoke',
         'paired validated Oracle minus prior-validated Oracle, one descriptor.validate bypass only',
         'whole request intervention; standalone component averages not subtracted',
         'UNRESOLVED', 'Requires immutable-fixture parity and actual paired measurements; no local GPU result'),
        ('attention_skip_claim', 'src/semcache/edgelora/batched_mapping_optimized.py', 'batched_v2_projection_path',
         'only q/k/v module.forward replaced; self_attn computation and FFN remain called',
         'full attention backend/kernel implementation depends on installed Transformers',
         'CONTRADICTED', 'Code provides no Attention or FFN bypass; exact kernel mechanism remains unresolved'),
    ]
    result=[]
    for component,path,symbol,boundary,relation,status,interpretation in records:
        text=(Path(root)/path).read_text(); lines=text.splitlines()
        needle=symbol.split('.')[-1]
        line=next((i for i,s in enumerate(lines,1) if 'def '+needle+'(' in s),None)
        result.append(dict(component=component,path=path,symbol=symbol,line=line,exact_measured_boundary=boundary,
            nesting=relation,status=status,evidence_kind='STATIC_CODE',sha256=hashlib.sha256(text.encode()).hexdigest(),
            measured_difference_ms=None,uncertainty='No new GPU measurement; existing traces must be supplied for event verification',interpretation=interpretation))
    return result


def source_contract(root):
    """Fail closed if the bypassed call or lookup-default contract has drifted."""
    path=Path(root)/'src/semcache/edgelora/batched_mapping_optimized.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PreparedMappingCache')
    get=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get')
    defaults=dict(zip([x.arg for x in get.args.args][-len(get.args.defaults):],get.args.defaults))
    if not isinstance(defaults.get('validate'),ast.Constant) or defaults['validate'].value is not True:
        raise MissingEvidence('Mapping get validation default drifted; re-audit')
    executor=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='batched_v2_projection_path')
    calls=[n for n in ast.walk(executor) if isinstance(n,ast.Call) and ast.unparse(n.func)=='mapping.descriptor.validate']
    if len(calls)!=1: raise MissingEvidence('Expected exactly one executor descriptor.validate call; re-audit')
    return dict(mapping_default_validation=True,executor_descriptor_validation_calls=len(calls),
        bypass='only runtime.plan.descriptor.validate at executor entry; all other checks remain',
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class ImmutableFixture:
    """Exclusive serial-fixture lease, with boundary validation and content evidence.

    Checks occur outside request timers. No asynchronous/concurrent fixture writers
    are supported. It is a controlled experiment, not runtime safety enforcement.
    """
    def __init__(self,runtime):
        started=perf_counter_ns()
        self.runtime=runtime; self.thread=threading.get_ident(); self.closed=False; self.calls=0
        self.outside_check_ms=[]
        self.descriptor=runtime.plan.descriptor
        self.original_validate=self.descriptor.validate
        self.original_validate(runtime.hits)
        self.baseline=self.fingerprint()
        self.before_hashes=self.backing_hashes()
        self.before_model_hashes=self.model_hashes()
        self.setup_ms=(perf_counter_ns()-started)/1e6

    def fingerprint(self):
        from semcache.diagnostics.r1b3_fast_selection import tensor_stamp
        r=self.runtime
        if r.descriptor is not r.plan.descriptor: raise MissingEvidence('Descriptor/plan ownership mismatch')
        if threading.get_ident()!=self.thread or threading.current_thread() is not threading.main_thread():
            raise MissingEvidence('Immutable fixture requires main-thread serial ownership')
        if any(t.is_alive() and t.ident!=self.thread for t in threading.enumerate()):
            raise MissingEvidence('Concurrent Python thread; bypass fixture UNVERIFIED')
        if r.model.training or any(m.training for m in r.model.modules()): raise MissingEvidence('Fixture requires eval mode')
        if list(r.tokenizer(r.text)['input_ids'])!=list(r.ids): raise MissingEvidence('Tokenizer/input fixture changed')
        expected=tuple((h.entry.qkv_metadata['source_query_id'],*h.entry.positions,h.window.start,h.window.end,h.entry.token_ids,id(h.entry)) for h in r.hits)
        if expected!=r.plan.selected_sequence: raise MissingEvidence('Mapping/selected hits differ')
        if (r.plan.namespace!=r.namespace or r.plan.weights!=r.weights or r.plan.adapter!='user_a'
            or r.plan.target_ids!=tuple(r.ids) or r.plan.length!=len(r.ids)
            or r.plan.device!=str(r.a.device) or r.descriptor.namespace!=r.namespace
            or r.descriptor.weight_identity!=r.weights):
            raise MissingEvidence('Mapping/model/adapter/input namespace mismatch')
        mask=bytearray(len(r.ids))
        for h in r.hits:
            start,end=h.window.start,h.window.end
            if not 0<=start<end<=len(mask) or any(mask[start:end]): raise MissingEvidence('Invalid immutable selected geometry')
            mask[start:end]=b'\x01'*(end-start)
        fresh=tuple(i for i,x in enumerate(mask) if not x)
        if bytes(mask)!=r.plan.mask or fresh!=r.plan.fresh or r.plan.reused!=sum(mask):
            raise MissingEvidence('Mapping mask/fresh/selected coverage mismatch')
        if any(m._forward_hooks or m._forward_pre_hooks for m in r.model.modules()):
            raise MissingEvidence('Uncontrolled model hooks present; bypass fixture UNVERIFIED')
        for i in range(len(r.adapter.layers)):
            for m in r.adapter.projection_modules(i).values():
                if m._forward_hooks or m._forward_pre_hooks or 'forward' in m.__dict__:
                    raise MissingEvidence('Projection instrumentation/wrapper present; fixture UNVERIFIED')
        params=tuple((name,id(p),tensor_stamp(p)) for name,p in r.model.named_parameters())
        owners=tuple((id(t),tensor_stamp(t)) for t,_ in r.descriptor.owner_stamps)
        indices=tuple((key,id(t),tensor_stamp(t)) for key,t in r.plan.index_tensors.items())
        groups=tuple((g['query'],id(g['backing']),tuple(g['source']),tuple(g['target']),g['is_contiguous']) for g in r.plan.groups)
        active=tuple(getattr(r.model,'active_adapters',()))
        return (id(r.model),id(r.adapter),id(r.plan),id(r.plan.descriptor),id(r.catalog),r.snapshot,r.oracle_snapshot,
            tuple(r.ids),tuple(r.oracle_ids),r.text,r.namespace,r.weights,r.storage,str(r.a.device),str(r.a.dtype),
            r.plan.adapter,r.plan.namespace,r.plan.weights,r.plan.device,r.plan.dtype,r.plan.length,r.plan.target_ids,
            r.plan.selected_sequence,r.plan.mask,r.plan.fresh,r.plan.reused,groups,params,owners,indices,active)

    def backing_hashes(self):
        from semcache.diagnostics.r1b3_gemm_shape import tensor_sha
        r=self.runtime
        # Include mapping indices; .data writes may bypass tensor version counters.
        tensors=[t for t,_ in r.descriptor.owner_stamps]+list(r.plan.index_tensors.values())
        return tuple((id(t),tensor_sha(t)) for t in tensors)

    def model_hashes(self):
        from semcache.diagnostics.r1b3_gemm_shape import tensor_sha
        return tuple((name,tensor_sha(p)) for name,p in self.runtime.model.named_parameters())

    def check(self,contents=False):
        started=perf_counter_ns()
        if self.closed: raise MissingEvidence('Immutable fixture lease already closed')
        if self.fingerprint()!=self.baseline: raise MissingEvidence('Fixture identity/version/geometry changed; UNVERIFIED')
        self.original_validate(self.runtime.hits)
        if contents:
            if self.backing_hashes()!=self.before_hashes: raise MissingEvidence('Backing/index content changed; UNVERIFIED')
            if self.model_hashes()!=self.before_model_hashes: raise MissingEvidence('Model weight content changed; UNVERIFIED')
        self.outside_check_ms.append(dict(wall_ms=(perf_counter_ns()-started)/1e6,includes_content_hashes=contents))

    @contextmanager
    def bypass(self):
        self.check()
        count_before=self.calls
        def skip(hits,book=None):
            if (threading.get_ident()!=self.thread or len(hits)!=len(self.runtime.hits)
                or any(a is not b for a,b in zip(hits,self.runtime.hits))):
                raise MissingEvidence('Bypass outside exact prior-validated fixture')
            if self.runtime.plan.descriptor is not self.descriptor: raise MissingEvidence('Bypass descriptor changed')
            self.calls+=1
            return None  # No fake measured elapsed time or success evidence.
        try:
            with patch.object(self.descriptor,'validate',skip): yield
        finally:
            self.check()
        if self.calls-count_before!=1: raise MissingEvidence('Expected exactly one repeated-validation bypass')

    def close(self):
        self.check(contents=True); self.closed=True
        return dict(status='VERIFIED_CONTROLLED_FIXTURE',bypassed_calls=self.calls,
            prevalidation_setup_ms=self.setup_ms,outside_request_checks=self.outside_check_ms,
            cost_scope='outside request; full validation plus stamps; final check also hashes all source/index/model tensors; do not add to minimal request wall',
            backing_index_hash=digest(self.before_hashes),model_weight_content_hash=digest(self.before_model_hashes),boundary_hashes_equal=True,
            limitation='exclusive serial experiment; no proof against external writers or malicious mutate-and-restore; no production safety')


def compare_execution(reference,actual,ref_record,record):
    import torch
    from semcache.edgelora.batched_mixed_projection import compare_projections
    for key in ('projection_audit','selection_hash','reused_tokens'):
        if ref_record[key]!=record[key]: raise ParityFailure('Controlled Oracle '+key+' differs')
    if not torch.equal(reference,actual): raise ParityFailure('Controlled Oracle logits differ exactly')
    if '_qkv' in ref_record or '_qkv' in record:
        if '_qkv' not in ref_record or '_qkv' not in record: raise MissingEvidence('Missing QKV parity evidence')
        n=next(iter(ref_record['_qkv'].values()))['q'].shape[1]
        mask=[False]*n
        for h in ref_record['selected_provenance']: mask[h['target'][0]:h['target'][1]]=[True]*(h['target'][1]-h['target'][0])
        result=compare_projections(ref_record['_qkv'],record['_qkv'],mask)
        if not result['passed']: raise ParityFailure('Controlled Oracle QKV differs')
        return result
    return dict(logits_exact=True,rows_equal=True,selection_equal=True,qkv='untimed validation separately required')


def audit_trace(trace):
    """Independent offline audit, not a replacement for frozen R2-A attribution.

    CPU-op External IDs and runtime correlations provide candidate origins. Multiple
    candidate origins stay ambiguous; kernel names alone cannot establish origin.
    Parent attention containment and closest-tag assignment are reported separately.
    """
    events=[e for e in trace.get('traceEvents',[]) if e.get('ph')=='X' and 'ts' in e and 'dur' in e]
    lanes=defaultdict(list)
    for e in events:
        if e.get('cat') in ('user_annotation','cpu_op','cuda_runtime','cuda_driver'):
            lanes[(e.get('pid'),e.get('tid'))].append(e)
    external=defaultdict(set); correlations=defaultdict(set)
    for lane in lanes.values():
        stack=[]
        for e in sorted(lane,key=lambda x:(x['ts'],-x['dur'])):
            stack=[x for x in stack if x['ts']+x['dur']>e['ts']]
            ancestors=tuple(x['name'][5:] for x in stack if x['ts']<=e['ts'] and e['ts']+e['dur']<=x['ts']+x['dur'] and x.get('name','').startswith('r2a::'))
            tags=tuple(x for x in ancestors if not x.startswith('request/'))
            origin=(tags[-1] if tags else 'unknown_or_unattributed','attention_inclusive' in ancestors,e.get('name'))
            args=e.get('args',{})
            if e.get('cat')=='cpu_op' and args.get('External id') is not None: external[str(args['External id'])].add(origin)
            if e.get('cat') in ('cuda_runtime','cuda_driver') and args.get('correlation') is not None:
                correlations[str(args['correlation'])].add(origin)
            if e.get('cat')=='user_annotation': stack.append(e)
    records=[]; totals=defaultdict(float); ambiguous=0
    for e in events:
        cat=str(e.get('cat','')).lower()
        if not any(s in cat for s in ('kernel','gpu_memcpy','gpu_memset')): continue
        args=e.get('args',{})
        origins=external.get(str(args.get('External id')),set())
        launches=correlations.get(str(args.get('correlation')),set())
        # A matching CPU op provides more precise schema than its CUDA runtime launch.
        launch_tags={(x[0],x[1]) for x in launches}; op_tags={(x[0],x[1]) for x in origins}
        conflict=bool(op_tags and launch_tags and op_tags!=launch_tags)
        candidates=op_tags or launch_tags
        known=len(candidates)==1 and not conflict
        tag,inside=next(iter(candidates)) if known else ('ambiguous_or_unattributed',False)
        ms=e['dur']/1000
        if 'kernel' in cat:
            totals['all_kernel_duration_ms']+=ms
            if not known: ambiguous+=1; totals['unknown_kernel_duration_ms']+=ms
            if known and inside: totals['attention_parent_inclusive_kernel_ms']+=ms
            if tag=='attention_inclusive': totals['attention_closest_tag_kernel_ms']+=ms
        ops=sorted({x[2] for x in origins})
        attention_math=known and inside and any(op in ('aten::bmm','aten::_softmax','aten::softmax','aten::_scaled_dot_product_attention_math','aten::scaled_dot_product_attention') for op in ops)
        if attention_math and 'kernel' in cat: totals['linked_attention_math_candidate_ms']+=ms
        records.append(dict(activity=cat,kernel=e.get('name'),duration_ms=ms,component=tag,
            attention_parent_contains=inside if known else None,cpu_ops=ops,attention_math_candidate=attention_math,
            attribution_status='LINKED' if known else 'AMBIGUOUS_OR_UNATTRIBUTED',stream=args.get('stream')))
    from semcache.diagnostics.r2a_profiler import trace_rows
    old,oldmeta=trace_rows(trace)
    return dict(status='AUDITED' if records else 'UNVERIFIED',activities=records,totals=dict(totals),
        ambiguous_kernel_count=ambiguous,frozen_attribution=old,frozen_metadata=oldmeta,
        interpretation='closest-tag attention is not inclusive; parent and child columns overlap and MUST NOT be added',
        attention_skipped='UNRESOLVED_FROM_TRACE_ALONE; executor code has no attention bypass')
