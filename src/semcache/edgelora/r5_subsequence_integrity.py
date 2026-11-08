"""D5 request-scoped validation; no cache-policy or projection arithmetic.

Single owner thread and ordered CUDA stream; no adversarial concurrent writers.
Never trust Tensor._version: data/alias writes can bypass it. Authenticated private
copies, rather than mutable cache aliases, are supplied to scatter in scoped mode.
Unconsumed tensor content is not continuously rehashed after authorization.
"""
from copy import deepcopy
import threading,time
from .communication import require,digest
from .r5_subsequence_profile import call

EAGER='EAGER_FULL_PROOF'
SCOPED='REQUEST_SCOPED_VERIFIED'
MODES=(EAGER,SCOPED)
SCHEMA='R5_REQUEST_SCOPED_INTEGRITY_V1'
THREAT_MODEL='single-owner-thread; ordered CUDA stream; no concurrent external writes; private consumed snapshots'

def header(entry):
    # Exactly the original aggregate's metadata operands, plus key in its binding.
    return dict(metadata=entry.qkv_metadata,positions=entry.positions,tokens=entry.token_ids,size=entry.size_bytes)

def layout(t):
    storage=t.untyped_storage()
    return (tuple(t.shape),str(t.dtype),str(t.device),tuple(t.stride()) if hasattr(t,'stride') else None,
            t.storage_offset() if hasattr(t,'storage_offset') else None,t.is_contiguous(),storage.data_ptr(),storage.nbytes())

def structure(entry):
    require(isinstance(entry.tensors,dict) and entry.tensors,'missing stored tensors')
    # Hold storage alive too: .data replacement must not free/reuse its address
    # while the verification record is active. Fixture lists have analogous pins.
    return tuple((layer,tuple((t,layout(t),t.values if hasattr(t,'fixture_bytes') else t.untyped_storage()) for t in ts)) for layer,ts in entry.tensors.items())

def same_structure(entry,expected):
    require(isinstance(entry.tensors,dict) and list(entry.tensors)==[x[0] for x in expected],'tensor layer structure changed')
    for layer,ts in expected:
        actual=entry.tensors[layer];require(len(actual)==len(ts),'tensor role structure changed')
        for t,(original,descriptor,storage_pin) in zip(actual,ts):
            require(t is original and layout(t)==descriptor,'tensor object/view/storage identity changed')

def policy_state(entry):
    return (entry.frequency,entry.impact,entry.created_at,entry.last_access,entry.updated_at)

class RequestValidation:
    def __init__(self,owner,mode,entry_proof,tensor_proof):
        require(mode in MODES,'unknown integrity validation mode')
        self.owner=owner;self.mode=mode;self.entry_proof=entry_proof;self.tensor_proof=tensor_proof
        self.epoch=0;self.open=False;self.verified={};self.selected={};self.metrics=None;self.exports=[]
    def start(self,req,selected):
        begin=time.monotonic_ns()
        require(not self.open and not self.verified,'previous validation scope still active')
        self.epoch+=1;self.open=True;self.thread=threading.get_ident();self.token=object();self.request=deepcopy(req)
        self.registry_hash=digest(self.owner.bridge.registry);self.selected={h.entry.key:(h.entry,self.owner.proofs.get(h.entry.key)) for h in selected}
        self.exports=[];self.metrics=dict(validation_mode=self.mode,request_id=req['request_id'],request_sha256=req['request_sha256'],request_epoch=self.epoch,
            request_authorized_entries=0,full_entry_proof_count=0,full_entry_proof_wall_ms=0.,full_entry_serialized_bytes=0,
            consumed_tensor_proof_count=0,consumed_tensor_proof_wall_ms=0.,consumed_tensor_serialized_bytes=0,consumed_snapshot_clone_wall_ms=0.,
            metadata_structure_check_wall_ms=0.,owner_valid_wall_ms=0.,consumed_operation_wall_ms=0.,request_scope_setup_wall_ms=(time.monotonic_ns()-begin)/1e6,request_scope_cleanup_wall_ms=0.,integrity_check_failures=0,admission_entry_proof_count=0,admission_entry_proof_wall_ms=0.,admission_entry_serialized_bytes=0)
    def fail(self):
        if self.metrics is not None:self.metrics['integrity_check_failures']+=1
        self.open=False;self.verified.clear();self.selected.clear();self.exports=[]
    def close(self):
        begin=time.monotonic_ns()
        self.open=False;self.verified.clear();self.selected.clear()
        if self.metrics is not None:self.metrics['request_scope_cleanup_wall_ms']+=(time.monotonic_ns()-begin)/1e6
    def _base(self,entry):
        require(self.open and threading.get_ident()==self.thread,'expired/foreign-thread request verification')
        require(self.owner.active is not None and self.owner.active['req']==self.request,'request identity changed')
        require(digest(self.owner.bridge.registry)==self.registry_hash,'authorization registry changed')
        selected=self.selected.get(entry.key)
        require(selected is not None and selected[0] is entry and self.owner.cache.entries.get(entry.key) is entry,'stale/replaced/unselected entry')
        require(self.owner.proofs.get(entry.key)==selected[1] and selected[1] is not None,'admission proof binding changed')
        binding=self.owner.admission_bindings.get(entry.key)
        require(binding is not None and binding['entry'] is entry and binding['proof']==selected[1],'admission resident object/proof identity changed')
        require(entry.qkv_metadata is binding['metadata_root'] and entry.tensors is binding['tensors_root'],'admission metadata/tensor container replaced')
        m=entry.qkv_metadata
        require(m['physical_identity']==self.request['physical_identity']==self.owner.bridge.identity and m['completed'] and m['component_scope']=='total_qkv','physical ownership/completion differs')
        from .r5_multi_artifacts import sharing
        require(sharing(self.owner.bridge.registry,m['source_user'],self.request['user_id'],m['source_adapter'],self.request['adapter_id']) is None,'unauthorized source')
    def _check(self,entry,record):
        start=time.monotonic_ns()
        try:
            self._base(entry)
            require(record['token'] is self.token and record['entry'] is entry and record['key']==entry.key,'stale request/entry verification record')
            require(entry.qkv_metadata is record['metadata_root'] and entry.tensors is record['tensors_root'],'entry metadata/tensor container replaced')
            require(digest(header(entry))==record['header_hash'] and policy_state(entry)==record['policy'],'entry metadata changed during request')
            same_structure(entry,record['structure'])
        finally:self.metrics['metadata_structure_check_wall_ms']+=(time.monotonic_ns()-start)/1e6
    def full(self,entry,*,admission=False,fingerprints=None):
        # Serialization/hash occurs once per tensor even when exporting fingerprints.
        start=time.monotonic_ns();prefix='admission_entry' if admission else 'full_entry'
        try:byte_count=sum(t.numel()*t.element_size() for ts in (entry.tensors or {}).values() for t in ts)
        except (AttributeError,TypeError):byte_count=0  # Failed/corrupt structure; profiler records attempted calls separately.
        try:
            if fingerprints is None:return self.entry_proof(entry)
            return self.entry_proof(entry,fingerprints=fingerprints)
        finally:
            if self.metrics is not None:
                self.metrics[prefix+'_proof_count']+=1;self.metrics[prefix+'_proof_wall_ms']+=(time.monotonic_ns()-start)/1e6
                self.metrics[prefix+'_serialized_bytes']+=byte_count
    def verify(self,entry):
        try:
            self._base(entry)
            record=self.verified.get(entry.key)
            if record is not None:self._check(entry,record);return record
            require(len(self.verified)<64,'bounded authorization inventory exceeded')
            same_structure(entry,self.owner.admission_bindings[entry.key]['structure'])
            original_header=deepcopy(header(entry));original_structure=structure(entry);original_policy=policy_state(entry);metadata_root=entry.qkv_metadata;tensors_root=entry.tensors
            fingerprints={};actual=call('request_full_entry_validation',self.full,entry,fingerprints=fingerprints)
            require(actual==self.selected[entry.key][1],'cached tensor/metadata integrity failure; no fallback')
            same_structure(entry,original_structure)
            require(entry.qkv_metadata is metadata_root and entry.tensors is tensors_root,'entry containers changed during verification')
            require(header(entry)==original_header and policy_state(entry)==original_policy,'entry changed during verification')
            require(digest(dict(original_header,tensors=fingerprints))==actual,'fingerprint/aggregate proof differs')
            record=dict(token=self.token,entry=entry,key=entry.key,header_hash=digest(original_header),structure=original_structure,policy=original_policy,fingerprints=fingerprints,metadata_root=metadata_root,tensors_root=tensors_root)
            export=dict(request_sha256=self.request['request_sha256'],request_epoch=self.epoch,key=entry.key,admission_proof=actual,header=original_header,tensors=fingerprints)
            export['authorization_id']=digest(export);record['authorization_id']=export['authorization_id']
            self.verified[entry.key]=record;self.exports.append(export);self.metrics['request_authorized_entries']+=1
            return record
        except BaseException:
            self.fail();raise
    def consume(self,entry,layer,role):
        begin=time.monotonic_ns()
        try:
            record=self.verify(entry);tensor=entry.tensors[layer]['qkv'.index(role)]
            start=time.monotonic_ns();owned=call('consumed_authenticated_snapshot_clone',lambda:tensor.detach().clone(),cuda=True)
            self.metrics['consumed_snapshot_clone_wall_ms']+=(time.monotonic_ns()-start)/1e6
            start=time.monotonic_ns()
            try:proof=call('consumed_tensor_validation',self.tensor_proof,owned)
            finally:
                self.metrics['consumed_tensor_proof_count']+=1;self.metrics['consumed_tensor_proof_wall_ms']+=(time.monotonic_ns()-start)/1e6
                self.metrics['consumed_tensor_serialized_bytes']+=owned.numel()*owned.element_size()
            self._check(entry,record)
            require(proof==record['fingerprints'][str(layer)]['qkv'.index(role)],'consumed cached tensor integrity failure; no fallback')
            return owned,dict(authorization_id=record['authorization_id'],expected=record['fingerprints'][str(layer)]['qkv'.index(role)],actual=proof,private_snapshot=True)
        except BaseException:
            if self.open:self.fail()
            raise
        finally:self.metrics['consumed_operation_wall_ms']+=(time.monotonic_ns()-begin)/1e6
    def report(self):
        return dict(schema=SCHEMA,**(self.metrics or dict(validation_mode=self.mode)),authorizations=deepcopy(self.exports),
                    scope='request validation across SETUP/PREFILL/DECODE; admission counted separately; inclusive scopes overlap',threat_model=THREAT_MODEL,
                    request_validation_wall_ms=0. if self.metrics is None else sum(self.metrics[k] for k in ('owner_valid_wall_ms','consumed_operation_wall_ms','request_scope_setup_wall_ms','request_scope_cleanup_wall_ms')),
                    unconsumed_content_after_authorization='NOT_CONTINUOUSLY_REHASHED',records_expired=not self.open)

def compare_validation(experiment,run,audit,verification,other_experiment,other_run,other_audit,other_verification,output_dir,expected_source_commit):
    """Read-only independently sealed EAGER/SCOPED pilot comparison, CPU only."""
    from pathlib import Path
    from .r5_subsequence_diagnostics import history,fresh,projections
    from .r5_subsequence_audit import replay_validation
    from .r5_i5r2_tensors import Evidence
    from .r3a_artifacts import write
    from .r5_i5_evidence import seal
    rows=[];anchors=[];plans=[];raw=[];audits=[]
    for mode,plan_path,run_path,audit_path,verification_path in ((EAGER,experiment,run,audit,verification),(SCOPED,other_experiment,other_run,other_audit,other_verification)):
        p,data,export,anchor=history(plan_path,run_path,audit_path,expected_source_commit)
        require(p['integrity_validation']['mode']==mode,'comparison requires explicit eager then scoped plans')
        ve=Evidence(verification_path);vr=ve.verify();v=ve.json('subsequence_verification.json')
        require(v['status']=='PASS' and v['audit_inventory']==anchor['audit'],'independent verification/audit link differs')
        for condition,item in data.items():
            sources={}
            for req,result in zip(item['requests'],item['results']):
                require(result['validation_mode']==mode,'raw validation mode differs');replay_validation(item['ev'],req,result,sources)
                sources[req['request_id']]=(req,projections(result))
                physical=result['completion']['physical'];validation=None if physical is None else physical['integrity_validation']
                rows.append(dict(validation_mode=mode,condition=condition,request_id=req['request_id'],application_e2e_ms=result['application_e2e_ms'],ttft_ms=result['application_ttft_ms'],prefill_ms=result['prefill_ms'],decode_ms=result['decode_ms'],
                    integrity_validation=None if validation is None else {k:x for k,x in validation.items() if k!='authorizations'},reused_projection_rows=0 if physical is None else physical['physical_projection_rows_reused'],
                    projection_tcp_bytes=sum(r['sent']['wire_bytes']+r['received']['wire_bytes'] for r in result['sparse_receipts']),gpu_uuid=item['config']['physical']['gpu_uuid'],gpu_peak_allocated_bytes=item['config']['gpu_peak_allocated_bytes']))
        plans.append(p);raw.append(data);audits.append(export);anchors.append(dict(inputs=anchor,verification=vr))
    require(plans[0]['tasks']==plans[1]['tasks'] and plans[0]['source']==plans[1]['source'] and plans[0]['mode']==plans[1]['mode']=='EXACT_CONTEXT_VERIFIED','unmatched tasks/execution source/context mode')
    require(plans[0]['diagnostics']['cuda_events']==plans[1]['diagnostics']['cuda_events'],'unequal profiling instrumentation')
    parity=[];physical_equal=True;qkv_equal=True;nonzero=True
    for condition in raw[0]:
        left,right=raw[0][condition],raw[1][condition]
        require({k:v for k,v in left['config']['physical'].items() if k!='gpu_uuid'}=={k:v for k,v in right['config']['physical'].items() if k!='gpu_uuid'},'model/adapter/backend identity differs across modes')
        for a,b in zip(left['results'],right['results']):
            same=left['ev'].tensor('es',a['logits'])==right['ev'].tensor('es',b['logits']) and a['generated_token_ids']==b['generated_token_ids'] and a['generated_text']==b['generated_text']
            parity.append(dict(condition=condition,request_id=a['request_id'],full_logit_bytes_tokens_text_equal=same))
            ar,br=projections(a),projections(b)
            require(set(ar)==set(br),'projection inventory differs across validation modes')
            components={name:all((ar[k].get(name) or {}).get('sha256')==(br[k].get(name) or {}).get('sha256') for k in ar) for name in ('input','fresh_base','fresh_delta','output')}
            parity[-1]['projection_components_equal']=components
            qkv_equal=qkv_equal and all(components.values())
            if a['completion']['physical'] is not None:
                equal=a['matching']['selected_spans']==b['matching']['selected_spans'] and all(ar[k]['fresh_positions']==br[k]['fresh_positions'] and ar[k]['reused_positions']==br[k]['reused_positions'] for k in ar)
                physical_equal=physical_equal and equal
        if condition=='SEMCACHE_SUBSEQUENCE':nonzero=all(x['results'][1]['completion']['physical']['physical_projection_rows_reused']>0 for x in (left,right))
    differences=[]
    for a in rows[:len(rows)//2]:
        b=next(x for x in rows[len(rows)//2:] if (x['condition'],x['request_id'])==(a['condition'],a['request_id']))
        differences.append(dict(condition=a['condition'],request_id=a['request_id'],scoped_minus_eager_ms={field:b[field]-a[field] for field in ('application_e2e_ms','ttft_ms','prefill_ms','decode_ms')}))
    fixture=any(r['TEST_FIXTURE_ONLY'] for r in raw[0]['FULL_RECOMPUTE']['requests'])
    result=dict(schema='R5_D5_COMPARISON_V1',status='COMPARISON_COMPLETE',TEST_FIXTURE_ONLY=fixture,inputs=anchors,rows=rows,differences=differences,parity=parity,
        PHYSICAL_REUSE_PRESERVED=physical_equal and nonzero,QKV_VALUES_UNCHANGED=qkv_equal,NUMERICAL_BEHAVIOR='UNCHANGED' if qkv_equal and all(x['full_logit_bytes_tokens_text_equal'] for x in parity) else 'CHANGED',
        frozen_numerical_fidelity_by_mode={mode:[x for x in report['comparisons'] if x['condition']=='SEMCACHE_SUBSEQUENCE'] for mode,report in zip(MODES,audits)},
        INTEGRITY_SAFETY='LIMITED_TO_DECLARED_THREAT_MODEL',GPU_PERFORMANCE='CPU_FIXTURE_ONLY' if fixture else 'DIAGNOSTIC_MEASUREMENTS_ONLY',CORRECTNESS_QUALIFIED_SPEEDUP='NOT_ESTABLISHED',A0_BREAK_EVEN='NOT_ESTABLISHED',
        limitations='single diagnostic run per mode; fixed A/B/C order; runtime variance; actual GPU UUID differences recorded; nested timers overlap; unused content not continuously rehashed')
    out=fresh(output_dir,(experiment,run,audit,verification,other_experiment,other_run,other_audit,other_verification));write(out/'d5_validation_comparison.json',result)
    text=['# D5 validation comparison','',f"PHYSICAL_REUSE_PRESERVED={result['PHYSICAL_REUSE_PRESERVED']}; NUMERICAL_BEHAVIOR={result['NUMERICAL_BEHAVIOR']}",'',result['limitations'],'','Frozen numerical failures remain failures; no single-run production speedup or fidelity-restoration claim.','', '| Mode | Condition | Request | App E2E ms | Prefill ms | Full proofs | Consumed proofs |','|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        v=r['integrity_validation'] or {};text.append(f"| {r['validation_mode']} | {r['condition']} | {r['request_id']} | {r['application_e2e_ms']} | {r['prefill_ms']} | {v.get('full_entry_proof_count',0)} | {v.get('consumed_tensor_proof_count',0)} |")
    (out/'d5_validation_comparison.md').write_text('\n'.join(text)+'\n');seal(out)
    return dict(status=result['status'],PHYSICAL_REUSE_PRESERVED=result['PHYSICAL_REUSE_PRESERVED'],NUMERICAL_BEHAVIOR=result['NUMERICAL_BEHAVIOR'])
