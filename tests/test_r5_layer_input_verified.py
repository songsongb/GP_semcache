"""TEST_FIXTURE_ONLY scalar QKV operators; never measured OPT/GPU evidence."""
from copy import deepcopy
import math
import multiprocessing
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import require, digest, header, Channel, connect
from semcache.edgelora.r3a_tiny import f32
from semcache.edgelora.r3a_artifacts import unpack
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.user_device import cpu_deltas
from semcache.edgelora.r5_sparse_qkv import blob, gather, SparseSession, finish_request
from semcache.edgelora.r5_layer_input_arithmetic import ScalarFixtureArithmetic
from semcache.edgelora.r5_layer_input_physical import LayerInputOwner, MODE, category
from semcache.edgelora.r5_subsequence_physical import EXACT, APPROX, PhysicalOwner
import test_r5_sparse_physical as fixture


class FixtureCompletion:
    TEST_FIXTURE_ONLY = True
    def send(self, h, payload=b''):
        self.last = h; return {'TEST_FIXTURE_ONLY':True}
    def recv(self, **expected):
        c = self.last['metadata']['sparse_complete']
        return header('READY', self.last['request_id'], -1, completion_sha256=digest(c)), b'', {'TEST_FIXTURE_ONLY':True}


class FixtureSession:
    """Same scalar CPU LoRA function, explicit in-process fixture transport."""
    def __init__(self, arithmetic):
        self.arithmetic_binding = arithmetic.binding; self.matrices = deepcopy(arithmetic.matrices)
        self.layer = self.role = 0; self.calls = []; self.bundle = None
    def correction(self, layer, role, x, positions):
        require((layer,role) == (self.layer,'qkv'[self.role]), 'fixture role order')
        self.calls.append((layer,role,list(positions)))
        n,h = len(positions),x.shape[2]
        if self.role == 0:
            self.positions = list(positions); self.input = blob(x)
            self.bundle = cpu_deltas(unpack(blob(gather(x,positions)),'float32'),self.matrices,layer,h,torch_backend=False) if n else None
        else: require(self.positions == positions and self.input == blob(x),'fixture QKV differs')
        result = None if not n else ArrayTensor(self.bundle[self.role*n*h:(self.role+1)*n*h],[1,n,h])
        self.role += 1
        if self.role == 3: self.role = 0; self.layer += 1
        return result
    def complete(self): require(self.layer == 2 and self.role == 0,'fixture incomplete')


def actual_fixture_probabilities(outputs):
    layers = []
    for layer in range(2):
        q,k = outputs[layer,'q'],outputs[layer,'k']; n,h = q.shape[1:]; rows = []
        for i in range(n):
            scores = [sum(q.values[i*h+j]*k.values[p*h+j] for j in range(h))/math.sqrt(h) for p in range(i+1)]
            maximum = max(scores); exps = [math.exp(v-maximum) for v in scores]; denominator = sum(exps)
            rows.append([v/denominator for v in exps]+[0.]*(n-i-1))
        layers.append(rows)
    return layers


def fixture_impact(attentions, start, end, valid_attention_mask=None):
    """Actual toy attention probabilities; original paper-row-L2 formula, one head."""
    return sum(sum(math.sqrt(sum(v*v for v in layer[t][:t+1])) for t in range(start,end)) for layer in attentions)


class Verified(unittest.TestCase):
    def setUp(self):
        self.f = fixture.Physical(); self.f.setUp(); self.events = []
        self.weights = {(l,r): [f32(.2 if i%5 == 0 else .01) for i in range(16)] for l in range(2) for r in 'qkv'}
        self.matrices = {k:[f32(v) for v in vals] for k,vals in self.f.matrices.items()}
        self.arithmetic = ScalarFixtureArithmetic(self.f.ident,self.weights,self.matrices)
        self.owner = self.new_owner(self.arithmetic)

    def new_owner(self, arithmetic, **kwargs):
        return LayerInputOwner(self.f.ident,self.f.registry,arithmetic=arithmetic,trace=self.events.append,**kwargs)

    def inputs(self, ids, layer, *, shifted_equal=False, offset=0.):
        return ArrayTensor([f32(token*.1+j*.01+(0 if shifted_equal else p*.03)+layer*.07+offset)
                            for p,token in enumerate(ids) for j in range(4)],[1,len(ids),4])

    def project(self, owner, req, *, later_mismatch=False, shifted_equal=False, offset=0., session=None):
        owner.begin(req,{'cluster_id':0},mode=MODE)
        session = session or FixtureSession(owner.arithmetic); outputs = {}
        for l in range(2):
            x = self.inputs(req['token_ids'],l,shifted_equal=shifted_equal,offset=offset+(10. if later_mismatch and l else 0.))
            for r in 'qkv': outputs[l,r] = owner.project(l,r,x,owner.arithmetic.native(l,r),session)
        session.complete(); return outputs,session

    def complete(self, owner, req, outputs, channel=None):
        c = owner.completion_channel(channel or FixtureCompletion())
        finish_request(c,req,[3])
        with patch('semcache.edgelora.r5_subsequence_physical.actual_attention_impact',side_effect=fixture_impact):
            return owner.complete(completed=True,attentions=actual_fixture_probabilities(outputs))

    def run_req(self, rid, ids, *, owner=None, user='User_01', **kwargs):
        owner = owner or self.owner; req = self.f.req(rid,ids,user=user)
        outputs, session = self.project(owner,req,**kwargs)
        return self.complete(owner,req,outputs),outputs,session

    def source(self, **kwargs): return self.run_req('SOURCE',[1,2,3,4,5,6],**kwargs)

    def test_mode_explicit_and_legacy_owner_rejects_new_mode(self):
        req = self.f.req('R',[1,2,3])
        for mode in (None,EXACT,APPROX):
            with self.assertRaises(ValueError): self.owner.begin(req,{'cluster_id':0},mode=mode)
        with self.assertRaises(ValueError): self.owner.begin(req,{'cluster_id':0},mode=MODE,allow_approximate=True)
        legacy = PhysicalOwner(self.f.ident,self.f.registry)
        with self.assertRaises(ValueError): legacy.begin(req,{'cluster_id':0},mode=MODE)
        with self.assertRaises(ValueError): self.new_owner(object())

    def test_same_position_nonprefix_verified_equal_inputs(self):
        self.source(); result,outputs,_ = self.run_req('TARGET',[9,2,3,4,5,6],user='User_02')
        self.assertGreater(result['reused_projection_rows_by_category']['B'],0)
        self.assertEqual(result['reused_projection_rows_by_category']['C'],0)
        self.assertGreater(result['projection_rows_skipped'],0)
        # Independent all-fresh source in an isolated owner has identical values.
        other = self.new_owner(self.arithmetic)
        _,fresh,_ = self.run_req('FRESH',[9,2,3,4,5,6],owner=other)
        self.assertEqual([blob(outputs[k]) for k in sorted(outputs)],[blob(fresh[k]) for k in sorted(fresh)])

    def test_later_layer_mismatch_recomputes(self):
        self.source(); r,_,_ = self.run_req('TARGET',[9,2,3,4,5,6],later_mismatch=True)
        self.assertTrue(any(x['reused_positions'] for x in r['records'] if x['layer'] == 0))
        self.assertTrue(all(not x['reused_positions'] for x in r['records'] if x['layer'] == 1))
        self.assertTrue(any(d['reason']=='PER_LAYER_INPUT_BYTES_MISMATCH' for x in r['records'] for d in x['decisions']))

    def test_shifted_position_unequal_inputs_is_fresh(self):
        self.source(); r,_,_ = self.run_req('SHIFT',[9,1,2,3,4,5,6])
        self.assertEqual(r['physical_projection_rows_reused'],0)
        self.assertTrue(any(d['category']=='C' for row in r['records'] for d in row['decisions']))

    def test_synthetic_shifted_equal_input_edge_case(self):
        self.source(shifted_equal=True)
        r,_,_ = self.run_req('SHIFT',[9,1,2,3,4,5,6],shifted_equal=True)
        self.assertGreater(r['reused_projection_rows_by_category']['C'],0)
        self.assertEqual(r['reused_projection_rows_by_category']['B'],0)

    def test_complete_fresh_fallback_and_mixed_rows(self):
        self.source(); r,_,s = self.run_req('MIXED',[9,2,3,4,5,6])
        self.assertTrue(any(0<len(row['reused_positions'])<6 for row in r['records']))
        self.assertEqual(sum(len(p) for _,_,p in s.calls),r['native_projection_rows'])
        r,_,_ = self.run_req('UNEQUAL',[9,2,3,4,5,6],offset=100.)
        self.assertEqual(r['physical_projection_rows_reused'],0)

    def test_signed_zero_requires_literal_input_bytes(self):
        self.inputs = lambda ids,layer,**kwargs:ArrayTensor([-0.0]*(len(ids)*4),[1,len(ids),4])
        self.source()
        self.inputs = lambda ids,layer,**kwargs:ArrayTensor([0.0]*(len(ids)*4),[1,len(ids),4])
        r,_,_ = self.run_req('ZERO',[1,2,3,4,5,6])
        self.assertEqual(r['physical_projection_rows_reused'],0)

    def test_reference_guard_consumption_is_not_saved_work(self):
        arithmetic = ScalarFixtureArithmetic(self.f.ident,self.weights,self.matrices,reference_guard=True)
        owner = self.new_owner(arithmetic); self.source(owner=owner)
        r,_,_ = self.run_req('TARGET',[9,2,3,4,5,6],owner=owner)
        self.assertGreater(r['physical_projection_rows_reused'],0)
        self.assertEqual(r['native_projection_rows'],6*2*3)
        self.assertEqual(r['projection_rows_skipped'],0)
        self.assertEqual(r['reference_projection_rows'],36)

    def test_shape_dependent_reference_outputs_fail_closed_to_fresh(self):
        arithmetic = ScalarFixtureArithmetic(self.f.ident,self.weights,self.matrices,reference_guard=True,shape_offset=True)
        owner = self.new_owner(arithmetic); self.source(owner=owner,shifted_equal=True)
        r,_,_ = self.run_req('TARGET',[9,1,2,3,4,5,6],owner=owner,shifted_equal=True)
        self.assertEqual(r['physical_projection_rows_reused'],0)
        self.assertTrue(any(d['reason']=='FULL_SHAPE_OUTPUT_BYTES_MISMATCH' for x in r['records'] for d in x['decisions']))
        self.assertFalse(any(e['kind']=='CHU' for e in self.events))

    def test_empty_and_fully_reused_row_sets(self):
        self.run_req('SOURCE',[1,2,3]); r,_,session = self.run_req('TARGET',[1,2,3])
        self.assertEqual(r['physical_projection_rows_reused'],18)
        self.assertEqual(r['native_projection_rows'],0)
        self.assertTrue(all(not positions for _,_,positions in session.calls))

    def test_unsupported_dtype_shape_layout_and_nonfinite(self):
        for kind in ('dtype','shape','layout','nan'):
            owner = self.new_owner(self.arithmetic); req = self.f.req('BAD_'+kind,[1,2,3]); owner.begin(req,{'cluster_id':0},mode=MODE)
            x = self.inputs(req['token_ids'],0)
            if kind == 'dtype': x.dtype = 'float16'
            if kind == 'shape': x = ArrayTensor([1.]*8,[1,2,4])
            if kind == 'layout': x.is_contiguous = lambda:False
            if kind == 'nan': x.values[0] = float('nan')
            with self.subTest(kind=kind),self.assertRaises(ValueError): owner.project(0,'q',x,self.arithmetic.native(0,'q'),FixtureSession(self.arithmetic))
            self.assertIsNone(owner.active)

    def test_changed_weights_bias_adapter_and_backend_binding(self):
        for kind in ('weights','bias','matrices','binding'):
            arithmetic = ScalarFixtureArithmetic(self.f.ident,self.weights,self.matrices); owner = self.new_owner(arithmetic)
            req = self.f.req('BAD',[1,2,3]); owner.begin(req,{'cluster_id':0},mode=MODE); session = FixtureSession(arithmetic)
            if kind == 'weights': arithmetic.weights[0,'q'][0] += 1.
            if kind == 'bias': arithmetic.biases[0,'q'][0] += 1.
            if kind == 'matrices': session.matrices[0,'q','A'][0] += 1.
            if kind == 'binding': session.arithmetic_binding = 'FOREIGN_BACKEND'
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                owner.project(0,'q',self.inputs([1,2,3],0),arithmetic.native(0,'q'),session)

    def test_current_target_identity_cannot_change(self):
        self.source(); req = self.f.req('TARGET',[9,2,3,4,5,6]); self.owner.begin(req,{'cluster_id':0},mode=MODE)
        self.owner.active['req']['request_sequence_number'] += 1
        with self.assertRaises(ValueError): self.owner.project(0,'q',self.inputs(req['token_ids'],0),self.arithmetic.native(0,'q'),FixtureSession(self.arithmetic))

    def test_qkv_identical_input_contract_is_verified(self):
        req = self.f.req('BAD',[1,2,3]); self.owner.begin(req,{'cluster_id':0},mode=MODE); s = FixtureSession(self.arithmetic)
        self.owner.project(0,'q',self.inputs([1,2,3],0),self.arithmetic.native(0,'q'),s)
        with self.assertRaisesRegex(ValueError,'Q/K/V actual input bytes'):
            self.owner.project(0,'k',self.inputs([1,2,3],0,offset=1.),self.arithmetic.native(0,'k'),s)

    def test_missing_altered_input_proof_and_literal_storage(self):
        for kind in ('missing','hash','storage'):
            owner = self.new_owner(self.arithmetic); self.source(owner=owner)
            entry = next(iter(owner.cache.entries.values()))
            if kind == 'missing': del entry.qkv_metadata['input_proofs']['0']
            if kind == 'hash': entry.qkv_metadata['input_proofs']['0']['sha256']='0'*64
            if kind == 'storage': owner.input_storage[entry.key]=tuple()
            req = self.f.req('TARGET',[1,2,3,4,5,6]); owner.begin(req,{'cluster_id':0},mode=MODE)
            with self.subTest(kind=kind),self.assertRaises((ValueError,KeyError)):
                owner.project(0,'q',self.inputs(req['token_ids'],0),self.arithmetic.native(0,'q'),FixtureSession(self.arithmetic))

    def test_corrupt_tensor_before_and_after_authorization(self):
        for after in (False,True):
            owner = self.new_owner(self.arithmetic); self.source(owner=owner)
            req = self.f.req('TARGET',[1,2,3,4,5,6]); owner.begin(req,{'cluster_id':0},mode=MODE)
            entry = owner.active['selected'][0].entry
            if after: owner.valid(entry)
            entry.tensors[0][0].values[0] += 1.
            with self.subTest(after=after),self.assertRaises(ValueError):
                owner.project(0,'q',self.inputs(req['token_ids'],0),self.arithmetic.native(0,'q'),FixtureSession(self.arithmetic))

    def test_evicted_entry_and_stale_authorization(self):
        self.source(); req = self.f.req('TARGET',[1,2,3,4,5,6]); self.owner.begin(req,{'cluster_id':0},mode=MODE)
        entry = self.owner.active['selected'][0].entry; self.owner.valid(entry)
        del self.owner.cache.entries[entry.key]
        with self.assertRaises(ValueError): self.owner.validation.consume(entry,0,'q')
        self.owner.abort()
        with self.assertRaises(ValueError): self.owner.validation.consume(entry,0,'q')

    def test_cross_adapter_and_denied_cross_user(self):
        self.source(); req = self.f.req('FOREIGN',[1,2,3],user='User_11')
        with self.assertRaises(ValueError): self.owner.begin(req,{'cluster_id':0},mode=MODE)
        self.f.registry['users']['User_02']['allow_cross_user_reuse'] = False
        r,_,_ = self.run_req('DENIED',[1,2,3,4,5,6],user='User_02')
        self.assertEqual(r['physical_projection_rows_reused'],0)

    def test_duplicate_key_first_owner_and_nonoverlap(self):
        self.run_req('SOURCE',[1,1,1,1,1,1]); self.assertEqual(len(self.owner.cache.entries),1)
        entry = next(iter(self.owner.cache.entries.values())); original = entry.qkv_metadata['source_request_id']
        r,_,_ = self.run_req('TARGET',[1,1,1,1,1,1])
        for row in r['records']:
            spans = [set(range(*i['target_positions'])) for i in row['consumed']]
            self.assertEqual(sum(map(len,spans)),len(set().union(*spans)))
        self.assertEqual(entry.qkv_metadata['source_request_id'],original)
        self.assertEqual(self.owner.bridge.max_candidates,64); self.assertEqual(self.owner.bridge.pbr_interval,100)

    def test_failed_completion_and_no_premature_admission(self):
        req = self.f.req('SOURCE',[1,2,3]); outputs,_ = self.project(self.owner,req)
        self.assertFalse(self.owner.cache.entries)
        with self.assertRaisesRegex(ValueError,'completion ACK'):
            self.owner.complete(completed=True,attentions=actual_fixture_probabilities(outputs))
        self.assertFalse(self.owner.cache.entries); self.assertIsNone(self.owner.active)

    def test_bad_completion_ack_and_transitive_admission_restriction(self):
        class Bad(FixtureCompletion):
            def recv(self,**expected):
                h,b,r = super().recv(**expected); h['metadata']['completion_sha256']='0'*64; return h,b,r
        req = self.f.req('SOURCE',[1,2,3]); outputs,_ = self.project(self.owner,req)
        with self.assertRaises(ValueError): self.complete(self.owner,req,outputs,Bad())
        self.owner.abort(); self.run_req('SOURCE_VALID',[1,2,3,4,5,6])
        r,_,_ = self.run_req('TARGET',[9,2,3,4,5,6])
        self.assertGreater(r['physical_projection_rows_reused'],0)
        self.assertTrue(all(e.qkv_metadata['source_request_id']=='SOURCE_VALID' for e in self.owner.cache.entries.values()))
        self.assertTrue(any(e['kind']=='VERIFIED_ADMISSION_SKIPPED' for e in self.events))

    def test_chu_only_after_consumption_and_completed_request(self):
        self.source(); entry = next(iter(self.owner.cache.entries.values())); before = entry.frequency
        req = self.f.req('TARGET',[1,2,3,4,5,6]); outputs,_ = self.project(self.owner,req)
        self.assertEqual(entry.frequency,before)
        self.complete(self.owner,req,outputs)
        self.assertGreater(entry.frequency,before)
        self.assertTrue(any(e['kind']=='CHU' for e in self.events)); self.assertFalse(any(e['kind']=='PBR' for e in self.events))

    def test_unique_repeated_invocations_keep_owner_and_counter(self):
        root = self.owner.cache; self.source()
        for n in range(4): self.run_req('INVOCATION_'+str(n),[1,2,3,4,5,6])
        self.assertIs(self.owner.cache,root); self.assertEqual(self.owner.cache.now,5)
        with self.assertRaises(ValueError): self.owner.begin(self.f.req('INVOCATION_0',[1,2,3]),{'cluster_id':0},mode=MODE)
        self.assertEqual(len(self.owner.bridge.seen),5)

    def test_approximate_or_legacy_sources_are_not_promoted(self):
        for mode in (EXACT,APPROX):
            owner = self.new_owner(self.arithmetic); self.source(owner=owner)
            # Explicit fixture re-sealing represents an authenticated historical
            # source. Production never rewrites provenance or imports this state.
            from semcache.edgelora.r5_subsequence_physical import entry_proof
            for e in owner.cache.entries.values():
                e.qkv_metadata['provenance_mode']=mode
                owner.proofs[e.key]=entry_proof(e); owner.admission_bindings[e.key]['proof']=owner.proofs[e.key]
            r,_,_ = self.run_req('TARGET',[9,2,3,4,5,6],owner=owner)
            self.assertEqual(r['physical_projection_rows_reused'],0)
            self.assertTrue(any(d['reason']=='SOURCE_PROVENANCE_MODE_MISMATCH' for row in r['records'] for d in row['decisions']))

    def test_input_storage_capacity_and_disposal(self):
        self.source()
        self.assertGreater(self.owner.cache.logical_cache_bytes,self.owner.cache.physical_tensor_bytes)
        self.assertLessEqual(self.owner.cache.logical_cache_bytes,512*2**20)
        self.assertTrue(self.owner.input_storage); self.owner.dispose()
        self.assertFalse(self.owner.input_storage); self.assertFalse(self.owner.input_bindings)
        self.assertFalse(self.owner.cache.entries)

    def test_policy_eviction_cleans_input_storage_and_entry_bound(self):
        for capacity,ids in ((1800,[1,2,3,4,5,6]),(512*2**20,list(range(70)))):
            owner=self.new_owner(self.arithmetic,capacity=capacity)
            self.run_req('ROOT_'+str(capacity),ids,owner=owner)
            self.assertLessEqual(len(owner.cache.entries),64)
            self.assertLessEqual(owner.cache.logical_cache_bytes,capacity)
            self.assertEqual(set(owner.input_storage),set(owner.cache.entries))
            self.assertEqual(set(owner.input_bindings),set(owner.cache.entries))
            self.assertTrue(any(e['kind']=='EVICT' for e in self.events))

    def test_real_fixture_tcp_completion_and_physical_rows(self):
        ctx = multiprocessing.get_context('fork'); parent,kid = ctx.Pipe()
        process = ctx.Process(target=fixture.child,args=(kid,self.f.ident,self.f.registry,self.matrices)); process.start(); channel=None
        try:
            self.assertTrue(parent.poll(10),'fixture CPU UD missing')
            channel = Channel(connect(parent.recv()),'fixture_session',self.f.ident['adapter_matrix_fingerprint'])
            results = []
            for rid,ids in (('SOURCE',[1,2,3,4,5,6]),('TARGET',[9,2,3,4,5,6])):
                req = self.f.req(rid,ids); self.owner.begin(req,{'cluster_id':0},mode=MODE)
                bound = self.owner.completion_channel(channel)
                session = SparseSession(bound,req,rid+'_PREFILL',fixture=True); session.arithmetic_binding=self.arithmetic.binding
                outputs={}
                for l in range(2):
                    x = self.inputs(ids,l)
                    for r in 'qkv': outputs[l,r]=self.owner.project(l,r,x,self.arithmetic.native(l,r),session)
                session.complete(); finish_request(bound,req,[3])
                with patch('semcache.edgelora.r5_subsequence_physical.actual_attention_impact',side_effect=fixture_impact):
                    results.append(self.owner.complete(completed=True,attentions=actual_fixture_probabilities(outputs)))
            self.assertEqual(results[0]['physical_projection_rows_reused'],0)
            self.assertGreater(results[1]['reused_projection_rows_by_category']['B'],0)
            channel.send(header('STOP','shutdown',-1)); channel.recv(kind='BYE')
            self.assertTrue(parent.poll(10)); parent.recv()
        finally:
            if channel: channel.sock.close()
            process.join(10)
            if process.is_alive(): process.terminate(); process.join()
            parent.close()
        self.assertEqual(process.exitcode,0)


if __name__=='__main__': unittest.main()
