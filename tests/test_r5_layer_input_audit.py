"""TEST_FIXTURE_ONLY independent value primitive; no OPT/GPU conclusions."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import os
from unittest.mock import patch

from semcache.edgelora.communication import digest, digest_bytes
from semcache.edgelora.r5_sparse_qkv import blob
from semcache.edgelora.r5_layer_input_arithmetic import ScalarFixtureArithmetic
from semcache.edgelora.r5_layer_input_audit import replay_projection, static_feasibility
from semcache.edgelora.r5_subsequence_natural_io import put, seal
import test_r5_layer_input_verified as fixtures


class MemoryEvidence:
    def __init__(self, registry): self.registry=registry; self.data={}
    def save(self,tensor,name):
        data=blob(tensor); self.data[name]=data
        return dict(file=name,shape=list(tensor.shape),dtype=str(tensor.dtype).removeprefix('torch.'),sha256=digest_bytes(data))
    def tensor(self,base,spec):
        data=self.data[spec['file']]
        if digest_bytes(data)!=spec['sha256']: raise ValueError('fixture evidence SHA differs')
        return data


class Audit(unittest.TestCase):
    def scenario(self,reference_guard=False):
        h=fixtures.Verified('runTest'); h.setUp(); ev=MemoryEvidence(h.f.registry)
        arithmetic=ScalarFixtureArithmetic(h.f.ident,h.weights,h.matrices,reference_guard=reference_guard)
        owner=h.new_owner(arithmetic,evidence=ev)
        sr=h.f.req('SOURCE',[1,2,3,4,5,6]); outputs,_=h.project(owner,sr)
        source=h.complete(owner,sr,outputs)
        sources={'SOURCE':dict(request=sr,projections={(r['layer'],r['role']):r for r in source['records']},
            completion_control=dict(request=sr,generated_token_ids=[3],termination='LENGTH_CAP'))}
        req=h.f.req('TARGET',[9,2,3,4,5,6]); outputs,_=h.project(owner,req); physical=h.complete(owner,req,outputs)
        return ev,physical,req,sources

    def test_scalar_and_paid_reference_value_replay(self):
        for guard in (False,True):
            ev,physical,req,sources=self.scenario(guard)
            results=[replay_projection(ev,r,req,physical,sources) for r in physical['records']]
            self.assertEqual(sum(r['reused_by_category']['B'] for r in results),physical['reused_projection_rows_by_category']['B'])
            self.assertGreater(physical['physical_projection_rows_reused'],0)
            self.assertEqual(sum(r['skipped_rows'] for r in results),physical['projection_rows_skipped'])

    def test_false_work_counts_category_input_equality_and_source_completion_rejected(self):
        for kind in ('work','category','input','completion','transitive','authorization'):
            ev,physical,req,sources=self.scenario()
            record=next(r for r in physical['records'] if r['consumed'])
            if kind=='work': record['native_projection_rows']+=1
            if kind=='category': record['consumed'][0]['category']='C'
            if kind=='input': record['consumed'][0]['target_input_proof']['sha256']='0'*64
            if kind=='completion': sources['SOURCE']['completion_control']['generated_token_ids']=[4]
            if kind=='transitive': sources['SOURCE']['projections'][0,'q']['consumed']=[{'TEST_FIXTURE_ONLY':True}]
            if kind=='authorization': physical['integrity_validation']['authorizations'][0]['admission_proof']='0'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError): replay_projection(ev,record,req,physical,sources)

    def test_tampered_saved_snapshot_and_reference_are_rejected(self):
        for kind in ('snapshot','reference'):
            ev,physical,req,sources=self.scenario(reference_guard=True)
            record=next(r for r in physical['records'] if r['consumed'])
            spec=record['consumed'][0]['tensor'] if kind=='snapshot' else record['full_shape_reference']
            ev.data[spec['file']]=b'changed'
            with self.subTest(kind=kind),self.assertRaises(ValueError): replay_projection(ev,record,req,physical,sources)

    def test_missing_real_audit_is_blocked_and_fixture_cannot_pass_as_real(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(static_feasibility(root/'missing',root/'audit')['status'],'BLOCKED')
            (root/'plan').mkdir(); (root/'audit').mkdir()
            p=dict(schema='D5_CONTINUOUS_NATURAL_PHASE2_CPU_V1',TEST_FIXTURE_ONLY=True)
            p['plan_id']=digest(p); put(root/'plan/phase2_plan.json',p); seal(root/'plan'); seal(root/'audit')
            with self.assertRaises(ValueError): static_feasibility(root/'plan',root/'audit')

    def test_relative_paths_normalized_before_shared_verifier(self):
        with tempfile.TemporaryDirectory(prefix='TEST_FIXTURE_ONLY_') as tmp:
            root=Path(tmp);(root/'plan').mkdir();(root/'audit').mkdir()
            with patch('semcache.edgelora.r5_layer_input_audit.verify',side_effect=ValueError('TEST_FIXTURE_STOP')) as verifier:
                with self.assertRaisesRegex(ValueError,'TEST_FIXTURE_STOP'):
                    static_feasibility(os.path.relpath(root/'plan'),os.path.relpath(root/'audit'))
                self.assertTrue(verifier.call_args.args[0].is_absolute())
            (root/'link').symlink_to(root/'plan',target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'source root symlink'):
                static_feasibility(root/'link',root/'audit')


if __name__=='__main__': unittest.main()
