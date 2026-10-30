"""Predeclared exact counts, immutable order and isolated extension contracts."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import digest,ProtocolError
from semcache.edgelora.r3cf_schedule import audit_schedule,budget,events,make_schedule,validate_schedule


class ScheduleTests(unittest.TestCase):
    def test_exact_126_and_primary_96_auxiliary_16_counts(self):
        manifest=make_schedule();counts=validate_schedule(manifest);b=budget(manifest)
        self.assertEqual(counts,dict(primary_measured=96,primary_warmup=6,condition_gates=6,
            auxiliary_measured=16,auxiliary_fills=2,total=126))
        self.assertEqual(b['model_request_bound'],126);self.assertEqual(b['planned_full_exchange_requests'],101)
        self.assertEqual(b['planned_full_hit_requests'],25);self.assertEqual(b['planned_hidden_frames'],3232)
        arms={(t['condition']['id'],t['arm']):t for t in manifest['schedule'] if t['role']=='PRIMARY'}
        self.assertEqual(len(arms),6)
        for cid,hits in (('C00',2),('C00',4),('C03',0),('C03',1)):
            task=arms[cid,f'CACHE_{hits:02d}_OF_16'];self.assertEqual(task['measured_modes'].count('WARM_HIT'),hits)
            self.assertEqual(task['warmup_modes'],['COLD_MISS'])

    def test_seed_reproducibility_no_grid_expansion_and_tamper_rejection(self):
        self.assertEqual(make_schedule(),make_schedule());self.assertNotEqual(make_schedule(),make_schedule(43))
        for bad in (-1,2**32,True):
            with self.assertRaises(ProtocolError):make_schedule(bad)
        altered=deepcopy(make_schedule());altered['schedule'][0]['measured_modes'].pop()
        altered['schedule_sha256']=digest({k:v for k,v in altered.items() if k!='schedule_sha256'})
        with self.assertRaises(ProtocolError):validate_schedule(altered)

    def test_auxiliary_roles_and_missing_reordered_or_duplicate_rows_fail(self):
        manifest=make_schedule();rows=[dict(e,request_id=f'TEST_FIXTURE_ONLY_{i}',execution_order=i,
            analysis_role=e['role'],schedule_sha256=manifest['schedule_sha256']) for i,e in enumerate(events(manifest))]
        self.assertEqual(audit_schedule(rows,manifest)['total'],126)
        for mutation in ('missing','duplicate','role','order'):
            altered=deepcopy(rows)
            if mutation=='missing':altered.pop()
            if mutation=='duplicate':altered[-1]['request_id']=altered[0]['request_id']
            if mutation=='role':altered[-1]['analysis_role']='PRIMARY'
            if mutation=='order':altered[0],altered[1]=altered[1],altered[0]
            with self.subTest(mutation=mutation),self.assertRaises(ProtocolError):audit_schedule(altered,manifest)

    def test_old_runtime_binding_restored_on_success_and_failure(self):
        from semcache.edgelora import r3c_runtime
        from semcache.edgelora.r3cf_runtime import FocusedExperiment,execute
        original=r3c_runtime.NetworkExperiment
        def check(*args):
            self.assertIs(r3c_runtime.NetworkExperiment,FocusedExperiment);return dict(execution_status='COMPLETE')
        with patch.object(r3c_runtime,'execute',side_effect=check):
            result=execute(dict(fixture=True),None);self.assertFalse(result['PRIMARY_PERFORMANCE_ELIGIBLE'])
        self.assertIs(r3c_runtime.NetworkExperiment,original)
        with patch.object(r3c_runtime,'execute',side_effect=ValueError('TEST_FIXTURE_ONLY failure')):
            with self.assertRaises(ValueError):execute(dict(fixture=True),None)
        self.assertIs(r3c_runtime.NetworkExperiment,original)

    def test_boundary_failure_does_not_copy_prior_success_as_a_new_request(self):
        from semcache.edgelora.r3c_runtime import NetworkExperiment
        from semcache.edgelora.r3cf_runtime import FocusedExperiment
        from semcache.edgelora.r3a_evidence import jsonl
        model=FocusedExperiment.__new__(FocusedExperiment);model.fixture=True
        model.network_rows=[dict(request_id='PRIOR_SUCCESS',observed_qkv_outputs=96)]
        model.plan=dict(schedule_manifest=make_schedule())
        with tempfile.TemporaryDirectory() as tmp:
            model.output=Path(tmp)
            with patch.object(NetworkExperiment,'one',side_effect=TimeoutError('TEST_FIXTURE_ONLY boundary failure')):
                with self.assertRaises(TimeoutError):model.one(make_schedule()['profiles'][0],'NO_REUSE','measured',0,'NO_REUSE')
            rows=jsonl(model.output/'r3cf_requests.jsonl')
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['status'],'FAILED')
            self.assertEqual(rows[0]['request_id'],'NOT_STARTED');self.assertNotIn('observed_qkv_outputs',rows[0])


if __name__=='__main__':unittest.main()
