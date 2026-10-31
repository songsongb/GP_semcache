"""Release regressions for the missing I5-P workload integration."""
import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from semcache.edgelora.r3a_artifacts import ROOT
from semcache.edgelora.r5_i5_workload import SCALE_SEMANTICS,routing,turn_order,pack_groups,continuity,select,requests,prepare_experiment

class WorkloadRelease(unittest.TestCase):
    def test_public_symbols_and_active_call_sites(self):
        self.assertEqual(SCALE_SEMANTICS,'INDEPENDENT_BLOCK_DATASET_SCALE')
        for fn,required in [(select,{'turn_order'}),(requests,{'routing'}),(prepare_experiment,{'pack_groups','continuity','requests'})]:
            calls={n.func.id for n in ast.walk(ast.parse(inspect.getsource(fn))) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
            self.assertTrue(required<=calls,(fn.__name__,calls))
    def test_actual_cli_without_registry_returns_structured_not_ready(self):
        r=subprocess.run([sys.executable,str(ROOT/'scripts/123_preflight_r5_i5.py'),'readiness'],capture_output=True,text=True,timeout=120)
        self.assertEqual(r.returncode,1,r.stderr)
        report=json.loads(r.stdout)
        self.assertEqual(report['status'],'NOT_READY');self.assertTrue(report['GPU_NOT_RUN'])
        self.assertEqual(len(report['datasets']),6)
        self.assertNotIn('ImportError',r.stderr)
    def test_route_ignores_turn_and_request_position(self):
        for dataset,version in [('coqa','1.0'),('multiwoz','2.2')]:
            sample=dict(dataset=dataset,version=version,group_id='dialogue_immutable',id='turn1')
            self.assertEqual(routing(sample,37),routing(dict(sample,id='turn200',turn_id=200),37))
    def test_numeric_order_and_continuity(self):
        rows=[dict(sample=dict(dataset='coqa',version='1.0',group_id='g',id=str(i),turn_id=str(i))) for i in [10,2,1]]
        cohorts=pack_groups(rows,2)
        self.assertEqual([p['sample']['turn_id'] for c in cohorts for p in c],['1','2','10'])
        self.assertEqual(continuity(cohorts)[0]['block_ordinals'],[0,1])
        self.assertFalse(continuity(cohorts)[0]['cache_survives'])
