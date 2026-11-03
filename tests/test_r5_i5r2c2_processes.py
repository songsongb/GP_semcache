"""CPU ES/UD/TCP lifecycle, three independently reset modes; no GPU evidence."""
import unittest,tempfile,subprocess,sys,os
from pathlib import Path
from semcache.edgelora.r3a_artifacts import ROOT,read
from semcache.edgelora.r5_i5r2c2 import prepare,validate,MODES
import test_r5_i5r2b as fixtures
class Processes(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(os.environ.get('C2_TEST_ROOT',cls.tmp.name));cls.root.mkdir(parents=True,exist_ok=True)
  multi,old=fixtures.fixture_inputs(cls.root/'inputs');plans=prepare(old,multi,None,None,None,None,cls.root/'plans',fixture=True)
  p=validate(plans/'OFF')
  result=subprocess.run([sys.executable,str(ROOT/'scripts/138_r5_i5r2c2.py'),'run','--experiment',str(plans),'--multi-prepared',str(multi),'--output-dir',str(cls.root/'run'),'--source-commit',p['i5r2b']['source']['commit'],'--test-fixture-only'],capture_output=True,text=True,timeout=1800)
  (cls.root/'stdout.log').write_text(result.stdout);(cls.root/'stderr.log').write_text(result.stderr)
  if result.returncode:raise AssertionError(str(cls.root)+'\n'+result.stderr)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def test_three_modes_independent_completed_and_audited(self):
  for m in MODES:
   d=read(self.root/'run'/(m+'_c2_audit')/'i5r2c2_decision.json');self.assertTrue(d['TEST_FIXTURE_ONLY']);self.assertEqual(d['PHYSICAL_REUSE_PRESERVED'],'CPU_FIXTURE_ONLY');self.assertEqual(d['independent_audit']['status'],'PASS')
 def test_real_rows_and_extra_native_work_are_separate(self):
  for m,count in (('OFF',0),('SOURCE_ONLY',192*96),('SOURCE_AND_SUFFIX',256*96)):
   a=read(self.root/'run'/(m+'_c2_audit')/'i5r2c2_accounting.json');self.assertEqual(a['real_reused_tokens'],64);self.assertEqual(a['extra_projection_rows'],count)
 def test_baseline_logits_unchanged_across_modes(self):
  from semcache.edgelora.r5_i5_evidence import jsonl
  for condition in ('FULL_RECOMPUTE','SEMANTIC_NO_REUSE'):
   values=[[r['logits_sha256'] for r in jsonl(self.root/'run'/m/condition/'generation.jsonl')] for m in MODES]
   self.assertEqual(values[0],values[1]);self.assertEqual(values[1],values[2])
if __name__=='__main__':unittest.main()
