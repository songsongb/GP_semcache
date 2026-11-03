"""CPU cast rounding and sealed synthetic S1 chain, never GPU fidelity evidence."""
import struct,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
from semcache.edgelora.r3a_artifacts import ROOT,pack
from semcache.edgelora.r5_i5r2_tensors import file_sha
from semcache.edgelora.r5_i5r2c0_analysis import cast_half,compare_four,analyze
from semcache.edgelora.r5_i5r2bs1_plan import prepare
from test_r5_i5r2bs1 import historical_fixture,synthetic_run

class Cast(unittest.TestCase):
 def test_round_to_nearest_even_at_midpoints(self):
  for value,expected in ((1+2**-11,1.),(1+3*2**-11,1+2**-9),(-1-2**-11,-1.),(2**-25,0.),(3*2**-25,2**-23)):
   with self.subTest(value=value):self.assertEqual(cast_half(pack([value],'float32')),pack([expected],'float16'))
 def test_adjacent_float32_values_across_boundary(self):
  midpoint=1+2**-11;bits=struct.unpack('<I',pack([midpoint],'float32'))[0]
  x,y=struct.pack('<I',bits-1),struct.pack('<I',bits+1)
  r=compare_four(pack([1.],'float16'),pack([1.],'float16'),x,y)
  self.assertTrue(r['CAST_PREFIX_STILL_DIFFERENT']);self.assertEqual(r['rounding_boundaries']['adjacent_bin_midpoint_straddles'],1)
 def test_equal_casts_do_not_imply_baseline_compatibility(self):
  r=compare_four(pack([1.],'float16'),pack([1.015625],'float16'),pack([1.],'float32'),pack([1.],'float32'))
  self.assertTrue(r['CAST_PREFIX_BITWISE_EQUAL']);self.assertTrue(r['CAST_PREFIX_NUMERICALLY_CLOSER']);self.assertTrue(r['SOURCE_ONLY_BASELINE_INCOMPATIBLE'])
 def test_identity_and_signed_zero(self):
  x=pack([0.,-0.,65504.,2**-24],'float32');self.assertEqual(cast_half(x),pack([0.,-0.,65504.,2**-24],'float16'))
 def test_invalid_values_fail_closed(self):
  for data in (b'x',struct.pack('<f',float('nan')),struct.pack('<f',float('inf')),pack([1e10],'float32')):
   with self.assertRaises(ValueError):cast_half(data)
 def test_frozen_tolerance_retained(self):
  r=compare_four(pack([1.],'float16'),pack([1.03125],'float16'),pack([1.],'float32'),pack([1.],'float32'))
  self.assertFalse(r['comparisons']['A_vs_B']['activation_tolerance_pass']);self.assertEqual(r['comparisons']['A_vs_B']['differing_elements_percent'],100.)

class EvidenceTests(unittest.TestCase):
 def fixture(self,root):
  run,review,base,commit=historical_fixture(root)
  source=dict(commit=commit,branch='exp/raw-semcache-r5-integration',files={'src/semcache/edgelora/r3b_runtime.py':__import__('hashlib').sha256(subprocess.check_output(['git','show','HEAD:src/semcache/edgelora/r3b_runtime.py'],cwd=ROOT)).hexdigest()})
  with patch('semcache.edgelora.r5_i5r2bs1_plan.source_identity',return_value=source):
   prepare(run,review,base,root/'plan',expected_commit=commit,fixture=True)
   raw=synthetic_run(root/'plan',root/'run')
   from semcache.edgelora.r5_i5r2bs1_analysis import analyze as s1
   s1(root/'plan',raw,root/'s1audit')
  return root/'plan',raw,root/'s1audit'
 def test_complete_chain_all_roles_repetitions_read_only(self):
  with tempfile.TemporaryDirectory() as d:
   p,r,a=self.fixture(Path(d));before=file_sha(r/'i5_output_manifest.json')
   result=analyze(p,r,a,Path(d)/'out');self.assertEqual(len(result['groups']),3)
   self.assertTrue(all(g['repetitions']==6 for g in result['groups'].values()));self.assertEqual(file_sha(r/'i5_output_manifest.json'),before)
   self.assertFalse(result['ROOT_CAUSE_CONFIRMED'])
 def test_corrupt_raw_and_missing_audit_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p,r,a=self.fixture(Path(d));f=next((r/'outputs').iterdir());f.write_bytes(b'bad')
   with self.assertRaises(ValueError):analyze(p,r,a,Path(d)/'bad')
   with self.assertRaises(ValueError):analyze(p,r,Path(d)/'missing',Path(d)/'bad2')

if __name__=='__main__':unittest.main()
