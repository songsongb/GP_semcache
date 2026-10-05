import copy
import importlib.util
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1]/'scripts/57_analyze_cachegen_c6b3_3_transport.py'
SPEC = importlib.util.spec_from_file_location('b33_transport',PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture(r=100,p=20,c=120):
    return dict(stage='C6-B3-2',transport_accounting={mode:dict(raw_total_delta_bytes=r,
        compressed_total_delta_bytes=p,per_packet_cdf_bytes=c,transmitted_total_including_cdf_bytes=p+c,
        transport_compression_ratio=r/(p+c),transport_byte_reduction_percentage=100*(1-(p+c)/r)) for mode in m.MODES})


class Tests(unittest.TestCase):
    def test_replay_factors_and_upper_bound(self):
        rows,br=m.analyze(fixture())
        self.assertEqual(len(rows),14)
        self.assertEqual([(r['mode'],r['amortization_n']) for r in rows],[(mode,n) for mode in m.MODES for n in m.FACTORS])
        self.assertEqual(list(m.FACTORS),[1,2,4,8,16,32,'inf'])
        first=rows[0];last=rows[6]
        self.assertEqual(first['analytical_total_bytes'],140)
        self.assertEqual(first['compression_ratio'],Fraction(5,7))
        self.assertEqual(first['byte_reduction_percentage'],-40)
        self.assertFalse(first['net_saving'])
        self.assertEqual(first['provenance'],'MEASURED_ACCOUNTING_REPLAY')
        self.assertEqual(rows[1]['provenance'],'ANALYTICAL_CDF_AMORTIZATION')
        self.assertEqual(last['provenance'],'ANALYTICAL_PRESHARED_CDF_UPPER_BOUND')
        self.assertEqual(last['amortized_cdf_bytes'],0)
        self.assertEqual(last['analytical_total_bytes'],20)
        self.assertEqual(last['compression_ratio'],br[0]['payload_only_compression_ratio'])
        self.assertEqual(last['byte_reduction_percentage'],80)
        self.assertEqual(br[0]['continuous_break_even_threshold_n'],Fraction(3,2))
        self.assertEqual(br[0]['minimum_integer_n_for_positive_saving'],2)
        self.assertEqual(br[0]['bytes_saved_at_minimum_integer_n'],20)

    def test_strict_integer_threshold(self):
        rows,br=m.analyze(fixture(c=160))
        self.assertEqual(br[0]['continuous_break_even_threshold_n'],2)
        self.assertEqual(br[0]['minimum_integer_n_for_positive_saving'],3)
        self.assertFalse(rows[1]['net_saving'])
        self.assertEqual(rows[1]['byte_reduction_percentage'],0)

    def test_zero_cdf_and_impossible_saving(self):
        self.assertEqual(m.analyze(fixture(c=0))[1][0]['minimum_integer_n_for_positive_saving'],1)
        for payload in (100,110):
            result=m.analyze(fixture(p=payload))[1][0]
            self.assertIsNone(result['minimum_integer_n_for_positive_saving'])
            self.assertIsNone(result['continuous_break_even_threshold_n'])

    def test_invalid_input(self):
        for field in m.FIELDS:
            bad=fixture();del bad['transport_accounting'][m.MODES[0]][field]
            with self.assertRaises(ValueError):m.analyze(bad)
        for mode in m.MODES:
            bad=fixture();del bad['transport_accounting'][mode]
            with self.assertRaises(ValueError):m.analyze(bad)
        for field in m.FIELDS[3:]:
            bad=fixture();bad['transport_accounting'][m.MODES[0]][field]+=1
            with self.assertRaisesRegex(ValueError,'N=1 reproduction failed'):m.analyze(bad)
        for value in (True,'100',Decimal('NaN'),Decimal('Infinity'),-1,0):
            bad=fixture();bad['transport_accounting'][m.MODES[0]]['raw_total_delta_bytes']=value
            with self.assertRaises(ValueError):m.analyze(bad)
        bad=fixture();bad['stage']='OTHER'
        with self.assertRaises(ValueError):m.analyze(bad)

    def test_precision_and_cli_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input.json';source.write_text(json.dumps(fixture()))
            original=source.read_bytes();out=root/'output'
            m.main(['--summary',str(source),'--output-root',str(out)])
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(len(json.loads((out/'transport_amortization.json').read_text())),14)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['stage'],'C6-B3-3')
            self.assertFalse(manifest['model_inference_performed'])
            for name,sha in manifest['output_hashes'].items():self.assertEqual(m.file_hash(out/name),sha)
            self.assertIn('does NOT establish',(out/'summary.md').read_text())
            with self.assertRaises(ValueError):m.main(['--summary',str(source),'--output-root',str(out)])
        encoded=m.json_text({'fraction':Fraction(1,3)})
        self.assertGreater(len(encoded.split(': ')[1].rstrip('}')),40)
        self.assertEqual(json.loads(encoded,parse_float=Decimal)['fraction'],Decimal('0.'+'3'*50))

if __name__=='__main__':unittest.main()
