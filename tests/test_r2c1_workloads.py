from types import SimpleNamespace as NS
import unittest
from semcache.diagnostics.r2c1_workloads import lengths,coverages,expected_tokens,memory_estimate,check_coverage


def physical(n,reused,layers=2):
    provenance=[dict(query='source',source=[i,i+3],target=[i,i+3],tokens=list(range(i,i+3)),
        prefix_token_ids_equal=True,adapter_identity_match=True) for i in range(0,reused,3)]
    return dict(reused_tokens=reused,selected_provenance=provenance,projection_audit=dict(
        skipped_projection_rows_all_layers_roles=reused*3*layers,native_projection_rows_all_layers_roles=(n-reused)*3*layers))


class Workloads(unittest.TestCase):
    def test_lengths_positions_and_fractions(self):
        self.assertEqual(lengths(None,'pilot'),[512,1024]); self.assertEqual(lengths(None,'smoke'),[32,64])
        for value in ('2049','4096','512,512'):
            with self.assertRaises(ValueError): lengths(value,'pilot')
        with self.assertRaises(ValueError): lengths('2048','pilot',1536)
        self.assertEqual(coverages('0,.5,max'),[0.,.5,'max'])
        for n,max_tokens in [(512,510),(1024,1023),(1536,1536),(2048,2046)]:
            self.assertEqual(expected_tokens(n,'max'),max_tokens)
            self.assertLessEqual(expected_tokens(n,.5)/n,.5)

    def test_physical_coverage_zero_partial_max_alignment(self):
        for reused in (0,15,30):
            e=check_coverage(physical(32,reused),list(range(32)),2)
            self.assertEqual(e['reused_tokens'],reused); self.assertEqual(e['native_projection_rows'],(32-reused)*6)
        for mutation in ('position','prefix','tokens','overlap','row_count'):
            r=physical(32,6)
            if mutation=='position': r['selected_provenance'][0]['source']=[3,6]
            if mutation=='prefix': r['selected_provenance'][0]['prefix_token_ids_equal']=False
            if mutation=='tokens': r['selected_provenance'][0]['tokens']=[99]*3
            if mutation=='overlap': r['selected_provenance'].append(r['selected_provenance'][0]); r['reused_tokens']=9
            if mutation=='row_count': r['projection_audit']['skipped_projection_rows_all_layers_roles']=0
            with self.assertRaises(ValueError): check_coverage(r,list(range(32)),2)

    def test_memory_has_raw_qkv_and_profiler_reserve(self):
        cfg=NS(num_hidden_layers=32,hidden_size=2560,num_attention_heads=32,vocab_size=50272)
        a=memory_estimate(cfg,1024,2,512*1024**2); b=memory_estimate(cfg,1024,2,512*1024**2,True)
        self.assertEqual(a['source_qkv_backing_bytes'],3*32*1024*2560*2)
        self.assertEqual(b['estimated_additional_bytes']-a['estimated_additional_bytes'],1024**3)
        self.assertIsNone(a['metadata_python_bytes'])


if __name__=='__main__': unittest.main()
