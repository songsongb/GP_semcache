"""Conversation routing, continuity, real readiness fail-closed and sealed array bounds."""
import tempfile
from pathlib import Path
from copy import deepcopy
import unittest
from unittest.mock import patch
from test_r5_i5 import samples,dataset
from semcache.edgelora.r4c_prepare import TinyTokenizer
from semcache.edgelora.r5_i5_workload import routing,turn_order,pack_groups,continuity,prepare_experiment,SCALE_SEMANTICS
from semcache.edgelora.r5_i5_preflight import readiness,inspect_plan,scorer_smokes

class PatchTests(unittest.TestCase):
    def test_conversation_sticky_and_registry(self):
        for name in ('coqa','multiwoz'):
            s=samples(name)[0];r=routing(s)
            self.assertEqual(r,routing(dict(s,id='another_turn',turn_id=100)))
            n=int(r['user_id'][5:]);self.assertEqual(r['adapter_id'],'Adapter_'+'ABC'[(n-1)//10])
    def test_nonconversation_sample_identity(self):
        for name in ('snips','qasper','2wikimqa','multi_news'):
            s=samples(name)[0];self.assertEqual(routing(s),routing(deepcopy(s)))
    def test_numeric_order(self):
        s=samples()[0];rows=[dict(s,id=str(i),turn_id=i) for i in ('10','2','1')]
        self.assertEqual([r['turn_id'] for r in sorted(rows,key=turn_order)],['1','2','10'])
    def test_packing_and_breaks_exact(self):
        s=samples()[0];ps=[{'sample':dict(s,group_id=g,id=f'{g}{i}',turn_id=i)} for g,n in [('a',2),('b',3),('c',6)] for i in range(n)]
        chunks=pack_groups(ps,5)
        self.assertEqual([len(x) for x in chunks],[5,5,1]);self.assertEqual(sum(map(len,chunks)),11)
        self.assertEqual(continuity(chunks),[dict(group_id='c',block_ordinals=[1,2],reason='GROUP_EXCEEDS_BLOCK_LIMIT',cache_survives=False)])
    def test_stability_across_scales_blocks_repetitions_and_tracks(self):
        with tempfile.TemporaryDirectory() as d:
            path=dataset(d,'snips',10);maps=[]
            for scale,size,count in [('SMOKE',3,6),('SMALL',5,10)]:
                m=prepare_experiment({'snips':path},TinyTokenizer(),{'fixture':True},2048,dict(bos_token_id=0,eos_token_id=6,pad_token_id=0),Path(d)/scale,
                    scale=scale,block_size=size,repetitions=2,overrides={'snips':count},budgets={'snips':2},fixture=True,tracks=('NATURAL','CONTROLLED'),prefixes=(64,))
                mapping={}
                for b in m['blocks']:
                    self.assertEqual(b['scale_semantics'],SCALE_SEMANTICS)
                    for t in b['tasks']:
                        route=(t['user_id'],t['adapter_id']);self.assertEqual(mapping.setdefault(t['sample_id'],route),route)
                maps.append(mapping)
                report=inspect_plan(Path(d)/scale)
                self.assertEqual(report['maximum_block_index'],len(m['blocks'])-1)
                self.assertEqual(report['slurm_array'],f"0-{len(m['blocks'])-1}%1")
                self.assertEqual(m['scale_summary']['unique_scored_samples'],count)
            for sid in maps[0]:self.assertEqual(maps[0][sid],maps[1][sid])
    def test_group_breaks_are_in_sealed_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            path=dataset(d,'coqa',2)
            m=prepare_experiment({'coqa':path},TinyTokenizer(),{'fixture':True},2048,dict(bos_token_id=0,eos_token_id=6,pad_token_id=0),Path(d)/'plan',
                block_size=1,repetitions=2,overrides={'coqa':4},budgets={'coqa':2},fixture=True)
            self.assertEqual(m['scale_summary']['unique_scored_samples'],4)
            self.assertEqual(len(m['blocks']),8)
            for b in m['blocks']:
                self.assertTrue(b['continuity_breaks'])
                self.assertFalse(b['continuity_breaks'][0]['cache_survives'])
                self.assertEqual(b['cache_lifetime_requests'],1)
    def test_missing_real_datasets_no_fixture_fallback(self):
        with patch('semcache.edgelora.r5_i5_preflight.installed_opt_check',side_effect=ImportError('absent')):
            r=readiness({})
        self.assertEqual(r['status'],'NOT_READY');self.assertEqual(len(r['datasets']),6)
        self.assertTrue(all(x['status']=='DATASET_NOT_READY' for x in r['datasets'].values()))
        self.assertEqual(r['runtime']['installed_opt']['status'],'NOT_RUN')
        self.assertTrue(r['coqa_smoke_blocker'])
    def test_fixture_snapshot_rejected_as_real(self):
        with tempfile.TemporaryDirectory() as d:
            path=dataset(d)
            with patch('semcache.edgelora.r5_i5_preflight.installed_opt_check',side_effect=ImportError('absent')):r=readiness({'coqa':str(path)})
            self.assertEqual(r['datasets']['coqa']['status'],'DATASET_NOT_READY')
    def test_known_scorers(self):
        r=scorer_smokes()
        for name in ('coqa','snips','multiwoz','qasper','2wikimqa'):self.assertEqual(r[name]['status'],'PASS')
        self.assertIn(r['multi_news']['status'],('PASS','NOT_RUN'))
