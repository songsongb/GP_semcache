import unittest
from semcache.experiments.cachegen import c6b3_selection as v


def fixture(content=True):
    rows=[]
    for i in range(200):
        depth=i%5
        rows.append(dict(source_id=str(i),conversation_id=str(i//5),dataset='multiwoz',source_split='train',
            token_ids=[2,9,9,50118,44518,35,70,71,72,1000+i],
            current_user_token_indices=[6,7,8,9] if content else [9],history_depth=depth,
            semantic_execution_provenance='MEASURED',model_revision=v.b.m9.MODEL_REVISION,
            tokenizer_id=f'{v.b.m9.MODEL_ID}@{v.b.m9.MODEL_REVISION}',
            semantic_assignment_source=v.b.m9.assignment_source('multiwoz'),cluster_id=0,
            reference_text='response',reference_sha256='hash',turn_id=str(depth*2)))
    return rows

class Tests(unittest.TestCase):
    def test_prefix_never_qualifies(self):
        with self.assertRaisesRegex(ValueError,'candidate conversation counts'):v.select(fixture(False))

    def test_strata_disjoint_deterministic(self):
        rows=fixture();episodes,counts=v.select(rows)
        self.assertEqual((episodes,counts),v.select(rows))
        self.assertEqual(len(episodes),32)
        self.assertEqual(len({rows[e['target_index']]['conversation_id'] for e in episodes}),32)
        for depth in range(1,5):self.assertEqual(sum(e['history_depth']==depth for e in episodes),8)
        for e in episodes:
            self.assertEqual(e['token_ids'],[70,71,72])
            self.assertGreaterEqual(e['source_start'],6)
            self.assertGreaterEqual(e['target_start'],6)
            self.assertLess(e['source_index'],e['target_index'])
            self.assertNotEqual(e['target_turn_id'],'0')
        for r in rows:r['bleu']=999;r['loss']=-1
        self.assertEqual((episodes,counts),v.select(rows))

    def test_offset_boundaries(self):
        text='Dialogue:\nUser: natural words here\nAssistant:'
        start=text.index('natural');end=start+len('natural words here')
        class Tokenizer:
            is_fast=True
            def __call__(self,*args,**kwargs):
                return dict(input_ids=[2,10,11,12,13,14],offset_mapping=[(0,0),(0,9),(start-1,start+7),(start+8,start+13),(start+14,end),(end,end+2)])
        row=dict(prompt_text=text,query_text=text,current_user_text='natural words here',prompt_version=v.b.VERSION,
                 token_ids=[2,10,11,12,13,14],history_source_ids=[],turn_id='0',history_policy={'last_complete_exchanges':4})
        out=v.annotate([row],Tokenizer())[0]
        self.assertEqual(out['current_user_token_indices'],[3,4]) # boundary-crossing token excluded
        row['token_ids']=[99]
        with self.assertRaises(ValueError):v.annotate([row],Tokenizer())

if __name__=='__main__':unittest.main()
