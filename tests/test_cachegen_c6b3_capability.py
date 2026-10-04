import unittest
from unittest.mock import patch
from semcache.experiments.cachegen import c6b3_capability as c

class Tests(unittest.TestCase):
    def test_corpus_only(self):
        cases=[dict(user=u,history_depth=k,generated_text='answer',reference_text='ref',generated_length=3,
                    normalized_edit_distance=.2,position_agreement=.5) for u in c.d.USERS for k in range(1,5)]
        calls=[]
        def bleu(g,r,**kwargs):
            calls.append((g,r,kwargs));return {'value':0}
        with patch.object(c,'compute_bleu',side_effect=bleu):s=c.aggregate(cases)
        self.assertEqual(len(calls[0][0]),8);self.assertEqual(len(calls),7)
        self.assertTrue(all(x[2]==c.d.b.BLEU for x in calls));self.assertEqual(s['nonempty_generation_count'],8)
        self.assertFalse(s['semantic_reuse_enabled']);self.assertFalse(s['compression_enabled'])

if __name__=='__main__':unittest.main()
