import copy
import tempfile
import unittest
from pathlib import Path
from semcache.experiments.cachegen import c6b3_multiwoz as b


def rows():
    return [dict(dataset='multiwoz',source_id=f'c{i//2}:{2*(i%2)}',conversation_id=f'c{i//2}',
                 global_query_index=i,original_order_index=i,source_split='train',
                 query_text=f'question {i}',reference_text=f'answer {i}',domain_or_intent='hotel',
                 metadata={'turn_id':2*(i%2)}) for i in range(160)]

class Tokenizer:
    eos_token_id=2
    def __call__(self,text,**kwargs):
        return {'input_ids':[10]*len(text.split())+([2] if kwargs['add_special_tokens'] else [])}

class Tests(unittest.TestCase):
    def test_history(self):
        r=rows(); out=b.reconstruct(r,2)
        self.assertEqual(out,b.reconstruct(r,2))
        self.assertEqual(out[1]['prompt_text'],'Dialogue:\nUser: question 0\nAssistant: answer 0\nUser: question 1\nAssistant:')
        self.assertNotIn('answer 1',out[1]['prompt_text'])
        self.assertNotIn('question 2',out[1]['prompt_text'])
        self.assertNotIn('answer 1',out[2]['prompt_text'])
        self.assertEqual(out[2]['history_source_ids'],[])
        self.assertEqual(r,rows())

    def test_guards(self):
        for change in (lambda r:r[0].update(reference_text=None),
                       lambda r:r[1]['metadata'].update(turn_id=4),
                       lambda r:r[1].update(source_id=r[0]['source_id'])):
            r=rows();change(r)
            with self.assertRaises(ValueError):b.reconstruct(r,1)

    def test_analysis(self):
        s=b.analyze(rows(),Tokenizer(),256)
        self.assertEqual(set(s),set('01234'))
        self.assertGreater(s['1']['total_tokens']['max'],s['0']['total_tokens']['max'])
        self.assertEqual(s['4']['prompt_tokens']['fraction_within_bound'],1)
        self.assertEqual(b.m9.MODEL_REVISION,'905a4b602cda5c501f1b3a2650a4152680238254')

    def test_frozen_plan(self):
        r=b.reconstruct(rows(),1)
        for i,row in enumerate(r):
            row.update(token_ids=[2,90,91,10,11,12,100+i],cluster_id=0,
                semantic_execution_provenance='MEASURED',model_revision=b.m9.MODEL_REVISION,
                semantic_assignment_source=b.m9.assignment_source('multiwoz'),
                tokenizer_id=f'{b.m9.MODEL_ID}@{b.m9.MODEL_REVISION}')
        counts=b.lengths(r,Tokenizer())
        selection,train=b.plan(r,counts)
        self.assertEqual((selection,train),b.plan(r,counts))
        altered=copy.deepcopy(r)
        for row in altered: row['generated_quality']=999
        self.assertEqual((selection,train),b.plan(altered,counts))
        self.assertEqual(len(selection['episodes']),32)
        self.assertEqual(selection['target_conversation_count'],32)
        hold=set(train['holdout_conversation_ids'])
        self.assertFalse(hold & set(train['training_conversation_ids']))
        a,c=train['users'].values()
        self.assertFalse(set(a['conversation_ids']) & set(c['conversation_ids']))
        for e in selection['episodes']:
            self.assertNotEqual(e['source_id'],e['target_id'])
            self.assertLess(e['source_index'],e['target_index'])
            for side in ('source','target'):
                self.assertIn(e[side+'_conversation_id'],hold)
                start=e[side+'_start']; self.assertGreaterEqual(start,3)
                self.assertEqual(r[e[side+'_index']]['token_ids'][start:start+3],e['token_ids'])

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'existing.json'; b.write(p,{'original':True})
            with self.assertRaises(FileExistsError): b.write(p,{})
            self.assertIn('original',p.read_text())

if __name__=='__main__':unittest.main()
