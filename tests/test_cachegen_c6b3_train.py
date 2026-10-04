import unittest
from semcache.experiments.cachegen import c6b3_train as d


def fixture():
    rows=[];training=dict(users={u:dict(row_ids=[]) for u in d.USERS},holdout_conversation_ids=['eval'])
    for user in d.USERS:
        for c in range(45):
            for t in range(5):
                sid=f'{user}:{c}:{t}'
                rows.append(dict(source_id=sid,conversation_id=f'{user}:{c}',turn_id=str(2*t),
                    prompt_sha256='p',reference_sha256='r',prompt_text='prompt',reference_text='response',token_ids=[2,4]))
                training['users'][user]['row_ids'].append(sid)
    return rows,training

class Tests(unittest.TestCase):
    def test_cohort_and_profiles(self):
        rows,training=fixture();cap=d.capability_cohort(rows,training)
        self.assertEqual(cap,d.capability_cohort(rows,training));self.assertEqual(len(cap['conversation_ids']),64)
        self.assertEqual(len(set(cap['conversation_ids'])),64)
        for u in d.USERS:
            for k in range(1,5):self.assertEqual(sum(e['user']==u and e['history_depth']==k for e in cap['examples']),8)
        for profile in ('pilot','full'):
            plan=d.training_subset(rows,training,cap,profile)
            for u in d.USERS:
                self.assertEqual(plan['users'][u]['row_count'],65)
                self.assertEqual(plan['users'][u]['conversation_count'],13)
                self.assertFalse(set(plan['users'][u]['conversation_ids'])&set(cap['conversation_ids']))
                ids=d.epoch_ids(plan,u,0)
                self.assertEqual(set(ids),set(plan['users'][u]['row_ids']))
                self.assertEqual(ids,d.epoch_ids(plan,u,0));self.assertNotEqual(ids,d.epoch_ids(plan,u,1))
    def test_pilot_cap_preserves_whole_conversations(self):
        rows=[];training={'users':{},'holdout_conversation_ids':['eval']}
        for user in d.USERS:
            ids=[]
            for c in range(60):
                for t in range(101):
                    sid=f'{user}:{c}:{t}';ids.append(sid)
                    rows.append(dict(source_id=sid,conversation_id=f'{user}:{c}'))
            training['users'][user]={'row_ids':ids}
        cap={'conversation_ids':[]}
        pilot=d.training_subset(rows,training,cap,'pilot')
        full=d.training_subset(rows,training,cap,'full')
        for user in d.USERS:
            self.assertEqual(pilot['users'][user]['row_count'],4949)
            self.assertEqual(pilot['users'][user]['conversation_count'],49)
            self.assertEqual(full['users'][user]['row_count'],6060)
            self.assertEqual(len(set(pilot['users'][user]['row_ids'])),4949)

    def test_encode(self):
        class Tokenizer:
            eos_token_id=2
            def __call__(self,text,**kwargs):
                if kwargs['add_special_tokens']:return {'input_ids':[2,4]}
                self.text=text;return {'input_ids':[8,9]}
        tok=Tokenizer();row=fixture()[0][0];e=d.encode_example(tok,row)
        self.assertEqual(tok.text,' response');self.assertEqual(e['labels'],[-100,-100,8,9,2])
        row['token_ids']=[8]
        with self.assertRaises(ValueError):d.encode_example(tok,row)
    def test_infeasible(self):
        rows,training=fixture();training['users']['user_a']['row_ids']=training['users']['user_a']['row_ids'][:20]
        with self.assertRaises(ValueError):d.capability_cohort(rows,training)

if __name__=='__main__':unittest.main()
