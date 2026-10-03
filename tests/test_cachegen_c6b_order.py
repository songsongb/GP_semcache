"""Pure-CPU regression tests for the frozen-cohort epoch permutation."""
from collections import Counter
import copy
from itertools import groupby
import json
import random
import unittest

from semcache.experiments.cachegen import c6b_snips as b1
from semcache.experiments.cachegen.c6b_order import POLICY, epoch_order, order_manifest


def pilot():
    rows=[dict(source_id=f'{user}:{label}:{i}',user=user,intent=label)
          for user in b1.USERS for label in b1.LABELS for i in range(200)]
    return dict(train_rows=rows,train_row_ids=[r['source_id'] for r in rows],
                holdout_ids=['frozen_source','frozen_target'],
                counts={u:{label:200 for label in b1.LABELS} for u in b1.USERS})


class EpochOrderTests(unittest.TestCase):
    def test_exact_balanced_cohort_and_no_holdout_each_epoch(self):
        plan=pilot(); before=copy.deepcopy(plan)
        lookup={r['source_id']:r['intent'] for r in plan['train_rows']}
        for user in b1.USERS:
            expected=[r['source_id'] for r in plan['train_rows'] if r['user']==user]
            for epoch in range(1,4):
                ordered,audit=epoch_order(plan,user,epoch,42)
                self.assertEqual(Counter(ordered),Counter(expected))
                self.assertEqual(len(set(ordered)),1400)
                self.assertFalse(set(ordered)&set(plan['holdout_ids']))
                labels=[lookup[sid] for sid in ordered]
                self.assertEqual(Counter(labels),Counter({l:200 for l in b1.LABELS}))
                for start in range(0,1400,7):
                    self.assertEqual(labels[start:start+7],list(b1.LABELS))
                self.assertEqual(audit['ordered_row_ids_sha256'],b1.digest(ordered))
        self.assertEqual(plan,before)

    def test_determinism_epoch_shuffles_and_rng_isolation(self):
        plan=pilot(); state=random.getstate()
        lookup={r['source_id']:r['intent'] for r in plan['train_rows']}
        seeds=[]
        for user in b1.USERS:
            first,audit=epoch_order(plan,user,1,42)
            second,audit2=epoch_order(plan,user,2,42)
            self.assertEqual((first,audit),epoch_order(copy.deepcopy(plan),user,1,42))
            self.assertNotEqual(first,epoch_order(plan,user,1,43)[0])
            seeds.extend([audit['derived_rng_seed'],audit2['derived_rng_seed']])
            for label in b1.LABELS:
                self.assertNotEqual([s for s in first if lookup[s]==label],
                                    [s for s in second if lookup[s]==label])
        self.assertEqual(len(set(seeds)),4)
        self.assertEqual(random.getstate(),state)

    def test_regression_no_terminal_200_example_search_screening_block(self):
        plan=pilot(); lookup={r['source_id']:r['intent'] for r in plan['train_rows']}
        old=[r['intent'] for r in plan['train_rows'] if r['user']=='user_a']
        self.assertEqual(old[-200:],['SearchScreeningEvent']*200)
        for user in b1.USERS:
            for epoch in range(1,4):
                ordered,_=epoch_order(plan,user,epoch,42)
                labels=[lookup[s] for s in ordered]
                self.assertNotEqual(labels[-200:],['SearchScreeningEvent']*200)
                self.assertEqual(max(len(list(g)) for _,g in groupby(labels)),1)

    def test_reject_duplicate_rows_and_holdout_leakage(self):
        for sid in ('frozen_source','frozen_target'):
            plan=pilot(); plan['train_rows'][0]['source_id']=sid
            with self.assertRaisesRegex(ValueError,'holdout leakage'):
                epoch_order(plan,'user_a',1,42)
        plan=pilot(); plan['train_rows'].append(plan['train_rows'][0].copy())
        with self.assertRaisesRegex(ValueError,'Duplicated'):
            epoch_order(plan,'user_a',1,42)

    def test_manifest_has_reproducible_hash_for_each_user_epoch(self):
        plan=pilot(); manifest=order_manifest(plan,42,3)
        self.assertEqual(manifest['training_order_policy'],
            'deterministic per-epoch within-intent shuffle + class-balanced interleave')
        self.assertEqual(manifest['training_order_policy'],POLICY)
        self.assertEqual(json.loads(json.dumps(manifest)),manifest)
        for user in b1.USERS:
            audits=manifest['training_order_epochs'][user]
            self.assertEqual(len(audits),3)
            for epoch,audit in enumerate(audits,1):
                self.assertEqual(audit,epoch_order(plan,user,epoch,42)[1])

    def test_old_frozen_selection_golden_digest_unchanged(self):
        from test_cachegen_c6b_snips import workload,selection
        rows=workload(); frozen=selection(rows); frozen_before=copy.deepcopy(frozen)
        plan=b1.build_plan(rows,frozen,200,42)
        # Recorded from the pre-fix implementation at e244e0dd.
        golden='2277009eea06735aa64ee5dfc456af67590b95c274c4aeea4b4cad423f9fcf8b'
        self.assertEqual(b1.digest(plan),golden)
        order_manifest(plan,42,3)
        self.assertEqual(b1.digest(plan),golden)
        self.assertEqual(frozen,frozen_before)


if __name__=='__main__': unittest.main()
