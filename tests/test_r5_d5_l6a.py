"""CPU structural fixtures. All synthetic measurements are FIXTURE/NOT_MEASURED."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import csv
import unittest
from unittest.mock import patch

from semcache.cache.cache_entry import CacheEntry
from semcache.edgelora import r5_d5_l6a as l6
from semcache.edgelora import r5_d5_l6a_analysis as analysis
from semcache.edgelora.r5_d5_lightperf_replay import replay_stream
from semcache.edgelora.r5_d5_lightperf_io import SummaryWriter, verify, read_json
from semcache.edgelora.r5_subsequence_physical import PhysicalOwner
from test_r5_d5_sweep import workload, policy, digest, run_fixtures, frozen_fixture


def stream(n):
    w,tasks=workload('2wikimqa',10)
    for task in tasks:
        task['prepared']['max_new_tokens']=8
        task['policy']=policy(w['special_tokens'],'quality',8)
    for block in w['blocks']:
        block['block_id']=digest({k:v for k,v in block.items() if k!='block_id'})
    return replay_stream(w,'2wikimqa',n,enable_continuous_replay=True)


class LongReplay(unittest.TestCase):
    def test_prefix_payload_and_determinism(self):
        long,short=stream(50),stream(20)
        l6.check_stream(long,short)
        self.assertEqual(long,stream(50))
        for field in ('request_id','adapter_id','policy','token_ids'):
            bad=deepcopy(long)
            bad['tasks'][34][field]=None
            bad['tasks_sha256']=digest(bad['tasks'])
            with self.assertRaises(ValueError):
                l6.check_stream(bad,short)

    def test_build_reproducible_and_bound_to_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,cpu,frozen=frozen_fixture(Path(tmp))
            frozen.update(thresholds=l6.THRESHOLDS,transport=dict(rows=list(range(20))))
            frozen['source']['commit']=l6.BASE
            reference=dict(plan=str(plan),cpu_audit=str(cpu))
            fields={k:frozen[k] for k in ('original_inventory','multi_inventory','seed','semantic_identity')}
            fields.update(stream=stream(50),transport=dict(rows=list(range(50))),
                          resources=dict(reasons=[],impact_layer_working_bytes=0,gpu_estimate_bytes=0,
                                         gpu_allocation_ceiling_bytes=22*2**30))
            with patch.object(l6,'historical_reference',return_value=(frozen,None,reference)), \
                 patch.object(l6,'material',return_value=fields):
                a=l6.build(Path(tmp),{'scope':'FIXTURE'},64)
                self.assertEqual(a,l6.build(Path(tmp),{'scope':'FIXTURE'},64))
                self.assertNotEqual(a['plan_id'],l6.build(Path(tmp),{'scope':'FIXTURE'},128)['plan_id'])
                self.assertEqual(a['cycles'][-1],dict(cycle=5,first=40,last=49))
                fields['transport']['rows'].reverse()
                with self.assertRaisesRegex(ValueError,'transport'):
                    l6.build(Path(tmp),{},64)

    def runs(self):
        s=stream(50)
        runs=run_fixtures()
        for ci,r in enumerate(runs):
            original=r['requests']
            rows=[]
            for i,(task,occ) in enumerate(zip(s['tasks'],s['occurrences'])):
                row=dict(original[i%20],request_id=task['request_id'],replay_provenance=occ,
                         adapter_id=task['adapter_id'],cache_entry_count=i,cache_occupancy_bytes=i*7,
                         application_e2e_ms=100+i-ci,session_ordinal=i)
                rows.append(row)
            r.update(requests=rows,scale=50,tasks_sha256=s['tasks_sha256'],replay_stream_sha256=digest(s))
        return runs,s

    def test_cycles_paired_and_persistent_adapter_occupancy(self):
        runs,s=self.runs()
        report=analysis.summarize(runs,s,fixture=True)
        self.assertEqual(report['scope'],'FIXTURE/NOT_MEASURED')
        self.assertEqual([r['requests'] for r in report['cycle_metrics']],[10]*15)
        self.assertEqual([r['requests'] for r in report['cumulative_metrics']],[10,20,30,40,50]*3)
        last=report['cumulative_metrics'][-1]
        self.assertEqual(last['mean_delta_e2e_ms'],-2)
        self.assertEqual(last['policy_counts']['INSERT'],30)
        rows=runs[-1]['requests']
        states={}
        totals=[]
        for row in rows:
            states[row['adapter_id']]=row['cache_entry_count']
            totals.append(sum(states.values()))
        self.assertEqual(last['cache']['total']['entry_count'],dict(mean=sum(totals)/50,maximum=max(totals),final=totals[-1]))
        self.assertEqual(report['cycle_metrics'][-1]['cache']['total']['entry_count']['mean'],sum(totals[-10:])/10)

    def test_fresh_cycle_csv_json_inventory(self):
        runs,selected=self.runs()
        report=analysis.summarize(runs,selected,fixture=True)
        with tempfile.TemporaryDirectory() as tmp, patch.object(l6,'NS',Path(tmp)):
            path=Path(tmp)/'fixture'
            analysis.write_report(report,path,[])
            verify(path)
            self.assertEqual(read_json(path/'long_replay_analysis.json'),report)
            for group in ('cycle_metrics','cumulative_metrics'):
                with (path/(group+'.csv')).open() as f:
                    rows=list(csv.DictReader(f))
                self.assertEqual([int(row['requests']) for row in rows],
                                 [row['requests'] for row in report[group]])
                self.assertTrue(all(r['scope']=='FIXTURE/NOT_MEASURED' for r in rows))
            with self.assertRaisesRegex(ValueError,'fresh output'):
                analysis.write_report(report,path,[])

    def test_rejects_mismatched_comparisons(self):
        runs,s=self.runs()
        for change in ('count','condition','hash','gpu','exact_tokens','provenance'):
            bad=deepcopy(runs)
            if change=='count': bad[1]['requests'].pop()
            if change=='condition': bad[2]['condition']=bad[1]['condition']
            if change=='hash': bad[1]['tasks_sha256']='different'
            if change=='gpu': bad[1]['gpu']['uuid']='different'
            if change=='exact_tokens': bad[1]['requests'][22]['generated_token_ids']=[-1]
            if change=='provenance': bad[1]['requests'][21]['replay_provenance']={}
            with self.subTest(change=change),self.assertRaises(ValueError):
                analysis.summarize(bad,s,fixture=True)

    def test_analyzer_requires_intact_light_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p=dict(plan_id='FIXTURE',entry_limit=64,**{k:'FIXTURE' for k in
                'source stream resources dataset count thresholds validation_mode conditions seed performance_knobs impact_implementation transport base_prepared multi_inventory original_inventory semantic_identity tokenizer_identity dataset_inventories workload_mode enable_continuous_replay warmup_requests'.split()})
            light=root/'light';light.mkdir()
            w=SummaryWriter(light);w.write('plan.json',dict(p,l6_plan_id=p['plan_id']));w.seal()
            args=SimpleNamespace(light_plan=light,light_audit=root/'audit',full=root/'full',exact=root/'exact',approx=root/'approx',output=root/'out')
            with self.assertRaises(ValueError):analysis.analyze(p,args)
            args.full.mkdir()
            w=SummaryWriter(args.full);w.write('run.json',{'scope':'FIXTURE'});w.seal()
            (args.full/'run.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed light receipt'):analysis.analyze(p,args)
            self.assertFalse(args.output.exists())


class EntryLimits(unittest.TestCase):
    def owner(self,**kwargs):
        return PhysicalOwner(dict(device='cpu'),{},**kwargs)

    def fill(self,owner,n):
        evicted=[]
        for i in range(n):
            owner.cache.insert(CacheEntry(0,(i,),(0,1),8),on_event=lambda e,v,_:evicted.append(v.key) if e=='EVICT' else None)
            self.assertEqual(len(owner.cache.entries),min(i+1,n,owner.cache.entry_limit,owner.cache.capacity_bytes//8))
        return evicted

    def test_default_and_128_independent_deterministic_eviction(self):
        default,explicit,big=self.owner(),self.owner(entry_limit=64),self.owner(entry_limit=128)
        old=self.fill(default,129)
        self.assertEqual(self.fill(explicit,129),old)
        self.assertEqual(list(default.cache.entries),list(explicit.cache.entries))
        self.assertEqual(len(old),65)
        self.assertEqual(self.fill(big,129),[(0,(0,))])
        self.assertEqual(len(default.cache.entries),64)

    def test_two_adapters_each_keep_128_entries(self):
        owners=[PhysicalOwner(dict(device='cpu',adapter_id=aid),{},entry_limit=128)
                for aid in ('Adapter_A','Adapter_B')]
        for owner in owners:
            self.assertEqual(self.fill(owner,128),[])
        self.assertEqual(sum(len(o.cache.entries) for o in owners),256)
        owners[0].cache.insert(CacheEntry(0,(128,),(0,1),8))
        self.assertIn((0,(0,)),owners[1].cache.entries)
        self.assertNotIn((0,(0,)),owners[0].cache.entries)

    def test_byte_cap_and_invalid_limits(self):
        self.assertEqual(len(self.fill(self.owner(entry_limit=128,capacity=16),4)),2)
        for invalid in (0,63,65,129,True,64.,'128'):
            with self.assertRaises(ValueError): self.owner(entry_limit=invalid)
        with self.assertRaises(ValueError): self.owner(entry_limit=128,capacity=512*2**20+1)

    def test_gpu_128_blocked_without_cuda(self):
        with self.assertRaisesRegex(ValueError,'ENTRY128_GPU_NOT_APPROVED'):
            PhysicalOwner(dict(device='cuda:0'),{},entry_limit=128)
        owner=self.owner(entry_limit=128)
        owner.active={'seen':set()}
        with self.assertRaisesRegex(ValueError,'ENTRY128_GPU_NOT_APPROVED'):
            owner.project(0,'q',SimpleNamespace(device=SimpleNamespace(type='cuda')),None,None)
