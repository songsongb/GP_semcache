"""Stdlib-only diagnostic tests; synthetic evidence is never GPU evidence."""
from copy import deepcopy
import hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from semcache.edgelora.r5_i5r0_io import Files,canonical,sha,load_blocks,UNKNOWN,CONDITIONS,provenance
from semcache.edgelora.r5_i5r0 import candidate_diagnostic,aggregate_requests,latency,run


def pair(source=(1,2,3),target=(1,2,3,4)):
    phys=dict(adapter_id='Adapter_A')
    s=dict(token_ids=list(source),positions=list(range(len(source))),attention_mask=[1]*len(source),physical_identity=phys,
           source_user='User_01',source_request_id='S',completed=True)
    t=dict(token_ids=list(target),positions=list(range(len(target))),attention_mask=[1]*len(target),physical_identity=phys,adapter_id='Adapter_A',user_id='User_01')
    return s,t
REG=dict(users={'User_01':dict(adapter_id='Adapter_A',share_group='A',allow_cross_user_reuse=True)})


def seal(root):
    files={str(p.relative_to(root)):sha(p.read_bytes()) for p in root.rglob('*') if p.is_file() and p.name!='i5_output_manifest.json'}
    (root/'i5_output_manifest.json').write_text(json.dumps(dict(files=files)))


def fixture(parent,index=0):
    root=Path(parent)/f'block_{index}_123_audit';root.mkdir()
    s,t=pair(target=(1,2,3));tasks=[]
    for i,ids in enumerate(([1,2,3],[1,2,3])):
        tasks.append(dict(request_id='S' if i==0 else 'T',token_ids=ids,adapter_id='Adapter_A',user_id='User_01',scored=True,sample_id=str(i),group_id='g'))
    block=dict(dataset='qasper',track='NATURAL',repetition=0,tasks=tasks,cache_lifetime_requests=2)
    block['block_id']=sha(canonical(block));config=dict(block=block,experiment_id='experiment',source_commit=None,sources={},fixture=True)
    rows=[];events=[]
    for c in CONDITIONS:
        for i,rid in enumerate(('S','T')):
            reasons=[] if i==0 or c==CONDITIONS[0] else [dict(entry_id='key',source_request_id='S',prefix_length=3,rejection='TARGET_NOT_LONGER_THAN_SOURCE')]
            rows.append(dict(condition=c,request_id=rid,status='COMPLETE',dataset='qasper',adapter_id='Adapter_A',user_id='User_01',scored=True,
                input_tokens=3,candidate_count=len(reasons),candidate_reasons=reasons,matched_identity_sha256=rid,request_sha256=rid,physical_identity={'adapter_id':'Adapter_A'},
                arrival_ns=i*10,completion_ns=i*10+5,source=None,reused_prefix_tokens=0,application_e2e_ms=10+(c==CONDITIONS[2])*2,
                admission=None,eviction_count=0,cache_before={'lru_to_mru':['key'] if i else []},truncated=True))
        if c!=CONDITIONS[0]:
            m=dict(s,entry_id='key',cluster_id=0,request_sha256='S')
            for seq,e in enumerate([dict(kind='DISCOVERY',request_id='S',candidates=[],candidate_reasons=[],assignment={'cluster_id':0}),
                dict(kind='INDEX_INSERT',metadata=m),dict(kind='DISCOVERY',request_id='T',candidates=[dict(metadata=m,matching_windows=[{}])],
                candidate_reasons=[dict(entry_id='key',source_request_id='S',prefix_length=3,rejection='TARGET_NOT_LONGER_THAN_SOURCE')],assignment={'cluster_id':0})]):
                events.append(dict(condition=c,adapter_id='Adapter_A',sequence=seq,event=e))
    for name,value in [('i5_execution_config.json',config),('i5_independent_audit.json',{'status':'PASS'}),('i5_cache_summary.json',{}),('i5_latency_summary.json',{}),('i5_network_summary.json',{})]:
        (root/name).write_text(json.dumps(value))
    for name,value in [('i5_request_metrics.jsonl',rows),('i5_cache_events.jsonl',events)]:
        (root/name).write_text(''.join(json.dumps(v)+'\n' for v in value))
    seal(root);return root


def archive(parent,out):
    with zipfile.ZipFile(out,'w') as z:
        for p in sorted(Path(parent).rglob('*')):
            if p.is_file() and p.resolve()!=Path(out).resolve():z.write(p,p.relative_to(parent).as_posix())


class CandidateChecks(unittest.TestCase):
    def diagnose(self,s,t,**kw):return candidate_diagnostic(s,t,resident=True,registry=REG,**kw)
    def test_short_prefix(self):
        d=self.diagnose(*pair());self.assertTrue(d['could_pass_unchanged_safety_policy']);self.assertEqual(d['longest_common_prefix_tokens'],3)
    def test_equal_exact_sequence_still_length_failure(self):
        d=self.diagnose(*pair(target=(1,2,3)));self.assertTrue(d['exact_full_sequence_equal']);self.assertTrue(d['checks']['exact_full_source_prefix'])
        self.assertFalse(d['could_pass_unchanged_safety_policy']);self.assertIn('TARGET_NOT_LONGER_THAN_SOURCE',d['independent_diagnostic_reasons'])
    def test_long_source(self):
        d=self.diagnose(*pair(source=(1,2,3,4),target=(1,2,3)));self.assertTrue(d['source_length_greater_target']);self.assertFalse(d['checks']['exact_full_source_prefix'])
    def test_shared_subsequence_is_not_prefix(self):
        d=self.diagnose(*pair(target=(9,1,2,3)));self.assertFalse(d['checks']['exact_full_source_prefix']);self.assertEqual(d['longest_common_prefix_tokens'],0)
    def test_adapter_mismatch(self):
        s,t=pair();t['physical_identity']={'adapter_id':'Adapter_B'};t['adapter_id']='Adapter_B'
        self.assertFalse(self.diagnose(s,t)['checks']['adapter_compatible'])
    def test_position_mask_checks_independent(self):
        s,t=pair();t['positions'][0]=8;t['attention_mask'][1]=0;d=self.diagnose(s,t)
        self.assertFalse(d['checks']['positions_compatible']);self.assertFalse(d['checks']['mask_compatible'])
    def test_evicted_entry(self):
        d=candidate_diagnostic(*pair(),resident=False,registry=REG);self.assertFalse(d['could_pass_unchanged_safety_policy'])
    def test_missing_tokens_not_invented(self):
        s,t=pair();del s['token_ids'];d=self.diagnose(s,t,source_length=3)
        self.assertEqual(d['longest_common_prefix_tokens'],UNKNOWN);self.assertEqual(d['checks']['exact_full_source_prefix'],UNKNOWN)
    def test_length_false_dominates_other_unknowns(self):
        d=candidate_diagnostic(*pair(target=(1,2,3)),resident=True);self.assertEqual(d['checks']['sharing_authorized'],UNKNOWN);self.assertIs(d['could_pass_unchanged_safety_policy'],False)


class InputAndAggregation(unittest.TestCase):
    def test_safe_zip_and_partial_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inputs=root/'in';inputs.mkdir();fixture(inputs);z=root/'audit.zip';archive(inputs,z)
            r=run(audit_zip=z,output_dir=root/'out',allow_partial=True,expected_blocks=1)
            m=r['workload']['integrated'];self.assertEqual(m['scored_requests'],2);self.assertEqual(m['requests_with_semantic_candidates'],1);self.assertEqual(m['candidate_occurrences'],1)
            self.assertTrue((root/'out/i5r0_report.md').exists());self.assertTrue(r['coverage']['partial_observability'])
            r2=run(audit_zip=z,output_dir=root/'out2',allow_partial=True,expected_blocks=1)
            self.assertEqual(r['workload'],r2['workload'])
    def test_missing_optional_requires_partial(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);inp=p/'in';inp.mkdir();fixture(inp);z=p/'a.zip';archive(inp,z)
            with self.assertRaisesRegex(ValueError,'partial'):run(audit_zip=z,output_dir=p/'out',expected_blocks=1)
    def test_multiple_candidates_are_not_multiple_requests(self):
        rows=[dict(block_id='b',condition=CONDITIONS[2],request_id='T',input_tokens=5,reused_prefix_tokens=0,has_candidates=True,candidate_count=2,has_safety_eligible_candidate=False,eligibility_unknown=False,actual_hit=False,admissions=0,evictions=0,cross_adapter_rejections=0,no_candidate_requests=False,no_candidates=False)]
        s,t=pair();d=candidate_diagnostic(s,t,resident=False,registry=REG)
        cs=[dict(d,block_id='b',condition=CONDITIONS[2],request_id='T',entry_id=str(i)) for i in range(2)]
        m=aggregate_requests(rows,cs);self.assertEqual(m['requests_with_semantic_candidates'],1);self.assertEqual(m['candidate_occurrences'],2)
    def test_known_historical_blob_mismatch_is_not_waived(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);inp=p/'in';inp.mkdir();fixture(inp);z=p/'a.zip';archive(inp,z)
            with patch('semcache.edgelora.r5_i5r0.provenance',return_value={'blocks':[{'verification':'RECORDED_SOURCE_MISMATCH'}]}):
                with self.assertRaisesRegex(ValueError,'historical Git blobs'):run(audit_zip=z,output_dir=p/'out',allow_partial=True,expected_blocks=1)
    def test_paired_latency(self):
        rs=[dict(block_id='b',condition=c,request_id='R',scored=True,dataset='d',repetition=0,original=dict(application_e2e_ms=v)) for c,v in zip(CONDITIONS,[10,11,12])]
        p=latency(rs)['paired'][CONDITIONS[0]+'_vs_'+CONDITIONS[2]]['summary']['application_e2e_ms']
        self.assertEqual(p['absolute_delta']['mean'],2);self.assertEqual(p['aggregate_percentage_delta'],20)
    def test_zip_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            z=Path(d)/'bad.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('../escape','x')
            with self.assertRaises(ValueError):Files(z,True)
    def test_zip_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            z=Path(d)/'bad.zip';i=zipfile.ZipInfo('link');i.external_attr=(0o120777<<16)
            with zipfile.ZipFile(z,'w') as f:f.writestr(i,'/etc/passwd')
            with self.assertRaises(ValueError):Files(z,True)
    def test_corrupt_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);b=fixture(p);(b/'i5_cache_summary.json').write_text('tampered');z=p/'bad.zip';archive(p,z)
            f=Files(z,True)
            try:
                with self.assertRaisesRegex(ValueError,'corrupt sealed'):f.verify(b.name)
            finally:f.close()
    def test_raw_run_mode_excludes_sibling_audits_and_rejects_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);exports=p/'exports';exports.mkdir();b=fixture(exports)
            c=json.loads((b/'i5_execution_config.json').read_text());rows=[json.loads(x) for x in (b/'i5_request_metrics.jsonl').read_text().splitlines()];events=[json.loads(x) for x in (b/'i5_cache_events.jsonl').read_text().splitlines()]
            runs=p/'runs';runs.mkdir();root=runs/'block_0_123';root.mkdir()
            for name,v in [('i5_execution_config.json',c),('i5_block_status.json',{'status':'COMPLETE'}),('i5_workload_manifest.json',{'block':c['block']}),('i5_adapter_registry.json',{'registry':REG})]:(root/name).write_text(json.dumps(v))
            for condition in CONDITIONS:
                dest=root/condition;dest.mkdir();(dest/'result.json').write_text(json.dumps({'status':'COMPLETE'}))
                for name,v in [('generation.jsonl',[r for r in rows if r['condition']==condition]),('events.jsonl',[{k:v for k,v in e.items() if k!='condition'} for e in events if e['condition']==condition]),('requests.jsonl',[])]:
                    (dest/name).write_text(''.join(json.dumps(r)+'\n' for r in v))
            seal(root)
            sibling=runs/'block_0_456_audit';sibling.mkdir();(sibling/'ignored').write_text('not a raw block')
            result=run(runs_root=runs,output_dir=p/'out',allow_partial=True,expected_blocks=1)
            self.assertEqual(result['workload']['blocks'],1)
            (root/'i5_block_status.json').write_text('{"status":"FAILED"}');seal(root)
            with self.assertRaisesRegex(ValueError,'incomplete'):run(runs_root=runs,output_dir=p/'out2',allow_partial=True,expected_blocks=1)
            (root/'i5_output_manifest.json').unlink()
            f=Files(runs)
            try:
                with self.assertRaisesRegex(ValueError,'missing sealed evidence manifest'):load_blocks(f,'RAW_RUNS',True,1)
            finally:f.close()
    def test_raw_incomplete_directory_cannot_be_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'block_0_123').mkdir()
            with self.assertRaisesRegex(ValueError,'incomplete block'):Files(d)
    def test_duplicate_and_missing_blocks(self):
        for duplicate in (True,False):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);fixture(p,0)
                if duplicate:
                    import shutil
                    shutil.copytree(p/'block_0_123_audit',p/'block_0_124_audit')
                z=Path(d).parent/(p.name+'.zip');archive(p,z);f=Files(z,True)
                try:
                    with self.assertRaises(ValueError):load_blocks(f,'AUDIT_ZIP',True,2)
                finally:f.close();z.unlink()
    def test_duplicate_block_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);b=fixture(p,0);b2=fixture(p,1);z=p/'x.zip';archive(b,z)
            # Both fixture configs deliberately have identical block identities.
            z2=Path(d).parent/(p.name+'.zip');archive(p,z2);f=Files(z2,True)
            try:
                with self.assertRaisesRegex(ValueError,'duplicate block ID'):load_blocks(f,'AUDIT_ZIP',True,2)
            finally:f.close();z2.unlink()
    def test_historical_snapshot_grouping_unavailable(self):
        blocks=[dict(index=i,config=dict(block={'block_id':str(i)},source_commit='0'*40,sources={})) for i in range(2)]
        r=provenance(blocks);self.assertEqual(r['historical_groups']['0'*40],[0,1]);self.assertTrue(all(b['verification']=='HISTORICAL_OBJECT_UNAVAILABLE' for b in r['blocks']))

if __name__=='__main__':unittest.main()
