"""TEST_FIXTURE_ONLY column-format conversion; never real MultiWOZ validation."""
from copy import deepcopy
from pathlib import Path
import contextlib,hashlib,importlib.util,io,json,subprocess,tempfile,unittest
from unittest.mock import patch
from semcache.edgelora import r5_multiwoz_native as api
from semcache.edgelora.communication import digest
from semcache.edgelora.r5_i5_data import DST_SLOTS,normalize,read_native
from semcache.edgelora.r5_i5_quality import score
from semcache.edgelora.r5_i5_evidence import verify as inventory_verify


def fixture_schema():
    """Artificial structural fixture; never an authenticated production ontology."""
    names=sorted({s.split('-')[0] for s in DST_SLOTS})
    schema=[dict(service_name=n,slots=[dict(name=s) for s in sorted(DST_SLOTS) if s.startswith(n+'-')]) for n in names]
    data=json.dumps(schema).encode()
    return dict(services=api.schema_index(schema),bytes=data,TEST_FIXTURE_ONLY=True,
        provenance=dict(TEST_FIXTURE_ONLY=True,sha256=hashlib.sha256(data).hexdigest()))


def state(names=(),values=()):
    return dict(active_intent='find_hotel',requested_slots=[],slots_values=dict(slots_values_name=list(names),slots_values_list=deepcopy(list(values))),
        preserved_state_metadata={'TEST_FIXTURE_ONLY':True})


def frame(services,states,slots=None):
    return dict(service=services,state=states,slots=slots if slots is not None else [{} for s in services])


def dialogue(rid='TEST_FIXTURE_ONLY_DIALOGUE'):
    annotations=dict(slot=['hotel-pricerange'],value=['Cheap'],start=[4],exclusive_end=[9],copy_from=[''],copy_from_value=[[]])
    return dict(TEST_FIXTURE_ONLY=True,dialogue_id=rid,services=['hotel','train'],source_metadata={'keep':'unchanged'},turns=dict(
        turn_id=['0','1','2','3','4'],speaker=[0,1,0,1,0],utterance=['A cheap hotel for 2–3 people','OK','Train to Cambridge','OK','Clear hotel constraints'],
        frames=[frame(['hotel'],[state(['hotel-pricerange','hotel-bookpeople'],[['Cheap','inexpensive'],['2','3']])],[annotations]),
                frame(['hotel'],[state(['hotel-name'],[['SYSTEM metadata, not gold']])]),
                frame(['train'],[state(['train-destination'],[['Cambridge','cambridge']])]),
                frame([],[]),frame(['hotel'],[state()])],
        dialogue_acts=[{'keep_numeric_value':n,'nested':['original']} for n in range(5)]))


class Conversion(unittest.TestCase):
    def setUp(self):self.schema=fixture_schema();self.source=dialogue()

    def converted(self):return api.convert_dialogue(self.source,self.schema['services'],fixture=True)

    def test_order_identity_speaker_service_and_alternatives_preserved(self):
        before=deepcopy(self.source);out=self.converted();stats=api.validate_dialogue(self.source,out,self.schema['services'],fixture=True)
        self.assertEqual(self.source,before);self.assertEqual(out['dialogue_id'],self.source['dialogue_id'])
        self.assertEqual([t['turn_id'] for t in out['turns']],self.source['turns']['turn_id'])
        self.assertEqual([t['speaker'] for t in out['turns']],['USER','SYSTEM','USER','SYSTEM','USER'])
        self.assertEqual([t['utterance'] for t in out['turns']],self.source['turns']['utterance'])
        self.assertEqual([t['dialogue_acts'] for t in out['turns']],self.source['turns']['dialogue_acts'])
        values=out['turns'][0]['frames'][0]['state']['slot_values']
        self.assertEqual(list(values),['hotel-pricerange','hotel-bookpeople']);self.assertEqual(values['hotel-bookpeople'],['2','3'])
        self.assertEqual(stats,dict(dialogue_id=self.source['dialogue_id'],turns=5,user_turns=3,frames=4,i5_labels_verified=True))

    def test_slot_annotation_column_types_and_state_metadata_are_preserved(self):
        out=self.converted();slot=out['turns'][0]['frames'][0]['slots'][0]
        self.assertEqual(slot,dict(slot='hotel-pricerange',value='Cheap',start=4,exclusive_end=9,copy_from='',copy_from_value=[]))
        self.assertIs(type(slot['start']),int);self.assertEqual(out['turns'][0]['frames'][0]['state']['preserved_state_metadata'],{'TEST_FIXTURE_ONLY':True})
        self.source['turns']['frames'][0]['slots']=[[slot]]
        self.assertEqual(self.converted()['turns'][0]['frames'][0]['slots'],[slot])

    def test_i5_accumulation_is_per_service_replacement_and_system_state_is_not_gold(self):
        out=self.converted();rows=normalize('multiwoz',[out],fixture=True,ontology=sorted(DST_SLOTS))
        self.assertEqual([r['id'] for r in rows],[self.source['dialogue_id']+':'+n for n in ('0','2','4')])
        self.assertNotIn('hotel-name',rows[1]['references'])
        self.assertEqual(rows[1]['references']['hotel-pricerange'],['Cheap','inexpensive'])
        self.assertEqual(rows[2]['references'],{'train-destination':['Cambridge','cambridge']})
        self.assertEqual(out['turns'][4]['frames'][0]['state']['slot_values'],{})
        self.assertEqual(score(rows[0],json.dumps({'hotel-pricerange':' INEXPENSIVE ','hotel-bookpeople':'3'}),'I5_PROMPTS_V1')['score'],1.)
        self.assertEqual(score(rows[2],json.dumps({'train-destination':' CAMBRIDGE '}),'I5_PROMPTS_V1')['score'],1.)
        self.assertIn('SYSTEM: OK',rows[1]['history']);self.assertEqual(rows[0]['references']['hotel-pricerange'][0],'Cheap')

    def test_empty_system_state_is_preserved_and_no_speaker_inferred_from_position(self):
        self.source['turns']['frames'][1]['state']=[None]
        self.assertIsNone(self.converted()['turns'][1]['frames'][0]['state'])
        self.source['turns']['speaker'][1]=0
        with self.assertRaises(ValueError):self.converted()

    def test_unequal_turn_frame_slot_value_and_annotation_arrays_fail(self):
        mutations=[lambda r:r['turns']['utterance'].pop(),lambda r:r['turns']['frames'][0]['service'].append('train'),
            lambda r:r['turns']['frames'][0]['state'][0]['slots_values']['slots_values_list'].pop(),
            lambda r:r['turns']['frames'][0]['slots'][0]['value'].append('extra')]
        for change in mutations:
            with self.subTest(mutation=change):
                self.source=dialogue();change(self.source)
                with self.assertRaises(ValueError):self.converted()

    def test_invalid_speakers_ids_services_and_frames_fail(self):
        for speaker in (True,-1,2,'0','USER',None):
            self.source=dialogue();self.source['turns']['speaker'][0]=speaker
            with self.assertRaises(ValueError):self.converted()
        for change in (lambda r:r['turns']['turn_id'].__setitem__(1,'0'),
            lambda r:r['services'].append('unsupported'),lambda r:r['turns']['frames'][0]['service'].__setitem__(0,'train'),
            lambda r:r['turns']['frames'].__setitem__(0,[]),lambda r:r['turns']['frames'].__setitem__(0,frame([],[]))):
            self.source=dialogue();change(self.source)
            with self.assertRaises((ValueError,KeyError)):self.converted()

    def test_missing_user_states_duplicate_slot_names_and_invalid_values_fail(self):
        for change in (lambda s:s.pop('slots_values'),lambda s:s.update(slot_values={}),
            lambda s:s.update(active_intent=None),lambda s:s.update(requested_slots=['train-day']),
            lambda s:s['slots_values']['slots_values_name'].__setitem__(1,'hotel-pricerange'),
            lambda s:s['slots_values']['slots_values_name'].__setitem__(0,'train-destination')):
            self.source=dialogue();change(self.source['turns']['frames'][0]['state'][0])
            with self.assertRaises(ValueError):self.converted()
        for value in ([],None,'cheap',[1],[True],[''],[' '],[None],[["cheap"]]):
            self.source=dialogue();self.source['turns']['frames'][0]['state'][0]['slots_values']['slots_values_list'][0]=value
            with self.assertRaises(ValueError):self.converted()

    def test_duplicate_service_and_unsupported_i5_slots_are_not_merged_or_dropped(self):
        self.source['turns']['frames'][0]=frame(['hotel','hotel'],[state(),state()])
        with self.assertRaises(ValueError):self.converted()
        self.source=dialogue();self.schema['services']['hotel'].add('hotel-extra')
        self.source['turns']['frames'][0]['state'][0]=state(['hotel-extra'],[['present, never drop']])
        with self.assertRaises(ValueError):self.converted()

    def test_independent_comparison_rejects_type_state_identity_and_order_loss(self):
        out=self.converted()
        for change in (lambda n:n['turns'][0]['frames'][0]['state']['slot_values']['hotel-bookpeople'].__setitem__(0,2),
            lambda n:n['turns'].reverse(),lambda n:n['turns'][0]['frames'][0]['state'].pop('preserved_state_metadata'),
            lambda n:n['turns'][0]['dialogue_acts'].clear()):
            bad=deepcopy(out);change(bad)
            with self.assertRaises((ValueError,KeyError)):api.validate_dialogue(self.source,bad,self.schema['services'],fixture=True)

    def test_file_conversion_jsonl_despite_json_suffix_and_i5_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'test.json';original=[self.source,dialogue('TEST_FIXTURE_ONLY_SECOND')]
            source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in original));before=source.read_bytes()
            out=root/'converted';m=api._convert(source,out,self.schema,api.file_sha(source),fixture=True)
            self.assertEqual(m['counts'],dict(dialogues=2,turns=10,user_turns=6,frames=8));self.assertEqual(source.read_bytes(),before)
            inventory_verify(out)
            rows,ontology=read_native(out/'native.jsonl','multiwoz','test','jsonl',out/'schema.json')
            self.assertEqual([r['dialogue_id'] for r in rows],[r['dialogue_id'] for r in original])
            self.assertEqual(len(normalize('multiwoz',rows,fixture=True,ontology=ontology)),6)
            self.assertEqual(api._verify_files(source,out/'native.jsonl',out/'records.jsonl',self.schema['services'],fixture=True),m['counts'])
            proofs=[r for _,_,r in api.records(out/'records.jsonl')]
            self.assertEqual(proofs[0]['source_record_sha256'],digest(original[0]));self.assertEqual(proofs[1]['record_ordinal'],1)
            with self.assertRaises(ValueError):api.verify(out)  # Fixtures never authorize real conversion.
            with self.assertRaises(ValueError):api._convert(source,out,self.schema,api.file_sha(source),fixture=True)

    def test_missing_source_wrong_hash_and_failed_conversion_do_not_publish_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'test.json';source.write_text(json.dumps(self.source)+'\n')
            with self.assertRaises(ValueError):api._convert(source,root/'wrong',self.schema,'0'*64,fixture=True)
            self.assertFalse((root/'wrong').exists())
            source.write_text(json.dumps(self.source)+'\n'+json.dumps(self.source)+'\n')
            with self.assertRaisesRegex(ValueError,'duplicate dialogue'):api._convert(source,root/'bad',self.schema,api.file_sha(source),fixture=True)
            self.assertFalse((root/'bad/native.jsonl').exists())
            self.assertEqual(json.loads((root/'bad/conversion_manifest.json').read_text())['status'],'FAILED');inventory_verify(root/'bad')

    def test_global_i5_sample_identity_collisions_fail_without_renaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=dialogue('TEST_FIXTURE_ONLY:a');b=dialogue('TEST_FIXTURE_ONLY');b['turns']['turn_id'][0]='a:0'
            source=root/'test.json';source.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n')
            with self.assertRaisesRegex(ValueError,'I5 USER sample identity'):api._convert(source,root/'out',self.schema,api.file_sha(source),fixture=True)
            self.assertFalse((root/'out/native.jsonl').exists())

    def test_strict_json_duplicate_fields_and_nonfinite_values_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'test.json'
            for text in ('{"dialogue_id":"a","dialogue_id":"b"}\n','{"x":NaN}\n','\n'):
                p.write_text(text)
                with self.assertRaises(ValueError):list(api.records(p))

    def test_schema_requires_authentic_pinned_git_bytes_and_complete_ontology(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'schema.json';p.write_bytes(self.schema['bytes']);revision='a'*40;sha=api.file_sha(p)
            def git(argv,**kwargs):
                self.assertIn('protocol.allow=never',argv);self.assertEqual(kwargs['env']['GIT_NO_LAZY_FETCH'],'1')
                command=argv[argv.index('-C')+2:]
                data={('remote','get-url','origin'):b'https://github.com/budzianowski/multiwoz.git\n',
                    ('rev-parse','--verify',revision+'^{commit}'):(revision+'\n').encode(),
                    ('show',revision+':'+api.SCHEMA_PATH):self.schema['bytes']}[tuple(command)]
                return subprocess.CompletedProcess(argv,0,data,b'')
            # Mocked local Git proof only; never provenance for the fixture schema.
            with patch.object(api.subprocess,'run',side_effect=git):
                self.assertEqual(api.pinned_schema(p,tmp,revision,sha)['services'],self.schema['services'])
                with self.assertRaises(ValueError):api.pinned_schema(p,tmp,revision,'0'*64)
                p.write_text('[]')
                with self.assertRaises(ValueError):api.pinned_schema(p,tmp,revision,sha)
            with patch.object(api.subprocess,'run',return_value=subprocess.CompletedProcess([],0,b'https://example.org/fake\n',b'')):
                with self.assertRaises(ValueError):api.pinned_schema(p,tmp,revision,sha)
            with self.assertRaises(ValueError):api.pinned_schema(p,tmp,'abbreviated',sha)
            with self.assertRaises(ValueError):api.pinned_schema(Path(tmp)/'missing',tmp,revision,sha)
            with self.assertRaises(ValueError):api.schema_index([dict(service_name='hotel',slots=[])])

    def test_cli_has_no_download_or_fixture_authorization_and_blocks_missing_schema(self):
        script=Path(api.__file__).resolve().parents[3]/'scripts/159_convert_r5_multiwoz.py'
        spec=importlib.util.spec_from_file_location('conversion_cli',script);cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        with patch.object(api,'convert',return_value={'TEST_FIXTURE_ONLY':True}) as call,contextlib.redirect_stdout(io.StringIO()):
            args=['convert']
            for key,value in dict(source='source',source_sha256='1'*64,output_dir='fresh',ontology='missing',schema_repository='repo',schema_revision='a'*40,schema_sha256='2'*64).items():args+=['--'+key.replace('_','-'),value]
            self.assertEqual(cli.main(args),0);self.assertNotIn('fixture',call.call_args.kwargs)
        with contextlib.redirect_stderr(io.StringIO()):self.assertEqual(cli.main(args),2)


if __name__=='__main__':unittest.main()
