"""Offline conversion of the observed M-stage column format; no inference code."""
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path
import hashlib,json,os,re,subprocess
from .communication import require,digest,json_loads
from .r5_i5_data import DST_SLOTS,normalize
from .r5_i5_evidence import fresh,seal,verify as verify_inventory

VERSION='I5_MULTIWOZ22_COLUMN_CONVERSION_V1'
SCHEMA_PATH='data/MultiWOZ_2.2/schema.json'
SPEAKERS={0:'USER',1:'SYSTEM'}
TURN_FIELDS={'turn_id','speaker','utterance','frames','dialogue_acts'}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def columns(value,label,required):
    require(isinstance(value,dict) and required<=value.keys(),label+': missing column fields')
    require(all(isinstance(v,list) for v in value.values()),label+': columns must be lists')
    lengths={len(v) for v in value.values()};require(len(lengths)<=1,label+': unequal column lengths')
    return [{k:deepcopy(v[i]) for k,v in value.items()} for i in range(next(iter(lengths),0))]


def schema_index(schema):
    """Structural validation only; production additionally requires pinned Git bytes."""
    require(isinstance(schema,list) and schema,'invalid MultiWOZ schema')
    services={};all_slots=set()
    for service in schema:
        name=service['service_name'];require(isinstance(name,str) and name and name not in services,'duplicate/invalid schema service')
        require(isinstance(service['slots'],list),'schema slots must be a list')
        names=[s['name'] for s in service['slots']]
        require(all(isinstance(s,str) and s.startswith(name+'-') for s in names) and len(set(names))==len(names),'invalid/duplicate schema slots')
        require(not all_slots.intersection(names),'slot names collide across services')
        services[name]=set(names);all_slots.update(names)
    require(DST_SLOTS<=all_slots,'verified schema lacks I5 tracked slots')
    return services


def pinned_schema(ontology,schema_repository,schema_revision,schema_sha256):
    require(re.fullmatch('[0-9a-f]{40}',schema_revision or '') and re.fullmatch('[0-9a-f]{64}',schema_sha256 or ''),'full upstream schema commit and SHA256 required')
    repo=Path(schema_repository).resolve();path=Path(ontology).resolve()
    require(repo.is_dir() and path.is_file(),'MISSING_SCHEMA_PREREQUISITE: existing official Git checkout and schema.json required; no download attempted')
    def git(*args):
        # Partial clones must not lazily fetch objects during this offline task.
        p=subprocess.run(['git','-c','protocol.allow=never','-C',str(repo),*args],capture_output=True,timeout=30,
            env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_NO_REPLACE_OBJECTS='1',GIT_TERMINAL_PROMPT='0'))
        require(p.returncode==0,'local schema Git proof unavailable: '+p.stderr.decode(errors='replace'))
        return p.stdout
    origin=git('remote','get-url','origin').decode().strip()
    require(origin.rstrip('/').removesuffix('.git') in (
        'https://github.com/budzianowski/multiwoz','git@github.com:budzianowski/multiwoz',
        'ssh://git@github.com/budzianowski/multiwoz'),'schema origin must be the official MultiWOZ repository')
    require(git('rev-parse','--verify',schema_revision+'^{commit}').decode().strip()==schema_revision,'schema revision must identify the pinned commit')
    blob=git('show',schema_revision+':'+SCHEMA_PATH);data=path.read_bytes()
    require(data==blob and hashlib.sha256(data).hexdigest()==schema_sha256,'schema bytes differ from pinned official-repository Git blob/SHA256')
    services=schema_index(json_loads(data))
    return dict(services=services,bytes=data,TEST_FIXTURE_ONLY=False,provenance=dict(repository=str(repo),origin=origin,
        revision=schema_revision,path=SCHEMA_PATH,sha256=schema_sha256,source_schema_file=str(path),
        trust='operator-verified upstream pin; local Git blob comparison; no remote attestation or fetch'))


def state_values(state,service,services,user):
    if not user and state in (None,{}):return None
    require(isinstance(state,dict) and {'active_intent','requested_slots','slots_values'}<=state.keys() and 'slot_values' not in state,
        'missing USER state or unsupported state schema')
    require(isinstance(state['active_intent'],str),'active_intent must be preserved text')
    requested=state['requested_slots'];require(isinstance(requested,list) and all(isinstance(s,str) and s in services[service] for s in requested),'invalid requested_slots/service')
    sv=state['slots_values']
    require(isinstance(sv,dict) and set(sv)=={'slots_values_name','slots_values_list'},'invalid slots_values columns')
    names,values=sv['slots_values_name'],sv['slots_values_list']
    require(isinstance(names,list) and isinstance(values,list) and len(names)==len(values),'unequal slot-value arrays')
    require(all(isinstance(s,str) and s in services[service] and s in DST_SLOTS for s in names),'unsupported or wrong-service I5 state slot')
    require(len(names)==len(set(names)),'duplicate slot names')
    require(all(isinstance(v,list) and v and all(isinstance(x,str) and x.strip() for x in v) for v in values),'invalid slot values; no coercion or discarded alternatives permitted')
    return dict(zip(names,deepcopy(values)))


def slot_rows(slots,service,services):
    if isinstance(slots,dict):rows=columns(slots,'frame slots',set())
    else:
        require(isinstance(slots,list) and all(isinstance(s,dict) for s in slots),'unsupported slot annotations')
        rows=deepcopy(slots)
    for row in rows:
        require(isinstance(row.get('slot'),str) and row['slot'] in services[service],'unsupported slot annotation/service')
    return rows


def convert_dialogue(record,services,*,fixture=False):
    require(isinstance(record,dict) and {'dialogue_id','services','turns'}<=record.keys(),'invalid dialogue record')
    require(record.get('TEST_FIXTURE_ONLY',False) is fixture,'fixture/real source scope mismatch')
    require(isinstance(record['dialogue_id'],str) and record['dialogue_id'],'invalid dialogue_id')
    declared=record['services']
    require(isinstance(declared,list) and declared and all(isinstance(s,str) and s in services for s in declared) and len(set(declared))==len(declared),'unsupported/duplicate dialogue services')
    turns=columns(record['turns'],'turns',TURN_FIELDS);require(turns,'empty dialogue')
    seen=set();users=0
    for turn in turns:
        tid=turn['turn_id'];speaker=turn['speaker']
        require(isinstance(tid,str) and tid and tid not in seen,'duplicate/invalid turn_id');seen.add(tid)
        require(type(speaker) is int and speaker in SPEAKERS,'speaker must be integer0 USER or1 SYSTEM')
        require(isinstance(turn['utterance'],str) and isinstance(turn['dialogue_acts'],dict),'invalid utterance/dialogue_acts')
        turn['speaker']=SPEAKERS[speaker];user=speaker==0;users+=user
        frames=columns(turn['frames'],'frames',{'service','state','slots'})
        require(not user or frames,'missing USER frames/state')
        frame_services=set()
        for frame in frames:
            service=frame['service']
            require(isinstance(service,str) and service in declared and service not in frame_services,'unsupported/duplicate frame service')
            frame_services.add(service)
            values=state_values(frame['state'],service,services,user)
            if values is not None:
                state=frame['state'];state.pop('slots_values');state['slot_values']=values
            frame['slots']=slot_rows(frame['slots'],service,services)
        turn['frames']=frames
    require(users>0,'dialogue has no USER turns')
    native=deepcopy(record);native['turns']=turns
    return native


def validate_dialogue(source,native,services,*,fixture=False):
    """Compare source columns with output rows and I5's actual accumulated labels."""
    require(digest({k:v for k,v in source.items() if k!='turns'})==digest({k:v for k,v in native.items() if k!='turns'}),'dialogue identity/metadata differs')
    cols=source['turns'];turns=native['turns'];require(len(turns)==len(cols['turn_id']),'turn count differs')
    by_service={};expected=[];history=[]
    for i,turn in enumerate(turns):
        require(set(turn)==set(cols),'turn fields differ')
        for key,value in cols.items():
            if key not in ('speaker','frames'):require(digest(turn[key])==digest(value[i]),'turn order/text/metadata differs')
        require(turn['speaker']==SPEAKERS[cols['speaker'][i]],'speaker mapping differs')
        history.append(turn['speaker']+': '+turn['utterance']);fc=cols['frames'][i]
        require(len(turn['frames'])==len(fc['service']),'frame count differs')
        for j,frame in enumerate(turn['frames']):
            require(set(frame)==set(fc),'frame fields differ')
            for key,value in fc.items():
                if key not in ('state','slots'):require(digest(frame[key])==digest(value[j]),'frame service/metadata differs')
            old=fc['state'][j];new=frame['state'];service=fc['service'][j]
            if old not in (None,{}):
                require(digest({k:v for k,v in old.items() if k!='slots_values'})==digest({k:v for k,v in new.items() if k!='slot_values'}),'state contents lost')
                names=old['slots_values']['slots_values_name'];values=old['slots_values']['slots_values_list']
                require(list(new['slot_values'])==names and digest(list(new['slot_values'].values()))==digest(values),'state names/alternatives/types/order differ')
            else:require(digest(old)==digest(new),'empty SYSTEM state changed')
            old_slots=fc['slots'][j];new_slots=frame['slots']
            if isinstance(old_slots,dict):
                require(all(set(s)==set(old_slots) for s in new_slots),'slot annotation fields differ')
                for k,v in old_slots.items():require(digest([s[k] for s in new_slots])==digest(v),'slot annotation column lost')
            else:require(digest(old_slots)==digest(new_slots),'slot annotations changed')
            if turn['speaker']=='USER':by_service[service]=dict(zip(old['slots_values']['slots_values_name'],old['slots_values']['slots_values_list']))
        if turn['speaker']=='USER':expected.append((f"{source['dialogue_id']}:{cols['turn_id'][i]}",{k:v for state in by_service.values() for k,v in state.items()},'\n'.join(history)))
    rows=normalize('multiwoz',[native],ontology=sorted(DST_SLOTS),fixture=fixture)
    require(len(rows)==len(expected),'I5 USER sample count differs')
    for row,(rid,refs,text) in zip(rows,expected):
        require(row['id']==rid and row['group_id']==source['dialogue_id'] and digest(row['references'])==digest(refs) and row['history']==text,'I5 accumulated state/history differs')
    return dict(dialogue_id=source['dialogue_id'],turns=len(turns),user_turns=len(rows),frames=sum(len(t['frames']) for t in turns),i5_labels_verified=True)


def records(path):
    with Path(path).open('rb') as f:
        for line_number,raw in enumerate(f,1):
            require(raw.strip(),'blank JSONL record at line '+str(line_number))
            yield line_number,raw,json_loads(raw)


def _convert(source,output_dir,schema,source_sha256,*,fixture=False):
    require(re.fullmatch('[0-9a-f]{64}',source_sha256 or ''),'explicit source SHA256 required')
    source=Path(source).resolve();require(source.is_file() and file_sha(source)==source_sha256,'missing or changed source JSONL')
    require(schema['TEST_FIXTURE_ONLY'] is fixture,'unverified/test schema cannot authorize real conversion')
    require(not Path(output_dir).is_symlink(),'output directory symlink forbidden')
    protected=[source,*[schema['provenance'][k] for k in ('repository','source_schema_file') if k in schema['provenance']]]
    out=fresh(output_dir,protected);seen=set();sample_ids=set();counts=dict(dialogues=0,turns=0,user_turns=0,frames=0);h=hashlib.sha256()
    manifest=dict(schema=VERSION,status='FAILED',TEST_FIXTURE_ONLY=fixture,source=str(source),source_sha256=source_sha256,
        ontology=schema['provenance'],speaker_encoding={'0':'USER','1':'SYSTEM'},
        converter_sha256=file_sha(__file__),GPU_status='GPU_NOT_RUN')
    try:
        with (out/'native.partial').open('x') as f,(out/'records.jsonl').open('x') as evidence:
            for number,raw,record in records(source):
                h.update(raw);require(record['dialogue_id'] not in seen,'duplicate dialogue_id');seen.add(record['dialogue_id'])
                native=convert_dialogue(record,schema['services'],fixture=fixture)
                stats=validate_dialogue(record,native,schema['services'],fixture=fixture)
                for t in native['turns']:
                    if t['speaker']=='USER':
                        sid=native['dialogue_id']+':'+t['turn_id'];require(sid not in sample_ids,'duplicate I5 USER sample identity');sample_ids.add(sid)
                for key in ('turns','user_turns','frames'):counts[key]+=stats[key]
                counts['dialogues']+=1
                f.write(json.dumps(native,ensure_ascii=False,allow_nan=False)+'\n')
                evidence.write(json.dumps(dict(stats,record_ordinal=counts['dialogues']-1,source_line=number,
                    source_line_sha256=hashlib.sha256(raw).hexdigest(),source_record_sha256=digest(record),native_record_sha256=digest(native)),sort_keys=True)+'\n')
        require(counts['dialogues'] and h.hexdigest()==source_sha256==file_sha(source),'empty or changed source during conversion')
        (out/'schema.json').write_bytes(schema['bytes'])
        os.rename(out/'native.partial',out/'native.jsonl')
        manifest.update(status='COMPLETE',counts=counts,native_sha256=file_sha(out/'native.jsonl'),
            validation='source/row state comparison and actual I5 normalize; no state accumulation in converted frames')
    except BaseException as exc:manifest['reason']=str(exc);raise
    finally:
        (out/'conversion_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n');seal(out)
    return manifest


def convert(source,output_dir,ontology,schema_repository,schema_revision,schema_sha256,source_sha256):
    schema=pinned_schema(ontology,schema_repository,schema_revision,schema_sha256)
    return _convert(source,output_dir,schema,source_sha256)


def verify(conversion_dir):
    out=Path(conversion_dir).resolve();verify_inventory(out);m=json_loads((out/'conversion_manifest.json').read_bytes())
    require(m['schema']==VERSION and m['status']=='COMPLETE' and m['TEST_FIXTURE_ONLY'] is False,'complete real conversion required')
    require(m['converter_sha256']==file_sha(__file__),'converter source changed; use the recorded version for verification')
    pin=m['ontology'];schema=pinned_schema(out/'schema.json',pin['repository'],pin['revision'],pin['sha256'])
    require(file_sha(m['source'])==m['source_sha256'] and file_sha(out/'native.jsonl')==m['native_sha256'],'source/native SHA256 differs')
    counts=_verify_files(m['source'],out/'native.jsonl',out/'records.jsonl',schema['services'])
    require(counts==m['counts'],'conversion totals differ')
    return dict(status='CPU_CONVERSION_VALIDATED',counts=counts,GPU_status='GPU_NOT_RUN')


def _verify_files(source_path,native_path,proof_path,services,*,fixture=False):
    counts=dict(dialogues=0,turns=0,user_turns=0,frames=0);seen=set();sample_ids=set()
    for ordinal,items in enumerate(zip_longest(records(source_path),records(native_path),records(proof_path))):
        a,b,c=items;require(a is not None and b is not None and c is not None,'dialogue/provenance counts differ')
        number,raw,source=a;native=b[2];proof=c[2]
        # Validate source shape independently before comparing saved row evidence.
        convert_dialogue(source,services,fixture=fixture)
        stats=validate_dialogue(source,native,services,fixture=fixture)
        require(source['dialogue_id'] not in seen,'duplicate dialogue_id');seen.add(source['dialogue_id'])
        for t in native['turns']:
            if t['speaker']=='USER':
                sid=native['dialogue_id']+':'+t['turn_id'];require(sid not in sample_ids,'duplicate I5 USER sample identity');sample_ids.add(sid)
        require(proof==dict(stats,record_ordinal=ordinal,source_line=number,source_line_sha256=hashlib.sha256(raw).hexdigest(),
            source_record_sha256=digest(source),native_record_sha256=digest(native)),'record identity/validation metadata differs')
        counts['dialogues']+=1
        for key in ('turns','user_turns','frames'):counts[key]+=stats[key]
    return counts
