"""C7-B0: provenance-bound, quality-blind TOTAL-Q calibration capture."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

from . import c6b3_train as b3
from .c6b3_2_multiwoz import WEIGHTS, SELECTION_SHA, verify_adapters
from .c6_quality import PROFILE_SHA

MODEL = b3.b.m9.MODEL_ID
REVISION = b3.b.m9.MODEL_REVISION
VERSION = b3.b.VERSION
USERS = ('user_a', 'user_b')
Q_OBJECT = 'resident TOTAL Q = base Q + active-user LoRA Q'
DEFAULT_ROOT = 'results/cachegen/c7/b_q_calibration'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return b3.b.digest(value)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def git():
    return {k: subprocess.check_output(['git', *v], text=True).strip() for k,v in
        [('branch',['branch','--show-current']), ('commit',['rev-parse','HEAD']), ('status',['status','--porcelain'])]}


def verify_files(hashes):
    for path, expected in hashes.items():
        require(Path(path).is_file() and sha(path)==expected, 'Provenance file changed/missing: '+path)


def bound_cohort(path):
    """Only known B3 schemas; every discovered evaluation cohort must be hash-bound."""
    path = Path(path)
    owners = []
    for name in ('manifest.json', 'training_manifest.json', 'capability_manifest.json'):
        owner = path.parent/name
        if not owner.is_file():
            continue
        m = read(owner)
        hashes = m.get('output_hashes', {})
        expected = hashes.get(str(path.resolve()), hashes.get(path.name))
        if path.name=='capability_validation.json':
            expected = expected or m.get('capability_validation_file_sha256')
        if expected == sha(path):
            owners.append(owner)
    require(owners, 'Unbound B3 exclusion cohort: '+str(path))
    if path.suffix=='.csv':
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        require(rows and all(r.get('conversation_id') for r in rows), 'Invalid B3 evaluation case schema')
        cids = {r['conversation_id'] for r in rows}
    else:
        data = read(path)
        if path.name=='evaluation_selection.json':
            require(data.get('episodes') and data.get('selection_sha256')==digest(data['episodes']), 'Invalid B3 selection digest')
            cids = {e[s+'_conversation_id'] for e in data['episodes'] for s in ('source','target')}
        else:
            require(data.get('cohort')=='capability64' and len(data.get('examples',[]))==64, 'Unknown capability schema')
            cids = {r['conversation_id'] for r in data['examples']}
            require(cids==set(data['conversation_ids']) and len(cids)==64, 'Capability IDs differ')
    return cids, {str(p.resolve()):sha(p) for p in [path, *owners]}


def load_provenance(args):
    # Existing B3 parser replays source/history/token/selection/training ownership.
    rows, training, frozen, provenance = b3.load_inputs(args.plan_dir, args.source, args.semantic)
    require(sha(args.plan_dir/'evaluation_selection.json')==SELECTION_SHA, 'Frozen32 file hash differs')
    trained = read(args.adapter_root/'training_manifest.json')
    verify_adapters(args.adapter_root, trained)
    for k,v in {**b3.SCOPE, **provenance}.items():
        require(trained.get(k)==v, 'Frozen training provenance mismatch: '+k)
    saved = read(args.adapter_root/'train_selection.json')
    capability = read(args.adapter_root/'capability_validation.json')
    require(capability==b3.capability_cohort(rows,training), 'Capability64 does not replay')
    require(sha(args.adapter_root/'capability_validation.json')==trained['capability_validation_file_sha256'], 'Capability hash mismatch')
    require(digest(saved)==trained['plan_sha256'] and sha(args.adapter_root/'train_selection.json')==trained['train_selection_file_sha256'], 'Saved full training pool differs')
    require(saved['profile']=='full' and saved['epochs']==2, 'Expected frozen full two-epoch pool')
    replay = b3.training_subset(rows, training, capability, 'full')
    for user in USERS:
        require(all(saved['users'][user].get(k)==v for k,v in replay['users'][user].items()), 'Saved user training pool differs: '+user)
    hold = {e[s+'_conversation_id'] for e in frozen['episodes'] for s in ('source','target')}
    caps = set(capability['conversation_ids'])
    excluded = hold|caps
    inputs = {str(p.resolve()):sha(p) for p in (args.source,args.semantic,
        args.plan_dir/'manifest.json',args.plan_dir/'evaluation_selection.json',args.plan_dir/'training_plan.json',
        args.plan_dir/'current_user_spans.json',args.adapter_root/'training_manifest.json',
        args.adapter_root/'train_selection.json',args.adapter_root/'capability_validation.json')}
    # Include old/pilot/final B3 cohorts, never silently ignore an unbound file.
    roots = {args.adapter_root.parent.resolve(), args.plan_dir.resolve()}
    scanned = set()
    for root in roots:
        for name in ('evaluation_selection.json','capability_validation.json','capability_per_case.csv'):
            for path in sorted(root.rglob(name)):
                if path.resolve() in scanned:
                    continue
                scanned.add(path.resolve())
                ids, hashes = bound_cohort(path)
                excluded |= ids
                inputs.update(hashes)
    for user in USERS:
        for p in (args.adapter_root/user).rglob('*'):
            if p.is_file(): inputs[str(p.resolve())]=sha(p)
    return rows, saved, dict(frozen32_conversation_ids=sorted(hold), capability64_conversation_ids=sorted(caps),
        all_excluded_conversation_ids=sorted(excluded), input_hashes=inputs,
        adapter_paths={u:str((args.adapter_root/u).resolve()) for u in USERS}, adapter_hashes=WEIGHTS,
        model=MODEL, model_revision=REVISION, tokenizer_revision=REVISION, prompt_version=VERSION,
        dataset_workload_hashes=provenance, exclusion_schema='B3 hash-bound evaluation_selection/capability_validation/capability_per_case artifacts')


def build_cohort(rows, pool, provenance, seed=42):
    require(seed==42, 'C7-B seed must be 42')
    excluded = set(provenance['all_excluded_conversation_ids'])
    by_id = {r['source_id']:r for r in rows}
    require(len(by_id)==len(rows), 'Duplicate source IDs')
    candidates = {}
    for user in USERS:
        for depth in range(1,5):
            groups = {}
            for sid in pool['users'][user]['row_ids']:
                r = by_id[sid]
                if r['conversation_id'] in excluded or r['history_depth']!=depth:
                    continue
                require(r['prompt_version']==VERSION and r['prompt_sha256']==digest(r['prompt_text']), 'Prompt changed')
                allowed = set(r['current_user_token_indices'])
                specials = {0,1,2}|set(r.get('special_token_ids',[]))
                for start in sorted(allowed):
                    ids = r['token_ids'][start:start+3]
                    if len(ids)!=3 or not {start,start+1,start+2}<=allowed or specials.intersection(ids):
                        continue
                    choice = dict(user=user,history_depth=depth,conversation_id=r['conversation_id'],source_id=sid,
                        token_ids=ids,w3_start=start,prompt_sha256=r['prompt_sha256'],
                        prompt_utf8_sha256=hashlib.sha256(r['prompt_text'].encode()).hexdigest(),
                        reference_sha256=r['reference_sha256'],current_user_token_indices=sorted(allowed),
                        current_user_char_span=r['current_user_char_span'],source_split=r.get('source_split'),
                        prompt_token_count=len(r['token_ids']),special_token_ids=sorted(specials))
                    groups.setdefault(r['conversation_id'],[]).append(choice)
            # Choose row/window by stable hash, without inspecting tensor values.
            candidates[user,depth] = {cid:min(values,key=lambda x:(digest(['C7-B-window',seed,x['source_id'],x['w3_start']]),x['source_id'],x['w3_start']))
                                      for cid,values in groups.items()}
    ordered = {key:sorted(values,key=lambda cid:(digest(['C7-B-conversation',seed,*key,cid]),cid)) for key,values in candidates.items()}
    owners, assigned = {}, {}
    def match(slot, seen):
        key = slot[:2]
        for cid in ordered[key]:
            if cid in seen:continue
            seen.add(cid)
            if cid not in owners or match(owners[cid],seen):
                owners[cid]=slot;assigned[slot]=candidates[key][cid];return True
        return False
    for user in USERS:
        for depth in range(1,5):
            for i in range(16):
                require(match((user,depth,i),set()), f'Insufficient eligible distinct conversations: {user}/depth{depth}')
    blocks = [dict(assigned[slot], calibration_id=f'c7b:{slot[0]}:d{slot[1]}:{slot[2]:02}',
                   split='profile_fit' if slot[2]<12 else 'candidate_select') for slot in sorted(assigned)]
    cohort = dict(stage='C7-B0',seed=seed,blocks=blocks,provenance=provenance,
        selection_rule='seed42 stable hash + deterministic bipartite conversation matching; one block per conversation; split fixed before capture',
        prompt_hash_encoding='B3 canonical JSON SHA256; UTF8 SHA256 recorded separately')
    validate_cohort(cohort)
    return cohort


def validate_cohort(cohort):
    blocks = cohort['blocks'];p = cohort['provenance']
    require(cohort['seed']==42 and len(blocks)==128, 'Need seed42 exactly128 blocks')
    require(len({b['calibration_id'] for b in blocks})==128 and len({b['conversation_id'] for b in blocks})==128, 'Duplicate block/conversation')
    expected = {(u,d,s):n for u in USERS for d in range(1,5) for s,n in [('profile_fit',12),('candidate_select',4)]}
    require(Counter((b['user'],b['history_depth'],b['split']) for b in blocks)==expected, 'User/depth/split imbalance')
    selected = {b['conversation_id'] for b in blocks}
    for key in ('frozen32_conversation_ids','capability64_conversation_ids','all_excluded_conversation_ids'):
        require(not selected&set(p[key]), 'Calibration/evaluation conversation overlap: '+key)
    for b in blocks:
        start=b['w3_start']
        require(len(b['token_ids'])==3 and all(type(t) is int and t>=0 for t in b['token_ids']), 'Invalid w3 tokens')
        require({start,start+1,start+2}<=set(b['current_user_token_indices']) and start>=0 and start+3<=b['prompt_token_count'], 'w3 outside current user')
        require(not set(b['token_ids'])&set(b['special_token_ids']), 'Special-token window')
    return dict(frozen32_overlap_count=0,capability64_overlap_count=0,
                fit_select_conversation_overlap_count=0,other_frozen_cohort_overlap_count=0)


@contextmanager
def total_q_capture(adapter, start, *, layers=32, hidden=2560):
    """Same passive native-projection hooks as models.capture, owning only w3 Q.

    Unlike qkv_capture's diagnostic records, never retain full prompt Q/K/V or
    hidden states and never re-project. Hook sees TOTAL PEFT q_proj output.
    """
    import torch
    require(len(adapter.layers)==layers, 'Unexpected projection layer count')
    captured={};handles=[]
    def hook(layer):
        def save(module, args, output):
            require(layer not in captured and output.ndim==3 and output.shape[0]==1 and output.shape[2]==hidden,
                    'Invalid/duplicate TOTAL-Q output')
            require(start>=0 and start+3<=output.shape[1], 'Q window outside prompt')
            captured[layer]=output[:,start:start+3].detach().to(device='cpu',dtype=torch.float16).clone().contiguous()
        return save
    try:
        for layer in range(layers):
            handles.append(adapter.projection_modules(layer)['q'].register_forward_hook(hook(layer)))
        yield captured
        validate_q_block(torch.cat([captured[l] for l in range(layers)],dim=0), hidden=hidden)
    finally:
        for handle in handles:handle.remove()


def validate_q_block(q, hidden=2560):
    import torch
    require(q.shape==(32,3,hidden) and q.dtype==torch.float16 and q.device.type=='cpu', 'Expected CPU FP16 Q [32,3,hidden]')
    require(torch.isfinite(q).all().item(), 'Nonfinite Q')


def seraph_paths(output):
    """Enforce data-backed runtime output/cache paths on SERAPH, before HF imports."""
    data=Path('/data/khuss')
    if data.exists():
        require(Path(output).resolve().is_relative_to(data), 'SERAPH output must be under /data/khuss')
    hf_home=Path(os.environ.get('HF_HOME','/data/khuss/huggingface'))
    hf_hub=os.environ.get('HF_HUB_CACHE',os.environ.get('HUGGINGFACE_HUB_CACHE',str(hf_home/'hub')))
    defaults = dict(HF_HOME=str(hf_home),HF_HUB_CACHE=hf_hub,
        HF_ASSETS_CACHE=str(hf_home/'assets'),HF_DATASETS_CACHE=str(hf_home/'datasets'),
        HF_MODULES_CACHE=str(hf_home/'modules'),TRANSFORMERS_CACHE=hf_hub,
        TORCH_HOME='/data/khuss/.cache/torch',XDG_CACHE_HOME='/data/khuss/.cache',
        TORCH_EXTENSIONS_DIR='/data/khuss/.cache/torch_extensions',TRITON_CACHE_DIR='/data/khuss/.cache/triton',
        CUDA_CACHE_PATH='/data/khuss/.cache/cuda',TMPDIR='/data/khuss/.cache/tmp')
    for key,value in defaults.items():
        os.environ.setdefault(key,value)
        require(not Path(os.environ[key]).resolve().is_relative_to(Path('/home/khuss')), 'Forbidden runtime cache: '+key)
        if data.exists():require(Path(os.environ[key]).resolve().is_relative_to(data), 'SERAPH cache must be under /data/khuss: '+key)
    for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[key]='1'
    if data.exists():Path(os.environ['TMPDIR']).mkdir(parents=True,exist_ok=True)


def capture_run(args):
    require(args.seed==42 and args.device.startswith('cuda:'), 'B0 real FP16 capture requires one selected CUDA device and seed42')
    root=args.output_root
    require(not root.exists() or not any(root.iterdir()), 'New/empty B0 output root required')
    rows,pool,provenance=load_provenance(args)
    seraph_paths(root)
    tokenizer,token_metadata=b3.b.tokenizer_only()
    annotated=b3.selection_v2.annotate(rows,tokenizer)
    annotated=[dict(r,special_token_ids=sorted(set(r.get('special_token_ids',[]))|set(tokenizer.all_special_ids))) for r in annotated]
    cohort=build_cohort(annotated,pool,provenance,args.seed)
    # Persist selection/split BEFORE model loading or observing any Q statistics.
    write(root/'cohort.json',cohort)
    cohort_sha=sha(root/'cohort.json')
    from .c6b_runtime import load_base
    from semcache.models.task_adapters import load_two_task_users,activate_task_user
    from semcache.models.model_adapter import OPTModelAdapter
    import torch
    model,loaded_tokenizer,metadata=load_base(args)
    require(model.config.num_hidden_layers==32 and model.config.hidden_size==2560, 'Expected OPT2.7B architecture')
    model=load_two_task_users(model,args.adapter_root/'user_a',args.adapter_root/'user_b')
    adapter=OPTModelAdapter(model);by_id={r['source_id']:r for r in annotated};corpus={}
    with torch.inference_mode():
        for index,block in enumerate(cohort['blocks']):
            row=by_id[block['source_id']]
            ids=loaded_tokenizer(row['prompt_text'],add_special_tokens=True,truncation=False)['input_ids']
            require(ids==row['token_ids'] and len(ids)<=384, 'Capture prompt IDs/bound changed')
            activate_task_user(model,block['user'])
            with total_q_capture(adapter,block['w3_start']) as captured:
                output=model(input_ids=torch.tensor([ids],device=args.device),use_cache=False)
                del output
            corpus[block['calibration_id']]=torch.cat([captured[l] for l in range(32)],dim=0)
            print(f'Captured TOTAL Q {index+1}/128',flush=True)
    verify_files(provenance['input_hashes'])
    require(sha(root/'cohort.json')==cohort_sha, 'Pre-capture cohort/split changed during capture')
    raw=sum(q.numel()*q.element_size() for q in corpus.values())
    capture=root/'capture';capture.mkdir(exist_ok=False)
    torch.save(dict(schema='c7b_total_q_v1',q_object=Q_OBJECT,cohort_sha256=cohort_sha,blocks=corpus),capture/'q_blocks.pt')
    write(capture/'capture_manifest.json',dict(stage='C7-B0',status='COMPLETE',**provenance,
        cohort_sha256=cohort_sha,q_object=Q_OBJECT,layers=32,hidden_dim=2560,dtype='float16',blocks=128,
        raw_q_bytes=raw,**validate_cohort(cohort),model_inference_performed=True,training_performed=False,
        layer_block_shape=[1,3,2560],stored_block_shape=[32,3,2560],
        tokenizer=token_metadata,model_tokenizer_provenance=metadata,single_gpu=True,
        git=git(),output_hashes={'q_blocks.pt':sha(capture/'q_blocks.pt'),'../cohort.json':sha(root/'cohort.json')}))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter-root',type=Path,default=Path('results/cachegen/c6b3/b1_full'))
    p.add_argument('--plan-dir',type=Path,default=Path('results/cachegen/c6b3/multiwoz_plan_v2'))
    p.add_argument('--source',type=Path,default=Path('results/workloads/multiwoz.jsonl'))
    p.add_argument('--semantic',type=Path,default=Path('results/workloads/c6b3_multiwoz_history_semantic.jsonl'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output-root',type=Path,default=Path(DEFAULT_ROOT))
    args=p.parse_args(argv);capture_run(args)
