"""C0 collection preparation; GPU jobs only read verified local inputs."""
import json
from pathlib import Path

from .communication import digest, digest_bytes, require
from .r3a_artifacts import append, read, write
from .r4b2_workload import tokenizer_configuration
from .r4c_dataset import DATASETS, load_dataset, select
from .r4c_prompts import FixtureTokenizer, TokenizerAdapter, prepare, validate_prepared, TEMPLATES
from .r4c_workload import natural_discovery, schedule
from .r4c_identity import sources


def opt_tokenizer(pm,tokenizer=None):
    from transformers import AutoConfig, AutoTokenizer
    i=pm['identity']; tok=tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(i['tokenizer_source'],revision=i['tokenizer_revision'],local_files_only=True)
    tok.truncation_side='right'
    config=AutoConfig.from_pretrained(i['model'],revision=i['base_revision'],local_files_only=True)
    require(config.model_type=='opt','C1 backend supports only OPT; Qwen requires C2 baseline validation')
    identity=dict(model=i['model'],model_revision=i['base_revision'],tokenizer_source=i['tokenizer_source'],tokenizer_revision=i['tokenizer_revision'],
                  vocab_sha256=digest(tok.get_vocab()),configuration=tokenizer_configuration(tok),model_config_sha256=digest(config.to_dict()),
                  raw_cache_bytes_per_token=3*int(config.num_hidden_layers)*int(config.hidden_size)*{'float16':2,'float32':4}[i['dtype']])
    return TokenizerAdapter(tok,identity,int(config.max_position_embeddings))


def prepare_collection(registry, tokenizer, output, *, datasets=DATASETS, count=2, seed=42, cohort='development', excluded_ids=(),
                       input_budget=256, qa_budget=32, summary_budget=64, prefix_length=128, policy_mode='quality',
                       strict_count=1, enable_512=False, fixture=False,min_input_tokens=None):
    output=Path(output); prompts=[]; manifests=[]
    require(len(set(datasets))==len(datasets) and 0<len(datasets)<=3,'invalid dataset selection')
    for name in datasets:
        require(name in registry, f'MISSING_DATASET: registry requires {name}; expected local {name}.jsonl or HF saved snapshot')
        rows, manifest=load_dataset(name,registry[name],fixture=fixture)
        before,after=TEMPLATES[name]
        lengths={r['original']['_id']:len(tokenizer.encode(before+r['original']['context']+after.format(question=r['original']['input']))) for r in rows}
        selected, selection=select(rows,count,seed,tokenizer=tokenizer,min_context_tokens=registry[name].get('min_context_tokens',0),
            cohort=cohort,excluded_ids=excluded_ids,input_lengths=lengths,min_input_tokens=prefix_length+1 if min_input_tokens is None else min_input_tokens)
        manifest['selection']=selection; manifests.append(manifest)
        generation_budget=registry[name].get('max_new_tokens',summary_budget if name=='multi_news' else qa_budget)
        manifest['generation_budget']=generation_budget
        for row in selected:
            prompt=prepare(row,tokenizer,input_budget=input_budget,max_new_tokens=generation_budget,enable_512=enable_512)
            require(len(prompt['token_ids'])>prefix_length,'INELIGIBLE_PROMPT: selected real record shorter than prefix; no silent replacement')
            prompts.append(prompt)
    special={k:getattr(tokenizer.tokenizer,k) for k in ('bos_token_id','eos_token_id','pad_token_id')}
    tasks=schedule(prompts,special,prefix_length=prefix_length,mode=policy_mode,strict_count=strict_count)
    manifest=dict(schema='R4C_DATASET_MANIFEST_V1',datasets=manifests,model_tokenizer=tokenizer.identity,context_limit=tokenizer.supported_context_length,
                  cohort=cohort,seed=seed,excluded_ids=list(excluded_ids),TEST_FIXTURE_ONLY=fixture,evaluation_label='LONGBENCH_DERIVED_TRUNCATED_EVALUATION',
                  special_tokens=special,input_budget=input_budget,prefix_length=prefix_length,qa_budget=qa_budget,summary_budget=summary_budget,
                  policy_mode=policy_mode,strict_count=strict_count,records=len(prompts),gold_truncated=False,preparation_sources=sources(),
                  generation_budgets={m['dataset']:m['generation_budget'] for m in manifests})
    for row in prompts:append(output/'r4c_prepared_prompts.jsonl',row)
    coverage=natural_discovery(prompts,raw_cache_bytes_per_token=tokenizer.identity.get('raw_cache_bytes_per_token'))
    unique={tuple(p['token_ids'][:prefix_length]) for p in prompts}
    coverage['controlled_proactive_sources']=dict(unique_token_prefixes=len(unique),prefix_length=prefix_length,
        retain_all_raw_qkv_bytes=None if 'raw_cache_bytes_per_token' not in tokenizer.identity else len(unique)*prefix_length*tokenizer.identity['raw_cache_bytes_per_token'],
        natural_hit_rate_inferred=False)
    write(output/'r4c_dataset_manifest.json',manifest);write(output/'r4c_prefix_coverage.json',coverage)
    write(output/'r4c_schedule.json',tasks)
    write(output/'r4c_prepared_manifest.json',dict(schema='R4C_PREPARED_V1',TEST_FIXTURE_ONLY=fixture,
        hashes={name:digest_bytes((output/name).read_bytes()) for name in ('r4c_dataset_manifest.json','r4c_prepared_prompts.jsonl','r4c_prefix_coverage.json','r4c_schedule.json')}))
    return manifest,prompts,tasks


def load_prepared(path, *, verify_datasets=True):
    path=Path(path); seal=read(path/'r4c_prepared_manifest.json')
    require(seal['schema']=='R4C_PREPARED_V1','unsupported C0 preparation')
    for name,h in seal['hashes'].items():
        p=(path/name).resolve();require(p.is_relative_to(path.resolve()) and digest_bytes(p.read_bytes())==h,'C0 prepared file changed')
    manifest=read(path/'r4c_dataset_manifest.json'); rows=[validate_prepared(json.loads(line)) for line in (path/'r4c_prepared_prompts.jsonl').read_text().splitlines()]
    from .r4a_evidence import verify_sources
    verify_sources(manifest['preparation_sources'])
    require(len(rows)==manifest['records'] and len({r['sample_id'] for r in rows})==len(rows),'prepared cohort size/IDs differ')
    tasks=read(path/'r4c_schedule.json')
    require(tasks==schedule(rows,manifest['special_tokens'],prefix_length=manifest['prefix_length'],mode=manifest['policy_mode'],strict_count=manifest['strict_count']), 'prepared schedule differs from cohort/policy')
    if verify_datasets:
        for source in manifest['datasets']:
            load_dataset(source['dataset'],dict(path=source['path'],format=source['format'],revision=source['revision'],sha256=source['files']),fixture=manifest['TEST_FIXTURE_ONLY'])
    return manifest,rows,tasks


class TinyTokenizer:
    bos_token_id=0; eos_token_id=6; pad_token_id=0
    def encode(self,text,**kwargs):return [1+ord(c)%5 for c in text]
    def decode(self,ids,**kwargs):return ''.join(chr(97+t) for t in ids)


def fixture_inputs(output, *, qa_budget=5, summary_budget=7, count=1):
    """Synthetic records can only be reached from --test-fixture-only."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True); registry={}
    for name in DATASETS:
        p=output/(name+'.jsonl')
        for j in range(count):
            append(p,dict(_id=f'SYNTHETIC_{name}_{j}',input='Where is the fixture?',context=('Synthetic article '+name+' '+str(j)+'. ')*80,
                answers=['fixture_token_3','fixture_token_4'],length=2000,TEST_FIXTURE_ONLY=True))
        registry[name]=dict(path=str(p.resolve()),revision='TEST_FIXTURE_ONLY_v1',format='jsonl',sha256={p.name:digest_bytes(p.read_bytes())})
    tok=TokenizerAdapter(TinyTokenizer(),dict(scope='TEST_FIXTURE_ONLY_MOD5_CHAR_TOKENIZER'),2048)
    return registry,tok
