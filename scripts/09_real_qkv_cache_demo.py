from _common import arguments, ROOT
from semcache.models.loader import load_model
from semcache.probe import run_probe
from semcache.utils.io import write_json, write_csv
from semcache.utils.seed import seed_everything

args, config = arguments('Physical base OPT cache MISS -> INSERT -> HIT; no attention substitution')
seed_everything(config['seed'])
model, tokenizer, metadata = load_model(config['model'])
rows, report = run_probe(model, tokenizer, metadata, config['subsequence_window'], config.get('probe', {}).get('layers'), physical=True)
output = args.output or ROOT / 'results/raw/physical_qkv_cache_demo.csv'
write_csv(output, rows)
write_json(output.with_suffix('.json'), report)
print('MISS -> INSERT -> HIT: physical cache hit confirmed')
print('Clustering uses controlled fixture vectors. Safe inference reuse is NOT YET CLAIMED.')
