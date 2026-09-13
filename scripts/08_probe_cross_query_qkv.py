from _common import arguments, ROOT
from semcache.models.loader import load_model
from semcache.probe import run_probe
from semcache.utils.io import write_json, write_csv
from semcache.utils.seed import seed_everything

args, config = arguments('Controlled base OPT Q/K/V probe; use --model-id facebook/opt-125m --dtype float32 for development')
seed_everything(config['seed'])
model, tokenizer, metadata = load_model(config['model'])
rows, report = run_probe(model, tokenizer, metadata, config['subsequence_window'], config.get('probe', {}).get('layers'))
output = args.output or ROOT / 'results/raw/qkv_cross_query_probe.csv'
write_csv(output, rows)
write_json(output.with_suffix('.json'), report)
print(f'Saved {len(rows)} per-layer measurements to {output}. Safe inference reuse is NOT YET CLAIMED.')
