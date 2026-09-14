from _common import arguments, ROOT
from semcache.models.loader import load_model
from semcache.models.lora_fixtures import create_controlled_users, base_weight_fingerprint
from semcache.models.model_adapter import OPTModelAdapter
from semcache.utils.seed import seed_everything
from semcache.utils.io import write_csv, write_json


def setup(description, all_layers=False):
    def options(parser):
        parser.set_defaults(config=ROOT / 'configs/development.yaml')
        parser.add_argument('--tolerance', type=float, default=1e-6)
    args, config = arguments(description, options)
    import math
    import importlib.util
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    if importlib.util.find_spec('peft') is None:
        raise SystemExit("Missing PEFT. Install: python -m pip install -e '.[lora,test]'")
    seed_everything(config['seed'])
    model, tokenizer, metadata = load_model(config['model'])
    model, fixtures = create_controlled_users(model, config.get('lora'))
    import peft
    metadata['peft_version'] = peft.__version__
    layers = config.get('probe', {}).get('layers', None if all_layers else [0, 1, 5, 11])
    if layers is None:
        layers = list(range(len(OPTModelAdapter(model).layers)))
    for layer in layers:
        OPTModelAdapter(model).projection_modules(layer)
    print('Controlled non-trained adapters; single-process ES/UD emulation. No safe reuse or quality claim.', flush=True)
    return args, config, model, tokenizer, metadata, fixtures, layers


def save(args, config, metadata, fixtures, rows, filename, model, layers):
    output = args.output or ROOT / ('results/raw/'+filename+'.csv')
    write_csv(output, rows)
    write_json(output.with_suffix('.json'), dict(metadata=metadata, config=config, fixtures=fixtures,
        layers=layers, tolerance=args.tolerance, metric_source='measured', safe_reuse_claimed=False,
        base_fingerprint=base_weight_fingerprint(OPTModelAdapter(model)), rows=rows))
    print(f'Saved {len(rows)} measurements to {output}', flush=True)
