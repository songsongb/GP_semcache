from _common import arguments, ROOT
from semcache.models.loader import load_model
from semcache.evaluation.output_impact import run_output_impact
from semcache.utils.seed import seed_everything
from semcache.utils.io import write_csv, write_json


def main(control=False):
    def options(parser):
        parser.add_argument('--all-layers', action='store_true')
        parser.add_argument('--modes', nargs='+', choices=['q', 'k', 'v', 'qkv'], default=['q', 'k', 'v', 'qkv'])
        parser.add_argument('--cases', nargs='+', choices=['A', 'B'] if control else list('ABCD'),
                            default=['A', 'B'] if control else list('ABCD'))
        parser.add_argument('--tolerance', type=float, default=1e-6)
    args, config = arguments('Controlled cached-QKV injection into OPT (development only)', options)
    if args.all_layers and args.layers is not None:
        raise ValueError('Choose --layers or --all-layers')
    seed_everything(config['seed'])
    model, tokenizer, metadata = load_model(config['model'])
    layers = None if args.all_layers else config.get('probe', {}).get('layers', [0, 1, 5, 11])
    print('Tolerance is an implementation-control check, NOT a reuse-quality threshold.', flush=True)
    def display(row):
        print(f"{row['probe_case']} layer={row['layer']} mode={row['injection_mode']} "
              f"max_abs={row['max_abs_logit_diff']:.12g} prefix={row['prefix_max_abs_logit_diff']:.12g} "
              f"suffix_KL(baseline||injected)={row['affected_suffix_mean_kl']:.12g} "
              f"cache_hit={row['cache_hit']} valid_candidate={row['valid_reuse_candidate']}", flush=True)
    rows = run_output_impact(model, tokenizer, metadata, config['subsequence_window'], layers,
                             args.modes, args.cases, config['seed'], args.tolerance, display)
    output = args.output or ROOT / ('results/raw/qkv_injection_control.csv' if control else 'results/raw/qkv_reuse_output_impact.csv')
    write_csv(output, rows)
    write_json(output.with_suffix('.json'), dict(metadata=metadata, config=config, tolerance=args.tolerance,
               scope='base_raw_unscaled_linear_projection', metric_source='measured', safe_reuse_claimed=False,
               kl_direction='baseline || injected', metric_precision='CPU float64', rows=rows))
    print(f'Saved {len(rows)} measurements to {output}. Safe reuse is NOT claimed.')
