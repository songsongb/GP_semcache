from _lora_common import setup, save
from semcache.evaluation.lora_probe import validate_forward, provenance


def main():
    args, config, model, tokenizer, metadata, fixtures, layers = setup('EdgeLoRA reconstructed QKV full-forward parity', all_layers=True)
    ids = tokenizer(config['inspection']['text'])['input_ids']
    row = dict(**provenance(metadata, fixtures['user_a']))
    row.update(validate_forward(model, ids, layers=layers, tolerance=args.tolerance))
    row['layers'] = layers
    print(f"max_abs_logit_diff={row['max_abs_logit_diff']:.12g} relative_l2={row['relative_l2_logit_diff']:.12g} "
          f"KL={row['affected_suffix_mean_kl']:.12g} argmax_agreement={row['last_argmax_agreement']}", flush=True)
    save(args, config, metadata, fixtures, [row], 'edgelora_projection_parity', model, layers)


if __name__ == '__main__':
    main()
