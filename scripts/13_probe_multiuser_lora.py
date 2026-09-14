from _lora_common import setup, save
from semcache.evaluation.lora_probe import run_multiuser_probe


def main():
    args, config, model, tokenizer, metadata, fixtures, layers = setup('Two-user LoRA isolation and observational context probe')
    rows = run_multiuser_probe(model, tokenizer, metadata, fixtures, layers, config['subsequence_window'], args.tolerance)
    for r in rows:
        if r['adapter_name'] == 'user_a':
            print(f"{r['probe_mode']} case={r['probe_case']} layer={r['layer']} tensor={r['tensor_type']} "
                  f"component={r['component']} cosine={r['cross_user_cosine_similarity']:.8g} "
                  f"relative_l2={r['cross_user_relative_l2']:.8g}", flush=True)
    save(args, config, metadata, fixtures, rows, 'multiuser_lora_probe', model, layers)


if __name__ == '__main__':
    main()
