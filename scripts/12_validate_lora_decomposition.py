from _lora_common import setup, save
from semcache.evaluation.lora_probe import capture_user, decomposition_rows, assert_return_identity
from semcache.models.lora_fixtures import base_weight_fingerprint
from semcache.models.model_adapter import OPTModelAdapter


def main():
    args, config, model, tokenizer, metadata, fixtures, layers = setup('Validate PEFT Q/K/V = base + nonzero LoRA')
    ids = tokenizer(config['inspection']['text'])['input_ids']
    before = base_weight_fingerprint(OPTModelAdapter(model))
    first, logits = capture_user(model, ids, 'user_a', layers)
    rows = decomposition_rows(model, first, metadata, fixtures['user_a'], args.tolerance)
    capture_user(model, ids, 'user_b', layers)
    returned, logits_returned = capture_user(model, ids, 'user_a', layers)
    assert_return_identity(first, returned, logits, logits_returned)
    if before != base_weight_fingerprint(OPTModelAdapter(model)):
        raise AssertionError('Base changed after adapter switching')
    for r in rows:
        print(f"layer={r['layer']} tensor={r['tensor_type']} base_norm={r['base_norm']:.8g} "
              f"lora_norm={r['lora_delta_norm']:.8g} ratio={r['lora_to_base_norm_ratio']:.8g} "
              f"max_abs_error={r['decomposition_max_abs_error']:.8g} "
              f"relative_l2={r['decomposition_relative_l2']:.8g} cosine={r['decomposition_cosine_similarity']:.12g}", flush=True)
    save(args, config, metadata, fixtures, rows, 'lora_decomposition_probe', model, layers)


if __name__ == '__main__':
    main()
