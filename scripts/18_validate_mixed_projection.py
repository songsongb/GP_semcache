from _semcache_common import setup
from _common import ROOT
from semcache.evaluation.mixed_control import validate_mixed_control
from semcache.utils.io import write_csv, write_json

if __name__ == '__main__':
    args, config, model, tokenizer, metadata, fixtures, adapter = setup()
    row = validate_mixed_control(model, tokenizer, adapter)
    output = args.output or ROOT/'results/raw/mixed_projection_validation.csv'
    write_csv(output, [row])
    write_json(output.with_suffix('.json'), dict(metadata=metadata, config=config, fixtures=fixtures, control=row))
    print(row)
