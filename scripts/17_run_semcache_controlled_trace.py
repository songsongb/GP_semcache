from _semcache_common import setup, engine_for, TRACE
from _common import ROOT
from semcache.evaluation.mixed_control import validate_mixed_control
from semcache.utils.io import write_csv, write_json, write_jsonl

if __name__ == '__main__':
    args, config, model, tokenizer, metadata, fixtures, adapter = setup()
    control = validate_mixed_control(model, tokenizer, adapter)  # STOP on failure before approximate reuse.
    print('Exact-control passed; starting approximate cross-user trace.', flush=True)
    engine = engine_for(config, model, tokenizer, metadata, adapter, args.logical_capacity_bytes)
    rows = []
    for query_id, user, text, _ in TRACE:
        result = engine.query(text, user, query_id, args.compare_baseline)
        rows.append(result['summary'])
        row = rows[-1]
        print(f"{query_id}: cluster={row['cluster_id']} hits={row['block_hit_count']} "
              f"reused={row['reused_unique_token_count']} fresh={row['recomputed_tokens']} "
              f"max_abs={row.get('max_abs_logit_diff')}", flush=True)
    engine.pbr(0)
    engine.pbr(1)
    types = {e['event_type'] for e in engine.events}
    if not {'MISS', 'ADMIT', 'INSERT', 'HIT', 'FETCH', 'MIXED_PROJECT', 'CHU'} <= types:
        raise AssertionError('Controlled trace did not complete required physical reuse transitions')
    output = args.output or ROOT/'results/raw/semcache_controlled_trace.csv'
    write_csv(output, rows)
    write_jsonl(output.parent/'semcache_events.jsonl', engine.events)
    write_json(output.with_suffix('.json'), dict(metadata=metadata, config=config, fixtures=fixtures,
        exact_control=control, rows=rows, safe_reuse_claimed=False,
        decode_scope=engine.lookup_latest_token(0, 1, 1)))
    print(f'Saved {output}; safe_reuse_claimed=false')
