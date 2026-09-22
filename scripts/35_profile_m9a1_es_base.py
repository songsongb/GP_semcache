#!/usr/bin/env python3
"""Fresh M8.5 length-32 fixture + bare-OPT ES latency profiles (local models only)."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from semcache.system_cost.common import MODELS, write_json, file_sha256
from semcache.system_cost.profiles import load_rows, pair_rows
from semcache.system_cost.es_base_profile import attach_profile, validate_fresh_m85


def fresh_m8_command(args, output):
    cmd = [sys.executable, str(ROOT/'scripts/30_run_m8_inference_timing.py'),
        '--model', args.model, '--device', args.device, '--dtype', args.dtype,
        '--prompt-lengths', '32', '--warmup-runs', str(args.warmup_runs),
        '--measured-runs', str(args.measured_runs), '--impact-reducer', 'paper_row_l2_sum',
        '--cluster-update-mode', 'buffered', '--cluster-update-interval', '100',
        '--rho', '0.8', '--history-lambda', '100', '--pbr-mode', 'interval', '--pbr-interval', '100',
        '--output-dir', str(output)]
    if args.revision:
        cmd.extend(['--revision', args.revision])
    return cmd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, default='facebook/opt-2.7b')
    p.add_argument('--dtype', choices=('float16', 'float32'), default='float16')
    p.add_argument('--device', default='cuda')
    p.add_argument('--prompt-length', type=int, choices=(32,), default=32)
    p.add_argument('--warmup-runs', type=int, default=3)
    p.add_argument('--measured-runs', type=int, default=10)
    p.add_argument('--revision', help='Optional local revision; resolved fresh M8 commit is used for base reload')
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/m9a1/opt27b_len32')
    args = p.parse_args(argv)
    if args.warmup_runs < 1 or args.measured_runs < 1:
        p.error('Use positive warmup and measured counts')
    if args.model == 'facebook/opt-2.7b' and args.dtype != 'float16':
        p.error('Primary 2.7B profile requires float16')
    if args.output_dir.exists():
        p.error('Choose a new output directory: fresh profiles must not reuse old sweep artifacts')
    args.output_dir.mkdir(parents=True)
    fresh_id = 'm9a1-' + uuid.uuid4().hex
    fresh_dir = args.output_dir/'fresh_m85'
    cmd = fresh_m8_command(args, fresh_dir)
    write_json(args.output_dir/'profile_manifest.json', dict(fresh_m85_profile_id=fresh_id,
        state='STARTED', started_at=datetime.now(timezone.utc).isoformat(), command=cmd,
        personalized_output_quality_source='fresh M8.5 only', base_logits_quality_evidence=False))
    # Separate process releases all PEFT/encoder GPU state before loading bare OPT.
    subprocess.run(cmd, check=True)
    path, env_path = fresh_dir/'inference_raw.jsonl', fresh_dir/'inference_environment.json'
    rows, env = load_rows(path), json.loads(env_path.read_text())
    pairs = sorted(pair_rows(rows, args.model), key=lambda pair: pair[0]['repeat_index'])
    for native, lookup, physical in pairs:
        for row in (native, lookup, physical):
            validate_fresh_m85(row)
    from semcache.models.loader import load_model
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.system_cost.base_runtime import profile_pairs
    from semcache.utils.seed import seed_everything
    seed_everything(42)
    first = pairs[0][0]
    model, tokenizer, metadata = load_model(dict(name=args.model, tokenizer=args.model,
        revision=first['model_revision'], tokenizer_revision=first['tokenizer_revision'],
        dtype=args.dtype, device=args.device, attention_implementation='eager', local_files_only=True))
    hashes = dict(m8=file_sha256(path), environment=file_sha256(env_path))
    profiles = profile_pairs(model, tokenizer, OPTModelAdapter(model), metadata, pairs, env, fresh_id, hashes)
    combined = []
    for native, lookup, physical in pairs:
        index = native['repeat_index']
        combined.extend([attach_profile(native, profiles[(index, 'ES_BASE_NATIVE')], fresh_id),
            lookup, attach_profile(physical, profiles[(index, 'ES_BASE_SEMCACHE_REUSE')], fresh_id)])
    output = args.output_dir/'strict_es_input.jsonl'
    with output.open('w') as f:
        for row in combined:
            f.write(json.dumps(row, allow_nan=False)+'\n')
    write_json(args.output_dir/'es_base_profiles.json', dict(measurement_label='MEASURED_ES_BASE_ONLY',
        profiles=list(profiles.values()), personalized_output_quality_claimed=False))
    write_json(args.output_dir/'profile_manifest.json', dict(fresh_m85_profile_id=fresh_id,
        state='COMPLETE', completed_at=datetime.now(timezone.utc).isoformat(), command=cmd,
        source_hashes=hashes, strict_input_sha256=file_sha256(output), model=metadata,
        base_lora_absent=True, personalized_output_quality_source='fresh M8.5 only',
        base_logits_quality_evidence=False))
    print(f'Saved strict base-only profiles to {output}; no personalized output claim from base logits')


if __name__ == '__main__':
    main()
