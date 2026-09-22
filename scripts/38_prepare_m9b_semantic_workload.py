#!/usr/bin/env python3
"""Prepare M9-B semantic JSONL with local pinned OPT tokenizer and actual TinyBERT."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from semcache.experiments.m9b_semantic_workload import (
    CLUSTERS, CURRENT_PREPARED_COUNTS, MODEL_ID, MODEL_REVISION, ENCODER_REVISION,
    prepare, read_raw, validate_generated,
)


def local_components(device, seed):
    # Hard offline boundary, even if a caller has not configured HF offline mode.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file
    from semcache.models.tokenizer_provenance import resolve_tokenizer_provenance
    from semcache.semantic.encoder import TinyBERTSemanticEncoder
    from semcache.utils.seed import seed_everything
    seed_everything(seed)
    # Resolve an actual tokenizer asset in the pinned local snapshot; no model load.
    asset = cached_file(MODEL_ID, 'tokenizer_config.json', revision=MODEL_REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(Path(asset).parent), local_files_only=True)
    token_metadata = resolve_tokenizer_provenance(tokenizer,
        model_source_id=MODEL_ID, tokenizer_source_id=MODEL_ID,
        model_requested_revision=MODEL_REVISION, tokenizer_requested_revision=MODEL_REVISION,
        model_resolved_revision=None, model_revision_explicit=True, tokenizer_revision_explicit=True)
    encoder = TinyBERTSemanticEncoder(revision=ENCODER_REVISION, pooling='masked_mean',
        max_length=512, device=device, dtype='float32', local_files_only=True)
    metadata = dict(tokenizer=token_metadata, semantic_encoder=encoder.metadata,
                    execution_provenance='MEASURED')
    environment = dict(hostname=platform.node(), python=platform.python_version(), device=device,
        packages={name: importlib.metadata.version(name) for name in ('torch','transformers','huggingface-hub')},
        local_files_only=True, downloads_performed=False, deterministic_algorithms=True,
        semantic_model_execution=True, opt_model_execution=False)
    return tokenizer, encoder, metadata, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=CLUSTERS, required=True)
    parser.add_argument('--source', type=Path, help='Default results/workloads/DATASET.jsonl')
    parser.add_argument('--output', type=Path, help='Default results/workloads/m9b_DATASET_semantic.jsonl')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--expect-rows', type=int, help='Optional check for a particular prepared artifact')
    parser.add_argument('--check-current-prepared-count', action='store_true',
                        help='Check this prepared SNIPS=13784 or MultiWOZ=56776 artifact only')
    parser.add_argument('--compare-manifest', type=Path, help='Require identical source/order/token/assignment/output hashes')
    parser.add_argument('--validate-only', action='store_true', help='Validate existing output+sidecar without model imports')
    args = parser.parse_args()
    source = args.source or ROOT/f'results/workloads/{args.dataset}.jsonl'
    output = args.output or ROOT/f'results/workloads/m9b_{args.dataset}_semantic.jsonl'
    try:
        if args.expect_rows is not None and args.check_current_prepared_count:
            raise ValueError('Choose one row-count check')
        expected = CURRENT_PREPARED_COUNTS[args.dataset] if args.check_current_prepared_count else args.expect_rows
        raw = read_raw(source, args.dataset)
        if expected is not None and len(raw) != expected:
            raise ValueError(f'Expected {expected} rows; found {len(raw)}')
        previous = json.loads(args.compare_manifest.read_text()) if args.compare_manifest else None
        if args.validate_only:
            manifest = json.loads(Path(str(output)+'.manifest.json').read_text())
            if manifest['dataset'] != args.dataset:
                raise ValueError('Manifest dataset mismatch')
            result = validate_generated(source, output, manifest, previous_manifest=previous)
            print(json.dumps(result, sort_keys=True))
            return
        if source.resolve() == output.resolve() or output.exists() or Path(str(output)+'.manifest.json').exists():
            raise ValueError('Output must be new and distinct from raw source; use --validate-only for existing output')
        if args.batch_size < 1:
            raise ValueError('batch-size must be positive')
        tokenizer, encoder, metadata, environment = local_components(args.device, args.seed)
        result = prepare(source, output, args.dataset, tokenizer, encoder, metadata,
            batch_size=args.batch_size, seed=args.seed, expected_rows=expected,
            previous_manifest=previous, environment=environment)
    except (ValueError, KeyError, TypeError, OSError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Saved {result['row_count']} actual TinyBERT semantic rows to {output}; raw source unchanged")


if __name__ == '__main__':
    main()
