#!/usr/bin/env python3
"""CPU-only post-hoc transport metadata amortization; no model/codec imports."""
import argparse
import csv
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

MODES = ('TRANSPORT_QKV_COMP', 'FULL_PIPELINE')
FACTORS = (1, 2, 4, 8, 16, 32, 'inf')
FIELDS = ('raw_total_delta_bytes', 'compressed_total_delta_bytes', 'per_packet_cdf_bytes',
          'transmitted_total_including_cdf_bytes', 'transport_compression_ratio',
          'transport_byte_reduction_percentage')
ROOT = Path(__file__).resolve().parents[1]
POLICY = dict(total='P + C/N; infinity: P', ratio='R / total',
              reduction_percentage='100 * (1 - total/R)',
              positive_saving='total < R; when R > P, N > C/(R-P)',
              minimum_integer='max(1, floor(C/(R-P)) + 1), verified exactly',
              arithmetic='exact rational arithmetic from JSON decimal values; 50 significant decimal digits for nonterminating output',
              replay_tolerance='total bytes: absolute 0.000001; ratio/percentage: max(absolute 1e-10, relative 1e-12)')


def decimal(value):
    if isinstance(value, Fraction):
        with localcontext() as ctx:
            ctx.prec = 50
            return Decimal(value.numerator) / Decimal(value.denominator)
    return value


def json_text(value):
    """Emit JSON numeric decimals without converting through binary floats."""
    value = decimal(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError('Nonfinite JSON output')
        return str(value)
    if isinstance(value, dict):
        return '{' + ',\n'.join(json.dumps(k) + ': ' + json_text(v) for k, v in value.items()) + '}'
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(json_text(v) for v in value) + ']'
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f'{name} must be a finite JSON number')
    n = Decimal(str(value))
    if not n.is_finite():
        raise ValueError(f'{name} must be finite')
    return Fraction(n)


def close(actual, expected, byte_count=False):
    tolerance = Fraction(1, 1000000) if byte_count else max(Fraction(1, 10**10), abs(expected)/10**12)
    return abs(actual - expected) <= tolerance


def metrics(r, total):
    return dict(analytical_total_bytes=total, compression_ratio=r/total,
                byte_reduction_percentage=100*(1-total/r), net_saving=total < r)


def analyze(summary):
    if summary.get('stage') != 'C6-B3-2':
        raise ValueError('Input stage must be C6-B3-2')
    accounting = summary.get('transport_accounting', {})
    rows, breaks = [], []
    for mode in MODES:
        if mode not in accounting:
            raise ValueError('Missing required mode: ' + mode)
        measured = accounting[mode]
        missing = set(FIELDS)-set(measured)
        if missing:
            raise ValueError(f'{mode}: missing required fields: {sorted(missing)}')
        values = {k: number(measured[k], mode+'.'+k) for k in FIELDS}
        r, p, c = (values[k] for k in FIELDS[:3])
        if r <= 0 or p <= 0 or c < 0:
            raise ValueError(f'{mode}: require R > 0, P > 0 and C >= 0')
        replay = metrics(r, p+c)
        for key, computed, byte_count in (
            ('transmitted_total_including_cdf_bytes', p+c, True),
            ('transport_compression_ratio', replay['compression_ratio'], False),
            ('transport_byte_reduction_percentage', replay['byte_reduction_percentage'], False)):
            if not close(computed, values[key], byte_count):
                raise ValueError(f'{mode}: N=1 reproduction failed for {key}')
        for n in FACTORS:
            amortized = Fraction(0) if n == 'inf' else c/n
            provenance = ('MEASURED_ACCOUNTING_REPLAY' if n == 1 else
                          'ANALYTICAL_PRESHARED_CDF_UPPER_BOUND' if n == 'inf' else
                          'ANALYTICAL_CDF_AMORTIZATION')
            rows.append(dict(mode=mode, amortization_n=n, provenance=provenance,
                raw_total_delta_bytes=r, compressed_payload_bytes=p, original_cdf_bytes=c,
                amortized_cdf_bytes=amortized, **metrics(r, p+amortized)))
        threshold = c/(r-p) if r > p else None
        minimum = max(1, threshold.numerator//threshold.denominator+1) if threshold is not None else None
        total = p+c/minimum if minimum is not None else None
        if minimum is not None:
            if not total < r or (minimum > 1 and p+c/(minimum-1) < r):
                raise ValueError('Strict minimum-integer verification failed')
        breaks.append(dict(mode=mode, raw_total_delta_bytes=r, compressed_payload_bytes=p, original_cdf_bytes=c,
            payload_only_compression_ratio=r/p, payload_only_byte_reduction_percentage=100*(1-p/r),
            payload_only_provenance='ANALYTICAL_PRESHARED_CDF_UPPER_BOUND',
            continuous_break_even_threshold_n=threshold,
            positive_saving_condition=f'positive saving requires N > {decimal(threshold)}' if threshold is not None else
                'No positive saving possible: compressed payload P >= raw bytes R, even with zero recurring CDF cost',
            minimum_integer_n_for_positive_saving=minimum, bytes_at_minimum_integer_n=total,
            bytes_saved_at_minimum_integer_n=r-total if total is not None else None,
            compression_ratio_at_minimum_integer_n=r/total if total is not None else None,
            byte_reduction_percentage_at_minimum_integer_n=100*(1-total/r) if total is not None else None,
            n1_reproduction_check='PASS'))
    return rows, breaks


def markdown(rows, breaks):
    text = ['# Transport metadata amortization accounting', '',
            '| Mode | N | Total bytes | Compression ratio | Byte reduction | Provenance |',
            '|---|---:|---:|---:|---:|---|']
    for r in rows:
        n = '∞' if r['amortization_n'] == 'inf' else str(r['amortization_n'])
        text.append(f"| {r['mode']} | {n} | {decimal(r['analytical_total_bytes']):.6f} | "
                    f"{decimal(r['compression_ratio']):.6f} | {decimal(r['byte_reduction_percentage']):.6f}% | {r['provenance']} |")
    text.extend(['', '## Strict positive-saving thresholds', ''])
    for result in breaks:
        threshold = result['continuous_break_even_threshold_n']
        if threshold is None:
            text.append(f"- {result['mode']}: {result['positive_saving_condition']}.")
        else:
            text.append(f"- {result['mode']}: positive saving requires N > {decimal(threshold):.6f} "
                        f"(display rounded); smallest positive integer N = {result['minimum_integer_n_for_positive_saving']}. "
                        f"Payload-only upper-bound reduction = {decimal(result['payload_only_byte_reduction_percentage']):.6f}%.")
    text.extend(['', '## Interpretation', '',
        '1. N=1 reproduces the measured C6-B3-2 accounting.',
        '2. N>1 values are analytical amortization scenarios only; they were not physically executed.',
        '3. N=∞ is a preshared/frozen-CDF payload-only upper bound.',
        '4. This analysis does NOT establish that reusing/fixing a CDF preserves the same reconstruction quality or compression rate.',
        '5. A future experiment is required to validate actual shared, session-level, per-layer, or frozen-CDF transport coding.', ''])
    return '\n'.join(text)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path('results/cachegen/c6b3/b2_frozen32/summary.json'))
    parser.add_argument('--output-root', type=Path, default=Path('results/cachegen/c6b3/b3_transport_accounting'))
    args = parser.parse_args(argv)
    if args.output_root.exists() and (not args.output_root.is_dir() or any(args.output_root.iterdir())):
        raise ValueError('Output root must be new or empty; overwrite is not supported')
    source_bytes = args.summary.read_bytes()
    summary = json.loads(source_bytes, parse_float=Decimal, parse_constant=Decimal, object_pairs_hook=unique_object)
    rows, breaks = analyze(summary)  # All validation precedes artifact creation.
    def git(*cmd):
        return subprocess.check_output(['git', *cmd], cwd=ROOT, text=True).strip()
    manifest = dict(stage='C6-B3-3', status='COMPLETE',
        analysis_type='post-hoc analytical transport CDF amortization',
        input_summary_path=str(args.summary.resolve()), input_summary_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_stage='C6-B3-2', analyzed_modes=list(MODES), amortization_factors=list(FACTORS),
        formulas=POLICY, n1_reproduction_checks={r['mode']:r['n1_reproduction_check'] for r in breaks},
        git=dict(branch=git('branch','--show-current'), commit=git('rev-parse','HEAD'), status=git('status','--porcelain')),
        measured_experiment_rerun=False, model_inference_performed=False, gpu_required=False,
        paper_claim_scope='analytical communication-accounting follow-up; no shared-CDF quality validation')
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, value in [('transport_amortization.json', rows), ('transport_break_even.json', breaks)]:
        with (args.output_root/name).open('x', encoding='utf8') as stream:
            stream.write(json_text(value)+'\n')
    with (args.output_root/'transport_amortization.csv').open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k:decimal(v) for k,v in row.items()} for row in rows)
    with (args.output_root/'summary.md').open('x', encoding='utf8') as stream:
        stream.write(markdown(rows, breaks))
    manifest['output_hashes'] = {name:file_hash(args.output_root/name) for name in
        ('transport_amortization.json','transport_amortization.csv','transport_break_even.json','summary.md')}
    with (args.output_root/'manifest.json').open('x', encoding='utf8') as stream:
        stream.write(json_text(manifest)+'\n')
    for result in breaks:
        print(f"{result['mode']}: N=1 replay PASS; {result['positive_saving_condition']}; "
              f"minimum integer N={result['minimum_integer_n_for_positive_saving']}; "
              f"payload-only upper-bound reduction={decimal(result['payload_only_byte_reduction_percentage']):.6f}%")


if __name__ == '__main__':
    main()
