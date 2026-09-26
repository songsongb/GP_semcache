"""B1 lossless symbol locality diagnostics; no entropy coder or profile fitting."""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .common import digest, environment, file_hash, load_fixture, read_json
from .harness import verify_capture
from .shared.device_contract import resolve_contract, verify_contract
from .shared.harness import existing_uniform, protect_output
from .shared.tensors import quantize, reconstruct

FAMILIES = ('RAW_GLOBAL', 'RAW_ROLE_SPLIT', 'ANCHOR_MOD_RESIDUAL')
ROLES = {'RAW_GLOBAL': ('all',), 'RAW_ROLE_SPLIT': ('anchor', 'nonanchor'),
         'ANCHOR_MOD_RESIDUAL': ('anchor', 'residual')}
SPLITS = ('calibration', 'evaluation')
DATASETS = ('snips', 'multiwoz')
LABELS = dict(provenance='MEASURED_RESEARCH_EXTENSION', inspiration='CACHEGEN_INSPIRED',
              experiment='B1_LOSSLESS_SYMBOL_TRANSFORM')
RULE = dict(provenance='REPRODUCTION_CHOICE', threshold_percent=3.0,
    metric='evaluation K+V weighted bits/symbol, with separate K/V probability models',
    comparison='ANCHOR_MOD_RESIDUAL versus RAW_ROLE_SPLIT',
    go='overall reduction >= 3% AND positive reduction in both SNIPS and MultiWOZ',
    statistical_significance_claim=False, threshold_tuned_on_evaluation=False)
CONFIG = dict(schema_version=1, token_group_size=10, alphabet=[-127, 127], modulus=255,
    input_domain='u=s+127 in [0,254]', anchor='u[:,0,:]',
    residual='r=(u_i-u_0) mod 255; d=r if r<=127 else r-255; i=1..9',
    inverse='r=d mod 255; u_i=(u_0+r) mod 255; s_i=u_i-127',
    histogram_index='original s+127 for anchors/raw; centered d+127 for residuals',
    arithmetic_dtype='wide Python integers; never int8 subtraction',
    families=list(FAMILIES), K_V_models='separate; combine code costs, never histograms',
    entropy='empirical Shannon, no smoothing, no CDF, no differential entropy',
    locality='float64 analysis of raw FP16: x_i-x_0; target-normalized relative L2; pooled cosine',
    decision_rule=RULE, no_arithmetic_coding=True, no_profile_fitting=True,
    no_new_lossy_quantization=True, official_cachegen_reproduction=False)


def symbol_to_u8_domain(x):
    if type(x) is not int or not -127 <= x <= 127:
        raise ValueError('Symbol must be an integer in [-127,127]')
    return x+127


def u8_domain_to_symbol(u):
    if type(u) is not int or not 0 <= u < 255:
        raise ValueError('Domain value must be an integer in [0,254]')
    return u-127


def validate_domain(data, shape):
    if (len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape)
            or shape[1] != 10 or len(data) != math.prod(shape)):
        raise ValueError('Expected [layers,10,channels]; T=3/padding is forbidden')
    if not isinstance(data, bytes) or 255 in data:
        raise ValueError('Expected bytes in the 255-symbol domain [0,254]')


def transform(data, shape, *, inverse=False):
    """Packed [L,T,C] domains. Anchors are untouched; residual slots hold d+127."""
    validate_domain(data, shape)
    layers, _, channels = shape
    out = bytearray(data)
    for layer in range(layers):
        start = layer*10*channels
        anchor = data[start:start+channels]
        for token in range(1, 10):
            offset = start+token*channels
            values = data[offset:offset+channels]
            if inverse:
                out[offset:offset+channels] = bytes((a+d-127) % 255 for a, d in zip(anchor, values))
            else:
                out[offset:offset+channels] = bytes((u-a+127) % 255 for a, u in zip(anchor, values))
    return bytes(out)


def role_domains(data, shape):
    validate_domain(data, shape)
    layers, _, channels = shape
    anchor, nonanchor = bytearray(), bytearray()
    for layer in range(layers):
        start = layer*10*channels
        anchor.extend(data[start:start+channels])
        nonanchor.extend(data[start+channels:start+10*channels])
    return bytes(anchor), bytes(nonanchor)


def counts(data):
    counter = Counter(data)
    return [counter[i] for i in range(255)]


def representation_counts(data, shape):
    transformed = transform(data, shape)
    if transform(transformed, shape, inverse=True) != data:
        raise ValueError('Lossless symbol inverse mismatch')
    anchor, nonanchor = role_domains(data, shape)
    new_anchor, residual = role_domains(transformed, shape)
    if new_anchor != anchor:
        raise ValueError('Token 0 anchor changed')
    return {('RAW_GLOBAL', 'all'): counts(data),
            ('RAW_ROLE_SPLIT', 'anchor'): counts(anchor),
            ('RAW_ROLE_SPLIT', 'nonanchor'): counts(nonanchor),
            ('ANCHOR_MOD_RESIDUAL', 'anchor'): counts(new_anchor),
            ('ANCHOR_MOD_RESIDUAL', 'residual'): counts(residual)}


def entropy(hist):
    if len(hist) != 255 or any(type(n) is not int or n < 0 for n in hist):
        raise ValueError('Expected 255 nonnegative integer counts')
    total = sum(hist)
    return -sum((n/total)*math.log2(n/total) for n in hist if n) if total else 0.0


def reduction(value, baseline):
    if baseline == 0:
        return 0.0 if value == 0 else None
    return 100*(baseline-value)/baseline


class EntropyAccumulator:
    def __init__(self):
        self.histograms = defaultdict(lambda: [0]*255)
        self.blocks = defaultdict(set)
        self.seen = set()

    def add(self, block, component, histograms):
        identity = block['block_id'], component
        if identity in self.seen:
            raise ValueError('Duplicate block/component')
        self.seen.add(identity)
        if block['token_group_size'] != 10 or block['partition'] not in SPLITS or block['dataset'] not in DATASETS:
            raise ValueError('Invalid B1 block/split/dataset')
        if component not in ('K', 'V') or set(histograms) != {(f, r) for f in FAMILIES for r in ROLES[f]}:
            raise ValueError('Invalid representation/component')
        total = sum(histograms['RAW_GLOBAL', 'all'])
        anchor = histograms['RAW_ROLE_SPLIT', 'anchor']
        nonanchor = histograms['RAW_ROLE_SPLIT', 'nonanchor']
        if (not total or [a+b for a, b in zip(anchor, nonanchor)] != histograms['RAW_GLOBAL', 'all'] or
                histograms['ANCHOR_MOD_RESIDUAL', 'anchor'] != anchor or
                sum(histograms['ANCHOR_MOD_RESIDUAL', 'residual']) != sum(nonanchor)):
            raise ValueError('Representation count conservation failed')
        for (family, role), hist in histograms.items():
            entropy(hist)  # Validate integer counts.
            for dataset in ('ALL', block['dataset']):
                key = block['partition'], dataset, component, family, role
                target = self.histograms[key]
                for i, n in enumerate(hist):
                    target[i] += n
                self.blocks[key[:3]].add(block['block_id'])

    def summary(self):
        rows = []
        for split in SPLITS:
            for dataset in ('ALL', *DATASETS):
                for component in ('K', 'V', 'K+V'):
                    components = ('K', 'V') if component == 'K+V' else (component,)
                    if component == 'K+V' and self.blocks[split, dataset, 'K'] != self.blocks[split, dataset, 'V']:
                        raise ValueError('K/V block coverage differs')
                    group = []
                    for family in FAMILIES:
                        by_role = {}
                        for role in ROLES[family]:
                            hist = [self.histograms[split, dataset, c, family, role] for c in components]
                            n = sum(sum(h) for h in hist)
                            if not n:
                                raise ValueError('Empty split/dataset/component population')
                            by_role[role] = (n, sum(sum(h)*entropy(h) for h in hist)/n,
                                             sum(v > 0 for v in hist[0]) if len(hist) == 1 else None)
                        n = sum(x[0] for x in by_role.values())
                        weighted = sum(x[0]*x[1] for x in by_role.values())/n
                        row = dict(split=split, dataset=dataset, token_group_size=10, component=component,
                            representation=family, block_count=len(self.blocks[split, dataset, components[0]]),
                            symbol_count=n, unique_symbol_count=(sum(v > 0 for v in self.histograms[
                                split, dataset, component, family, ROLES[family][0]]) if family == 'RAW_GLOBAL' and component != 'K+V' else None),
                            entropy_bits_per_symbol=weighted, weighted_bits_per_symbol=weighted,
                            H_raw=by_role['all'][1] if family == 'RAW_GLOBAL' else None,
                            H_anchor_raw=by_role['anchor'][1] if family == 'RAW_ROLE_SPLIT' else None,
                            H_nonanchor_raw=by_role['nonanchor'][1] if family == 'RAW_ROLE_SPLIT' else None,
                            H_anchor=by_role['anchor'][1] if family == 'ANCHOR_MOD_RESIDUAL' else None,
                            H_residual=by_role['residual'][1] if family == 'ANCHOR_MOD_RESIDUAL' else None,
                            anchor_symbol_count=by_role.get('anchor', (0,))[0],
                            nonanchor_or_residual_symbol_count=by_role.get('nonanchor', by_role.get('residual', (0,)))[0],
                            anchor_unique_symbol_count=by_role.get('anchor', (None, None, None))[2],
                            nonanchor_or_residual_unique_symbol_count=by_role.get('nonanchor', by_role.get('residual', (None, None, None)))[2],
                            empirical_symbol_counts_json=json.dumps({c: {r: self.histograms[split, dataset, c, family, r]
                                for r in ROLES[family]} for c in components}, separators=(',', ':')),
                            K_V_model_policy='SEPARATE_MODELS_WEIGHTED_COSTS',
                            count_semantics='roles and K/V are separate distributions; unique count undefined for model mixtures',
                            result_kind='EMPIRICAL_ENTROPY_NOT_ENCODED_STORAGE', **LABELS)
                        group.append(row)
                    for row in group:
                        row['reduction_vs_RAW_GLOBAL_percent'] = reduction(row['weighted_bits_per_symbol'], group[0]['weighted_bits_per_symbol'])
                        row['reduction_vs_RAW_ROLE_SPLIT_percent'] = reduction(row['weighted_bits_per_symbol'], group[1]['weighted_bits_per_symbol'])
                    rows.extend(group)
        return rows


def decision(rows):
    relevant = {r['dataset']: r for r in rows if r['split'] == 'evaluation' and
                r['component'] == 'K+V' and r['representation'] == 'ANCHOR_MOD_RESIDUAL'}
    gains = {d: relevant[d]['reduction_vs_RAW_ROLE_SPLIT_percent'] for d in ('ALL', *DATASETS)}
    go = (gains['ALL'] is not None and gains['ALL'] >= RULE['threshold_percent'] and
          all(gains[d] is not None and gains[d] > 0 for d in DATASETS))
    return dict(rule=RULE, reduction_percent=gains, recommendation='GO_TO_B2' if go else 'STOP',
                B2_executed=False, interpretation='Practical screening only; no statistical-significance or actual-storage claim')


def select_blocks(capture):
    verify_capture(capture)
    if capture.get('scope') != 'base_raw_unscaled_linear_projection; no LoRA adapter':
        raise ValueError('Expected frozen C1 base-projection capture')
    selected = [b for split in SPLITS for b in capture['blocks']
                if b['partition'] == split and b['token_group_size'] == 10]
    populations = {s: {d: sum(b['partition'] == s and b['dataset'] == d for b in selected)
                       for d in DATASETS} for s in SPLITS}
    for split, numbers in populations.items():
        if not all(numbers.values()):
            raise ValueError('Both datasets must have T=10 blocks in each frozen split')
        numbers['ALL'] = sum(numbers.values())
    if populations['evaluation']['ALL'] != 236:
        raise ValueError('Expected all 236 frozen evaluation T=10 blocks')
    return selected, populations


def tensor_diagnostics(root, rec, block, contract):
    """One unchanged C1 quantization; inverse representation; unchanged scales."""
    import torch
    if block['token_group_size'] != 10:
        raise ValueError('B1 rejects T=3 and incomplete groups')
    qkv = load_fixture(root, block)
    encoded = quantize(qkv['k'], qkv['v'], device=contract['quantization_device_resolved'])
    recovered, histograms, locality = [], {}, {}
    for component, (symbols, scales) in zip(('K', 'V'), encoded):
        if symbols.dtype != torch.int8 or scales.dtype != torch.float32:
            raise ValueError('C1 symbol/scale dtype mismatch')
        shape = tuple(symbols.shape)
        domain = bytes((symbols.to(torch.int16)+127).to(torch.uint8).flatten().tolist())
        transformed = transform(domain, shape)
        inverse = transform(transformed, shape, inverse=True)
        if inverse != domain:
            raise ValueError('Symbol roundtrip mismatch')
        restored = (torch.frombuffer(bytearray(inverse), dtype=torch.uint8).to(torch.int16)-127).to(torch.int8).reshape(shape)
        if not torch.equal(symbols, restored):
            raise ValueError('Reconstructed int8 symbols differ from C1')
        recovered.append((restored, scales))  # Original scales are preserved, never re-estimated.
        histograms[component] = representation_counts(domain, shape)
        x = qkv[component.lower()].double()
        anchor = x[:, 0, :]
        entries = []
        for token in range(1, 10):
            target = x[:, token, :]
            delta = target-anchor
            entries.append(dict(token_distance=token, element_count=delta.numel(),
                abs_delta_sum=delta.abs().sum().item(), delta_sq_sum=(delta*delta).sum().item(),
                target_sq_sum=(target*target).sum().item(), anchor_sq_sum=(anchor*anchor).sum().item(),
                dot_sum=(target*anchor).sum().item()))
        locality[component] = entries
    actual = reconstruct(tuple(recovered), device=contract['reconstruction_device'])
    direct = reconstruct(encoded, device=contract['reconstruction_device'])
    if not all(torch.equal(a, b) for a, b in zip(actual, direct)):
        raise ValueError('Inverse representation changed C1 reconstruction')
    saved_exact = None  # C1 saved only evaluation reconstructions.
    if block['partition'] == 'evaluation':
        existing_uniform(root, rec, block, actual)
        saved_exact = True
    return histograms, locality, dict(block_id=block['block_id'], split=block['partition'],
        dataset=block['dataset'], K_symbol_roundtrip_exact=True, V_symbol_roundtrip_exact=True,
        scales_preserved_exact=True, unchanged_quantizer_reconstruction_exact=True,
        saved_c1_reconstruction_exact=saved_exact,
        saved_c1_check='CHECKED' if saved_exact else 'NOT_APPLICABLE_CALIBRATION_NOT_SAVED_BY_C1')


def locality_summary(acc):
    rows = []
    for (split, dataset, component, distance), sums in sorted(acc.items(), key=lambda x: str(x[0])):
        n = sums['element_count']
        rows.append(dict(split=split, dataset=dataset, token_group_size=10, component=component,
            token_distance=distance, element_count=n, mean_absolute_delta=sums['abs_delta_sum']/n,
            rms_delta=math.sqrt(sums['delta_sq_sum']/n),
            relative_L2=math.sqrt(sums['delta_sq_sum']/sums['target_sq_sum']) if sums['target_sq_sum'] else None,
            cosine_similarity=sums['dot_sum']/math.sqrt(sums['target_sq_sum']*sums['anchor_sq_sum'])
                if sums['target_sq_sum'] and sums['anchor_sq_sum'] else None,
            relative_L2_denominator='L2 of matching non-anchor original tensor',
            cosine_definition='dot/norm of concatenated target and repeated-anchor values; not mean per-vector cosine',
            result_kind='DESCRIPTIVE_RAW_FP16_LOCALITY_ONLY', **LABELS))
    return rows


def atomic_json(path, data):
    atomic_bytes(path, (json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())


def atomic_bytes(path, data):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def write_csv(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())


def bind_inputs(args):
    if args.capture_manifest.name != 'capture_manifest.json':
        raise ValueError('Use the frozen C1 capture_manifest.json')
    root, aroot = args.capture_manifest.resolve().parent, args.c15a_dir.resolve()
    capture = read_json(args.capture_manifest)
    selected, populations = select_blocks(capture)
    rec = read_json(root/'reconstruction_manifest.json')
    if rec['capture_manifest_sha256'] != digest(capture):
        raise ValueError('C1 reconstruction manifest/capture mismatch')
    contract = resolve_contract(root)
    if not contract['quantization_device_resolved'].startswith('cuda:'):
        raise ValueError('B1 requires recorded C1 CUDA contract, no CPU fallback')
    am = read_json(aroot/'full_storage/manifest.json')
    if (am['status'] != 'COMPLETED' or am['primary_result_eligible'] is not True or
            am['completed_block_mode_pairs'] != 2616 or am['failed_pairs'] != 0):
        raise ValueError('C1.5-A completed primary baseline required')
    verify_contract(am['run_contract']['device_contract'], contract)
    for name, key in [('capture_manifest.json', 'c1_capture_manifest_sha256'),
                      ('reconstruction_manifest.json', 'c1_reconstruction_manifest_sha256')]:
        if file_hash(root/name) != am[key]:
            raise ValueError('A/C1 input hash mismatch')
    if file_hash(Path(__file__).parent/'codecs.py') != am['baseline_source_sha256']:
        raise ValueError('C1.5-A Baseline source changed')
    pm = aroot/'profile_manifest.json'
    if file_hash(pm) != am['run_contract']['profile_manifest_sha256']:
        raise ValueError('Frozen A profile manifest hash mismatch')
    for mode, name in [('SHARED_CDF_GLOBAL', 'shared_cdf_global.bin'),
                       ('SHARED_CDF_LAYERGROUP', 'shared_cdf_layergroup.bin')]:
        if file_hash(aroot/name) != am['profile_hashes'][mode]:
            raise ValueError('Frozen A profile hash mismatch')
    paths = [root/name for name in ('capture_manifest.json', 'reconstruction_manifest.json',
        'c1_block_raw.csv', 'c1_summary.csv', 'c1_quality.csv', 'environment.json')]
    paths += [aroot/name for name in ('manifest.json', 'profile_manifest.json',
        'shared_cdf_global.bin', 'shared_cdf_layergroup.bin', 'full_storage/manifest.json',
        'full_storage/storage_manifest.json', 'full_storage/c15_full_block_raw.csv',
        'full_storage/c15_full_summary.csv', 'full_storage/environment.json')]
    hashes = {str(p): file_hash(p) for p in paths}
    for block in selected:
        if not (root/block['file']).resolve().is_relative_to(root):
            raise ValueError('Fixture path escapes C1 root')
    return root, selected, populations, rec, contract, hashes


def run(args):
    parent = args.output_dir.resolve()
    root = args.capture_manifest.resolve().parent
    for protected in (root, args.c15a_dir.resolve()):
        protect_output(parent, protected)
    design_path = parent/'design_manifest.json'
    design = read_json(design_path)
    out = parent/'b1'
    # Exclusive directory creation makes this a single-writer, non-overwriting job.
    out.mkdir()  # Existing complete, failed or interrupted runs must be preserved.
    state = dict(status='INCOMPLETE', diagnostic_result_eligible=False, primary_storage_result=False,
        config=CONFIG, config_sha256=digest(CONFIG), **LABELS, completed_blocks=0, attempted_blocks=0,
        no_quality_change='Exactly inverted symbols and unchanged scales preserve C1 UNIFORM_INT8 reconstruction.',
        official_cachegen_reproduction=False, no_arithmetic_coding=True, no_profile_fitting=True,
        no_new_lossy_quantization=True, decision=None)
    started = time.monotonic()
    original_term = signal.getsignal(signal.SIGTERM)
    def interrupt(signum, frame):
        raise KeyboardInterrupt('B1 interrupted')
    signal.signal(signal.SIGTERM, interrupt)
    def publish(status):
        state['status'] = status
        state['diagnostic_result_eligible'] = status == 'COMPLETED'
        atomic_json(out/'manifest.json', state)
        design['stages']['B1']['status'] = status
        design['stages']['B1']['execution_status'] = status
        design['stages']['B1']['decision_rule'] = RULE
        design['stages']['B1']['config_sha256'] = digest(CONFIG)
        design['stages']['B1']['execution_manifest'] = str((out/'manifest.json').resolve())
        design['stages']['B1']['decision'] = state['decision']
        design['experiment_implemented'] = True
        design['experiment_run'] = state['attempted_blocks'] > 0
        design['status'] = 'B1_'+status
        design['no_experiment_statement'] = 'Only B1 symbol/locality diagnostics implemented. No B2/B3, arithmetic coding, profile fitting, new lossy quantization or inference.'
        atomic_json(design_path, design)
    raw_rows, roundtrips = [], []
    accumulator, locality = EntropyAccumulator(), defaultdict(Counter)
    try:
        publish('INCOMPLETE')  # Freeze rule/transform before any calibration/evaluation fixture reads.
        root, selected, populations, rec, contract, hashes = bind_inputs(args)
        state.update(populations=populations, device_contract=contract, input_file_sha256=hashes,
            calibration_evaluation_overlap=0, selected_block_ids={s: [b['block_id'] for b in selected
                if b['partition'] == s] for s in SPLITS},
            implementation_sha256={str(p): file_hash(p) for p in (
                Path(__file__), Path(__file__).parent/'codecs.py',
                Path(__file__).parent/'shared/tensors.py', Path(__file__).parent/'shared/device_contract.py')},
            git_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, text=True, capture_output=True).stdout.strip())
        atomic_json(out/'environment.json', environment())
        publish('INCOMPLETE')
        for block in selected:
            state['attempted_blocks'] += 1
            histograms, local, exact = tensor_diagnostics(root, rec, block, contract)
            if (not all(exact[k] is True for k in ('K_symbol_roundtrip_exact', 'V_symbol_roundtrip_exact',
                    'scales_preserved_exact', 'unchanged_quantizer_reconstruction_exact')) or
                    (block['partition'] == 'evaluation' and exact['saved_c1_reconstruction_exact'] is not True)):
                raise ValueError('B1 exactness invariant failed')
            roundtrips.append(exact)
            for component in ('K', 'V'):
                accumulator.add(block, component, histograms[component])
                for (family, role), hist in histograms[component].items():
                    raw_rows.append(dict(block_id=block['block_id'], query_id=block.get('query_id'),
                        split=block['partition'], dataset=block['dataset'], token_group_size=10,
                        component=component, representation=family, role=role, symbol_count=sum(hist),
                        unique_symbol_count=sum(n > 0 for n in hist), entropy_bits_per_symbol=entropy(hist),
                        symbol_counts_json=json.dumps(hist, separators=(',', ':')),
                        histogram_alphabet='index 0..254 denotes symbol -127..127', **LABELS))
                for entry in local[component]:
                    sums = {k: v for k, v in entry.items() if k != 'token_distance'}
                    for dataset in ('ALL', block['dataset']):
                        for distance in (entry['token_distance'], 'ALL_1_9'):
                            locality[block['partition'], dataset, component, distance].update(sums)
            state['completed_blocks'] += 1
            publish('INCOMPLETE')
            print(f'B1 {state["completed_blocks"]}/{len(selected)} {block["partition"]} {block["block_id"]}', flush=True)
        if any(file_hash(path) != expected for path, expected in hashes.items()):
            raise ValueError('Frozen C1/C1.5-A artifacts changed during B1')
        verify_contract(contract, resolve_contract(root))
        if state['completed_blocks'] != len(selected) or len(accumulator.seen) != 2*len(selected):
            raise ValueError('Incomplete B1 coverage')
        summary = accumulator.summary()
        write_csv(out/'b1_entropy_raw.csv', raw_rows)
        write_csv(out/'b1_entropy_summary.csv', summary)
        write_csv(out/'b1_locality_summary.csv', locality_summary(locality))
        atomic_json(out/'b1_roundtrip.json', dict(status='COMPLETED', blocks=roundtrips,
            checked_blocks=len(roundtrips), all_symbol_roundtrips_exact=True, all_scales_preserved=True,
            evaluation_saved_c1_checks=populations['evaluation']['ALL'], **LABELS))
        state['decision'] = decision(summary)
        state['invocation_wall_seconds'] = time.monotonic()-started
        publish('COMPLETED')
    except BaseException as exc:
        state['decision'] = None
        state['error'] = f'{type(exc).__name__}: {exc}'
        publish('INCOMPLETE' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'FAILED')
        atomic_json(out/'b1_roundtrip.json', dict(status=state['status'], blocks=roundtrips,
            checked_blocks=len(roundtrips), complete=False, error=state['error'], **LABELS))
        raise
    finally:
        signal.signal(signal.SIGTERM, original_term)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-manifest', type=Path, default=Path('results/cachegen/c1/capture_manifest.json'))
    parser.add_argument('--c15a-dir', type=Path, default=Path('results/cachegen/c1_5'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/cachegen/c1_5b'),
                        help='B parent containing design_manifest.json; writes a new b1/ directory')
    run(parser.parse_args(argv))
