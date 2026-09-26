# C1.5-B2: actual storage for the frozen B1 representation

B2 is a `MEASURED_RESEARCH_EXTENSION`, `CACHEGEN_INSPIRED`,
`LOSSLESS_SYMBOL_TRANSFORM`. It is not an official CacheGen reproduction.
It uses the unchanged C1 CUDA `Baseline(UNIFORM_INT8)` quantizer and decoder,
original FP32 per-layer/token scales, the exact B1 transform, and the unchanged
C1.5-A deterministic CPU integer arithmetic coder. Q storage remains FP16.
There is no new lossy quantization, model inference, B3, C2, or production cache
integration. Exact inverse symbols and unchanged scales preserve C1 quality.

## Frozen population and modes

Only existing T=10 blocks are selected from the frozen C1 split, in capture
order. Calibration: 241 blocks (SNIPS 103, MultiWOZ 138). Evaluation: 236 blocks
(SNIPS 93, MultiWOZ 143). Overlap must be zero. No padding, T=3 execution, or
construction of new groups is allowed. The completed matching B1 GO result
and completed C1.5-A primary result are required as input evidence.

| Mode | K nonanchor stream | V nonanchor stream | Classification |
|---|---|---|---|
| `B2_RAW_ROLE_SPLIT` | Original symbols | Original symbols | `FAIR_CONTROL` |
| `B2_ANCHOR_MOD_RESIDUAL_KV` | B1 residual | B1 residual | `PRIMARY`, `PRIMARY_B2`, `B1_PREDECLARED_TRANSFORM` |
| `B2_K_RESIDUAL_V_RAW` | B1 residual | Original symbols | `POST_HOC_EXPLORATORY`, `NOT_PREDECLARED_PRIMARY` |

Every mode has four shared CDFs in order: K anchor, K nonanchor/residual,
V anchor, V nonanchor/residual. Token 0 always retains its original symbols.
CDFs pool layers and datasets; K, V, and token roles remain separate. All modes
use exactly the same CDF granularity and representation. The hybrid cannot be
promoted to the primary mode.

For C1 symbols `s` in `[-127,127]`, use `u=s+127`. The unchanged B1 transform
references token 0 independently for each layer/channel:

```text
anchor = u[:,0,:]
r_i = (u[:,i,:] - anchor) mod 255           # i=1..9
d_i = r_i if r_i <= 127 else r_i - 255
arithmetic alphabet index for residual = d_i + 127
inverse r_i = d_i mod 255
inverse u_i = (anchor + r_i) mod 255
inverse s_i = u_i - 127
```

This is a lossless transform of already quantized symbols, not a floating-point
delta quantizer. The subtraction sign and mapping are the frozen B1 research
choices, not claims about undocumented official-source behavior.

## Commands on SERAPH

Run from the repository root with the existing C1 Python/CUDA environment and
one CUDA GPU available. All three commands share these defaults: capture
`results/cachegen/c1/capture_manifest.json`, A directory
`results/cachegen/c1_5`, B1 directory `results/cachegen/c1_5b/b1`, output
`results/cachegen/c1_5b/b2`. They also accept explicit `--capture-manifest`,
`--c15a-dir`, `--b1-dir`, and `--output-dir` paths.

Freeze calibration-only profiles first:

```bash
python scripts/42_cachegen_anchor_storage.py profile-fit \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --output-dir results/cachegen/c1_5b/b2
```

Then run the deterministic smoke: first evaluation SNIPS block and first
evaluation MultiWOZ block, three modes each, six pairs. Smoke is always
`DIAGNOSTIC_ONLY` and `primary_result_eligible=false`.

```bash
python scripts/42_cachegen_anchor_storage.py smoke \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --output-dir results/cachegen/c1_5b/b2
```

After that smoke completes with matching hashes, run all 236 evaluation blocks
and three modes, 708 pairs:

```bash
python scripts/42_cachegen_anchor_storage.py storage-full \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --output-dir results/cachegen/c1_5b/b2
```

Repeat the identical smoke or storage-full command to resume interruption.
Profile fitting is deliberately write-once; it refuses to overwrite initialized
or frozen profiles. Preserve an interrupted/failed fit directory before fitting
into a new empty directory. `FAILED` evaluation runs are sealed and must be
preserved/archived before replacement. Run smoke before starting full storage;
a missing or incompatible smoke fails the full job without encoding pairs.

The optional Slurm wrapper runs full storage with one GPU, no embedded account
or partition. Activate the existing environment before submission and provide
cluster-specific flags externally:

```bash
sbatch scripts/slurm_c15b2_full_storage.sh
```

Commit/freeze the implementation before profile fitting. Profiles, smoke and
full jobs bind the implementation hashes, git revision, C1 CUDA contract,
runtime, input hashes, exact selected IDs, and frozen profile hashes. Changing
that contract requires a separate output directory and newly declared run.

## Actual bytes and correctness

Calibration fitting reads only the 241 calibration fixtures. Counts use the
unchanged C1.5-A Laplace +1 and deterministic integer normalization to 65,536.
Profiles freeze before any evaluation encoding. Evaluation never updates a CDF
or selects a variant. The profile manifest preserves calibration counts,
normalized profiles, byte sizes, SHA256 hashes, and the declared decision rule.

The four-stream adapter reuses C1.5-A's little-endian profile/block header
structs, uint32 CDF entries, uint32 stream lengths, FP32 scale order, and SHA256
trailer. Distinct B2 magic/mode values prevent confusion with A files. Each
profile occupies 4,108 bytes (12 header + 4 × 256 × 4 CDF bytes). Per block,
original scales occupy 2,560 bytes. Actual framing occupies 99 bytes
(51 header + 16 stream lengths + 32 checksum). There is no per-block anchor
index. `local_transform_metadata_bytes` includes all serialized framing;
`local_metadata_bytes` is framing plus scales. There is one container per pair
holding four actual arithmetic streams.

Each evaluation pair quantizes once on the recorded C1 CUDA device, preserves
scales, encodes each stream once, atomically writes its bitstream, reads it back,
decodes each stream once, and requires exact arithmetic-symbol roundtrip,
inverse-symbol equality, scale-byte/tensor equality, and exact saved C1
UNIFORM_INT8 reconstruction through the unchanged CUDA decoder. No tolerance
checks exist. File sizes determine stored bytes. Bitstreams are preserved.

For each independent evaluation pool:

```text
encoded_KV = one shared profile + sum(payload + original scales + framing)
encoded_SemCache = one shared profile + sum(FP16 Q + payload + scales + framing)
KV ratio = sum(raw_KV) / encoded_KV
SemCache ratio = sum(raw_Q + raw_KV) / encoded_SemCache
payload-only bits/symbol = 8 * payload / actual symbol count
profile-amortized bits/symbol = 8 * (payload + shared profile) / symbol count
all-stored KV bits/symbol = 8 * encoded_KV / symbol count
```

The summary reports ALL, SNIPS, and MultiWOZ T=10 pools, separate K/V payload
bytes, and frozen `UNIFORM_INT8`, GLOBAL, LAYERGROUP reference rows. Exact source
rows and hashes come from A's `full_storage/c15_full_summary.csv`. They are
reconciled against the exact evaluation IDs and A's block raw CSV. Smoke
reference pools use only its two matching IDs, not the full 236-block pool.
Shared profiles are charged once per independent stratum. Payload/raw/local
totals reconcile additively across datasets; independently charged dataset
profiles must be removed before reconciling stored bytes to the overall pool.

## Predeclared primary decision and resumability

The primary causal test is K+V residual versus raw role split on all 236 blocks.
`B2_PRIMARY_SUCCESS` requires all exactness checks and a smaller actual encoded
KV pool with **at least 3% relative KV-byte reduction**. The threshold is a
`REPRODUCTION_CHOICE` aligned with B1 and is written before evaluation.
Dataset directions are reported separately; they do not retrospectively change
the criterion. Hybrid storage and comparisons remain exploratory. A valid
complete primary evaluation may return `B2_PRIMARY_NO_SUCCESS`; eligibility
does not mean the transform improved storage.

One output-root lock excludes concurrent writers across fit/smoke/full.
Durable atomic bitstream writes precede durable atomic per-pair checkpoints.
Only completed checkpoints matching the run contract, provenance, correctness
flags, file checksum, shape and exact byte accounting are skipped. Their files
are inspected, without another arithmetic encode/decode. Capture and saved C1
tensor hashes are revalidated. Deterministic block/mode order survives resume;
duplicates, incompatible state, and corruption fail closed. A crash after a
checkpoint but before progress publication does not recompute that pair. An
uncommitted interrupted pair can be retried.

`INCOMPLETE` and `FAILED` have no final summary and no primary eligibility.
Full `COMPLETED` requires all 708 unique pairs and final input/profile/source/
device/correctness/accounting validation. The full primary summary is written
only then, with the completion manifest published last in its run directory.
Timing is solely `SINGLE_PASS_OPERATIONAL_TIMING`,
`NOT_PRIMARY_LATENCY_RESULT`, with zero warmups and zero measurement repeats.
It cannot be compared to C1's repeated codec timings.

## Outputs and validation

All newly measured output stays under `results/cachegen/c1_5b/b2/`:

```text
manifest.json
b2_profile_manifest.json
profiles/raw_role_split.bin
profiles/anchor_mod_residual_kv.bin
profiles/k_residual_v_raw.bin
smoke/{manifest.json,progress.json,b2_block_raw.csv,b2_summary.csv,
       run_diagnostics.json,environment.json,checkpoints/,bitstreams/}
full_storage/{manifest.json,progress.json,b2_block_raw.csv,b2_summary.csv,
              run_diagnostics.json,environment.json,checkpoints/,bitstreams/}
smoke/failures.jsonl or full_storage/failures.jsonl  # only on failure
```

Profile fitting is a calibration histogram/transform scan; smoke is a small
actual coding job. Full storage is an hours-class CPU reference-coder job with
one CUDA GPU used for C1 quantization/reconstruction. Exact duration depends on
the host; no real timing was measured by implementation tests.

Tests require no model downloads or inference:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_cachegen_*.py' -v
```

Packed-symbol/coder, persistence, coverage, fairness, reference-accounting and
failure-gate tests run without PyTorch. Real tensor roundtrips additionally run
when PyTorch is installed; recorded CUDA parity must pass the SERAPH smoke.
C1, C1.5-A, B1, their profiles, source implementations, and artifacts are read-only.
