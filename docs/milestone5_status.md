# Milestone 5 status — 2026-09-14

**IMPLEMENTED / CPU-TESTED. PRETRAINED-MODEL TESTED: NOT YET VERIFIED.
REAL-GPU VERIFIED: NOT YET VERIFIED for M5.**

M2–M4 SERAPH real-GPU verification is user-reported and preserved. M5 local
execution used Python 3.10.12, Torch 2.6.0+cpu, Transformers 4.57.6 and PEFT
0.20.0. CUDA is unavailable; local OPT-125m files are unavailable. No downloads,
pretrained M5 forwards, CUDA M5 forwards, trained adapters or datasets were used.

## Implemented flow and evidence

- Existing semantic encoder/clusterer/window/matcher APIs feed `SemCacheEngine`.
  Fixture and optional explicitly configured HF modes; no hidden TinyBERT ID.
- Cluster plus exact token tuple indexes the existing `GlobalCache`. Different
  source/target positions work; another cluster cannot hit the same tokens.
- Deterministic greedy overlap selection creates one reused-position mask.
- Admission uses existing Eq. 11 and strict `>`; allocation is deferred until
  acceptance. Overflow uses existing maximum Eq. 12, including candidate and
  deterministic ties. Tests check every actual victim against the full population.
- Appearance counters and actual reuse counters remain distinct. Age advances on
  the query clock; static S is preserved. Lookup alone does not increment reuse F.
- Actual attention row L2, head mean, token/layer sum provides impact. Existing
  CHU arithmetic and explicit PBR operate on bounded per-cluster query history.
- Cached component is source-user TOTAL QKV, compact CPU tensors per layer.
  Approximate forwards can populate subsequent entries from their actual mixed
  outputs. No target-user delta correction or safe-reuse policy is introduced.
- All-layer mixed projection gathers only unmatched hidden rows into native PEFT,
  then scatters fresh and cached totals before normal model attention/FFN.
  Native base and LoRA A pre-hooks in tests independently observe the subset.
  Multiple windows, zero fresh rows, shape errors, exception cleanup and exact
  output controls are covered. Installed Transformers/PEFT source is unchanged.
- Summary CSV and event JSONL separate query quality, lookup/reuse accounting,
  policy changes and physical storage. All retain `safe_reuse_claimed=false`.

Measured random two-layer OPT (hidden 16, two heads, rank 8), using the character
fixture tokenizer in a separate local five-query health/weather trace:

| Control / target | Observation |
|---|---|
| Exact control | 66 full rows; 6 reused in two windows; 60 native fresh rows for each Q/K/V per layer |
| Exact control projection error | 0 |
| Exact control max/mean logit error, relative L2, KL | 0 / 0 / 0 / 0; last argmax agrees |
| Exact-control physical cache | 2,304 bytes |
| Health cross-user target | 57 unique reused tokens; max logit error 0.03132577799; relative L2 0.03637225281; last KL 7.418492139e-7 |
| Weather source in separate cluster | 0 cache hits despite shared text tokens |
| Weather cross-user target | 3 unique reused tokens; max logit error 0.01405426487; relative L2 0.02401853770 |
| Five-query trace | 71 inserts, 118 denials, 41 fetched/reused windows, 41 CHU updates, 71 manual PBR recalculations |
| Policy numerical fixture | Eq. 11/12 scores .46; strict .3 denial; CHU 1→2 with current 6; PBR→3 |
| Tiny 20-byte logical policy pool | Maximum-score victim .9 evicted; denied factory never called |

These numbers are CPU fixture correctness evidence, not pretrained OPT-125m
measurements, paper hit rate, performance results or personalized task quality.
Generated local evidence is `results/raw/milestone5_cpu_validation.json` (ignored
by Git). Scripts 17/18 are also tested through offline tiny-model loader
substitution, keeping their pretrained default result names unpopulated locally.

## Verification commands

Final normal and integration-enabled suites: **83 passed, 4 skipped** each
(all M1–M4 tests preserved; local pretrained files unavailable).
Scripts 15/16, script 17/18 offline CLI substitution tests, compile checks and
whitespace checks passed. The expected PEFT synthetic save/load warning remains.

Local suite (normal and integration-enabled) uses:

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 /tmp/semcache-m3-venv/bin/python -m pytest -q
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 SEMCACHE_OPT_INTEGRATION=1 /tmp/semcache-m3-venv/bin/python -m pytest -q -rs
/tmp/semcache-m3-venv/bin/python -m compileall -q src scripts tests
git diff --check
```

The optional pretrained tests skip missing local weights cleanly. Normal pytest
never downloads. SERAPH commands, with locally provisioned model/tokenizer:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
conda activate semcache
cd /data/khuss/repos/GP_semcache
python -m pytest -q
SEMCACHE_OPT_INTEGRATION=1 python -m pytest -q
python scripts/15_validate_semantic_frontend.py --config configs/development.yaml
python scripts/16_validate_cache_policy.py --config configs/development.yaml
python scripts/18_validate_mixed_projection.py --config configs/development.yaml --device cuda --revision 27dcfa74d334bc871f3234de431e71c6eeba5dd6
python scripts/17_run_semcache_controlled_trace.py --config configs/development.yaml --device cuda --revision 27dcfa74d334bc871f3234de431e71c6eeba5dd6 --compare-baseline
```

Do not proceed past an exact-control failure or relax its fixed tolerance.
Script 17 repeats the exact gate before approximation. Optional development
`--logical-capacity-bytes N` forces eviction without changing the paper default.
The optional pretrained pytest checks CPU; only successful CUDA scripts establish
M5 REAL-GPU VERIFIED.

## Boundaries and remaining ambiguities

M5 verifies **prefill only**. The explicit newest-token lookup interface uses a
singleton key; it cannot reuse w=3 entries. Decode population, past-key-values
and the full generation loop are deferred to M6. The controller accepts a
ModelAdapter and contains no OPT internal module paths.

Paper-unspecified checkpoint, centroid initialization/update scheduling, exact
matching, executable key, normalization edge cases, head aggregation, overlap
ties, repeated-occurrence history aggregation and manual PBR are documented in
[reproduction_choices.md](reproduction_choices.md#milestone-5-integrated-prefill-system).
The standalone optional HF probe 19 is omitted; the HF interface is implemented
but no real semantic encoder is tested. ES/UD roles remain local emulation;
analytical savings do not establish speedup or emulate a 200 Mbps network.

No safe reuse threshold, BLEU, Fig. 6–11, Table II, MultiWOZ/CoQA/SNIPS evaluation,
50-user run, trained personalized adapters or OPT-6.7B run is included.
