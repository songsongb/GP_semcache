# Milestone 4 status — 2026-09-14

**IMPLEMENTED / CPU-TESTED. PRETRAINED-MODEL TESTED / REAL-GPU VERIFIED:
user-reported SERAPH validation, as supplied at the start of Milestone 5.**

The user reports M2–M4 verification with pretrained OPT-125m, pinned revision
27dcfa74d334bc871f3234de431e71c6eeba5dd6, float32, Torch 2.6.0+cu118,
Transformers 4.57.6, PEFT 0.20.0 and RTX A2000 12GB. M4 total projection
decomposition and reconstructed full-forward logits matched exactly. The local
CPU evidence and historical pending-SERAPH notes below are retained separately.

## Measured local evidence

Python 3.10.12, Torch 2.6.0+cpu, Transformers 4.57.6, PEFT 0.20.0.
Random tiny OPT: two layers, hidden dimension 16, two attention heads,
float32. No pretrained weights, CUDA, adapter training or personalized data.

| Check | Observed result |
|---|---|
| Existing suite before changes | 33 passed, 2 skipped |
| Complete suite after M4 | 62 passed, 3 skipped |
| Integration-enabled, offline suite | 62 passed, 3 skipped: local OPT-125m unavailable |
| Nonzero user_a delta norms, input IDs `[2,3,4]`, six Q/K/V projections | 0.005306176976020236–0.008305296586608684 |
| Decomposition maximum absolute / mean absolute / relative L2 error | 0 / 0 / 0 |
| Decomposition cosine | 0.9999999999999999–1.0000000000000002 (float64 reduction rounding) |
| Base Q/K/V weights and biases | SHA-256 fingerprints unchanged through creation/switching |
| Base ownership | Same original model and Q/K/V parameter objects |
| User isolation | Distinct A/B matrices and fixed-input deltas; exact user_a return identity |
| Fixed-input base contribution | Exactly equal between users |
| Full-forward divergence | Later-layer base differs for some rows as user hidden states diverge |
| Communication, batch=1, n=3, d=16, float32 | 48 sent + 144 returned = 192 elements; 768 bytes |
| Communication dtype tests | float16, bfloat16, float32, float64; batch 1/2; mixed dtype accounting |
| Full-forward reconstructed Q/K/V, all two layers | 6 replacements; max/mean logit error, relative L2 and KL exactly 0; last argmax agrees |
| CLI scripts 12/13/14 | Passed offline tiny-model loader substitution tests; CSV/JSON schema validated |
| External interface | Local synthetic-adapter save/load roundtrip passed; no trained adapter supplied |
| Compilation / whitespace checks | Passed |

The PEFT save/load roundtrip emits its expected warning about the random model's
empty pretrained config path; it is not a model download or correctness failure.

Local measured details are in `results/raw/milestone4_cpu_validation.json`
(generated, ignored by Git). Its model scope explicitly says random tiny OPT,
not pretrained OPT or paper results. The report contains 6 decomposition rows,
144 multiuser rows, and a full-forward control. Script 13 on the default four
pretrained layers is expected to produce 288 rows; that run has not occurred here.
No `lora_decomposition_probe.csv` pretrained measurements were fabricated.

## Local validation commands

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 /tmp/semcache-m3-venv/bin/python -m pytest -q
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 SEMCACHE_OPT_INTEGRATION=1 /tmp/semcache-m3-venv/bin/python -m pytest -q -rs
/tmp/semcache-m3-venv/bin/python -m compileall -q src scripts tests
git diff --check
```

A direct script 12 invocation with CPU and the pinned OPT-125m revision also
stopped at AutoConfig loading: the local Hugging Face cache lacks the required
model/config/tokenizer. It did not run a pretrained forward. This environment
has a CPU-only Torch build and `torch.cuda.is_available() == False`.

## Remaining verification and scope

SERAPH must run pretrained OPT-125m rank-8 decomposition, two-user isolation and
all-layer reconstructed logits using the exact commands in the README's
Milestone 4 section. Pin revision
`27dcfa74d334bc871f3234de431e71c6eeba5dd6` and export
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python. Numerical failures must be
investigated, never bypassed by automatically relaxing tolerance.

CPU UD offload is deferred. ES and UD are logical roles on one device in one
process; input/output layer placement and networking are not emulated. PCIe or
host timings are not paper network latency. Safe cross-user reuse, personalized
quality, BLEU, Fig. 6 and full SemCache policy/workloads remain unverified and
out of scope. Synthetic fixtures are not trained adapters.
