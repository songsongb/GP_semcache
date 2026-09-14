# Milestone 6A status

**IMPLEMENTED / CPU-TESTED.** Dataset ingestion adapters, deterministic workloads,
static audits, config/model metadata, analytical cost replay, M5-engine logical
smoke, physical runner interfaces, aggregation, manifests and read-only paper
references are implemented. **PAPER-RESULT REPRODUCED: NO.**

M2–M5 pretrained OPT-125m REAL-GPU VERIFIED on SERAPH is **user-reported** at
the start of this milestone. This does not establish M6A pretrained/GPU evidence.
Historical M5 local CPU records remain unchanged in the earlier status document.

## Evidence and boundaries

- Baseline before M6A: 83 passed, 4 skipped; all M1–M5 tests preserved.
- M6A normal suite: 104 passed, 8 skipped (final validation below).
- Integration-enabled suite: 104 passed, 8 skipped. Four pretrained checks skip
  missing local OPT-125m artifacts; three dataset checks skip missing explicit
  source paths; optional SacreBLEU numerical check skips the absent package.
- CPU random tiny OPT runs validate normalized workload records through M5 in
  MEASURED_MODEL and HYBRID, including physical QKV storage and measured prefill.
- Subprocess tests execute prepare → validate → logical smoke over synthetic
  SNIPS-shaped fixtures, verify 50 users, SHA stability across processes, no
  workload mutation and persisted manifests/raw events/aggregates.
- Script 22 passes saved FLOPs/elements, Mbps conversion and 512/300 checks.
- Local environment: Python 3.10.12, Torch 2.6.0+cpu, Transformers 4.57.6,
  PEFT 0.20.0. No CUDA, downloaded weights, real dataset, model training or
  pretrained inference was used. No packages were installed.

| Status label | M6A evidence |
|---|---|
| IMPLEMENTED | Yes, foundation scope |
| CPU-TESTED | Yes, synthetic data and random tiny OPT |
| DATASET-VERIFIED | NOT YET VERIFIED; zero real datasets loaded |
| PRETRAINED-MODEL TESTED | NOT YET VERIFIED for M6A |
| REAL-GPU VERIFIED | NOT YET VERIFIED for M6A |
| PAPER-RESULT REPRODUCED | No |

Full-generation SemCache, baseline execution other than SEMCACHE, trained
personalization, paper BLEU and full analytical system memory remain deferred.
Logical mode has no attention impact provider; CHU/PBR are unavailable, not
synthetically estimated. Unspecified compute rates and wire precision leave
latency null. Dataset smoke does not establish safe approximate QKV reuse.

See [paper experiment specification](paper_experiment_spec.md) and
[workload reconstruction](workload_reconstruction.md) for equations, assumptions,
unknowns and exact transformation rules. Optional dependency groups are
`datasets` and `evaluation` (SacreBLEU); core JSON ingestion needs neither.
Install them only through the environment's explicit dependency provisioning.
There is no pandas dependency or installer inside scripts.

## Exact SERAPH commands

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
conda activate semcache
cd /data/khuss/repos/GP_semcache
python -m pytest -q
SEMCACHE_OPT_INTEGRATION=1 python -m pytest -q
python scripts/22_validate_simulation_model.py --config configs/paper/common.yaml
```

Supply actual local source paths first. Each must contain only the chosen split
(or a supported split wrapper); do not point these at synthetic fixtures when
claiming dataset verification. Sources can be local JSON/JSONL or HF saved data.

```bash
export SEMCACHE_MULTIWOZ_PATH=/absolute/path/to/multiwoz_train.json
export SEMCACHE_COQA_PATH=/absolute/path/to/coqa-train-v1.0.json
export SEMCACHE_SNIPS_PATH=/absolute/path/to/snips_train.json

SEMCACHE_DATASET_INTEGRATION=1 python -m pytest -q tests/test_m6_workloads.py -rs

python scripts/20_prepare_paper_workloads.py --dataset multiwoz --config configs/paper/multiwoz.yaml --input-path "$SEMCACHE_MULTIWOZ_PATH" --output results/workloads/multiwoz.jsonl --seed 42
python scripts/21_validate_paper_workloads.py results/workloads/multiwoz.jsonl --config configs/paper/multiwoz.yaml
python scripts/20_prepare_paper_workloads.py --dataset coqa --config configs/paper/coqa.yaml --input-path "$SEMCACHE_COQA_PATH" --output results/workloads/coqa.jsonl --seed 42
python scripts/21_validate_paper_workloads.py results/workloads/coqa.jsonl --config configs/paper/coqa.yaml
python scripts/20_prepare_paper_workloads.py --dataset snips --config configs/paper/snips.yaml --input-path "$SEMCACHE_SNIPS_PATH" --output results/workloads/snips.jsonl --seed 42
python scripts/21_validate_paper_workloads.py results/workloads/snips.jsonl --config configs/paper/snips.yaml

python scripts/23_run_workload_smoke.py --workload results/workloads/multiwoz.jsonl --config configs/paper/multiwoz.yaml --max-queries 100
python scripts/23_run_workload_smoke.py --workload results/workloads/coqa.jsonl --config configs/paper/coqa.yaml --max-queries 100
python scripts/23_run_workload_smoke.py --workload results/workloads/snips.jsonl --config configs/paper/snips.yaml --max-queries 100
```

Alternatively set `dataset_source.input_path` in a copied dataset config, then
omit `--input-path`. Source release/revision is not guessed. For external HF
loading explicitly supply `--hf-id`, optional `--hf-config`, `--revision` and
`--allow-download`. No package is installed automatically.

For model-token audits append the following to script 21 or 23 once provisioned:

```bash
--tokenizer-id facebook/opt-125m --tokenizer-revision 27dcfa74d334bc871f3234de431e71c6eeba5dd6
```

This is development tokenization, not OPT-6.7B reproduction. For real semantic
encoding set `semantic_encoder.model_id` and `.revision` in a copied config,
then use `--encoder real`; at least C queries are required. Actual checkpoint
selection/resolved revision is recorded. We have not selected or tested one.
Script 23's default fixture and FP16 logical activation payload are explicit
smoke choices. Override activation payload with `--qkv-precision-bits`.

Local validation commands used here:

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 /tmp/semcache-m3-venv/bin/python -m pytest -q
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 SEMCACHE_OPT_INTEGRATION=1 SEMCACHE_DATASET_INTEGRATION=1 /tmp/semcache-m3-venv/bin/python -m pytest -q -rs
/tmp/semcache-m3-venv/bin/python scripts/22_validate_simulation_model.py --config configs/paper/common.yaml
/tmp/semcache-m3-venv/bin/python -m compileall -q src scripts tests
 git diff --check
```
