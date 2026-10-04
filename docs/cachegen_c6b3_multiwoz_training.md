# C6-B3-1 training and FULL_RECOMPUTE validation

All commands use frozen `multiwoz_plan_v2`, the original prepared source, and
`results/workloads/c6b3_multiwoz_history_semantic.jsonl`. No clustering or prompt
changes occur. K=4, training sequence bound384 and B3 prompt version remain fixed.
Input hashes, frozen v2 selection replay, original source reconstruction, whole
conversation assignments, and all holdout sets are checked before training.

Capability64 is constructed identically for both jobs, without any quality data.
Each user gets eight targets per available depth1/2/3/4+, each from a distinct
conversation. Within a bucket, the earliest eligible row represents each
conversation. Seed42 SHA ordering and deterministic augmenting matching select
32 disjoint conversations/user. Lack of candidates fails. Both source and target
conversations of frozen32 remain excluded. Capability conversations are removed
entirely from both training profiles.

Pilot shuffles remaining conversations deterministically per user, then appends
whole groups when they fit within5000 rows, preserving internal source order.
Oversize groups are skipped, not split. Pilot runs3 epochs. Full uses all remaining
conversations and runs2 epochs. Both shuffle selected training rows afresh per
epoch using a local seed derived from user/epoch/42; no labels or scores influence
ordering. Every selected row appears once per epoch. Order hashes are recorded.

The objective separately tokenizes one ASCII space + original response, appends
EOS, and supervises only those tokens. Frozen prompt IDs must reproduce exactly;
no truncation is allowed. Standard PEFT QKV LoRA uses rank8/alpha8/dropout0, no
bias. Frozen base weights are FP16; adapter matrices and Adam moments are FP32.
Batch1, accumulation8, AdamW lr1e-4, weight decay0, clip1.0, FP16 AMP and gradient
checkpointing match B1. Incomplete accumulation groups use their actual size.

Checkpoints every500 successful optimizer updates save both adapters, optimizer,
scaler, RNG states and the next epoch/batch position. They are written only after
an accumulation group; no partial gradients are needed. Resume uses a new output
root, the same profile/optimizer, and `--resume PATH/TO/checkpoint`. A completed
checkpoint requires `checkpoint.json`; interrupted partial checkpoints fail.
Only locally created, hash-verified tensor-only torch state is loaded. No
checkpoint is chosen by BLEU. Final adapters are standard `user_a/`, `user_b/`.
The frozen base QKV fingerprint must remain unchanged.

## Running

Use the existing SERAPH conda setup where `conda` is available to batch shells.
Jobs activate `semcache`, request exactly one GPU on `batch_eebme_ugrad`, and keep
cache/temp/log/output under `/data/khuss`. Pilot6h and full18h are independent.
They do not cancel, depend on, or select checkpoints from each other.

```bash
sbatch jobs/c6b3_b1_pilot.sh
sbatch jobs/c6b3_b1_full.sh
```

Each trains, then evaluates its capability64 cohort. Output roots:

- `results/cachegen/c6b3/b1_pilot`
- `results/cachegen/c6b3/b1_pilot_capability64`
- `results/cachegen/c6b3/b1_full`
- `results/cachegen/c6b3/b1_full_capability64`

Training directories contain capability_validation.json, train_selection.json,
training_manifest.json, per-user adapters and checkpoints. All roots must be new.
The two jobs write distinct logs `/data/khuss/c6b3_PROFILE_JOBID.{out,err}`.

Model-free planning (offline tokenizer only, no OPT weights):

```bash
python scripts/54_train_cachegen_c6b3_multiwoz.py --profile pilot --dry-run \
  --output-root results/cachegen/c6b3/b1_pilot_plan_check
```

Run the untrained BASE OPT baseline exactly once, independently, on the same64:

```bash
python scripts/55_eval_cachegen_c6b3_multiwoz.py --base \
  --output-root results/cachegen/c6b3/base_capability64
```

Use the same cache/offline exports as the jobs for interactive commands. Baseline
is not run by both jobs, avoiding duplicate baseline execution.

## Capability decision and final32

Evaluation is exclusively FULL_RECOMPUTE. Greedy generation has max_new_tokens160,
no sampling, beam1, normal EOS. The training384 bound is verified for reference
examples; generated continuation may extend beyond384 but must fit OPT's actual
position limit. The prompt itself is never changed or truncated.

Corpus SacreBLEU is computed once per corpus, with13a/exp/effective_order false/
lowercase false and one reference, points0..100. Version/signature and explicit
REPRODUCTION_CHOICE protocol are saved. Per-user BLEU and per-depth diagnostic
BLEU use separate corpora, not mean sentence BLEU. Also saved: generated text/IDs,
references, counts, reference-token edit distance/aligned-position agreement,
completion+EOS teacher-forced NLL and perplexity. Those token diagnostics are not
compression fidelity. No SemCache paper BLEU reproduction is claimed.

Review pilot/full capability64 manually. There is no automatic checkpoint choice,
stopping threshold or cancellation. If desired, cancel full manually with
`scancel <FULL_JOB_ID>`. Freeze the chosen completed adapter before final32.
Create a manual JSON decision containing:

```
{
  "training_manifest_sha256": "<chosen training_manifest.json file SHA256>",
  "capability_manifest_path": "/data/khuss/repos/GP_semcache/results/cachegen/c6b3/b1_pilot_capability64/capability_manifest.json",
  "capability_manifest_sha256": "<that manifest's file SHA256>",
  "final_output_root": "/data/khuss/repos/GP_semcache/results/cachegen/c6b3/frozen32_selected"
}
```

Then run once:

```bash
python scripts/55_eval_cachegen_c6b3_multiwoz.py \
  --adapter-root results/cachegen/c6b3/b1_pilot --cohort frozen32 \
  --freeze-decision /data/khuss/b3_adapter_freeze.json \
  --output-root results/cachegen/c6b3/frozen32_selected
```

The decision binds one new output root, the exact adapter training manifest and
completed capability64 evidence. Reusing the output root fails. No frozen32
result is consumed by training or checkpoint selection. The manual process must
not issue new decisions to repeatedly probe frozen32.

Capability outputs: capability_per_case.csv, capability_summary.json,
capability_manifest.json and capability_validation.json. Full provenance and
output hashes accompany completed results. Failed runs retain STARTING manifests,
which cannot pass completed-adapter/capability checks.

Local tests use synthetic data. Exact SERAPH row/token counts, available64 cohort,
GPU compatibility, training duration, BLEU and task capability remain unmeasured.
Synthetic fixture:45 five-turn conversations/user;32 held out for capability,
13 conversations /65 training rows per user remain. This is not a server count.
