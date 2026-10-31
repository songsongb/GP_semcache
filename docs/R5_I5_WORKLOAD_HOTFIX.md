# R5-I5-P workload delivery hotfix

## Root cause confirmed locally

HEAD is 45578c1eb0e1d4ba8dce8e498baadc84dcf7222d. `git show --stat HEAD`
shows six committed paths: the preflight module/CLI/test/report and two launcher
files. The committed workload is still positional and lacks SCALE_SEMANTICS,
routing, turn_order, pack_groups and continuity. Its preflight consumer imports
those names, producing the reported ImportError on a clean checkout.

At the start of this hotfix, all five symbols and their integration were present
in local **uncommitted** workload edits. Related runner/audit/summary and fixture
edits were also uncommitted. The prior validation described that worktree, not
the incomplete commit delivered to SERAPH. This is a partial-commit/delivery
failure; the local Git evidence does not establish why those paths were omitted
from the original staging command or who selected that set of files.

The restored dependency set is delivered together in a portable patch against
45578c1. It includes the pre-existing intended tracked edits rather than discarding
them. A clean archive of 45578c1 is tested before and after applying the patch,
so symbols existing only in the developer's dirty checkout cannot satisfy the
delivery check. No remote repository inspection, push or GPU execution is used.

## Source and manifest consistency

The workload actively uses all five exported symbols:

* select calls turn_order for integer turn order (1,2,10).
* requests calls routing with routing_seed, never request position for ownership.
* prepare_experiment calls pack_groups, continuity and requests with the fixed seed.
* SHA-256 canonical `[dataset, version, conversation ID, seed]` selects user
  `int(hash,16)%30+1`; non-conversational datasets use original sample ID.
  Users 1–10 own A, 11–20 B, 21–30 C. Track/repetition/block position do not enter.
* Whole groups fit together; oversized groups split chronologically with explicit
  `cache_survives=false` break metadata. Partial sample-selection groups are marked.
* Every block records physical request lifetime, per-condition reset, dataset
  continuity boundary, scored targets and priming count. The sealed manifest
  records scale_summary and INDEPENDENT_BLOCK_DATASET_SCALE.
* The independent analyzer recomputes ownership from dataset identity and seed,
  verifies request hashes across A/B/C and retains separate numerical gates.
  Consolidation exports the independent-block scale/lifetime interpretation.

Raw-QKV inference, CPU LoRA/TCP arithmetic, same-adapter authorization, three
synthetic adapters, clusters=1, LRU and frozen thresholds are unchanged.
ROOT_CAUSE_UNRESOLVED remains the historical numerical status; I3F-B is Future Work.

## Exact source changes relative to 45578c1

Restored tracked changes:

1. src/semcache/edgelora/r5_i5_workload.py
2. src/semcache/edgelora/r5_i5_analysis.py
3. src/semcache/edgelora/r5_i5_summary.py
4. scripts/119_run_r5_i5.py
5. tests/test_r5_i5.py
6. tests/test_r5_i5_processes.py
7. docs/R5_I5_INTEGRATED_EVALUATION.md

Added by this hotfix:

8. tests/test_r5_i5_hotfix.py — public exports, active call sites, real CLI,
   conversation ownership and numeric/group continuity regressions.
9. docs/R5_I5_WORKLOAD_HOTFIX.md — this report.

The existing preflight module/CLI/test already in 45578c1 require no source edits.
Unrelated untracked files, historical results and real data artifacts are excluded
from the patch. No local source-control commit or push is implied by a patch file.

## Mandatory local commands

```bash
PYTHONPATH=src python3 -c 'from semcache.edgelora.r5_i5_workload import SCALE_SEMANTICS,routing,turn_order,pack_groups,continuity; print(SCALE_SEMANTICS)'
python scripts/123_preflight_r5_i5.py readiness
PYTHONPATH=src:tests python3 -m unittest test_r5_i5 test_r5_i5_controller \
  test_r5_i5_preflight test_r5_i5_hotfix test_r5_contracts test_r5_semantic \
  test_r5_controller test_r5_multi test_r5_fidelity test_r5_perf_evidence \
  test_r4c_opt_contract -v
R5_I5_TEST_OUTPUT_ROOT=/home/songsong/SemCache/results/r5/i5/hotfix_tcp_45578c1_20261101 \
  PYTHONPATH=src:tests python3 -m unittest test_r5_i5_processes test_r5_multi_processes -v
```

The no-registry CLI must exit 1 with valid JSON `status: NOT_READY`, six dataset
entries and `GPU_NOT_RUN: true`; missing dependencies/data are not import failures.
This workstation does not have SERAPH's registry_longbench3.json; that absence is
not evidence against the user-reported 200 Qasper, 200 2WikiMQA and 200 MultiNews
samples already prepared on SERAPH. They must be retained and checked there.

## Safe SERAPH application

Transfer the delivered patch and its SHA-256 using your approved file-transfer
method. These are operator instructions, not remote commands run by this task.
Use a clean tracked checkout at exactly 45578c1; leave unrelated untracked files
alone. Abort on any patch collision or different base rather than resetting it.

```bash
cd /data/khuss/repos/GP_semcache
# Inspect these; proceed only on the required branch/base with no tracked edits.
git branch --show-current
git rev-parse HEAD
git status --porcelain --untracked-files=no
sha256sum /tmp/r5_i5_workload_hotfix.patch

git apply --check /tmp/r5_i5_workload_hotfix.patch
git apply /tmp/r5_i5_workload_hotfix.patch
PYTHONPATH=src python -c 'from semcache.edgelora.r5_i5_workload import SCALE_SEMANTICS,routing,turn_order,pack_groups,continuity; print(SCALE_SEMANTICS)'
python scripts/123_preflight_r5_i5.py readiness
# Structured NOT_READY without a registry is expected, exit code 1.
python scripts/123_preflight_r5_i5.py readiness \
  --registry results/r5/i5/registry_longbench3.json \
  --base-prepared /ABS/VERIFIED_BASE --multi-prepared /ABS/EXPANDED_ADAPTERS
```

Use the actual base/adapter paths on SERAPH. Do not run the dataset normalizer,
regenerate the 200-example snapshots, overwrite manifests or edit old execution
plans. New group-aware plans require fresh output directories after readiness.
GPU preflight still requires reviewed committed sources: stage precisely the nine
paths above and make a local hotfix commit after reviewing/testing the applied
patch. Do not use `git add .`, `git clean`, `git reset --hard` or overwrite evidence.
No GPU jobs or pushes are part of this hotfix. Retain a before/after SHA inventory
of results/r5/i5/data and registry_longbench3.json if verifying the server transfer.
