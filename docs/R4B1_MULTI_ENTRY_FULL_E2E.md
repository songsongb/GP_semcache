# R4-B1: multi-entry exact full-prompt Raw QKV

Work branch: `exp/raw-semcache-r4b`, based on `exp/raw-semcache-r4`.
R3 and R4-A source and existing evidence remain frozen. The reported R4-A
Matrix Job 152050 baseline (108 requests, 1296 steps, 64/64 distributed bitwise
comparisons, Native strict FAIL) is user-supplied historical evidence; B1 never
rewrites that run or converts its Native failure into a pass.

## Scope and storage

`MultiPromptCache` owns multiple independent, sealed R3-B `ExactRawCache`
instances. Their physical tensors remain genuine compact, detached
`CacheEntry` copies on the ES GPU, using the original `GlobalCache` admission
policy and integrity checks. B1 adds a shared byte limit and exact LRU across
full-prompt identities. LRU advances only on a successfully validated lookup;
physical reuse is recorded after generation succeeds. Victims are cleared
before new tensors are cloned. Evicted views retain no tensor owners. CUDA
allocator reserve, model weights, temporary forward captures and in-request KV
are outside this cache occupancy count, and are not presented as cached QKV.

The full namespace includes actual complete token IDs, attention mask,
logical positions, model/revisions/base weight fingerprint, tokenizer source,
revision, actual vocabulary hash and special IDs, adapter matrices, dtype,
device/GPU UUID, original prepared-manifest hash, endpoint/precision settings,
and execution source hashes. The original R3-B seals additionally validate every
layer/QKV tensor's shape, device, dtype, compact storage and content SHA, plus
source request and completed-generation provenance. Hashes index candidates;
the full context and sealed tensor contents must also validate.

`COLD_MISS` invalidates only the requested identity before doing a real lookup,
full prefill, and G-1 decode forwards. Admission occurs only after successful
generation. `NO_REUSE` leaves the multi-entry cache untouched. `WARM_HIT` must
find the exact resident entry, physically reuse all 96 prefill QKV projections,
create fresh in-request HF KV, and recompute each decode via independent CPU
UD and localhost TCP. A requested HIT that misses raises before any forward;
it never silently becomes recomputation. No partial-prefix matching, semantic
matching, cross-request KV, compression or remote-link claims are implemented.

The generation adapter delegates to frozen R4-A math and finite/position/KV
guards. It does not alter R4-A's single-entry router or any R3 primitive.

## Bounded physical pilot

The primary pilot uses the first two distinct exact L32 real-token prompts
from fresh frozen R4-A Native preparation. GPU uses fixed G8; the tiny CPU
fixture uses fixed G3. In each of one warmup block and two measured blocks
(101, 202), the order is:

1. A NO_REUSE, B NO_REUSE.
2. A COLD, B COLD.
3. A WARM, B WARM.

The main budget defaults to 64 MiB and must hold both entries. For the frozen
FP16 OPT-2.7B shape each L32 entry owns exactly 15,728,640 bytes (15 MiB).
The two completed cold generations remain resident across prompt switches.

After releasing the main cache, a separate one-entry capacity-pressure pool
runs A COLD, B COLD (evict A), then explicitly requests A WARM and verifies a
MISS/refusal with zero new wire frames. It then serves A COLD, A WARM, B COLD,
B WARM. The refusal is an expected negative diagnostic, never a served HIT or
performance row. Every served pressure row is compared to a matched NO_REUSE.

Total: 24 generation requests, 192 GPU or 72 fixture generation steps, plus
two unmeasured original-input Native/ES triage prefills per endpoint. Native
references run and exit before the bare ES loads; the UD never loads the model
or uses CUDA. This is a diagnostic pilot, not the R4-A matrix or R4-B2.

All requests preserve generated IDs/text, full serialized next-token vocabulary
logits for every emitted token, per-step 96-QKV hashes, full logit nonfinite
diagnostics, actual per-layer KV summaries and observed OPT position lookups.
Nonfinite values still abort; no clamping or masking is added. Every cache arm
must be bitwise identical to distributed NO_REUSE for IDs, text, serialized
logits and every step's QKV hashes. Failure stops subsequent requests.

## Evidence and analysis

`r4b1_requests.jsonl`, `r4b1_tokens.jsonl`, `r4b1_generation_outputs.jsonl`,
`es/logits/*.f32.bin`, `es/r4b1_logit_diagnostics.jsonl`,
`r4b1_cache_events.jsonl`, `es/r4b1_contexts.json`, `es/r4b1_schedule.json`,
`es/r4b1_hit_refusal.json`, `es/fidelity_checks.jsonl`, independent ES/UD wire
and UD layer logs retain the actual experiment. Events contain occupancy,
resident identities/source IDs, LRU order, byte capacity, measured admission,
lookup/validation and eviction durations. Request rows record TTFT/E2E,
cache operation times, and separate Prefill/Decode directional network bytes.
Insertion includes eviction time; these nested timers must not be added.
Checks, copies, content hashing and diagnostic writes are inside the
instrumented request wall; completed request/vector exports are outside it.

The independent analysis phase verifies file/source hashes, reconstructs
cache admission/LRU/eviction/occupancy from events, checks completed cold-source
provenance, actual KV/decode/wire ledgers and bitwise output fidelity. It uses
the unchanged R4-A matched-history strict Native analysis with original R3
thresholds; Native remains a separate gate, including dtype and original-95
triage checks. All original failures and quantified metrics remain visible in
`r4b1_parity_diagnostics.json` and `r4b1_decision.json`.

`r4b1_quality.json` and `r4b1_sacrebleu_signatures.json` retain corpus BLEU,
exact output fidelity, matched Native comparison and external-gold status via
the frozen quality implementation. Missing SacreBLEU is explicitly reported in
the CPU fixture; GPU preflight requires it. Optional run-only `--gold-jsonl`
uses the original external reference schema. No model pseudo-gold is generated.
`r4b1_latency_contrasts.csv` retains paired TTFT/E2E and block intervals.
Two prompts/two blocks supply weak conditional timing evidence.

GPU B1 pass fields remain false in fixtures and until an actual complete,
audited GPU run. `PRIMARY_PERFORMANCE_ELIGIBLE` and
`correctness_qualified_speedup` are always false in B1; no Native-failed run
can claim correctness-qualified speedup. Expected artifacts also include
`r4b1_results.json`, `r4b1_decision.json`, `r4b1_preflight.json`,
`r4b1_summary.txt`, worker plans/results and lifecycle logs. Failed runs retain
partial evidence and no pass flags. Analysis is read-only and cannot overwrite
inputs. Every output directory must be new/empty below `results/r4/b1`.

## Commands (not executed on a GPU by this implementation task)

```bash
PYTHONPATH=src:tests python3 -m unittest tests.test_r4b1_cache tests.test_r4b1_processes -v
python3 scripts/103_run_r4b1_multi_entry.py --phase fixture --test-fixture-only \
  --output-dir results/r4/b1/TEST_FIXTURE_ONLY_example
python3 scripts/103_run_r4b1_multi_entry.py --phase analyze \
  --inputs results/r4/b1/TEST_FIXTURE_ONLY_example --output-dir results/r4/b1/analysis_example
```

Within a separately obtained RTX3090 allocation on the work branch, prepare
fresh original-95 base artifacts, then run:

```bash
python3 scripts/103_run_r4b1_multi_entry.py --phase prepare --enable-gpu-experiment \
  --base-prepared <fresh_original_95_prepare> --output-dir results/r4/b1/prepare_unique
python3 scripts/103_run_r4b1_multi_entry.py --phase run --enable-gpu-experiment \
  --prepared results/r4/b1/prepare_unique/native --capacity-bytes 67108864 \
  --output-dir results/r4/b1/run_unique
python3 scripts/103_run_r4b1_multi_entry.py --phase analyze \
  --inputs results/r4/b1/run_unique --output-dir results/r4/b1/analysis_unique
```

`jobs/r4b1_opt_multi_entry.sh` is a separate Slurm pilot template gated by
`R4B1_ENABLE_GPU=1`, branch, allocation, offline dependencies and an actual
RTX3090 UUID. Create `logs/` before any future submission. No job is submitted
and no git push is performed by this task.

Remaining work: actual allocated RTX3090 measurement; resolve Native strict
parity separately; broader token budgets and larger working sets after the
pilot. Partial-prefix reuse and semantic matching are explicitly R4-B2 work.
CPU fixtures validate physical TCP, cache policy/accounting and protocol gates,
not OPT/PEFT/FP16 CUDA behavior or GPU benefit.

## Changed files

All implementation files are new; no existing R3 or R4-A file is edited.

- `src/semcache/edgelora/r4b1_cache.py`: sealed multi-entry ownership, shared capacity and LRU.
- `src/semcache/edgelora/r4b1_generation.py`: frozen-generation adapter and cache timing/occupancy.
- `src/semcache/edgelora/r4b1_runtime.py`: physical worker, interleaved requests and pressure/refusal experiment.
- `src/semcache/edgelora/r4b1_evidence.py`: schedule, file/provenance/wire validation and independent cache replay.
- `src/semcache/edgelora/r4b1_analysis.py`: Native, quality and latency analysis with conservative qualification.
- `scripts/103_run_r4b1_multi_entry.py`: new CLI, worker lifecycle, preparation, fixture and explicit analysis.
- `jobs/r4b1_opt_multi_entry.sh`: separate opt-in RTX3090 Slurm template.
- `tests/test_r4b1_cache.py`: capacity, ownership, LRU, identity and integrity controls.
- `tests/test_r4b1_processes.py`: physical CPU fixture, read-only analysis, negative audits and GPU opt-in.
- `docs/R4B1_MULTI_ENTRY_FULL_E2E.md`: scope, contracts, commands, artifacts and limitations.
