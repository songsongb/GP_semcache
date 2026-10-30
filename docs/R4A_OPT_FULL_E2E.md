# R4-A: physical OPT generation and exact total-Raw-QKV replay

This is an add-only physical generation diagnostic. R3 sources, outputs and
strict Native PEFT thresholds remain frozen. GPU findings are NOT_RUN on the
personal Ubuntu implementation machine. No SERAPH artifacts are accessed.

R3's original ProjectionSession and UD hard-code the prepared length 32;
R3-B HIT replaces 96 total Q/K/V projections, then still runs attention, MLP
and LM head. R4 does not claim that HIT skips the entire prefill or substitutes
saved logits/text. Its dynamic metadata adapter calls the unchanged original
ProjectionSession with a per-forward length view. Its independent new CPU UD
uses the unchanged `cpu_deltas` and safe original Channel framing. CUDA is
hidden on UD; actual original FP32 A/B are loaded once, with no full model.
Each recompute pass has 32 HIDDEN and 32 bundled DELTAS frames.

The ES retains a bare FP16 OPT-2.7B; no A/B matrices or LoRA computation run
there. Prepared Native reference and actual adapter weights come from the
unchanged original 95 preparation. A separate Native worker restores every
original matrix into native PEFT, verifies full base/matrix fingerprints,
records matched references, then exits before the bare ES starts. UUID,
versions, revisions, token IDs, adapter identity and source SHA gates apply.

R4 uses direct unshaped 127.0.0.1 TCP. R3-C's proxy has a 2 MiB payload cap,
whereas a bundled L128 FP32 delta is 3.75 MiB. The original Channel supports
this bounded payload; no frozen proxy limit is changed. No remote-link,
bandwidth-equivalence, CacheGen or natural semantic-hit claim is made.

## Generation and cache semantics

All modes use normal in-request HF past_key_values. Prefill selects the first
new token; each later forward consumes only the preceding generated token.
G output tokens therefore require one prefill and G-1 decode forwards, not G
decode forwards or repeatedly recomputed growing prompts. Actual HF KV output
is inspected for all 32 layers at every step. Unsupported/static layouts,
missing KV, wrong lengths/dtypes/positions or malformed frames fail closed.

Direct forward calls use eager attention, eval mode, use_cache=True and
output_attentions=True. Optional position_ids/cache_position are supplied only
when the loaded OPT API explicitly accepts them; otherwise OPT derives learned
positions from the full attention mask and past length. The mask/position rule
matches [HF OPT's positional embedding implementation](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/opt/modeling_opt.py).
The actual loaded version/API, rather than that example version, governs the
runtime. The primary controlled workload uses no padding: L32 and L128 mean
exactly 32 and 128 real source-token IDs, respectively, with all-one masks.
Tokenization disables automatic BOS/EOS insertion, uses right truncation,
and validates the retained IDs against the full source prefix. Sources are
four deterministic, distinct multi-paragraph texts, each checked after actual
tokenization to contain enough real tokens. A short source, pad/special ID,
non-one mask, wrong count, or collapsed prompt identity fails preparation.
The suite records source-token counts, retained real counts and zero pad counts;
the same contract is checked when loading prepared artifacts. Left-padded inputs
remain supported by the diagnostic/generation helpers and regression controls.

Each actual forward passively observes the OPT positional embedding module's
mask, past length and optional position IDs, then checks its returned learned
embeddings against the expected lookup (including OPT's offset). The hook
returns None and is removed on success or exception. Input and output actual
KV tensors are checked for every layer's shape, dtype, device and length;
cache-reported sequence length is also checked when that API is available.
Both the logical nonpadding positions and padding-inclusive cache slots are
recorded. These observations add instrumented serving overhead to all modes.

Greedy selection, BOS/EOS/pad and max-new-token policy are immutable. Fixed
latency trials explicitly ignore EOS. Separately requested natural-eos runs
stop at EOS and are quality diagnostics, not fixed-output latency claims.
Only newly generated IDs are decoded; the prompt is never echoed into BLEU.

NO_REUSE bypasses all cross-request cache lookup and insertion. It still uses
normal KV for its own decode. It is not charged namespace binding/validation.
Cold requests bind/reset the measured namespace, recompute the prefix, decode
normally, and admit owned total QKV only after the entire generation succeeds.
Cold insertion is within E2E finalization; it does not delay the first token
when max-new-tokens is greater than one. WARM_HIT validates all full-prompt
identities/content and physically reuses 96 prefill QKV with zero prefill data
frames/UD LoRA. It obtains fresh actual HF KV from that full prefill forward,
then recomputes every decode token using CPU UD and TCP. Warm generation thus
has decode traffic; it does not have zero total-request communication.

The new single-active-prompt adapter uses the genuine R3-B ExactRawCache,
CacheEntry, GlobalCache, ExactTokenMatcher and admission machinery. Changing
prompt namespace explicitly evicts its previous entry. Full IDs, masks,
positions, model, adapter, dtype/device/shape, completion and source provenance
are sealed. Only exact full-prompt replay is authorized. No semantic/window,
partial-prefix, SemCacheEngine.query or cross-request KV reuse is implemented.

## Stage 0 and correctness gates

Unmeasured native and distributed triage separately run identical original
inputs with use_cache=False and use_cache=True. Native False is compared with
the unchanged original 95 logits. Passive bounded captures include QKV,
attention output, fc2/FFN output and decoder-layer output plus actual inputs
at every layer. First output divergence and same-input status are reported;
later changed-input contrasts are consequences, not independent causes.
Native layer0 Q/K/V probes reuse the existing same-hidden reconstruction/CPU
delta diagnostic. They never replace Native outputs. No precision sweep or
tolerance changes occur. Captures are outside request performance timing.

Strict Native failure is independent of physical Full-vs-reuse fidelity.
Each measured/warmup triad must have identical generated IDs, text, serialized
next-token logits and all per-step 96-QKV hashes. Failure stops subsequent
triads and expansion. Native equivalence additionally requires both strict
triage gates and matched-history generation checks using original THRESHOLDS.
Argmax agreement or BLEU alone cannot pass Native strict parity. No numerical
root cause is asserted solely from tensor hashes or low last-token KL.

## Bounded workload and timing

Default prepare scope is pilot: four distinct declared texts, L32, G8, two
repeat blocks (101/202), all three modes. One warmup triad plus 24 measured
requests gives 27 generation requests. The launcher explicitly requests matrix
scope: (L32,G8), (L32,G16), (L128,G8), (L128,G16), at most 108 requests and
1296 generation forward passes, plus two unmeasured triage prefills. Native
references for requested cells are prepared beforehand in the isolated worker;
only distributed measurement expands after the pilot fidelity gate.

Before expansion, observed GPU peak must be <=20 GiB, free >=2 GiB, and the
declared pilot-time admission estimate must fit 7200 seconds. It uses the
declared sum of prompt+decode token-work relative to the pilot plus a 900 s
allowance; this is a conservative resource assumption, not performance evidence. Resource-gated
omissions are explicitly recorded, never imputed. OOM/unsupported GPU fails.
Three-hour allocation, 7500-second ES worker deadline and bounded prompt/output
caps prevent unbounded sweeps. Every triad contains its own real Cold fill.
NO_REUSE position is counterbalanced while Cold always precedes Warm.

Request start precedes actual cache work. TTFT ends at first selected token-ID
availability. E2E ends at final response finalization, including actual cold
admission and text decoding. Tokenization/model loading/handshake are outside.
Per-step copies, full next-vector/QKV evidence hashing and selection are inside
these instrumented serving walls on every mode.
Completed-request/vector exports happen outside; the per-forward diagnostic
JSONL is written inside the wall so a failed request retains its evidence.
Only one last-token vector per step is held on CPU (bounded <=16 vectors), not
all-token-per-step GPU logits. Per-position NaN/+Inf/-Inf integer reductions
are copied for diagnostics, bounded to 128 rows; no full vocabulary tensor is
copied for this check. Diagnostic persistence is inside the instrumented wall.
This is instrumented sequential serving, not
concurrent throughput or kernel-only time. CUDA events include host/socket
idle and are not added to host/UD/transport clocks.

Report per-cell mean/median/p90, raw paired sample/block contrasts and a seeded
10000-resample whole-block percentile interval. Two blocks provide weak
conditional uncertainty, not independent deployments. A benefit flag requires
real GPU execution, fixed matched output lengths, fidelity, >=8 pairs/two
blocks, upper interval below zero and every paired difference favorable.
TTFT and E2E flags are separate; decode traffic can erase an E2E advantage.
Cold+one-resident cost is separately amortized across the two served requests.

## BLEU and future metrics

The existing `evaluation.bleu.compute_bleu` is used with the C6 convention:
SacreBLEU corpus, 13a, mixed case, exponential smoothing, effective_order=False,
one aligned reference corpus, 0–100 scale. Version and full signature are saved.
CacheGen experiment/codec modules are not imported. Model/corpus/decoding/reference
differences prevent numerical comparability claims with C6/C8 or INFOCOM results.

`bleu_vs_distributed_full_recompute` is output fidelity; exact IDs/text,
positional agreement and changed-output count are also reported.
`bleu_vs_native_peft` needs actually computed matched Native generations and
remains separate from strict Native parity. Gold scores require external JSONL
records with sample_id, prompt_sha256, reference_text, dataset and revision.
Without these, status is NOT_ESTIMABLE_NO_GOLD_REFERENCES, never zero or a
model-generated pseudo-label. The adapter is untrained, so even supplied gold
cannot establish trained/personalized task capability. Missing SacreBLEU is
MISSING_DEPENDENCY; no installation or fake score occurs. GPU preflight requires
the package, while CPU fixtures may finish with the metric explicitly missing.

The optional quality API accepts a dataset_metric callable over aligned output
and external-gold records; future LongBench/SCBench adapters must record the
original dataset/paper metric (QA EM/F1, summary ROUGE-L etc.), implementation
and version alongside BLEU. Open-ended QA BLEU is not an accuracy proof.

## Commands and artifacts

```bash
cd /data/khuss/repos/GP_semcache
mkdir -p logs
bash -n jobs/r4a_opt_full_e2e.sh
sbatch jobs/r4a_opt_full_e2e.sh
```

Launcher uses one GPU, four CPUs, 64 GiB, batch_eebme_ugrad, three hours, no node
pin, conda semcache (optional R4A_CONDA_ROOT), offline HF cache
/data/khuss/huggingface and unique results/r4/a/opt_full_e2e_<job>_<UTC>.
R4A_SCOPE=pilot restricts it to the conservative pilot. It runs unchanged 95
prepare, new matched Native preparation, physical generation, then independent
analysis. Native FAIL, no benefit or unresolved contrasts do not fail a valid
diagnostic; cache/identity/transport/KV/lifecycle failure does.

```bash
python scripts/102_run_r4a_opt_full_e2e.py --phase prepare --enable-gpu-experiment \
  --base-prepared <95_prepare> --scope matrix --generation-policy fixed --output-dir <new_R4_prepare>
python scripts/102_run_r4a_opt_full_e2e.py --phase run --enable-gpu-experiment \
  --prepared <new_R4_prepare>/native --output-dir <new_R4_run>
python scripts/102_run_r4a_opt_full_e2e.py --phase analyze \
  --inputs <new_R4_run> --output-dir <new_R4_analysis>
```

All outputs must be new/empty below results/r4/a and outside inputs. Core files:
r4a_results.json, r4a_summary.txt, r4a_requests.jsonl, r4a_tokens.jsonl,
r4a_generation_outputs.jsonl, r4a_quality.json, r4a_sacrebleu_signatures.json,
r4a_parity_diagnostics.json, r4a_latency_contrasts.csv, r4a_decision.json,
r4a_preflight.json, r4a_audit.json. Native/ES/UD raw vectors, component captures,
frame hashes, timing, process stdout/stderr, identities and lifecycle are retained.
Failed runs keep partial evidence and do not enter latency/benefit analysis.

`native/r4a_logit_diagnostics.jsonl` (or fixture `prepared/`) and
`es/r4a_logit_diagnostics.jsonl` persist one bounded record for every generation
forward, **before** the finite guard and token selection. Records contain
sample/request IDs, zero-based generation step (0 = prefill), total input/context
length, current query length, real and pad counts, full mask, positions, actual
input/output KV summaries and positional observations. `all_logits`,
`last_position_logits`, `valid_position_logits` and `padded_position_logits`
each report elements, finite/nonfinite, NaN, positive Inf and negative Inf
counts. Group masks describe the returned query positions: on decode, cached
padding contributes to the context pad count but not the one valid query row.
At most 1500 records per worker are allowed, covering the declared experiment.
No nonfinite value is masked, clamped, replaced, serialized as a valid logit or
ignored, including a NaN confined to padding. Any such output still aborts the
request and blocks completed-reference/cold-admission/performance claims.
Successful serving audits require these records to match the per-token ledger.

## Pilot 151991 failure attribution

The user supplied these original L32 observations; they are preserved here as
reported evidence, not a local tokenizer reproduction:

| Original sample | Real tokens | Left pad tokens |
| --- | ---: | ---: |
| text_0 | 17 | 15 |
| text_1 | 17 | 15 |
| text_2 | 16 | 16 |
| text_3 | 15 | 17 |

**Confirmed workload defect:** the previous code tokenized short sentences with
`padding='max_length'`. Its declared length was a padded slot budget rather
than a real-token budget. The exact previous source sentences remain available
as `FAILED_PADDED_PILOT_TEXTS`; existing results/logs are not rewritten or reused
as new experiment output. Fresh v2 preparation is required.

**Confirmed diagnostic limitation:** `last_logits()` checked *all* model logits
before selecting the final-token vector, and gave no sample/step/group counts.
The exception establishes that some model logits were nonfinite; it does not
establish that final-token logits were nonfinite or identify prefill versus
decode. Both original Native R3-A-input triages completed, according to the
user's report; those unpadded controls do not reproduce the short padded prompts.
The historical failing sample ID, step and nonfinite counts cannot be recovered
from this exception. No raw Pilot 151991 files are present in this checkout.

**Unconfirmed numerical cause:** fully masked left-padding query rows and FP16
attention arithmetic are plausible candidates; autoregressive cache/position
or later decode arithmetic remains possible. The exact loaded Transformers
implementation matters. For example, the pinned
[HF OPT reference implementation](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/opt/modeling_opt.py)
adds its causal/padding bias before softmax and derives logical positions from
the full mask and past length; this source inspection is not a reproduction
of the SERAPH runtime. No claim that left padding caused this NaN/Inf failure
is justified without per-step runtime evidence or a matched padded/unpadded
comparison on that model/runtime. The primary pilot now removes padding as a
workload confounder and records any nonfinite outputs without changing the
strict Native PEFT or distributed fidelity gates.

CPU `--phase fixture --test-fixture-only` uses an explicit tiny scalar causal
decoder with actual KV storage and TCP: L32, G3, two texts/two blocks, 15
generation requests plus two triage prefills. It is not OPT/PEFT/CUDA evidence;
all real R4 pass/benefit flags stay false. Optional GPU tests require explicit
opt-in and an allocation. No SERAPH access or job submission occurs locally.

Local CPU coverage includes finite padded/unpadded last-vector selection,
padding-only/valid/final NaN/Inf rejection, a successful prefill followed by a
fatal decode with durable partial diagnostics, multi-step cached versus full
prefix math, exact real-token workload validation, and the physical TCP fixture
with independent audit/reanalysis. An optional random-weight actual HF OPT CPU
test checks padded/unpadded prefill plus two decode steps and observed position
lookups without downloading a model. It is explicitly skipped when Torch or
Transformers is absent. The local environment also lacks NumPy, PEFT and
SacreBLEU. Actual OPT-2.7B FP16/GPU, Native PEFT, the original failing sample,
and a matched GPU left-padding causal comparison remain NOT_RUN. CPU controls
validate plumbing and fail-closed behavior, not the numerical cause of Pilot
151991 or GPU benefit. No Slurm submission or git push is part of this fix.
