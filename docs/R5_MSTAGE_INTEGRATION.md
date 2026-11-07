# M-stage / R5 integration audit

PARTIAL_IMPLEMENTATION: opt-in metadata/policy integration; physical subsequence
executor NOT_IMPLEMENTED. GPU_NOT_RUN. Phase2 E2E NOT_READY. No production
forward, policy, transport, threshold or historical result was changed.

| Mechanism | Existing M-stage | Existing R5 | This integration / gap |
|---|---|---|---|
| TinyBERT, clusters | encoder + IntentClusterer | SemanticIndex uses those classes | Reuse actual R5 assignment; no second encoding |
| Exact w3 | SubsequenceExtractor / ExactTokenMatcher | candidate discovery already connected | Direct M reuse, occurrence positions retained |
| Nonoverlap | select_nonoverlapping | full-source prefix selection | M selector reused for shadow spans |
| Admission F/I/S | AdmissionPolicy + normalized population | LRU complete-source fit | M GlobalCache directly reused, metadata only |
| Eviction F/A/I/S | EvictionPolicy | LRU | M score reused;512MiB/adapter and64-entry bounds |
| CHU | actual-reuse frequency + EMA impact | actual attention probabilities not exposed | Formula preview only; no false frequency increment |
| PBR | QueryImpactHistory / CacheMetricManager | no connected probability history | Actual supplied attention observations only; missing data is not zero |
| Total QKV reuse | PEFT mixed_projection_path | bare Linear + CPU UD corrections | Incompatible validators, ownership and wire schemas |
| Attention / KV | full prefill after projection assembly | complete prefix KV + suffix forward | Arbitrary span KV cannot substitute causal-context KV |
| Evidence | passive projection audit | independent raw/KV/logit/TCP audits | No sparse consumption proof/schema yet |

## What is connected

`r5_mstage_bridge.shadow_controller_factory(runtime, trace=...)` is an explicit,
process-local opt-in context. It restores the original factory in `finally`.
Existing A and Exact-Prefix C calls stay on the original code. B additionally
observes actual semantic assignments and checks metadata spans in per-physical-
identity/adapter M caches. Original B results retain their original fields.
Additional callbacks are evidence-inclusive overhead, not performance-free.

The standalone `MStageShadow` API validates R5 request schemas and physical
identity, original30-user authorization and source completion. It uses exact
M-stage keys, nonoverlap selection, policy normalization, frequency history,
impact reducers, CHU formula and PBR. It stores **no tensors**. All events say
physical_hit=false, skipped_projection_rows=0, skipped_wire_bytes=0. Lookup
counts and candidate rows cannot be interpreted as GPU reuse. Eviction is local
to the metadata cache, not an R5 tensor eviction or UD EVICT_PREFIX operation.
There is no new TTL; age uses the M query clock; disposal explicitly clears state.

`complete(completed=True, attentions=...)` accepts actual eager attention
probability tensors through the existing M reducer, never a guessed impact
scalar. The default R5 driver has `output_attentions=False`; its controller
receives no probability tensor. Consequently the connected shadow factory
records ACTUAL_ATTENTION_PROBABILITIES_NOT_RECORDED and does not admit entries.
CPU tests inject labeled fixture impact values or, when installed, actual Torch
probability tensors. They do not establish GPU impact capture. F remains zero
without executed reuse; appearance frequency for admission is separate.
`chu_preview` operates on a copy and is never an actual-use receipt. PBR uses
observed matching history only. Metadata policy tests are not an E2E ablation.

## Mathematical and architectural blockers

M mixed projection skips selected total Q/K/V Linear rows but executes attention,
normalization, FFN and LM head for the whole request. R5 prefix HIT additionally
skips those prefix forward rows and supplies actual past K/V. They are different
mechanisms. Never translate an M span candidate into an R5 prefix grant.

For layer L, total projection depends on h_L, base weights, CPU LoRA matrices and
cast/add order. Matching token IDs, even at equal positions, does not establish
equal h_L under changed causal history. Shifted matches also have OPT positional
embedding differences. Later-layer attention changes propagate through every
subsequent projection. Even equal causal input values can have FP16 shape-related
differences; the prior numerical FAIL remains unresolved. Source prefix K/V
integrity is not equivalence to target-context K/V.

The M executor validates PEFT Linear instances and locally owned total tensors.
R5 deliberately has bare GPU Linear and CPU-only LoRA matrices. Wrapping it as
PEFT, injecting GPU LoRA, or bypassing its validator would change the model split.
The existing M fresh_projection callback is useful algebraically, but its gathered
rows are incompatible with R5 DynamicSession/ForwardOrder's exact input_length,
contiguous positions, masks and all-layer/all-role order. Sending sparse rows
under an old header would violate the protocol. Keeping full CPU corrections
would not save those bytes and must not be labeled distributed total-QKV reuse.

No approximate mode is silently enabled: it requires a reviewed versioned sparse
transport and separate numerical/ownership audit. `physical_grant()` always fails.
This deliberately does not fulfill the requested C physical execution yet.

## Smallest next implementation gate (before any GPU allocation)

1. A separate sparse-projection protocol carrying real absolute gather/scatter
   positions, adapter/model hashes, executed roles, completion and exact wire
   payload accounting. Do not modify the existing prefix version in place.
2. A distributed adapter for M mixed projection over bare Linear + CPU LoRA,
   without installing ES LoRA parameters. Fresh rows use the new CPU endpoint;
   retrieved total QKV spans carry all-layer ownership and saved-byte proofs.
3. Either prove per-layer hidden-input/context equivalence, or predeclare
   APPROXIMATE_SUBSEQUENCE_TOTAL_QKV. Position/authorization guards remain
   mandatory. In approximate mode recompute full attention/FFN/LM head; never
   authorize arbitrary-span past-KV consumption or claim equivalence.
4. Passive, bounded actual attention probability capture for impact. Prefer the
   inspected eager operation observer to changing attention configuration;
   fail closed if probabilities/dropout boundaries cannot be validated. Include
   its reduction, synchronization and history cost inside application E2E.
5. Separate tensor owner removing index entries, proofs and CPU grants together.
   Independent audit must prove consumed span hashes, real fresh positions and
   sparse wire bytes; metadata admission is not tensor materialization.

Only after these gates prepare actual A/B/C with identical arrivals, native
Qasper/2Wiki F1 and MultiNews rouge==1.0.1 ROUGE-L, full logits and frozen
thresholds. C must be called SEMCACHE_SUBSEQUENCE, not Exact Prefix C. Preserve
natural/controlled labels. Include source materialization, impact, all policy
updates, semantic processing, admission and eviction in instrumented application
E2E and source-inclusive economics; report warm latency separately. No
correctness-qualified speedup if numerical fidelity fails.

## Manual CPU commands after user-managed review/commit/push/pull

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=''
PYTHONPATH=src:tests python3 -m unittest test_r5_mstage_bridge -q
PYTHONPATH=src:tests python3 -m unittest test_r5_mstage_bridge \
 test_r5_i5 test_r5_i5_controller test_r5_controller test_r5_semantic \
 test_r5_multi test_r5_i5r4 -q
export MS_SOURCE_COMMIT="$(git rev-parse HEAD)"
export MS_EXPERIMENT="$PWD/results/r5/i5r/parallel/mstage_$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/150_r5_mstage.py prepare --output-dir "$MS_EXPERIMENT"
bash jobs/r5_mstage_subsequence.sh
```

The last command is a CPU gate and intentionally exits2 with BLOCKED_NOT_GPU_READY.
It is **not** a submission-ready GPU script and never invokes119 or sbatch.
The sealed integration plan records sources, blockers and non-executable A/B/C
design. Do not allocate GPU resources for it. Existing149 controlled Exact Prefix
pilots remain independent and must not be reported as this subsequence mode.

## Local validation

Focused bridge:16 PASS /0 FAIL /1 SKIP (17 tests). Combined bridge + I5,
I5 controller, R5 controller/semantic/multi-user and I4 regressions:
151 PASS /0 FAIL /3 SKIP (154 tests; includes focused). Skips: real Torch
attention reduction, native OPT CPU integration and rouge==1.0.1; dependencies
are absent locally. Three Python AST checks, CLI help, `bash -n` and whitespace
checks passed. The connected factory was tested against the actual I5 B
controller over CPU fixtures: no physical tensors read/admitted; missing real
attention is explicitly reported. Existing baseline regressions passed.
No GPU execution, physical subsequence consumption, real-data quality/performance
measurement or native PEFT/GPU mixed projection validation was performed.
