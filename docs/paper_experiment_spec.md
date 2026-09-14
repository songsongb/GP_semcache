# M6A paper experiment specification

**No paper results are reproduced by this harness.** The supplied SemCache PDF,
DOI 10.1109/INFOCOM59046.2026.11571717, is the primary specification. Eqs. 15–21,
Table I, §V-A, Table IV and figure labels were checked against its extracted text.
Paper references are comparison data only; no fitting or calibration is performed.

## PAPER-DEFINED

| Setting | Value |
|---|---|
| ES | NVIDIA A100 80GB; Intel Xeon Gold 6338 |
| Each UD | 4 CPU cores, 2.3 GHz, 8GB RAM |
| ES–UD link | 200 Mbps |
| Models | GPT2-Large FP32; LLaMA-7B INT4; OPT-6.7B FP16 |
| LoRA | HF PEFT; Q/K/V targets; rank 8 |
| Users | 50 |
| Intent clusters | MultiWOZ 20; CoQA 40; SNIPS 30 |
| Window; logical cache | 3; 20 GB |
| Admission | alpha=.5, beta=.3, delta=.2; strict score > .3 |
| Eviction | alpha=.4, beta=.3, gamma=.2, delta=.1 |
| Impact | rho=.8, Lambda=100 |
| Cluster updates | Every 100 queries |

Dataset configs explicitly inherit `common.yaml` through the M6 loader. The
legacy M1–M5 config loader is unchanged. The runner accepts explicit cluster
counts for later sweeps; no automatic reduction when warmup has fewer than C
examples. No sweep is executed in M6A.

## PAPER-UNCLEAR — unresolved, not hidden defaults

- Exact adjusted dataset queries, dataset release/split and personalized task construction.
- Exact per-user partition, query rewriting and user/query order.
- LoRA training data, optimizer, learning rate, epochs, scaling and dropout.
- Exact TinyBERT checkpoint, immutable revision and embedding bridge in Eq. 7.
- Generation length, sampling, EOS behavior and seeds.
- BLEU package/version, tokenization, smoothing and aggregation protocol.
- Paper hit-rate denominator; `reported_hit_rate_definition` remains null.
- Activation QKV and network element precision, especially for INT4 weight models.
- Effective FLOP/s, hardware utilization, overheads, scheduling and system-memory accounting.
- Fig. 8 labels bandwidth “Bps”; setup and narrative explicitly say 200 Mbps.
  Figure labels 200M/500M/1G are retained but resolved sweep units remain null.

These unknowns prevent direct Table II quality/performance comparison. Paper
Table II memory is not equivalent to CUDA allocated memory. No hidden fixed
query count or chosen encoder checkpoint is introduced.

## REPRODUCTION-CHOICE

All config leaves have a provenance ledger in each run manifest. Overrides of
known paper defaults become REPRODUCTION_CHOICE. Unspecified/null config fields
remain explicit unresolved choices. Workload source, transformation, SHA-based
assignment/order, seed 42, decimal GB, exact token matching, greedy overlap,
first-C warmup and the inherited M5 normalization decisions are choices.

Execution modes are mandatory:

- `MEASURED_MODEL`: actual local model prefill; instrumented wall time is MEASURED.
- `ANALYTICAL_SIMULATION`: logical cache events and optional equation timing, SIMULATED.
- `HYBRID`: actual model/cache behavior plus analytical system timing; sources stay separate.

Scopes are `prefill_only` or `full_generation`. The latter currently raises:
M5 decode lookup is an interface, not a complete autoregressive loop. The Python
runner accepts a provisioned M5 physical engine for measured/hybrid subsets;
script 23 deliberately runs logical-only simulation. `physical_cpu` requires
CPU tensor cache storage; `physical_model` accepts a real model engine and its
explicit storage device. There is no automatic weight loading or quantization.
OPT-125m on RTX A2000 cannot directly reproduce OPT-6.7B FP16/A100 results.

Baseline interfaces: UD_ONLY means full UD inference; ES_ONLY means base and
user LoRA on ES; FBC means frequency cache with LRU replacement; SEMCACHE means
EdgeLoRA plus semantic-aware global QKV caching. Only SEMCACHE is executable
here. FBC admission/frequency details are unresolved; other baselines raise.

## Authoritative analytical model and units

`simulation/cost_model.py` owns Eqs. 15–20; the M5 savings fields delegate to it.
For one layer, n input tokens, n' reused, hidden width d and rank r:

```
saved base FLOPs = 6 n' d²
saved LoRA FLOPs = 6 n' d r
saved communication elements = 4 n' d
no-cache communication elements = 4 n d
rest ES FLOPs = 18 n d² + 4 n² d + 16 n d
L' = max(6(n-n')d²/f_ES,
         6(n-n')dr/f_UD + 4(n-n')d * element_bytes * 8 / B)
     + rest_ES_FLOPs/f_ES
```

f_ES and f_UD are FLOP/s, B is bit/s. 1 TFLOP/s=10^12 FLOP/s;
1 Mbps=10^6 bit/s; one byte=8 bits. Equation D counts elements, not bytes.
Wire precision must be supplied separately. Eqs. 18/19 do not remove attention
or FFN computation for reused tokens. Whole-model estimates sum this per-layer
expression over L; they omit encoder, lookup, input/output layers, queueing,
protocol overhead and generation. They are not full measured query latency.
The equation is the paper abstraction, not architecture-specific FFN profiling.

The 512/300=1.7066667 ratio is Eq. 20's bottleneck approximation, not the ratio
of full Eq. 19 timings and not measured speedup. Script 22 uses explicitly named
fixture rates of 1 TFLOP/s ES and .001 TFLOP/s UD, FP16 wire elements; these
are hand-calculation fixtures, not inferred A100/Xeon capabilities.

## Memory and model metadata

Logical block payload bytes are `ceil(L*w*(d_Q+2*d_KV)*bits_QKV/8)`.
Metadata, packing metadata and allocator overhead are excluded. INT4 weights
do not imply INT4 activations. Paper activation precision remains null; script
23 explicitly declares a development FP16 payload choice in its CLI/help and
manifest. No 20GB tensor is allocated. OPT-6.7B simulation uses its own 32-layer,
4096-wide metadata, never OPT-125m physical block sizes.

Model YAMLs contain hidden size, KV width, layers, weight precision, rank,
checkpoint/revision placeholders and local execution availability. OPT's
4096/32 dimensions are confirmed by its
[official config](https://huggingface.co/facebook/opt-6.7b/raw/main/config.json).
GPT2-Large's explicit 1280/36 and LLaMA v1 7B's explicit 4096/32 are recorded as
REPRODUCTION_CHOICE architecture inputs pending pinned authoritative snapshots;
no exact LLaMA paper checkpoint is asserted. These entries describe simulation
metadata, not loaded models or available large-model execution.

Metrics separately record logical cache payload, physical tensor storage, GPU
allocated/reserved bytes, base parameters, all loaded adapters and analytical
system-memory estimate. Unimplemented/unobserved quantities are null. Each has
its own scope and source. No analytical system memory is fabricated.

## References, aggregation and quality

`experiments/paper_reference.py` contains deeply read-only Table II/Fig. 6
mappings marked PAPER_REFERENCE. Execution modules never import it. The cost
model accepts validated dimensional/rate scalars, not reference result objects.
No reference latency is used to choose rates or cache policy.

Aggregation preserves source/scope/unit and rejects mixed inputs. Counters and
savings sum; occupancy/latency/quality average; latency p50/p95 interpolate sorted
samples. Ratios pool raw numerators/denominators. Generic typed quality fields
can be aggregated when available; corpus BLEU should be computed once over the
aligned corpus, not averaged over sentence BLEU. The optional SacreBLEU interface
requires every protocol field, reports package version/signature and never
claims the unspecified paper protocol. It does not generate answers.

Comparison requires complete contexts. Model/precision/execution-scope/memory
scope/unit mismatches cannot be overridden. Other documented differences may
be APPROXIMATE; incomplete metadata is NOT_COMPARABLE. Percent differences are
null for NOT_COMPARABLE, including OPT-125m/A2000 versus OPT-6.7B/A100.

Sweep specs retain C=[10,20,30,40,50], Table IV w=[1,2,3,4,5], Fig.7
cache=[5,10,20,40] GB and admission=[.1,.2,.3,.4,.5], users=[10,50,100,200],
ranks=[4,8,16], Lambda=[100,500,1000], approximate query lengths=[50,100,200,400].
Network sweep units remain PAPER_UNCLEAR. The old `configs/sweeps/planned.yaml`
is historical; use `configs/paper/sweeps.yaml` for M6.
