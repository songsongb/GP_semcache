# Qualitative SemCache sweeps

These are qualitative sensitivity experiments using the current
reproduction-choice SemCache implementation.
They are not full reproduction of SemCache's paper results because
actual TinyBERT semantic embeddings and attention-based CHU/PBR are
not yet integrated into the full logical workload runner.

All cache results are `SIMULATED` / `ANALYTICAL_SIMULATION`.
The semantic frontend is the existing deterministic fixture encoder; impact
is unavailable and CHU/PBR are inactive. `safe_reuse_claimed` remains false.
Matching, admission, eviction, user assignment and workload ordering are unchanged.
No parameters are tuned against paper outcomes.

## What each sweep tests

- **Cache size:** capacity sensitivity. Only `logical_cache_capacity_gb` changes.
  GB means decimal 1e9 bytes. Peak bytes means maximum resident occupancy after
  eviction on each insertion, excluding transient insertion overflow.
- **Admission threshold:** admission selectivity / cache-quality tradeoff.
  Only `admission.threshold` changes; alpha=.5, beta=.3, delta=.2 are required,
  and eviction weights stay unchanged. Observe occupancy, admission rate,
  evictions, hits and token reuse together; monotonic improvement is not assumed.
- **Bandwidth:** communication sensitivity, not cache-policy quality. One
  SEMCACHE simulation supplies decisions for every point. Remaining communication
  elements equal total minus saved elements and do not vary with bandwidth.

The base configuration, per-leaf provenance ledger, parameter source for each
point, effective config hashes, model specification and workload hashes are saved.
20 GB, theta=.3 and 200 Mbps match paper-defined defaults in the existing ledger;
other values and the experiment design are reproduction choices. The inherited
`semantic_encoder.family: TinyBERT` configuration describes the intended family,
not the actual executed encoder; explicit availability fields and
`reproduction_choices.semantic_encoder` describe execution.
As in script 24, null QKV storage precision resolves to 16 bits, tagged as a
reproduction choice. It is **not** communication wire precision.

## Full SNIPS commands (run manually)

From the repository root with its Python environment activated:

```bash
python scripts/25_run_semcache_sweeps.py --workload results/workloads/snips.jsonl --config configs/paper/snips.yaml --sweep cache_size --values 5 10 15 20 --max-queries 13784 --seed 42 --compact --output results/sweeps/snips_cache_size.json
python scripts/25_run_semcache_sweeps.py --workload results/workloads/snips.jsonl --config configs/paper/snips.yaml --sweep admission_threshold --values 0.1 0.2 0.3 0.4 0.5 --max-queries 13784 --seed 42 --compact --output results/sweeps/snips_theta.json
python scripts/25_run_semcache_sweeps.py --workload results/workloads/snips.jsonl --config configs/paper/snips.yaml --sweep bandwidth --values 200 500 1000 --max-queries 13784 --seed 42 --compact --output results/sweeps/snips_bandwidth.json

python scripts/26_plot_semcache_sweeps.py --input results/sweeps/snips_cache_size.json --output results/sweeps/snips_cache_size.png
python scripts/26_plot_semcache_sweeps.py --input results/sweeps/snips_theta.json --output results/sweeps/snips_theta.png
python scripts/26_plot_semcache_sweeps.py --input results/sweeps/snips_bandwidth.json --output results/sweeps/snips_bandwidth.png
```

Plotting requires matplotlib (`pip install matplotlib`); no seaborn is used.
For MultiWOZ or CoQA substitute the workload, dataset config and query limit.
Capacity and threshold jobs run SEMCACHE only (four and five simulations).
Bandwidth runs one simulation. Other baselines are never invoked.
Script 24 also accepts `--baseline SEMCACHE --compact`.

## Timing and output

To enable communication timing, explicitly set
`system.communication_element_bytes` in a separate configuration, for example 2.
The conversion is `elements * bytes_per_element * 8 / (Mbps * 1e6)`.
This is analytical timing under a reproduction choice, not measured wall-clock
latency. With null wire precision, communication time stays null. The plot then
shows constant communication volume, not an invented bandwidth-dependent cost.
Total latency always stays null in this communication-only sweep framework,
with `latency_available=false` and a reason; missing ES/UD effective throughput
is never inferred. It does not calculate full Eq.19 latency even if supplied.

Compact JSON contains shared metadata and one aggregate per point, with no query
IDs, user-ID arrays or per-query results. Expected size is roughly 10–25 KB per
sweep (dependent on configuration), essentially independent of query count.
Without `--compact`, query summaries are included for debugging. Compact mode
also discards accumulated engine events between queries; internal query summaries
remain available to the existing accounting code. Workload loading and fixture
initialization retain their existing memory requirements.

Outputs are deterministic for identical workload/config/model, seed and code;
there are no timestamps in sweep JSON. Record the repository revision alongside
meeting artifacts. Fairness checks reuse script 24's workload hash, identity,
ordering, user assignment and model-rank validation.
