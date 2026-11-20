# Assumptions and limitations

All changed-bandwidth/throughput results are MODELED, never GPU observations.
Official receipt verification is required in receipts mode. Summary mode checks
attachment consistency only; it does not reissue LIGHT integrity or CPU approval.
Missing wire bytes/LoRA wall costs remain NOT_MEASURED and dependent predictions
are NOT_ESTIMABLE. TCP receive wait and generic exclusive profile times are never
used as LoRA compute; planned wire bounds and skip ratios never replace wire data.

T(B,f) = T0 + exposure * [0.008 * effective_bytes / B_Mbps
                         + measured_cpu_deltas_wall_ms * (1/f - 1)].
Unlimited has zero incremental pacing. T0 already includes baseline loopback,
serialization/hash, observer and ES computation; none is subtracted or rescaled.
This models extra pacing, not replacement of measured socket wait. The baseline
must be unlimited/factor1/no separate RTT. No GPU/model/semantic/quality term is
slowed by the UD factor. UDServiceEmulator executes the same LoRA function once,
measures each call, and targets cost/factor with observable sleep. Hardware
equivalence is NOT_ESTABLISHED; CPU wall cost includes function checks/stacking.

SERIAL_BLOCKING_REFERENCE sums both directions: the actual sparse RPC receives
the complete hidden frame before LoRA and waits for deltas before native Linear.
BytePacer reserves <=64KiB chunks before sending with no idle credit; ideal total
reserved time is bytes/rate, but OS oversleep, partial sends/refunds, buffering,
framing work, frame-size scheduling and contention are not reconstructed from
aggregate receipts. PARTIAL_EXPOSURE_50PCT_ASSUMED uses the same assumed exposed
fraction in every condition. DUPLEX_MAX_STRESS_ASSUMED uses max(direction bytes)
as a hypothetical overlap stress case, not the observed request/response schedule.
These scenarios are assumptions, not calibrated center/confidence bounds or a
guarantee that actual E2E lies within the range. RTT is not silently introduced.

Wire bytes mean application header/framing/payload bytes, excluding TCP/IP packet
headers/retransmissions. STOP/BYE CONTROL bytes are included once, assigned to
the last request of their adapter, where existing runtime performs shutdown.
Endpoint socket times are inclusive and mirrored; never sum them as E2E or CPU.
Per-request phase wire/compute totals are not retained by existing summaries.
TTFT at changed knobs has no point estimate: only explicit 0..all-extra-delay
allocation bounds are emitted where E2E can be modeled. These are conditional
model bounds, not phase measurements or rigorous real-system bounds.

Session E2E is sum(request E2E), matching the official comparison; it is not
initialization/sealing-inclusive run wall or request sequence span. Cycles use
authenticated cycle_index, not an assumed split at request10. Adapter groupings
retain routing. Predictions freeze tokens/cache decisions/quality and cannot
model deadline-induced aborts, changed contention, thermals or CPU clock effects.
Aggregate worker bounds are flagged; initialization and per-frame timeout
feasibility remain NOT_ESTABLISHED. No run is automatically launched.

Analytic roots are MODELED crossings, never observed break-even. Equality is not
Speedup>1. Nonpositive savings can yield no crossing or the opposite win region.
Below50Mbps / factors other than1,.5,.25 are outside existing runtime knobs even
if requested as model search points. The extreme UD bound assumes all FULL E2E
is LoRA and cache LoRA is zero; it is ASSUMED, not a measurement. Optional
hypothetical compute-fraction grids are separate ASSUMED_MODELED tables and are
never merged into measured extractions. Input summary rounding is retained.

APPROX remains a separate fidelity-unqualified curve; unrelated quality metrics
are not pooled. Receipt integrity and token agreement do not establish independent
full-vocabulary numerical fidelity. correctness_qualified_speedup=NOT_ESTABLISHED.
