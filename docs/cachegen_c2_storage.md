# C2: frozen K20/V16 physical cache storage

The existing physical cache is named `GlobalCache` in this repository. Its
`CacheEntry` stores per-layer `(Q,K,V)` detached tensors. For an OPT-2.7B
`w=3` FP16 entry, each projection has shape `[1,3,2560]` per layer across
32 layers. Raw Q, K, and V each occupy 491,520 bytes, so K+V occupy 983,040
bytes and the whole entry occupies 1,474,560 bytes. The engine defaults to
CPU cache storage; the projection hook produces tensors on the model device.
Logical capacity and eviction continue to use the raw `size_bytes` field,
not compressed physical bytes. Existing cache entries are not serialized.

`GlobalCache` now accepts `physical_storage_mode='RAW_FP16'` (default) or
`physical_storage_mode='COMPRESSED_KV_K20_V16'` with an injected
`FrozenK20V16Codec`. The codec accepts configurable `profile_path`,
`quantization_device`, `decode_device`, and instrumentation settings. This
keeps production cache code independent of the CacheGen experiment package.
The compressed mode stores Q as detached FP16 tensors
on the configured cache storage device and stores K/V as one CPU byte string
containing the existing B2 frame. The frame includes the shape and FP32
maxabs metadata. K20/V16 symbols use the released C1.5C formula and the
frozen four-role `B2_ANCHOR_MOD_RESIDUAL_KV` profile. No CDF is fitted at
runtime. The profile file must match SHA256
`8defa27a83e8ad16e9c7efa0f40e68c302aef90363e14a1451288a3bf433cb2c`.

On lookup, `GlobalCache` returns a temporary decoded view with the same key,
metadata, and logical resident entry. This view supplies `(Q,K,V)` tensors to
the existing mixed-projection path. The resident entry retains only Q and the
encoded byte frame. Reuse frequency updates apply to the resident entry.
`CacheEntry.storage_accounting` reports raw Q/K/V bytes, actual stored Q
and K/V bytes, and compression ratios; the shared profile is charged once
per cache/workload, never per entry. Optional timing records contain
quantize, encode, insert, decode, dequantize, and lookup milliseconds.

Run the eight-block, no-inference smoke manually on SERAPH from
`/data/khuss/repos/GP_semcache`:

```bash
python scripts/44_cachegen_c2_storage_smoke.py \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --profile-path results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin \
  --output-dir results/cachegen/c2/storage_smoke \
  --per-dataset 4 --seed 42
```

It selects the first four sorted evaluation `w=3` blocks from each of
MultiWOZ and SNIPS. The C1 fixtures contain real Q alongside K/V. The smoke
uses one fixture read per block, one cache insert and lookup, and direct
quantization only to check symbol and reconstruction equality. It writes
`manifest.json`, `storage_summary.json`, and compact `per_block.csv` under
`results/cachegen/c2/storage_smoke/`; no model is loaded and no payload
files are written. The expected runtime is minutes rather than 30 minutes,
subject to SERAPH filesystem and arithmetic coder throughput. This command
has not been run locally against real captures.
