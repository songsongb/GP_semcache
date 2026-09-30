# CacheGen HIT/MISS integration smoke test

`scripts/48_validate_cachegen_hit_miss_pipeline.py` runs exactly two requests,
with synthetic deterministic projections: 32 layers, three tokens, width eight.
No model downloads, datasets, engine changes, or experiment result writes occur.

The current `main` branch has the teammate's `encode_lora_delta` /
`decode_lora_delta` transport implementation. The existing storage implementation
is on local branch `exp/cachegen-c5`: `FrozenK20V16Codec.make_entry` /
`decode_entry` in `experiments/cachegen/c2/physical_storage.py`.

To expose that existing storage code without switching or merging branches,
export its source and frozen profile into a fresh temporary directory:

```bash
storage_checkout=$(mktemp -d /tmp/semcache-storage.XXXXXX)
git archive exp/cachegen-c5 src/semcache/experiments/cachegen results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin | tar -x -C "$storage_checkout"
CACHEGEN_LMCACHE_ROOT=/path/to/CacheGen/LMCache python3 scripts/48_validate_cachegen_hit_miss_pipeline.py \
  --storage-src "$storage_checkout/src" \
  --profile "$storage_checkout/results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin" \
  --device cuda:0
```

Use a Python environment with PyTorch, CUDA, a compatible legacy LMCache checkout,
and its prebuilt `torchac_cuda` extension. `CACHEGEN_LMCACHE_ROOT` points to the
directory containing `lmcache/`; the old transport path remains the fallback.
On a branch already containing storage code and its profile, omit `--storage-src`
and `--profile`. The script does not build or install dependencies.

MISS invokes the real transport codec separately for each Q/K/V projection,
simulates delivery in process, then passes decoded tensors to the real frozen
storage codec. Its entry factory retains only `CompressedKVPayload`; raw Q
copies created temporarily inside `make_entry` are discarded. HIT never invokes
transport, projects only Q, and supplies it in a temporary storage decode view.
Both requests feed finite reconstructed tensors into a tiny attention operation.
The resident schema, codec call counts, and routing records are checked.
Cosine similarity and relative L2 reuse `evaluation.qkv_metrics.similarity`,
comparing both stages against the original projections without quality thresholds.

The transport implementation ordinarily encodes LoRA deltas, followed by
base-plus-delta recombination. This fixture uses synthetic projections with an
implicit zero base to exercise its exact codec API without loading PEFT or a
model. Delivery is simulated, matching the existing projection context manager;
it does not test a socket protocol. The storage path is the existing CacheGen
inspired C2 K20/V16 research codec, not the same packet format as transport.
The frozen real profile is reused; tiny entries need not save bytes after framing.

Validation in this workspace: two dependency-free routing/adapter tests pass;
real smoke execution exits 2 with `No module named 'torch'`. No real codec PASS
or reconstruction scores are claimed. Run the routing checks with:

```bash
python3 -m unittest discover -s tests -p test_cachegen_hit_miss_pipeline.py -v
```

Full engine integration remains separate work: resolve the storage source branch,
provide the compatible GPU runtime, replace synthetic projections with model
projections and base recombination, and connect Q-free entries to engine lookup,
accounting, and eviction. Existing C2 engine entries retain raw Q; this smoke test
changes only its standalone resident-entry boundary.
