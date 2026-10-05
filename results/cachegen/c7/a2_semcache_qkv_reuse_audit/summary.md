# C7-A2 SemCache Q/K/V reuse audit

The cache hit is a w=3 token-subsequence hit, not a Q/K/V-specific hit.
Q/K/V are the cached payload associated with the selected hit.
SemCache intentionally reuses Q as well as K and V because its target is projection reuse in EdgeLoRA, not merely standard autoregressive KV-cache reuse.
Transport LoRA Q delta and cached resident TOTAL Q are distinct objects.
Fresh target Q is separately computed on unmatched positions.

Flow: source mixed projection capture -> source w=3 slices -> CacheEntry.tensors[layer][q/k/v index] -> (cluster, exact token IDs) lookup -> CacheHit -> common nonoverlap mask -> mixed_projection_path -> target hit rows.

Raw Q/K/V reuse confirmed: True. Same reuse mask: True.
CPU fixture: 3 layers, one selected three-token hit at positions [2,3,4], four fresh rows. Per-role base and LoRA-A hooks check actual input rows. Per-layer reuse/skipped counts and value comparisons are in the JSON.
Transport boundary bypass confirmed: True. Only an uncompressed base+delta callback probe ran; actual transport-reconstructed row counts are null.
Compressed-storage / FULL_PIPELINE confirmation: unknown. No frozen codec/profile execution in this audit. C2 source is missing. Installed GlobalCache lacks physical_codec. Decoded-view contract probe is not a real compressed-storage lookup.
The constructed decoded-view input passes the real C6 validator and exercises the real mixed consumer; this proves the consumer contract, not external codec view construction or compressed lookup.
C6 raw/shared mixed execution retains Q as an active SemCache reuse payload. The broader claim that compressed C6 preserved it end-to-end remains unverified here.

Counterfactual status: COMPLETE. Each independent Q/K/V corruption tests payload consumption only.
Previous C7-A framing is superseded; its implementation and artifacts remain unchanged and are not removal evidence.

Recommendation: **INCONCLUSIVE**.
Next stage: **FURTHER_AUDIT_ONLY**. Inspect and validate the actual frozen C2 codec and extended GlobalCache path before C7-B.

