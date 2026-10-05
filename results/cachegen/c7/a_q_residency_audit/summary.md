# C7-A resident TOTAL Q audit

Question: is resident source TOTAL Q consumed after insertion on the target HIT path?

Recommendation: **COMPRESS_RESIDENT_Q**.

Static dataflow: source mixed projections → CacheEntry.tensors[layer][0] → GlobalCache.lookup → CacheHit → mixed_projection_path.forward → source.to(output) copied into matched target Q rows → attention. Compressed C6 contracts use q_tensors plus a temporary decoded QKV view; external C2 source is missing here.

Exact cluster/token-w3 identity and selected entry do not inspect Q values. Byte accounting reads Q storage; the mixed projection consumer reads its numerical values. Same-layer K/V reuse and fresh LoRA recombination do not need resident Q. Attention-dependent impact updates can indirectly feed later policy decisions.

Runtime counters (normal fixture): {"resident_k_read_count": 2, "resident_k_write_count": 1, "resident_q_hit_read_count": 1, "resident_q_read_count": 2, "resident_q_value_consumer_count": 1, "resident_q_write_count": 1, "resident_v_read_count": 2, "resident_v_write_count": 1, "transport_q_delta_decode_count": 0, "transport_q_delta_encode_count": 0}

Counterfactual status: COMPLETE. 

Zero-Q changes: {"logical_safety_assertion": false, "lookup_success": false, "reconstructed_kv": false, "reuse_execution_failed": false, "selected_cache_entry": false, "target_reuse_output": true}
Zero-Q maximum absolute attention-output delta: 0.559219479560852
None-Q changes: {"logical_safety_assertion": false, "lookup_success": false, "reconstructed_kv": false, "reuse_execution_failed": true, "selected_cache_entry": false, "target_reuse_output": null}
None-Q error: AttributeError: 'NoneType' object has no attribute 'shape'
Field-presence dependency != value dependency: None exposes schema assumptions; zero-Q independently tests numerical necessity.

Transported LoRA Q-delta activity does not by itself justify keeping resident TOTAL Q.
Fresh target Q is a separate runtime object and is not evidence that cached resident source Q is required.

Storage implication: Q remains raw FP16 in the C6 policy; compressed K/V frame metadata is already included in the frame. No Q removal benchmark or byte-savings-based recommendation is made. Optional artifact context is in the audit JSON.

Scope: tiny CPU PEFT projection modules and causal attention, no language-model inference, no CUDA/transport codec execution, no training, no compression implementation, no production cache changes. The shared raw HIT consumer is tested; external compressed storage internals remain unverified.
