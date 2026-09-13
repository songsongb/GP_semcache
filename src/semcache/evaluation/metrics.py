FIELDS = """run_id seed model model_revision tokenizer dtype dataset method num_users lora_rank cluster_count window_size semantic_encoder match_rule cache_size_gb theta_admit rho pbr_interval bandwidth_mbps num_requests total_tokens cache_hits cache_misses hit_rate reused_tokens reuse_ratio encoder_latency_s base_projection_latency_s lora_projection_latency_s forward_comm_latency_s backward_comm_latency_s attention_latency_s ffn_latency_s system_latency_s logical_memory_gb physical_peak_memory_gb bleu metric_source""".split()


def empty_row():
    return dict.fromkeys(FIELDS)
