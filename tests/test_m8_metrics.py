import time

import pytest

from semcache.metrics.m8 import (MODES, RAW_FIELDS, TRAINING_FIELDS,
    aggregate_raw, analytical_communication, cache_transfer_path, incremental_memory,
    mode_identity, model_config, raw_record, repetition_schedule, token_ids_sha256,
    request_timing_accounting, training_record, training_step_diagnostics,
    exact_parity_passed, reuse_delta_summary, training_timing_summary)
from semcache.metrics.scaling_workload import (construct_natural_prompt,
    controlled_length_trace, parse_prompt_lengths, subsequence_occurrence_report)
from semcache.metrics.timing import (CPUWallTimer, CUDATimer, TimingRegistry,
    close_request_timing, resolve_cuda_event_pairs)


def test_cpu_wall_timer_and_scope_metadata():
    registry = TimingRegistry()
    with registry.cpu("work_ms", parent="request_wall_ms"):
        time.sleep(.001)
    values, scopes = registry.export()
    assert values["work_ms"] >= .5
    assert scopes["work_ms"] == dict(timing_scope="work_ms", timing_parent="request_wall_ms",
        inclusive_or_exclusive="exclusive", clock="cpu_perf_counter_ns")
    with CPUWallTimer() as timer:
        pass
    assert timer.elapsed_ms >= 0


def test_correctness_work_is_outside_request_wall():
    request = CPUWallTimer()
    request.__enter__()
    time.sleep(.001)
    values, scopes = close_request_timing(request, TimingRegistry(), True)
    with CPUWallTimer() as correctness:
        time.sleep(.025)
    assert values["request_wall_ms"] < correctness.elapsed_ms
    assert scopes["request_wall_ms"]["timing_parent"] is None


def test_cuda_timer_or_clean_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with CUDATimer() as timer:
        torch.ones(8, device="cuda").square_()
    assert timer.elapsed_ms >= 0


def test_cuda_event_batch_resolver_synchronizes_once_not_per_layer():
    calls = []
    class Event:
        def __init__(self, value): self.value = value
        def synchronize(self): calls.append(self.value)
        def elapsed_time(self, end): return end.value - self.value
    pairs = [(Event(i * 10), Event(i * 10 + 2)) for i in range(12)]
    assert resolve_cuda_event_pairs(pairs) == [2.] * 12
    assert calls == [112]
    calls.clear()
    assert resolve_cuda_event_pairs(pairs, synchronize=False) == [2.] * 12
    assert calls == []


def test_mixed_projection_capture_has_no_per_projection_cpu_transfer():
    import inspect
    from semcache.edgelora.mixed_projection import mixed_projection_path
    source = inspect.getsource(mixed_projection_path)
    assert "output.detach().cpu()" not in source
    assert "output.detach()" in source


def test_attention_reducer_has_one_host_materialization_after_layers():
    import inspect
    from semcache.cache.attention_impact import MeanLayerHeadFrobeniusReducer
    source = inspect.getsource(MeanLayerHeadFrobeniusReducer.reduce)
    assert source.count(".tolist()") == 1
    assert "torch.cat(norm_tensors).tolist()" in source


def test_aggregation_p50_p95_and_nulls():
    rows = []
    for i, value in enumerate((1., 2., 3., 4., None)):
        rows.append(raw_record(experiment_id="x", model_id="facebook/opt-125m",
            mode="NATIVE_NO_CACHE", query_id="q", repeat_index=i, request_wall_ms=value))
    summary = next(x for x in aggregate_raw(rows) if x["metric"] == "request_wall_ms")
    assert summary["count"] == 4 and summary["mean"] == 2.5
    assert summary["p50"] == 2.5 and summary["p95"] == pytest.approx(3.85)
    empty = next(x for x in aggregate_raw(rows) if x["metric"] == "attention_ms")
    assert empty["count"] == 0 and empty["mean"] is None


class WhitespaceTokenizer:
    def __call__(self, text):
        return {"input_ids": [0] + [sum(map(ord, word)) for word in text.split()]}


def test_controlled_prompt_construction_is_deterministic_and_exact_repeats_match():
    tokenizer = WhitespaceTokenizer()
    a = construct_natural_prompt(tokenizer, 32, topic="hotel")
    b = construct_natural_prompt(tokenizer, 32, topic="hotel")
    assert a == b and a["requested_prompt_tokens"] == 32
    assert a["actual_prompt_tokens"] == len(a["token_ids"])
    trace = controlled_length_trace(tokenizer, 64)
    assert [item["condition"] for item in trace] == [
        "cold_miss", "same_user_exact", "cross_user_exact", "unrelated"]
    assert trace[0]["token_ids"] == trace[1]["token_ids"] == trace[2]["token_ids"]
    assert trace[0]["text"] == trace[1]["text"] == trace[2]["text"]
    assert trace[0]["subsequence_audit"]["duplicated_key_count"] > 0
    assert parse_prompt_lengths("32,64,128,256") == [32, 64, 128, 256]
    with pytest.raises(ValueError):
        parse_prompt_lengths("32,32")


def test_duplicate_w3_occurrence_detection_reports_positions():
    report = subsequence_occurrence_report([1, 2, 3, 1, 2, 3, 1], 3)
    assert report["total_windows"] == 5
    assert report["unique_windows"] == 3
    assert report["duplicated_key_count"] == 2
    assert report["max_occurrences"] == 2
    assert report["duplicated_windows"][0] == dict(
        token_ids=[1, 2, 3], positions=[0, 3], occurrences=2)


def test_prompt_length_is_an_explicit_summary_group():
    rows = [raw_record(experiment_id="x", model_id="m", mode=MODES[0], query_id="q",
        requested_prompt_tokens=length, actual_prompt_tokens=length + 1, request_wall_ms=float(length))
        for length in (32, 64)]
    summary = [row for row in aggregate_raw(rows) if row["metric"] == "request_wall_ms"]
    assert {(row["requested_prompt_tokens"], row["actual_prompt_tokens"]) for row in summary} == {
        (32, 33), (64, 65)}


def test_reuse_delta_sign_convention():
    rows = []
    for mode, qkv, prefill, request in [
        ("SEMCACHE_LOOKUP_NO_REUSE", 100., 200., 300.),
        ("SEMCACHE_PHYSICAL_REUSE", 80., 220., 330.),
    ]:
        rows.append(raw_record(experiment_id="x", model_id="m", mode=mode,
            query_id="same_user_exact", requested_prompt_tokens=64, actual_prompt_tokens=65,
            candidate_blocks=63, attention_impact_ms=30., attention_impact_block_count=63,
            reused_tokens=60, recomputed_tokens=5, token_reuse_ratio=60/65, block_hits=61,
            mixed_qkv_execution_ms=qkv, prefill_wall_ms=prefill, request_wall_ms=request))
    result = reuse_delta_summary(rows)[0]
    assert result["qkv_reuse_delta_ms"] == 20.
    assert result["qkv_reuse_delta_percent"] == 20.
    assert result["qkv_observed_direction"] == "physical_faster"
    assert result["prefill_reuse_delta_ms"] == -20.
    assert result["request_reuse_delta_ms"] == -30.
    assert result["request_observed_direction"] == "physical_slower"
    assert result["physical_reused_tokens"] == 60
    assert result["physical_token_reuse_ratio"] == pytest.approx(60/65)
    assert result["physical_correctness_validation_status"] == "correctness_not_validated_for_lengths_ge_64"


def test_parity_field_means_numerical_success_not_fixture_name():
    divergent = raw_record(controlled_exact_fixture=True, max_abs_logit_diff=1.,
        relative_l2_logit_diff=.2, last_position_kl=1., argmax_agreement=False)
    assert divergent["controlled_exact_fixture"] is True
    assert divergent["controlled_exact_parity_passed"] is False
    assert exact_parity_passed(divergent) is False
    exact = raw_record(controlled_exact_fixture=True, max_abs_logit_diff=0.,
        relative_l2_logit_diff=0., last_position_kl=0., argmax_agreement=True)
    assert exact["controlled_exact_parity_passed"] is True


def test_position_aligned_diagnostic_checks_source_position_adapter_and_prompt():
    from semcache.cache.cache_entry import CacheEntry
    from semcache.semantic.hit_selection import CacheHit, select_position_aligned_diagnostic
    from semcache.semantic.subsequence import Subsequence
    ids = (1, 2, 3, 1, 2, 3)
    entry = CacheEntry(0, (1, 2, 3), (0, 3), 1,
        qkv_metadata=dict(source_query_token_ids=ids, source_user="user_a",
                          source_adapter="user_a", source_query_id="cold_miss"))
    aligned = CacheHit(Subsequence((1, 2, 3), 0, 3), entry)
    repeated_wrong_position = CacheHit(Subsequence((1, 2, 3), 3, 6), entry)
    selected, mask = select_position_aligned_diagnostic(
        [aligned, repeated_wrong_position], len(ids), ids, "user_a")
    assert selected == [aligned] and sum(mask) == 3
    assert select_position_aligned_diagnostic([aligned], len(ids), ids, "user_b")[0] == []
    assert select_position_aligned_diagnostic([aligned], len(ids), ids + (4,), "user_a")[0] == []


def test_length64_admission_audit_schema():
    audit = dict(cache_key=(0, (1, 2, 3)), source_start=0, source_end=3,
        admission_score=.29, admission_decision=False,
        rejection_reason="admission_score_at_or_below_threshold",
        admission_candidate_order=1, admission_order=None, source_occurrence_count=1)
    row = raw_record(admission_audit=[audit], admission_candidate_count=1,
        admitted_block_count=0, rejected_block_count=1, deduplicated_window_count=0)
    assert row["admission_audit"][0]["rejection_reason"]
    assert row["admission_candidate_count"] == (
        row["admitted_block_count"] + row["rejected_block_count"])


def test_attention_impact_schema_and_residual_accounting_are_nonoverlapping():
    row = raw_record(experiment_id="x", model_id="m", mode=MODES[2], query_id="q",
        request_wall_ms=100., tokenization_ms=1., semantic_encode_ms=10.,
        cluster_assign_update_ms=1., subsequence_extract_ms=1., cache_lookup_ms=1.,
        hit_selection_ms=1., prefill_wall_ms=50., prefill_gpu_ms=500., attention_impact_ms=20.,
        chu_update_ms=1., cache_policy_ms=5., cache_materialization_ms=500.,
        pbr_update_ms=1., mixed_qkv_execution_ms=500.,
        correctness_reference_ms=800., quality_diagnostics_ms=900.,
        attention_impact_block_count=9, attention_impact_ms_per_block=20/9)
    assert row["timing_accounted_ms"] == 92.
    assert row["unaccounted_request_ms"] == 8.
    assert row["timing_coverage_ratio"] == .92
    assert row["attention_impact_ms_per_block"] == pytest.approx(20/9)
    assert row["prefill_host_overhead_ms"] == -450.
    # Inclusive children and outside-request correctness work are not subtracted.
    assert request_timing_accounting(row)["timing_accounted_ms"] == 92.


def test_derived_timing_fields_null_and_populated_rules():
    populated = raw_record(request_wall_ms=20., prefill_wall_ms=12., prefill_gpu_ms=10.,
        attention_impact_ms=9., attention_impact_block_count=3)
    assert populated["attention_impact_ms_per_block"] == 3.
    assert populated["prefill_host_overhead_ms"] == 2.
    assert populated["timing_accounted_ms"] == 21.  # top-level wall + impact only
    empty = raw_record(request_wall_ms=1., attention_impact_ms=9., attention_impact_block_count=0)
    assert empty["attention_impact_ms_per_block"] is None
    assert empty["prefill_host_overhead_ms"] is None


def test_raw_memory_and_training_schemas_are_complete():
    row = raw_record(experiment_id="x", model_id="facebook/opt-125m", mode=MODES[0], query_id="q")
    assert tuple(row) == RAW_FIELDS
    assert row["attention_ms"] is None and row["peak_cuda_allocated_bytes"] is None
    assert row["cuda_allocated_before_bytes"] is None
    assert row["incremental_peak_allocated_bytes"] is None
    assert row["correctness_reference_ms"] is None
    assert row["physical_cache_storage_device"] is None
    assert row["safe_reuse_claimed"] is False
    training = training_record(experiment_id="t", model_id="facebook/opt-125m")
    assert tuple(training) == TRAINING_FIELDS
    assert training["total_training_wall_s"] is None


def test_mode_and_large_model_guards():
    assert MODES == ("NATIVE_NO_CACHE", "SEMCACHE_LOOKUP_NO_REUSE", "SEMCACHE_PHYSICAL_REUSE")
    assert model_config("facebook/opt-125m")["dtype"] == "float32"
    assert model_config("facebook/opt-2.7b")["dtype"] == "float16"
    with pytest.raises(PermissionError):
        model_config("facebook/opt-6.7b")
    with pytest.raises(ValueError):
        model_config("facebook/opt-6.7b", allow_large_model=True, measured_runs=6)


def test_repeat_schedule_reconstructs_state_and_discards_warmup_state():
    schedule = repetition_schedule(2, 3)
    assert [x["phase"] for x in schedule] == ["warmup", "warmup", "measured", "measured", "measured"]
    # Model the script's fresh engine factory: mutations never cross traces.
    starts, discarded = [], []
    for item in schedule:
        state = {"cache": [], "centroid_count": 1}
        starts.append((list(state["cache"]), state["centroid_count"]))
        state["cache"].append("primed")
        state["centroid_count"] += 1
        if item["phase"] == "warmup":
            discarded.append(state)
    assert starts == [([], 1)] * 5
    assert len(discarded) == 2


def test_mode_identity_excludes_mode_and_is_equivalent():
    token_hash = token_ids_sha256([1, 2, 3])
    identities = [mode_identity("model-sha", "tokenizer-sha", "torch.float16",
        token_hash, "user_a", 42, "eager") for _ in MODES]
    assert identities[0] == identities[1] == identities[2]
    assert "mode" not in identities[0]


def test_incremental_peak_memory_math_and_nulls():
    assert incremental_memory(100, 200, 160, 300) == dict(
        incremental_peak_allocated_bytes=60, incremental_peak_reserved_bytes=100)
    assert incremental_memory(None, None, None, None) == dict(
        incremental_peak_allocated_bytes=None, incremental_peak_reserved_bytes=None)


def test_cache_storage_transfer_provenance_uses_runtime_devices():
    assert cache_transfer_path("cpu", "cuda:0") == "cpu_to_cuda_on_reuse"
    assert cache_transfer_path("cuda:1", "cuda:1") == "cuda_local_on_reuse"


def test_training_step_diagnostic_keeps_first_step_separate():
    result = training_step_diagnostics([100., 20., 40.])
    assert result == dict(first_step_ms=100., steady_state_step_mean_ms=30.,
        steady_state_step_p50_ms=30.)
    one = training_step_diagnostics([100.])
    assert one["first_step_ms"] == 100.
    assert one["steady_state_step_mean_ms"] is None
    summary = training_timing_summary(0.175, [100., 20., 40.])
    # Total remains caller-measured/startup-inclusive; diagnostics do not rewrite it.
    assert summary["total_training_wall_s"] == 0.175
    assert summary["first_step_ms"] == 100.


def test_communication_is_analytical_only():
    result = analytical_communication(100, 2, 1000)
    assert result == dict(saved_communication_elements=100, saved_communication_bytes=200,
        estimated_communication_ms=200., measured_or_analytical="ANALYTICAL / SIMULATED")


def test_engine_mode_separation_and_physical_skip():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from test_lora import tiny_base
    from test_mixed_semcache import make_engine
    from semcache.models.lora_fixtures import create_controlled_users
    users = create_controlled_users(tiny_base())[0]
    engine = make_engine(users)
    source = engine.query("2 3 4 5 6 7 8 9", "user_a", "source")
    with pytest.raises(ValueError, match="outside request timing"):
        engine.query("2 3 4 5 6 7 8 9", "user_a", "invalid-reference",
            compare_baseline=True, collect_timing=True)
    lookup = engine.query("2 3 4 5 6 7 8 9", "user_a", "lookup",
        execution_mode="SEMCACHE_LOOKUP_NO_REUSE", collect_timing=True,
        baseline_logits=source["logits"])["summary"]
    assert lookup["block_hit_count"] and lookup["reused_unique_token_count"] == 0
    assert not lookup["physical_reuse_used"] and not lookup["projection_skip_used"]
    physical = engine.query("2 3 4 5 6 7 8 9", "user_a", "physical",
        execution_mode="SEMCACHE_PHYSICAL_REUSE", collect_timing=True,
        baseline_logits=source["logits"])["summary"]
    assert physical["reused_unique_token_count"] > 0
    assert physical["physical_reuse_used"] and physical["projection_skip_used"]
    assert physical["timing"]["mixed_qkv_execution_ms"] is None  # CPU fixture: no fake GPU zero.
    assert physical["timing"]["quality_diagnostics_ms"] is not None
    assert physical["timing"]["attention_impact_ms"] is not None
    assert physical["timing"]["attention_impact_ms_per_block"] is not None
    assert physical["timing"]["prefill_wall_ms"] is not None
    assert physical["timing"]["prefill_host_overhead_ms"] is None  # CPU fixture has no CUDA event.
    scope = physical["timing_scopes"]["attention_impact_ms"]
    assert scope["timing_parent"] == "request_wall_ms"
    assert scope["inclusive_or_exclusive"] == "exclusive"
    assert physical["attention_impact_block_count"] == physical["candidate_windows"]
    assert physical["max_abs_logit_diff"] is not None
    assert physical["relative_l2_logit_diff"] is not None
    assert physical["last_position_kl_baseline_to_injected"] is not None
    assert physical["last_argmax_agreement"] is not None
    provenance = physical["reuse_block_provenance"]
    assert provenance and all(item["source_query_id"] == "source" for item in provenance)
    assert all(item["source_user"] == item["source_adapter"] == "user_a" for item in provenance)
    assert all("positions_identical" in item for item in provenance)
    diagnostic = engine.query("2 3 4 5 6 7 8 9", "user_a", "aligned",
        execution_mode="SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC", collect_timing=True,
        baseline_logits=source["logits"])["summary"]
    assert diagnostic["projection_skip_used"]
    assert all(item["positions_identical"] for item in diagnostic["reuse_block_provenance"])
    assert all(item["source_user"] == item["destination_user"] for item in diagnostic["reuse_block_provenance"])
