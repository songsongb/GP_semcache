"""M8.5 prefill profile and artifact provenance, without model imports."""
from semcache.cache.attention_impact import IMPACT_MODES


ALIGNMENT_FIELDS = (
    "impact_reducer_type", "impact_reducer_metadata", "impact_reducer_applied",
    "cluster_update_interval_queries", "cluster_schedule_mode", "rho", "history_lambda",
    "pbr_interval_queries", "pbr_schedule_provenance",
    "admission_frequency_semantics", "eviction_frequency_semantics",
    "admission_frequency_window_queries", "cache_addressing_mode", "execution_scope",
)


def add_alignment_arguments(parser):
    parser.add_argument("--impact-reducer", choices=IMPACT_MODES, default="paper_row_l2_sum")
    parser.add_argument("--cluster-update-mode", choices=("buffered", "immediate_eq9"), default="buffered")
    parser.add_argument("--cluster-update-interval", type=int, default=100)
    parser.add_argument("--rho", type=float, default=.8)
    parser.add_argument("--history-lambda", type=int, default=100)
    parser.add_argument("--pbr-mode", choices=("manual", "interval"), default="interval")
    parser.add_argument("--pbr-interval", type=int, default=100,
                        help="Global query interval; REPRODUCTION_CHOICE, not low-load detection")


def validate_alignment_arguments(args):
    if (args.cluster_update_interval < 1 or args.history_lambda < 1
            or args.pbr_interval < 1 or not 0 <= args.rho < 1):
        raise ValueError("Intervals and Lambda must be positive; rho must be in [0,1)")


def artifact_alignment(row):
    """Missing provenance in historical artifacts stays unknown, never inferred."""
    return {field: row.get(field) for field in ALIGNMENT_FIELDS}
