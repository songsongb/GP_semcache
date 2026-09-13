import json
from _common import arguments, ROOT
from semcache.demo import run_demo
from semcache.utils.io import write_json, write_csv
args, config = arguments("Controlled two-request MISS -> HIT index demo; no LLM inference")
report = run_demo(config)
output = args.output or ROOT / "results/raw/milestone1_demo.json"
write_json(output, report)
row = report["summary"].copy()
row["metric_source"] = json.dumps(row["metric_source"], sort_keys=True)
write_csv(output.with_suffix(".csv"), [row])
for e in report["events"]:
    print(f"request {e['request']} cluster {e['cluster_id']} {e['token_ids']}: {e['status']}")
for insertion in report["insertions"]:
    print(f"request {insertion['request']} admission {insertion['token_ids']}: {insertion['admitted']}")
print(f"logical bytes={report['logical_cache_bytes']}; physical tensor bytes={report['physical_tensor_bytes']}")
print("Index hits only; numerical cross-query QKV reuse is NOT verified.")
