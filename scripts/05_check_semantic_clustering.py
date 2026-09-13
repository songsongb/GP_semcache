import json
from _common import arguments
from semcache.demo import run_demo
args, config = arguments("Controlled centroid assignment smoke test")
report = run_demo(config)
print(json.dumps({"assignments": sorted({(e["request"], e["cluster_id"]) for e in report["events"]}), "encoder": "controlled_vectors_v1"}))
