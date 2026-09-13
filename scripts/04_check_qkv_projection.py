import json
from _common import arguments, ROOT
from semcache.models.loader import load_model
from semcache.models.model_adapter import OPTModelAdapter
from semcache.models.capture import qkv_capture
from semcache.utils.seed import seed_everything
from semcache.utils.timing import timer
from semcache.utils.io import write_json

args, config = arguments("Inspect OPT projection hooks; loads only explicitly configured local weights")
seed_everything(config["seed"])
model, tokenizer, metadata = load_model(config["model"])
adapter = OPTModelAdapter(model)
inputs = tokenizer(config["inspection"]["text"], return_tensors="pt").to(config["model"]["device"])
with timer(config["model"]["device"]) as timing:
    import torch
    layer = config['inspection']['layer_idx']
    with qkv_capture(model, layers=[layer]) as capture:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    record = capture.records[layer]
    qkv = tuple(record[n] for n in ('q','k','v'))
report = {"metadata": metadata, "layer_idx": config["inspection"]["layer_idx"], "seed": config["seed"],
          "shapes": [list(t.shape) for t in qkv], "tensor_bytes": sum(t.numel()*t.element_size() for t in qkv),
          "inspection_total_latency_s": timing["seconds"], "metric_source": "measured",
          "projection_validation": record["validation"], "layer_metadata": record["metadata"],
          "same_input_projection_parity": True, "cross_query_reuse_verified": False}
write_json(args.output or ROOT / "results/raw/qkv_inspection.json", report)
print(json.dumps(report, indent=2))
