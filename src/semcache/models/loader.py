def load_model(config):
    """Explicit invocation only. No model download occurs at import/test time."""
    from .runtime import require_models
    require_models()
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    from .runtime import deterministic_cuda_preflight
    deterministic_cuda_preflight(config["device"])
    c = config
    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}.get(c["dtype"])
    if dtype is None:
        raise ValueError("Unsupported dtype; INT4 quantization is deferred")
    common = {"local_files_only": c["local_files_only"], "trust_remote_code": False}
    if c.get("revision"):
        common["revision"] = c["revision"]
    model_config = AutoConfig.from_pretrained(c["name"], **common)
    if model_config.model_type != "opt":
        raise ValueError("Only OPT is supported")
    tokenizer_args = {"local_files_only": c["local_files_only"], "trust_remote_code": False}
    if c.get("tokenizer_revision"):
        tokenizer_args["revision"] = c["tokenizer_revision"]
    tokenizer = AutoTokenizer.from_pretrained(c["tokenizer"], **tokenizer_args)
    model = AutoModelForCausalLM.from_pretrained(c["name"], config=model_config, dtype=dtype, use_safetensors=False,
                attn_implementation=c["attention_implementation"], **common).to(c["device"]).eval()
    metadata = {"model": c["name"], "model_revision": c.get("revision"),
                "model_revision_requested": c.get("revision"),
                "resolved_model_revision": getattr(model.config, "_commit_hash", None),
                "tokenizer": c["tokenizer"], "tokenizer_revision": c.get("tokenizer_revision"),
                "tokenizer_revision_requested": c.get("tokenizer_revision"),
                "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
                "dtype": str(next(model.parameters()).dtype), "device": str(next(model.parameters()).device),
                "torch_version": torch.__version__, "transformers_version": transformers.__version__}
    return model, tokenizer, metadata
