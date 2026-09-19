from abc import ABC, abstractmethod


def masked_mean_pool(last_hidden_state, attention_mask):
    """Mean valid token states; padding contributes neither sum nor divisor."""
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


class SemanticEncoder(ABC):
    @abstractmethod
    def encode(self, queries):
        """Return one semantic vector per query."""

    @property
    @abstractmethod
    def metadata(self):
        pass


class ControlledEncoder(SemanticEncoder):
    """Fixture only: no learned semantics or TinyBERT performance claim."""
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, queries):
        return [list(self.vectors[q]) for q in queries]

    @property
    def metadata(self):
        return {"checkpoint": "controlled_vectors_v1", "revision": "1"}


class HuggingFaceTextEncoder(SemanticEncoder):
    """Reproduction choice: re-tokenize text, masked mean of last hidden layer.

    Does not implement Eq. 7's unspecified OPT-embedding to TinyBERT bridge.
    """
    def __init__(self, checkpoint=None, revision="main", pooling="masked_mean", max_length=512,
                 device="cpu", dtype=None, local_files_only=True, backend="huggingface_text",
                 tokenizer=None, model=None, model_id=None):
        if checkpoint is not None and model_id is not None and checkpoint != model_id:
            raise ValueError("checkpoint and model_id aliases disagree")
        checkpoint = model_id or checkpoint
        if not checkpoint:
            raise ValueError('Supply an explicit encoder checkpoint; the paper does not specify one')
        if pooling != "masked_mean" or backend not in {"huggingface_text", "tinybert_huggingface"}:
            raise ValueError("Unsupported semantic encoder policy")
        import torch
        if dtype is not None and isinstance(dtype, str):
            dtype = {"float16": torch.float16, "float32": torch.float32,
                     "bfloat16": torch.bfloat16}.get(dtype)
            if dtype is None:
                raise ValueError("Unsupported semantic encoder dtype")
        if tokenizer is None or model is None:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint, revision=revision, local_files_only=local_files_only)
            kwargs = dict(revision=revision, local_files_only=local_files_only)
            if dtype is not None:
                kwargs["dtype"] = dtype
            model = AutoModel.from_pretrained(checkpoint, **kwargs)
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device, self.max_length, self.pooling = device, max_length, pooling
        actual_dtype = str(next(self.model.parameters()).dtype)
        self._metadata = {"checkpoint": checkpoint, "revision": revision,
                          "resolved_revision": getattr(self.model.config, "_commit_hash", None),
                          "model_id": checkpoint, "tokenizer": getattr(tokenizer, "name_or_path", checkpoint),
                          "tokenizer_class": type(tokenizer).__name__,
                          "hidden_size": getattr(self.model.config, "hidden_size", None),
                          "pooling": pooling, "max_length": max_length, "backend": backend,
                          "device": str(device), "dtype": actual_dtype,
                          "family": "TinyBERT" if backend == "tinybert_huggingface" else "unspecified_hf_encoder",
                          "provenance": "REPRODUCTION_CHOICE"}

    @property
    def metadata(self):
        return self._metadata.copy()

    def encode(self, queries):
        import torch
        if isinstance(queries, str):
            queries = [queries]
        batch = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True,
                               max_length=self.max_length)
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            h = self.model(**batch).last_hidden_state
            return masked_mean_pool(h, batch["attention_mask"]).cpu().tolist()


class TinyBERTSemanticEncoder(HuggingFaceTextEncoder):
    """Long-lived Hugging Face TinyBERT encoder.

    The family is paper-defined. Checkpoint, revision, pooling, truncation and
    tokenizer details are explicit reproduction choices.
    """
    DEFAULT_MODEL_ID = "huawei-noah/TinyBERT_General_4L_312D"

    def __init__(self, model_id=DEFAULT_MODEL_ID, revision="main", checkpoint=None,
                 backend="tinybert_huggingface", **kwargs):
        if backend != "tinybert_huggingface":
            raise ValueError("TinyBERT encoder requires tinybert_huggingface backend")
        if checkpoint is not None and checkpoint != model_id:
            raise ValueError("checkpoint and model_id aliases disagree")
        super().__init__(model_id, revision, backend=backend, **kwargs)
