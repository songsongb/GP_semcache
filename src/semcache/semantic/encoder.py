from abc import ABC, abstractmethod


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
    def __init__(self, checkpoint, revision, pooling="masked_mean", max_length=512,
                 device="cpu", local_files_only=True, backend="huggingface_text"):
        if pooling != "masked_mean" or backend != "huggingface_text":
            raise ValueError("Unsupported semantic encoder policy")
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(checkpoint, revision=revision, local_files_only=local_files_only).to(device).eval()
        self.device, self.max_length = device, max_length
        self._metadata = {"checkpoint": checkpoint, "revision": revision,
                          "resolved_revision": getattr(self.model.config, "_commit_hash", None),
                          "pooling": pooling, "max_length": max_length, "backend": backend}

    @property
    def metadata(self):
        return self._metadata.copy()

    def encode(self, queries):
        import torch
        batch = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
        with torch.inference_mode():
            h = self.model(**batch).last_hidden_state
            mask = batch.attention_mask.unsqueeze(-1)
            return ((h * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu().tolist()
