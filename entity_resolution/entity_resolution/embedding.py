import hashlib
import logging
import numpy as np

log = logging.getLogger(__name__)

class HashEmbedder:
    """Offline test embedder; never use for production retrieval."""
    def __init__(self, dim=1024): self.dim = dim
    def encode(self, texts, batch_size=64):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            for j in range(self.dim): out[i, j] = ((digest[j % len(digest)] / 255.0) * 2) - 1
        norms = np.linalg.norm(out, axis=1, keepdims=True); return out / np.maximum(norms, 1e-12)

class SentenceTransformerEmbedder:
    def __init__(self, model_name="BAAI/bge-m3", device=None, dim=1024):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = dim
    def encode(self, texts, batch_size=64):
        x = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        x = np.asarray(x, dtype="float32")
        if x.shape[1] != self.dim: raise ValueError(f"Expected {self.dim} dimensions, got {x.shape[1]}")
        return x

def make_embedder(config, device=None):
    if config.backend == "hash": return HashEmbedder(config.embedding_dim)
    return SentenceTransformerEmbedder(config.model_name, device=device, dim=config.embedding_dim)
