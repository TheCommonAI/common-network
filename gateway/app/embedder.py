from __future__ import annotations
import numpy as np

from app.config import settings

_model: SentenceTransformer | None = None


def load() -> None:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.embed_model_name)


def embed(text: str) -> list[float]:
    if _model is None:
        raise RuntimeError("embedder not loaded — call load() at startup")
    vec: np.ndarray = _model.encode(text, normalize_embeddings=True)
    return vec.tolist()
