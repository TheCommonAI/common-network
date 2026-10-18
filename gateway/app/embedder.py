from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.config import settings

_model: SentenceTransformer | None = None

# One thread, deliberately. The encode is CPU-bound, and two encodes running at
# once on a small box split the same cores between them and both get slower.
# A single thread keeps the event loop free while the encodes queue up in order.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")


def load() -> None:
    from sentence_transformers import SentenceTransformer
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embed_model_name)


def embed(text: str) -> list[float]:
    if _model is None:
        raise RuntimeError("embedder not loaded — call load() at startup")
    vec: np.ndarray = _model.encode(text, normalize_embeddings=True)
    return vec.tolist()


async def embed_async(text: str) -> list[float]:
    """`embed` on the embedding thread, so the event loop keeps serving."""
    return await asyncio.get_running_loop().run_in_executor(_executor, embed, text)
