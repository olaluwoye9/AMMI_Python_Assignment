"""Embedding generation pipeline for document chunks using sentence-transformers."""

import numpy as np
import logging
from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENT_TRANSFORMERS = True
except Exception:
    # Do not fail import-time — provide a lightweight fallback below.
    _HAS_SENT_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for document chunks.

    This class uses `sentence-transformers` when available. If the package or
    required system libraries are missing, a deterministic lightweight
    fallback is used so downstream code (retriever/orchestrator) can run in
    constrained environments.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        if _HAS_SENT_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"✅ Loaded model: {model_name} (dim={self.embedding_dim})")
                self._use_fallback = False
            except Exception:
                logger.warning("Failed to load sentence-transformers model — using fallback embeddings")
                self._use_fallback = True
                self.embedding_dim = 384
                self.model = None
        else:
            logger.info("sentence-transformers not available — using fallback embeddings")
            self._use_fallback = True
            self.embedding_dim = 384
            self.model = None

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """Deterministic fallback embedding using SHA256 hashing.

        Produces normalized float32 vectors of length `self.embedding_dim`.
        """
        import hashlib

        out = []
        dim = self.embedding_dim
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # Expand hash bytes to required dim
            reps = (dim + len(h) - 1) // len(h)
            data = (h * reps)[:dim]
            arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
            arr = arr - arr.mean()
            norm = np.linalg.norm(arr) + 1e-9
            arr = arr / norm
            out.append(arr)
        return np.vstack(out)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        if not self._use_fallback:
            logger.info(f"Embedding {len(texts)} texts with model={self.model_name}")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)
        return self._fallback_embed(texts).astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        emb = self.embed_texts([text])
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        return emb

    def embed_from_files(self, file_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        texts = []
        for fpath in tqdm(file_paths, desc="Reading files"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            except Exception as e:
                logger.warning(f"Failed to read {fpath}: {e}")
        embeddings = self.embed_texts(texts)
        return embeddings, texts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = EmbeddingGenerator("all-MiniLM-L6-v2")
    
    # Test embedding
    test_texts = [
        "What is the maternity leave policy?",
        "Employee conduct guidelines.",
        "Payroll and compensation details."
    ]
    embeddings = gen.embed_texts(test_texts)
    print(f"Shape: {embeddings.shape}")
    print(f"✅ Sample embedding shape: {embeddings[0].shape}")
