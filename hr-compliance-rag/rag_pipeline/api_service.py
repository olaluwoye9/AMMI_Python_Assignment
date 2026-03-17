from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from pathlib import Path
from rag_pipeline.configs.config import METADATA_CSV

try:
    from vectorstore.retriever import SemanticRetriever
except Exception:
    # Fall back to a lightweight stub when heavy deps (sentence-transformers/faiss)
    # are not available in the runtime (e.g., local dev without full build).
    class SemanticRetriever:
        def __init__(self, *args, **kwargs):
            pass

        def retrieve(self, query, top_k=5, filters=None):
            return []

app = FastAPI(title="HR Compliance RAG API")


class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list
    raw: Optional[Dict[str, Any]] = None


# Lazily instantiate retriever and orchestrator to avoid importing heavy deps at startup
index_path = Path("vectorstore/faiss.index")
retriever = None
orchestrator = None

def _init_orchestrator_if_possible():
    global retriever, orchestrator
    if orchestrator is not None:
        return
    # only attempt to create a real retriever if index + metadata exist
    if index_path.exists() and Path(METADATA_CSV).exists():
        try:
            from vectorstore.retriever import SemanticRetriever
            from rag_pipeline.rag_orchestrator import RAGOrchestrator

            retriever = SemanticRetriever(index_path=str(index_path), metadata_path=METADATA_CSV)
            orchestrator = RAGOrchestrator(retriever)
        except Exception:
            try:
                # Lightweight in-process retriever fallback using embeddings
                import numpy as np
                import pandas as pd
                from vectorstore.embedding_generator import EmbeddingGenerator
                from rag_pipeline.rag_orchestrator import RAGOrchestrator

                class SimpleRetriever:
                    def __init__(self, metadata_path: str = METADATA_CSV, embedding_model: str = "all-MiniLM-L6-v2"):
                        self.metadata_path = metadata_path
                        self.metadata_df = pd.read_csv(metadata_path)
                        data_dir = Path(metadata_path).parent
                        processed_dir = data_dir / "processed"
                        self.chunks = []
                        for _, row in self.metadata_df.iterrows():
                            chunk_id = str(row.get("chunk_id") or "")
                            file_name = str(row.get("file_name") or "")
                            text = ""
                            if chunk_id and processed_dir.exists():
                                matches = list(processed_dir.rglob(f"*{chunk_id}*"))
                                if matches:
                                    try:
                                        text = matches[0].read_text(encoding="utf-8")
                                    except Exception:
                                        text = ""
                            if not text:
                                text = str(row.get("chunk_text") or row.get("text") or "")
                            meta = {c: row.get(c) for c in self.metadata_df.columns}
                            self.chunks.append({"file_name": file_name, "chunk_id": chunk_id, "text": text, "metadata": meta})
                        self.embedding_gen = EmbeddingGenerator(embedding_model)
                        texts = [c["text"] or "" for c in self.chunks]
                        if texts:
                            embs = self.embedding_gen.embed_texts(texts)
                            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
                            self.embeddings = embs / norms
                        else:
                            self.embeddings = np.empty((0, self.embedding_gen.embedding_dim), dtype=np.float32)

                    def retrieve(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None):
                        if self.embeddings.shape[0] == 0:
                            return []
                        q_emb = self.embedding_gen.embed_single(query).reshape(-1)
                        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
                        scores = (self.embeddings @ q_emb).astype(float)
                        idxs = np.arange(len(self.chunks))
                        if filters:
                            mask = np.ones(len(self.chunks), dtype=bool)
                            for field, value in (filters or {}).items():
                                if field in self.metadata_df.columns:
                                    vals = np.array([c["metadata"].get(field) for c in self.chunks])
                                    if isinstance(value, (list, tuple)):
                                        mask &= np.isin(vals, value)
                                    else:
                                        mask &= (vals == value)
                                else:
                                    pass
                            idxs = idxs[mask]
                            scores = scores[mask]
                        if len(scores) == 0:
                            return []
                        topk = min(k, len(scores))
                        top_idx_local = np.argsort(-scores)[:topk]
                        results = []
                        for rank, local_pos in enumerate(top_idx_local, start=1):
                            orig_idx = int(idxs[local_pos]) if filters else int(local_pos)
                            chunk = self.chunks[orig_idx]
                            results.append({"rank": rank, "file_name": chunk.get("file_name"), "chunk_id": chunk.get("chunk_id"), "text": chunk.get("text"), "score": float(scores[local_pos])})
                        return results

                retriever = SimpleRetriever(metadata_path=METADATA_CSV)
                orchestrator = RAGOrchestrator(retriever)
            except Exception:
                retriever = None
                orchestrator = None
    else:
        retriever = None
        orchestrator = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    # initialize lazily on first request
    if orchestrator is None:
        _init_orchestrator_if_possible()
    if orchestrator is None or orchestrator.retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not configured")
    result = orchestrator.answer(req.query, req.filters or {})
    return QueryResponse(answer=result["answer"], sources=result["sources"], raw=result.get("raw"))
