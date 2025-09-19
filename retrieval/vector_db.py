import faiss
import numpy as np
from typing import List, Optional
from my_config import config

class VectorDB:
    def __init__(self, dimension: int, index: Optional[faiss.Index] = None, documents: Optional[List[dict]] = None):
        # Allow loading existing index from disk to avoid rebuilding each time
        self.index = index if index is not None else faiss.IndexFlatIP(dimension)
        self.documents = documents[:] if documents else []

    def add_documents(self, embeddings: np.ndarray, documents: List[dict]):
        """Add documents to vector database"""
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[dict]:
        """Search similar documents (threshold filtering + fallback: if all below threshold, return at least top-scoring ones)."""
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx >= 0 and score >= config.SIMILARITY_THRESHOLD:
                doc = dict(self.documents[idx])
                doc["score"] = float(score)
                results.append(doc)

        if not results:
            # Fallback: even if below threshold, return top-scoring ones to ensure upstream has candidates for full-text expansion
            for idx, score in zip(indices[0], distances[0]):
                if idx >= 0:
                    doc = dict(self.documents[idx])
                    doc["score"] = float(score)
                    results.append(doc)
                    if len(results) >= min(k, 3):
                        break

        results.sort(key=lambda x: x["score"], reverse=True)
        return results


    def save_index(self, path: str):
        faiss.write_index(self.index, path)


    def persist_index(self, path: str) -> None:
        """Persist index to disk"""
        faiss.write_index(self.index, path)

