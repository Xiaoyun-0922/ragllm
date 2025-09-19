from typing import List, Dict
import re
import numpy as np
from models.embedding import EmbeddingModel
from retrieval.vector_db import VectorDB
from my_config import config

class Retriever:
    def __init__(self, vector_db: VectorDB, embedding_model: EmbeddingModel):
        self.vector_db = vector_db
        self.embedding_model = embedding_model

        # Common microorganism/genus keywords (expandable)
        self._microbe_terms_seed = [
            "staphylococcus aureus", "s. aureus", "escherichia coli", "e. coli",
            "pseudomonas aeruginosa", "klebsiella pneumoniae", "acinetobacter baumannii",
            "salmonella", "enterococcus faecalis", "enterococcus faecium", "listeria",
            "bacillus", "mycobacterium", "streptococcus", "gram-positive", "gram-negative",
        ]

    def _extract_microbe_terms(self, text: str) -> List[str]:
        """Extract target microorganism keywords from query (simple heuristic)."""
        text_l = text.lower()
        hits = [t for t in self._microbe_terms_seed if t in text_l]
        # Simple genus+species pattern (e.g., "Staphylococcus aureus")
        pat = re.compile(r"\b([A-Z][a-z]+\s+[a-z]{3,})\b")
        for m in pat.findall(text):
            hits.append(m.lower())
        # Deduplicate while preserving order
        seen, result = set(), []
        for h in hits:
            if h not in seen:
                seen.add(h)
                result.append(h)
        return result

    def _build_expanded_queries(self, base_query: str, terms: List[str]) -> List[str]:
        """Build expanded sub-queries based on microorganism keywords (covering MIC/mechanism/hemolysis evidence)."""
        expansions: List[str] = []
        for t in terms[:2]:  # Take at most first 2 to control call volume
            expansions.extend([
                f"{t} antimicrobial peptide MIC",
                f"{t} MIC",
                f"{t} antimicrobial peptide mechanism",
                f"{t} antibacterial mechanism",
                f"{t} hemolysis",
            ])
        queries = [base_query]
        for q in expansions:
            if q not in queries:
                queries.append(q)
        return queries[:6]




    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        """Keyword-aware retrieval: sub-query expansion and weighted reranking around user-mentioned bacteria/genus."""
        if k is None:
            k = config.TOP_K

        terms = self._extract_microbe_terms(query)
        subqueries = self._build_expanded_queries(query, terms)
        per_query_k = max(k, min(10, k * 2))

        merged: Dict[tuple, Dict] = {}
        alpha = 0.05  # Keyword hit weighting

        for q in subqueries:
            emb = self.embedding_model.encode([q]).cpu().numpy()
            cand = self.vector_db.search(emb, per_query_k)
            for d in cand:
                key = (d.get("source"), d.get("chunk_id"))
                base_score = float(d.get("score", 0.0))
                text_l = d.get("text", "").lower()
                hit_cnt = sum(1 for t in terms if t in text_l)
                boosted = base_score + alpha * hit_cnt
                prev = merged.get(key)
                if (prev is None) or (boosted > prev["score"]):
                    dd = dict(d)
                    dd["base_score"] = base_score
                    dd["score"] = boosted
                    dd["keyword_hits"] = hit_cnt
                    merged[key] = dd

        results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return results[:k]
