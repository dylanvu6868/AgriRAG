"""
DeepRAG — Retriever (FAISS)
Loads the FAISS index + metadata, embeds query, returns top-k chunks.
"""

import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from pyvi import ViTokenizer

from config import (
    FAISS_INDEX, FAISS_META, BM25_INDEX,
    EMBEDDING_MODEL, EMBEDDING_DEVICE,
    TOP_K,
)

# Thêm Cross-Encoder cho Re-ranking (Tiếng Việt)
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

class Retriever:
    _instance = None

    def __init__(self):
        if not Path(FAISS_INDEX).exists() or not Path(BM25_INDEX).exists():
            raise FileNotFoundError(
                f"Index not found. Please run `python ingest.py` first."
            )

        print("Loading retriever (Hybrid Search + Re-ranking) ...")
        self._embedder = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
        self._cross_enc = CrossEncoder(CROSS_ENCODER_MODEL, device=EMBEDDING_DEVICE)
        
        self._index = faiss.read_index(str(FAISS_INDEX))
        with open(FAISS_META, "r", encoding="utf-8") as f:
            self._meta: List[Dict] = json.load(f)
            
        with open(BM25_INDEX, "rb") as f:
            self._bm25 = pickle.load(f)
            
        print(f"   OK: {self._index.ntotal} vectors & BM25 loaded.")

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        Hybrid Retrieve: FAISS (Dense) + BM25 (Sparse) -> RRF Fusion -> CrossEncoder Rerank
        """
        # 1. FAISS Search (lấy danh sách top 20)
        retrieval_limit = max(top_k * 3, 20)
        vec = self._embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores_faiss, indices_faiss = self._index.search(vec, retrieval_limit)
        
        # 2. BM25 Search (lấy danh sách top 20)
        tokenized_query = ViTokenizer.tokenize(query.lower()).split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        indices_bm25 = np.argsort(bm25_scores)[::-1][:retrieval_limit]
        
        # 3. RRF (Reciprocal Rank Fusion)
        k = 60
        rrf_scores = {}
        for rank, idx in enumerate(indices_faiss[0]):
            if idx == -1: continue
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
            
        for rank, idx in enumerate(indices_bm25):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
            
        # Sắp xếp lại theo điểm RRF và lấy Top 15 để rerank
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # Lấy thông tin meta cho các chunk này
        candidate_meta = [self._meta[idx] for idx, _ in sorted_rrf]
        
        # 4. Cross-Encoder Re-ranking
        cross_inp = [[query, m["text"]] for m in candidate_meta]
        cross_scores = self._cross_enc.predict(cross_inp)
        
        # Sort candidates theo Cross-Encoder score
        reranked_indices = np.argsort(cross_scores)[::-1]
        
        results = []
        for i in reranked_indices[:top_k]:
            m = candidate_meta[i]
            results.append({
                "chunk_id":        m["chunk_id"],
                "source":          m["source"],
                "page":            m.get("page", 0),
                "disease_name":    m.get("disease_name", ""),
                "subsection_name": m.get("subsection_name", ""),
                "text":            m["text"],
                "score":           round(float(cross_scores[i]), 4),
            })
            
        return results

    def collection_stats(self) -> Dict:
        return {"points": self._index.ntotal, "vectors": self._index.ntotal}


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    r = Retriever()
    for i, c in enumerate(r.retrieve("Bệnh đạo ôn trên cây lúa", top_k=3), 1):
        label = c["disease_name"] or f"p.{c['page']}"
        sub   = c["subsection_name"] or ""
        print(f"\n[{i}] score={c['score']} | {c['source']} | {label} {sub}")
        print(c["text"][:250], "…")
