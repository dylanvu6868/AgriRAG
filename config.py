"""
DeepRAG — Configuration
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
PDF_DIR       = BASE_DIR / "pdf"
FAISS_DIR     = BASE_DIR / "faiss_db"      # FAISS index + metadata stored here
FAISS_INDEX   = FAISS_DIR / "index.faiss"
FAISS_META    = FAISS_DIR / "metadata.json"
BM25_INDEX    = FAISS_DIR / "bm25_index.pkl"

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# ─── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM    = 384
EMBEDDING_DEVICE = "cpu"   # change to "cuda" if GPU available

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 8   # Reduced from 15 — fewer chunks = less cross-disease noise

# ─── Groq / LLM ───────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Load from environment, fallback to a dummy key if not set
_keys_str = os.getenv("GROQ_API_KEYS", "")
GROQ_API_KEYS = [k.strip() for k in _keys_str.split(",") if k.strip()]

GROQ_MODEL        = "llama-3.3-70b-versatile"   # model hợp lệ trên Groq
LLM_TEMPERATURE   = 0.2   # Reduced from 0.5 — factual Q&A needs lower temperature
LLM_MAX_TOKENS    = 4096
LLM_TOP_P         = 0.9
