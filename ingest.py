"""
DeepRAG — PDF Ingestion Pipeline (FAISS + Disease-Based Chunking)

Strategy for Benh_lua-1-33.pdf:
  - Detect numbered disease sections: "N. Bệnh ..." or "N. Tuyến trùng ..."
  - Detect subsections: "N.M. <title>"
  - Each (disease, subsection) pair becomes ONE chunk
  - Full disease text (all subsections merged) becomes an extra summary chunk

Other PDFs fall back to sliding-window chunking.
"""

import os
from pathlib import Path
os.environ["HF_HOME"] = str(Path(__file__).parent / "hf_cache")

import json
import re
import uuid
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import faiss
import fitz
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer

from config import (
    PDF_DIR, FAISS_DIR, FAISS_INDEX, FAISS_META, BM25_INDEX,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_DEVICE,
)

def tokenize_vn(text: str) -> List[str]:
    """Tokenize text for BM25 (lowercase + PyVi segmentation)"""
    return ViTokenizer.tokenize(text.lower()).split()


# ─── PDF text extraction ──────────────────────────────────────────────────────

def extract_full_text_with_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Returns list of (page_number, page_text)."""
    doc = fitz.open(str(pdf_path))
    result = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            result.append((i + 1, text))
    doc.close()
    return result

# ─── Hierarchy-based Chunking (No overlap) ──────────────────────────────────
RE_H0 = re.compile(r"^PH[ỤU]\s*L[ỤU]C\s+\d+", re.IGNORECASE)
RE_H1 = re.compile(r"^(\d+|[IVI]+)\.\s+\S|^B[ỆE]NH\s+", re.IGNORECASE)
RE_H2 = re.compile(r"^(\d+\.\d+)\.?\s+\S")
RE_H3 = re.compile(r"^([a-zđ]\)|-|\+)\s+\S", re.IGNORECASE)

def heading_level(line: str):
    s = line.strip()
    if RE_H0.match(s): return 0
    if RE_H2.match(s): return 2
    if RE_H1.match(s): return 1
    if RE_H3.match(s): return 3
    return None

class Section:
    def __init__(self, level: int, title: str, parent=None):
        self.level    = level
        self.title    = title
        self.parent   = parent
        self.body     = []
        self.children = []

    def body_text(self) -> str:
        return " ".join(self.body).strip()

    def breadcrumb(self) -> list[str]:
        if self.parent is None or self.parent.level == -1:
            return [self.title] if self.title != "ROOT" else []
        return self.parent.breadcrumb() + [self.title]

    def disease_name(self) -> str:
        crumb = self.breadcrumb()
        for c in crumb:
            if RE_H1.match(c.strip()):
                return c.strip()
        return crumb[0] if crumb else ""

def parse_sections(text: str) -> Section:
    root = Section(-1, "ROOT")
    stack = [root]

    # Clean text to preserve line structure better
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)
    text = re.sub(r"\n\s*-{3,}\s*\n",   "\n", text)
    text = re.sub(r"[ \t]+",            " ",  text)
    text = re.sub(r"\n{3,}",            "\n\n", text)
    
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line: continue
        lvl = heading_level(line)
        if lvl is not None:
            while len(stack) > 1 and stack[-1].level >= lvl:
                stack.pop()
            parent = stack[-1]
            sec = Section(lvl, line, parent=parent)
            parent.children.append(sec)
            stack.append(sec)
        else:
            stack[-1].body.append(line)
    return root

MIN_CHARS = 120
MAX_CHARS = 1400

def _add_chunk(target_list, title: str, body: str, crumb: list[str], disease: str):
    if len(body) < 30: return
    content = f"{title}\n{body}" if title else body
    target_list.append({
        "disease_name": disease,
        "subsection_name": " > ".join(crumb[1:]) if len(crumb) > 1 else "[Full]",
        "text": content,
    })

def _split_long(target_list, title: str, body: str, crumb: list[str], disease: str):
    if len(body) <= MAX_CHARS:
        _add_chunk(target_list, title, body, crumb, disease)
        return
    sentences = re.split(r"(?<=[.!?])\s+", body)
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > MAX_CHARS and current:
            _add_chunk(target_list, title, current.strip(), crumb, disease)
            current = sent + " "
        else:
            current += sent + " "
    if current.strip():
        _add_chunk(target_list, title, current.strip(), crumb, disease)

def extract_advanced_chunks(root: Section) -> List[Dict]:
    raw_chunks = []
    def visit(sec: Section):
        if sec.level == -1:
            for child in sec.children: visit(child)
            return

        crumb   = sec.breadcrumb()
        disease = sec.disease_name()
        body    = sec.body_text()
        is_leaf = len(sec.children) == 0

        if is_leaf:
            if body: _split_long(raw_chunks, sec.title, body, crumb, disease)
        else:
            if body: _add_chunk(raw_chunks, sec.title, body, crumb, disease)
            for child in sec.children: visit(child)
    
    visit(root)
    
    # Merge small chunks
    merged = []
    for c in raw_chunks:
        if (merged and len(merged[-1]["text"]) < MIN_CHARS 
            and merged[-1]["disease_name"] == c["disease_name"]):
            merged[-1]["text"] += "\n" + c["text"]
        else:
            merged.append(c)
    return merged


# ─── Fallback sliding-window chunker ─────────────────────────────────────────

def sliding_window_chunks(pages: List[Tuple[int, str]]) -> List[Dict]:
    buffer, page_map, pos = "", [], 0
    pages_dict = {}
    for pg, text in pages:
        page_map.append((pos, pg))
        pages_dict[pg] = text
        buffer += text + "\n"
        pos += len(text) + 1

    results = []
    start = 0
    while start < len(buffer):
        end = min(start + CHUNK_SIZE, len(buffer))
        text = re.sub(r'\s+', ' ', buffer[start:end]).strip()
        if text:
            page_no = 1
            for char_pos, pg in page_map:
                if char_pos <= start:
                    page_no = pg
            results.append({
                'text': text, 
                'page': page_no, 
                'label': '', 
                'parent_text': pages_dict.get(page_no, text)
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return results


# ─── Build metadata list ──────────────────────────────────────────────────────

def process_pdf(pdf_path: Path) -> List[Dict]:
    pages = extract_full_text_with_pages(pdf_path)
    if not pages:
        return []

    records = []

    # Use disease chunking only for the rice disease PDF
    use_disease = 'benh' in pdf_path.name.lower() or 'bệnh' in pdf_path.name.lower()

    if use_disease:
        full_text = "\n".join(t for _, t in pages)
        root = parse_sections(full_text)
        disease_chunks = extract_advanced_chunks(root)

        if disease_chunks:
            print(f"   → Advanced Section-based: {len(disease_chunks)} chunks")
            for c in disease_chunks:
                text_content = c['text']
                # Xây dựng văn bản cấy meta
                injected_text = f"[Sách: {pdf_path.name} | Bệnh: {c['disease_name']}] {text_content}"
                records.append({
                    'chunk_id':       str(uuid.uuid4()),
                    'source':         pdf_path.name,
                    'page':           0,           # not page-specific (relies on hierarchy)
                    'disease_name':   c['disease_name'],
                    'subsection_name': c['subsection_name'],
                    'text':           injected_text,
                    'embed_text':     injected_text
                })
            return records
        else:
            print("   → Smart Sliding Window (Metadata + Parent-Child)")

    # Áp dụng thuật toán Smart Sliding Window
    sw = sliding_window_chunks(pages)
    if 'disease_chunks' not in locals() or not disease_chunks:
        print(f"   → Smart Sliding Window: {len(sw)} chunks")
    for c in sw:
        child_text = c['text']
        parent_text = c.get('parent_text', child_text)
        
        # Tiêm (Inject) siêu dữ liệu vào đầu chunk tìm kiếm
        injected_child = f"[Sách: {pdf_path.name} | Trang: {c['page']}] {child_text}"
        # Xây dựng Chunk Mẹ (Parent Chunk) để mớm cho LLM
        injected_parent = f"--- Bắt đầu trích đoạn từ Sách: {pdf_path.name} (Trang {c['page']}) ---\n{parent_text}\n--- Kết thúc trích đoạn ---"

        records.append({
            'chunk_id':       str(uuid.uuid4()),
            'source':         pdf_path.name,
            'page':           c['page'],
            'disease_id':     None,
            'disease_name':   '',
            'subsection_id':  '',
            'subsection_name': '',
            'text':           injected_parent,
            'embed_text':     injected_child
        })
    return records


# ─── Main ingestion ───────────────────────────────────────────────────────────

def ingest_all():
    print("🔧 Loading embedding model …")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)

    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in", PDF_DIR)
        return

    all_meta: List[Dict] = []
    all_texts: List[str] = []

    for pdf_path in pdf_files:
        print(f"\n📄 Processing: {pdf_path.name}")
        records = process_pdf(pdf_path)
        for r in records:
            embed_title = r.get('embed_text', r['text'])
            all_texts.append(embed_title)
            all_meta.append(r)

    print(f"\n🔢 Embedding {len(all_texts)} chunks …")
    vectors = embedder.encode(
        all_texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True
    )

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(np.array(vectors, dtype=np.float32))

    print(f"💾 Saving FAISS index → {FAISS_INDEX}")
    faiss.write_index(index, str(FAISS_INDEX))

    print(f"💾 Saving metadata  → {FAISS_META}")
    with open(FAISS_META, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    print(f"📊 Building BM25 index ({len(all_texts)} docs) …")
    tokenized_corpus = [tokenize_vn(t) for t in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"💾 Saving BM25 index → {BM25_INDEX}")
    with open(BM25_INDEX, "wb") as f:
        pickle.dump(bm25, f)

    print(f"\n✅ Done! FAISS: {index.ntotal} vectors | BM25: {len(tokenized_corpus)} docs — disease chunks: "
          f"{sum(1 for m in all_meta if m.get('disease_name') is not None)}")


if __name__ == "__main__":
    ingest_all()
