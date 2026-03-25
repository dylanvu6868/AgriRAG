"""
DeepRAG — FastAPI backend
Serves the HTML UI + /api/chat (SSE streaming) + /api/stats
"""

import os
from pathlib import Path

# Override HuggingFace cache directory to use D drive instead of full C drive
os.environ["HF_HOME"] = str(Path(__file__).parent / "hf_cache")

import json
import asyncio
import shutil
import subprocess
from typing import AsyncGenerator, List

from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retriever import Retriever
from rag_engine import stream_deeprag
import database
import log_interactions

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="DeepRAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

# ─── Retriever (loaded once at startup) ───────────────────────────────────────

_retriever: Retriever | None = None

def load_retriever():
    global _retriever
    try:
        _retriever = Retriever()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Retriever load error: {e}")

load_retriever()
database.init_db()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/api/stats")
async def stats():
    if _retriever is None:
        return JSONResponse({"error": "Index not built. Run ingest.py first."}, status_code=503)
    s = _retriever.collection_stats()
    return JSONResponse(s)


@app.post("/api/upload-doc")
def upload_doc(file: UploadFile = File(...)):
    import json
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from config import PDF_DIR, FAISS_INDEX, FAISS_META, BM25_INDEX, EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_DIM
    from ingest import process_pdf, tokenize_vn
    from pathlib import Path

    # 1. Save the uploaded file
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    file_path = PDF_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 2. Extract + chunk the NEW file
        records = process_pdf(file_path)
        if not records:
            return JSONResponse({"error": "Không trích xuất được nội dung từ file."}, status_code=400)

        # 3. Load existing metadata (if any) and REMOVE old chunks for this source (dedup)
        existing_meta = []
        if Path(FAISS_META).exists():
            with open(FAISS_META, "r", encoding="utf-8") as mf:
                existing_meta = json.load(mf)

        old_count = len(existing_meta)
        # Filter out any chunk that came from the same file
        existing_meta = [m for m in existing_meta if m.get("source") != file.filename]
        removed_count = old_count - len(existing_meta)
        if removed_count > 0:
            print(f"   🗑  Removed {removed_count} old chunks for '{file.filename}' (dedup)")

        # 4. Merge remaining + new records
        all_meta = existing_meta + records

        # 5. Re-embed ALL chunks (use embed_text if available, else text — consistent with ingest.py)
        embedder = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
        all_embed_texts = [m.get("embed_text", m["text"]) for m in all_meta]
        all_vectors = embedder.encode(
            all_embed_texts, show_progress_bar=False, batch_size=32, normalize_embeddings=True
        ).astype(np.float32)

        # 6. Rebuild FAISS index from scratch (clean — no duplicate vectors)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(all_vectors)

        # 7. Save back to disk
        faiss.write_index(index, str(FAISS_INDEX))
        with open(FAISS_META, "w", encoding="utf-8") as mf:
            json.dump(all_meta, mf, ensure_ascii=False)

        # 8. Rebuild BM25 index
        from rank_bm25 import BM25Okapi
        import pickle
        tokenized_corpus = [tokenize_vn(m.get("embed_text", m["text"])) for m in all_meta]
        bm25 = BM25Okapi(tokenized_corpus)
        with open(BM25_INDEX, "wb") as bf:
            pickle.dump(bm25, bf)

        # 9. Hot-reload retriever with new index
        global _retriever
        _retriever = Retriever()

        return {
            "status": "ok",
            "message": (
                f"✅ Đã thêm '{file.filename}' — {len(records)} chunks mới"
                + (f" (xóa {removed_count} chunks cũ)" if removed_count else "")
                + f", tổng {index.ntotal} vectors."
            )
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse({"error": f"Lỗi xử lý: {str(e)}"}, status_code=500)



# --- Chat History Endpoints ---

@app.get("/api/sessions")
def get_sessions():
    return database.get_sessions()

@app.post("/api/sessions")
def create_session():
    # We create an empty session
    sid = database.create_session("Đoạn chat mới")
    return {"session_id": sid}

@app.get("/api/sessions/{session_id}")
def get_session_messages(session_id: str):
    return database.get_messages(session_id)

@app.get("/api/sessions/{session_id}/messages")
def get_session_messages_v2(session_id: str):
    return database.get_messages(session_id)

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    database.delete_session(session_id)
    return {"status": "ok"}


# Global state to hold the latest image prediction
_latest_image_context = None

@app.post("/api/upload-image")
def upload_image(file: UploadFile = File(...)):
    global _latest_image_context
    try:
        image_bytes = file.file.read()
        
        # Save placeholder for reference locally
        img_dir = Path("data/images")
        img_dir.mkdir(parents=True, exist_ok=True)
        with open(img_dir / file.filename, "wb") as f:
            f.write(image_bytes)
            
        # Run inference
        from vision_engine import predict_disease
        pred = predict_disease(image_bytes)
        
        if "error" in pred:
            return JSONResponse({"status": "error", "error": pred["error"]}, status_code=500)
            
        vi_label = pred["vietnamese_label"]
        conf = pred["confidence"]
        inference_time_s = pred.get("inference_time_s", None)
        low_conf = pred.get("low_confidence", False)
        top3 = pred.get("top3", [])
        
        # Build image context — include top-2 when uncertain
        if low_conf and len(top3) >= 2:
            ctx_parts = " hoặc ".join(
                f"{c['vietnamese_label']} ({c['confidence']*100:.0f}%)" for c in top3[:2]
            )
            _latest_image_context = f"Ảnh chụp có thể cho thấy triệu chứng của {ctx_parts} — mức độ tin cậy thấp, cần xem xét thêm."
        else:
            _latest_image_context = f"Ảnh chụp cho thấy triệu chứng giống {vi_label} (độ tin cậy {conf*100:.1f}%)."
        
        timing_str = f" ({inference_time_s:.2f}s)" if inference_time_s is not None else ""
        conf_warn = " ⚠️ Độ tin cậy thấp — xem top-3 bên dưới." if low_conf else ""
        
        return {
            "status": "ok",
            "message": f"AI nhận diện: {vi_label.upper()} ({conf*100:.1f}%){timing_str}.{conf_warn} Khách có thể tiếp tục đặt câu hỏi về bệnh này!",
            "inference_time_s": inference_time_s,
            "low_confidence": low_conf,
            "top3": top3,
        }
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)



@app.post("/api/chat")
async def chat(request: Request):
    global _latest_image_context
    """
    SSE endpoint. Streams:
      data: {"type": "chunk", "text": "..."}
      data: {"type": "done", "result": {...}, "chunks": [...]}
      data: {"type": "error", "message": "..."}
    """
    if _retriever is None:
        return JSONResponse({"error": "Index not built."}, status_code=503)

    body       = await request.json()
    question   = body.get("question", "").strip()
    top_k      = int(body.get("top_k", 5))
    session_id = body.get("session_id", None)
    is_new_session = body.get("is_new_session", False)

    if not question:
        return JSONResponse({"error": "Empty question."}, status_code=400)

    # Database logic
    if session_id:
        # Auto update title based on first query (limit to 30 chars)
        if is_new_session:
            title = question[:30] + '...' if len(question) > 30 else question
            # Ensure no image contexts clutter the title
            database.update_session_title(session_id, title.replace("Hãy chẩn đoán bệnh dựa trên ảnh này.", "Chẩn đoán ảnh"))
        database.add_message(session_id, "user", question)

    # Capture and reset the global image context
    img_ctx = _latest_image_context
    _latest_image_context = None

    search_query = question
    if img_ctx:
        search_query = f"{question} {img_ctx}"
        # Gài thêm lệnh bắt LLM phải xuất Nhãn bệnh ra trước khi trả lời
        question = (
            f"THÔNG TIN ẢNH: {img_ctx}\n"
            f"CÂU HỎI CỦA NGƯỜI DÙNG: {question}\n\n"
            f"YÊU CẦU ĐẶC BIỆT: Hãy mở đầu câu trả lời bằng việc tóm tắt lại chẩn đoán bệnh từ ảnh (ví dụ: 'Dựa vào phân tích ảnh, cây đang có dấu hiệu...'). Sau đó, đưa ra các biện pháp phòng trừ và điều trị chi tiết dựa trên tài liệu theo yêu cầu của người dùng."
        )
        
    retrieved = _retriever.retrieve(search_query, top_k=top_k)

    async def event_stream() -> AsyncGenerator[str, None]:
        import re
        full_text = ""
        last_extracted_len = 0
        
        # Regex to capture content inside "answer": { "text": "..." }
        pattern = re.compile(r'"answer"\s*:\s*\{.*?"text"\s*:\s*"((?:[^"\\]|\\.)*)', re.DOTALL)

        try:
            for token in stream_deeprag(question, retrieved, image_context=img_ctx):
                full_text += token
                
                # Check if we have reached the answer text
                if '"answer"' in full_text and '"text"' in full_text.split('"answer"')[-1]:
                    # Extract the current value of answer.text
                    match = pattern.search(full_text)
                    if match:
                        current_text = match.group(1)
                        # Unescape basic JSON characters for display
                        current_text = current_text.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                        
                        # Find what's new
                        if len(current_text) > last_extracted_len:
                            new_chunk = current_text[last_extracted_len:]
                            last_extracted_len = len(current_text)
                            
                            payload = json.dumps({"type": "chunk", "text": new_chunk}, ensure_ascii=False)
                            yield f"data: {payload}\n\n"
                            
                await asyncio.sleep(0)

            # Parse final JSON
            text = full_text.strip()
            # Robust JSON extractor
            text = full_text.strip()
            # Remove markdown code fences if present
            if text.startswith("```"):
                lines = text.split('\n')
                if len(lines) > 2:
                    text = "\n".join(lines[1:-1])
            
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                text = text[start_idx:end_idx+1]

            try:
                result = json.loads(text, strict=False)
                
                # Save AI answer to DB
                if session_id:
                     final_text = ""
                     if "answer" in result and "text" in result["answer"]:
                         final_text = result["answer"]["text"]
                     elif "raw" in result:
                         final_text = result["raw"]
                         
                     # Construct html representation if wanted, or just raw text. For this we will save the raw output.
                     database.add_message(session_id, "ai", json.dumps({"result": result, "chunks": retrieved}, ensure_ascii=False))
                
                # Log to CSV
                try:
                    log_question = body.get("question", "").strip() if hasattr(body, 'get') else question
                    log_interactions.log_interaction(
                        question=question,
                        answer=final_text if session_id else (result.get("answer", {}).get("text") or result.get("raw", "")),
                    )
                except Exception as log_err:
                    print(f"[log_interactions] Error: {log_err}")

            except json.JSONDecodeError:
                # Fallback: remove all unescaped control characters (ASCII 0-31) that break JSON strings
                import re
                cleaned_text = re.sub(r'[\x00-\x1F]+', ' ', text)
                try:
                    result = json.loads(cleaned_text)
                except json.JSONDecodeError:
                    result = {"raw": full_text}
                if session_id:
                     database.add_message(session_id, "ai", json.dumps({"result": result, "chunks": []}, ensure_ascii=False))

            done_payload = json.dumps(
                {"type": "done", "result": result, "chunks": retrieved},
                ensure_ascii=False
            )
            yield f"data: {done_payload}\n\n"

        except Exception as e:
            err_payload = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
