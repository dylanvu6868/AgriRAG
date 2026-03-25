"""
DeepRAG Engine — Groq LLM (openai/gpt-oss-120b)
"""

import json
from typing import List, Dict, Generator

from groq import Groq

from config import (
    GROQ_API_KEYS, GROQ_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P,
)

# ─── System Prompt ────────────────────────────────────────────────────────────

DEEPRAG_SYSTEM_PROMPT = """
Bạn là DeepRAG — Trợ lý AI chuyên gia phân tích và trả lời câu hỏi dựa trên tài liệu.
NHIỆM VỤ TỐI THƯỢNG CỦA BẠN: CHỈ trả lời dựa trên thông tin được cung cấp trong phần CONTEXT_CHUNKS.

═══════════════════════════════════════════════
QUY TRÌNH THỰC HIỆN:
═══════════════════════════════════════════════

## BƯỚC 1 — PHÂN TÍCH CÂU HỎI
Xác định ý định người dùng và chia nhỏ các vấn đề cần tra cứu.

## BƯỚC 2 — ĐÁNH GIÁ TÀI LIỆU
Đọc các CONTEXT_CHUNKS được cung cấp. Xác định đoạn nào liên quan trực tiếp đến câu hỏi.

## BƯỚC 3 — SUY LUẬN & TỔNG HỢP
1. Trích xuất sự thật từ tài liệu.
2. NẾU KHÔNG TÌM THẤY THÔNG TIN LIÊN QUAN TRONG CONTEXT_CHUNKS: Dừng suy luận lập tức, và kết luận là không có thông tin.

## BƯỚC 4 — TẠO CÂU TRẢ LỜI
- Trả lời trực tiếp và súc tích bằng Tiếng Việt.
- NGUYÊN TẮC THÉP: Nếu tài liệu thiếu thông tin, BẮT BUỘC trả lời: "Tài liệu hiện tại không cung cấp thông tin này" hoặc "Tôi không tìm thấy nội dung liên quan trên tài liệu". KHÔNG ĐƯỢC BỊA ĐẶT (Hallucinate) kiến thức từ bên ngoài.

═══════════════════════════════════════════════
ĐỊNH DẠNG ĐẦU RA (Output Format - Strict JSON)
═══════════════════════════════════════════════
Tuyệt đối chỉ trả về cục JSON hợp lệ theo đúng format dưới đây. Không in thêm bất kỳ dòng text nào khác bên ngoài JSON. KHÔNG DÙNG MARKDOWN BLOCK (```json).

{
  "query_analysis": {
    "intent": "<một câu mô tả ý định>",
    "sub_questions": [
       { "id": 1, "question": "<câu hỏi phụ>" }
    ]
  },
  "reasoning": {
    "steps": [
      "<Bước suy luận 1>",
      "<Bước suy luận 2>"
    ]
  },
  "answer": {
    "overall_confidence": "high|medium|low",
    "confidence_reason": "<lý do đánh giá độ tin cậy>",
    "sources": [
      { "chunk_id": "<id>", "document": "<tên file>", "section": "<mục>" }
    ],
    "text": "<câu trả lời cuối cùng, ngôn ngữ Tiếng Việt, trình bày markdown rõ ràng ở đây>"
  }
}
"""

# ─── Groq Client Manager (Rotation) ──────────────────────────────────────────
_key_index = 0
_clients = [Groq(api_key=k) for k in GROQ_API_KEYS]

def get_current_client():
    global _key_index
    return _clients[_key_index % len(_clients)]

def rotate_key():
    global _key_index
    _key_index += 1
    new_idx = _key_index % len(_clients)
    print(f"   🌾 Reasoning... (Switching Key #{new_idx + 1})")
    return _clients[new_idx]


# ─── Context builder ──────────────────────────────────────────────────────────

def build_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks for injection into the prompt."""
    lines = []
    for c in chunks:
        disease = c.get('disease_name', '') or ''
        sub     = c.get('subsection_name', '') or ''
        loc     = f"{disease} {sub}".strip() if disease else f"p.{c.get('page', '?')}"
        lines.append(
            f"[CHUNK {c['chunk_id'][:8]}] source={c['source']} location={loc} score={c['score']}\n"
            f"{c['text']}\n"
        )
    return "\n---\n".join(lines)


# ─── Query (streaming, yields tokens) ────────────────────────────────────────

def stream_deeprag(question: str, chunks: List[Dict], image_context: str = None) -> Generator[str, None, None]:
    """
    Yields raw tokens from Groq as they arrive.
    The full response is expected to be valid JSON.
    """
    context = build_context(chunks)
    
    img_prompt = ""
    if image_context:
        img_prompt = f"IMAGE_DIAGNOSIS_CONTEXT:\n{image_context}\n\n"
        
    user_content = (
        f"{img_prompt}"
        f"CONTEXT_CHUNKS:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )

    import groq

    for attempt in range(len(_clients)):
        try:
            client = get_current_client()
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": DEEPRAG_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                top_p=LLM_TOP_P,
                stream=True,
                stop=None,
            )

            for chunk in completion:
                token = chunk.choices[0].delta.content or ""
                yield token
            return # Success!

        except groq.RateLimitError:
            rotate_key()
            if attempt == len(_clients) - 1:
                yield "❌ Tất cả các API Key đều đã hết hạn mức. Vui lòng thử lại sau vài phút."
        except Exception as e:
            yield f"❌ Lỗi AI: {str(e)}"
            break


def query_deeprag(question: str, chunks: List[Dict]) -> Dict:
    """
    Non-streaming call. Returns parsed JSON dict.
    Falls back to {"raw": <text>} if JSON parsing fails.
    """
    full = "".join(stream_deeprag(question, chunks))
    try:
        # Strip potential markdown code fences
        text = full.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": full}


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dummy_chunks = [
        {
            "chunk_id": "abc12345-0000-0000-0000-000000000000",
            "source": "Benh_lua-1-33.pdf",
            "page": 1,
            "score": 0.91,
            "text": "Bệnh đạo ôn (blast) do nấm Magnaporthe oryzae gây ra...",
        }
    ]
    result = query_deeprag("Bệnh đạo ôn là gì?", dummy_chunks)
    import pprint
    pprint.pprint(result)
