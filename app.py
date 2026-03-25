"""
DeepRAG — Streamlit UI
"""

import json
import streamlit as st

from retriever import Retriever
from rag_engine import stream_deeprag, build_context

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DeepRAG",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0d1117 100%); color: #e2e8f0; }
    section[data-testid="stSidebar"] { background: rgba(255,255,255,0.04); border-right: 1px solid rgba(255,255,255,0.08); }
    .answer-card { background: linear-gradient(135deg,rgba(99,179,237,0.08),rgba(159,122,234,0.08)); border:1px solid rgba(99,179,237,0.25); border-radius:16px; padding:24px 28px; margin:12px 0; line-height:1.75; }
    .badge-high   { background:#22543d; color:#9ae6b4; border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600; }
    .badge-medium { background:#744210; color:#fbd38d; border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600; }
    .badge-low    { background:#63171b; color:#feb2b2; border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600; }
    .source-chip { display:inline-block; background:rgba(99,179,237,0.12); border:1px solid rgba(99,179,237,0.3); border-radius:8px; padding:4px 10px; margin:3px; font-size:0.78rem; color:#90cdf4; }
    .step-bubble { background:rgba(255,255,255,0.04); border-left:3px solid #667eea; padding:8px 14px; margin:6px 0; border-radius:0 8px 8px 0; font-size:0.85rem; color:#cbd5e0; }
    .sub-q { background:rgba(167,139,250,0.12); border:1px solid rgba(167,139,250,0.3); border-radius:8px; padding:5px 12px; margin:4px 2px; display:inline-block; font-size:0.82rem; color:#c4b5fd; }
    .stChatInput textarea { background:#1e2333 !important; border:1px solid rgba(99,179,237,0.3) !important; }
    .stSpinner > div { border-top-color:#63b3ed !important; }
    h1,h2,h3 { color:#e2e8f0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session state ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    try:
        with st.spinner("Loading knowledge base …"):
            st.session_state.retriever = Retriever()
    except FileNotFoundError as e:
        st.session_state.retriever = None
        st.session_state.retriever_error = str(e)

retriever = st.session_state.get("retriever")

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌾 DeepRAG")
    st.markdown("*Hệ thống hỏi-đáp nông nghiệp dựa trên AI*")
    st.divider()

    if retriever:
        stats = retriever.collection_stats()
        st.metric("📚 Chunks trong KB", stats["points"])
    else:
        st.error("⚠️ Chưa có FAISS index.\nChạy `python ingest.py` rồi khởi động lại app.")

    top_k = st.slider("🔍 Số chunks lấy về (top-k)", 1, 10, 5)
    st.divider()

    if st.button("🗑 Xoá lịch sử chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<div style='font-size:0.75rem;color:#718096;'>"
        "Powered by <b>Groq</b> · openai/gpt-oss-120b<br>"
        "Embed: paraphrase-multilingual-MiniLM-L12-v2"
        "</div>",
        unsafe_allow_html=True,
    )

# ─── Main header ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style='text-align:center;padding:10px 0 4px;'>
        <span style='font-size:2.4rem;'>🌾</span>
        <h1 style='margin:4px 0;font-size:2rem;
                   background:linear-gradient(90deg,#63b3ed,#b794f4);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            DeepRAG
        </h1>
        <p style='color:#718096;font-size:0.9rem;margin:0;'>
            Hệ thống hỏi-đáp kiến thức nông nghiệp · Powered by Groq
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ─── No-index guard ───────────────────────────────────────────────────────────

if not retriever:
    st.warning(
        "**Chưa có FAISS index.**  \n"
        "Mở terminal và chạy:  \n"
        "```\npython ingest.py\n```\n"
        "Sau đó reload trang này."
    )
    st.stop()

# ─── Render helpers ───────────────────────────────────────────────────────────

CONF_BADGE = {
    "high":   '<span class="badge-high">✅ Cao</span>',
    "medium": '<span class="badge-medium">⚠️ Trung bình</span>',
    "low":    '<span class="badge-low">❌ Thấp</span>',
}


def chunk_label(c: dict) -> str:
    """Human-readable location label for a chunk."""
    disease = c.get("disease_name", "")
    sub     = c.get("subsection_name", "")
    if disease:
        return f"{disease} {sub}".strip()
    return f"{c['source']} p.{c.get('page', '?')}"


def render_deeprag_response(data: dict, chunks: list):
    """Render structured DeepRAG JSON response."""

    if "raw" in data:
        st.markdown(
            f'<div class="answer-card">{data["raw"]}</div>',
            unsafe_allow_html=True,
        )
        return

    qa         = data.get("query_analysis", {})
    reasoning  = data.get("reasoning", {})
    answer     = data.get("answer", {})
    confidence = answer.get("overall_confidence", "low")

    # Answer text
    st.markdown(
        f'<div class="answer-card">{answer.get("text", "")}</div>',
        unsafe_allow_html=True,
    )

    # Confidence badge
    badge = CONF_BADGE.get(confidence, CONF_BADGE["low"])
    st.markdown(
        f"**Độ tin cậy:** {badge} &nbsp;— {answer.get('confidence_reason', '')}",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Reasoning expander
    with st.expander("🧠 Phân tích câu hỏi & lập luận"):
        st.markdown("**Intent:** " + qa.get("intent", ""))
        st.markdown("**Sub-questions:**")
        for sq in qa.get("sub_questions", []):
            kws = ", ".join(sq.get("keywords", []))
            st.markdown(
                f'<span class="sub-q">Q{sq["id"]}: {sq["question"]}</span>'
                f'<span style="font-size:0.75rem;color:#718096;margin-left:6px;">[{kws}]</span>',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        for step in reasoning.get("steps", []):
            st.markdown(f'<div class="step-bubble">{step}</div>', unsafe_allow_html=True)
        if reasoning.get("contradictions"):
            st.markdown("**⚡ Mâu thuẫn:** " + "; ".join(reasoning["contradictions"]))
        if reasoning.get("gaps"):
            st.markdown("**🕳 Thiếu thông tin:** " + "; ".join(reasoning["gaps"]))

    # Sources expander
    with st.expander("📄 Nguồn tài liệu"):
        sources = answer.get("sources", [])
        if sources:
            for s in sources:
                st.markdown(
                    f'<span class="source-chip">📑 {s.get("document","?")} '
                    f'· {s.get("section","")}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("(Không có nguồn cụ thể)")

        st.markdown("---")
        st.markdown("**Top retrieved chunks:**")
        for c in chunks:
            label = chunk_label(c)
            st.markdown(
                f'<span class="source-chip">score {c["score"]} · {label}</span>',
                unsafe_allow_html=True,
            )
            st.caption(c["text"][:300] + "…")


# ─── Chat history ─────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_deeprag_response(msg["meta"]["data"], msg["meta"]["chunks"])

# ─── Chat input ───────────────────────────────────────────────────────────────

if prompt := st.chat_input("Đặt câu hỏi về cây trồng, bệnh lúa, … (Enter để gửi)"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("🔍 Tìm kiếm trong kho kiến thức …"):
        chunks = retriever.retrieve(prompt, top_k=top_k)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        placeholder.markdown("⏳ *DeepRAG đang suy luận …*")

        try:
            for token in stream_deeprag(prompt, chunks):
                full_text += token
                placeholder.markdown(
                    f"⏳ *Đang nhận phản hồi …* `({len(full_text)} chars)`"
                )

            placeholder.empty()

            text = full_text.strip()
            # Strip markdown code fences if model wraps in ```json ... ```
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:])
            if text.endswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[:-1])
            text = text.strip()

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {"raw": full_text}

            render_deeprag_response(data, chunks)

            st.session_state.messages.append({
                "role": "assistant",
                "content": "",
                "meta": {"data": data, "chunks": chunks},
            })

        except Exception as e:
            placeholder.error(f"❌ Lỗi khi gọi Groq API: {e}")
