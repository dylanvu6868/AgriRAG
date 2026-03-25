<h1 align="center">AgriRAG: AI Chatbot & Chẩn Đoán Bệnh Nông Nghiệp 🌾</h1>

<p align="center">
  Hệ thống kết hợp <b>Thị giác Máy tính (Computer Vision)</b> nhận diện hình ảnh dịch bệnh và hệ thống <b>Retrieval-Augmented Generation (RAG)</b> để tư vấn nông nghiệp chính xác với sự hỗ trợ của LLM (LLaMA-3 / Groq).
</p>

---

## 🚀 Tính Năng Nổi Bật

1. **Nhận Diện Bệnh Lúa Chính Xác Cao (Top-3 Inference):**
   - Đọc và phân loại hình ảnh bệnh lúa (Đạo ôn, Bạc lá, Khô vằn, Bọ trĩ, v.v.).
   - Khả năng tiền xử lý ảnh tự động: Áp dụng thuật toán cân bằng sáng **CLAHE** (Contrast Auto-Leveling) và **Sharpness (1.5x)** giúp tăng độ chính xác ngay cả với ảnh mờ, tối hoặc tự chụp ngoài đồng bằng điện thoại.
   - Luôn trả về xác suất của **3 bệnh khả nghi nhất** kèm thang đo mức độ tin cậy.

2. **RAG "Chống Ảo Giác" (Hybrid Search):**
   - Sử dụng kết hợp **FAISS (Vector Search)** và **BM25 (Keyword Search)** để truy xuất sách/hướng dẫn chăm sóc cây trồng (thư mục `pdf/`).
   - Tự động tách đoạn thông minh (Chunking) theo cấu trúc Mục lục của tài liệu, phân tách rạch ròi `embed_text` (dành cho tìm kiếm) và đoạn gốc dài (dành cho LLM đọc).
   - Tự động lọc trùng lặp Vector (Deduplication) khi cập nhật tài liệu mới qua UI.

3. **Giao Diện Siêu Mượt (Cơ chế Streaming):**
   - Giao diện Web được build bằng HTML/CSS/JS thuần, nhẹ và có tốc độ phản hồi tính bằng mili-giây.
   - Thanh tiến trình Confidence Bars, cảnh báo Độ tin cậy thấp và chi tiết phần phân tích RAG (Sources/Chunks) xem được trực tiếp trên khung chat.
   - Hỗ trợ gọi bật **Camera trực tiếp trên Điện thoại** (`capture="environment"`).

4. **Self-Logging & Giám Sát:** Tự động thu thập (Log) lại lịch sử tương tác, loại câu hỏi, câu trả lời và thời gian model xử lý hình ảnh vào file CSV (`data/interaction_log.csv`) nhằm phục vụ công tác rà soát, đánh giá độ chính xác của AI.

---

## 📂 Cấu Trúc Thư Mục Hệ Thống

```text
📦 AgriRAG
 ┣ 📂 faiss_db/             # Chứa Index Vector (index.faiss) & siêu dữ liệu tìm kiếm
 ┣ 📂 pdf/                  # Thư viện sách/tài liệu nông nghiệp đầu vào
 ┣ 📂 static/               # Giao diện Web Client (index.html, logo.png)
 ┣ 📂 data/                 # File log tương tác người dùng (CSV, SQLite)
 ┣ 📜 server.py             # File khởi chạy FastAPI Backend chính (Cổng giao tiếp UI)
 ┣ 📜 app.py                # Giao diện phụ chạy bằng Streamlit (nếu cần đổi giao diện)
 ┣ 📜 rag_engine.py         # Não bộ RAG: Lấy Context + Đẩy Prompt (System/User) vào LLaMA-3
 ┣ 📜 vision_engine.py      # Tiền xử lý ảnh (CLAHE) & dự đoán bằng Model Keras
 ┣ 📜 retriever.py          # Thuật toán nhúng (Sentence-Transformers) + BM25 Search
 ┣ 📜 ingest.py             # Script nạp PDF, băm nhỏ (chunking) và tạo faiss_db mới
 ┣ 📜 log_interactions.py   # Mô-đun phụ trách ghi log tự động ra file CSV
 ┣ 📜 best_rice_model.keras # Tệp Model AI Computer Vision đã train (EfficientNet)
 ┣ 📜 config.py             # Cấu hình đường dẫn, tên Model, và thiết lập Tokenizer
 ┣ 📜 .env                  # Tệp ẩn chứa API Key của Groq (tránh lộ bảo mật GitHub)
 ┣ 📜 render.yaml           # Mã kịch bản (Blueprint) cấu hình tự động Deploy lên Cloud
 ┗ 📜 requirements.txt      # Danh sách các thư viện cần cài đặt
```

---

## ⚙️ Hướng Dẫn Cài Đặt (Localhost)

Yêu cầu máy tính có cài sẵn **Python 3.9+** và **Git**.

### Bước 1: Clone mã nguồn & Tạo môi trường ảo
```bash
git clone https://github.com/dylanvu6868/AgriRAG.git
cd AgriRAG
python -m venv .venv

# Kích hoạt môi trường:
# - Trên Windows:
.venv\Scripts\activate
# - Trên Mac/Linux:
source .venv/bin/activate
```

### Bước 2: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 3: Cấu hình Khóa API (API Keys)
Tạo một file có tên là `.env` ở ngay thư mục gốc của dự án (cùng cấp với `server.py`). Điền khóa Groq API của bạn vào:
```env
GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3
```

### Bước 4: Khởi chạy máy chủ FastAPI (Bắt đầu sử dụng)
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
👉 Mở trình duyệt web của bạn và truy cập: [http://localhost:8000](http://localhost:8000)

*(Ngoài ra: Nếu bạn muốn nạp (ingest) lại một file PDF mới hoàn toàn từ đầu vào FAISS thay vì gọi Upload trên giao diện, bạn có thể chạy file độc lập `python ingest.py`).*

---

## ☁️ Hướng Dẫn Triển Khai Lên Đám Mây (Deploy Render.com)

Hệ thống đã được thiết kế sẵn mã kịch bản **Infrastructure-as-Code** để triển khai miễn phí cực kỳ nhanh chóng thông qua nền tảng Render.com.

1. Đăng ký / Đăng nhập [Render.com](https://render.com/) bằng tài khoản GitHub.
2. Trên góc phải màn hình, chọn **New** ➔ **Blueprint**.
3. Cấp quyền & lựa chọn Repository `dylanvu6868/AgriRAG` của bạn từ danh sách.
4. Render sẽ tự động đọc cấu hình trong file `render.yaml`. 
5. Tại bảng khai báo biến môi trường (Environment Variables) hiện ra, Hãy Paste mảng API Keys của bạn vào biến `GROQ_API_KEYS`.
6. Bấm **Apply** và chờ máy chủ ảo của Render tải về thư viện trong 5 phút. Khi hoàn tất, hệ thống sẽ cấp cho bạn đường dẫn URL public để bạn có thể truy cập bằng mọi điện thoại!

---

*© AgriRAG System Architecture. Mọi vấn đề lỗi hoặc thắc mắc vui lòng kiểm tra tại file nhật ký trong thư mục `data/` trước khi tiến hành fix lỗi phân tích ảnh.*
