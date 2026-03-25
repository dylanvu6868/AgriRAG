<h1 align="center">AgriRAG: AI Chatbot & Agricultural Disease Diagnosis 🌾</h1>

<p align="center">
  A system combining <b>Computer Vision</b> for plant disease detection and <b>Retrieval-Augmented Generation (RAG)</b> for accurate agricultural consultation powered by LLM (LLaMA-3 / Groq).
</p>

---

## 🚀 Key Features

1. **High-Accuracy Rice Disease Detection (Top-3 Inference):**
   - Scans and detects rice diseases (Blast, Blight, Sheath Blight, Thrips, etc.).
   - Automated Image Preprocessing: Applies **CLAHE** (Contrast Auto-Leveling) and **Sharpness (1.5x)** to enhance recognition accuracy even on blurry, dark, or field-captured smartphone photos.
   - Always returns the **Top 3 most probable diseases** along with a confidence level badge.

2. **"Anti-Hallucination" RAG (Hybrid Search):**
   - Utilizes both **FAISS (Vector Search)** and **BM25 (Keyword Search)** to accurately retrieve agricultural documents and care instructions (from the `pdf/` directory).
   - Smart Chunking: Hierarchical splitting based on document headings, cleanly separating `embed_text` (dense metadata for searching) from the original paragraph (context for the LLM).
   - Auto-Deduplication: Automatically filters duplicate vectors when updating or re-uploading documents via the UI.

3. **Ultra-Smooth UI (Streaming Enabled):**
   - The web interface is built with vanilla HTML/CSS/JS, ensuring lightweight performance and millisecond-level responsiveness.
   - Real-time Confidence Bars, Low Confidence Warnings, and expandable RAG analysis details (Sources/Chunks) right inside the chat window.
   - Supports **Direct Mobile Camera Capture** (`capture="environment"`).

4. **Self-Logging & Monitoring:** Automatically records interaction history, queries, responses, and vision model processing time into a CSV file (`data/interaction_log.csv`) to facilitate system review and accuracy evaluation.

---

## 📂 Project Structure

```text
📦 AgriRAG
 ┣ 📂 faiss_db/             # FAISS Vector Index (index.faiss) & search metadata
 ┣ 📂 pdf/                  # Input agricultural books/documents library
 ┣ 📂 static/               # Frontend Web Client (index.html, logo.png)
 ┣ 📂 data/                 # User interaction logs (CSV, SQLite)
 ┣ 📜 server.py             # Main FastAPI Backend starting point (UI Communication Port)
 ┣ 📜 app.py                # Secondary Streamlit UI (Alternative interface)
 ┣ 📜 rag_engine.py         # RAG Brain: Fetches Context + Injects Prompt into LLaMA-3
 ┣ 📜 vision_engine.py      # Image Preprocessing (CLAHE) & Keras Model Prediction
 ┣ 📜 retriever.py          # Embedding logic (Sentence-Transformers) + BM25 Search
 ┣ 📜 ingest.py             # Script to ingest PDFs, chunk data, and build faiss_db
 ┣ 📜 log_interactions.py   # Module responsible for auto-logging interactions to CSV
 ┣ 📜 best_rice_model.keras # Trained AI Computer Vision Model (EfficientNet)
 ┣ 📜 config.py             # Path settings, Model configurations, and Constants
 ┣ 📜 .env                  # Hidden file containing Groq API Keys (Prevents GitHub leaks)
 ┣ 📜 render.yaml           # Infrastructure-as-Code Blueprint for Cloud Deployment
 ┗ 📜 requirements.txt      # List of required Python dependencies
```

---

## ⚙️ Local Installation Guide

Requires **Python 3.9+** and **Git**.

### Step 1: Clone the repository & Create a Virtual Environment
```bash
git clone https://github.com/dylanvu6868/AgriRAG.git
cd AgriRAG
python -m venv .venv

# Activate the environment:
# - Windows:
.venv\Scripts\activate
# - Mac/Linux:
source .venv/bin/activate
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys
Create a file named `.env` in the root directory (same level as `server.py`). Add your Groq API key(s) here:
```env
GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3
```

### Step 4: Run the FastAPI Server (Start the App)
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
👉 Open your web browser and navigate to: [http://localhost:8000](http://localhost:8000)

*(Bonus: If you want to ingest a completely new PDF file into FAISS from scratch without using the UI Upload, you can run the standalone script: `python ingest.py`).*

---

## ☁️ Cloud Deployment Guide (Render.com)

The system is pre-configured with **Infrastructure-as-Code** for incredibly fast and free deployment via Render.com.

1. Sign up / Log in to [Render.com](https://render.com/) using your GitHub account.
2. In the top right corner, click **New** ➔ **Blueprint**.
3. Grant access and select your `dylanvu6868/AgriRAG` repository from the list.
4. Render will automatically read the configuration inside the `render.yaml` file.
5. When the Environment Variables prompt appears, paste your array of API Keys into the `GROQ_API_KEYS` variable.
6. Click **Apply** and wait approximately 5 minutes for the Render virtual server to download the dependencies. Once completed, a public URL will be generated, allowing you to access the chatbot from any smartphone!

---

*© AgriRAG System Architecture. For any issues or bugs, please review the log files in the `data/` directory prior to debugging image analysis.*
