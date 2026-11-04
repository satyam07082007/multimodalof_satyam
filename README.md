# 🌿 HAVEN AI — Multimodal Intelligent Assistant

> **"One AI. Every Possibility."**

Haven AI is a modern, full-featured, general-purpose multimodal AI platform inspired by ChatGPT. It combines text conversations, image understanding, real image generation, voice conversations, document intelligence (PDF/DOCX/PPTX/TXT), live web research, software engineering & debugging, interactive data analysis (CSV/Excel), custom RAG vector indexing, and dedicated emotional support & crisis safety.

---

## 🌐 Live Localhost Access Links

When the servers are running:
- 🌿 **Haven AI Web Studio (Streamlit)**: [http://localhost:8501](http://localhost:8501)
- ⚙️ **FastAPI Backend Root / Health**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- 📖 **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 📑 **FastAPI ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🌟 Capability Matrix

| Capability | Module | Description |
| :--- | :--- | :--- |
| **🤖 General Conversation** | `modules/chat_engine.py` | Multi-turn dialogues, structured markdown, LaTeX math, summaries, emails, language translation, context memory. |
| **🎨 Image Generation** | `modules/image_generation.py` | **Generates real images** via Neural Diffusion / Pollinations / FLUX / Hugging Face SDXL with in-chat preview, regeneration, and download. |
| **🖼️ Image Understanding** | `modules/image_understanding.py` | Object recognition, scene description, OCR text extraction, diagram interpretation, and code screenshot debugging. |
| **🎙️ Voice Input (STT)** | `modules/speech_to_text.py` | Voice recording transcription via Whisper / Gemini Multimodal audio. |
| **🔊 Natural Speech (TTS)** | `modules/text_to_speech.py` | Converts assistant text responses into soothing natural speech audio (`.mp3`) with player controls. |
| **📄 Document Intelligence** | `modules/document_processing.py` | Parses **PDF, DOCX, PPTX, TXT, CSV, XLSX**. Generates executive summaries, structured study notes, Q&A pairs, and MCQs. |
| **🌐 Web Research Engine** | `modules/web_search.py` | Real-time internet search via DuckDuckGo with verified source URLs and citations. |
| **💻 Code Studio & Debugger** | `modules/coding_assistant.py` | Multi-language code generation (Python, JS, Java, C++, SQL), line-by-line explanations, and syntax/logic debugging. |
| **📊 Data Analysis & Charts** | `modules/data_analysis.py` | Upload CSV/Excel datasets to inspect statistics, missing values, and generate interactive **Plotly** visualizations. |
| **📚 Custom Knowledge RAG** | `modules/rag_engine.py` | Session-isolated document chunking and TF-IDF semantic vector retrieval with document citations. |
| **🌿 Emotional Support Mode** | `modules/emotional_support.py` | Empathetic active listening, 5-4-3-2-1 grounding, Box Breathing, PMR, 988 Crisis Lifeline escalation, and medical boundaries. |

---

## 🏗️ System Architecture

```text
Streamlit Frontend (Port 8501)
  │
  ├── Sidebar: Mode Selector, Chat History, Multimodal Dropzone, Voice Toggle, Helplines
  │
  ├── Main Chat Canvas:
  │     ├── General AI Dialogue (Markdown, Code Blocks, LaTeX)
  │     ├── Generated Images with Direct Download Button
  │     ├── Interactive Plotly Analytical Charts
  │     ├── Audio Voice Responses (TTS .mp3)
  │     └── Collapsible Verified Citations
  ▼
Haven AI Modular Backend & Engine Layer
  ├── modules/chat_engine.py          (Gemini / Hugging Face / Intelligent Engine)
  ├── modules/image_generation.py     (Neural Diffusion / FLUX / SDXL)
  ├── modules/image_understanding.py  (Multimodal Vision & OCR)
  ├── modules/speech_to_text.py       (Speech Recognition)
  ├── modules/text_to_speech.py       (gTTS Speech Synthesis)
  ├── modules/document_processing.py  (PDF, DOCX, PPTX, XLSX, CSV Parser)
  ├── modules/web_search.py           (Live Internet Search with DuckDuckGo)
  ├── modules/coding_assistant.py     (Algorithmic & Code Studio)
  ├── modules/data_analysis.py        (Pandas & Plotly EDA)
  ├── modules/rag_engine.py           (Semantic Vector Store)
  └── modules/emotional_support.py    (Empathetic Care & 988 Safety)
  ▼
Persistence & Security
  ├── database: MySQL (+ SQLite Fallback for Instant Local Testing)
  └── utils/security.py (PII Redaction, Prompt-Injection Filter, File Validation)
```

---

## 📁 Updated Project Folder Structure

```text
antigravity ide/
├── frontend/
│   └── streamlit_app.py            # ChatGPT-style Streamlit UI with multi-capability drawer
├── modules/
│   ├── chat_engine.py              # Conversational reasoning engine
│   ├── image_understanding.py      # Vision, OCR, scene analysis
│   ├── image_generation.py         # Real text-to-image neural generation
│   ├── audio_processing.py         # Voice note processing & audio Q&A
│   ├── speech_to_text.py           # Speech recognition & transcription
│   ├── text_to_speech.py           # Audio synthesis (TTS)
│   ├── document_processing.py      # PDF, DOCX, PPTX, TXT, CSV, XLSX processor
│   ├── web_search.py               # Live web search & citation extractor
│   ├── coding_assistant.py         # Code generator & debugger
│   ├── data_analysis.py            # Dataset EDA & interactive Plotly charts
│   ├── rag_engine.py               # Dynamic session vector index
│   └── emotional_support.py        # Dedicated emotional support & crisis safety
├── utils/
│   ├── security.py                 # PII sanitizer, prompt-injection defense, 988 hotline
│   └── file_handler.py             # Server file validator
├── app/
│   ├── main.py                     # FastAPI REST API
│   ├── config.py                   # Pydantic settings & environment manager
│   ├── database.py                 # SQLAlchemy MySQL/SQLite manager
│   ├── models/                     # Relational database models
│   ├── schemas/                    # Pydantic schemas
│   ├── api/                        # Health, Chat, and Upload endpoints
│   └── services/                   # Backend orchestration services
├── tests/
│   ├── conftest.py                 # Offline mock test fixtures
│   └── test_haven_ai.py            # Comprehensive 11-capability verification suite
├── requirements.txt                # Pinned dependencies
├── .env.example                    # Configuration template
├── run_backend.bat                 # One-click FastAPI backend launcher
└── run_frontend.bat                # One-click Streamlit frontend launcher
```

---

## 🛠️ Installation & Setup Guide

### 1. Prerequisites
- **Python 3.10+** (Python 3.14 fully supported)
- Windows PowerShell or Command Prompt

### 2. Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### 4. Configure Credentials (`.env`)
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

| Variable | Required? | Purpose |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | Optional | Unlocks Gemini 1.5 Flash for vision, audio, and large-scale general reasoning. |
| `HUGGINGFACE_API_KEY` | Optional | Unlocks Hugging Face inference models (Llama 3, SDXL). |
| `MYSQL_*` | Optional | Connects to MySQL. If left blank, automatically uses SQLite fallback. |

*(Note: Haven AI is designed with built-in zero-key fallback engines so every feature runs immediately even if API keys are not yet configured).*

### 5. Run the Automated Tests
```powershell
.\.venv\Scripts\pytest.exe -v tests/test_haven_ai.py
```
*(All 11 capability tests pass out-of-the-box).*

---

## 🏃 Launch Commands

### Option A: Using Windows Batch Launchers
- Double-click **`run_frontend.bat`** to start Streamlit (port 8501).
- Double-click **`run_backend.bat`** to start FastAPI (port 8000).

### Option B: PowerShell Commands

**Terminal 1 (Streamlit Web Interface):**
```powershell
.\.venv\Scripts\streamlit.exe run frontend/streamlit_app.py --server.port 8501
```

**Terminal 2 (FastAPI Backend):**
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## ✅ Verified Testing Prompts (Checklist Results)

1. **General Chat**: `"Explain artificial intelligence in simple language."` → Delivers structured overview with pillars, table, and concepts.
2. **Image Understanding**: Upload drawing or photo → Returns dimensions, color profile, and visual breakdown.
3. **Image Generation**: `"Generate an image of a futuristic AI research laboratory."` → Invokes neural diffusion engine and renders high-definition image with direct download button.
4. **Audio / Voice Input**: Upload voice note → Transcribes audio recording into conversational context.
5. **Voice Output (TTS)**: Toggle "Read Responses Aloud" in sidebar → Synthesizes `.mp3` audio player inside chat.
6. **Document Intelligence**: Upload PDF/DOCX/PPTX/TXT → Generates executive summary, study notes, Q&A pairs, or MCQs.
7. **Web Search**: `"Find the latest developments in artificial intelligence."` → Queries DuckDuckGo and surfaces real-time citations.
8. **Coding Assistant**: `"Write a Python program to implement binary search."` → Generates clean `O(log n)` algorithm with line-by-line breakdown.
9. **Data Analysis**: Upload CSV/Excel → Inspects columns, missing values, and displays interactive Plotly charts.
10. **Custom RAG**: Indexes uploaded documents into vector store and answers with grounded excerpts.
11. **Emotional Support**: `"I am feeling stressed today. Can we talk?"` → Empathetic active listening, 5-4-3-2-1 grounding exercise, and 988 lifeline protection.
