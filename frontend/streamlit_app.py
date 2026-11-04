import sys
import os
import uuid
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

# Core Configuration
from app.config import settings

# Core Haven AI Modules
from modules.chat_engine import ChatEngine
from modules.image_understanding import ImageUnderstandingService
from modules.image_generation import ImageGenerationService
from modules.audio_processing import AudioProcessingModule
from modules.document_processing import DocumentProcessor
from modules.web_search import WebSearchModule
from modules.coding_assistant import CodingAssistantModule
from modules.data_analysis import DataAnalysisModule
from modules.rag_engine import DynamicRAGEngine
from modules.emotional_support import EmotionalSupportModule
from utils.security import sanitize_pii, check_crisis_intent

# Streamlit Page Setup
st.set_page_config(
    page_title="Haven AI | One AI. Every Possibility.",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Services
@st.cache_resource
def get_services():
    return {
        "chat": ChatEngine(),
        "vision": ImageUnderstandingService(),
        "image_gen": ImageGenerationService(),
        "audio": AudioProcessingModule(),
        "doc": DocumentProcessor(),
        "search": WebSearchModule(),
        "coder": CodingAssistantModule(),
        "data": DataAnalysisModule(),
        "rag": DynamicRAGEngine(),
        "care": EmotionalSupportModule()
    }

services = get_services()

# Custom Modern Styling: Premium Dark Theme & Clean Typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Background Aesthetics */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #080c14 0%, #0f172a 50%, #060910 100%);
        color: #f1f5f9;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.5) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(90deg, #34d399, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-tagline {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Mode Pill */
    .mode-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        margin-bottom: 12px;
    }

    /* Quick Action Chips */
    .quick-chip {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 8px;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #cbd5e1;
        font-size: 12px;
        margin: 4px;
        text-decoration: none;
    }

    /* Message Styling */
    [data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    /* Helpline Alert Box */
    .helpline-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(153, 27, 27, 0.12) 100%);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 12px;
        padding: 12px 14px;
        color: #fecaca;
        font-size: 13px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = f"chat_{uuid.uuid4().hex[:8]}"

if st.session_state.current_chat_id not in st.session_state.conversations:
    st.session_state.conversations[st.session_state.current_chat_id] = {
        "title": "New Conversation",
        "messages": []
    }

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "🤖 General Assistant"

if "enable_tts" not in st.session_state:
    st.session_state.enable_tts = False

current_chat = st.session_state.conversations[st.session_state.current_chat_id]

# Sidebar: Branding, Navigation, Mode Selection, Settings
with st.sidebar:
    st.markdown("### 🌿 HAVEN AI")
    st.caption("**One AI. Every Possibility.**")
    st.markdown(f'<div class="mode-badge">{st.session_state.app_mode}</div>', unsafe_allow_html=True)

    # New Chat
    if st.button("➕ New Chat", use_container_width=True):
        new_id = f"chat_{uuid.uuid4().hex[:8]}"
        st.session_state.conversations[new_id] = {"title": "New Conversation", "messages": []}
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.divider()

    # Capability Mode Selector
    st.markdown("#### 🧭 Intelligence Modes")
    selected_mode = st.radio(
        "Select Mode",
        [
            "🤖 General Assistant",
            "🎨 Image Generation Studio",
            "📄 Document Intelligence",
            "📊 Data Analysis & Charts",
            "🌐 Web Research Engine",
            "💻 Code Studio & Debugger",
            "📚 Custom Knowledge RAG",
            "🌿 Emotional Support & Care"
        ],
        index=[
            "🤖 General Assistant",
            "🎨 Image Generation Studio",
            "📄 Document Intelligence",
            "📊 Data Analysis & Charts",
            "🌐 Web Research Engine",
            "💻 Code Studio & Debugger",
            "📚 Custom Knowledge RAG",
            "🌿 Emotional Support & Care"
        ].index(st.session_state.app_mode),
        label_visibility="collapsed"
    )
    if selected_mode != st.session_state.app_mode:
        st.session_state.app_mode = selected_mode
        st.rerun()

    st.divider()

    # Multimodal Uploads Drawer
    st.markdown("#### 📎 Attachments & Media")
    uploaded_file = st.file_uploader(
        "Upload image, audio, document, or spreadsheet",
        type=["png", "jpg", "jpeg", "webp", "wav", "mp3", "pdf", "docx", "txt", "csv", "xlsx", "pptx"],
        key="main_uploader"
    )

    if uploaded_file:
        st.success(f"Loaded: `{uploaded_file.name}` ({round(uploaded_file.size / 1024, 1)} KB)")

    st.divider()

    # Settings & Audio Controls
    with st.expander("⚙️ Controls & Preferences", expanded=False):
        st.session_state.enable_tts = st.toggle("🔊 Read Responses Aloud (TTS)", value=st.session_state.enable_tts)
        st.markdown(f"**Session ID**: `{st.session_state.current_chat_id}`")
        if st.button("🗑️ Clear Current Chat", use_container_width=True):
            current_chat["messages"] = []
            st.rerun()

    # Crisis Hotline (Accessible in sidebar)
    with st.expander("🆘 Crisis & Safety Helplines", expanded=False):
        st.markdown("""
        <div class="helpline-box">
            <strong>If you or someone you know needs support:</strong><br><br>
            • <strong>988 Lifeline</strong>: Call or text <code>988</code> (24/7, Free, US & Canada)<br>
            • <strong>Crisis Text Line</strong>: Text <code>HOME</code> to <code>741741</code><br>
            • <strong>International</strong>: <a href="https://www.befrienders.org" target="_blank" style="color:#fca5a5;">befrienders.org</a>
        </div>
        """, unsafe_allow_html=True)

# Main Hero Header
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🌿 HAVEN AI</div>
    <div class="hero-tagline">One AI. Every Possibility. — Multimodal Intelligence, Vision, Coding, Research & Care.</div>
</div>
""", unsafe_allow_html=True)

# Display Dialogue Messages
for msg in current_chat["messages"]:
    role = msg["role"]
    avatar = "🌱" if role == "user" else "🌿"
    with st.chat_message(role, avatar=avatar):
        # Render prompt text
        st.markdown(msg["content"])

        # Render generated or attached image
        if msg.get("image_path") and Path(msg["image_path"]).exists():
            st.image(str(msg["image_path"]), caption="Generated / Attached Image", use_container_width=True)
            with open(msg["image_path"], "rb") as img_f:
                st.download_button(
                    label="⬇️ Download Image",
                    data=img_f.read(),
                    file_name=Path(msg["image_path"]).name,
                    mime="image/png",
                    key=f"dl_{msg.get('image_path')}"
                )

        # Render interactive Plotly chart if present
        if msg.get("chart_fig"):
            st.plotly_chart(msg["chart_fig"], use_container_width=True)

        # Render audio synthesis
        if msg.get("audio_path") and Path(msg["audio_path"]).exists():
            st.audio(str(msg["audio_path"]))

        # Render source citations
        if msg.get("sources"):
            with st.expander("📘 Verified Knowledge & Web Citations", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f"**{s.get('document_name', 'Source')}** — {s.get('chunk_reference', '')}\n> {s.get('excerpt', '')}")

# Chat Input Composer
chat_input = st.chat_input("Ask a question, request code, generate an image, or share your thoughts...")

# Process Input
if chat_input or (uploaded_file and st.button("🚀 Process Uploaded File", key="process_btn")):
    user_prompt = chat_input or f"Analyze uploaded file: {uploaded_file.name}"
    active_mode = st.session_state.app_mode

    # Auto-title conversation on first message
    if len(current_chat["messages"]) == 0:
        current_chat["title"] = user_prompt[:30] + ("..." if len(user_prompt) > 30 else "")

    # Append user bubble
    current_chat["messages"].append({"role": "user", "content": user_prompt})

    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("Haven AI is thinking and formulating response..."):
            reply_text = ""
            generated_img_path = None
            generated_audio_path = None
            chart_fig = None
            citations = []

            # ----------------------------------------------------
            # 1. IMAGE GENERATION STUDIO
            # ----------------------------------------------------
            if active_mode == "🎨 Image Generation Studio" or any(p in user_prompt.lower() for p in ["generate an image", "create an image", "draw a ", "generate a picture"]):
                clean_img_prompt = user_prompt.replace("generate an image of", "").replace("create an image of", "").replace("generate an image", "").strip()
                if not clean_img_prompt:
                    clean_img_prompt = "futuristic AI research laboratory with holographic neural interfaces"
                
                success, img_path, msg = services["image_gen"].generate_image(clean_img_prompt)
                if success and img_path:
                    generated_img_path = str(img_path)
                    reply_text = f"### 🎨 Image Generated Successfully\n\n**Prompt**: *\"{clean_img_prompt}\"*\n\n{msg}"
                else:
                    reply_text = f"⚠️ Image generation error: {msg}"

            # ----------------------------------------------------
            # 2. WEB RESEARCH ENGINE
            # ----------------------------------------------------
            elif active_mode == "🌐 Web Research Engine" or any(p in user_prompt.lower() for p in ["search the web", "latest news on", "find latest"]):
                search_res = services["search"].search(user_prompt, max_results=4)
                reply_text = services["search"].synthesize_research_report(user_prompt, search_res)
                citations = [{"document_name": r.get("title"), "chunk_reference": r.get("href"), "excerpt": r.get("body")} for r in search_res]

            # ----------------------------------------------------
            # 3. CODING ASSISTANT
            # ----------------------------------------------------
            elif active_mode == "💻 Code Studio & Debugger" or any(p in user_prompt.lower() for p in ["write a program", "write code", "debug this", "binary search"]):
                if "debug" in user_prompt.lower():
                    reply_text = services["coder"].debug_code(user_prompt)
                else:
                    reply_text = services["coder"].generate_code_solution(user_prompt)

            # ----------------------------------------------------
            # 4. DATA ANALYSIS & VISUALIZATION
            # ----------------------------------------------------
            elif (active_mode == "📊 Data Analysis & Charts" or (uploaded_file and uploaded_file.name.endswith((".csv", ".xlsx")))):
                if uploaded_file and uploaded_file.name.endswith((".csv", ".xlsx")):
                    temp_data_path = settings.UPLOAD_PATH / uploaded_file.name
                    with open(temp_data_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    df, err = services["data"].load_dataset(temp_data_path)
                    if df is not None:
                        eda = services["data"].analyze_dataset(df)
                        chart_fig = services["data"].generate_chart(df)
                        reply_text = (
                            f"### 📊 Dataset Analysis: `{uploaded_file.name}`\n\n"
                            f"- **Shape**: {eda['rows']} rows × {eda['columns']} columns\n"
                            f"- **Identified Columns**: {', '.join(eda['column_names'])}\n"
                            f"- **Numeric Metrics**: {', '.join(eda['numeric_columns']) if eda['numeric_columns'] else 'None'}\n\n"
                            f"#### Key Observations\n"
                            f"- Missing values profile: {sum(eda['missing_values'].values())} null entries detected.\n"
                            f"- Interactive visual distribution generated below."
                        )
                    else:
                        reply_text = f"Error reading spreadsheet: {err}"
                else:
                    reply_text = "Please upload a CSV or Excel dataset using the attachment box in the sidebar to inspect statistics and generate charts."

            # ----------------------------------------------------
            # 5. DOCUMENT INTELLIGENCE
            # ----------------------------------------------------
            elif (active_mode == "📄 Document Intelligence" or (uploaded_file and uploaded_file.name.endswith((".pdf", ".docx", ".pptx", ".txt")))):
                if uploaded_file:
                    temp_doc_path = settings.UPLOAD_PATH / uploaded_file.name
                    with open(temp_doc_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    doc_data = services["doc"].extract_content(temp_doc_path)
                    if "text" in doc_data:
                        # Determine study material mode
                        sub_mode = "summary"
                        if "mcq" in user_prompt.lower():
                            sub_mode = "mcq"
                        elif "note" in user_prompt.lower():
                            sub_mode = "notes"
                        elif "question" in user_prompt.lower() or "q&a" in user_prompt.lower():
                            sub_mode = "qa"

                        study_content = services["doc"].generate_study_materials(doc_data["text"], mode=sub_mode)
                        reply_text = f"**Document Parsed**: `{uploaded_file.name}` ({doc_data['metadata'].get('size_kb')} KB)\n\n{study_content}"
                    else:
                        reply_text = doc_data.get("error", "Failed to parse document.")
                else:
                    reply_text = "Please upload a PDF, Word document, PowerPoint, or text file using the sidebar attachment box to summarize or generate MCQs."

            # ----------------------------------------------------
            # 6. EMOTIONAL SUPPORT & CARE
            # ----------------------------------------------------
            elif active_mode == "🌿 Emotional Support & Care" or any(w in user_prompt.lower() for w in ["stress", "anxious", "depressed", "lonely", "sad", "burnout"]):
                care_data = services["care"].process_emotional_input(user_prompt)
                reply_text = care_data["response"]
                citations = care_data.get("sources", [])

            # ----------------------------------------------------
            # 7. GENERAL ASSISTANT / MULTIMODAL INQUIRY
            # ----------------------------------------------------
            else:
                multimodal_ctx = ""
                # If image attached
                if uploaded_file and uploaded_file.name.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    temp_img_path = settings.UPLOAD_PATH / uploaded_file.name
                    with open(temp_img_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    multimodal_ctx = services["vision"].analyze_image(temp_img_path, prompt_instruction=user_prompt)

                # If audio attached
                elif uploaded_file and uploaded_file.name.endswith((".wav", ".mp3", ".m4a", ".ogg")):
                    temp_aud_path = settings.UPLOAD_PATH / uploaded_file.name
                    with open(temp_aud_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    aud_res = services["audio"].process_voice_input(temp_aud_path)
                    multimodal_ctx = f"Spoken Audio Transcript: {aud_res['transcript']}"

                reply_text = services["chat"].generate_chat_response(
                    prompt=user_prompt,
                    conversation_history=current_chat["messages"][-6:],
                    context_data=multimodal_ctx
                )

            # Generate TTS audio if toggle enabled
            if st.session_state.enable_tts:
                audio_dest = services["audio"].generate_voice_reply(reply_text)
                if audio_dest:
                    generated_audio_path = str(audio_dest)

            # Display response content
            st.markdown(reply_text)

            if generated_img_path:
                st.image(generated_img_path, caption="Generated Image", use_container_width=True)

            if chart_fig:
                st.plotly_chart(chart_fig, use_container_width=True)

            if generated_audio_path:
                st.audio(generated_audio_path)

            if citations:
                with st.expander("📘 Verified Knowledge & Web Citations", expanded=False):
                    for s in citations:
                        st.markdown(f"**{s.get('document_name', 'Source')}** — {s.get('chunk_reference', '')}\n> {s.get('excerpt', '')}")

            # Append assistant message to chat history
            current_chat["messages"].append({
                "role": "assistant",
                "content": reply_text,
                "image_path": generated_img_path,
                "audio_path": generated_audio_path,
                "chart_fig": chart_fig,
                "sources": citations
            })

    st.rerun()
