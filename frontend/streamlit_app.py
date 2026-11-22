import sys
import os
import re
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
from modules.image_search import ImageSearchModule
from modules.audio_processing import AudioProcessingModule
from modules.document_processing import DocumentProcessor
from modules.web_search import WebSearchModule
from modules.coding_assistant import CodingAssistantModule
from modules.data_analysis import DataAnalysisModule
from modules.rag_engine import DynamicRAGEngine
from modules.emotional_support import EmotionalSupportModule
from modules.router import SmartRouter, RouteIntent
from utils.security import sanitize_pii, check_crisis_intent

# Streamlit Page Setup
st.set_page_config(
    page_title="HAVEN AI | One AI. Every Possibility.",
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
        "image_search": ImageSearchModule(),
        "audio": AudioProcessingModule(),
        "doc": DocumentProcessor(),
        "search": WebSearchModule(),
        "coder": CodingAssistantModule(),
        "data": DataAnalysisModule(),
        "rag": DynamicRAGEngine(),
        "care": EmotionalSupportModule()
    }

services = get_services()


def enhance_image_prompt(raw_prompt: str) -> str:
    """Cleans and expands user image prompts into descriptive diffusion prompts."""
    clean = raw_prompt
    for prefix in [
        "generate an image of", "generate image of", "create an image of", "create image of",
        "make an image of", "make image of", "draw a picture of", "draw an image of",
        "draw a ", "draw ", "paint a ", "paint ", "photo of a ", "photo of ",
        "picture of a ", "picture of ", "image of a ", "image of ",
        "image banao ", "photo banao ", "ki image banao", "ki photo banao",
        "generate a photo of", "generate a picture of", "generate a ", "generate ",
        "create a ", "create "
    ]:
        clean = re.sub(re.escape(prefix), "", clean, flags=re.IGNORECASE)
    clean = clean.strip()
    if not clean:
        clean = "stunning futuristic neon art landscape"

    # Translate / enhance prompt using Gemini if available
    try:
        import google.genai as genai
        api_key = getattr(settings, "LLM_API_KEY", "") or os.getenv("LLM_API_KEY", "")
        if api_key:
            client = genai.Client(api_key=api_key)
            res = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=(
                    "Translate into English (if non-English) and expand this image prompt into a vivid, "
                    "high-quality visual prompt for diffusion image generation. "
                    "Return ONLY the enhanced prompt in English under 40 words, nothing else.\n"
                    f"Prompt: {clean}"
                )
            )
            if res and res.text and len(res.text.strip()) > 5:
                return res.text.strip()
    except Exception:
        pass
    return clean


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
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.55) 100%);
        border: 1px solid rgba(148, 163, 184, 0.18);
        backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 30px;
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

    /* Message Styling */
    [data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.12);
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

    /* Image Search Card Styling */
    .img-search-box {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 10px;
        background: rgba(15, 23, 42, 0.6);
        padding: 8px;
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
    st.session_state.app_mode = "🤖 Auto-Detect (All-in-One)"

if "enable_tts" not in st.session_state:
    st.session_state.enable_tts = False

current_chat = st.session_state.conversations[st.session_state.current_chat_id]

# Sidebar: Branding, Navigation, Mode Selection, Settings
with st.sidebar:
    st.markdown("### 🌿 HAVEN AI")
    st.caption("**One AI. Every Possibility.** — ChatGPT & Gemini Class AI")
    st.markdown(f'<div class="mode-badge">{st.session_state.app_mode}</div>', unsafe_allow_html=True)

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = f"chat_{uuid.uuid4().hex[:8]}"
        st.session_state.conversations[new_id] = {"title": "New Conversation", "messages": []}
        st.session_state.current_chat_id = new_id
        st.rerun()

    # Chat History List
    st.markdown("#### 💬 Recent Conversations")
    for chat_id, chat_data in list(st.session_state.conversations.items())[-6:]:
        title = chat_data.get("title", "Conversation")
        is_active = (chat_id == st.session_state.current_chat_id)
        btn_label = f"{'🟢 ' if is_active else '💬 '}{title[:22]}"
        if st.button(btn_label, key=f"hist_{chat_id}", use_container_width=True):
            st.session_state.current_chat_id = chat_id
            st.rerun()

    st.divider()

    # Mode Selector
    st.markdown("#### 🧭 Intelligence Modes")
    mode_options = [
        "🤖 Auto-Detect (All-in-One)",
        "🔍 Web Image Search",
        "🎨 Image Generation Studio",
        "💻 Code Studio & Debugger",
        "🌐 Web Research Engine",
        "📊 Data Analysis & Charts",
        "📄 Document Intelligence",
        "🌿 Emotional Support & Care"
    ]
    cur_idx = mode_options.index(st.session_state.app_mode) if st.session_state.app_mode in mode_options else 0
    selected_mode = st.radio("Select Mode", mode_options, index=cur_idx, label_visibility="collapsed")
    if selected_mode != st.session_state.app_mode:
        st.session_state.app_mode = selected_mode
        st.rerun()

    st.divider()

    # Multimodal Attachments Drawer
    st.markdown("#### 📎 Attachments & Media")
    uploaded_file = st.file_uploader(
        "Upload image, document, spreadsheet, or audio",
        type=["png", "jpg", "jpeg", "webp", "pdf", "docx", "txt", "csv", "xlsx", "pptx", "wav", "mp3"],
        key="main_uploader"
    )

    if uploaded_file:
        st.success(f"Attached: `{uploaded_file.name}` ({round(uploaded_file.size / 1024, 1)} KB)")

    st.divider()

    # Settings & Audio Controls
    with st.expander("⚙️ Preferences & Controls", expanded=False):
        st.session_state.enable_tts = st.toggle("🔊 Read Responses Aloud (TTS)", value=st.session_state.enable_tts)
        st.markdown(f"**Session**: `{st.session_state.current_chat_id}`")

        # Export Chat History
        if current_chat["messages"]:
            chat_export = f"# Haven AI Conversation Export\n**Session**: {st.session_state.current_chat_id}\n\n"
            for m in current_chat["messages"]:
                role_label = "User" if m["role"] == "user" else "Haven AI"
                chat_export += f"### {role_label}:\n{m['content']}\n\n"
            st.download_button(
                label="📥 Export Chat (Markdown)",
                data=chat_export,
                file_name=f"haven_chat_{st.session_state.current_chat_id}.md",
                mime="text/markdown",
                use_container_width=True
            )

        if st.button("🗑️ Clear Current Chat", use_container_width=True):
            current_chat["messages"] = []
            st.rerun()

    # Crisis Helpline
    with st.expander("🆘 Crisis & Safety Helplines", expanded=False):
        st.markdown("""
        <div class="helpline-box">
            <strong>Need immediate help?</strong><br><br>
            • <strong>988 Lifeline</strong>: Call/Text <code>988</code> (24/7, US & CA)<br>
            • <strong>Crisis Text Line</strong>: Text <code>HOME</code> to <code>741741</code><br>
            • <strong>International</strong>: <a href="https://www.befrienders.org" target="_blank" style="color:#fca5a5;">befrienders.org</a>
        </div>
        """, unsafe_allow_html=True)


# Main Hero Header
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🌿 HAVEN AI</div>
    <div class="hero-tagline">One AI. Every Possibility. — Smart Search, Image Generation, Code, Web Research, Vision & Care.</div>
</div>
""", unsafe_allow_html=True)

# Starter Suggestion Cards for Empty Chat
pending_prompt = None
if len(current_chat["messages"]) == 0:
    st.markdown("##### 💡 What would you like to do today?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 **Search Existing Images**\n\n*Find real photos of the Taj Mahal*", use_container_width=True):
            pending_prompt = "Find images of the Taj Mahal"
        if st.button("🎨 **Generate an Image**\n\n*A futuristic cybernetic tiger with glowing neon stripes*", use_container_width=True):
            pending_prompt = "Generate an image of a cybernetic tiger with neon stripes"
        if st.button("💻 **Write Python Code**\n\n*Build a FastAPI REST API with rate limiting*", use_container_width=True):
            pending_prompt = "Write a complete Python program for a FastAPI REST API with rate limiting and error handling"
    with col2:
        if st.button("👤 **Search People / Celebrities**\n\n*Give me images of Akhilesh Yadav*", use_container_width=True):
            pending_prompt = "Give me images of Akhilesh Yadav"
        if st.button("🌐 **Live Web Search**\n\n*Latest breakthroughs in artificial intelligence in 2026*", use_container_width=True):
            pending_prompt = "Search the web for the latest artificial intelligence breakthroughs in 2026"
        if st.button("🌿 **Emotional Care**\n\n*I had a stressful day, guide me through a calming exercise*", use_container_width=True):
            pending_prompt = "I had a stressful day today, can you guide me through a calming grounding exercise?"

# Render Chat Dialogue
for msg in current_chat["messages"]:
    role = msg["role"]
    avatar = "🌱" if role == "user" else "🌿"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

        # Render Image Search Gallery if present
        if msg.get("image_search_results"):
            st.markdown("##### 🖼️ Retrieved Web Images:")
            results_to_show = msg["image_search_results"][:6]
            cols = st.columns(min(len(results_to_show), 3))
            for idx, item in enumerate(results_to_show):
                with cols[idx % 3]:
                    img_src = item.get("thumbnail_url") or item.get("image_url")
                    if img_src:
                        st.image(img_src, caption=item.get("title", "")[:35], use_container_width=True)
                        st.markdown(f"[🔗 Source Website]({item.get('source_url', '#')}) • [Full Image]({item.get('image_url', '#')})")

        # Render generated or attached image
        if msg.get("image_path") and Path(msg["image_path"]).exists():
            st.image(str(msg["image_path"]), caption="Generated / Attached Image", use_container_width=True)
            with open(msg["image_path"], "rb") as img_f:
                st.download_button(
                    label="⬇️ Download Generated Image",
                    data=img_f.read(),
                    file_name=Path(msg["image_path"]).name,
                    mime="image/png",
                    key=f"dl_{msg.get('image_path')}_{uuid.uuid4().hex[:6]}"
                )

        # Render Plotly Chart
        if msg.get("chart_fig"):
            st.plotly_chart(msg["chart_fig"], use_container_width=True)

        # Render Audio Player
        if msg.get("audio_path") and Path(msg["audio_path"]).exists():
            st.audio(str(msg["audio_path"]))

        # Render Citations
        if msg.get("sources"):
            with st.expander("📘 Verified Knowledge & Web Citations", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f"**{s.get('document_name', 'Source')}** — {s.get('chunk_reference', '')}\n> {s.get('excerpt', '')}")

# Voice Input Microphone Bar
voice_prompt = None
with st.expander("🎙️ Voice Input (Speak into your Microphone)", expanded=False):
    recorded_audio = st.audio_input("Record your voice query:")
    if recorded_audio:
        temp_voice_file = settings.UPLOAD_PATH / f"voice_{uuid.uuid4().hex[:8]}.wav"
        with open(temp_voice_file, "wb") as vf:
            vf.write(recorded_audio.getvalue())
        voice_res = services["audio"].process_voice_input(temp_voice_file)
        if voice_res.get("transcript"):
            voice_prompt = voice_res["transcript"]
            st.info(f"🎤 Voice Heard: \"{voice_prompt}\"")

# Chat Input Composer
chat_input = st.chat_input("Ask anything, search images, generate art, write code, search web, or attach files...")

# Resolve final prompt from user input, voice, card clicks, or file upload
active_prompt = pending_prompt or voice_prompt or chat_input
if not active_prompt and uploaded_file and st.button("🚀 Process Attached File", key="process_btn"):
    active_prompt = f"Analyze attached file: {uploaded_file.name}"

# Process Interaction
if active_prompt:
    active_mode = st.session_state.app_mode

    # Auto-title conversation on first message
    if len(current_chat["messages"]) == 0:
        current_chat["title"] = active_prompt[:30] + ("..." if len(active_prompt) > 30 else "")

    # Append user message
    current_chat["messages"].append({"role": "user", "content": active_prompt})

    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("Haven AI is routing and processing your request..."):
            reply_text = ""
            generated_img_path = None
            generated_audio_path = None
            chart_fig = None
            citations = []
            image_search_results = []

            # ----------------------------------------------------
            # Central Smart Router Intent Classification
            # ----------------------------------------------------
            intent = SmartRouter.route(
                user_prompt=active_prompt,
                active_mode=active_mode,
                has_upload=bool(uploaded_file),
                upload_name=uploaded_file.name if uploaded_file else ""
            )

            # ----------------------------------------------------
            # 1. IMAGE SEARCH & RETRIEVAL (Real People, Places, Landmarks)
            # ----------------------------------------------------
            if intent == RouteIntent.IMAGE_SEARCH:
                search_query = SmartRouter.extract_clean_query(active_prompt, intent)
                st.caption(f"🔍 Searching live web for existing images of: *\"{search_query}\"*...")
                image_search_results = services["image_search"].search(search_query, max_results=6)
                reply_text = services["image_search"].format_search_results(image_search_results, search_query)

            # ----------------------------------------------------
            # 2. AI IMAGE GENERATION (Neural Diffusion / FLUX)
            # ----------------------------------------------------
            elif intent == RouteIntent.IMAGE_GENERATION:
                st.caption("🎨 Synthesizing original artwork with Neural Diffusion...")
                gen_prompt = SmartRouter.extract_clean_query(active_prompt, intent)
                enhanced_prompt = enhance_image_prompt(gen_prompt)
                success, img_path, msg = services["image_gen"].generate_image(enhanced_prompt)
                if success and img_path:
                    generated_img_path = str(img_path)
                    reply_text = (
                        f"### 🎨 Image Generated Successfully\n\n"
                        f"- **Your Request**: *\"{active_prompt}\"*\n"
                        f"- **Diffusion Prompt**: *\"{enhanced_prompt}\"*\n\n"
                        f"{msg}"
                    )
                else:
                    reply_text = f"⚠️ Image generation notice: {msg}"

            # ----------------------------------------------------
            # 3. WEB RESEARCH & LIVE NEWS
            # ----------------------------------------------------
            elif intent == RouteIntent.WEB_SEARCH:
                st.caption("🌐 Conducting live web research across public internet index...")
                search_query = SmartRouter.extract_clean_query(active_prompt, intent)
                search_res = services["search"].search(search_query, max_results=5)
                reply_text = services["search"].synthesize_research_report(search_query, search_res)
                citations = [
                    {"document_name": r.get("title"), "chunk_reference": r.get("href"), "excerpt": r.get("body")}
                    for r in search_res
                ]

            # ----------------------------------------------------
            # 4. CODE STUDIO & DEBUGGER
            # ----------------------------------------------------
            elif intent == RouteIntent.CODE_ASSISTANT:
                st.caption("💻 Generating software engineering solution...")
                if any(w in active_prompt.lower() for w in ["debug", "error", "traceback", "fix"]):
                    reply_text = services["coder"].debug_code(active_prompt)
                else:
                    reply_text = services["coder"].generate_code_solution(active_prompt)

            # ----------------------------------------------------
            # 5. DATA ANALYSIS & CHARTS (Spreadsheets)
            # ----------------------------------------------------
            elif intent == RouteIntent.DATA_ANALYSIS:
                st.caption("📊 Analyzing spreadsheet data and calculating statistics...")
                if uploaded_file and uploaded_file.name.lower().endswith((".csv", ".xlsx")):
                    temp_data_path = settings.UPLOAD_PATH / uploaded_file.name
                    with open(temp_data_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    df, err = services["data"].load_dataset(temp_data_path)
                    if df is not None:
                        eda = services["data"].analyze_dataset(df)
                        chart_fig = services["data"].generate_chart(df)
                        reply_text = (
                            f"### 📊 Dataset Analysis: `{uploaded_file.name}`\n\n"
                            f"- **Dimensions**: {eda['rows']} rows × {eda['columns']} columns\n"
                            f"- **Detected Columns**: {', '.join(eda['column_names'])}\n"
                            f"- **Numeric Metrics**: {', '.join(eda['numeric_columns']) if eda['numeric_columns'] else 'None'}\n\n"
                            f"#### Key Observations\n"
                            f"- Missing Values: {sum(eda['missing_values'].values())} null entries detected.\n"
                            f"- Interactive visual chart rendered below."
                        )
                    else:
                        reply_text = f"Error reading spreadsheet: {err}"
                else:
                    reply_text = "Please upload a CSV or Excel dataset in the left sidebar attachment drawer to inspect statistics and generate charts."

            # ----------------------------------------------------
            # 6. DOCUMENT INTELLIGENCE & RAG (PDF / DOCX / PPTX / TXT)
            # ----------------------------------------------------
            elif intent == RouteIntent.DOCUMENT_RAG:
                if uploaded_file and uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    # Vision analysis for attached image
                    st.caption("🖼️ Analyzing image with Google Gemini Vision...")
                    temp_img_path = settings.UPLOAD_PATH / uploaded_file.name
                    with open(temp_img_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    reply_text = services["vision"].analyze_image(temp_img_path, prompt_instruction=active_prompt)
                elif uploaded_file:
                    st.caption("📄 Extracting document text and generating intelligent synthesis...")
                    temp_doc_path = settings.UPLOAD_PATH / uploaded_file.name
                    with open(temp_doc_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    doc_data = services["doc"].extract_content(temp_doc_path)
                    if "text" in doc_data:
                        sub_mode = "summary"
                        lower_p = active_prompt.lower()
                        if "mcq" in lower_p or "question" in lower_p:
                            sub_mode = "mcq"
                        elif "note" in lower_p:
                            sub_mode = "notes"
                        elif "q&a" in lower_p:
                            sub_mode = "qa"

                        study_content = services["doc"].generate_study_materials(
                            document_text=doc_data["text"],
                            mode=sub_mode,
                            user_query=active_prompt
                        )
                        reply_text = f"**Document**: `{uploaded_file.name}` ({doc_data['metadata'].get('size_kb')} KB)\n\n{study_content}"
                    else:
                        reply_text = doc_data.get("error", "Failed to parse document content.")
                else:
                    reply_text = "Please upload a PDF, DOCX, PPTX, or TXT file using the sidebar attachment drawer to summarize or analyze it."

            # ----------------------------------------------------
            # 7. AUDIO FILE PROCESSING
            # ----------------------------------------------------
            elif intent == RouteIntent.AUDIO_PROCESSING:
                st.caption("🎧 Transcribing audio recording...")
                temp_aud_path = settings.UPLOAD_PATH / uploaded_file.name
                with open(temp_aud_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                aud_res = services["audio"].process_voice_input(temp_aud_path)
                transcript = aud_res.get("transcript", "")
                reply_text = services["chat"].generate_chat_response(
                    prompt=f"The user uploaded an audio recording with transcript: '{transcript}'. Please respond thoughtfully to their query.",
                    conversation_history=current_chat["messages"][-6:]
                )

            # ----------------------------------------------------
            # 8. EMOTIONAL SUPPORT & CARE
            # ----------------------------------------------------
            elif intent == RouteIntent.EMOTIONAL_CARE:
                st.caption("🌿 Providing empathetic active listening and grounding support...")
                care_data = services["care"].process_emotional_input(active_prompt)
                reply_text = care_data["response"]
                citations = care_data.get("sources", [])

            # ----------------------------------------------------
            # 9. GENERAL CONVERSATIONAL AI (Gemini 3.5 Flash)
            # ----------------------------------------------------
            else:
                reply_text = services["chat"].generate_chat_response(
                    prompt=active_prompt,
                    conversation_history=current_chat["messages"][-6:]
                )

            # Read Aloud (TTS) if enabled
            if st.session_state.enable_tts and reply_text:
                audio_dest = services["audio"].generate_voice_reply(reply_text[:600])
                if audio_dest:
                    generated_audio_path = str(audio_dest)

            # Display Response Content
            st.markdown(reply_text)

            # Render Image Search Gallery if present
            if image_search_results:
                st.markdown("##### 🖼️ Retrieved Web Images:")
                results_to_show = image_search_results[:6]
                cols = st.columns(min(len(results_to_show), 3))
                for idx, item in enumerate(results_to_show):
                    with cols[idx % 3]:
                        img_src = item.get("thumbnail_url") or item.get("image_url")
                        if img_src:
                            st.image(img_src, caption=item.get("title", "")[:35], use_container_width=True)
                            st.markdown(f"[🔗 Source Website]({item.get('source_url', '#')}) • [Full Image]({item.get('image_url', '#')})")

            # Render Generated Image
            if generated_img_path:
                st.image(generated_img_path, caption="Generated Image", use_container_width=True)
                with open(generated_img_path, "rb") as img_f:
                    st.download_button(
                        label="⬇️ Download Image",
                        data=img_f.read(),
                        file_name=Path(generated_img_path).name,
                        mime="image/png",
                        key=f"dl_instant_{uuid.uuid4().hex[:6]}"
                    )

            # Render Plotly Chart
            if chart_fig:
                st.plotly_chart(chart_fig, use_container_width=True)

            # Render Audio Player
            if generated_audio_path:
                st.audio(generated_audio_path)

            # Render Citations
            if citations:
                with st.expander("📘 Verified Knowledge & Web Citations", expanded=False):
                    for s in citations:
                        st.markdown(f"**{s.get('document_name', 'Source')}** — {s.get('chunk_reference', '')}\n> {s.get('excerpt', '')}")

            # Append assistant message to chat history
            current_chat["messages"].append({
                "role": "assistant",
                "content": reply_text,
                "image_path": generated_img_path,
                "image_search_results": image_search_results,
                "audio_path": generated_audio_path,
                "chart_fig": chart_fig,
                "sources": citations
            })

    st.rerun()
