import re
import logging
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger("haven.router")


class RouteIntent(str, Enum):
    IMAGE_SEARCH = "image_search"
    IMAGE_GENERATION = "image_gen"
    WEB_SEARCH = "web_search"
    CODE_ASSISTANT = "code"
    DOCUMENT_RAG = "document"
    DATA_ANALYSIS = "data_analysis"
    AUDIO_PROCESSING = "audio_file"
    EMOTIONAL_CARE = "care"
    GENERAL_CHAT = "general_chat"


class SmartRouter:
    """
    Central Intelligent AI Router for HAVEN AI.
    Accurately classifies user requests across Multimodal capabilities:
    - Image Search & Retrieval (Celebrities, places, products, things)
    - AI Image Generation (Text-to-Image neural diffusion)
    - Real-Time Web Research (Search engine & live news)
    - Code Studio & Debugger (Multi-language programming)
    - Document Intelligence & RAG (PDF, DOCX, TXT Q&A)
    - Data Analysis (CSV/XLSX statistics & interactive charts)
    - Audio Speech Recognition (Transcriptions & voice input)
    - Emotional Support (Empathetic listening & grounding)
    - General Conversational AI (Questions, answers, reasoning)
    """

    # 1. AI Image Generation patterns (Creating new artificial images)
    IMAGE_GEN_PATTERNS = [
        r'\b(generate|create|make|draw|paint|sketch|render)\b.*\b(image|picture|photo|wallpaper|logo|avatar|drawing|art|portrait|illustration)\b',
        r'\b(image|photo|picture|tasveer|chhavi|drawing)\b.*\b(generate|banao|bana do|banaye|banaiye)\b',
        r'\b(draw|paint|sketch)\s+(a|an|the)?\b',
        r'\b(create|generate|make)\s+(a|an)?\s*(futuristic|realistic|cybernetic|surreal|cinematic|fantasy|3d|cartoon|anime|sci-fi)?\s*(city|tiger|landscape|car|animal|lion|forest|robot|portrait|wallpaper|scene|artwork)\b'
    ]

    # 2. Image Search & Retrieval patterns (Fetching existing photos of people, places, landmarks, products)
    IMAGE_SEARCH_PATTERNS = [
        r'\b(give me|show me|find|search|fetch|get|look for)\b.*\b(image|images|picture|pictures|photo|photos|pic|pics)\b',
        r'\b(image|images|picture|pictures|photo|photos|pic|pics)\s+(of|for)\b',
        r'\b(ki|ka)\s+(photo|image|picture|tasveer)\s+(dikhao|dhoondo|lao|bhejo)\b',
        r'\b(photo|image|picture|tasveer)\s+(dikhao|dhoondo)\b'
    ]

    # 3. Web Search patterns
    WEB_SEARCH_PATTERNS = [
        r'\b(search\b.*\b(web|news|internet|online|google)|latest news|current weather|live score|news today|search latest)\b',
        r'\b(search the web|search web|google this|internet search|web research)\b'
    ]

    # 4. Code Studio patterns
    CODE_PATTERNS = [
        r'\b(write\s+(a\s+)?(python|code|program|script|function|app|api)|debug|traceback|fix error|algorithm|binary search|sql query|quick sort|stack trace)\b',
        r'\b(write\s+(a\s+)?(javascript|typescript|c\+\+|java|html|css|rust|golang|bash)\s+(code|script|program)?)\b',
        r'\b(code likho|code banao|program banao)\b'
    ]

    # 5. Emotional Support patterns
    CARE_PATTERNS = [
        r'\b(feel sad|depressed|anxious|panic|lonely|heartbroken|stressed|bohot tension|pareshan|rough day|overwhelmed|crying|suicidal|hopeless)\b'
    ]

    @classmethod
    def route(
        cls,
        user_prompt: str,
        active_mode: str = "auto",
        has_upload: bool = False,
        upload_name: str = ""
    ) -> RouteIntent:
        """
        Routes the user prompt to the most appropriate engine.
        Respects manual mode override when specified; otherwise applies intelligent auto-routing.
        """
        # Manual Mode Override
        if active_mode == "🎨 Image Generation Studio":
            return RouteIntent.IMAGE_GENERATION
        if active_mode == "🔍 Web Image Search":
            return RouteIntent.IMAGE_SEARCH
        if active_mode == "🌐 Web Research Engine":
            return RouteIntent.WEB_SEARCH
        if active_mode == "💻 Code Studio & Debugger":
            return RouteIntent.CODE_ASSISTANT
        if active_mode == "📊 Data Analysis & Charts":
            return RouteIntent.DATA_ANALYSIS
        if active_mode == "📄 Document Intelligence":
            return RouteIntent.DOCUMENT_RAG
        if active_mode == "🌿 Emotional Support & Care":
            return RouteIntent.EMOTIONAL_CARE

        # File Attachment Priority in Auto-Detect
        if has_upload and upload_name:
            ext = upload_name.lower()
            if ext.endswith((".csv", ".xlsx")):
                return RouteIntent.DATA_ANALYSIS
            if ext.endswith((".pdf", ".docx", ".pptx", ".txt")):
                return RouteIntent.DOCUMENT_RAG
            if ext.endswith((".png", ".jpg", ".jpeg", ".webp")):
                return RouteIntent.DOCUMENT_RAG  # Vision inspection handled here
            if ext.endswith((".wav", ".mp3", ".m4a", ".ogg")):
                return RouteIntent.AUDIO_PROCESSING

        text = user_prompt.lower().strip()

        # 1. Check AI Image Generation (Creating new visuals)
        for pattern in cls.IMAGE_GEN_PATTERNS:
            if re.search(pattern, text):
                return RouteIntent.IMAGE_GENERATION

        # 2. Check Image Search (Finding existing real-world pictures)
        for pattern in cls.IMAGE_SEARCH_PATTERNS:
            if re.search(pattern, text):
                return RouteIntent.IMAGE_SEARCH

        # 3. Check Live Web Search
        for pattern in cls.WEB_SEARCH_PATTERNS:
            if re.search(pattern, text):
                return RouteIntent.WEB_SEARCH

        # 4. Check Code Assistant
        for pattern in cls.CODE_PATTERNS:
            if re.search(pattern, text):
                return RouteIntent.CODE_ASSISTANT

        # 5. Check Emotional Support
        for pattern in cls.CARE_PATTERNS:
            if re.search(pattern, text):
                return RouteIntent.EMOTIONAL_CARE

        # 6. Default to General AI Assistant
        return RouteIntent.GENERAL_CHAT

    @classmethod
    def extract_clean_query(cls, prompt: str, intent: RouteIntent) -> str:
        """
        Extracts clean target subject from prompts for specialized engines.
        """
        p = prompt.strip()

        if intent == RouteIntent.IMAGE_SEARCH:
            # Strip common retrieval phrases
            cleaned = re.sub(r'^(give me|show me|find|search for|search|fetch|get|look for)\s+', '', p, flags=re.IGNORECASE)
            cleaned = re.sub(r'\b(images of|image of|pictures of|picture of|photos of|photo of|pics of|pic of|images for|image for|photos for|photo for|pictures for)\b', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s+(ki|ka)\s+(photo|image|picture|tasveer)(\s+(dikhao|dhoondo|lao|bhejo))?$', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s+(photo|image|picture|tasveer)\s+(dikhao|dhoondo)$', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"'s\s*(image|images|photo|photos|picture|pictures)?", '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\b(image|images|picture|pictures|photo|photos)\b', '', cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip(" ?.,!'\"")
            return cleaned or p

        if intent == RouteIntent.IMAGE_GENERATION:
            cleaned = p
            for prefix in [
                "generate an image of", "generate image of", "create an image of", "create image of",
                "make an image of", "make image of", "draw a picture of", "draw an image of",
                "draw a ", "draw ", "paint a ", "paint ", "photo of a ", "photo of ",
                "picture of a ", "picture of ", "image of a ", "image of ",
                "image banao ", "photo banao ", "ki image banao", "ki photo banao",
                "generate a photo of", "generate a picture of", "generate a ", "generate ",
                "create a ", "create "
            ]:
                cleaned = re.sub(re.escape(prefix), "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip(" ?.,!'\"")
            return cleaned or p

        if intent == RouteIntent.WEB_SEARCH:
            cleaned = re.sub(r'^(search the web for|search web for|search the web|search web|google this|google|search for|find latest|latest news on)\s+', '', p, flags=re.IGNORECASE)
            cleaned = cleaned.strip(" ?.,!'\"")
            return cleaned or p

        return p

