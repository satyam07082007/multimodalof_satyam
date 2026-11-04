import logging
from pathlib import Path
from typing import Optional
from PIL import Image
from pypdf import PdfReader
import google.generativeai as genai
from app.config import settings

logger = logging.getLogger("emotional_support.multimodal")

class MultimodalService:
    """
    Handles multimodal perception across Vision (images), Audio (STT),
    Document parsing (PDF/TXT), and sentiment detection.
    """

    @classmethod
    def analyze_image(cls, image_path: Path) -> str:
        """
        Interprets user-submitted image (drawing, journal note, emotion scene)
        using vision-capable model or graceful perceptual fallback.
        """
        if not image_path.exists():
            return ""

        if settings.LLM_API_KEY and settings.LLM_PROVIDER.lower() == "gemini":
            try:
                model = genai.GenerativeModel(settings.LLM_MODEL)
                pil_img = Image.open(image_path)
                prompt = (
                    "You are an empathetic emotional support observer. Look at this image provided by a user "
                    "(which may be artwork, a journal entry, an emotion photo, or surroundings). "
                    "In 2-3 gentle sentences, describe the emotional mood, colors, and visual feeling conveyed."
                )
                response = model.generate_content([prompt, pil_img])
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.error("Vision model analysis failed: %s", e)

        # Fallback offline perceptual summary
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                return f"[Visual Context: Image uploaded ({w}x{h} px). Expressive drawing/journal shared for emotional context.]"
        except Exception:
            return "[Visual Context: Image shared for emotional context.]"

    @classmethod
    def transcribe_audio(cls, audio_path: Path) -> str:
        """
        Transcribes speech from audio recording (.wav, .mp3, .ogg)
        using multimodal model or structured voice tag.
        """
        if not audio_path.exists():
            return ""

        if settings.LLM_API_KEY and settings.LLM_PROVIDER.lower() == "gemini":
            try:
                uploaded_file = genai.upload_file(path=str(audio_path))
                model = genai.GenerativeModel(settings.LLM_MODEL)
                response = model.generate_content([
                    "Please transcribe the spoken emotional message from this audio clip accurately.",
                    uploaded_file
                ])
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.error("Audio STT transcription failed: %s", e)

        return "[Voice note shared: Spoken emotional reflection received.]"

    @classmethod
    def extract_document_context(cls, doc_path: Path, max_chars: int = 1500) -> str:
        """
        Extracts readable text from user-uploaded PDF or TXT attachment for conversation context.
        """
        if not doc_path.exists():
            return ""

        text = ""
        suffix = doc_path.suffix.lower()
        try:
            if suffix == ".txt":
                with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            elif suffix == ".pdf":
                reader = PdfReader(str(doc_path))
                pages = [p.extract_text() or "" for p in reader.pages[:3]]
                text = "\n".join(pages)
        except Exception as e:
            logger.error("Document extraction error: %s", e)

        if text:
            cleaned = text.strip()
            if len(cleaned) > max_chars:
                return cleaned[:max_chars] + "\n...[truncated document content]"
            return cleaned
        return ""

    @classmethod
    def detect_mood(cls, text: str) -> str:
        """
        Detects predominant emotional state for empathetic tone tuning.
        """
        lower = text.lower()
        if any(w in lower for w in ["anxious", "panic", "scared", "fear", "nervous", "dread", "shaking", "racing heart"]):
            return "anxious"
        if any(w in lower for w in ["sad", "depressed", "crying", "tears", "grief", "heartbroken", "down", "hurting"]):
            return "sad"
        if any(w in lower for w in ["exhausted", "burnout", "tired", "drained", "overworked", "can't focus", "heavy"]):
            return "exhausted"
        if any(w in lower for w in ["lonely", "alone", "isolated", "empty", "abandoned", "no friends"]):
            return "lonely"
        if any(w in lower for w in ["angry", "furious", "mad", "frustrated", "hate", "irritated", "unfair"]):
            return "frustrated"
        if any(w in lower for w in ["better", "hope", "thank you", "calmer", "relieved", "good", "grateful"]):
            return "hopeful"
        return "neutral"

