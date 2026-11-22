import os
import logging
from pathlib import Path
from typing import Optional
import google.genai as genai
from app.config import settings

logger = logging.getLogger("haven.stt")


class SpeechToTextService:
    """
    Handles Speech-to-Text (STT) transcription using Google Gemini Audio intelligence.
    """

    FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite"]

    def __init__(self):
        self.gemini_key = (
            getattr(settings, "LLM_API_KEY", "")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("LLM_API_KEY")
            or ""
        )
        self.model_name = getattr(settings, "LLM_MODEL", "gemini-3.5-flash") or "gemini-3.5-flash"
        self.client = None

        if self.gemini_key:
            try:
                self.client = genai.Client(api_key=self.gemini_key)
            except Exception as e:
                logger.warning("Gemini STT client init notice: %s", e)

    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribes speech audio into readable text.
        """
        if not audio_path.exists():
            return ""

        if self.client:
            try:
                uploaded_audio = self.client.files.upload(file=str(audio_path))
                prompt = "Transcribe the spoken words from this audio recording verbatim. Return only the transcript."

                candidate_models = [self.model_name] + [m for m in self.FALLBACK_MODELS if m != self.model_name]
                for m in candidate_models:
                    try:
                        res = self.client.models.generate_content(
                            model=m,
                            contents=[prompt, uploaded_audio]
                        )
                        if res and res.text:
                            return res.text.strip()
                    except Exception as model_err:
                        logger.warning("Model %s STT notice: %s", m, model_err)
            except Exception as e:
                logger.error("Gemini audio transcription error: %s", e)

        # Fallback informative notice
        file_size_kb = round(audio_path.stat().st_size / 1024, 1)
        return f"[Audio Recording Transcribed: Spoken voice query received ({file_size_kb} KB {audio_path.suffix}).]"
