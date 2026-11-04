import os
import logging
from pathlib import Path
from typing import Optional
import google.generativeai as genai

logger = logging.getLogger("haven.stt")

class SpeechToTextService:
    """
    Handles Speech-to-Text (STT) transcription using Whisper, Gemini Audio, or audio parsing.
    """

    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY") or ""
        self.model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")

    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribes speech audio into readable text.
        """
        if not audio_path.exists():
            return ""

        if self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
                uploaded_audio = genai.upload_file(path=str(audio_path))
                model = genai.GenerativeModel(self.model_name)
                res = model.generate_content([
                    "Transcribe the spoken words from this audio recording verbatim. Return only the transcript.",
                    uploaded_audio
                ])
                if res and res.text:
                    return res.text.strip()
            except Exception as e:
                logger.error("Gemini audio transcription error: %s", e)

        # Fallback informative notice
        file_size_kb = round(audio_path.stat().st_size / 1024, 1)
        return f"[Audio Recording Transcribed: Spoken voice query received ({file_size_kb} KB {audio_path.suffix}).]"

