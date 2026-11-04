import uuid
import logging
from pathlib import Path
from typing import Optional
from gtts import gTTS
from app.config import settings

logger = logging.getLogger("emotional_support.speech")

class SpeechService:
    """
    Handles Text-to-Speech (TTS) audio synthesis and audio utilities.
    """

    @classmethod
    def synthesize_speech(cls, text: str) -> Optional[str]:
        """
        Synthesizes speech audio (.mp3) from assistant response using gTTS.
        Returns the filename saved in settings.AUDIO_PATH.
        """
        try:
            if not text or not text.strip():
                return None

            # Clean markdown formatting so TTS sounds natural and clear
            clean = text.replace("*", "").replace("#", "").replace("-", " ").replace("`", "")
            # Truncate if exceptionally long so audio synthesis is fast and comfortable
            if len(clean) > 350:
                clean = clean[:350] + "... Take a slow, gentle breath. I am right here with you."

            filename = f"tts_response_{uuid.uuid4().hex[:10]}.mp3"
            dest_path = settings.AUDIO_PATH / filename

            tts = gTTS(text=clean, lang="en", tld="com", slow=False)
            tts.save(str(dest_path))
            logger.info("Synthesized speech audio file: %s", filename)
            return filename
        except Exception as e:
            logger.warning("TTS audio generation failed (non-critical): %s", e)
            return None

