import uuid
import re
import logging
from pathlib import Path
from typing import Optional
from gtts import gTTS

logger = logging.getLogger("haven.tts")

class TextToSpeechService:
    """
    Synthesizes natural speech audio responses from text.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        try:
            from app.config import settings
            default_dir = getattr(
                settings,
                "AUDIO_PATH",
                getattr(settings, "AUDIO_DIR", Path("media") / "audio_responses")
            )
        except Exception as e:
            logger.warning("Could not load settings in TextToSpeechService: %s", e)
            default_dir = Path("media") / "audio_responses"

        self.output_dir = Path(output_dir) if output_dir else default_dir
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Ensures destination audio directory exists and is writable."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("Failed to create audio output directory %s: %s", self.output_dir, e)

    def synthesize(self, text: str, voice_lang: str = "en", max_chars: int = 400) -> Optional[Path]:
        """
        Synthesizes text into natural audio (.mp3).
        Returns Path to the generated audio file.
        """
        try:
            if not text or not text.strip():
                return None

            self._ensure_output_dir()

            # Clean markdown formatting, asterisks, brackets, and emojis
            clean = re.sub(r"[\*#_`~>\[\]]", "", text)
            clean = re.sub(r"https?://\S+", "", clean)
            clean = re.sub(r"\s+", " ", clean).strip()

            if len(clean) > max_chars:
                clean = clean[:max_chars] + "... That concludes this summary."

            filename = f"haven_voice_{uuid.uuid4().hex[:10]}.mp3"
            dest_file = self.output_dir / filename

            tts = gTTS(text=clean, lang=voice_lang, slow=False)
            tts.save(str(dest_file))
            logger.info("Successfully synthesized audio: %s", filename)
            return dest_file
        except Exception as e:
            logger.warning("TTS synthesis notice (non-fatal): %s", e)
            return None
