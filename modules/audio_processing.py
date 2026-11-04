import logging
from pathlib import Path
from typing import Optional, Dict, Any
from modules.speech_to_text import SpeechToTextService
from modules.text_to_speech import TextToSpeechService

logger = logging.getLogger("haven.audio_processing")

class AudioProcessingModule:
    """
    Unified Audio & Voice processing module.
    Manages speech-to-text, audio summaries, and voice synthesis.
    """

    def __init__(self):
        self.stt = SpeechToTextService()
        self.tts = TextToSpeechService()

    def process_voice_input(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribes incoming audio recording and prepares it for the chat engine."""
        transcript = self.stt.transcribe(audio_path)
        return {
            "transcript": transcript,
            "filename": audio_path.name,
            "size_bytes": audio_path.stat().st_size if audio_path.exists() else 0
        }

    def generate_voice_reply(self, text: str) -> Optional[Path]:
        """Converts assistant text reply into playable speech audio."""
        return self.tts.synthesize(text)

