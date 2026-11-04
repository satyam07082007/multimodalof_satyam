import os
import uuid
import logging
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
from gtts import gTTS
import google.generativeai as genai
from backend.app.config import settings

logger = logging.getLogger("emotional_support.multimodal")

# Initialize Gemini if key is provided
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)

class MultimodalProcessor:
    """
    Handles multimodal perception (Vision, Audio STT, Audio TTS, Emotion Tagging).
    """

    @classmethod
    def analyze_image(cls, image_path: Path) -> str:
        """
        Analyzes user-submitted image (e.g. emotion drawing, journal, environment)
        using Gemini Vision or returns an empathetic fallback description.
        """
        if not image_path.exists():
            return ""

        if settings.GEMINI_API_KEY:
            try:
                model = genai.GenerativeModel(settings.GEMINI_MODEL)
                pil_image = Image.open(image_path)
                prompt = (
                    "You are an empathetic emotional support observer. Look at this image provided by a user "
                    "(which might be a drawing, handwritten journal, or scene). "
                    "Describe the emotional undertone, dominant colors, atmosphere, and implied feelings gently in 2-3 sentences."
                )
                response = model.generate_content([prompt, pil_image])
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.error("Gemini Vision analysis error: %s", e)

        # Fallback offline perceptual summary
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                return f"[Image uploaded: {w}x{h} px format. Expressive visual artwork/journal entry shared for emotional context.]"
        except Exception:
            return "[Visual image shared for emotional context]"

    @classmethod
    def transcribe_audio(cls, audio_path: Path) -> str:
        """
        Transcribes audio note using Gemini Multimodal or audio recognition.
        """
        if not audio_path.exists():
            return ""

        if settings.GEMINI_API_KEY:
            try:
                # Gemini 1.5 multimodal audio upload
                audio_file = genai.upload_file(path=str(audio_path))
                model = genai.GenerativeModel(settings.GEMINI_MODEL)
                response = model.generate_content([
                    "Please transcribe the spoken emotional message from this audio clip exactly as spoken.",
                    audio_file
                ])
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.error("Gemini audio transcription error: %s", e)

        return "[Audio voice note received. Emotional vocal expression recorded.]"

    @classmethod
    def generate_speech_response(cls, text_to_speak: str) -> Optional[str]:
        """
        Generates a soothing audio file (.mp3) from the assistant's text response using gTTS.
        Returns the filename stored in the media directory.
        """
        try:
            # Clean markdown formatting for clean audio synthesis
            clean_text = text_to_speak.replace("*", "").replace("#", "").replace("-", " ")
            # Truncate for audio synthesis if exceptionally long
            if len(clean_text) > 400:
                clean_text = clean_text[:400] + "... Take a deep breath, I am here with you."

            audio_filename = f"voice_response_{uuid.uuid4().hex[:10]}.mp3"
            dest_file = settings.AUDIO_PATH / audio_filename

            tts = gTTS(text=clean_text, lang="en", tld="com", slow=False)
            tts.save(str(dest_file))
            logger.info("Generated TTS audio: %s", audio_filename)
            return audio_filename
        except Exception as e:
            logger.error("TTS generation error: %s", e)
            return None

    @classmethod
    def detect_sentiment_and_mood(cls, text: str) -> str:
        """
        Tags the predominant emotional mood of the message for session tracking.
        """
        lower = text.lower()
        if any(w in lower for w in ["anxious", "panic", "scared", "fear", "nervous", "dread", "shaking"]):
            return "anxious"
        if any(w in lower for w in ["sad", "depressed", "crying", "tears", "grief", "heartbroken", "down"]):
            return "sad"
        if any(w in lower for w in ["exhausted", "burnout", "tired", "drained", "can't focus", "heavy"]):
            return "exhausted"
        if any(w in lower for w in ["lonely", "alone", "isolated", "empty", "abandoned"]):
            return "lonely"
        if any(w in lower for w in ["angry", "furious", "mad", "frustrated", "hate", "irritated"]):
            return "frustrated"
        if any(w in lower for w in ["better", "hope", "thank you", "calmer", "relieved", "good"]):
            return "hopeful"
        return "neutral"
