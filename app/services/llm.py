import logging
from typing import Optional, List, Dict, Any
import google.genai as genai
from app.config import settings

logger = logging.getLogger("emotional_support.llm")

# Configure Google GenAI Client if key is provided
_genai_client = None
if settings.LLM_API_KEY:
    try:
        _genai_client = genai.Client(api_key=settings.LLM_API_KEY)
    except Exception as e:
        logger.warning("Could not initialize Google GenAI SDK: %s", e)


class LLMService:
    """
    Modular LLM service with provider abstraction.
    Supports Google Gemini via google-genai SDK, and local therapeutic fallback responses.
    """

    SYSTEM_INSTRUCTION = (
        "You are Haven, a compassionate, gentle, and non-judgmental emotional support companion.\n"
        "Core Guidelines:\n"
        "1. Active Listening & Validation: Always acknowledge and validate the user's emotions before offering suggestions.\n"
        "2. Grounded & Empathetic Tone: Maintain warmth and patience. Never use toxic positivity or invalidate their struggle.\n"
        "3. Knowledge Attribution: If retrieved grounding or coping techniques are provided, weave them in gently as invitations, not rigid orders. Mention when advice is drawn from evidence-based grounding frameworks.\n"
        "4. Strict Ethical Boundaries: Never diagnose mental health disorders, prescribe medication, or claim to be a human therapist.\n"
        "5. Manageable Next Steps: Keep responses concise and avoid overwhelming the user with long lists of chores."
    )

    FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash"]

    @classmethod
    def generate_response(
        cls,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        rag_context: Optional[str] = None,
        multimodal_context: Optional[str] = None,
        mood_tag: str = "neutral"
    ) -> str:
        """
        Generates empathetic response using configured LLM or fallback engine.
        """
        global _genai_client
        if _genai_client is None and settings.LLM_API_KEY:
            try:
                _genai_client = genai.Client(api_key=settings.LLM_API_KEY)
            except Exception as e:
                logger.warning("Failed to initialize GenAI client on request: %s", e)

        if _genai_client and settings.LLM_PROVIDER.lower() == "gemini":
            try:
                prompt_parts = [cls.SYSTEM_INSTRUCTION]

                if rag_context:
                    prompt_parts.append(rag_context)

                if multimodal_context:
                    prompt_parts.append(f"### Multimodal Context (Audio/Visual/Document):\n{multimodal_context}\n")

                if conversation_history:
                    history_str = "### Recent Conversation History:\n"
                    for h in conversation_history[-4:]:
                        role = "User" if h["role"] == "user" else "Assistant"
                        history_str += f"{role}: {h['content']}\n"
                    prompt_parts.append(history_str)

                prompt_parts.append(f"User Detected Mood: {mood_tag}")
                prompt_parts.append(f"User Current Message: {user_message}")
                prompt_parts.append("Empathetic Assistant Response:")

                full_prompt = "\n\n".join(prompt_parts)

                primary_model = settings.LLM_MODEL or "gemini-3.5-flash"
                candidate_models = [primary_model] + [m for m in cls.FALLBACK_MODELS if m != primary_model]

                for m in candidate_models:
                    try:
                        response = _genai_client.models.generate_content(
                            model=m,
                            contents=full_prompt
                        )
                        if response and response.text:
                            return response.text.strip()
                    except Exception as model_err:
                        logger.warning("LLM model %s failed: %s. Trying candidate...", m, model_err)

            except Exception as e:
                logger.error("LLM Provider API call failed: %s. Using therapeutic fallback.", e)

        # Fallback offline therapeutic response engine
        return cls._generate_therapeutic_fallback(
            user_message=user_message,
            rag_context=rag_context,
            multimodal_context=multimodal_context,
            mood_tag=mood_tag
        )

    @classmethod
    def _generate_therapeutic_fallback(
        cls,
        user_message: str,
        rag_context: Optional[str],
        multimodal_context: Optional[str],
        mood_tag: str
    ) -> str:
        """
        Offline empathetic response generator when API key is unconfigured or during provider downtime.
        """
        openers = {
            "anxious": "I hear the tension and overwhelm in what you're sharing. When anxiety runs high, even small things can feel overwhelming.",
            "sad": "I am so sorry you're carrying such heaviness right now. It takes courage to put words to feelings when things hurt, and I'm right here with you.",
            "exhausted": "It sounds like you have been pushing yourself through burnout and exhaustion. Your mind and body are asking for genuine, guilt-free rest.",
            "lonely": "Feeling lonely or unseen is such a painful ache. I want you to remember that your feelings are valid, and you are not invisible here.",
            "frustrated": "It makes total sense that you're feeling frustrated. Feeling blocked or overwhelmed naturally stirs up irritation and anger.",
            "hopeful": "I am so glad to notice that spark of lightness and hope in what you've shared. Celebrating these moments of ease is so valuable."
        }
        opener = openers.get(mood_tag, "Thank you for reaching out and sharing what is on your mind. I am here to listen without judgment.")

        techniques = ""
        if rag_context:
            if "5-4-3-2-1" in rag_context:
                techniques = (
                    "\n\n🌱 **Grounding Exercise (5-4-3-2-1)**:\n"
                    "Let's ground our senses together: Look around and name 3 things you can see, "
                    "notice 2 textures your fingertips can touch, and take one deep, steady breath."
                )
            elif "Box Breathing" in rag_context:
                techniques = (
                    "\n\n🌬️ **Box Breathing (4-4-4-4)**:\n"
                    "Let's reset together: Inhale smoothly for 4 seconds... gently hold for 4... "
                    "release softly for 4... and pause for 4. Feel your shoulders drop away from your ears."
                )
            elif "Physiological Sigh" in rag_context:
                techniques = (
                    "\n\n🫁 **Physiological Sigh**:\n"
                    "Try taking two deep inhales through your nose (one deep, followed immediately by a quick top-up sniff), "
                    "then a long, slow sigh out through your mouth. This quickly signals your nervous system that you are safe."
                )
            elif "R.A.I.N." in rag_context:
                techniques = (
                    "\n\n💙 **Mindful Compassion (R.A.I.N.)**:\n"
                    "Place a gentle hand over your chest. Recognize this feeling, allow it to be here without fighting it, "
                    "and whisper to yourself: 'This is a moment of difficulty, and I will be kind to myself.'"
                )

        closing = "\n\nTake things one step at a time. What would feel like the most gentle thing you could do for yourself right now?"
        return f"{opener}{techniques}{closing}"
