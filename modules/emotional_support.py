import os
import logging
from typing import Dict, Any, List, Optional
import google.genai as genai
from app.config import settings
from utils.security import check_crisis_intent, check_medical_boundary
from app.services.rag import rag_service

logger = logging.getLogger("haven.emotional_support")


class EmotionalSupportModule:
    """
    Dedicated Empathetic Emotional Support & Grounding Engine.
    Uses Google Gemini AI for warm, personalized responses enriched with RAG-retrieved
    evidence-based grounding content. Includes crisis escalation (988 Lifeline).
    Supports both instance and classmethod invocation.
    """

    SYSTEM_PROMPT = (
        "You are Haven's dedicated Emotional Care Companion — warm, empathetic, and grounded.\n"
        "Core Principles:\n"
        "1. Active Listening: Always start by validating and naming the user's emotional experience.\n"
        "2. Empathetic Tone: Warm, calm, non-judgmental. Avoid toxic positivity or dismissiveness.\n"
        "3. Evidence-Based Grounding: When appropriate, offer calming techniques as gentle invitations:\n"
        "   - 5-4-3-2-1 Grounding (senses)\n"
        "   - Box Breathing (4-4-4-4 inhale/hold/exhale/hold)\n"
        "   - Physiological Sigh (double inhale + long exhale)\n"
        "   - RAIN Method (Recognize, Allow, Investigate, Nurture)\n"
        "4. Clear Boundaries: Never diagnose clinical conditions, prescribe medication, or claim to replace therapy.\n"
        "5. Transparency: Remind users you are an AI, not a human therapist, when discussing serious concerns.\n"
        "6. Length: Responses should be warm and meaningful — not too short, not overwhelming. 3-5 paragraphs max.\n"
        "7. Closing: End with an open, curious question to invite continued sharing."
    )

    FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash"]
    _instance = None

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
                logger.warning("Gemini config in EmotionalSupport: %s", e)

    @classmethod
    def _get_active_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def _call_gemini(cls, user_text: str, context_chunks: str = "") -> Optional[str]:
        """Calls Gemini with the emotional support system prompt."""
        inst = cls._get_active_instance()
        if not inst.client:
            return None
        parts = [cls.SYSTEM_PROMPT]
        if context_chunks:
            parts.append(
                f"### Relevant Grounding Resources (from Haven Knowledge Base):\n{context_chunks}"
            )
        parts.append(f"User shares: {user_text}\n\nHaven responds:")
        full_prompt = "\n\n".join(parts)

        candidate_models = [inst.model_name] + [m for m in cls.FALLBACK_MODELS if m != inst.model_name]
        for m in candidate_models:
            try:
                res = inst.client.models.generate_content(
                    model=m,
                    contents=full_prompt
                )
                if res and res.text:
                    return res.text.strip()
            except Exception as e:
                logger.warning("Model %s emotional support notice: %s", m, e)
        return None

    @classmethod
    def _detect_mood_tag(cls, text: str) -> str:
        lower = text.lower()
        if any(w in lower for w in ["anxious", "panic", "scared", "fear", "nervous", "worry"]):
            return "anxious"
        if any(w in lower for w in ["sad", "depressed", "crying", "heartbroken", "down", "grief"]):
            return "sad"
        if any(w in lower for w in ["exhausted", "burnout", "tired", "drained", "overwhelm"]):
            return "exhausted"
        if any(w in lower for w in ["lonely", "alone", "isolated", "empty", "disconnected"]):
            return "lonely"
        if any(w in lower for w in ["angry", "mad", "frustrated", "irritated", "rage"]):
            return "frustrated"
        if any(w in lower for w in ["calm", "better", "hope", "grateful", "peace"]):
            return "hopeful"
        return "reflective"

    @classmethod
    def _fallback_care_response(cls, text: str, mood: str, guides: List[Dict[str, Any]]) -> str:
        """Built-in empathetic fallback response when Gemini is not available."""
        openers = {
            "anxious": "I hear how much anxiety and tension you're feeling right now. It is completely natural to feel overwhelmed — your nervous system is trying to protect you, and that takes real energy.",
            "sad": "I am so sorry you are hurting right now. It takes real courage to open up when your heart feels heavy, and you don't have to carry this alone.",
            "exhausted": "It sounds like you've been carrying a tremendous weight for a long time. Burnout is a real signal that your body and mind need gentle, compassionate care.",
            "lonely": "Feeling lonely or isolated can be one of the heaviest aches to sit with. I want you to know your presence matters, and I am right here with you.",
            "frustrated": "It makes complete sense that you're feeling frustrated. When things feel blocked or unfair, anger and irritation are natural responses — not flaws.",
            "hopeful": "I am so glad to hear a glimmer of hope and lightness in your words. These small moments of ease are worth cherishing and building on.",
            "reflective": "Thank you for sharing what is on your mind. I am here to listen without judgment and support you however I can.",
        }
        opener = openers.get(mood, openers["reflective"])

        # Find most relevant grounding exercise from RAG context
        exercise = ""
        for g in guides:
            content = g.get("content", "")
            if "5-4-3-2-1" in content:
                exercise = (
                    "\n\n🌱 **Grounding Exercise — 5-4-3-2-1**:\n"
                    "Let's slow down together. Can you name:\n"
                    "- **5** things you can *see* around you\n"
                    "- **4** things you can *touch*\n"
                    "- **3** things you can *hear*\n"
                    "- **2** things you can *smell*\n"
                    "- **1** thing you can *taste*\n\n"
                    "Take your time. There is no rush."
                )
                break
            elif "Box Breathing" in content or "box breathing" in content:
                exercise = (
                    "\n\n🌬️ **Box Breathing**:\n"
                    "Let's try this together:\n"
                    "- *Inhale* slowly for **4 counts**...\n"
                    "- *Hold* gently for **4 counts**...\n"
                    "- *Exhale* smoothly for **4 counts**...\n"
                    "- *Hold* at the bottom for **4 counts**.\n\n"
                    "Repeat 2-3 times. Notice your shoulders soften."
                )
                break

        closing = "\n\nWhat feels like the kindest thing you can do for yourself in the next hour?"
        return f"{opener}{exercise}{closing}"

    @classmethod
    def process_emotional_input(cls, user_text: str) -> Dict[str, Any]:
        """
        Evaluates emotional input, checks safety guardrails, retrieves RAG context,
        and generates a warm Gemini-powered (or fallback) empathetic response.
        Works both when called as `instance.process_emotional_input(...)` or `EmotionalSupportModule.process_emotional_input(...)`.
        """
        # 1. Critical Crisis & Self-Harm Safety Check — always runs first
        is_crisis, crisis_msg = check_crisis_intent(user_text)
        if is_crisis:
            return {
                "is_crisis": True,
                "response": crisis_msg,
                "sources": [],
                "mood_tag": "crisis"
            }

        # 2. Medical Boundary Check
        is_med, med_disclaimer = check_medical_boundary(user_text)

        # 3. Retrieve evidence-based grounding from knowledge base (RAG)
        retrieved_guides = rag_service.query(user_text, top_k=2)

        # Build context string from RAG results for Gemini enrichment
        context_chunks = ""
        if retrieved_guides:
            context_chunks = "\n\n".join(
                f"[Source: {g.get('document_name', 'Knowledge Base')}]\n{g.get('content', '')[:400]}"
                for g in retrieved_guides
            )

        # 4. Detect mood tag
        mood = cls._detect_mood_tag(user_text)

        # 5. Generate empathetic response — Gemini AI first, fallback second
        ai_response = cls._call_gemini(user_text, context_chunks)
        empathetic_reply = ai_response or cls._fallback_care_response(user_text, mood, retrieved_guides)

        # 6. Prepend medical disclaimer if triggered
        if is_med and med_disclaimer:
            empathetic_reply = f"{med_disclaimer}\n\n---\n\n{empathetic_reply}"

        return {
            "is_crisis": False,
            "response": empathetic_reply,
            "sources": [
                {
                    "document_name": g.get("document_name"),
                    "chunk_reference": g.get("chunk_reference"),
                    "excerpt": g.get("excerpt")
                }
                for g in retrieved_guides
            ],
            "mood_tag": mood
        }
