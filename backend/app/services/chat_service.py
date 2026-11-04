import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
import google.generativeai as genai

from backend.app.config import settings
from backend.app.db.models import ChatSession, ChatMessage, GuardrailLog
from backend.app.guardrails.safety import GuardrailEngine, GuardrailResult
from backend.app.rag.retriever import WellnessRAGRetriever
from backend.app.services.multimodal import MultimodalProcessor

logger = logging.getLogger("emotional_support.chat_service")

# Initialize RAG retriever
rag_retriever = WellnessRAGRetriever()

class ChatService:
    """
    Core orchestrator handling the full Multimodal Emotional Support flow:
    Session -> Input Normalization -> Guardrails -> RAG -> LLM -> TTS -> DB Logging
    """

    @classmethod
    def get_or_create_session(cls, db: Session, session_id: Optional[str] = None) -> ChatSession:
        if not session_id:
            session_id = f"sess_{uuid.uuid4().hex[:12]}"

        session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if not session:
            session = ChatSession(session_id=session_id, current_mood="neutral")
            db.add(session)
            db.commit()
            db.refresh(session)
        return session

    @classmethod
    def process_chat(
        cls,
        db: Session,
        session_id: Optional[str] = None,
        user_text: Optional[str] = None,
        image_path: Optional[Path] = None,
        audio_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        session = cls.get_or_create_session(db, session_id)
        current_session_id = session.session_id

        # 1. Process Multimodal Inputs
        transcribed_text = ""
        if audio_path and audio_path.exists():
            transcribed_text = MultimodalProcessor.transcribe_audio(audio_path)

        visual_context = ""
        if image_path and image_path.exists():
            visual_context = MultimodalProcessor.analyze_image(image_path)

        # Merge input text
        effective_user_text = (user_text or "").strip()
        if not effective_user_text and transcribed_text:
            effective_user_text = transcribed_text

        display_user_text = effective_user_text
        if visual_context:
            effective_user_text += f"\n[Visual Context]: {visual_context}"

        if not effective_user_text:
            effective_user_text = "Hello, I am reaching out for emotional support."

        # 2. Evaluate Safety Guardrails
        guardrail_result: GuardrailResult = GuardrailEngine.evaluate(effective_user_text)

        # Log guardrail triggers if any
        if guardrail_result.triggered:
            log_entry = GuardrailLog(
                session_id=current_session_id,
                trigger_type=guardrail_result.trigger_type or "guardrail_alert",
                severity=guardrail_result.severity or "medium",
                user_input_snippet=effective_user_text[:300],
                mitigation_action=guardrail_result.mitigation_action or "Flagged"
            )
            db.add(log_entry)
            db.commit()

        # If critical guardrail triggered (self-harm/crisis), bypass LLM and immediately prioritize safety
        if guardrail_result.override_response and not guardrail_result.is_safe:
            emergency_reply = guardrail_result.override_response
            audio_file = MultimodalProcessor.generate_speech_response("I care deeply about your safety. You are not alone. Please reach out to the crisis resources.")

            # Record in DB
            cls._save_messages(
                db=db,
                session_id=current_session_id,
                user_text=display_user_text,
                assistant_text=emergency_reply,
                image_path=str(image_path) if image_path else None,
                audio_path=str(audio_path) if audio_path else None,
                sentiment_tag="critical",
                guardrail_flagged=True
            )

            return {
                "session_id": current_session_id,
                "response": emergency_reply,
                "audio_filename": audio_file,
                "sentiment": "crisis",
                "guardrail_triggered": True,
                "guardrail_type": guardrail_result.trigger_type,
                "rag_sources": []
            }

        # 3. Detect Sentiment & Mood
        sentiment = MultimodalProcessor.detect_sentiment_and_mood(effective_user_text)
        session.current_mood = sentiment
        db.commit()

        # 4. RAG Knowledge Retrieval
        rag_query = f"{effective_user_text} {sentiment}"
        retrieved_docs = rag_retriever.retrieve(query=rag_query, top_k=2)
        rag_context = rag_retriever.format_context_for_prompt(retrieved_docs)

        # 5. Multimodal LLM Generation
        assistant_reply = cls._generate_empathetic_response(
            user_text=guardrail_result.sanitized_text,
            visual_context=visual_context,
            rag_context=rag_context,
            sentiment=sentiment
        )

        # If medical boundary was triggered, append disclaimer gently
        if guardrail_result.override_response and guardrail_result.trigger_type == "medical_advice_seeking":
            assistant_reply = f"{guardrail_result.override_response}\n\n---\n{assistant_reply}"

        # 6. Generate Voice Speech (TTS)
        audio_filename = MultimodalProcessor.generate_speech_response(assistant_reply)

        # 7. Persist to MySQL
        cls._save_messages(
            db=db,
            session_id=current_session_id,
            user_text=display_user_text,
            assistant_text=assistant_reply,
            image_path=str(image_path) if image_path else None,
            audio_path=str(audio_path) if audio_path else None,
            sentiment_tag=sentiment,
            guardrail_flagged=guardrail_result.triggered
        )

        return {
            "session_id": current_session_id,
            "response": assistant_reply,
            "audio_filename": audio_filename,
            "sentiment": sentiment,
            "guardrail_triggered": guardrail_result.triggered,
            "guardrail_type": guardrail_result.trigger_type,
            "rag_sources": [
                {"title": doc.get("title"), "topic": doc.get("topic"), "score": doc.get("relevance_score")}
                for doc in retrieved_docs
            ]
        }

    @classmethod
    def _generate_empathetic_response(
        cls,
        user_text: str,
        visual_context: str,
        rag_context: str,
        sentiment: str
    ) -> str:
        """Calls Gemini API or falls back to therapeutic response template."""
        system_instruction = (
            "You are a dedicated, compassionate, and gentle personal emotional support companion. "
            "Your objective is to provide a safe, non-judgmental space for the user.\n"
            "Principles to follow:\n"
            "1. Practice Active Listening and Emotional Validation first (let them know their feelings make sense).\n"
            "2. Maintain a warm, soothing, and grounded tone without toxic positivity.\n"
            "3. If relevant psychological coping frameworks or grounding exercises are provided in the context, "
            "naturally weave them into your response as an invitation, never an obligation.\n"
            "4. Never pretend to be a medical doctor or diagnose illnesses.\n"
            "5. Keep responses concise, warm, and readable."
        )

        if settings.GEMINI_API_KEY:
            try:
                model = genai.GenerativeModel(settings.GEMINI_MODEL)
                prompt = (
                    f"{system_instruction}\n\n"
                    f"{rag_context}\n\n"
                    f"User Detected Mood: {sentiment}\n"
                    f"Visual/Image Insight: {visual_context if visual_context else 'None'}\n\n"
                    f"User Message: {user_text}\n\n"
                    f"Empathetic Response:"
                )
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.error("Gemini API call failed: %s", e)

        # Empathetic Fallback Response Engine
        return cls._offline_empathetic_reply(user_text, rag_context, sentiment)

    @classmethod
    def _offline_empathetic_reply(cls, user_text: str, rag_context: str, sentiment: str) -> str:
        """Fallback response if Gemini API key is pending."""
        validation_openers = {
            "anxious": "I hear how much anxiety and tension you're feeling right now. It is completely understandable to feel overwhelmed when everything feels moving so fast.",
            "sad": "I am so sorry you are hurting right now. It takes real courage to open up when your heart feels heavy, and I want you to know you don't have to carry this all alone.",
            "exhausted": "It sounds like you've been carrying a tremendous weight for a long time. Burnout and exhaustion are real signals that your body and mind need gentle care.",
            "lonely": "Feeling lonely or isolated can be one of the heaviest ache to sit with. I want to remind you that your presence matters, and I am here with you right now.",
            "frustrated": "It makes total sense that you're feeling frustrated. When things feel unfair or blocked, feeling anger or irritation is completely natural.",
            "hopeful": "I am so glad to hear a glimmer of hope and lightness in your words. Cherishing these small moments of ease is a wonderful gift to yourself."
        }
        opener = validation_openers.get(sentiment, "Thank you for sharing what is on your mind. I am here to listen and support you.")

        exercise_suggestion = ""
        if "5-4-3-2-1" in rag_context:
            exercise_suggestion = "\n\n🌱 **Grounding Exercise (5-4-3-2-1)**:\nTake a gentle breath and look around your room: Can you find 3 distinct textures you can touch, and take a sip of cool water? Notice the sensation of the ground supporting you."
        elif "Box Breathing" in rag_context:
            exercise_suggestion = "\n\n🌬️ **Gentle Breathing Exercise**:\nLet's try a 4-second breath together: Inhale slowly for 4 seconds... gently hold for 4... and release smoothly for 4. Feel your shoulders drop down."
        elif "R.A.I.N." in rag_context:
            exercise_suggestion = "\n\n💙 **Compassion Practice (R.A.I.N.)**:\nAllow yourself to feel this without judgment. Place a gentle, comforting hand over your chest and whisper: 'This is hard right now, but I am safe in this moment.'"

        closing = "\n\nTake things one breath at a time. What feels like the kindest thing you can do for yourself in the next hour?"
        return f"{opener}{exercise_suggestion}{closing}"

    @classmethod
    def _save_messages(
        cls,
        db: Session,
        session_id: str,
        user_text: str,
        assistant_text: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        sentiment_tag: str = "neutral",
        guardrail_flagged: bool = False
    ):
        """Saves dialogue interaction to MySQL database."""
        try:
            # User message
            user_msg = ChatMessage(
                session_id=session_id,
                sender="user",
                text_content=user_text,
                has_image=bool(image_path),
                image_path=image_path,
                has_audio=bool(audio_path),
                audio_path=audio_path,
                sentiment_tag=sentiment_tag,
                guardrail_flagged=guardrail_flagged
            )
            # Assistant message
            assistant_msg = ChatMessage(
                session_id=session_id,
                sender="assistant",
                text_content=assistant_text,
                has_audio=True,
                sentiment_tag=sentiment_tag,
                guardrail_flagged=guardrail_flagged
            )
            db.add(user_msg)
            db.add(assistant_msg)
            db.commit()
        except Exception as e:
            logger.error("Failed to persist messages to database: %s", e)
            db.rollback()
