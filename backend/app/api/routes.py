import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db.database import get_db, DB_ENGINE_TYPE
from backend.app.db.models import ChatSession, ChatMessage, GuardrailLog
from backend.app.services.chat_service import ChatService

logger = logging.getLogger("emotional_support.api")
router = APIRouter(prefix="/api/v1")

@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check verifying API operational status and database connection."""
    try:
        session_count = db.query(ChatSession).count()
        return {
            "status": "healthy",
            "app_name": settings.APP_NAME,
            "version": settings.VERSION,
            "database_engine": DB_ENGINE_TYPE,
            "active_sessions_tracked": session_count,
            "guardrails_active": settings.ENABLE_STRICT_GUARDRAILS,
            "gemini_api_configured": bool(settings.GEMINI_API_KEY)
        }
    except Exception as e:
        logger.error("Health check error: %s", e)
        return {
            "status": "degraded",
            "database_error": str(e),
            "database_engine": DB_ENGINE_TYPE
        }

@router.post("/chat")
async def chat_endpoint(
    session_id: Optional[str] = Form(None),
    message: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """
    Multimodal Chat Endpoint.
    Accepts text, image (file upload), and audio (voice note upload).
    Processes through Guardrails -> RAG -> Multimodal LLM -> TTS -> MySQL.
    """
    image_saved_path: Optional[Path] = None
    audio_saved_path: Optional[Path] = None

    try:
        # Save uploaded image if present
        if image and image.filename:
            file_ext = Path(image.filename).suffix or ".jpg"
            img_name = f"img_{uuid.uuid4().hex[:10]}{file_ext}"
            image_saved_path = settings.UPLOAD_PATH / img_name
            with open(image_saved_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

        # Save uploaded audio if present
        if audio and audio.filename:
            file_ext = Path(audio.filename).suffix or ".wav"
            aud_name = f"voice_{uuid.uuid4().hex[:10]}{file_ext}"
            audio_saved_path = settings.UPLOAD_PATH / aud_name
            with open(audio_saved_path, "wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)

        # Process through chat service
        result = ChatService.process_chat(
            db=db,
            session_id=session_id,
            user_text=message,
            image_path=image_saved_path,
            audio_path=audio_saved_path
        )

        audio_url = None
        if result.get("audio_filename"):
            audio_url = f"/audio/{result['audio_filename']}"

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "session_id": result["session_id"],
                "response": result["response"],
                "audio_url": audio_url,
                "sentiment": result["sentiment"],
                "guardrail_triggered": result["guardrail_triggered"],
                "guardrail_type": result["guardrail_type"],
                "rag_sources": result["rag_sources"]
            }
        )

    except Exception as e:
        logger.error("Error processing multimodal chat: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your message: {str(e)}"
        )

@router.get("/sessions/{session_id}")
def get_session_history(session_id: str, db: Session = Depends(get_db)):
    """Fetches chat dialogue history for the given anonymous session."""
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    return {
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "current_mood": session.current_mood,
        "message_count": len(messages),
        "history": [
            {
                "message_id": m.message_id,
                "sender": m.sender,
                "text": m.text_content,
                "has_image": m.has_image,
                "has_audio": m.has_audio,
                "sentiment": m.sentiment_tag,
                "created_at": m.created_at.isoformat() if m.created_at else None
            }
            for m in messages
        ]
    }

@router.get("/sessions/{session_id}/guardrails")
def get_session_guardrail_logs(session_id: str, db: Session = Depends(get_db)):
    """Audits guardrail events recorded for this session."""
    logs = (
        db.query(GuardrailLog)
        .filter(GuardrailLog.session_id == session_id)
        .order_by(GuardrailLog.created_at.desc())
        .all()
    )
    return {
        "session_id": session_id,
        "total_flags": len(logs),
        "logs": [
            {
                "log_id": l.log_id,
                "trigger_type": l.trigger_type,
                "severity": l.severity,
                "snippet": l.user_input_snippet,
                "mitigation": l.mitigation_action,
                "timestamp": l.created_at.isoformat() if l.created_at else None
            }
            for l in logs
        ]
    }
