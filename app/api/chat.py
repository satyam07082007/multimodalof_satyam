import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.chat import Conversation, Message, GuardrailAudit
from app.models.document import Document, MessageSource
from app.schemas.request import ChatRequest
from app.schemas.response import (
    ChatResponse,
    SourceReference,
    ConversationSummary,
    ConversationDetail,
    MessageDetail
)
from app.services.safety import SafetyEngine
from app.services.rag import rag_service
from app.services.llm import LLMService
from app.services.speech import SpeechService
from app.services.multimodal import MultimodalService

logger = logging.getLogger("emotional_support.api.chat")
router = APIRouter(prefix="/api", tags=["Chat & Conversations"])

def get_or_create_conversation(db: Session, conversation_id: Optional[str] = None, title: Optional[str] = None) -> Conversation:
    """Retrieves an existing conversation or creates a new one."""
    if not conversation_id:
        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"

    conv = db.query(Conversation).filter(Conversation.conversation_id == conversation_id).first()
    if not conv:
        conv = Conversation(
            conversation_id=conversation_id,
            title=title or "Supportive Session"
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
    return conv

@router.post("/chat", response_model=ChatResponse)
def chat_text_endpoint(payload: ChatRequest, db: Session = Depends(get_db)):
    """
    Standard text chat endpoint.
    Processes: Safety Guardrails -> RAG Retrieval -> LLM -> Speech Synthesis -> MySQL Persistence.
    """
    conv = get_or_create_conversation(db, payload.conversation_id)
    conversation_id = conv.conversation_id
    user_text = payload.message.strip()

    # 1. Safety Guardrail Evaluation
    safety = SafetyEngine.evaluate(user_text)

    # Audit log if guardrail triggered
    if safety.triggered:
        audit = GuardrailAudit(
            conversation_id=conversation_id,
            trigger_type=safety.trigger_type or "guardrail",
            severity=safety.severity,
            action_taken=safety.mitigation_action or "Flagged"
        )
        db.add(audit)
        db.commit()

    # Critical Guardrail Bypass (Crisis/Violence)
    if safety.override_response and not safety.is_safe:
        # Save messages to database
        user_msg = Message(
            conversation_id=conversation_id,
            role="user",
            content=user_text,
            input_type="text"
        )
        asst_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=safety.override_response,
            input_type="text"
        )
        db.add(user_msg)
        db.add(asst_msg)
        db.commit()

        # Generate audio for crisis reassurance
        audio_file = SpeechService.synthesize_speech(
            "I care deeply about your safety and you are not alone. Please reach out to the crisis resources."
        )
        audio_url = f"/audio/{audio_file}" if audio_file else None

        return ChatResponse(
            conversation_id=conversation_id,
            response=safety.override_response,
            input_type="text",
            audio_url=audio_url,
            sources=[],
            guardrail_triggered=True,
            guardrail_type=safety.trigger_type,
            mood_tag="crisis"
        )

    # 2. Mood & Sentiment Detection
    mood = MultimodalService.detect_mood(user_text)

    # 3. RAG Retrieval
    retrieved_chunks = rag_service.query(user_text, top_k=settings.TOP_K_RETRIEVAL)
    rag_context = rag_service.format_prompt_context(retrieved_chunks)

    # 4. Fetch Conversation History
    recent_msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .limit(6)
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in recent_msgs]

    # 5. LLM Response Generation
    assistant_text = LLMService.generate_response(
        user_message=safety.sanitized_text,
        conversation_history=history,
        rag_context=rag_context,
        mood_tag=mood
    )

    # Prepend medical boundary notice if applicable
    if safety.override_response and safety.trigger_type == "medical_boundary":
        assistant_text = f"{safety.override_response}\n\n---\n{assistant_text}"

    # 6. Speech Synthesis (TTS)
    audio_file = SpeechService.synthesize_speech(assistant_text)
    audio_url = f"/audio/{audio_file}" if audio_file else None

    # 7. Persist to MySQL
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=user_text,
        input_type="text"
    )
    asst_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=assistant_text,
        input_type="text"
    )
    db.add(user_msg)
    db.add(asst_msg)
    db.commit()
    db.refresh(asst_msg)

    # Save source attribution records
    sources_out: List[SourceReference] = []
    for chunk in retrieved_chunks:
        doc_record = db.query(Document).filter(Document.filename == chunk.get("document_name")).first()
        doc_id = doc_record.document_id if doc_record else None
        
        src_link = MessageSource(
            message_id=asst_msg.message_id,
            document_id=doc_id,
            chunk_reference=chunk.get("chunk_reference"),
            excerpt=chunk.get("excerpt"),
            similarity_score=chunk.get("relevance_score", 0.0)
        )
        db.add(src_link)
        sources_out.append(SourceReference(
            document_name=chunk.get("document_name", "Curated Guide"),
            chunk_reference=chunk.get("chunk_reference"),
            relevance_score=chunk.get("relevance_score"),
            excerpt=chunk.get("excerpt")
        ))
    db.commit()

    return ChatResponse(
        conversation_id=conversation_id,
        response=assistant_text,
        input_type="text",
        audio_url=audio_url,
        sources=sources_out,
        guardrail_triggered=safety.triggered,
        guardrail_type=safety.trigger_type,
        mood_tag=mood
    )

@router.get("/conversations", response_model=List[ConversationSummary])
def list_conversations(db: Session = Depends(get_db)):
    """Retrieves all stored conversations with their message counts."""
    convs = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    summaries = []
    for c in convs:
        count = db.query(Message).filter(Message.conversation_id == c.conversation_id).count()
        summaries.append(ConversationSummary(
            conversation_id=c.conversation_id,
            title=c.title,
            created_at=c.created_at.isoformat() if c.created_at else "",
            updated_at=c.updated_at.isoformat() if c.updated_at else "",
            message_count=count
        ))
    return summaries

@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Retrieves single conversation history with attached source references."""
    conv = db.query(Conversation).filter(Conversation.conversation_id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    msg_details = []
    for m in messages:
        src_records = db.query(MessageSource).filter(MessageSource.message_id == m.message_id).all()
        sources = [
            SourceReference(
                document_name=s.document.filename if s.document else "Knowledge Reference",
                chunk_reference=s.chunk_reference,
                relevance_score=s.similarity_score,
                excerpt=s.excerpt
            )
            for s in src_records
        ]
        msg_details.append(MessageDetail(
            message_id=m.message_id,
            role=m.role,
            content=m.content,
            input_type=m.input_type,
            created_at=m.created_at.isoformat() if m.created_at else "",
            sources=sources
        ))

    return ConversationDetail(
        conversation_id=conv.conversation_id,
        title=conv.title,
        created_at=conv.created_at.isoformat() if conv.created_at else "",
        updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
        messages=msg_details
    )

@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Deletes conversation and associated records from database."""
    conv = db.query(Conversation).filter(Conversation.conversation_id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")

    db.delete(conv)
    db.commit()
    logger.info("Deleted conversation %s and associated messages.", conversation_id)
    return {"status": "success", "message": f"Conversation {conversation_id} deleted."}

