import shutil
import logging
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.chat import Conversation, Message, GuardrailAudit
from app.models.document import Document, MessageSource
from app.schemas.response import ChatResponse, SourceReference, DocumentIngestResponse
from app.utils.file_validation import validate_uploaded_file
from app.services.safety import SafetyEngine
from app.services.rag import rag_service
from app.services.llm import LLMService
from app.services.speech import SpeechService
from app.services.multimodal import MultimodalService
from app.api.chat import get_or_create_conversation

logger = logging.getLogger("emotional_support.api.upload")
router = APIRouter(prefix="/api", tags=["Multimodal & Ingestion"])

@router.post("/chat/multimodal", response_model=ChatResponse)
async def chat_multimodal_endpoint(
    conversation_id: Optional[str] = Form(None),
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """
    Multimodal Chat Endpoint.
    Accepts text message alongside an image, voice recording, or document attachment.
    Applies security validation, multimodal processing, RAG, safety guardrails, and DB logging.
    """
    conv = get_or_create_conversation(db, conversation_id)
    conversation_id = conv.conversation_id

    saved_file_path: Optional[Path] = None
    input_type = "text"
    multimodal_context = ""
    transcribed_user_speech = ""

    try:
        # Validate and temporarily save file if present
        if file and file.filename:
            category, safe_filename = validate_uploaded_file(file)
            input_type = category
            saved_file_path = settings.UPLOAD_PATH / safe_filename
            
            with open(saved_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Process according to category
            if category == "image":
                visual_summary = MultimodalService.analyze_image(saved_file_path)
                multimodal_context = f"[Visual Artwork / Image Context]: {visual_summary}"
            elif category == "audio":
                transcribed_user_speech = MultimodalService.transcribe_audio(saved_file_path)
                multimodal_context = f"[Transcribed Spoken Voice]: {transcribed_user_speech}"
            elif category == "document":
                doc_text = MultimodalService.extract_document_context(saved_file_path)
                multimodal_context = f"[Attached Document Reference ({file.filename})]:\n{doc_text}"

        # Combine text content
        raw_user_msg = (message or "").strip()
        if not raw_user_msg and transcribed_user_speech:
            raw_user_msg = transcribed_user_speech

        effective_text = raw_user_msg
        if not effective_text:
            if input_type == "image":
                effective_text = "I shared this image with you to reflect how I feel."
            elif input_type == "audio":
                effective_text = "I sent a voice note sharing my thoughts."
            elif input_type == "document":
                effective_text = "I shared this document for emotional context."
            else:
                effective_text = "Hello, I am looking for some gentle emotional support."

        # 1. Safety Guardrail Evaluation
        safety = SafetyEngine.evaluate(effective_text)

        if safety.triggered:
            audit = GuardrailAudit(
                conversation_id=conversation_id,
                trigger_type=safety.trigger_type or "guardrail",
                severity=safety.severity,
                action_taken=safety.mitigation_action or "Flagged"
            )
            db.add(audit)
            db.commit()

        # Handle Critical Safety Bypass
        if safety.override_response and not safety.is_safe:
            user_msg = Message(
                conversation_id=conversation_id,
                role="user",
                content=effective_text,
                input_type=input_type
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

            audio_file = SpeechService.synthesize_speech("I care deeply about your safety. You are not alone. Please reach out to the crisis resources.")
            audio_url = f"/audio/{audio_file}" if audio_file else None

            return ChatResponse(
                conversation_id=conversation_id,
                response=safety.override_response,
                input_type=input_type,
                audio_url=audio_url,
                sources=[],
                guardrail_triggered=True,
                guardrail_type=safety.trigger_type,
                mood_tag="crisis"
            )

        # 2. Mood Detection
        mood = MultimodalService.detect_mood(f"{effective_text} {multimodal_context}")

        # 3. RAG Knowledge Query
        retrieved_chunks = rag_service.query(effective_text, top_k=settings.TOP_K_RETRIEVAL)
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

        # 5. LLM Response
        assistant_text = LLMService.generate_response(
            user_message=safety.sanitized_text,
            conversation_history=history,
            rag_context=rag_context,
            multimodal_context=multimodal_context,
            mood_tag=mood
        )

        if safety.override_response and safety.trigger_type == "medical_boundary":
            assistant_text = f"{safety.override_response}\n\n---\n{assistant_text}"

        # 6. Speech Synthesis (TTS)
        audio_file = SpeechService.synthesize_speech(assistant_text)
        audio_url = f"/audio/{audio_file}" if audio_file else None

        # 7. Persist to MySQL
        user_msg = Message(
            conversation_id=conversation_id,
            role="user",
            content=effective_text,
            input_type=input_type
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

        # Link sources in DB
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
            input_type=input_type,
            audio_url=audio_url,
            sources=sources_out,
            guardrail_triggered=safety.triggered,
            guardrail_type=safety.trigger_type,
            mood_tag=mood
        )

    finally:
        # Privacy policy: Clean up temporary multimodal files to avoid storing personal audio/images permanently
        if saved_file_path and saved_file_path.exists():
            try:
                saved_file_path.unlink()
            except Exception as e:
                logger.warning("Could not delete temporary upload %s: %s", saved_file_path, e)

@router.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_documents_endpoint(
    files: Optional[List[UploadFile]] = File(None),
    db: Session = Depends(get_db)
):
    """
    Ingests approved knowledge-base documents into the RAG vector store and database.
    Restricted to local development / admin environment.
    """
    saved_paths: List[Path] = []
    try:
        if files:
            for f in files:
                if not f.filename:
                    continue
                ext = Path(f.filename).suffix.lower()
                if ext not in {".pdf", ".txt"}:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Unsupported knowledge document format '{ext}'. Must be .pdf or .txt"
                    )
                dest = settings.KNOWLEDGE_BASE_PATH / Path(f.filename).name
                with open(dest, "wb") as buffer:
                    shutil.copyfileobj(f.file, buffer)
                saved_paths.append(dest)

        result = rag_service.ingest_documents()

        # Update MySQL documents table
        for path in settings.KNOWLEDGE_BASE_PATH.glob("*.*"):
            if path.suffix.lower() in {".pdf", ".txt"}:
                doc_id = f"doc_{path.stem}"
                existing = db.query(Document).filter(Document.document_id == doc_id).first()
                if not existing:
                    db.add(Document(
                        document_id=doc_id,
                        filename=path.name,
                        document_type=path.suffix.lstrip(".").lower(),
                        processing_status="completed"
                    ))
        db.commit()

        return DocumentIngestResponse(
            status="success",
            total_documents=result["documents_indexed"],
            total_chunks=result["chunks_indexed"],
            message="Knowledge base successfully ingested and synchronized with vector store."
        )
    except Exception as e:
        logger.error("Error during document ingestion: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )

