from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.config import settings
from app.database import get_db, DB_ENGINE_TYPE
from app.services.rag import rag_service
from app.schemas.response import HealthResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
def get_health(db: Session = Depends(get_db)):
    """
    Health check endpoint returning application, database, and RAG status.
    """
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.VERSION,
        database_engine=DB_ENGINE_TYPE,
        rag_indexed_chunks=len(rag_service.chunks),
        llm_configured=bool(settings.LLM_API_KEY),
        guardrails_active=settings.ENABLE_STRICT_GUARDRAILS
    )

