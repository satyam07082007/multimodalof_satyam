from typing import Optional, List
from pydantic import BaseModel

class SourceReference(BaseModel):
    document_name: str
    chunk_reference: Optional[str] = None
    relevance_score: Optional[float] = None
    excerpt: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    input_type: str = "text"
    audio_url: Optional[str] = None
    sources: List[SourceReference] = []
    guardrail_triggered: bool = False
    guardrail_type: Optional[str] = None
    mood_tag: str = "neutral"

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    database_engine: str
    rag_indexed_chunks: int
    llm_configured: bool
    guardrails_active: bool

class MessageDetail(BaseModel):
    message_id: int
    role: str
    content: str
    input_type: str
    created_at: str
    sources: List[SourceReference] = []

class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int

class ConversationDetail(BaseModel):
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[MessageDetail]

class DocumentIngestResponse(BaseModel):
    status: str
    total_documents: int
    total_chunks: int
    message: str

