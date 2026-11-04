from typing import Optional, List
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(None, description="Unique conversation ID. Generated if omitted.")
    message: str = Field(..., min_length=1, max_length=4000, description="User's input message.")

class DocumentIngestRequest(BaseModel):
    reindex: bool = Field(False, description="Whether to rebuild index from scratch.")
    filenames: Optional[List[str]] = Field(None, description="Specific filenames to ingest. Ingests all if None.")

