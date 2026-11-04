import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class Document(Base):
    __tablename__ = "documents"

    document_id = Column(String(64), primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    document_type = Column(String(50), nullable=False)  # 'pdf', 'txt'
    processing_status = Column(String(50), default="completed", nullable=False)  # 'pending', 'completed', 'failed'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    sources = relationship("MessageSource", back_populates="document")

class MessageSource(Base):
    __tablename__ = "message_sources"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.message_id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(String(64), ForeignKey("documents.document_id", ondelete="SET NULL"), nullable=True, index=True)
    chunk_reference = Column(String(255), nullable=True)
    excerpt = Column(Text, nullable=True)
    similarity_score = Column(Float, default=0.0)

    message = relationship("Message", back_populates="sources")
    document = relationship("Document", back_populates="sources")

