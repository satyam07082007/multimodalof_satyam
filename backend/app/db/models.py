import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from backend.app.db.database import Base

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id = Column(String(64), primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_active = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    current_mood = Column(String(50), default="neutral")

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    guardrail_logs = relationship("GuardrailLog", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    message_id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    session_id = Column(String(64), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    sender = Column(String(20), nullable=False) # 'user', 'assistant', 'system'
    text_content = Column(Text, nullable=True)
    has_image = Column(Boolean, default=False)
    image_path = Column(String(255), nullable=True)
    has_audio = Column(Boolean, default=False)
    audio_path = Column(String(255), nullable=True)
    sentiment_tag = Column(String(50), default="neutral")
    guardrail_flagged = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")

class GuardrailLog(Base):
    __tablename__ = "guardrail_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    session_id = Column(String(64), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    trigger_type = Column(String(50), nullable=False) # 'self_harm', 'violence', 'medical_advice_seeking', 'pii_leak', 'toxic_language'
    severity = Column(String(20), nullable=False) # 'low', 'medium', 'critical'
    user_input_snippet = Column(Text, nullable=True)
    mitigation_action = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("ChatSession", back_populates="guardrail_logs")
