import os
from pathlib import Path
from typing import List, Union
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MEDIA_DIR = BASE_DIR / "media"
UPLOAD_DIR = MEDIA_DIR / "uploads"
AUDIO_OUTPUT_DIR = MEDIA_DIR / "audio_responses"
GENERATED_IMAGES_DIR = MEDIA_DIR / "generated_images"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base" / "sample_documents"
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"

# Ensure directories exist
for directory in [
    MEDIA_DIR,
    UPLOAD_DIR,
    AUDIO_OUTPUT_DIR,
    GENERATED_IMAGES_DIR,
    KNOWLEDGE_BASE_DIR,
    DATA_DIR,
    CACHE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    APP_NAME: str = "HAVEN AI - Multimodal Intelligent Assistant"
    VERSION: str = "2.0.0"
    DEBUG: bool = True
    
    # AI Engine
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # MySQL Configuration
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DB: str = os.getenv("MYSQL_DB", "emotional_support_db")
    
    # Guardrail Controls
    ENABLE_STRICT_GUARDRAILS: bool = True
    CRISIS_HELPLINE_PHONE: str = "988"
    CRISIS_TEXT_LINE: str = "Text HOME to 741741"
    
    # Directory Paths (Both _DIR and _PATH aliases for maximum compatibility)
    BASE_PATH: Path = BASE_DIR
    BASE_DIR: Path = BASE_DIR

    MEDIA_PATH: Path = MEDIA_DIR
    MEDIA_DIR: Path = MEDIA_DIR

    UPLOAD_PATH: Path = UPLOAD_DIR
    UPLOAD_DIR: Path = UPLOAD_DIR

    AUDIO_PATH: Path = AUDIO_OUTPUT_DIR
    AUDIO_DIR: Path = AUDIO_OUTPUT_DIR

    GENERATED_IMAGES_PATH: Path = GENERATED_IMAGES_DIR
    GENERATED_IMAGES_DIR: Path = GENERATED_IMAGES_DIR

    KNOWLEDGE_BASE_PATH: Path = KNOWLEDGE_BASE_DIR
    KNOWLEDGE_BASE_DIR: Path = KNOWLEDGE_BASE_DIR

    DATA_PATH: Path = DATA_DIR / "wellness_knowledge.json"
    DATA_DIR: Path = DATA_DIR

    CACHE_PATH: Path = CACHE_DIR
    CACHE_DIR: Path = CACHE_DIR

    @property
    def mysql_database_url(self) -> str:
        """Constructs MySQL SQLAlchemy connection string."""
        if self.MYSQL_PASSWORD:
            return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}?charset=utf8mb4"
        return f"mysql+pymysql://{self.MYSQL_USER}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}?charset=utf8mb4"

    @property
    def fallback_database_url(self) -> str:
        """Fallback local SQLite DB in case MySQL server is offline during initial local runs."""
        db_file = BASE_DIR / "fallback_dev.db"
        return f"sqlite:///{db_file}"

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
