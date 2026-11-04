import os
from pathlib import Path
from typing import List, Union
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_DIR = BASE_DIR / "media"
UPLOAD_DIR = MEDIA_DIR / "uploads"
AUDIO_OUTPUT_DIR = MEDIA_DIR / "audio_responses"
GENERATED_IMAGES_DIR = MEDIA_DIR / "generated_images"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base" / "sample_documents"
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"

# Ensure all application directories exist automatically
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
    # App Settings
    APP_NAME: str = "HAVEN AI - Multimodal Intelligent Assistant"
    VERSION: str = "2.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    STREAMLIT_PORT: int = 8501
    BACKEND_URL: str = "http://127.0.0.1:8000"
    ALLOWED_CORS_ORIGINS: Union[str, List[str]] = "http://localhost:8501,http://127.0.0.1:8501"

    # MySQL Database Settings
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DATABASE: str = "emotional_support_db"
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""

    # LLM Settings
    LLM_PROVIDER: str = "gemini"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gemini-1.5-flash"

    # RAG Settings
    EMBEDDING_MODEL: str = "tfidf_semantic"
    VECTOR_STORE_PATH: str = "./data/vector_index.json"
    SIMILARITY_THRESHOLD: float = 0.1
    TOP_K_RETRIEVAL: int = 3

    # Uploads & Security
    MAX_UPLOAD_SIZE_MB: int = 25
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

    DATA_PATH: Path = DATA_DIR
    DATA_DIR: Path = DATA_DIR

    CACHE_PATH: Path = CACHE_DIR
    CACHE_DIR: Path = CACHE_DIR

    @property
    def cors_origins(self) -> List[str]:
        if isinstance(self.ALLOWED_CORS_ORIGINS, list):
            return self.ALLOWED_CORS_ORIGINS
        return [o.strip() for o in self.ALLOWED_CORS_ORIGINS.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def mysql_database_url(self) -> str:
        """Constructs MySQL SQLAlchemy connection string."""
        if self.MYSQL_PASSWORD:
            return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}?charset=utf8mb4"
        return f"mysql+pymysql://{self.MYSQL_USER}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}?charset=utf8mb4"

    @property
    def sqlite_database_url(self) -> str:
        """Fallback local SQLite DB in case MySQL is unconfigured or offline."""
        db_file = DATA_DIR / "multimodal_chat.db"
        return f"sqlite:///{db_file}"

    model_config = ConfigDict(env_file=".env", extra="allow")


settings = Settings()
