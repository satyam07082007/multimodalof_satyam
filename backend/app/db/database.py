import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.app.config import settings

logger = logging.getLogger("emotional_support.db")
Base = declarative_base()

def create_app_engine():
    """
    Attempts to connect to MySQL database as specified.
    If MySQL server is unavailable or credentials are unconfigured,
    gracefully falls back to local SQLite to ensure uninterrupted local testing.
    """
    mysql_url = settings.mysql_database_url
    try:
        engine = create_engine(
            mysql_url,
            pool_recycle=3600,
            pool_pre_ping=True
        )
        # Test connection
        with engine.connect() as conn:
            logger.info("Successfully connected to MySQL database: %s", settings.MYSQL_DB)
        return engine, "MySQL"
    except Exception as e:
        logger.warning(
            "Could not connect to MySQL (%s). Falling back to SQLite local database for development. Error: %s",
            mysql_url, e
        )
        sqlite_engine = create_engine(
            settings.fallback_database_url,
            connect_args={"check_same_thread": False}
        )
        return sqlite_engine, "SQLite Fallback"

engine, DB_ENGINE_TYPE = create_app_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """FastAPI dependency for yielding database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initializes tables in database."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized with engine type: %s", DB_ENGINE_TYPE)
