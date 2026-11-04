import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

from app.config import settings
from app.utils.logging_config import setup_logging
from app.database import init_db, DB_ENGINE_TYPE
from app.api.health import router as health_router
from app.api.chat import router as chat_router
from app.api.upload import router as upload_router
from app.services.rag import rag_service

# Initialize logging
logger = setup_logging()

# Initialize FastAPI App
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Multimodal Emotional Support Chatbot API with RAG, Guardrails, and MySQL persistence."
)

# CORS Middleware (Strictly configured for Streamlit UI origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global safe exception handler to avoid exposing stack traces or credentials
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled server exception at %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred. Please try again shortly."}
    )

# Static file mount for synthesized TTS audio playback
app.mount("/audio", StaticFiles(directory=str(settings.AUDIO_PATH)), name="audio")

# Register API Routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(upload_router)

@app.on_event("startup")
def on_startup():
    """Initializes database schema and verifies knowledge base on startup."""
    logger.info("Initializing application database schema...")
    init_db()
    logger.info("Database active using engine: %s", DB_ENGINE_TYPE)
    logger.info("RAG vector store loaded with %d knowledge chunks.", len(rag_service.chunks))
    logger.info("Multimodal Emotional Support API is live on %s:%d", settings.API_HOST, settings.API_PORT)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )

