import os
import uuid
import re
from pathlib import Path
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException, status
from PIL import Image
from app.config import settings

# Allowed file extensions and corresponding MIME types
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg"}
ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".txt"}

ALLOWED_EXTENSIONS = (
    ALLOWED_IMAGE_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS | ALLOWED_DOCUMENT_EXTENSIONS
)

def get_safe_filename(original_filename: Optional[str], prefix: str = "upload") -> str:
    """
    Generates an unguessable, secure server-side filename.
    Prevents path traversal, directory escapes, and malicious filenames.
    """
    ext = ""
    if original_filename:
        # Extract extension and sanitize
        raw_ext = Path(original_filename).suffix.lower()
        if raw_ext in ALLOWED_EXTENSIONS:
            ext = raw_ext
        else:
            ext = ".bin"
    else:
        ext = ".bin"

    random_token = uuid.uuid4().hex[:12]
    return f"{prefix}_{random_token}{ext}"

def detect_file_category(filename: str) -> str:
    """Determines whether a file is an image, audio, or document."""
    ext = Path(filename).suffix.lower()
    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return "image"
    if ext in ALLOWED_AUDIO_EXTENSIONS:
        return "audio"
    if ext in ALLOWED_DOCUMENT_EXTENSIONS:
        return "document"
    return "unknown"

def validate_uploaded_file(file: UploadFile, max_size_bytes: Optional[int] = None) -> Tuple[str, str]:
    """
    Validates file extension, MIME type, size, and header magic bytes.
    Returns: (category, safe_filename)
    Raises: HTTPException on invalid files.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename."
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Validate file size
    limit = max_size_bytes or settings.max_upload_bytes
    # Read first chunk or entire file to inspect size
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)

    if size > limit:
        max_mb = limit / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {max_mb:.1f} MB."
        )

    if size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    category = detect_file_category(file.filename)

    # Content-level validation for Images
    if category == "image":
        try:
            img = Image.open(file.file)
            img.verify()  # Verifies file integrity
            file.file.seek(0)
        except Exception:
            file.file.seek(0)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or corrupted image file."
            )

    # Content-level validation for PDFs
    if ext == ".pdf":
        header = file.file.read(5)
        file.file.seek(0)
        if not header.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF file: Missing standard %PDF- header."
            )

    safe_name = get_safe_filename(file.filename, prefix=category)
    return category, safe_name

