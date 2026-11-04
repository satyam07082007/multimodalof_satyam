import re
import uuid
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from PIL import Image

# Allowed extensions across all multimodal formats
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg"}
ALLOWED_DOC_EXTS = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".pptx"}
ALL_ALLOWED_EXTS = ALLOWED_IMAGE_EXTS | ALLOWED_AUDIO_EXTS | ALLOWED_DOC_EXTS

# PII regex patterns
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Prompt injection detection
INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"disregard system instructions",
    r"system override",
    r"bypass safety guidelines",
    r"reveal system prompt",
    r"you are now in developer mode",
    r"jailbreak"
]

# Crisis / Self-harm patterns
CRISIS_PATTERNS = [
    r"\bkill myself\b", r"\bsuicide\b", r"\bend my life\b", r"\bwant to die\b",
    r"\bhurt myself\b", r"\bcut myself\b", r"\boverdose\b", r"\bno reason to live\b",
    r"\bbetter off dead\b", r"\bgoodbye forever\b", r"\btired of living\b",
    r"\bcommit suicide\b", r"\bend it all\b"
]

# Medical diagnosis boundary
MEDICAL_PATTERNS = [
    r"\bdiagnose me\b", r"\bwhat medication should i take\b", r"\bprescribe\b",
    r"\bdo i have bipolar\b", r"\bdo i have schizophrenia\b", r"\bdosage of\b",
    r"\bmedical treatment plan\b", r"\bstop taking my pills\b", r"\bclinical diagnosis\b"
]

def sanitize_pii(text: str) -> Tuple[str, bool]:
    """Redacts emails, phone numbers, and SSNs from text."""
    redacted = text
    found = False
    if EMAIL_PATTERN.search(redacted):
        redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)
        found = True
    if PHONE_PATTERN.search(redacted):
        redacted = PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
        found = True
    if SSN_PATTERN.search(redacted):
        redacted = SSN_PATTERN.sub("[REDACTED_SSN]", redacted)
        found = True
    return redacted, found

def check_prompt_injection(text: str) -> bool:
    """Returns True if text contains prompt-injection attack vectors."""
    lower = text.lower()
    return any(re.search(pat, lower) for pat in INJECTION_PATTERNS)

def check_crisis_intent(text: str) -> Tuple[bool, Optional[str]]:
    """Checks for self-harm or suicidal ideation and returns crisis intervention message."""
    lower = text.lower()
    for pat in CRISIS_PATTERNS:
        if re.search(pat, lower):
            crisis_msg = (
                "🤍 **I care deeply about your safety and you do not have to carry this alone.**\n\n"
                "It sounds like you are carrying immense distress right now. While I am an AI assistant and cannot provide emergency intervention, compassionate help is available 24/7:\n\n"
                "- 📞 **Call or Text 988** (Suicide & Crisis Lifeline — Free, Confidential, 24/7 in US & Canada)\n"
                "- 💬 **Text HOME to 741741** (Crisis Text Line)\n"
                "- 🌍 **International Support**: Visit [Befrienders Worldwide](https://www.befrienders.org/) or call local emergency services.\n\n"
                "Please reach out to one of these resources or a trusted person right now. Your safety and life matter."
            )
            return True, crisis_msg
    return False, None

def check_medical_boundary(text: str) -> Tuple[bool, Optional[str]]:
    """Checks if user is seeking medical prescriptions or psychiatric diagnosis."""
    lower = text.lower()
    for pat in MEDICAL_PATTERNS:
        if re.search(pat, lower):
            disclaimer = (
                "⚠️ **Medical Boundary Notice**: I am an AI assistant, not a licensed physician, psychiatrist, or medical professional. "
                "I cannot provide clinical diagnoses or recommend prescription medications. Please consult a qualified healthcare provider for medical evaluations."
            )
            return True, disclaimer
    return False, None

def generate_safe_filename(original_name: str, prefix: str = "file") -> str:
    """Generates an unguessable, safe server-side filename."""
    ext = Path(original_name).suffix.lower() if original_name else ".bin"
    if ext not in ALL_ALLOWED_EXTS:
        ext = ".bin"
    return f"{prefix}_{uuid.uuid4().hex[:12]}{ext}"

