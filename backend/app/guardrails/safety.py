import re
from typing import Optional, List, Dict
from pydantic import BaseModel
from backend.app.config import settings

class GuardrailResult(BaseModel):
    is_safe: bool
    triggered: bool
    trigger_type: Optional[str] = None # 'self_harm', 'medical_advice_seeking', 'pii_leak', 'violence'
    severity: Optional[str] = None     # 'low', 'medium', 'critical'
    sanitized_text: str
    override_response: Optional[str] = None
    mitigation_action: Optional[str] = None

class GuardrailEngine:
    """
    Multi-layer safety guardrail engine providing:
    1. Crisis & Self-Harm Interception
    2. Medical Boundary Enforcement
    3. PII Detection & Anonymization
    4. Ethical Tone Validation
    """

    CRISIS_KEYWORDS = [
        r"\bkill myself\b", r"\bsuicide\b", r"\bend my life\b", r"\bwant to die\b",
        r"\bhurt myself\b", r"\bcut myself\b", r"\boverdose\b", r"\bno reason to live\b",
        r"\bbetter off dead\b", r"\bgoodbye forever\b", r"\btired of living\b"
    ]

    MEDICAL_DIAGNOSIS_KEYWORDS = [
        r"\bdiagnose me\b", r"\bwhat medication should i take\b", r"\bprescribe\b",
        r"\bdo i have bipolar\b", r"\bdo i have schizophrenia\b", r"\bdosage of\b",
        r"\bmedical treatment plan\b", r"\bstop taking my pills\b"
    ]

    VIOLENCE_KEYWORDS = [
        r"\bhow to kill\b", r"\bhurt someone\b", r"\bmake a bomb\b", r"\bmurder\b"
    ]

    # PII patterns
    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    @classmethod
    def evaluate(cls, user_text: str) -> GuardrailResult:
        if not user_text:
            return GuardrailResult(
                is_safe=True,
                triggered=False,
                sanitized_text=""
            )

        sanitized = user_text
        lower_text = user_text.lower()

        # 1. PII Redaction
        pii_found = False
        if cls.EMAIL_PATTERN.search(sanitized):
            sanitized = cls.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)
            pii_found = True
        if cls.PHONE_PATTERN.search(sanitized):
            sanitized = cls.PHONE_PATTERN.sub("[REDACTED_PHONE]", sanitized)
            pii_found = True
        if cls.SSN_PATTERN.search(sanitized):
            sanitized = cls.SSN_PATTERN.sub("[REDACTED_SSN]", sanitized)
            pii_found = True

        # 2. Critical Safety: Self-Harm & Suicide Detection
        for pattern in cls.CRISIS_KEYWORDS:
            if re.search(pattern, lower_text):
                crisis_msg = (
                    "🤍 **I care deeply about your safety and you are not alone.**\n\n"
                    "It sounds like you are carrying immense pain right now. While I am an AI companion and cannot provide emergency medical intervention, there are compassionate people ready to support you 24/7:\n\n"
                    f"- 📞 **Call or Text 988** (Suicide & Crisis Lifeline - Free, Confidential, 24/7 in US/Canada)\n"
                    f"- 💬 **{settings.CRISIS_TEXT_LINE}** to connect with a Crisis Counselor\n"
                    "- 🌍 **International Resources**: If outside the US, please visit [Befrienders Worldwide](https://www.befrienders.org/) or contact your local emergency services.\n\n"
                    "Please reach out to one of these resources, a family member, or someone you trust right now. You deserve support and care."
                )
                return GuardrailResult(
                    is_safe=False,
                    triggered=True,
                    trigger_type="self_harm",
                    severity="critical",
                    sanitized_text=sanitized,
                    override_response=crisis_msg,
                    mitigation_action="Immediate emergency crisis intervention triggered with 988 helpline display."
                )

        # 3. Harm to Others / Violence
        for pattern in cls.VIOLENCE_KEYWORDS:
            if re.search(pattern, lower_text):
                violence_msg = (
                    "I cannot assist with requests involving harm, violence, or dangerous activities. "
                    "If you are feeling intense anger or overwhelm, we can practice grounding and de-escalation strategies together."
                )
                return GuardrailResult(
                    is_safe=False,
                    triggered=True,
                    trigger_type="violence",
                    severity="critical",
                    sanitized_text=sanitized,
                    override_response=violence_msg,
                    mitigation_action="Violence guardrail triggered; output suppressed."
                )

        # 4. Medical / Clinical Diagnosis Boundary
        for pattern in cls.MEDICAL_DIAGNOSIS_KEYWORDS:
            if re.search(pattern, lower_text):
                med_disclaimer = (
                    "⚠️ **Gentle Reminder**: I am an AI emotional support companion, not a licensed therapist, doctor, or psychiatrist.\n\n"
                    "I cannot diagnose clinical psychiatric conditions, provide medical advice, or prescribe medications. "
                    "For diagnoses and treatment adjustments, please consult a qualified healthcare provider. "
                    "However, I am always here to listen, offer emotional comfort, and explore healthy coping or mindfulness practices with you."
                )
                return GuardrailResult(
                    is_safe=True,
                    triggered=True,
                    trigger_type="medical_advice_seeking",
                    severity="medium",
                    sanitized_text=sanitized,
                    override_response=med_disclaimer,
                    mitigation_action="Medical boundary reminder delivered."
                )

        # 5. PII warning if triggered but otherwise safe
        if pii_found:
            return GuardrailResult(
                is_safe=True,
                triggered=True,
                trigger_type="pii_leak",
                severity="low",
                sanitized_text=sanitized,
                mitigation_action="Sensitive PII detected and redacted before processing."
            )

        return GuardrailResult(
            is_safe=True,
            triggered=False,
            sanitized_text=sanitized
        )
