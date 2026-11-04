import re
from typing import Optional, Tuple
from pydantic import BaseModel
from app.config import settings

class SafetyEvaluation(BaseModel):
    is_safe: bool
    triggered: bool
    trigger_type: Optional[str] = None  # 'self_harm', 'medical_boundary', 'violence', 'prompt_injection', 'pii_redacted'
    severity: str = "none"              # 'none', 'low', 'medium', 'critical'
    sanitized_text: str
    override_response: Optional[str] = None
    mitigation_action: Optional[str] = None

class SafetyEngine:
    """
    Dedicated AI Safety and Emotional Guardrail Engine.
    Executes independent rule-based validation before and after LLM orchestration.
    """

    CRISIS_PATTERNS = [
        r"\bkill myself\b", r"\bsuicide\b", r"\bend my life\b", r"\bwant to die\b",
        r"\bhurt myself\b", r"\bcut myself\b", r"\boverdose\b", r"\bno reason to live\b",
        r"\bbetter off dead\b", r"\bgoodbye forever\b", r"\btired of living\b",
        r"\bcommit suicide\b", r"\bend it all\b"
    ]

    MEDICAL_PATTERNS = [
        r"\bdiagnose me\b", r"\bwhat medication should i take\b", r"\bprescribe\b",
        r"\bdo i have bipolar\b", r"\bdo i have schizophrenia\b", r"\bdosage of\b",
        r"\bmedical treatment plan\b", r"\bstop taking my pills\b", r"\bclinical diagnosis\b",
        r"\bam i clinically depressed\b"
    ]

    VIOLENCE_PATTERNS = [
        r"\bhow to kill\b", r"\bhurt someone\b", r"\bmake a bomb\b", r"\bmurder\b",
        r"\battack someone\b", r"\bharm others\b"
    ]

    PROMPT_INJECTION_PATTERNS = [
        r"ignore (all )?previous instructions",
        r"disregard system instructions",
        r"system override",
        r"bypass safety guidelines",
        r"reveal system prompt",
        r"you are now in developer mode",
        r"jailbreak"
    ]

    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    @classmethod
    def evaluate(cls, user_text: str) -> SafetyEvaluation:
        """
        Evaluates input text against safety, crisis, medical boundary, and prompt-injection policies.
        """
        if not user_text:
            return SafetyEvaluation(
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

        # 2. Critical Safety: Imminent Danger & Self-Harm Interception
        for pattern in cls.CRISIS_PATTERNS:
            if re.search(pattern, lower_text):
                crisis_msg = (
                    "🤍 **I care deeply about your safety, and you do not have to carry this alone.**\n\n"
                    "It sounds like you are carrying immense pain right now. While I am an AI companion and cannot provide emergency medical intervention, there are compassionate people ready to support you 24/7:\n\n"
                    f"- 📞 **Call or Text {settings.CRISIS_HELPLINE_PHONE}** (Suicide & Crisis Lifeline — Free, Confidential, 24/7 in US/Canada)\n"
                    f"- 💬 **{settings.CRISIS_TEXT_LINE}** to connect with a Crisis Counselor\n"
                    "- 🌍 **International Support**: If outside North America, visit [Befrienders Worldwide](https://www.befrienders.org/) or contact your local emergency services.\n\n"
                    "Please reach out to one of these services, a family member, or a professional right now. Your safety and life matter."
                )
                return SafetyEvaluation(
                    is_safe=False,
                    triggered=True,
                    trigger_type="self_harm",
                    severity="critical",
                    sanitized_text=sanitized,
                    override_response=crisis_msg,
                    mitigation_action="Immediate emergency crisis intervention triggered with 988 helpline display."
                )

        # 3. Harm to Others / Violence Prevention
        for pattern in cls.VIOLENCE_PATTERNS:
            if re.search(pattern, lower_text):
                violence_msg = (
                    "I cannot assist with requests involving harm, violence, or dangerous activities. "
                    "If you are feeling overwhelmed, angry, or distressed, we can explore healthy grounding and de-escalation practices together."
                )
                return SafetyEvaluation(
                    is_safe=False,
                    triggered=True,
                    trigger_type="violence",
                    severity="critical",
                    sanitized_text=sanitized,
                    override_response=violence_msg,
                    mitigation_action="Violence guardrail triggered; output suppressed."
                )

        # 4. Prompt Injection Defense
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, lower_text):
                injection_msg = (
                    "I am programmed to operate as a safe and empathetic emotional companion. "
                    "I cannot alter system guidelines or disregard safety protocols. How can I support your well-being today?"
                )
                return SafetyEvaluation(
                    is_safe=False,
                    triggered=True,
                    trigger_type="prompt_injection",
                    severity="medium",
                    sanitized_text=sanitized,
                    override_response=injection_msg,
                    mitigation_action="Prompt injection pattern detected and neutralized."
                )

        # 5. Medical & Clinical Boundary Enforcement
        for pattern in cls.MEDICAL_PATTERNS:
            if re.search(pattern, lower_text):
                med_disclaimer = (
                    "⚠️ **Important Boundary**: I am an AI emotional support assistant, not a licensed therapist, doctor, or psychiatrist.\n\n"
                    "I cannot diagnose clinical mental health conditions, prescribe medication, or formulate medical treatment plans. "
                    "For clinical diagnoses and medication guidance, please consult a qualified healthcare provider. "
                    "I am always here to listen, offer emotional comfort, and practice evidence-based grounding techniques with you."
                )
                return SafetyEvaluation(
                    is_safe=True,
                    triggered=True,
                    trigger_type="medical_boundary",
                    severity="medium",
                    sanitized_text=sanitized,
                    override_response=med_disclaimer,
                    mitigation_action="Medical boundary reminder delivered."
                )

        # 6. PII Notice if detected
        if pii_found:
            return SafetyEvaluation(
                is_safe=True,
                triggered=True,
                trigger_type="pii_redacted",
                severity="low",
                sanitized_text=sanitized,
                mitigation_action="Sensitive PII was detected and redacted before processing."
            )

        return SafetyEvaluation(
            is_safe=True,
            triggered=False,
            severity="none",
            sanitized_text=sanitized
        )

