import pytest
from unittest.mock import MagicMock

@pytest.fixture(autouse=True)
def mock_external_calls(monkeypatch):
    """
    Automatically mocks gTTS and external network calls during pytest execution
    so tests run offline, deterministic, and instantly.
    """
    # Mock SpeechService.synthesize_speech
    from app.services.speech import SpeechService
    monkeypatch.setattr(SpeechService, "synthesize_speech", lambda text: "test_audio.mp3")

    # Mock MultimodalService vision & audio external calls
    from app.services.multimodal import MultimodalService
    monkeypatch.setattr(MultimodalService, "analyze_image", lambda path: "Calming visual artwork with soothing blue tones.")
    monkeypatch.setattr(MultimodalService, "transcribe_audio", lambda path: "I have been feeling overwhelmed by my daily tasks.")

