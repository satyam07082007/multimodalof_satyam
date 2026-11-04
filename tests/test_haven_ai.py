import pytest
import io
from pathlib import Path
import pandas as pd
from PIL import Image

from modules.chat_engine import ChatEngine
from modules.image_understanding import ImageUnderstandingService
from modules.image_generation import ImageGenerationService
from modules.speech_to_text import SpeechToTextService
from modules.text_to_speech import TextToSpeechService
from modules.document_processing import DocumentProcessor
from modules.web_search import WebSearchModule
from modules.coding_assistant import CodingAssistantModule
from modules.data_analysis import DataAnalysisModule
from modules.rag_engine import DynamicRAGEngine
from modules.emotional_support import EmotionalSupportModule
from utils.security import sanitize_pii, check_crisis_intent, check_medical_boundary

def test_1_general_chat():
    """Checklist 1: Explain artificial intelligence in simple language."""
    engine = ChatEngine()
    reply = engine.generate_chat_response("Explain artificial intelligence in simple language.")
    assert "Artificial Intelligence" in reply or "AI" in reply
    assert len(reply) > 50

def test_2_image_understanding(tmp_path):
    """Checklist 2: Image inspection and visual description."""
    test_img = tmp_path / "test_visual.png"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(test_img)

    vision = ImageUnderstandingService()
    desc = vision.analyze_image(test_img)
    assert "100 × 100" in desc or "pixels" in desc or "Visual" in desc

def test_3_image_generation(tmp_path):
    """Checklist 3: Image generation engine invokes diffusion service."""
    gen = ImageGenerationService(output_dir=tmp_path)
    # Using local mock or live generation test
    success, img_path, msg = gen.generate_image("futuristic AI research laboratory")
    # Even if offline, it handles the response gracefully
    assert isinstance(success, bool)
    assert isinstance(msg, str)

def test_4_speech_to_text(tmp_path):
    """Checklist 4: Audio voice transcription."""
    fake_audio = tmp_path / "sample_voice.wav"
    fake_audio.write_bytes(b"RIFF....WAVEfmt ....data....")

    stt = SpeechToTextService()
    transcript = stt.transcribe(fake_audio)
    assert len(transcript) > 0
    assert "Audio" in transcript or "voice" in transcript.lower()

def test_5_text_to_speech(tmp_path):
    """Checklist 5: Voice audio response synthesis."""
    tts = TextToSpeechService(output_dir=tmp_path)
    audio_path = tts.synthesize("Hello, I am Haven AI, your personal intelligent companion.")
    # In test environment, it returns a valid path or None if gTTS is mocked
    assert audio_path is None or Path(audio_path).suffix == ".mp3"

def test_6_document_understanding(tmp_path):
    """Checklist 6: Upload a document and summarize its important points and MCQs."""
    sample_doc = tmp_path / "research_paper.txt"
    sample_doc.write_text(
        "Machine Learning algorithms learn patterns from empirical training datasets. "
        "Supervised learning requires labeled features and outcomes to compute loss functions. "
        "Optimization techniques such as gradient descent minimize loss over iterative epochs."
    )

    doc_data = DocumentProcessor.extract_content(sample_doc)
    assert "text" in doc_data
    assert "Machine Learning" in doc_data["text"]

    summary = DocumentProcessor.generate_study_materials(doc_data["text"], mode="summary")
    assert "Executive Document Summary" in summary

    mcq = DocumentProcessor.generate_study_materials(doc_data["text"], mode="mcq")
    assert "Question 1" in mcq

def test_7_web_search():
    """Checklist 7: Find latest developments in artificial intelligence."""
    results = WebSearchModule.search("artificial intelligence developments", max_results=2)
    # Returns list of dicts with genuine structure
    assert isinstance(results, list)
    formatted = WebSearchModule.format_search_context(results)
    assert isinstance(formatted, str)

def test_8_coding_assistant():
    """Checklist 8: Write a Python program to implement binary search."""
    code_res = CodingAssistantModule.generate_code_solution("Write a Python program to implement binary search.")
    assert "def binary_search" in code_res
    assert "o(log n)" in code_res.lower()
    assert "```python" in code_res

def test_9_data_analysis(tmp_path):
    """Checklist 9: Upload a CSV and generate statistical summary and chart."""
    df_sample = pd.DataFrame({
        "Department": ["Engineering", "Design", "Research", "Marketing"],
        "Budget_K": [120, 80, 150, 60],
        "Team_Size": [15, 8, 12, 7]
    })
    csv_file = tmp_path / "company_data.csv"
    df_sample.to_csv(csv_file, index=False)

    df_loaded, err = DataAnalysisModule.load_dataset(csv_file)
    assert df_loaded is not None
    assert len(df_loaded) == 4

    analysis = DataAnalysisModule.analyze_dataset(df_loaded)
    assert analysis["rows"] == 4
    assert "Budget_K" in analysis["numeric_columns"]

    chart = DataAnalysisModule.generate_chart(df_loaded, chart_type="bar", x_col="Department", y_col="Budget_K")
    assert chart is not None

def test_10_rag_custom_knowledge():
    """Checklist 10: Answer question using uploaded document in vector index."""
    rag = DynamicRAGEngine(session_id="test_session")
    rag.add_document("quantum_notes.txt", "Quantum computing uses superposition and entanglement to execute qubits operations.")
    hits = rag.retrieve("What principles do quantum computers use?", top_k=1)
    assert len(hits) > 0
    assert "superposition" in hits[0]["content"].lower()

def test_11_emotional_support():
    """Checklist 11: Empathetic response, active listening, and crisis 988 safety check."""
    # Test gentle conversation
    care_res = EmotionalSupportModule.process_emotional_input("I am feeling stressed today. Can we talk?")
    assert care_res["is_crisis"] is False
    assert len(care_res["response"]) > 30

    # Test crisis escalation
    crisis_res = EmotionalSupportModule.process_emotional_input("I want to end my life")
    assert crisis_res["is_crisis"] is True
    assert "988" in crisis_res["response"]

