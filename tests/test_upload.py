import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

client = TestClient(app)

def test_reject_unsupported_file_extension():
    """Verifies that malicious or unsupported file extensions (.exe) are rejected with 400 Bad Request."""
    fake_exe = io.BytesIO(b"MZ\x90\x00\x03\x00\x00\x00")
    files = {"file": ("malware.exe", fake_exe, "application/octet-stream")}
    data = {"message": "Please review this"}

    response = client.post("/api/chat/multimodal", data=data, files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_reject_corrupted_image():
    """Verifies that corrupted image bytes fail image validation."""
    fake_img = io.BytesIO(b"corrupted binary random non-image bytes")
    files = {"file": ("corrupt.png", fake_img, "image/png")}
    data = {"message": "Here is my drawing"}

    response = client.post("/api/chat/multimodal", data=data, files=files)
    assert response.status_code == 400
    assert "Invalid or corrupted image" in response.json()["detail"]

def test_valid_image_multimodal_chat():
    """Verifies that a valid PNG image is accepted and processed through multimodal pipeline."""
    img_byte_arr = io.BytesIO()
    image = Image.new("RGB", (60, 60), color="teal")
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    files = {"file": ("sketch.png", img_byte_arr, "image/png")}
    data = {
        "conversation_id": "test_img_conv",
        "message": "I made this calming drawing today."
    }

    response = client.post("/api/chat/multimodal", data=data, files=files)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["input_type"] == "image"
    assert len(res_data["response"]) > 0

    # Clean up
    client.delete("/api/conversations/test_img_conv")

def test_valid_document_attachment_chat():
    """Verifies that a text document attachment is processed as conversational context."""
    doc_bytes = io.BytesIO(b"Reflective Journal Entry:\nToday was challenging but I took three deep breaths.")
    files = {"file": ("my_thoughts.txt", doc_bytes, "text/plain")}
    data = {
        "conversation_id": "test_doc_conv",
        "message": "Here are my reflections from this afternoon."
    }

    response = client.post("/api/chat/multimodal", data=data, files=files)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["input_type"] == "document"

    # Clean up
    client.delete("/api/conversations/test_doc_conv")

