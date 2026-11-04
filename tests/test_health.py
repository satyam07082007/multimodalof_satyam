import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check_endpoint():
    """Verifies that GET /health returns 200 OK and expected diagnostic metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "database_engine" in data
    assert "rag_indexed_chunks" in data
    assert "app_name" in data

