import pytest
from app.services.rag import RAGService

def test_rag_chunking():
    """Verifies that text is cleanly chunked into overlapping windows."""
    long_text = "Mindfulness is the practice of purposely focusing your attention on the present moment. " * 30
    chunks = RAGService.chunk_text(long_text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 250 for c in chunks)

def test_rag_retrieval_and_source_attribution():
    """Verifies that semantic search returns matching wellness chunks with scores."""
    rag = RAGService()
    results = rag.query("How do I practice 5-4-3-2-1 grounding?", top_k=2)
    assert len(results) > 0
    top_result = results[0]
    assert "document_name" in top_result
    assert "chunk_reference" in top_result
    assert top_result["relevance_score"] > 0.05
    assert "grounding" in top_result["content"].lower() or "5-4-3-2-1" in top_result["content"]

def test_rag_irrelevant_query_empty():
    """Verifies that completely irrelevant queries yield no low-confidence results."""
    rag = RAGService()
    # High threshold query for unrelated topic
    results = rag.query("quantum mechanics astrophysics black holes", top_k=2, threshold=0.7)
    assert len(results) == 0

def test_rag_prompt_context_security():
    """Verifies that formatted prompt context isolates untrusted data with security rules."""
    rag = RAGService()
    chunks = [{
        "document_name": "test.txt",
        "chunk_reference": "Section 1",
        "content": "Deep breathing calms the vagus nerve."
    }]
    formatted = rag.format_prompt_context(chunks)
    assert "UNTRUSTED REFERENCE DATA" in formatted
    assert "SECURITY RULE" in formatted
    assert "Deep breathing" in formatted

