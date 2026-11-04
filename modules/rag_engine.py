import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.services.rag import RAGService

logger = logging.getLogger("haven.rag_engine")

class DynamicRAGEngine:
    """
    User Knowledge Base RAG Engine.
    Allows on-the-fly document ingestion and conversational semantic retrieval per session.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    def add_document(self, filename: str, content: str) -> int:
        """
        Chunks and indexes user-uploaded document for this session's RAG memory.
        """
        doc_chunks = RAGService.chunk_text(content, chunk_size=500, overlap=100)
        for i, chunk in enumerate(doc_chunks):
            self.chunks.append({
                "chunk_id": f"{self.session_id}_{filename}_{i+1}",
                "document_name": filename,
                "chunk_reference": f"Section {i+1}",
                "content": chunk,
                "excerpt": chunk[:200] + ("..." if len(chunk) > 200 else "")
            })

        self._build_index()
        logger.info("Session RAG indexed %d chunks for document: %s", len(doc_chunks), filename)
        return len(doc_chunks)

    def _build_index(self):
        """Builds TF-IDF semantic vector space for retrieved chunks."""
        if not self.chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        corpus = [c["content"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.08) -> List[Dict[str, Any]]:
        """
        Retrieves top relevant chunks from session's custom knowledge base.
        """
        if not self.chunks or self.vectorizer is None or not query.strip():
            return []

        try:
            q_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
            ranked = scores.argsort()[::-1]

            results = []
            for idx in ranked[:top_k]:
                score = float(scores[idx])
                if score >= threshold:
                    matched = dict(self.chunks[idx])
                    matched["relevance_score"] = round(score, 3)
                    results.append(matched)

            return results
        except Exception as e:
            logger.error("Session RAG search error: %s", e)
            return []

