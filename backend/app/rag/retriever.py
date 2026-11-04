import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.app.config import settings

logger = logging.getLogger("emotional_support.rag")

class WellnessRAGRetriever:
    """
    RAG Retriever indexing mental health evidence-based exercises,
    grounding techniques, and emotional coping resources.
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or settings.DATA_PATH
        self.documents: List[Dict[str, Any]] = []
        self.corpus: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self):
        """Loads wellness knowledge items and builds index."""
        try:
            if not self.data_path.exists():
                logger.warning("Knowledge base file not found at %s. Initializing empty RAG index.", self.data_path)
                return

            with open(self.data_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)

            self.corpus = []
            for doc in self.documents:
                # Combine title, topic, content, and keywords for rich retrieval representation
                keywords_str = " ".join(doc.get("keywords", []))
                text_blob = f"{doc.get('title', '')} {doc.get('topic', '')} {doc.get('content', '')} {keywords_str}"
                self.corpus.append(text_blob)

            if self.corpus:
                self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
                logger.info("Successfully indexed %d mental wellness items into RAG engine.", len(self.documents))
        except Exception as e:
            logger.error("Error during RAG initialization: %s", e)

    def retrieve(self, query: str, top_k: int = 2, threshold: float = 0.08) -> List[Dict[str, Any]]:
        """
        Retrieves top relevant mental wellness guides for the given emotional query.
        """
        if not self.documents or self.vectorizer is None or not query.strip():
            return []

        try:
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            ranked_indices = scores.argsort()[::-1]

            results = []
            for idx in ranked_indices[:top_k]:
                score = float(scores[idx])
                if score >= threshold:
                    matched_doc = dict(self.documents[idx])
                    matched_doc["relevance_score"] = round(score, 3)
                    results.append(matched_doc)

            return results
        except Exception as e:
            logger.error("Error retrieving RAG documents: %s", e)
            return []

    def format_context_for_prompt(self, docs: List[Dict[str, Any]]) -> str:
        """Formats retrieved documents into a context block for the LLM prompt."""
        if not docs:
            return ""

        context_parts = ["### Relevant Psychological & Coping Evidence-Based Frameworks:"]
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"[{i}] {doc.get('title')} ({doc.get('topic')}):\n{doc.get('content')}"
            )
        return "\n\n".join(context_parts)
