#!/usr/bin/env python
"""
Standalone Knowledge Base Ingestion Script.
Reads all PDF and TXT files from knowledge_base/sample_documents,
generates chunks and vector indices, and records them in MySQL and the vector index.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path so app modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.rag import RAGService
from app.database import SessionLocal, init_db
from app.models.document import Document

def run_ingestion(rebuild: bool = True):
    print("=" * 60)
    print("🌿 Haven Knowledge Base Ingestion Tool")
    print("=" * 60)

    # Initialize DB tables if not present
    init_db()
    db = SessionLocal()

    rag = RAGService()
    docs_dir = settings.KNOWLEDGE_BASE_PATH
    print(f"[*] Scanning knowledge directory: {docs_dir}")

    files = list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.pdf"))
    if not files:
        print("[!] No .txt or .pdf files found in knowledge directory.")
        return

    print(f"[*] Found {len(files)} document(s) to process:")
    for f in files:
        print(f"    - {f.name} ({f.stat().st_size} bytes)")

    result = rag.ingest_documents(files)

    # Record documents in relational database
    try:
        for f in files:
            doc_id = f"doc_{f.stem}"
            existing = db.query(Document).filter(Document.document_id == doc_id).first()
            if not existing:
                doc_record = Document(
                    document_id=doc_id,
                    filename=f.name,
                    document_type=f.suffix.lstrip(".").lower(),
                    processing_status="completed"
                )
                db.add(doc_record)
        db.commit()
        print("[*] Successfully synchronized document metadata with application database.")
    except Exception as e:
        print(f"[!] Database metadata sync warning: {e}")
        db.rollback()
    finally:
        db.close()

    print("=" * 60)
    print(f"[✓] Ingestion Complete!")
    print(f"    Total Documents: {result['documents_indexed']}")
    print(f"    Total Chunks Indexed: {result['chunks_indexed']}")
    print(f"    Vector Index Saved: {settings.VECTOR_STORE_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest knowledge base documents into RAG vector index.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild vector store.")
    args = parser.parse_args()

    run_ingestion(rebuild=args.rebuild)

