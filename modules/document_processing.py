import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from pypdf import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
import google.generativeai as genai

logger = logging.getLogger("haven.document_processing")


class DocumentProcessor:
    """
    Multimodal Document Intelligence Engine.
    Extracts text from PDF, DOCX, TXT, CSV, XLSX, PPTX.
    Uses Gemini AI to generate smart summaries, study notes, MCQs, and Q&A pairs.
    Falls back to rule-based extraction when AI is unavailable.
    """

    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY") or ""
        self.model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")

        if self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
            except Exception as e:
                logger.warning("Gemini config in DocumentProcessor: %s", e)

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Calls Gemini with a document-focused prompt."""
        if not self.gemini_key:
            return None
        try:
            model = genai.GenerativeModel(self.model_name)
            res = model.generate_content(prompt)
            if res and res.text:
                return res.text.strip()
        except Exception as e:
            logger.error("Gemini document AI call failed: %s", e)
        return None

    def extract_content(self, file_path: Path, max_pages: int = 50) -> Dict[str, Any]:
        """
        Extracts structured text and metadata from the document based on file extension.
        Supports: PDF, DOCX, TXT, CSV, XLSX, PPTX.
        """
        if not file_path.exists():
            return {"error": "File does not exist."}

        ext = file_path.suffix.lower()
        extracted_text = ""
        metadata: Dict[str, Any] = {
            "filename": file_path.name,
            "format": ext,
            "size_kb": round(file_path.stat().st_size / 1024, 1)
        }

        try:
            if ext == ".txt":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    extracted_text = f.read()

            elif ext == ".pdf":
                reader = PdfReader(str(file_path))
                metadata["page_count"] = len(reader.pages)
                pages_text = []
                for i, page in enumerate(reader.pages[:max_pages]):
                    p_text = page.extract_text() or ""
                    if p_text.strip():
                        pages_text.append(f"--- [Page {i+1}] ---\n{p_text}")
                extracted_text = "\n\n".join(pages_text)

            elif ext == ".docx":
                doc = DocxDocument(str(file_path))
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                metadata["paragraph_count"] = len(paragraphs)
                extracted_text = "\n\n".join(paragraphs)

            elif ext == ".pptx":
                prs = Presentation(str(file_path))
                metadata["slide_count"] = len(prs.slides)
                slides_text = []
                for i, slide in enumerate(prs.slides):
                    slide_content = [
                        shape.text.strip()
                        for shape in slide.shapes
                        if hasattr(shape, "text") and shape.text.strip()
                    ]
                    if slide_content:
                        slides_text.append(f"--- [Slide {i+1}] ---\n" + "\n".join(slide_content))
                extracted_text = "\n\n".join(slides_text)

            elif ext in [".csv", ".xlsx"]:
                df = pd.read_csv(file_path, nrows=500) if ext == ".csv" else pd.read_excel(file_path, nrows=500)
                metadata["rows"] = len(df)
                metadata["columns"] = list(df.columns)
                extracted_text = (
                    f"Spreadsheet Dataset Summary:\n"
                    f"Columns: {', '.join(df.columns)}\n"
                    f"Rows Sampled: {len(df)}\n\n"
                    f"First 10 Rows:\n{df.head(10).to_string()}\n\n"
                    f"Statistical Summary:\n{df.describe().to_string()}"
                )

            metadata["char_count"] = len(extracted_text)
            metadata["word_count"] = len(extracted_text.split())

            return {"text": extracted_text, "metadata": metadata}

        except Exception as e:
            logger.error("Document extraction error for %s: %s", file_path, e)
            return {"error": f"Failed to parse {ext} file: {str(e)}"}

    def generate_study_materials(self, document_text: str, mode: str = "summary", user_query: str = "") -> str:
        """
        Uses Gemini AI to transform document text into smart study materials.
        Modes: summary | notes | mcq | qa
        Falls back to rule-based generation when AI unavailable.
        """
        clean_text = document_text.strip()
        if not clean_text:
            return "No document text available to process."

        # Truncate to ~6000 chars for Gemini context efficiency
        context_text = clean_text[:6000] + ("..." if len(clean_text) > 6000 else "")

        # --- Gemini AI Generation ---
        if mode == "summary":
            prompt = (
                "You are an expert research analyst. Summarize the following document clearly and concisely.\n\n"
                "Requirements:\n"
                "- Use markdown with ## headings, bullet points, and bold for key terms.\n"
                "- Identify: Main Topic, Key Arguments/Findings, Important Data/Facts, Conclusions.\n"
                "- Length: 300-500 words.\n\n"
                f"**Document Text**:\n{context_text}\n\n"
                "**Executive Summary**:"
                + (f"\n\nAlso specifically address: {user_query}" if user_query else "")
            )
        elif mode == "notes":
            prompt = (
                "Generate structured study notes from the following document.\n\n"
                "Requirements:\n"
                "- Organize under clear ## headings.\n"
                "- Use bullet points for key facts and definitions.\n"
                "- Highlight important terms in **bold**.\n"
                "- Keep notes concise and exam-ready.\n\n"
                f"**Document Text**:\n{context_text}\n\n"
                "**Study Notes**:"
            )
        elif mode == "mcq":
            prompt = (
                "Generate 5 high-quality multiple-choice questions (MCQs) based on this document.\n\n"
                "Format each question exactly as:\n"
                "**Q1**: [Question text]\n"
                "- A) [Option]\n"
                "- B) [Option]\n"
                "- C) [Option]\n"
                "- D) [Option]\n"
                "✅ **Correct Answer**: [Letter] — [Brief explanation]\n\n"
                f"**Document Text**:\n{context_text}\n\n"
                "**MCQ Questions**:"
            )
        elif mode == "qa":
            prompt = (
                "Generate 5 insightful Q&A pairs from this document that test deep understanding.\n\n"
                "Format:\n"
                "**Q**: [Question]\n"
                "**A**: [Detailed answer based on the document]\n\n"
                f"**Document Text**:\n{context_text}\n\n"
                + (f"Focus especially on answering: {user_query}\n\n" if user_query else "")
                + "**Q&A Pairs**:"
            )
        else:
            prompt = ""

        if prompt:
            ai_result = self._call_gemini(prompt)
            if ai_result:
                mode_labels = {
                    "summary": "📑 AI Document Summary",
                    "notes": "📝 AI Study Notes",
                    "mcq": "🎯 AI-Generated MCQs",
                    "qa": "❓ AI Q&A Pairs"
                }
                return f"### {mode_labels.get(mode, '📄 Document Analysis')}\n\n{ai_result}"

        # --- Offline Fallback ---
        lines = [line.strip() for line in clean_text.split("\n") if line.strip() and len(line.strip()) > 30]

        if mode == "notes":
            notes = [f"- **Key Point {i+1}**: {line[:180]}" for i, line in enumerate(lines[:8])]
            return (
                "### 📝 Structured Study Notes\n\n"
                "**Core Themes Identified**:\n" + "\n".join(notes) +
                "\n\n*Connect your Gemini API key for AI-powered smart notes.*"
            )

        if mode == "mcq":
            return (
                "### 🎯 Multiple-Choice Questions\n\n"
                "**Q1**: What is the primary subject of this document?\n"
                "- A) Implementation and background\n- B) Historical context\n- C) Future predictions\n- D) None\n"
                "✅ **Correct**: A\n\n"
                "**Q2**: Which approach is recommended for best results?\n"
                "- A) Random experimentation\n- B) Systematic structured application\n- C) Skipping review\n- D) Partial engagement\n"
                "✅ **Correct**: B\n\n"
                "*Connect your Gemini API key to generate real MCQs from your document content.*"
            )

        if mode == "qa":
            return (
                "### ❓ Question & Answer Pairs\n\n"
                f"**Q1: What is the core topic of this document?**\n"
                f"> **A**: {lines[0][:200] if lines else 'The document covers the subject outlined in its introduction.'}\n\n"
                "**Q2: What are the key conclusions?**\n"
                f"> **A**: The material presents structured evidence with clear methodology and practical recommendations.\n\n"
                "*Connect your Gemini API key for real AI-generated Q&A from your document.*"
            )

        # Default summary fallback
        word_count = len(clean_text.split())
        return (
            f"### 📑 Document Overview\n\n"
            f"**Length**: ~{word_count} words\n\n"
            f"#### 1. Opening Content\n"
            f"{clean_text[:600]}...\n\n"
            f"#### 2. Key Themes\n"
            f"- Explores core theoretical and operational aspects.\n"
            f"- Presents structured principles with actionable recommendations.\n\n"
            f"*Connect your Gemini API key for a full AI executive summary.*"
        )
