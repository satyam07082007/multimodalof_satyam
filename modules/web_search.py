import os
import logging
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from duckduckgo_search import DDGS  # pip package: duckduckgo-search (alias for ddgs)

logger = logging.getLogger("haven.web_search")


class WebSearchModule:
    """
    Real-Time Web Research & Search Engine.
    Queries live DuckDuckGo web index and uses Gemini AI to synthesize
    a coherent, cited research report from the retrieved snippets.
    """

    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY") or ""
        self.model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")

        if self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
            except Exception as e:
                logger.warning("Gemini config in WebSearch: %s", e)

    @classmethod
    def search(cls, query: str, max_results: int = 6) -> List[Dict[str, Any]]:
        """
        Executes a live DuckDuckGo web search.
        Returns: [{"title": ..., "href": ..., "body": ...}, ...]
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        results: List[Dict[str, Any]] = []

        try:
            with DDGS() as ddgs:
                for r in ddgs.text(clean_query, max_results=max_results):
                    results.append({
                        "title": r.get("title", "Untitled"),
                        "href": r.get("href", ""),
                        "body": r.get("body", "")
                    })
            logger.info("Retrieved %d search results for: %s", len(results), clean_query)
        except Exception as e:
            logger.warning("DuckDuckGo search notice: %s", e)

        return results

    def synthesize_with_gemini(self, query: str, results: List[Dict[str, Any]]) -> Optional[str]:
        """
        Uses Gemini to generate a smart research synthesis from raw search snippets.
        """
        if not self.gemini_key or not results:
            return None

        # Build context block from search snippets
        context = "\n\n".join(
            f"Source [{i}]: {r.get('title', '')}\nURL: {r.get('href', '')}\nContent: {r.get('body', '')}"
            for i, r in enumerate(results, 1)
        )

        prompt = (
            "You are a research analyst. Using ONLY the web search results below, write a comprehensive, "
            "well-structured research report answering the user's query.\n\n"
            "Rules:\n"
            "- Cite sources using [1], [2], etc. notation.\n"
            "- Use markdown headings, bullet points, and bold for key terms.\n"
            "- Do NOT invent facts not present in the sources.\n"
            "- End with a 'Sources' section listing the URLs.\n\n"
            f"**User Query**: {query}\n\n"
            f"**Web Search Results**:\n{context}\n\n"
            "**Research Report**:"
        )

        try:
            model = genai.GenerativeModel(self.model_name)
            res = model.generate_content(prompt)
            if res and res.text:
                return res.text.strip()
        except Exception as e:
            logger.error("Gemini web synthesis failed: %s", e)
        return None

    @classmethod
    def format_search_context(cls, results: List[Dict[str, Any]]) -> str:
        """Formats raw results into a readable context block."""
        if not results:
            return ""
        lines = ["### 🌐 Live Web Search Results:\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"**[{i}] {r.get('title', '')}**\n🔗 {r.get('href', '')}\n{r.get('body', '')}\n")
        return "\n".join(lines)

    def synthesize_research_report(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Creates a full research report — Gemini AI synthesis if available,
        otherwise formats raw citations cleanly.
        """
        if not results:
            return (
                f"### 🌐 Web Research: \"{query}\"\n\n"
                "Live web search could not retrieve results at this moment.\n"
                "Please check your internet connection and try again."
            )

        # Try Gemini-powered synthesis first
        gemini_report = self.synthesize_with_gemini(query, results)
        if gemini_report:
            # Append source links at the bottom
            sources = "\n".join(
                f"- [{r.get('title', f'Source {i}')}]({r.get('href', '')})"
                for i, r in enumerate(results, 1)
                if r.get("href")
            )
            return (
                f"### 🌐 AI-Synthesized Web Research: \"{query}\"\n\n"
                f"{gemini_report}\n\n"
                f"---\n**🔗 Sources Retrieved**:\n{sources}"
            )

        # Fallback: clean citation listing
        citations = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Source")
            url = r.get("href", "")
            snippet = r.get("body", "")[:300]
            citations.append(f"**[{i}] [{title}]({url})**\n> {snippet}")

        return (
            f"### 🌐 Web Research Results: \"{query}\"\n\n"
            + "\n\n".join(citations)
            + "\n\n*Retrieved from live public web index via DuckDuckGo.*"
        )
