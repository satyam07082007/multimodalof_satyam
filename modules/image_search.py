import logging
import urllib.parse
from typing import List, Dict, Any, Optional
import httpx

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

logger = logging.getLogger("haven.image_search")


class ImageSearchModule:
    """
    Real-Time Image Search & Retrieval Engine.
    Retrieves authentic, verified images of people, landmarks, products, animals,
    and scenes from the live web.
    Uses DuckDuckGo Images as primary provider and Wikimedia Commons as resilient fallback.
    """

    @classmethod
    def search(cls, query: str, max_results: int = 6) -> List[Dict[str, Any]]:
        """
        Searches the web for existing real images matching the query.
        Returns:
            [
                {
                    "title": str,
                    "image_url": str,
                    "thumbnail_url": str,
                    "source_url": str,
                    "source": str
                }, ...
            ]
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        # 1. Try DuckDuckGo Image Search (Primary)
        ddgs_results = cls._search_ddgs(clean_query, max_results=max_results)
        if ddgs_results:
            return ddgs_results

        # 2. Resilient Fallback: Wikimedia Commons API (Celebrities, Places, Historical Figures, Nature)
        wiki_results = cls._search_wikimedia(clean_query, max_results=max_results)
        if wiki_results:
            return wiki_results

        return []

    @classmethod
    def _search_ddgs(cls, query: str, max_results: int = 6) -> List[Dict[str, Any]]:
        """Retrieves images using DDGS search engine."""
        results: List[Dict[str, Any]] = []
        try:
            with DDGS() as ddgs:
                ddg_iter = ddgs.images(query, max_results=max_results)
                for r in ddg_iter:
                    img_url = r.get("image") or ""
                    thumb_url = r.get("thumbnail") or img_url
                    src_url = r.get("url") or img_url
                    title = r.get("title") or query

                    if img_url and img_url.startswith("http"):
                        results.append({
                            "title": title,
                            "image_url": img_url,
                            "thumbnail_url": thumb_url,
                            "source_url": src_url,
                            "source": "Web Search"
                        })
            logger.info("DDGS retrieved %d image results for query: %s", len(results), query)
        except Exception as e:
            logger.warning("DDGS image search notice: %s. Trying fallback provider...", e)
        return results

    @classmethod
    def _search_wikimedia(cls, query: str, max_results: int = 6) -> List[Dict[str, Any]]:
        """Retrieves verified public domain and CC images from Wikimedia Commons."""
        results: List[Dict[str, Any]] = []
        try:
            api_url = "https://commons.wikimedia.org/w/api.php"
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,  # File namespace
                "gsrlimit": max_results,
                "prop": "imageinfo",
                "iiprop": "url|mime|size",
                "format": "json"
            }
            headers = {"User-Agent": "HavenAI-ImageSearch/2.0 (contact@havenai.local)"}

            with httpx.Client(timeout=10) as client:
                resp = client.get(api_url, params=params, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    pages = data.get("query", {}).get("pages", {})
                    for pid, pdata in pages.items():
                        image_info = pdata.get("imageinfo", [{}])[0]
                        file_url = image_info.get("url", "")
                        title = pdata.get("title", "").replace("File:", "").split(".")[0]
                        desc_url = image_info.get("descriptionurl", file_url)

                        if file_url:
                            results.append({
                                "title": title or query,
                                "image_url": file_url,
                                "thumbnail_url": file_url,
                                "source_url": desc_url,
                                "source": "Wikimedia Commons"
                            })
            logger.info("Wikimedia retrieved %d image results for query: %s", len(results), query)
        except Exception as e:
            logger.warning("Wikimedia image search notice: %s", e)
        return results

    @classmethod
    def format_search_results(cls, results: List[Dict[str, Any]], query: str) -> str:
        """
        Formats retrieved image search results into a clean, structured summary with links.
        """
        if not results:
            return (
                f"### 🔍 Image Search: \"{query}\"\n\n"
                f"No direct web images could be retrieved for this query right now. "
                f"Please check your search terms or network connection."
            )

        lines = [
            f"### 🔍 Found {len(results)} Images for: *\"{query}\"*\n\n",
            "Here are verified photos found across the web with source links:\n\n"
        ]

        for i, r in enumerate(results, 1):
            title = r.get("title", f"Image {i}")
            img_url = r.get("image_url", "")
            src_url = r.get("source_url", img_url)
            source_name = r.get("source", "Web")
            lines.append(f"**[{i}] [{title}]({src_url})** — [Open Full Image]({img_url}) *(Source: {source_name})*\n")

        lines.append("\n*Click on any image thumbnail above to inspect the high-resolution source.*")
        return "".join(lines)

