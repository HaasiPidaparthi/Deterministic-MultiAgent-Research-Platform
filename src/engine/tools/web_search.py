from typing import List, Optional
from langchain_core.tools import tool
from urllib.parse import urlparse

from engine.tools.web_types import SearchResult, FetchResult
from langchain_tavily import TavilySearch

def _domain(url: str) -> str:
    return (urlparse(url).netloc or "").lower().replace("www.", "")

def _normalize_tavily_results(raw: dict) -> List[dict]:
    """
    Convert Tavily response into SearchResult-compatible dicts.
    Tavily results - url, title, content, score, etc.
    """
    out: List[dict] = []
    for r in raw.get("results", []) or []:
        out.append(
            {
                "url": r.get("url"),
                "title": r.get("title"),
                "snippet": (r.get("content") or r.get("snippet") or "")[:400],
                "source": "tavily",
            }
        )
    return out

@tool
def web_search(
    query: str,
    max_results: int = 8,
    search_depth: str = "advanced",
    allow_domains: Optional[List[str]] = None,
    include_answer: bool = False,
    include_raw_content: bool = False,
) -> List[dict]:
    """
    Search the web using Tavily Search API and return a list of SearchResult dicts.
    search_depth: one of {"basic","advanced","fast"}
    """
    tool_impl = TavilySearch(
        max_results=max_results,
        search_depth=search_depth,
        include_answer=include_answer,
        include_raw_content=include_raw_content,
    )
    raw = tool_impl.invoke({"query": query})
    results = _normalize_tavily_results(raw)

    if allow_domains:
        allow = {d.lower().replace("www.", "") for d in allow_domains}
        results = [r for r in results if _domain(r["url"]) in allow]

    return results