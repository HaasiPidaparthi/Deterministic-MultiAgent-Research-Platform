from typing import Optional, Dict, Any
from langchain_core.tools import tool
from langchain_tavily import TavilyExtract

def _guess_title_from_text(text: str) -> Optional[str]:
    # MVP: first non-empty line as "title"
    for line in (text or "").splitlines():
        s = line.strip()
        if s:
            return s[:120]
    return None

def _error_response(url: str, status_code: int = 500) -> Dict[str, Any]:
    """Return a standard error response dict."""
    return {
        "url": url,
        "status_code": status_code,
        "title": None,
        "publisher": None,
        "text": "",
    }

@tool
def fetch_url(
    url: str,
    extract_depth: str = "basic",   # "basic" or "advanced"
    format: str = "text",           # "markdown" or "text"
    include_images: bool = False,
    include_favicon: bool = False,
) -> Dict[str, Any]:
    """
    Fetch/extract content for a single URL using Tavily Extract.

    Returns a dict compatible with engine.tools.web_types.FetchResult:
    {url, status_code, title, publisher, text}
    """
    extractor = TavilyExtract(
        extract_depth=extract_depth,
        include_images=include_images,
        include_favicon=include_favicon,
        format=format,
    )

    raw = extractor.invoke({"urls": [url]})
    
    # Handle case where Tavily returns an error string instead of dict
    if isinstance(raw, str):
        return _error_response(url)
    
    results = raw.get("results", []) or []
    failed = raw.get("failed_results", []) or []

    if url in failed or not results:
        return _error_response(url)

    r0 = results[0]
    text = (r0.get("raw_content") or "").strip()

    return {
        "url": r0.get("url") or url,
        "status_code": 200 if text else 502,
        "title": _guess_title_from_text(text),
        "publisher": None,
        "text": text,
    }