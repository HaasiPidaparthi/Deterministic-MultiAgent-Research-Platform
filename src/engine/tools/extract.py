import hashlib
from urllib.parse import urlparse
from typing import Optional

from engine.tools.web_types import FetchResult

HIGH_RELIABILITY_DOMAINS = {
    "sec.gov",
    "bls.gov",
    "census.gov",
    "oecd.org",
    "worldbank.org",
    "who.int",
    "europa.eu",
}

MEDIUM_RELIABILITY_HINTS = {
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "economist.com",
    "hbr.org",
    "mckinsey.com",
    "gartner.com",
    "forrester.com",
    "pwc.com",
    "deloitte.com",
    "kpmg.com",
}

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def domain(url: str) -> str:
    return urlparse(url).netloc.lower().replace("www.", "")

def reliability_score(url: str) -> float:
    d = domain(url)
    if d in HIGH_RELIABILITY_DOMAINS:
        return 0.9
    if any(h in d for h in MEDIUM_RELIABILITY_HINTS):
        return 0.75
    if d.endswith(".gov") or d.endswith(".edu"):
        return 0.85
    return 0.5

def relevance_score(question: str, text: str, title: Optional[str]) -> float:
    # MVP heuristic: keyword overlap
    q_tokens = {t.lower() for t in question.split() if len(t) > 3}
    hay = (title or "") + " " + (text[:2000] if text else "")
    h_tokens = {t.lower().strip(".,:;()[]{}") for t in hay.split() if len(t) > 3}
    if not q_tokens:
        return 0.5
    overlap = len(q_tokens.intersection(h_tokens))
    return max(0.1, min(1.0, overlap / max(4, len(q_tokens))))