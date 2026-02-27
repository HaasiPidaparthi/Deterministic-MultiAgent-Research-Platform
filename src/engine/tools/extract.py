import math
import hashlib
from urllib.parse import urlparse
from typing import Optional, List, Dict

from engine.tools.web_types import FetchResult

try:
    import ollama
except Exception:
    ollama = None  # embeddings optional

_EMBED_CACHE = {}

DEFAULT_DOMAIN_WEIGHTS: Dict[str, float] = {
    ".gov": 0.95,
    ".edu": 0.90,
    ".org": 0.75,
}

HIGH_TRUST_DOMAINS = {
    "sec.gov",
    "bls.gov",
    "census.gov",
    "oecd.org",
    "worldbank.org",
    "who.int",
    "europa.eu",
}

MED_TRUST_DOMAINS = {
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

LOW_TRUST_HINTS = [
    "medium.com",
    "substack.com",
    "blogspot.",
    "wordpress.",
]


# --- HELPER FUNCTIONS ---
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def _hostname(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# --- RELIABILITY SCORE ---
def reliability_score(url: str, publisher: Optional[str] = None) -> float:
    host = _hostname(url).lower()
    if not host:
        return 0.2

    # Hard boosts
    if any(host.endswith(d) for d in HIGH_TRUST_DOMAINS):
        base = 0.90
    elif any(host.endswith(d) for d in MED_TRUST_DOMAINS):
        base = 0.75
    elif any(h in host for h in LOW_TRUST_HINTS):
        base = 0.45
    else:
        base = 0.50  # neutral default

    # TLD heuristic
    for tld, w in DEFAULT_DOMAIN_WEIGHTS.items():
        if host.endswith(tld):
            base = max(base, w)

    # HTTPS boost (tiny)
    if url.startswith("https://"):
        base += 0.02

    # Publisher hint (tiny)
    if publisher:
        p = publisher.lower()
        if "government" in p or "department" in p:
            base += 0.03

    return float(max(0.0, min(1.0, base)))


# --- RELEVANCE SCORE USING HEURISTICS AND EMBEDDINGS ---
def relevance_score(question: str, text: str, title: Optional[str]) -> float:
    # heuristic: keyword overlap
    q_tokens = {t.lower() for t in question.split() if len(t) > 3}
    hay = (title or "") + " " + (text[:2000] if text else "")
    h_tokens = {t.lower().strip(".,:;()[]{}") for t in hay.split() if len(t) > 3}
    if not q_tokens:
        return 0.5
    overlap = len(q_tokens.intersection(h_tokens))
    return max(0.1, min(1.0, overlap / max(4, len(q_tokens))))

def _cache_key(text: str, model: str) -> str:
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return f"{model}:{h}"

def _embed_ollama(text: str, model: str = "nomic-embed-text") -> List[float]:
    if ollama is None:
        return []
    text = (text or "").strip()
    if not text:
        return []
    key = _cache_key(text, model)
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]
    resp = ollama.embeddings(model=model, prompt=text)
    vec = resp.get("embedding", []) or []
    _EMBED_CACHE[key] = vec
    return vec

def relevance_score_embed(
    question: str,
    text: str,
    title: Optional[str],
    model: str = "nomic-embed-text",
) -> float:
    """
    Embedding-based relevance in [0, 1]. Falls back to heuristic if embeddings unavailable.
    """
    q = (question or "").strip()
    if not q:
        return 0.5

    doc = ((title or "") + "\n" + (text[:4000] if text else "")).strip()
    if not doc:
        return 0.0

    qv = _embed_ollama(q, model=model)
    dv = _embed_ollama(doc, model=model)
    if not qv or not dv:
        # fallback to heuristic
        return relevance_score(question, text, title)

    sim = _cosine(qv, dv)          # [-1, 1] in practice
    sim01 = 0.5 * (sim + 1.0)      # map to [0,1]
    return max(0.0, min(1.0, sim01))


# --- CONFIDENCE SCORE FOR CLAIMS ---
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def claim_confidence_embed(
    claim_text: str,
    cited_texts: List[str],
    cited_reliabilities: List[float],
    model: str = "nomic-embed-text",
) -> float:
    """
    Compute confidence for a claim using embeddings.

    Logic:
      - similarity = max cosine(claim, evidence_i) mapped to [0,1]
      - rel = average reliability
      - confidence = 0.65*similarity + 0.35*rel + bonus(log(#citations))
    """
    claim_text = (claim_text or "").strip()
    if not claim_text:
        return 0.2
    if not cited_texts:
        return 0.2

    # Embed claim once
    c_vec = _embed_ollama(claim_text, model=model)
    if not c_vec:
        # fallback: weaker score if embeddings unavailable
        avg_rel = sum(cited_reliabilities) / max(1, len(cited_reliabilities))
        return _clamp01(0.35 * avg_rel + 0.15)

    sims01: List[float] = []
    for t in cited_texts:
        t = (t or "").strip()
        if not t:
            continue
        e_vec = _embed_ollama(t, model=model)
        if not e_vec:
            continue
        sim = _cosine(c_vec, e_vec)       # [-1,1]
        sim01 = 0.5 * (sim + 1.0)         # -> [0,1]
        sims01.append(_clamp01(sim01))

    if not sims01:
        avg_rel = sum(cited_reliabilities) / max(1, len(cited_reliabilities))
        return _clamp01(0.35 * avg_rel + 0.15)

    best_sim = max(sims01)

    # avg reliability (guard mismatch lengths)
    if cited_reliabilities:
        avg_rel = sum(cited_reliabilities) / len(cited_reliabilities)
    else:
        avg_rel = 0.5

    # diminishing returns bonus for multiple citations
    n = min(len(cited_texts), len(cited_reliabilities) or len(cited_texts))
    bonus = 0.06 * math.log(1 + n)

    conf = 0.65 * best_sim + 0.35 * avg_rel + bonus
    return _clamp01(conf)
