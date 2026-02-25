from dataclasses import dataclass, field
from typing import Callable, List, Set, Any

from engine.schemas.evidence import EvidenceItem
from engine.tools.web_types import SearchResult, FetchResult
from engine.tools.extract import hash_text, reliability_score, relevance_score

# Accepts plain callable or LangChain Tool object with .invoke()
SearchFn = Any
FetchFn = Any

@dataclass
class ResearcherConfig:
    max_results_per_query: int = 5
    max_sources_total: int = 12
    min_reliability: float = 0.4

@dataclass
class ResearcherAgent:
    web_search: SearchFn
    fetch_url: FetchFn
    cfg: ResearcherConfig = field(default_factory=ResearcherConfig)

    def _call_search(self, query: str) -> List[dict]:
        """
        Supports either a plain callable web_search(query) or a LangChain Tool web_search.invoke().
        """
        if hasattr(self.web_search, "invoke"):
            # TavilySearch tool expects a dict input
            return self.web_search.invoke({"query": query, "max_results": self.cfg.max_results_per_query})
        # Plain callable (unit tests)
        return self.web_search(query)

    def _call_fetch(self, url: str) -> dict:
        """
        Supports either a plain callable fetch_url(url) or a LangChain Tool fetch_url.invoke().
        """
        if hasattr(self.fetch_url, "invoke"):
            return self.fetch_url.invoke({"url": url, "extract_depth": "basic", "format": "text"})
        return self.fetch_url(url)

    def research(self, question: str, search_queries: List[str]) -> List[EvidenceItem]:
        """
        Executes search + fetch to produce deduped EvidenceItems.
        """
        # 1) search -> candidate urls
        candidates: List[SearchResult] = []
        seen_urls: Set[str] = set()

        for q in search_queries:
            raw_result = self._call_search(q)
            for r in raw_result[: self.cfg.max_results_per_query]:
                sr = SearchResult.model_validate(r)
                if not sr.url or sr.url in seen_urls:
                    continue
                seen_urls.add(sr.url)
                candidates.append(sr)

        # 2) fetch -> content -> evidence
        evidence: List[EvidenceItem] = []
        seen_hashes: Set[str] = set()
        sid = 1

        for c in candidates:
            if len(evidence) >= self.cfg.max_sources_total:
                break
            
            fetched = self._call_fetch(c.url)
            fr = FetchResult.model_validate(fetched)
            if fr.status_code >= 400 or not fr.text or not fr.text.strip():
                continue

            content_h = hash_text(fr.text)
            if content_h in seen_hashes:
                continue
            seen_hashes.add(content_h)

            rel = reliability_score(fr.url)
            if rel < self.cfg.min_reliability:
                continue

            rev = relevance_score(question, fr.text, fr.title or c.title)
            snippet = (c.snippet or fr.text[:280]).strip().replace("\n", " ")

            item = EvidenceItem(
                id=f"S{sid}",
                url=fr.url,
                title=fr.title or c.title,
                publisher=fr.publisher or c.source,
                snippet=snippet[:400],
                reliability_score=rel,
                relevance_score=rev,
                content_hash=content_h,
            )
            sid += 1
            evidence.append(item)

        # 3) sort best-first
        evidence.sort(
            key=lambda e: (e.reliability_score * 0.6 + e.relevance_score * 0.4), 
            reverse=True,
        )
        return evidence