from dataclasses import dataclass, field
from typing import List, Set, Any

from engine.schemas.evidence import EvidenceItem
from engine.tools.web_types import SearchResult, FetchResult
from engine.tools.extract import hash_text, reliability_score, relevance_score_embed
from engine.events.emitter import Emitter

# Accepts plain callable or LangChain Tool object with .invoke()
SearchFn = Any
FetchFn = Any

@dataclass
class ResearcherConfig:
    max_results_per_query: int = 3
    max_sources_total: int = 5
    min_reliability: float = 0.4
    embedding_model: str = "nomic-embed-text"
    min_relevance: float = 0.20

@dataclass
class ResearcherAgent:
    web_search: SearchFn
    fetch_url: FetchFn
    cfg: ResearcherConfig = field(default_factory=ResearcherConfig)

    def _call_search(self, query: str, emitter=None) -> List[dict]:
        """
        Supports either a plain callable web_search(query) or a LangChain Tool web_search.invoke().
        """
        if emitter:
            emitter and emitter.emit("ToolCallRequested", agent="researcher", tool="web_search", query=query, max_results=self.cfg.max_results_per_query)
        try:
            if hasattr(self.web_search, "invoke"):
                # TavilySearch tool expects a dict input
                out = self.web_search.invoke({"query": query, "max_results": self.cfg.max_results_per_query})
            else:
                # Plain callable (unit tests)
                out = self.web_search(query)
            if emitter:
                emitter and emitter.emit("ToolCallCompleted", agent="researcher", tool="web_search", query=query, results_count=len(out) if isinstance(out, list) else None)
            return out
        except Exception as e:
            if emitter:
                emitter and emitter.emit("ToolCallFailed", agent="researcher", tool="web_search", query=query, error=str(e))
            raise

    def _call_fetch(self, url: str, emitter=None) -> dict:
        """
        Supports either a plain callable fetch_url(url) or a LangChain Tool fetch_url.invoke().
        """
        if emitter:
            emitter and emitter.emit("ToolCallRequested", agent="researcher", tool="fetch_url", url=url)
        try:
            if hasattr(self.fetch_url, "invoke"):
                out = self.fetch_url.invoke({"url": url, "extract_depth": "basic", "format": "text"})
            else:
                out = self.fetch_url(url)
            if emitter:
                emitter and emitter.emit("ToolCallCompleted", agent="researcher", tool="fetch_url", url=url, status_code=out.get("status_code"))
            return out
        except Exception as e:
            if emitter:
                emitter and emitter.emit("ToolCallFailed", agent="researcher", tool="fetch_url", url=url, error=str(e))
            raise
    
    def _reject(self, emitter, url: str, reason: str, **data):
        if emitter:
            emitter.emit(
                "EvidenceItemRejected",
                agent="researcher",
                url=url,
                reason=reason,
                **data,
            )


    def research(self, question: str, search_queries: List[str], emitter=None) -> List[EvidenceItem]:
        """
        Executes search + fetch to produce deduped EvidenceItems.
        Applies reliability + embedding-based relevance scoring.
        """
        if emitter:
            emitter and emitter.emit("AgentStarted", agent="researcher", question=question, queries_count=len(search_queries))

        # Use query context for relevance embedding
        query_context = (question or "").strip()
        if search_queries:
            query_context = query_context + "\n\nSearch queries:\n" + "\n".join(search_queries)

        # 1) search -> candidate urls
        candidates: List[SearchResult] = []
        seen_urls: Set[str] = set()
        rejected = 0

        for q in search_queries:
            raw_result = self._call_search(q, emitter=emitter)
            for r in raw_result[: self.cfg.max_results_per_query]:
                sr = SearchResult.model_validate(r)
                if not sr.url:
                    self._reject(emitter, "", reason="missing_url", query=q)
                    rejected += 1
                    continue
                if sr.url in seen_urls:
                    self._reject(emitter, sr.url, reason="duplicate_url", query=q)
                    rejected += 1
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
            
            fetched = self._call_fetch(c.url, emitter=emitter)
            fr = FetchResult.model_validate(fetched)

            if fr.status_code >= 400:
                self._reject(emitter, c.url, reason="fetch_http_error", status_code=fr.status_code) 
                rejected += 1
                continue
            
            if not fr.text or not fr.text.strip():
                self._reject(emitter, c.url, reason="fetch_empty_text", status_code=fr.status_code)
                rejected += 1
                continue

            content_h = hash_text(fr.text)
            if content_h in seen_hashes:
                self._reject(emitter, c.url, reason="duplicate_content", content_hash=content_h)
                rejected += 1
                continue
            seen_hashes.add(content_h)

            rel = reliability_score(fr.url)
            if rel < self.cfg.min_reliability:
                self._reject(emitter, c.url, reason="low_reliability", reliability=rel, min_reliability=self.cfg.min_reliability)
                rejected += 1
                continue

            rev = relevance_score_embed(
                question=query_context,
                text=fr.text,
                title=fr.title or c.title,
                model=self.cfg.embedding_model,
            )
            if rev < self.cfg.min_relevance:
                self._reject(emitter, c.url, reason="low_relevance", relevance=rev, min_relevance=self.cfg.min_relevance)
                rejected += 1
                continue
            
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

            emitter and emitter.emit(
                "EvidenceItemCreated",
                agent="researcher",
                evidence_id=item.id,
                url=item.url,
                reliability=item.reliability_score,
                relevance=item.relevance_score,
            )

        # 3) sort best-first
        evidence.sort(
            key=lambda e: (e.reliability_score * 0.6 + e.relevance_score * 0.4), 
            reverse=True,
        )

        emitter and emitter.emit(
            "ResearchCompleted",
            agent="researcher",
            candidates_count=len(candidates),
            evidence_count=len(evidence),
            rejected_count=rejected,
        )
        emitter and emitter.emit("AgentFinished", agent="researcher")

        return evidence