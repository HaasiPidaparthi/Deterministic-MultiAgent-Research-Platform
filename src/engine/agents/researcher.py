"""
Researcher Agent Module

This module contains the ResearcherAgent class which is responsible for gathering
evidence from web sources. The researcher performs web searches and fetches content
from relevant URLs, then scores the results for reliability and relevance to the
research question.
"""

from dataclasses import dataclass, field
from typing import List, Set, Any, Optional
import concurrent.futures

from engine.schemas.evidence import EvidenceItem
from engine.tools.web_types import SearchResult, FetchResult
from engine.tools.extract import hash_text, reliability_score, relevance_score_embed
from engine.tools.rag import RAGRetriever, RAGConfig
from engine.events.emitter import Emitter
from engine.metrics.run_metrics import inc_tool, inc_reject

# Accepts plain callable or LangChain Tool object with .invoke()
SearchFn = Any
FetchFn = Any

@dataclass
class ResearcherConfig:
    """
    Configuration for the ResearcherAgent.

    Attributes:
        max_results_per_query: Maximum search results to process per query
        max_sources_total: Maximum total evidence items to collect
        min_reliability: Minimum reliability score required for evidence
        embedding_model: Model name for relevance embeddings
        min_relevance: Minimum relevance score required for evidence
        evidence_quality_threshold: Threshold for triggering refetch of low-quality evidence
        enable_rag: Whether to use RAG (vector DB search) before web search
        rag_config: Configuration for RAG functionality
    """
    max_results_per_query: int = 3
    max_sources_total: int = 5
    min_reliability: float = 0.4
    embedding_model: str = "nomic-embed-text"
    min_relevance: float = 0.20
    evidence_quality_threshold: float = 0.6  # same threshold used by verifier for consistency
    enable_rag: bool = True
    rag_config: RAGConfig = field(default_factory=RAGConfig)

@dataclass
class ResearcherAgent:
    """
    Agent responsible for gathering and scoring web evidence.

    The ResearcherAgent performs web searches using the provided search function,
    fetches content from promising URLs, and scores each piece of evidence for
    reliability and relevance. It can operate in normal research mode or refetch
    mode for specific URLs. Includes RAG functionality for internal knowledge base search.
    """
    web_search: SearchFn
    fetch_url: FetchFn
    cfg: ResearcherConfig = field(default_factory=ResearcherConfig)
    rag_retriever: Optional[RAGRetriever] = None

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
    
    def _reject(self, emitter, url: str, reason: str, metrics=None, **data):
        if metrics is not None:
            inc_reject(metrics, reason, 1)
        if emitter:
            emitter.emit(
                "EvidenceItemRejected",
                agent="researcher",
                url=url,
                reason=reason,
                **data,
            )

    def _parallel_fetch_urls(self, urls: List[str], emitter=None, metrics=None):
        """Fetch multiple URLs in parallel and preserve the original order."""
        results = [None] * len(urls)
        future_to_index = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for index, url in enumerate(urls):
                if emitter:
                    emitter.emit("ToolCallRequested", agent="researcher", tool="fetch_url", url=url)
                future = executor.submit(self._call_fetch, url, emitter=None)
                future_to_index[future] = index

            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                url = urls[index]
                try:
                    fetched = future.result()
                    if emitter:
                        status_code = fetched.get("status_code") if isinstance(fetched, dict) else None
                        emitter.emit("ToolCallCompleted", agent="researcher", tool="fetch_url", url=url, status_code=status_code)
                    if metrics is not None:
                        inc_tool(metrics, "fetch_url", 1)
                    results[index] = (url, fetched, None)
                except Exception as exc:
                    if emitter:
                        emitter.emit("ToolCallFailed", agent="researcher", tool="fetch_url", url=url, error=str(exc))
                    if metrics is not None:
                        inc_tool(metrics, "fetch_url", 1)
                    results[index] = (url, None, exc)

        return results


    def _init_rag_if_needed(self):
        """Initialize RAG retriever if enabled and not already initialized."""
        if self.cfg.enable_rag and self.rag_retriever is None:
            self.rag_retriever = RAGRetriever(self.cfg.rag_config)

    def research(self, question: str, search_queries: List[str], emitter=None, metrics=None, refetch_urls: Optional[List[str]] = None) -> List[EvidenceItem]:
        """
        Executes search + fetch to produce deduped EvidenceItems.
        Applies reliability + embedding-based relevance scoring.
        If refetch_urls provided, skips search and fetches only those URLs.
        """
        if emitter:
            count = len(refetch_urls) if refetch_urls else len(search_queries)
            emitter.emit("AgentStarted", agent="researcher", question=question, queries_count=count)

        if refetch_urls:
            # Refetch mode: fetch specific URLs
            query_context = (question or "").strip()

            evidence = []
            seen_hashes = set()
            sid = 1

            fetched_results = self._parallel_fetch_urls(refetch_urls, emitter=emitter, metrics=metrics)

            for url, fetched, exc in fetched_results:
                if exc is not None:
                    self._reject(emitter, url, reason="fetch_exception", error=str(exc))
                    continue
                try:
                    fr = FetchResult.model_validate(fetched)
                except Exception as validation_error:
                    self._reject(
                        emitter,
                        url,
                        reason="fetch_validation_error",
                        error=str(validation_error),
                    )
                    continue

                if fr.status_code >= 400:
                    self._reject(emitter, url, reason="fetch_http_error", status_code=fr.status_code)
                    continue

                if not fr.text or not fr.text.strip():
                    self._reject(emitter, url, reason="fetch_empty_text", status_code=fr.status_code)
                    continue

                content_h = hash_text(fr.text)
                if content_h in seen_hashes:
                    self._reject(emitter, url, reason="duplicate_content", content_hash=content_h)
                    continue
                seen_hashes.add(content_h)

                rel = reliability_score(fr.url)
                if rel < self.cfg.min_reliability:
                    self._reject(emitter, url, reason="low_reliability", reliability=rel, min_reliability=self.cfg.min_reliability)
                    continue

                rev = relevance_score_embed(
                    question=query_context,
                    text=fr.text,
                    title=fr.title or "",
                    model=self.cfg.embedding_model,
                )
                if rev < self.cfg.min_relevance:
                    self._reject(emitter, url, reason="low_relevance", relevance=rev, min_relevance=self.cfg.min_relevance)
                    continue

                snippet = (fr.text[:280]).strip().replace("\n", " ")

                item = EvidenceItem(
                    id=f"S{sid}",
                    url=fr.url,
                    title=fr.title or "",
                    publisher=fr.publisher or "",
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

            evidence.sort(
                key=lambda e: (e.reliability_score * 0.6 + e.relevance_score * 0.4),
                reverse=True,
            )

            emitter and emitter.emit(
                "ResearchCompleted",
                agent="researcher",
                candidates_count=len(refetch_urls),
                evidence_count=len(evidence),
                rejected_count=len(refetch_urls) - len(evidence),
            )
            return evidence

        # Normal research mode
        # Use query context for relevance embedding
        query_context = (question or "").strip()
        if search_queries:
            query_context = query_context + "\n\nSearch queries:\n" + "\n".join(search_queries)

        # Initialize RAG if needed
        self._init_rag_if_needed()

        # 0) Try RAG search first if enabled
        rag_evidence = []
        if self.cfg.enable_rag and self.rag_retriever:
            if emitter:
                emitter.emit("RAGSearchStarted", agent="researcher", question=question, queries_count=len(search_queries))

            # Search RAG for each query
            for q in search_queries:
                try:
                    rag_results = self.rag_retriever.search_and_convert_to_evidence(
                        query=q,
                        question_context=query_context,
                        min_relevance=self.cfg.min_relevance
                    )
                    rag_evidence.extend(rag_results)

                    if emitter:
                        emitter.emit("RAGSearchCompleted", agent="researcher", query=q, results_count=len(rag_results))
                except Exception as e:
                    if emitter:
                        emitter.emit("RAGSearchFailed", agent="researcher", query=q, error=str(e))

            # Remove duplicates from RAG results
            seen_hashes = set()
            unique_rag_evidence = []
            for item in rag_evidence:
                if item.content_hash not in seen_hashes:
                    seen_hashes.add(item.content_hash)
                    unique_rag_evidence.append(item)

            # Enforce reliability threshold on RAG results as well
            rag_evidence = [
                item for item in unique_rag_evidence
                if item.reliability_score >= self.cfg.min_reliability
            ]
            rag_evidence = rag_evidence[: self.cfg.max_sources_total]  # Limit RAG results

        # 1) search -> candidate urls
        candidates: List[SearchResult] = []
        seen_urls: Set[str] = set()
        rejected = 0

        for q in search_queries:
            raw_result = self._call_search(q, emitter=emitter)
            if metrics is not None:
                inc_tool(metrics, "web_search", 1)
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

        fetched_results = self._parallel_fetch_urls([c.url for c in candidates], emitter=emitter, metrics=metrics)

        for i, c in enumerate(candidates):
            if len(evidence) >= self.cfg.max_sources_total:
                break

            url, fetched, exc = fetched_results[i]
            if exc is not None:
                self._reject(emitter, url, reason="fetch_exception", error=str(exc))
                rejected += 1
                continue

            try:
                fr = FetchResult.model_validate(fetched)
            except Exception as validation_error:
                self._reject(
                    emitter,
                    url,
                    reason="fetch_validation_error",
                    error=str(validation_error),
                )
                rejected += 1
                continue

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

        # 3) Combine RAG and web evidence, remove duplicates, sort best-first
        all_evidence = rag_evidence + evidence

        # Remove duplicates based on content hash
        seen_hashes = set()
        unique_evidence = []
        for item in all_evidence:
            if item.content_hash not in seen_hashes:
                seen_hashes.add(item.content_hash)
                unique_evidence.append(item)

        # Sort by quality score and limit to max_sources_total
        unique_evidence.sort(
            key=lambda e: (e.reliability_score * 0.6 + e.relevance_score * 0.4),
            reverse=True,
        )
        final_evidence = unique_evidence[:self.cfg.max_sources_total]

        if emitter:
            for item in final_evidence:
                emitter.emit(
                    "EvidenceItemCreated",
                    agent="researcher",
                    evidence_id=item.id,
                    url=item.url,
                    reliability=item.reliability_score,
                    relevance=item.relevance_score,
                )

        emitter and emitter.emit(
            "ResearchCompleted",
            agent="researcher",
            candidates_count=len(candidates),
            rag_evidence_count=len(rag_evidence),
            web_evidence_count=len(evidence),
            final_evidence_count=len(final_evidence),
            evidence_count=len(final_evidence),
            rejected_count=rejected,
        )
        emitter and emitter.emit("AgentFinished", agent="researcher")

        return final_evidence