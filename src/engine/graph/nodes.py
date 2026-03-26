from typing import Any, Dict
from langchain_core.runnables import RunnableConfig

from engine.agents.planner import PlannerAgent
from engine.agents.researcher import ResearcherAgent
from engine.agents.synthesizer import SynthesizerAgent
from engine.agents.verifier import VerifierAgent
from engine.graph.state import WorkflowState

def planner_node(agent: PlannerAgent):
    """
    Returns a LangGraph-compatible planner node function.
    """
    def _node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
        emitter = config.get("configurable", {}).get("emitter")
        metrics = state.get("metrics")
        plan = agent.plan(
            question=state["question"], 
            budget_usd=float(state.get("budget_usd", 2.5)), 
            time_limit_s=int(state.get("time_limit_s", 0)),
            emitter=emitter,
            metrics=metrics,
        )
        return {"plan": plan}
    return _node

def researcher_node(agent: ResearcherAgent):
    """
    Returns a LangGraph-compatible researcher node function.
    """
    def _node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
        emitter = config.get("configurable", {}).get("emitter")
        overrides = state.get("researcher_overrides") or {}
        # Apply overrides temporarily
        old = agent.cfg.model_dump() if hasattr(agent.cfg, "model_dump") else agent.cfg.__dict__.copy()
        try:
            for k, v in overrides.items():
                if hasattr(agent.cfg, k):
                    setattr(agent.cfg, k, v)

            refetch_urls = state.get("refetch_urls")
            if refetch_urls:
                # Refetch mode: fetch additional URLs
                additional_evidence = agent.research(
                    question=state["question"], 
                    search_queries=[],
                    emitter=emitter,
                    refetch_urls=refetch_urls,
                )

                existing = state.get("evidence", []) or []
                existing_by_url = {e.url: e for e in existing}

                # Replace existing evidence entries with new fetch results when available.
                # If refetch URL has no new valid evidence, keep the existing entry.
                for ref_e in additional_evidence:
                    existing_by_url[ref_e.url] = ref_e

                merged_evidence = list(existing_by_url.values())
                return {"evidence": merged_evidence, "refetch_urls": []}  # clear refetch_urls
            else:
                # Normal research mode
                plan = state["plan"]
                search_queries = getattr(plan, "search_queries", []) if not isinstance(plan, dict) else plan.get("search_queries", [])
                evidence = agent.research(
                    question=state["question"], 
                    search_queries=search_queries,
                    emitter=emitter,
                )
                return {"evidence": evidence}
        finally:
            # Restore original config
            for k, v in old.items():
                if hasattr(agent.cfg, k):
                    setattr(agent.cfg, k, v)
    return _node

def synthesizer_node(agent: SynthesizerAgent):
    def _node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
        emitter = config.get("configurable", {}).get("emitter")
        metrics = state.get("metrics")
        mode = state.get("synthesizer_mode", "normal")

        brief = agent.synthesize(
            question=state["question"],
            plan=state["plan"],
            evidence=state["evidence"],
            emitter=emitter,
            mode=mode,
            metrics=metrics,
        )
        return {"brief": brief}
    return _node

def verifier_node(agent: VerifierAgent):
    def _node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
        emitter = config.get("configurable", {}).get("emitter")

        report = agent.verify(
            plan=state["plan"],
            evidence=state["evidence"],
            brief=state["brief"],
            emitter=emitter,
        )

        updates = {"report": report}

        # Check for retry: if verification failed and low overall scores, refetch low-scoring URLs
        if not report.passed and state.get("evidence"):
            avg_rel = sum(e.reliability_score for e in state["evidence"]) / len(state["evidence"])
            avg_rev = sum(e.relevance_score for e in state["evidence"]) / len(state["evidence"])
            threshold = getattr(agent.cfg, "evidence_quality_threshold", 0.6)
            if avg_rel < threshold or avg_rev < threshold:
                low_urls = list(set(e.url for e in state["evidence"] if e.reliability_score < threshold or e.relevance_score < threshold))
                updates["refetch_urls"] = low_urls

        return updates
    return _node