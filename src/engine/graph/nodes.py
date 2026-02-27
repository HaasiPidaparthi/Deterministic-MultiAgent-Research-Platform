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
        plan = agent.plan(
            question=state["question"], 
            budget_usd=float(state.get("budget_usd", 2.5)), 
            time_limit_s=int(state.get("time_limit_s", 0)),
            emitter=emitter,
        )
        return {"plan": plan}
    return _node

def researcher_node(agent: ResearcherAgent):
    """
    Returns a LangGraph-compatible researcher node function.
    """
    def _node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
        emitter = config.get("configurable", {}).get("emitter")
        plan = state["plan"]
        evidence = agent.research(
            question=state["question"], 
            search_queries=plan.search_queries,
            emitter=emitter,
        )
        return {"evidence": evidence}
    return _node

def synthesizer_node(agent: SynthesizerAgent):
    def _node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
        emitter = config.get("configurable", {}).get("emitter")
        brief = agent.synthesize(
            question=state["question"],
            plan=state["plan"],
            evidence=state["evidence"],
            emitter=emitter,
        )
        print("SYNTH IN keys:", list(state.keys()))
        out = {"brief": brief}
        print("SYNTH OUT keys:", list(out.keys()))
        return out
    return _node

def verifier_node(agent: VerifierAgent):
    def _node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
        emitter = config.get("configurable", {}).get("emitter")

        print("VERIFIER IN keys:", list(state.keys()))
        report = agent.verify(
            plan=state["plan"],
            evidence=state["evidence"],
            brief=state["brief"],
            emitter=emitter,
        )
        return {"report": report}
    return _node