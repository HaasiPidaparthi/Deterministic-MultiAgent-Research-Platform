from langgraph.graph import StateGraph, START, END
from typing import Optional

from engine.graph.state import WorkflowState
from engine.graph.nodes import planner_node, researcher_node, synthesizer_node, verifier_node
from engine.graph.loop_controller import decide_next_step, LoopConfig
from engine.graph.retry_policy import apply_retry_policy
from engine.graph.instrumentation import instrument_node

from engine.agents.planner import PlannerAgent
from engine.agents.researcher import ResearcherAgent
from engine.agents.synthesizer import SynthesizerAgent
from engine.agents.verifier import VerifierAgent

from engine.metrics.run_metrics import init_metrics

def _route(state: WorkflowState) -> str:
    workflow_cfg = state.get("workflow", {}) or {}
    max_iters = int(workflow_cfg.get("max_iterations", 3))

    nxt = decide_next_step(state, cfg=LoopConfig(max_iters=max_iters))
    if nxt == "end":
        return END
    return nxt

def _bump_iter(state: WorkflowState) -> dict:
    return {"iter": int(state.get("iter", 0)) + 1}

def build_graph(
        planner: PlannerAgent,
        researcher: ResearcherAgent,
        synthesizer: Optional[SynthesizerAgent] = None,
        verifier: Optional[VerifierAgent] = None,
    ):
    g = StateGraph(WorkflowState)

    g.add_node("init_metrics", init_metrics)
    g.add_edge(START, "init_metrics")
    g.add_edge("init_metrics", "planner")

    g.add_node("planner", instrument_node("planner", planner_node(planner)))
    g.add_node("researcher", instrument_node("researcher", researcher_node(researcher)))

    # Support lighter flows (unit tests) without full synthesis/verification.
    if synthesizer is None or verifier is None:
        g.add_edge("planner", "researcher")
        g.add_edge("researcher", END)
        return g.compile()

    g.add_node("synthesizer", instrument_node("synthesizer", synthesizer_node(synthesizer)))
    g.add_node("verifier", instrument_node("verifier", verifier_node(verifier)))

    # nodes for retry_policy, bump iteration count, and metrics
    g.add_node("retry_policy", instrument_node("retry_policy", apply_retry_policy))
    g.add_node("bump_iter", instrument_node("bump_iter", _bump_iter))

    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "synthesizer")
    g.add_edge("synthesizer", "verifier")

    # loop path
    g.add_edge("verifier", "retry_policy")
    g.add_edge("retry_policy", "bump_iter")

    # conditional route after iter bump
    g.add_conditional_edges(
        "bump_iter", 
        _route, 
        {"researcher": "researcher", 
         "synthesizer": "synthesizer", 
         END: END,
        },
    )

    return g.compile()