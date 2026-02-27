from langgraph.graph import StateGraph, START, END

from engine.graph.state import WorkflowState
from engine.graph.nodes import planner_node, researcher_node, synthesizer_node, verifier_node

from engine.agents.planner import PlannerAgent
from engine.agents.researcher import ResearcherAgent
from engine.agents.synthesizer import SynthesizerAgent
from engine.agents.verifier import VerifierAgent

def build_graph(
        planner: PlannerAgent, 
        researcher: ResearcherAgent, 
        synthesizer: SynthesizerAgent, 
        verifier: VerifierAgent
    ):
    g = StateGraph(WorkflowState)

    g.add_node("planner", planner_node(planner))
    g.add_node("researcher", researcher_node(researcher))
    g.add_node("synthesizer", synthesizer_node(synthesizer))
    g.add_node("verifier", verifier_node(verifier))

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "synthesizer")
    g.add_edge("synthesizer", "verifier")
    g.add_edge("verifier", END)

    return g.compile()