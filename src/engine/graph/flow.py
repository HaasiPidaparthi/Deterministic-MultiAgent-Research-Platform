from langgraph.graph import StateGraph, START, END

from engine.graph.state import WorkflowState
from engine.graph.nodes import planner_node, researcher_node

from engine.agents.planner import PlannerAgent
from engine.agents.researcher import ResearcherAgent

def build_graph(planner: PlannerAgent, researcher: ResearcherAgent):
    g = StateGraph(WorkflowState)

    g.add_node("planner", planner_node(planner))
    g.add_node("researcher", researcher_node(researcher))

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", END)

    return g.compile()