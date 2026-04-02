from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END

from engine.agents.planner import PlannerAgent
from engine.agents.researcher import ResearcherAgent, ResearcherConfig
from engine.graph.flow_loop import build_graph
from engine.graph.nodes import researcher_node
from engine.schemas.evidence import EvidenceItem


PLAN_JSON = """
{
  "subquestions": ["A","B","C"],
  "search_queries": ["q1","q2","q3","q4","q5"],
  "stop_criteria": {"min_sources": 5, "min_claim_coverage": 0.8},
  "assumptions": [],
  "risks_to_check": []
}
""".strip()

def fake_web_search(query: str):
    return [
        {"url": "https://sec.gov/example", "title": "SEC", "snippet": "Primary", "source": "sec"},
        {"url": "https://reuters.com/article", "title": "Reuters", "snippet": "News", "source": "reuters"},
    ]

def fake_fetch_url(url: str):
    return {
        "url": url,
        "status_code": 200,
        "title": "Title",
        "publisher": "Pub",
        "text": "payroll compliance requirements " * 50,
    }

def test_planner_researcher_flow_unit():
    llm = FakeMessagesListChatModel(responses=[AIMessage(content=PLAN_JSON)])
    planner = PlannerAgent(llm=llm)

    researcher = ResearcherAgent(
        web_search=fake_web_search,
        fetch_url=fake_fetch_url,
        cfg=ResearcherConfig(max_results_per_query=5, max_sources_total=5, min_reliability=0.0),
    )

    app = build_graph(planner, researcher)

    out = app.invoke({"question": "test question", "budget_usd": 1.0, "time_limit_s": 60})
    assert "plan" in out
    assert "evidence" in out
    assert len(out["evidence"]) >= 1


def test_loop_route_respects_max_iterations():
    from engine.graph.flow_loop import _route

    state = {
        "iter": 5,
        "workflow": {"max_iterations": 5},
        "metrics": {"elapsed_s": 0.0, "cost_usd": 0.0},
        "report": None,
    }
    next_step = _route(state)
    assert next_step == END
    assert state["stop_reason"] == "max_iters"


def test_researcher_node_refetch_keeps_low_existing_evidence():
    existing_evidence = [
        EvidenceItem(
            id="S1",
            url="https://example.com/low",
            title="Low",
            snippet="Low quality",
            reliability_score=0.4,
            relevance_score=0.3,
            content_hash="abc",
        )
    ]

    def fake_fetch_low(url: str):
        return {
            "url": url,
            "status_code": 200,
            "title": "Low",
            "publisher": "Pub",
            "text": "low relevance content",
        }

    researcher = ResearcherAgent(
        web_search=fake_web_search,
        fetch_url=fake_fetch_low,
        cfg=ResearcherConfig(max_results_per_query=5, max_sources_total=5, min_reliability=0.6, min_relevance=0.6),
    )

    node = researcher_node(researcher)

    out = node(
        {
            "question": "test question",
            "evidence": existing_evidence,
            "refetch_urls": ["https://example.com/low"],
        },
        {"configurable": {"emitter": None}},
    )

    assert any(e.url == "https://example.com/low" for e in out["evidence"])
