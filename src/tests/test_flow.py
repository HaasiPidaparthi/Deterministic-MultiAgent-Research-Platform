from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from engine.agents.planner import PlannerAgent
from engine.agents.researcher import ResearcherAgent, ResearcherConfig
from engine.graph.flow import build_graph

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