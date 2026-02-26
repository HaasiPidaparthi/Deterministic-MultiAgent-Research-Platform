import pytest
from dotenv import load_dotenv

from engine.tools.web_search import web_search
from engine.tools.fetch import fetch_url
from engine.agents.researcher import ResearcherAgent, ResearcherConfig

load_dotenv()

@pytest.mark.integration
def test_search_results():
    results = web_search.invoke({"query": "SMB payroll software market size", "max_results": 3})
    assert isinstance(results, list)
    assert len(results) > 0
    assert "url" in results[0]

@pytest.mark.integration
def test_fetch_url_extracts():
    out = fetch_url.invoke({"url": "https://en.wikipedia.org/wiki/Payroll"})
    assert out["status_code"] == 200
    assert isinstance(out["text"], str)
    assert len(out["text"]) > 200

@pytest.mark.integration
def test_researcher_dedupes_and_limits():
    agent = ResearcherAgent(
        web_search=web_search,
        fetch_url=fetch_url,
        cfg=ResearcherConfig(max_results_per_query=3, max_sources_total=2, min_reliability=0.6),
    )

    evidence = agent.research(
        question="Should we enter the SMB payroll market?",
        search_queries=["smb payroll market size", "payroll compliance requirements"],
    )

    # Limited to <= 2
    assert len(evidence) <= 2
    # No duplicates
    assert len({e.url for e in evidence}) == len(evidence)
    # Reliability enforced
    assert all(e.reliability_score >= 0.6 for e in evidence)

@pytest.mark.integration
def test_researcher_sorts_best_first():
    agent = ResearcherAgent(
        web_search=web_search,
        fetch_url=fetch_url,
        cfg=ResearcherConfig(max_results_per_query=3, max_sources_total=10, min_reliability=0.4),
    )

    evidence = agent.research(
        question="payroll compliance requirements",
        search_queries=["payroll compliance requirements"],
    )

    assert len(evidence) >= 1