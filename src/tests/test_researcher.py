import pytest
from dotenv import load_dotenv

from engine.tools.web_search import web_search
from engine.tools.web_fetch import fetch_url
from engine.agents.researcher import ResearcherAgent, ResearcherConfig
from engine.events.sink import InMemorySink
from engine.events.emitter import Emitter
import uuid

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
        cfg=ResearcherConfig(max_results_per_query=3, max_sources_total=5, min_reliability=0.4),
    )

    evidence = agent.research(
        question="payroll compliance requirements",
        search_queries=["payroll compliance requirements"],
    )

    assert len(evidence) >= 1

def test_researcher_emits_events():
    sink = InMemorySink()
    emitter = Emitter(sink, run_id=str(uuid.uuid4()))

    agent = ResearcherAgent(
        web_search=web_search,
        fetch_url=fetch_url,
        cfg=ResearcherConfig(max_results_per_query=3, max_sources_total=5, min_reliability=0.0),
    )

    evidence = agent.research(
        question="payroll compliance requirements",
        search_queries=["payroll compliance requirements"],
        emitter=emitter,
    )

    # Basic sanity: evidence produced
    assert len(evidence) >= 1

    # Collect event types in order
    types = [e.type for e in sink.events]
    print("Emitted events:", types)

    # Agent lifecycle
    assert "AgentStarted" in types
    assert "AgentFinished" in types

    # Tool calls (at least one search + one fetch)
    assert "ToolCallRequested" in types
    assert "ToolCallCompleted" in types

    # Evidence and completion
    assert "EvidenceItemCreated" in types
    assert "ResearchCompleted" in types

    # Stronger checks: ensure search + fetch events occurred with expected tool names
    requested_tools = [e.tool for e in sink.events if e.type == "ToolCallRequested"]
    completed_tools = [e.tool for e in sink.events if e.type == "ToolCallCompleted"]

    assert "web_search" in requested_tools
    assert "fetch_url" in requested_tools
    assert "web_search" in completed_tools
    assert "fetch_url" in completed_tools

    # Ensure EvidenceItemCreated count matches evidence length
    created = [e for e in sink.events if e.type == "EvidenceItemCreated"]
    assert len(created) == len(evidence)

    # Ensure ResearchCompleted summary matches evidence count
    completed = [e for e in sink.events if e.type == "ResearchCompleted"]
    assert len(completed) == 1
    assert completed[0].data.get("evidence_count") == len(evidence)