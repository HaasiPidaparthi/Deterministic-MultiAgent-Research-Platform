import pytest

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from engine.agents.planner import PlannerAgent
from engine.graph.nodes import planner_node
from dotenv import load_dotenv

load_dotenv()

PLAN_JSON = """
{
  "subquestions": [
    "What is the market size and growth for SMB payroll solutions in the US?",
    "Who are the primary buyers and what are their unmet needs?",
    "Who are the major competitors and how do they differentiate?",
    "What are typical pricing models and unit economics?",
    "What regulatory or compliance constraints apply (tax, labor, data)?",
    "What distribution channels and integrations are required to compete?"
  ],
  "search_queries": [
    "SMB payroll software market size CAGR 2025 2026",
    "Gusto market share revenue SMB payroll",
    "ADP RUN vs Paychex Flex SMB pricing comparison",
    "SMB payroll software switching costs pain points survey",
    "US payroll tax compliance requirements software vendor",
    "SOC 2 requirements payroll software vendors",
    "SMB payroll API integrations QuickBooks Xero requirements",
    "SMB payroll churn drivers Gusto Paychex ADP"
  ],
  "stop_criteria": {
    "min_sources": 8,
    "min_claim_coverage": 0.85,
    "max_minutes": 10
  },
  "assumptions": [
    "Geography is US-focused unless stated otherwise",
    "Target segment is SMBs under 500 employees"
  ],
  "risks_to_check": [
    "Regulatory compliance complexity",
    "High CAC due to incumbents and sticky integrations"
  ]
}
""".strip()


def test_planner_agent_returns_research_plan():
    # Fake model returns a single AI message containing JSON.
    llm = FakeMessagesListChatModel(responses=[AIMessage(content=PLAN_JSON)])
    agent = PlannerAgent(llm=llm)

    plan = agent.plan("Should we enter the SMB payroll market?", budget_usd=2.5, time_limit_s=180)
    print("Plan: ", plan)

    assert len(plan.subquestions) >= 3
    assert len(plan.search_queries) >= 5
    assert plan.stop_criteria.min_sources >= 1
    # sanity check: queries shouldn't be empty
    assert all(q.strip() for q in plan.search_queries)


def test_planner_node_writes_plan_into_state():
    llm = FakeMessagesListChatModel(responses=[AIMessage(content=PLAN_JSON)])
    agent = PlannerAgent(llm=llm)

    node = planner_node(agent)
    state = {"question": "Should we enter the SMB payroll market?", "budget_usd": 2.5, "time_limit_s": 180}

    out = node(state, config={})
    print("Out: ", out)
    assert "plan" in out
    assert out["plan"].stop_criteria.min_claim_coverage <= 1.0