from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from engine.agents.planner import PlannerAgent, ResearchPlan
from engine.graph.nodes import planner_node
from dotenv import load_dotenv

load_dotenv()

PLAN_JSON = """
{
  "subquestions": [
    {
      "question": "What is the current size and growth rate of the SMB payroll market?",
      "search_queries": ["SMB payroll market size", "small business payroll market CAGR"]
    },
    {
      "question": "Who are the main competitors and how do they differentiate?",
      "search_queries": ["Gusto vs ADP RUN vs Paychex Flex", "SMB payroll competitors pricing"]
    },
    {
      "question": "What regulatory and compliance requirements apply to SMB payroll providers?",
      "search_queries": ["US payroll tax compliance requirements", "SOC 2 payroll software"]
    }
  ],
  "stop_criteria": {
    "min_sources": 8,
    "min_claim_coverage": 0.85,
    "max_minutes": 10
  },
  "assumptions": [
    "SMB is defined as 1-500 employees",
    {"assumption": "Focus on the US market", "rationale": "Initial GTM and compliance scope"}
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

    # subquestions are objects now
    assert len(plan.subquestions) >= 3
    assert all(sq.question.strip() for sq in plan.subquestions)
    assert all(len(sq.search_queries) >= 2 for sq in plan.subquestions)

    # top-level search_queries should be present (flattened)
    assert len(plan.search_queries) >= 5
    assert all(q.strip() for q in plan.search_queries)

    # stop criteria sanity
    assert plan.stop_criteria.min_sources >= 1
    assert 0.0 <= plan.stop_criteria.min_claim_coverage <= 1.0

    # assumptions are normalized into Assumption objects
    assert len(plan.assumptions) >= 1
    assert all(hasattr(a, "assumption") and a.assumption.strip() for a in plan.assumptions)


def test_planner_node_writes_plan_into_state():
    llm = FakeMessagesListChatModel(responses=[AIMessage(content=PLAN_JSON)])
    agent = PlannerAgent(llm=llm)

    node = planner_node(agent)
    state = {"question": "Should we enter the SMB payroll market?", "budget_usd": 2.5, "time_limit_s": 180}

    out = node(state, config={})
    assert "plan" in out
    assert out["plan"].stop_criteria.min_claim_coverage <= 1.0
    assert len(out["plan"].subquestions) >= 3
    

def test_assumptions_accept_strings_or_objects():
    data_strings = {
        "subquestions": [{"question": "Q0001", "search_queries": ["a","b"]},
                         {"question": "Q0002", "search_queries": ["c","d"]},
                         {"question": "Q0003", "search_queries": ["e","f"]}],
        "assumptions": ["A0001", "A0002"]
    }
    plan = ResearchPlan.model_validate(data_strings)
    assert all(hasattr(a, "assumption") for a in plan.assumptions)

    data_objs = {
        **data_strings,
        "assumptions": [{"assumption": "A0001", "rationale": "R0001"}]
    }
    plan2 = ResearchPlan.model_validate(data_objs)
    assert plan2.assumptions[0].rationale == "R0001"