from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

from engine.agents.synthesizer import SynthesizerAgent
from engine.schemas.evidence import EvidenceItem
from engine.schemas.planner import ResearchPlan


BRIEF_JSON = """
{
  "title": "SMB Payroll Market Entry: Preliminary Brief",
  "executive_summary": "Entering the SMB payroll market could be attractive if we can differentiate on compliance automation and pricing. Evidence suggests a competitive landscape with incumbents, but opportunities may exist in underserved segments. This brief summarizes market signals, risks, and recommended next steps.",
  "key_findings": [
    {"text": "The SMB payroll market is competitive with established providers.", "citations": ["S1"], "confidence": 0.7},
    {"text": "Compliance requirements are a core execution risk and must be addressed early.", "citations": ["S2"], "confidence": 0.75},
    {"text": "Differentiation may come from workflow automation and integrations.", "citations": ["S1","S2"], "confidence": 0.6}
  ],
  "risks": [
    {"text": "Regulatory compliance complexity can increase cost and time-to-market.", "citations": ["S2"], "confidence": 0.7}
  ],
  "recommendation": "Proceed with a deeper validation sprint focusing on TAM/SAM/SOM, competitive teardown, and compliance roadmap before committing build resources.",
  "next_steps": ["Collect 3–5 recent market sizing sources", "Do a competitor feature/pricing matrix", "Draft compliance requirements checklist"],
  "assumptions": ["US-only scope for initial analysis"],
  "limitations": ["This is preliminary desk research."]
}
""".strip()


def test_synthesizer_returns_briefdraft():
    llm = FakeMessagesListChatModel(responses=[AIMessage(content=BRIEF_JSON)])
    agent = SynthesizerAgent(llm=llm)

    plan = ResearchPlan.model_validate({
        "subquestions": ["A", "B", "C"],
        "search_queries": ["q1", "q2", "q3"],
        "assumptions": ["US-only scope"]
    })

    evidence = [
        EvidenceItem(id="S1", url="https://sec.gov/x", title="SEC", snippet="...", reliability_score=0.9, relevance_score=0.6),
        EvidenceItem(id="S2", url="https://oecd.org/y", title="OECD", snippet="...", reliability_score=0.9, relevance_score=0.7),
    ]

    brief = agent.synthesize("Should we enter SMB payroll?", plan=plan, evidence=evidence)

    assert len(brief.key_findings) >= 3
    assert all(c.startswith("S") for k in brief.key_findings for c in k.citations)


def test_synthesizer_fills_missing_key_findings():
    brief_json_incomplete = """
    {
      "title": "SMB Payroll Market Entry: Incomplete Brief",
      "executive_summary": "The summary is sufficiently long to pass the schema checks. This is a placeholder narrative describing the high-level outcome and trends in the SMB payroll segment.",
      "key_findings": [
        {"text": "The SMB payroll market is competitive with established providers.", "citations": ["S1"], "confidence": 0.0},
        {"text": "Compliance requirements are a core risk and must be addressed early.", "citations": ["S2"], "confidence": 0.0}
      ],
      "risks": [],
      "recommendation": "Proceed with a deeper validation sprint focusing on compliance and market sizing.",
      "next_steps": [],
      "assumptions": [],
      "limitations": []
    }
    """.strip()

    llm = FakeMessagesListChatModel(responses=[AIMessage(content=brief_json_incomplete)])
    agent = SynthesizerAgent(llm=llm)

    plan = ResearchPlan.model_validate({
        "subquestions": ["A", "B", "C"],
        "search_queries": ["q1", "q2", "q3"],
        "assumptions": ["US-only scope"]
    })

    evidence = [
        EvidenceItem(id="S1", url="https://sec.gov/x", title="SEC", snippet="...", reliability_score=0.9, relevance_score=0.6),
    ]

    brief = agent.synthesize("Should we enter SMB payroll?", plan=plan, evidence=evidence)

    assert len(brief.key_findings) == 3
    assert brief.key_findings[-1].text.startswith("Insufficient key findings")
