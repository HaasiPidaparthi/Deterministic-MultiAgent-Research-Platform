from engine.agents.verifier import VerifierAgent, VerifierConfig
from engine.graph.nodes import verifier_node
from engine.schemas.brief import BriefDraft
from engine.schemas.evidence import EvidenceItem
from engine.schemas.planner import ResearchPlan


def test_verifier_flags_missing_and_invalid_citations():
    verifier = VerifierAgent(cfg=VerifierConfig(min_reliability_required=0.6))

    plan = ResearchPlan.model_validate({
        "subquestions": ["A","B","C"],
        "search_queries": ["q1","q2","q3","q4","q5"],
        "stop_criteria": {"min_sources": 2, "min_claim_coverage": 0.8}
    })

    evidence = [
        EvidenceItem(id="S1", url="https://sec.gov/x", title="SEC", snippet="...", reliability_score=0.9, relevance_score=0.6),
        EvidenceItem(id="S2", url="https://example.com/y", title="Blog", snippet="...", reliability_score=0.4, relevance_score=0.6),
    ]

    brief = BriefDraft.model_validate({
        "title": "Draft",
        "executive_summary": "X" * 80,
        "key_findings": [
            {"text": "Claim with valid citation", "citations": ["S1"], "confidence": 0.7},
            {"text": "Claim missing citation", "citations": [], "confidence": 0.7},
            {"text": "Claim with invalid citation", "citations": ["S999"], "confidence": 0.7},
        ],
        "risks": [
            {"text": "Risk cited with low reliability source", "citations": ["S2"], "confidence": 0.6}
        ],
        "recommendation": "Y" * 60,
        "next_steps": ["Do more research"],
        "assumptions": [],
        "limitations": []
    })

    report = verifier.verify(plan=plan, evidence=evidence, brief=brief)

    assert report.passed is False
    codes = [i.code for i in report.issues]
    assert "MISSING_CITATION" in codes
    assert "INVALID_CITATION" in codes
    # low reliability should be warning
    assert "LOW_RELIABILITY_CITATION" in codes


def test_verifier_node_refetch_urls_use_config_threshold():
    verifier = VerifierAgent(cfg=VerifierConfig(evidence_quality_threshold=0.9))
    node = verifier_node(verifier)

    plan = ResearchPlan.model_validate({
        "subquestions": ["A", "B", "C"],
        "search_queries": ["q1", "q2", "q3"],
        "stop_criteria": {"min_sources": 1, "min_claim_coverage": 0.8},
    })

    evidence = [
        EvidenceItem(id="S1", url="https://example.com/low1", title="X", snippet="...", reliability_score=0.4, relevance_score=0.3),
        EvidenceItem(id="S2", url="https://example.com/low2", title="Y", snippet="...", reliability_score=0.45, relevance_score=0.35),
    ]

    brief = BriefDraft.model_validate({
        "title": "Draft",
        "executive_summary": "This is a descriptive summary text with more than fifty characters.",
        "key_findings": [
            {"text": "Claim about payroll system quality.", "citations": ["S1"], "confidence": 0.7},
            {"text": "Another claim about support options.", "citations": [], "confidence": 0.6},
            {"text": "Third claim about vendor competition.", "citations": ["S2"], "confidence": 0.6},
        ],
        "risks": [],
        "recommendation": "Recommendation text should be at least 30 characters long.",
        "next_steps": [],
        "assumptions": [],
        "limitations": [],
    })

    updated = node({"plan": plan, "evidence": evidence, "brief": brief}, {"configurable": {"emitter": None}})

    assert "refetch_urls" in updated
    assert set(updated["refetch_urls"]) == {"https://example.com/low1", "https://example.com/low2"}