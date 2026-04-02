"""Tests for the evaluation framework.

These tests verify that the evaluation metrics and scoring are calculated correctly.
"""

import pytest
from engine.evaluation.metrics import (
    EvaluationMetrics,
    EvaluationScore,
    CitationMetrics,
    CompletenessMetrics,
    evaluate_run,
)
from engine.schemas.brief import BriefDraft, Claim
from engine.schemas.evidence import EvidenceItem


# Test fixtures
@pytest.fixture
def sample_evidence():
    """Create sample evidence items."""
    return [
        EvidenceItem(
            id="S1",
            url="https://example.com/article1",
            title="Article 1",
            snippet="This is about governance",
            reliability_score=0.85,
            relevance_score=0.9,
            publisher="Example Org",
        ),
        EvidenceItem(
            id="S2",
            url="https://example.com/article2",
            title="Article 2",
            snippet="This discusses regulations",
            reliability_score=0.75,
            relevance_score=0.85,
            publisher="Example Org",
        ),
        EvidenceItem(
            id="S3",
            url="https://example.com/article3",
            title="Article 3",
            snippet="Implementation challenges",
            reliability_score=0.80,
            relevance_score=0.8,
            publisher="Example Org",
        ),
    ]


@pytest.fixture
def sample_brief_well_cited(sample_evidence):
    """Create a well-cited brief."""
    return BriefDraft(
        title="AI Governance Framework",
        executive_summary="A comprehensive overview of AI governance",
        key_findings=[
            Claim(
                text="Federal guidelines provide clear direction for AI policy",
                citations=["S1"],
                confidence=0.85,
            ),
            Claim(
                text="Regulatory compliance is essential for trusted AI",
                citations=["S1", "S2"],
                confidence=0.80,
            ),
            Claim(
                text="Implementation challenges require careful planning",
                citations=["S2", "S3"],
                confidence=0.75,
            ),
        ],
        risks=[
            Claim(
                text="Inadequate oversight could lead to misuse",
                citations=["S1"],
                confidence=0.70,
            ),
        ],
        recommendation="Adopt comprehensive governance framework",
        assumptions=["Stakeholder cooperation"],
        limitations=["Limited to US context", "Rapidly evolving landscape"],
        next_steps=["Review current policies", "Consult stakeholders"],
    )


@pytest.fixture
def sample_brief_poorly_cited(sample_evidence):
    """Create a poorly-cited brief."""
    return BriefDraft(
        title="AI Implementation",
        executive_summary="How to implement AI systems",
        key_findings=[
            Claim(
                text="Companies need comprehensive AI strategies for success",
                citations=[],  # No citations
                confidence=0.5,
            ),
            Claim(
                text="Workforce training is very important for adoption",
                citations=["S99"],  # Orphaned citation
                confidence=0.5,
            ),
            Claim(
                text="Budget planning should be carefully considered",
                citations=[],  # No citations
                confidence=0.5,
            ),
        ],
        risks=[],
        recommendation="Start with focused pilot programs",
        assumptions=[],
        limitations=[],
        next_steps=[],
    )


class TestCitationMetrics:
    """Test citation metrics calculation."""
    
    def test_well_cited_brief(self, sample_brief_well_cited, sample_evidence):
        """Test metrics for a well-cited brief."""
        metrics = EvaluationMetrics.calculate_citation_metrics(
            sample_brief_well_cited, sample_evidence
        )
        
        assert metrics.total_claims == 4  # 3 findings + 1 risk
        assert metrics.cited_claims == 4  # All cited
        assert metrics.citation_recall == 1.0
        assert metrics.unique_sources_cited == 3
        assert metrics.orphaned_citations == 0
        assert metrics.avg_citations_per_claim > 0
    
    def test_poorly_cited_brief(self, sample_brief_poorly_cited, sample_evidence):
        """Test metrics for a poorly-cited brief."""
        metrics = EvaluationMetrics.calculate_citation_metrics(
            sample_brief_poorly_cited, sample_evidence
        )
        
        assert metrics.total_claims == 3
        assert metrics.cited_claims == 1  # Only one claim has citations
        assert metrics.citation_recall < 0.5  # Low recall
        assert metrics.orphaned_citations == 1  # S99 doesn't exist
    
    def test_no_citations(self, sample_evidence):
        """Test metrics for brief with no citations."""
        brief = BriefDraft(
            title="Test Brief",
            executive_summary="No citations example",
            key_findings=[
                Claim(text="First important claim without citations", citations=[], confidence=0.5),
                Claim(text="Second important claim without citations", citations=[], confidence=0.5),
                Claim(text="Third important claim without citations", citations=[], confidence=0.5),
            ],
            recommendation="Next steps",
            risks=[],
        )
        
        metrics = EvaluationMetrics.calculate_citation_metrics(brief, sample_evidence)
        
        assert metrics.total_claims == 3
        assert metrics.cited_claims == 0
        assert metrics.citation_recall == 0.0


class TestCompletenessMetrics:
    """Test completeness metrics calculation."""
    
    def test_full_coverage(self, sample_brief_well_cited):
        """Test metrics when all aspects are covered."""
        expected = [
            "governance",
            "Federal guidelines",
            "regulatory compliance",
            "implementation",
        ]
        
        metrics = EvaluationMetrics.calculate_completeness_metrics(
            sample_brief_well_cited, expected
        )
        
        assert len(metrics.expected_aspects) == 4
        assert metrics.aspect_coverage_ratio >= 0.75  # At least 3 of 4
        assert len(metrics.missing_aspects) <= 1
    
    def test_partial_coverage(self, sample_brief_poorly_cited):
        """Test metrics when only some aspects are covered."""
        expected = [
            "governance",
            "compliance",
            "security",
            "deployment",
        ]
        
        metrics = EvaluationMetrics.calculate_completeness_metrics(
            sample_brief_poorly_cited, expected
        )
        
        assert metrics.aspect_coverage_ratio < 1.0
        assert len(metrics.covered_aspects) < len(expected)
        assert len(metrics.missing_aspects) > 0
    
    def test_no_expected_aspects(self, sample_brief_well_cited):
        """Test with no expected aspects specified."""
        metrics = EvaluationMetrics.calculate_completeness_metrics(
            sample_brief_well_cited, None
        )
        
        assert metrics.aspect_coverage_ratio == 0.0
        assert len(metrics.expected_aspects) == 0


class TestCoherenceMetrics:
    """Test coherence and structure metrics."""
    
    def test_well_structured_brief(self, sample_brief_well_cited):
        """Test metrics for well-structured brief."""
        metrics = EvaluationMetrics.calculate_coherence_metrics(sample_brief_well_cited)
        
        assert metrics.key_findings_count == 3
        assert metrics.risks_count == 1
        assert metrics.has_recommendations is True
        assert metrics.has_limitations is True
        assert metrics.has_assumptions is True
        assert metrics.brief_length_tokens > 0
    
    def test_minimal_brief(self, sample_brief_poorly_cited):
        """Test metrics for minimal brief."""
        metrics = EvaluationMetrics.calculate_coherence_metrics(sample_brief_poorly_cited)
        
        assert metrics.key_findings_count == 3
        assert metrics.risks_count == 0
        assert metrics.has_recommendations is True
        assert metrics.has_limitations is False
        assert metrics.has_assumptions is False


class TestOverallScoring:
    """Test overall score calculation."""
    
    def test_perfect_score(self):
        """Test calculation of perfect score."""
        citation_m = CitationMetrics(
            total_claims=5,
            cited_claims=5,
            citation_recall=1.0,
            avg_citations_per_claim=1.5,
            unique_sources_cited=5,
            orphaned_citations=0,
        )
        
        completeness_m = CompletenessMetrics(
            expected_aspects=["A", "B", "C"],
            covered_aspects=["A", "B", "C"],
            aspect_coverage_ratio=1.0,
            missing_aspects=[],
        )
        
        from engine.evaluation.metrics import CoherenceMetrics, EfficiencyMetrics
        
        coherence_m = CoherenceMetrics(
            brief_length_tokens=500,
            key_findings_count=5,
            risks_count=2,
            has_recommendations=True,
            has_limitations=True,
            has_assumptions=True,
            avg_claim_confidence=0.85,
        )
        
        efficiency_m = EfficiencyMetrics(
            total_cost_usd=0.01,
            total_time_s=10.0,
            total_tokens=200,
        )
        
        cit_s, comp_s, coh_s, eff_s, overall = (
            EvaluationMetrics.calculate_overall_score(
                citation_m, completeness_m, coherence_m, efficiency_m,
                passed_verification=True
            )
        )
        
        assert cit_s > 0.8
        assert comp_s >= 1.0
        assert overall > 0.85  # Should be very high


class TestEvaluateRunIntegration:
    """Test end-to-end evaluation."""
    
    def test_evaluate_run(self, sample_brief_well_cited, sample_evidence):
        """Test complete evaluation of a run."""
        metrics = {
            "cost_usd": 0.05,
            "elapsed_s": 45.0,
            "llm_total_tokens": 1500,
            "tool_calls": {"web_search": 3, "fetch_url": 5},
        }
        
        score = evaluate_run(
            run_id="test_001",
            question="Test question?",
            difficulty="medium",
            category="Test",
            brief=sample_brief_well_cited,
            evidence=sample_evidence,
            metrics=metrics,
            expected_aspects=["governance", "compliance", "implementation"],
            passed_verification=True,
        )
        
        assert isinstance(score, EvaluationScore)
        assert score.run_id == "test_001"
        assert score.overall_score > 0.0
        assert score.overall_score <= 1.0
        assert score.citation_score > 0.0
        assert score.completeness_score > 0.0
        assert score.passed_verification is True


class TestEvaluationScoreSerialization:
    """Test serialization of evaluation scores."""
    
    def test_to_dict(self, sample_brief_well_cited, sample_evidence):
        """Test conversion to dictionary."""
        metrics = {
            "cost_usd": 0.02,
            "elapsed_s": 30.0,
            "llm_total_tokens": 1000,
            "tool_calls": {"web_search": 2, "fetch_url": 4},
        }
        
        score = evaluate_run(
            run_id="test_001",
            question="Test?",
            difficulty="medium",
            category="Test",
            brief=sample_brief_well_cited,
            evidence=sample_evidence,
            metrics=metrics,
        )
        
        score_dict = score.to_dict()
        
        assert isinstance(score_dict, dict)
        assert "run_id" in score_dict
        assert "overall_score" in score_dict
        assert "citation_metrics" in score_dict
        assert isinstance(score_dict["citation_metrics"], dict)


def test_runner_integration_with_dummy_graph(sample_evidence):
    """Validate BenchmarkRunner integrates with graph->evaluate_run."""
    from engine.evaluation.runner import create_evaluation_benchmark_runner

    class DummyGraph:
        def invoke(self, state):
            return {
                "plan": state.get("question"),
                "evidence": sample_evidence,
                "brief": BriefDraft(
                    title="Dummy",
                    executive_summary="Dummy output",
                    key_findings=[
                        Claim(text="Key finding one is important", citations=["S1"], confidence=0.9),
                        Claim(text="Key finding two is noteworthy", citations=["S2"], confidence=0.85),
                        Claim(text="Key finding three is also relevant", citations=["S3"], confidence=0.8),
                    ],
                    risks=[Claim(text="Risk exists", citations=["S1"], confidence=0.7)],
                    recommendation="Do something",
                    next_steps=["Step 1"],
                    assumptions=["Assumption 1"],
                    limitations=["Limitation 1"],
                ),
                "report": type("R", (), {"passed": True})(),
                "metrics": {"cost_usd": 0.01, "elapsed_s": 1.0, "llm_total_tokens": 42, "tool_calls": {"web_search": 1, "fetch_url": 1}},
            }

    runner = create_evaluation_benchmark_runner(
        agents={},
        graph=DummyGraph(),
        benchmark_file="data/benchmark_dataset.jsonl",
        output_dir="out/evaluation_runs",
    )

    results = runner.run_benchmark(benchmark_id="bench_001", verbose=False)

    assert len(results) == 1
    score = results[0]
    assert isinstance(score, EvaluationScore)
    assert 0.0 <= score.overall_score <= 1.0
    assert score.passed_verification is True
