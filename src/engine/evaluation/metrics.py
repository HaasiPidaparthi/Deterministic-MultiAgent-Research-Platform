"""Evaluation metrics for assessing quality of synthesized research briefs.

This module provides a comprehensive set of metrics for evaluating:
- Factual accuracy of claims
- Citation relevance and recall
- Answer completeness
- System efficiency
- Output coherence
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re
from collections import Counter

from engine.schemas.brief import BriefDraft, Claim
from engine.schemas.evidence import EvidenceItem
from engine.schemas.planner import ResearchPlan


@dataclass
class CitationMetrics:
    """Metrics for citation quality and coverage."""
    total_claims: int = 0
    cited_claims: int = 0
    citation_recall: float = 0.0  # cited_claims / total_claims
    avg_citations_per_claim: float = 0.0
    unique_sources_cited: int = 0
    orphaned_citations: int = 0  # citations to non-existent sources


@dataclass
class CompletenessMetrics:
    """Metrics for coverage of expected aspects."""
    expected_aspects: List[str] = None
    covered_aspects: List[str] = None
    aspect_coverage_ratio: float = 0.0  # covered / expected
    missing_aspects: List[str] = None
    
    def __post_init__(self):
        if self.covered_aspects is None:
            self.covered_aspects = []
        if self.missing_aspects is None:
            self.missing_aspects = []
        if self.expected_aspects is None:
            self.expected_aspects = []


@dataclass
class EfficiencyMetrics:
    """Metrics for system resource usage."""
    total_cost_usd: float = 0.0
    total_time_s: float = 0.0
    total_tokens: int = 0
    web_search_calls: int = 0
    fetch_url_calls: int = 0
    cost_per_query: float = 0.0
    tokens_per_query: float = 0.0


@dataclass
class CoherenceMetrics:
    """Metrics for output quality and consistency."""
    brief_length_tokens: int = 0
    key_findings_count: int = 0
    risks_count: int = 0
    has_recommendations: bool = False
    has_limitations: bool = False
    has_assumptions: bool = False
    avg_claim_confidence: float = 0.0


@dataclass
class EvaluationScore:
    """Comprehensive evaluation score for a complete run."""
    run_id: str
    question: str
    difficulty: str
    category: str
    
    # Sub-scores (0-1 range)
    citation_score: float = 0.0  # How well are claims cited?
    completeness_score: float = 0.0  # Are expected aspects covered?
    coherence_score: float = 0.0  # Is the brief well-structured?
    efficiency_score: float = 0.0  # Was it cost/time efficient? (inverted)
    
    # Weighted overall score
    overall_score: float = 0.0
    
    # Detailed metrics
    citation_metrics: CitationMetrics = None
    completeness_metrics: CompletenessMetrics = None
    coherence_metrics: CoherenceMetrics = None
    efficiency_metrics: EfficiencyMetrics = None
    
    # Metadata
    passed_verification: bool = False
    issues: List[str] = None
    
    def __post_init__(self):
        if self.citation_metrics is None:
            self.citation_metrics = CitationMetrics()
        if self.completeness_metrics is None:
            self.completeness_metrics = CompletenessMetrics()
        if self.coherence_metrics is None:
            self.coherence_metrics = CoherenceMetrics()
        if self.efficiency_metrics is None:
            self.efficiency_metrics = EfficiencyMetrics()
        if self.issues is None:
            self.issues = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert nested dataclasses
        result['citation_metrics'] = asdict(self.citation_metrics)
        result['completeness_metrics'] = asdict(self.completeness_metrics)
        result['coherence_metrics'] = asdict(self.coherence_metrics)
        result['efficiency_metrics'] = asdict(self.efficiency_metrics)
        return result


class EvaluationMetrics:
    """Comprehensive evaluation metrics calculator."""
    
    @staticmethod
    def calculate_citation_metrics(
        brief: BriefDraft,
        evidence: List[EvidenceItem]
    ) -> CitationMetrics:
        """Calculate citation quality metrics."""
        evidence_ids = {e.id for e in evidence}
        
        # Collect all claims
        all_claims = (brief.key_findings or []) + (brief.risks or [])
        total_claims = len(all_claims)
        
        cited_claims = 0
        all_citations = []
        orphaned = 0
        
        for claim in all_claims:
            citations = claim.citations or []
            if citations:
                cited_claims += 1
                all_citations.extend(citations)
                # Check for orphaned citations
                for cit in citations:
                    if cit not in evidence_ids:
                        orphaned += 1
        
        unique_sources = len(set(all_citations))
        avg_citations = len(all_citations) / total_claims if total_claims > 0 else 0.0
        recall = cited_claims / total_claims if total_claims > 0 else 0.0
        
        return CitationMetrics(
            total_claims=total_claims,
            cited_claims=cited_claims,
            citation_recall=recall,
            avg_citations_per_claim=avg_citations,
            unique_sources_cited=unique_sources,
            orphaned_citations=orphaned
        )
    
    @staticmethod
    def calculate_completeness_metrics(
        brief: BriefDraft,
        expected_aspects: Optional[List[str]] = None
    ) -> CompletenessMetrics:
        """Calculate coverage of expected aspects."""
        if not expected_aspects:
            return CompletenessMetrics(expected_aspects=[])
        
        # Combine brief text
        brief_text = (
            (brief.executive_summary or "") + " " +
            (brief.recommendation or "") + " " +
            " ".join([c.text or "" for c in (brief.key_findings or [])]) + " " +
            " ".join([c.text or "" for c in (brief.risks or [])])
        ).lower()
        
        covered = []
        missing = []
        
        for aspect in expected_aspects:
            # Simple presence check (can be enhanced with NLP)
            if any(word in brief_text for word in aspect.lower().split()):
                covered.append(aspect)
            else:
                missing.append(aspect)
        
        coverage_ratio = len(covered) / len(expected_aspects) if expected_aspects else 0.0
        
        return CompletenessMetrics(
            expected_aspects=expected_aspects,
            covered_aspects=covered,
            aspect_coverage_ratio=coverage_ratio,
            missing_aspects=missing
        )
    
    @staticmethod
    def calculate_coherence_metrics(
        brief: BriefDraft
    ) -> CoherenceMetrics:
        """Calculate output structure and quality metrics."""
        key_findings = brief.key_findings or []
        risks = brief.risks or []
        
        # Estimate token count (rough: ~1.3 tokens per word)
        brief_text = (
            (brief.title or "") + " " +
            (brief.executive_summary or "") + " " +
            (brief.recommendation or "") + " " +
            " ".join([c.text or "" for c in key_findings]) + " " +
            " ".join([c.text or "" for c in risks]) + " " +
            " ".join(brief.assumptions or []) + " " +
            " ".join(brief.limitations or []) + " " +
            " ".join(brief.next_steps or [])
        )
        word_count = len(brief_text.split())
        token_estimate = int(word_count * 1.3)
        
        # Average confidence
        all_claims = key_findings + risks
        avg_confidence = (
            sum(getattr(c, 'confidence', 0.0) or 0.0 for c in all_claims) /
            len(all_claims) if all_claims else 0.0
        )
        
        return CoherenceMetrics(
            brief_length_tokens=token_estimate,
            key_findings_count=len(key_findings),
            risks_count=len(risks),
            has_recommendations=bool(brief.recommendation),
            has_limitations=bool(brief.limitations),
            has_assumptions=bool(brief.assumptions),
            avg_claim_confidence=avg_confidence
        )
    
    @staticmethod
    def calculate_efficiency_metrics(
        metrics: Dict[str, Any],
        evidence_count: int = 0
    ) -> EfficiencyMetrics:
        """Calculate resource efficiency metrics."""
        total_cost = metrics.get("cost_usd", 0.0)
        total_time = metrics.get("elapsed_s", 0.0)
        total_tokens = metrics.get("llm_total_tokens", 0)
        tool_calls = metrics.get("tool_calls", {})
        
        web_search = tool_calls.get("web_search", 0)
        fetch_url = tool_calls.get("fetch_url", 0)
        
        cost_per_query = total_cost / max(1, web_search)
        tokens_per_query = total_tokens / max(1, evidence_count)
        
        return EfficiencyMetrics(
            total_cost_usd=total_cost,
            total_time_s=total_time,
            total_tokens=total_tokens,
            web_search_calls=web_search,
            fetch_url_calls=fetch_url,
            cost_per_query=cost_per_query,
            tokens_per_query=tokens_per_query
        )
    
    @staticmethod
    def calculate_overall_score(
        citation_metrics: CitationMetrics,
        completeness_metrics: CompletenessMetrics,
        coherence_metrics: CoherenceMetrics,
        efficiency_metrics: EfficiencyMetrics,
        passed_verification: bool = False,
        weights: Optional[Dict[str, float]] = None
    ) -> tuple[float, float, float, float, float]:
        """Calculate weighted overall score with sub-scores.
        
        Returns:
            Tuple of (citation_score, completeness_score, coherence_score, 
                     efficiency_score, overall_score)
        """
        if weights is None:
            weights = {
                "citation": 0.35,
                "completeness": 0.30,
                "coherence": 0.20,
                "efficiency": 0.15
            }
        
        # Citation score: recall and orphan-free
        citation_score = (
            citation_metrics.citation_recall * 0.8 +  # Recall is important
            max(0, 1 - (citation_metrics.orphaned_citations * 0.1)) * 0.2  # Penalize orphans
        )
        
        # Completeness score: aspect coverage
        completeness_score = completeness_metrics.aspect_coverage_ratio
        
        # Coherence score: structure and confidence
        coherence_score = (
            min(1.0, coherence_metrics.key_findings_count / 5) * 0.5 +  # Expect ~5 findings
            min(1.0, coherence_metrics.avg_claim_confidence) * 0.3 +  # Confidence score
            (0.2 if coherence_metrics.has_limitations else 0.0)  # Bonus for limitations
        )
        
        # Efficiency score: lower cost/time is better (inverted)
        # Normalize: < $0.05 = 1.0, > $0.50 = 0.0
        cost_efficiency = max(0, min(1.0, 1 - (efficiency_metrics.total_cost_usd / 0.5)))
        # Normalize: < 30s = 1.0, > 300s = 0.0
        time_efficiency = max(0, min(1.0, 1 - (efficiency_metrics.total_time_s / 300)))
        efficiency_score = cost_efficiency * 0.6 + time_efficiency * 0.4
        
        # Verification bonus
        verification_bonus = 0.1 if passed_verification else 0.0
        
        # Weighted overall
        overall = (
            citation_score * weights["citation"] +
            completeness_score * weights["completeness"] +
            coherence_score * weights["coherence"] +
            efficiency_score * weights["efficiency"]
        ) + verification_bonus
        
        return (
            min(1.0, citation_score),
            completeness_score,
            min(1.0, coherence_score),
            efficiency_score,
            min(1.0, overall)
        )


def evaluate_run(
    run_id: str,
    question: str,
    difficulty: str,
    category: str,
    brief: BriefDraft,
    evidence: List[EvidenceItem],
    metrics: Dict[str, Any],
    expected_aspects: Optional[List[str]] = None,
    passed_verification: bool = False
) -> EvaluationScore:
    """Evaluate a complete research run.
    
    Args:
        run_id: Unique identifier for the run
        question: The research question
        difficulty: Difficulty level (easy/medium/hard)
        category: Category/topic of the question
        brief: The generated brief
        evidence: The evidence items used
        metrics: Execution metrics (cost, time, tokens, etc.)
        expected_aspects: Expected aspects to cover
        passed_verification: Whether verification passed
    
    Returns:
        EvaluationScore with detailed metrics and scores
    """
    citation_m = EvaluationMetrics.calculate_citation_metrics(brief, evidence)
    completeness_m = EvaluationMetrics.calculate_completeness_metrics(
        brief, expected_aspects
    )
    coherence_m = EvaluationMetrics.calculate_coherence_metrics(brief)
    efficiency_m = EvaluationMetrics.calculate_efficiency_metrics(metrics, len(evidence))
    
    citation_s, complete_s, coherence_s, efficiency_s, overall_s = (
        EvaluationMetrics.calculate_overall_score(
            citation_m, completeness_m, coherence_m, efficiency_m,
            passed_verification
        )
    )
    
    issues = []
    if citation_m.citation_recall < 0.7:
        issues.append("Low citation recall")
    if citation_m.orphaned_citations > 0:
        issues.append(f"Found {citation_m.orphaned_citations} orphaned citations")
    if completeness_m.aspect_coverage_ratio < 0.6:
        issues.append(f"Only {len(completeness_m.covered_aspects)}/{len(completeness_m.expected_aspects)} aspects covered")
    if efficiency_m.total_cost_usd > 0.5:
        issues.append("High cost execution")
    
    return EvaluationScore(
        run_id=run_id,
        question=question,
        difficulty=difficulty,
        category=category,
        citation_score=citation_s,
        completeness_score=complete_s,
        coherence_score=coherence_s,
        efficiency_score=efficiency_s,
        overall_score=overall_s,
        citation_metrics=citation_m,
        completeness_metrics=completeness_m,
        coherence_metrics=coherence_m,
        efficiency_metrics=efficiency_m,
        passed_verification=passed_verification,
        issues=issues
    )
