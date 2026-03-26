"""Verification agent for validating research claims and evidence quality.

This module contains the VerifierAgent class, which is responsible for verifying
the quality and validity of research briefs by checking citations, evidence
reliability, and claim coverage. The agent validates that claims are properly
supported by evidence and meets quality thresholds defined in the research plan.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from engine.schemas.verify import VerificationReport, VerificationIssue
from engine.schemas.brief import BriefDraft, Claim
from engine.schemas.evidence import EvidenceItem
from engine.schemas.planner import ResearchPlan
from engine.events.emitter import Emitter
from engine.tools.extract import claim_confidence_embed


@dataclass
class VerifierConfig:
    """Configuration settings for the verification agent.

    Defines thresholds and parameters used by the VerifierAgent to evaluate
    the quality and validity of research briefs and evidence.

    Attributes:
        min_reliability_required: Minimum reliability score required for cited evidence.
        embedding_model: Name of the embedding model used for claim confidence scoring.
        evidence_quality_threshold: Threshold below which evidence quality triggers refetch.
    """
    min_reliability_required: float = 0.5
    embedding_model: str = "nomic-embed-text"
    evidence_quality_threshold: float = 0.6  # threshold for low-quality evidence to trigger refetch


@dataclass
class VerifierAgent:
    """Agent responsible for verifying the quality and validity of research briefs.

    This agent validates that claims in the brief are properly supported by evidence,
    checks citation validity, assesses evidence reliability, and ensures the brief
    meets quality thresholds defined in the research plan.

    Attributes:
        cfg: Configuration settings for verification parameters and thresholds.
    """
    cfg: VerifierConfig = field(default_factory=VerifierConfig())

    def verify(
        self,
        plan: ResearchPlan,
        evidence: List[EvidenceItem],
        brief: BriefDraft,
        emitter: Optional[Emitter] = None,
    ) -> VerificationReport:
        """Verify the quality and validity of a research brief.

        Performs comprehensive validation of the brief including:
        - Citation validity and existence
        - Evidence reliability assessment
        - Claim coverage and support
        - Compliance with plan requirements (min sources, coverage thresholds)
        - Confidence scoring for claims using embedding similarity

        Args:
            plan: The research plan containing stop criteria and requirements.
            evidence: List of evidence items available for citation validation.
            brief: The brief draft to be verified.
            emitter: Optional event emitter for logging verification progress.

        Returns:
            A VerificationReport containing the verification results, including
            pass/fail status, coverage metrics, identified issues, and quality scores.
        """
        emitter and emitter.emit("AgentStarted", agent="verifier")

        evidence_by_id: Dict[str, EvidenceItem] = {e.id: e for e in evidence}
        evidence_ids: Set[str] = set(evidence_by_id.keys())

        # Collect claims to verify
        claims: List[tuple[str, Claim]] = []
        for i, c in enumerate(brief.key_findings):
            claims.append((f"key_findings[{i}]", c))
        for i, c in enumerate(brief.risks):
            claims.append((f"risks[{i}]", c))

        issues: List[VerificationIssue] = []

        claim_count = len(claims)
        cited_claim_count = 0

        sources_used_ids: Set[str] = set()
        min_rel_observed = 1.0 if evidence else 0.0

        for loc, claim in claims:
            cits = claim.citations or []

            # Missing citations
            if not cits:
                issues.append(VerificationIssue(
                    severity="error",
                    code="MISSING_CITATION",
                    message="Claim has no citations.",
                    location=loc,
                ))
                claim.confidence = 0.2
                continue

            # Validate citations exist
            missing = [cid for cid in cits if cid not in evidence_ids]
            if missing:
                issues.append(VerificationIssue(
                        severity="error",
                        code="INVALID_CITATION",
                        message=f"Citations not found in evidence: {missing}",
                        location=loc,
                        evidence_ids=cits,
                ))
            
            # Count valid citations for coverage and scoring
            valid_cits = [cid for cid in cits if cid in evidence_ids]
            if valid_cits:
                cited_claim_count += 1

            # Gather cited evidence texts + rel scores
            cited_texts = []
            cited_rels = []
            for cid in valid_cits:
                ei = evidence_by_id.get(cid)
                cited_rels.append(float(ei.reliability_score))
                cited_texts.append(f"{ei.title or ''}\n{ei.snippet or ''}".strip())
                sources_used_ids.add(cid)
                min_rel_observed = min(min_rel_observed, ei.reliability_score)
            
            # overwrite model-provided confidence
            claim.confidence = claim_confidence_embed(
                claim_text=claim.text,
                cited_texts=cited_texts,
                cited_reliabilities=cited_rels,
                model=getattr(self.cfg, "embedding_model", "nomic-embed-text"),
            )

        # Stop criteria
        min_sources_required = getattr(getattr(plan, "stop_criteria", None), "min_sources", 0) or 0
        min_claim_coverage_required = getattr(getattr(plan, "stop_criteria", None), "min_claim_coverage", 0.0) or 0.0

        citation_coverage = (cited_claim_count / claim_count) if claim_count else 0.0

        if min_sources_required and len(sources_used_ids) < min_sources_required:
            issues.append(
                VerificationIssue(
                    severity="error",
                    code="INSUFFICIENT_SOURCES",
                    message=f"Used {len(sources_used_ids)} sources but requires at least {min_sources_required}.",
                )
            )

        if min_claim_coverage_required and citation_coverage < float(min_claim_coverage_required):
            issues.append(
                VerificationIssue(
                    severity="error",
                    code="INSUFFICIENT_COVERAGE",
                    message=f"Citation coverage {citation_coverage:.2f} below required {float(min_claim_coverage_required):.2f}.",
                )
            )

        if evidence and min_rel_observed < self.cfg.min_reliability_required:
            issues.append(
                VerificationIssue(
                    severity="warning",
                    code="LOW_RELIABILITY_CITATION",
                    message=f"Minimum cited reliability {min_rel_observed:.2f} below threshold {self.cfg.min_reliability_required:.2f}.",
                )
            )

        passed = not any(i.severity == "error" for i in issues)

        report = VerificationReport(
            passed=passed,
            claim_count=claim_count,
            cited_claim_count=cited_claim_count,
            citation_coverage=citation_coverage,
            min_sources_required=min_sources_required,
            sources_used=len(sources_used_ids),
            min_reliability_required=self.cfg.min_reliability_required,
            min_reliability_observed=min_rel_observed if evidence else 0.0,
            issues=issues,
        )

        emitter and emitter.emit(
            "AgentFinished",
            agent="verifier",
            passed=report.passed,
            issues=len(report.issues),
            citation_coverage=report.citation_coverage,
            sources_used=report.sources_used,
        )
        return report