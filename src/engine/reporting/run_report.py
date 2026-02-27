from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import List

from engine.schemas.evidence import EvidenceItem
from engine.schemas.planner import ResearchPlan
from engine.schemas.brief import BriefDraft
from engine.schemas.verify import VerificationReport


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _md_escape(s: str) -> str:
    return (s or "").replace("\n", " ").strip()


@dataclass
class RunReportPaths:
    run_dir: Path
    report_md: Path
    report_json: Path


def make_run_paths(run_id: str, base_dir: str = "out/reports") -> RunReportPaths:
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunReportPaths(
        run_dir=run_dir,
        report_md=run_dir / "report.md",
        report_json=run_dir / "report.json",
    )

def build_markdown_report(
    run_id: str,
    question: str,
    budget_usd: float,
    time_limit_s: int,
    plan: ResearchPlan,
    evidence: List[EvidenceItem],
    brief: BriefDraft,
    report: VerificationReport,
    events_path: str | None = None,
) -> str:
    # plan sections
    subq_lines = []
    for i, sq in enumerate(plan.subquestions, start=1):
        q = getattr(sq, "question", str(sq))
        subq_lines.append(f"{i}. {_md_escape(q)}")

    query_lines = "\n".join([f"- {_md_escape(q)}" for q in plan.search_queries[:30]])

    # evidence table (top N)
    top_evidence = evidence[:15]
    evidence_rows = []
    for e in top_evidence:
        evidence_rows.append(
            f"| {e.id} | {e.reliability_score:.2f} | {e.relevance_score:.2f} | {_md_escape(e.title or '')} | {e.url} |"
        )
    evidence_table = "\n".join(evidence_rows) if evidence_rows else "| - | - | - | - | - |"

    # brief findings
    finding_lines = []
    for i, k in enumerate(brief.key_findings, start=1):
        cits = ", ".join(k.citations or [])
        finding_lines.append(f"{i}. {_md_escape(k.text)}  \n   - Citations: `{cits}`  \n   - Confidence: {k.confidence:.2f}")

    risk_lines = []
    for i, r in enumerate(brief.risks or [], start=1):
        cits = ", ".join(r.citations or [])
        risk_lines.append(f"{i}. {_md_escape(r.text)}  \n   - Citations: `{cits}`  \n   - Confidence: {r.confidence:.2f}")

    # verifier issues
    issue_lines = []
    for iss in report.issues:
        loc = f" @ {iss.location}" if iss.location else ""
        ev = f" evidence={iss.evidence_ids}" if iss.evidence_ids else ""
        issue_lines.append(f"- **[{iss.severity.upper()}] {iss.code}**{loc}: {_md_escape(iss.message)}{ev}")

    assumptions = []
    for a in getattr(plan, "assumptions", []):
        assumptions.append(getattr(a, "assumption", str(a)))

    md = f"""# Run Report: {run_id}

**Generated:** {_now_iso()}  
**Question:** {question}  
**Budget:** ${budget_usd:.2f}  
**Time Limit:** {time_limit_s}s  
{f"**Event Log:** `{events_path}`" if events_path else ""}

---

## Plan

### Subquestions
{chr(10).join(subq_lines)}

### Search Queries (top)
{query_lines if query_lines else "- (none)"}

### Assumptions
{chr(10).join([f"- {_md_escape(a)}" for a in assumptions]) if assumptions else "- (none)"}

---

## Evidence (top {len(top_evidence)})

| ID | Reliability | Relevance | Title | URL |
|---:|---:|---:|---|---|
{evidence_table}

---

## Brief Draft

### Title
{_md_escape(brief.title)}

### Executive Summary
{brief.executive_summary.strip()}

### Key Findings
{chr(10).join(finding_lines) if finding_lines else "- (none)"}

### Risks
{chr(10).join(risk_lines) if risk_lines else "- (none)"}

### Recommendation
{brief.recommendation.strip()}

### Next Steps
{chr(10).join([f"- {_md_escape(s)}" for s in (brief.next_steps or [])]) if brief.next_steps else "- (none)"}

### Limitations
{chr(10).join([f"- {_md_escape(s)}" for s in (brief.limitations or [])]) if brief.limitations else "- (none)"}

---

## Verification Report

**PASSED:** {report.passed}  
**Claims:** {report.claim_count}  
**Cited Claims:** {report.cited_claim_count}  
**Citation Coverage:** {report.citation_coverage:.2f}  
**Sources Used:** {report.sources_used} (min required: {report.min_sources_required})  
**Min Reliability Cited:** {report.min_reliability_observed:.2f} (threshold: {report.min_reliability_required:.2f})

### Issues
{chr(10).join(issue_lines) if issue_lines else "- (none)"}
"""
    return md