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
    run_id=None,
    question=None,
    budget_usd=None,
    time_limit_s=None,
    plan=None,
    evidence=None,
    brief=None,
    report=None,
    events_path: str | None = None,
    metrics: dict | None = None,
) -> str:
    # Support state-based invocation for unit tests
    state = None
    out_dir = None
    stop_reason = None
    iter_count = None

    if isinstance(run_id, dict):
        state = run_id
        out_dir = question
        run_id = state.get("run_id", "run")
        question = state.get("question", "")
        budget_usd = float(state.get("budget_usd", 0.0))
        time_limit_s = int(state.get("time_limit_s", 0))
        plan = state.get("plan")
        evidence = state.get("evidence", [])
        brief = state.get("brief")
        report = state.get("report")
        events_path = state.get("events_path")
        metrics = state.get("metrics", {})
        stop_reason = state.get("stop_reason")
        iter_count = state.get("iter")
    else:
        if run_id is None:
            run_id = "run"
        # Use the passed metrics parameter if provided, otherwise default to empty dict
        if metrics is None:
            metrics = {}

    # plan sections
    subq_lines = []
    for i, sq in enumerate(getattr(plan, "subquestions", []) or [], start=1):
        q = getattr(sq, "question", str(sq))
        subq_lines.append(f"{i}. {_md_escape(q)}")

    query_lines = "\n".join(
        [f"- {_md_escape(q)}" for q in (getattr(plan, "search_queries", []) or [])[:30]]
    )

    # evidence table (top N)
    top_evidence = (evidence or [])[:15]
    evidence_rows = []
    for e in top_evidence:
        evidence_rows.append(
            f"| {e.id} | {e.reliability_score:.2f} | {e.relevance_score:.2f} | {_md_escape(e.title or '')} | {e.url} |"
        )
    evidence_table = "\n".join(evidence_rows) if evidence_rows else "| - | - | - | - | - |"

    # brief findings
    finding_lines = []
    for i, k in enumerate(getattr(brief, "key_findings", []) or [], start=1):
        cits = ", ".join(getattr(k, "citations", []) or [])
        finding_lines.append(
            f"{i}. {_md_escape(getattr(k, 'text', ''))}  \n   - Citations: `{cits}`  \n   - Confidence: {getattr(k, 'confidence', 0.0):.2f}"
        )

    risk_lines = []
    for i, r in enumerate(getattr(brief, "risks", []) or [], start=1):
        cits = ", ".join(getattr(r, "citations", []) or [])
        risk_lines.append(
            f"{i}. {_md_escape(getattr(r, 'text', ''))}  \n   - Citations: `{cits}`  \n   - Confidence: {getattr(r, 'confidence', 0.0):.2f}"
        )

    # verifier issues
    issue_lines = []
    for iss in getattr(report, "issues", []) or []:
        loc = f" @ {getattr(iss, 'location', '')}" if getattr(iss, "location", None) else ""
        ev = f" evidence={getattr(iss, 'evidence_ids', '')}" if getattr(iss, "evidence_ids", None) else ""
        issue_lines.append(
            f"- **[{getattr(iss, 'severity', '').upper()}] {getattr(iss, 'code', '')}**{loc}: {_md_escape(getattr(iss, 'message', ''))}{ev}"
        )

    assumptions = [getattr(a, "assumption", str(a)) for a in getattr(plan, "assumptions", []) or []]

    metric_lines = []
    if metrics:
        if iter_count is not None:
            metric_lines.append(f"- iter: {iter_count}")
        if stop_reason is not None:
            metric_lines.append(f"- stop_reason: {stop_reason}")
        for k, v in metrics.items():
            metric_lines.append(f"- {k}: {v}")
    else:
        metric_lines.append("- (none)")

    # Generate performance summary
    performance_lines = []
    if metrics:
        elapsed_s = float(metrics.get("elapsed_s", 0.0))
        cost_usd = float(metrics.get("cost_usd", 0.0))
        total_tokens = int(metrics.get("llm_total_tokens", 0))
        tool_calls = metrics.get("tool_calls", {})
        total_tool_calls = sum(tool_calls.values()) if tool_calls else 0
        successful_calls = total_tool_calls - sum(metrics.get("rejected_counts", {}).values())
        
        if elapsed_s > 0:
            performance_lines.append(f"- **Efficiency:** ${cost_usd/elapsed_s:.4f} per second")
        if total_tokens > 0 and elapsed_s > 0:
            performance_lines.append(f"- **Token Efficiency:** {total_tokens/elapsed_s:.0f} tokens per second")
        if total_tool_calls > 0:
            success_rate = (successful_calls / total_tool_calls) * 100
            performance_lines.append(f"- **Tool Success Rate:** {success_rate:.1f}% ({successful_calls} successful out of {total_tool_calls} total calls)")
        
        # Evidence quality summary
        if evidence:
            avg_reliability = sum(e.reliability_score for e in evidence) / len(evidence)
            performance_lines.append(f"- **Evidence Quality:** {len(evidence)} sources collected, {avg_reliability:.2f} average reliability")
    else:
        performance_lines.append("- (none)")

    md = f"""# Run Report: {run_id}

**Generated:** {_now_iso()}  
**Question:** {question}  
**Budget:** ${budget_usd:.2f}  
**Time Limit:** {time_limit_s}s  
{f"**Event Log:** `{events_path}`" if events_path else ""}

---

## Run Metrics
{chr(10).join(metric_lines)}

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
{_md_escape(getattr(brief, 'title', ''))}

### Executive Summary
{getattr(brief, 'executive_summary', '').strip()}

### Key Findings
{chr(10).join(finding_lines) if finding_lines else "- (none)"}

### Risks
{chr(10).join(risk_lines) if risk_lines else "- (none)"}

### Recommendation
{getattr(brief, 'recommendation', '').strip()}

### Next Steps
{chr(10).join([f"- {_md_escape(s)}" for s in (getattr(brief, 'next_steps', []) or [])]) if getattr(brief, 'next_steps', None) else "- (none)"}

### Limitations
{chr(10).join([f"- {_md_escape(s)}" for s in (getattr(brief, 'limitations', []) or [])]) if getattr(brief, 'limitations', None) else "- (none)"}

---

## Verification Report

**PASSED:** {getattr(report, 'passed', False)}  
**Claims:** {getattr(report, 'claim_count', 0)}  
**Cited Claims:** {getattr(report, 'cited_claim_count', 0)}  
**Citation Coverage:** {getattr(report, 'citation_coverage', 0.0):.2f}  
**Sources Used:** {getattr(report, 'sources_used', 0)} (min required: {getattr(report, 'min_sources_required', 0)})  
**Min Reliability Cited:** {getattr(report, 'min_reliability_observed', 0.0):.2f} (threshold: {getattr(report, 'min_reliability_required', 0.0):.2f})

### Issues
{chr(10).join(issue_lines) if issue_lines else "- (none)"}

---

## Cost Report

### Run Metrics
{chr(10).join(metric_lines)}

### Performance Summary
{chr(10).join(performance_lines)}
"""


    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(md, encoding='utf-8')

    return md