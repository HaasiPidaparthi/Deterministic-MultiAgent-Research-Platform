import os
import uuid
import json
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from engine.graph.flow import build_graph
from engine.agents.planner import PlannerAgent
from engine.agents.researcher import ResearcherAgent, ResearcherConfig
from engine.agents.synthesizer import SynthesizerAgent
from engine.agents.verifier import VerifierAgent, VerifierConfig
from engine.tools.web_search import web_search
from engine.tools.web_fetch import fetch_url
from engine.events.emitter import Emitter
from engine.events.sink import InMemorySink, JsonlFileSink
from engine.reporting.run_report import make_run_paths, build_markdown_report


def _print_plan(plan):
    print("\n=== PLAN ===")
    print(f"Subquestions: {len(plan.subquestions)}")
    for i, sq in enumerate(plan.subquestions[:6]):
        q = getattr(sq, "question", str(sq))
        print(f"  {i+1}. {q}")
    print(f"Search queries: {len(plan.search_queries)}")
    for q in plan.search_queries[:8]:
        print(f"  - {q}")

def _print_evidence(evidence):
    print("\n=== EVIDENCE (top) ===")
    for e in evidence[:8]:
        print(f"{e.id} | rel={e.reliability_score:.2f} rev={e.relevance_score:.2f} | {e.title or ''}")
        print(f"    {e.url}")
        if e.snippet:
            print(f"    {e.snippet[:180].strip()}")
        print()

def _print_brief(brief):
    print("\n=== BRIEF DRAFT ===")
    print(brief.title)
    print("\nExecutive summary:")
    print(brief.executive_summary.strip())

    print("\nKey findings:")
    for i, k in enumerate(brief.key_findings, start=1):
        cits = ", ".join(k.citations or [])
        print(f"  {i}. {k.text.strip()}  [{cits}] (conf={k.confidence:.2f})")

    if brief.risks:
        print("\nRisks:")
        for i, r in enumerate(brief.risks, start=1):
            cits = ", ".join(r.citations or [])
            print(f"  {i}. {r.text.strip()}  [{cits}] (conf={r.confidence:.2f})")

    print("\nRecommendation:")
    print(brief.recommendation.strip())

    if brief.next_steps:
        print("\nNext steps:")
        for s in brief.next_steps:
            print(f"  - {s}")

def _print_report(report):
    print("\n=== VERIFICATION REPORT ===")
    print(f"PASSED: {report.passed}")
    print(f"Claims: {report.claim_count} | Cited: {report.cited_claim_count} | Coverage: {report.citation_coverage:.2f}")
    print(f"Sources used: {report.sources_used} (min required: {report.min_sources_required})")
    print(f"Min reliability cited: {report.min_reliability_observed:.2f} (threshold: {report.min_reliability_required:.2f})")

    if report.issues:
        print("\nIssues:")
        for iss in report.issues:
            loc = f" @ {iss.location}" if iss.location else ""
            ev = f" evidence={iss.evidence_ids}" if iss.evidence_ids else ""
            print(f"  - [{iss.severity.upper()}] {iss.code}{loc}: {iss.message}{ev}")
 

def main():
    load_dotenv()

    # --- Event logging per run ---
    run_id = str(uuid.uuid4())
    events_path = f"out/events/{run_id}.jsonl"
    sink = JsonlFileSink(events_path)  # or InMemorySink()
    emitter = Emitter(sink=sink, run_id=run_id)

    print("\n--- TIMELINE ---")
    print(f"Run ID: {run_id}")
    print(f"Events: {events_path}")

    # --- LLMs ---
    planner_llm = ChatGroq(
        model=os.getenv("PLANNER_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    )
    synth_llm = ChatGroq(
        model=os.getenv("SYNTH_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    )

    # --- Agents ---
    planner = PlannerAgent(llm=planner_llm)

    researcher = ResearcherAgent(
        web_search=web_search, 
        fetch_url=fetch_url, 
        cfg=ResearcherConfig(max_results_per_query=5, max_sources_total=5, min_reliability=0.4)
    )

    synthesizer = SynthesizerAgent(llm=synth_llm)

    verifier = VerifierAgent(cfg=VerifierConfig(min_reliability_required=0.5))

    # --- Graph ---
    app = build_graph(planner, researcher, synthesizer, verifier)

    # --- Run ---
    question = "Should we enter the SMB payroll market?"
    state = {"question": question, "budget_usd": 2.5, "time_limit_s": 180}

    out = app.invoke(state, config={"configurable": {"emitter": emitter}})

    paths = make_run_paths(run_id)

    md = build_markdown_report(
        run_id=run_id,
        question=question,
        budget_usd=float(state["budget_usd"]),
        time_limit_s=int(state["time_limit_s"]),
        plan=out["plan"],
        evidence=out["evidence"],
        brief=out["brief"],
        report=out["report"],
        events_path=events_path,
    )

    paths.report_md.write_text(md, encoding="utf-8")

    # Save machine-readable JSON summary
    summary = {
        "run_id": run_id,
        "question": question,
        "budget_usd": float(state["budget_usd"]),
        "time_limit_s": int(state["time_limit_s"]),
        "plan": out["plan"].model_dump(),
        "evidence": [e.model_dump() for e in out["evidence"]],
        "brief": out["brief"].model_dump(),
        "verification_report": out["report"].model_dump(),
        "events_path": events_path,
    }
    paths.report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


    # --- Output ---
    _print_plan(out["plan"])
    _print_evidence(out["evidence"])
    _print_brief(out["brief"])
    _print_report(out["report"])

    print(f"\nSaved report: {paths.report_md}")
    print(f"Saved summary: {paths.report_json}")
    print("\nDone.")
    

if __name__ == "__main__":
    main()