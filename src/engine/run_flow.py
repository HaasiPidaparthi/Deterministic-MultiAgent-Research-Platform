"""
Deterministic Multi-Agent Workflow Orchestrator

This module provides the main entry point for running the multi-agent workflow
that researches, analyzes, and verifies business questions using a coordinated
team of AI agents.
"""

import os
import uuid
import json
import yaml
import argparse
from typing import Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from engine.graph.flow_loop import build_graph
from engine.agents.planner import PlannerAgent
from engine.agents.researcher import ResearcherAgent, ResearcherConfig
from engine.agents.synthesizer import SynthesizerAgent
from engine.agents.verifier import VerifierAgent, VerifierConfig
from engine.tools.web_search import web_search
from engine.tools.web_fetch import fetch_url
from engine.events.emitter import Emitter
from engine.events.sink import InMemorySink, JsonlFileSink
from engine.reporting.run_report import make_run_paths, build_markdown_report
from engine.reporting.dashboard import build_workflow_dashboard, make_dashboard_paths as make_dashboard_paths_dashboard
from engine.tools.rag import RAGConfig


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration YAML file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {config_path}")

    return config


def _print_plan(plan):
    """
    Print a formatted summary of the research plan.

    Args:
        plan: ResearchPlan object containing subquestions and search queries
    """
    print("\n=== PLAN ===")
    print(f"Subquestions: {len(plan.subquestions)}")
    for i, sq in enumerate(plan.subquestions[:6]):
        q = getattr(sq, "question", str(sq))
        print(f"  {i+1}. {q}")
    print(f"\nSearch queries: {len(plan.search_queries)}")
    for q in plan.search_queries[:8]:
        print(f"  - {q}")

def _print_evidence(evidence):
    """
    Print a formatted summary of the collected evidence.

    Args:
        evidence: List of EvidenceItem objects
    """
    print("\n=== EVIDENCE (top) ===")
    for e in evidence[:8]:
        print(f"{e.id} | rel={e.reliability_score:.2f} rev={e.relevance_score:.2f} | {e.title or ''}")
        print(f"    {e.url}")
        if e.snippet:
            print(f"    {e.snippet[:180].strip()}")
        print()

def _print_brief(brief):
    """
    Print a formatted summary of the synthesized brief.

    Args:
        brief: BriefDraft object containing the analysis results
    """
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
    """
    Print a formatted summary of the verification report.

    Args:
        report: VerificationReport object containing verification results
    """
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
 

def main(config_path: str = "config.yaml", override_question: Optional[str] = None):
    """
    Main entry point for the multi-agent workflow orchestrator.

    This function loads configuration from a YAML file and sets up the complete workflow including:
    - Event logging and monitoring
    - LLM configuration for different agents
    - Agent initialization with appropriate tools and configs
    - Graph construction and execution
    - Result reporting and output formatting

    Args:
        config_path: Path to the YAML configuration file
        override_question: Optional new research question passed at runtime.

    The workflow follows this sequence:
    1. Planner: Breaks down the question into subquestions and search queries
    2. Researcher: Searches and fetches relevant evidence
    3. Synthesizer: Analyzes evidence and creates a comprehensive brief
    4. Verifier: Validates the brief for accuracy and completeness

    Results are saved to both markdown and JSON formats for review.
    """
    load_dotenv()

    # Load configuration
    config = load_config(config_path)
    if override_question:
        config["question"] = override_question.strip()

    # --- Event logging per run ---
    run_id = str(uuid.uuid4())
    events_path = f"out/events/{run_id}.jsonl"
    sink = JsonlFileSink(events_path)  # or InMemorySink()
    emitter = Emitter(sink=sink, run_id=run_id)

    question = config["question"]

    print("\n=== TIMELINE ===")
    print(f"Run ID: {run_id}")
    print(f"Events: {events_path}")
    print(f"Config: {os.path.abspath('config.yaml')}")

    print(f"\nQuestion: {question}")

    # --- LLMs ---
    planner_llm = ChatGroq(
        model=os.getenv("PLANNER_MODEL", config["llm"]["planner_model"]),
        temperature=config["llm"]["temperature"],
    )
    synth_llm = ChatGroq(
        model=os.getenv("SYNTH_MODEL", config["llm"]["synthesizer_model"]),
        temperature=config["llm"]["temperature"],
    )

    # --- Agents ---
    planner = PlannerAgent(llm=planner_llm)

    researcher_search_mode = config["researcher"].get("search_mode")
    if researcher_search_mode is None:
        researcher_search_mode = "both" if config["researcher"].get("enable_rag", True) else "web"

    researcher = ResearcherAgent(
        web_search=web_search, 
        fetch_url=fetch_url, 
        cfg=ResearcherConfig(
            max_results_per_query=config["researcher"]["max_results_per_query"],
            max_sources_total=config["researcher"]["max_sources_total"],
            min_reliability=config["researcher"]["min_reliability"],
            enable_rag=config["researcher"].get("enable_rag", True),
            search_mode=researcher_search_mode,
            rag_config=RAGConfig(
                collection_name=config["researcher"].get("rag", {}).get("collection_name", "research_knowledge_base"),
                embedding_model=config["researcher"].get("rag", {}).get("embedding_model", "nomic-embed-text"),
                persist_directory=config["researcher"].get("rag", {}).get("persist_directory", "./data/chroma_db"),
                similarity_threshold=config["researcher"].get("rag", {}).get("similarity_threshold", 0.7),
                max_results=config["researcher"].get("rag", {}).get("max_results", 5),
                min_relevance=config["researcher"].get("rag", {}).get("min_relevance", 0.3),
            )
        )
    )

    synthesizer = SynthesizerAgent(llm=synth_llm)

    verifier = VerifierAgent(
        cfg=VerifierConfig(
            min_reliability_required=config["verifier"]["min_reliability_required"]
        )
    )

    # --- Graph ---
    app = build_graph(planner, researcher, synthesizer, verifier)

    # --- Run ---
    question = config["question"]
    state = {
        "question": question, 
        "budget_usd": config["budget_usd"], 
        "time_limit_s": config["time_limit_seconds"],
        "iter": 0,
        "researcher_overrides": {},  # start with no overrides
        "synthesizer_mode": config["workflow"]["synthesizer_mode"],  # start in configured mode
    }

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
        metrics=out.get("metrics", {}),
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

    dashboard_paths = make_dashboard_paths_dashboard(run_id)
    dashboard_html = build_workflow_dashboard(
        run_id=run_id,
        question=question,
        metrics=out.get("metrics", {}),
        time_limit_s=int(state["time_limit_s"]),
        budget_usd=float(state["budget_usd"]),
    )
    dashboard_paths.dashboard_html.write_text(dashboard_html, encoding="utf-8")


    # --- Output ---
    _print_plan(out["plan"])
    _print_evidence(out["evidence"])
    _print_brief(out["brief"])
    _print_report(out["report"])

    print(f"\nSaved report: {paths.report_md}")
    print(f"Saved summary: {paths.report_json}")
    print(f"Saved dashboard: {dashboard_paths.dashboard_html}")
    print("\nDone.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deterministic Multi-Agent Workflow Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/engine/run_flow.py                    # Use default config.yaml
  python src/engine/run_flow.py -c my_config.yaml  # Use custom config file
        """
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)"
    )

    parser.add_argument(
        "--question",
        default=None,
        help="Override the configured research question with a new question."
    )

    args = parser.parse_args()
    main(config_path=args.config, override_question=args.question)