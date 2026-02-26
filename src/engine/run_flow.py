import os
from dotenv import load_dotenv
from pathlib import Path

from engine.graph.flow import build_graph
from engine.agents.planner import PlannerAgent

from engine.agents.researcher import ResearcherAgent, ResearcherConfig
from engine.tools.web_search import web_search
from engine.tools.fetch import fetch_url

from langchain_groq import ChatGroq

def main():
    load_dotenv()

    # --- Planner LLM ---
    llm = ChatGroq(
        model=os.getenv("PLANNER_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    )
    planner = PlannerAgent(llm=llm)

    # --- Researcher tools ---
    cfg = ResearcherConfig(max_results_per_query=5, max_sources_total=8, min_reliability=0.4)
    researcher = ResearcherAgent(web_search=web_search, fetch_url=fetch_url, cfg=cfg)

    app = build_graph(planner, researcher)

    state = {
        "question": "Should we enter the SMB payroll market?",
        "budget_usd": 2.5,
        "time_limit_s": 180,
    }

    out = app.invoke(state)
    plan = out["plan"]
    evidence = out["evidence"]

    print("\n--- RESEARCH ---")
    print(f"Question: {state['question']}"
          f"\nBudget: ${state['budget_usd']}"
          f"\nTime limit: {state['time_limit_s']} seconds")

    print("\n--- PLAN ---")
    print(f"Subquestions: {len(plan.subquestions)}")
    print(f"Search queries: {len(plan.search_queries)}")

    print("\n--- EVIDENCE ---")
    print(f"ID | Reliability | Relevance | URL")
    for e in evidence[:5]:
        print(f"{e.id} | {e.reliability_score:.2f} | {e.relevance_score:.2f} | {e.url}")


if __name__ == "__main__":
    main()