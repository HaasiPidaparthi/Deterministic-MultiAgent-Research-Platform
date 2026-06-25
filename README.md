<div align="center">
<img src="https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python" alt="Python 3.11+"/>
<img src="https://img.shields.io/badge/LangChain-1.2+-orange?style=flat-square" alt="LangChain"/>
<img src="https://img.shields.io/badge/LLM-Groq%20%7C%20OpenAI-8A2BE2?style=flat-square" alt="LLM Providers"/>
<img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-teal?style=flat-square" alt="ChromaDB"/>

# Cerebra: Multi-Agent Research Platform

**A budget-aware, DAG-orchestrated multi-agent engine that researches, synthesizes, and verifies complex business questions — with full cost tracking, RAG integration, and structured evaluation.**

[Quickstart](#️-quickstart) · [Architecture](#️-architecture) · [Benchmarks](#-benchmarks) · [Example Output](#-example-output) · [Configuration](#️-configuration)

</div>

---

## 🚀 Overview

Most LLM research demos are single-prompt wrappers. Cerebra is a **production-style runtime engine**: it decomposes a question into a DAG of tasks, dispatches specialized agents, enforces token budgets in real time, falls back to web search when the internal knowledge base falls short, and verifies answers before returning them.

Built to demonstrate the engineering disciplines required in real AI systems. 

## ✨ Key features

| Capability | Detail |
|---|---|
| **Multi-agent DAG orchestration** | Planner → Researcher → Synthesizer → Verifier pipeline with retry policies and convergence checks |
| **Hybrid RAG** | Dense vector search (ChromaDB) + BM25 sparse retrieval; falls back to live web search when confidence is low |
| **Budget enforcement** | Hard token + USD cost caps per run — enforced at agent level, not just logged after the fact |
| **Structured evaluation** | `EVALUATION.md` and `evaluate_system.py` ship with the repo; 90–95% task completion in benchmark suite |
| **Full observability** | Every agent action emits structured events → `events.jsonl`; `metrics.json` captures latency, cost, and reliability scores per run |
| **Configurable via YAML** | Swap models, adjust search depth, tune reliability thresholds — no code changes needed |

## 🏗️ Architecture

```
User question
      │
      ▼
┌─────────────┐     decomposes into tasks
│   Planner   │ ──────────────────────────────────────────┐
└─────────────┘                                           │
                                                          ▼
                                              ┌────────────────────┐
                                              │  Task DAG (graph/) │
                                              └────────────────────┘
                                                          │
                          ┌───────────────────────────────┤
                          ▼                               ▼
               ┌──────────────────┐            ┌──────────────────┐
               │   Researcher     │            │   Researcher     │  (parallel tasks)
               │  RAG → Web srch  │            │  RAG → Web srch  │
               └────────┬─────────┘            └────────┬─────────┘
                        │                               │
                        └──────────────┬────────────────┘
                                       ▼
                            ┌─────────────────────┐
                            │    Synthesizer      │
                            │  merges + generates │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │     Verifier        │
                            │  reliability gate   │
                            └──────────┬──────────┘
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                   report.md                   events.jsonl
                   (answer)                    metrics.json
```

### Agent responsibilities
 
**Planner** — takes the raw question and emits a list of research sub-tasks, each with priority and dependency constraints. Uses a structured output schema (`schemas/planner.py`) so the DAG engine can parse it deterministically.
 
**Researcher** — executes one sub-task. Queries ChromaDB first (semantic similarity ≥ 0.7 threshold); if confidence is below `min_reliability`, falls back to Tavily web search. Evidence items are scored and deduplicated.
 
**Synthesizer** — receives all evidence briefs and generates a comprehensive answer. Supports two modes: `normal` (structured synthesis) and `fast` (single-pass, lower cost).
 
**Verifier** — checks the synthesized answer against minimum reliability and source coverage thresholds. If it fails, the workflow loops back to the Researcher with a gap-fill prompt — up to `max_iterations`.

### Loop controller & retry policy
 
The `graph/loop_controller.py` manages the feedback loop between Verifier and Researcher. It uses exponential backoff on repeated failures and a convergence check to break infinite loops.
 
---

## 📊 Benchmarks
 
Evaluated on a 50-question business intelligence benchmark suite (see `EVALUATION.md` for methodology and full question set):
 
| Metric | Result |
|---|---|
| Task completion rate | **90–95%** |
| Average run cost (Groq / llama-3.3-70b) | **< $0.03 per question** |
| Average end-to-end latency | **< 45 seconds** |
| Verifier pass rate (first attempt) | **~78%** |
| Retry convergence (≤ 3 iterations) | **~97%** of failing runs |
 
> Benchmarks run locally with Groq free-tier API. Results vary by question complexity and model choice.
 
---

## 💡 Example output
 
**Input question:**
```
"What are the high-impact use cases for AI across manufacturing supply chain resilience?"
```
 
**Output summary** *(from `out/reports/run_abc123/report.md`)*:
```
Answer: Five high-impact areas identified: (1) demand forecasting with LSTM models,
(2) supplier risk scoring via NLP on earnings calls, (3) visual defect detection
at inspection points, (4) logistics route optimization, (5) predictive maintenance
on CNC equipment. Sources: McKinsey 2024 Operations Report, NIST AI RMF, MIT
Sloan supply chain review. Reliability score: 0.87. Cost: $0.021. Elapsed: 38s.
```

Each run also produces a full `events.jsonl` trace — every agent action, tool call, and retry — making it straightforward to debug, audit, or feed into downstream analysis.

---

## ⚡️ Quickstart
 
### Prerequisites
- Python 3.11+
- Conda (recommended)
- API keys: at minimum `GROQ_API_KEY` (free tier works); optionally `OPENAI_API_KEY` and `TAVILY_API_KEY` for web search

## Setup
 
```bash
# 1. Clone
git clone https://github.com/HaasiPidaparthi/Cerebra-Multi-Agent-Research-Platform.git
cd Cerebra-Multi-Agent-Research-Platform
 
# 2. Create environment
conda env create -f environment.yml
conda activate agents
 
# 3. Install
pip install -e .
 
# 4. Configure secrets
cp .env.example .env
# edit .env — add your GROQ_API_KEY at minimum
 
# 5. Configure run parameters
cp config.example.yaml config.yaml
# edit config.yaml — set your question, budget, and model preferences
```

### Run
 
```bash
# Ask a question (edit config.yaml first)
python src/engine/run_flow.py
 
# Or pass a config directly
python src/engine/run_flow.py -c my_config.yaml
```
 
Output lands in `out/reports/{run_id}/`: a `report.md` answer, `events.jsonl` trace, and `metrics.json`.
 
### Populate the RAG knowledge base (optional)
 
```bash
# Load curated AI business use-case sources (government + org references)
python populate_rag.py --usecases
 
# Add your own files
python populate_rag.py --files research_docs/*.pdf
 
# Add web pages
python populate_rag.py --urls "https://example.com/doc"
 
# Check what's loaded
python populate_rag.py --stats
```
 
---

## ⚙️ Configuration
 
Key fields in `config.yaml`:
 
```yaml
question: "Should we enter the SMB payroll market?"
 
budget_usd: 2.50          # hard cost cap — run aborts if exceeded
time_limit_seconds: 180   # hard time cap
 
llm:
  planner_model: "llama-3.3-70b-versatile"
  synthesizer_model: "llama-3.3-70b-versatile"
  temperature: 0           # deterministic by default
 
researcher:
  enable_rag: true
  search_mode: "both"      # "rag" | "web" | "both"
  max_sources_total: 5
  min_reliability: 0.4
 
verifier:
  min_reliability_required: 0.5
 
workflow:
  max_iterations: 10
  synthesizer_mode: "normal"   # "normal" | "fast"
```
 
---

## 📁 Project structure
 
```
cerebra/
├── src/
│   └── engine/
│       ├── agents/           # planner · researcher · synthesizer · verifier
│       ├── events/           # event emitter, models, sinks
│       ├── graph/            # DAG flow loop, loop controller, retry policy, state
│       ├── metrics/          # LLM usage tracker, run metrics
│       ├── reporting/        # Markdown report builder, event reporter
│       ├── schemas/          # Pydantic models: briefs, evidence, plans, verify
│       ├── tools/            # web_fetch, web_search, content extractor
│       └── run_flow.py       # entrypoint
│   └── tests/
├── data/
│   └── ai_business_use_cases.json   # curated RAG seed sources
├── EVALUATION.md             # benchmark methodology + question set
├── evaluate_system.py        # run the evaluation suite
├── populate_rag.py           # RAG ingestion CLI
├── config.example.yaml
├── environment.yml
└── pyproject.toml
```
 
---

## 🧪 Testing & evaluation
 
```bash
# Unit + integration tests
pytest src/tests/
 
# Full benchmark evaluation (runs 50 questions, writes results to out/eval/)
python evaluate_system.py
```

The evaluation script computes task completion rate, average cost, and per-question reliability scores. Results are written to `out/eval/summary.json` and `out/eval/results.jsonl`.

---

## 🗺️ Roadmap
 
- [ ] LangGraph migration (replace custom DAG with LangGraph state machine)
- [ ] Streaming output support (token-by-token via FastAPI + SSE)
- [ ] HuggingFace Spaces demo (no API key required to try)
- [ ] RAGAS integration for automated RAG quality scoring
- [ ] Async agent execution (parallel Researcher tasks)
