# Deterministic Multi-Agent Workflow Orchestrator

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2+-orange.svg)](https://www.langchain.com/)

A sophisticated, tool-first, budget-aware runtime engine for orchestrating deterministic multi-agent workflows. This project demonstrates advanced AI engineering principles through a coordinated team of specialized agents that research, analyze, and verify complex business questions while maintaining strict cost and time constraints.

## 🚀 Overview

The Deterministic Multi-Agent Workflow Orchestrator is designed to automate complex research and decision-making processes using a collaborative AI agent architecture. By leveraging specialized agents (Planner, Researcher, Synthesizer, and Verifier), the system provides reliable, cost-effective solutions to business intelligence questions with full traceability and evaluation metrics.

### Key Capabilities
- **Multi-Agent Architecture**: Specialized agents for planning, research, synthesis, and verification
- **RAG Integration**: Vector database search before web search for internal knowledge bases
- **Deterministic Workflows**: Ensures reproducible results through structured agent interactions
- **Cost Awareness**: Real-time cost tracking and enforcement for LLM API calls
- **Tool Integration**: Web search and content fetching capabilities for data-driven insights
- **Comprehensive Reporting**: Detailed execution reports with metrics and event logs
- **Configurable Architecture**: Flexible YAML-based configuration for various use cases

## 🏗️ Architecture

The system follows a modular, event-driven architecture built on LangChain and modern Python practices:

### Core Components

#### Agents
- **Planner Agent**: Decomposes complex questions into actionable research tasks
- **Researcher Agent**: Conducts web searches and gathers relevant information
- **Synthesizer Agent**: Analyzes collected data and generates comprehensive answers
- **Verifier Agent**: Validates findings against reliability thresholds

#### Workflow Engine
- **Graph-Based Execution**: Directed acyclic graph (DAG) for deterministic flow control
- **Loop Controller**: Manages iteration logic with retry policies and convergence checks
- **State Management**: Persistent state tracking across workflow executions

#### Tools & Integration
- **Web Search**: Integrated search capabilities for information gathering
- **Web Fetch**: Content extraction from URLs with reliability scoring
- **Event System**: Comprehensive event emission and logging for observability

#### Monitoring & Reporting
- **Metrics Collection**: LLM usage tracking, execution times, and performance metrics
- **Run Reports**: Markdown-based reports with detailed execution summaries
- **Event Sinks**: In-memory and file-based event storage for analysis

## 🔍 RAG (Retrieval-Augmented Generation)

The orchestrator includes RAG functionality to search internal knowledge bases before falling back to web search. This improves efficiency and allows incorporation of proprietary or internal documents.

### Setting up RAG

1. **Install dependencies** (already included in pyproject.toml):
   ```bash
   pip install -e .
   ```

2. **Populate the vector database** with your documents:
   ```bash
   # Add curated AI business use case sources
   python populate_rag.py --usecases

   # Add local files
   python populate_rag.py --files research_docs/*.txt internal_reports/*.pdf

   # Add web pages (requires TAVILY_API_KEY)
   python populate_rag.py --urls "https://example.com/internal-doc1"

   # Check database stats
   python populate_rag.py --stats
   ```

3. **Configure RAG** in your `config.yaml`:
   ```yaml
   researcher:
     enable_rag: true
     rag:
       collection_name: "research_knowledge_base"
       embedding_model: "nomic-embed-text"
       persist_directory: "./data/chroma_db"
       similarity_threshold: 0.7
       max_results: 5
       min_relevance: 0.3
   ```

### RAG Workflow

1. **Query Analysis**: Agent analyzes the research question
2. **RAG Search**: Searches vector database for relevant internal documents
3. **Web Search**: If insufficient results from RAG, performs web search
4. **Evidence Integration**: Combines and deduplicates evidence from both sources
5. **Synthesis**: Creates comprehensive brief using all available evidence

## 💼 Example Use Cases: AI Business & Industry

The orchestrator is designed for business and industry-focused AI research. Use the RAG pipeline to augment answers with authoritative government and non-profit knowledge sources.

Example use cases:
- **AI Governance and Trust**: Ingest federal policy guidance and AI standards for public-sector modernization and regulated-industry planning.
- **Enterprise AI Risk Management**: Use NIST risk management guidance to support reliable AI adoption across operations and critical infrastructure.
- **AI Resilience for Cybersecurity and Supply Chain**: Use CISA and infrastructure guidance to frame risk-aware AI deployment and supply chain resilience.
- **AI Adoption for Small Business Intelligence**: Support market research, competitive analysis, and digital transformation recommendations for SMBs.
- **AI Business Transformation Strategy**: Augment executive strategy with industry-readiness analysis from trusted .org sources.

### Load curated AI business use case sources

Run the following command to populate the RAG database with curated summaries and source links for current AI business topics:

```bash
python populate_rag.py --usecases
```

The source metadata is loaded from `data/ai_business_use_cases.json`, which contains a curated list of authoritative .org and .gov references. You can extend or update the use case list by editing that file directly.

### Testing RAG

Run the test script to verify RAG functionality:
```bash
python test_rag.py
```

This demonstrates semantic search across the vector database with relevance scoring.

## 📁 Project Structure

```
deterministic-multi-agent-engine/
├── src/
│   ├── engine/
│   │    ├── agents/                # Specialized AI agents
│   │    │   ├── planner.py         # Task decomposition agent
│   │    │   ├── researcher.py      # Information gathering agent
│   │    │   ├── synthesizer.py     # Analysis and synthesis agent
│   │    │   └── verifier.py        # Validation and verification agent
│   │    ├── events/                # Event-driven architecture
│   │    │   ├── emitter.py         # Event emission system
│   │    │   ├── models.py          # Event data models
│   │    │   └── sink.py            # Event storage and handling
│   │    ├── graph/                 # Workflow execution engine
│   │    │   ├── flow_loop.py       # Main workflow orchestration
│   │    │   ├── loop_controller.py # Iteration control
│   │    │   ├── nodes.py           # Graph node definitions
│   │    │   ├── retry_policy.py    # Error handling strategies
│   │    │   └── state.py           # Workflow state management
│   │    ├── metrics/               # Performance monitoring
│   │    │   ├── llm_usage.py       # API cost tracking
│   │    │   └── run_metrics.py     # Execution metrics
│   │    ├── reporting/             # Output generation
│   │    │   ├── events.py          # Event-based reporting
│   │    │   └── run_report.py      # Markdown report builder
│   │    ├── schemas/               # Data models and validation
│   │    │   ├── brief.py           # Task briefing schemas
│   │    │   ├── evidence.py        # Evidence collection schemas
│   │    │   ├── planner.py         # Planning data structures
│   │    │   └── verify.py          # Verification schemas
│   │    ├── tools/                 # External integrations
│   │    │   ├── extract.py         # Content extraction utilities
│   │    │   ├── web_fetch.py       # URL content fetching
│   │    │   ├── web_search.py      # Search engine integration
│   │    │   └── web_types.py       # Web-related type definitions
│   │    └── run_flow.py            # Main application entry point
│   └── tests/                      # Comprehensive test suite
├── config.example.yaml             # Configuration template
├── pyproject.toml                  # Python project configuration
├── environment.yml                 # Conda environment specification
└── pytest.ini                      # Test configuration
```

## 🛠️ Installation

### Prerequisites
- Python 3.11 or higher
- Conda (recommended for environment management)
- API keys for LLM providers (Groq, OpenAI, etc.)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/deterministic-multi-agent-workflow-orchestrator.git
   cd deterministic-multi-agent-workflow-orchestrator
   ```

2. **Create and activate the Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate agents
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

4. **Configure environment variables:**
   Create a `.env` file in the project root with your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

5. **Set up configuration:**
   Copy the example configuration:
   ```bash
   cp config.example.yaml config.yaml
   ```
   Edit `config.yaml` with your desired parameters.

## 🚀 Usage

### Basic Execution

Run the orchestrator with the default configuration:

```bash
python src/engine/run_flow.py
```

Run with a custom configuration file:

```bash
python src/engine/run_flow.py -c my_custom_config.yaml
```

### Example Research Questions

Try these business-focused prompts to generate reports:
- "What are the top AI investment opportunities in SMB customer support automation?"
- "How should a financial services company prioritize AI risk controls for a new regulatory reporting system?"
- "What are the high-impact use cases for AI across manufacturing supply chain resilience?"
- "How can a healthcare provider deploy AI safely for clinical decision support while meeting compliance requirements?"
- "What internal capabilities and governance should an enterprise build before launching a generative AI product?"
- "What are the market entry risks and revenue potential for an AI-driven payroll solution in North America?"

### Configuration

The system uses YAML configuration files to define workflow parameters. Key configuration sections:

#### Research Question
```yaml
question: "Should we enter the SMB payroll market?"
```

#### Resource Constraints
```yaml
budget_usd: 2.5          # Maximum budget in USD
time_limit_seconds: 180  # Time limit for execution
```

#### LLM Configuration
```yaml
llm:
  planner_model: "llama-3.3-70b-versatile"
  synthesizer_model: "llama-3.3-70b-versatile"
  temperature: 0
```

#### Agent-Specific Settings
```yaml
researcher:
  max_results_per_query: 5
  max_sources_total: 5
  min_reliability: 0.4
  search_mode: "both"  # Options: "rag", "web", or "both"

verifier:
  min_reliability_required: 0.5

workflow:
  max_iterations: 10
  synthesizer_mode: "normal"
```

### Output

Execution generates:
- **Console Output**: Real-time progress and results
- **Report Directory**: `out/reports/{run_id}/` containing:
  - `report.md`: Comprehensive markdown report
  - `events.jsonl`: Complete event log
  - `metrics.json`: Performance metrics

## 🧪 Testing

Run the test suite to ensure everything is working:

```bash
pytest src/tests/
```

## 📊 Performance & Metrics

The orchestrator tracks:
- **LLM Usage**: Token counts and API costs by provider
- **Execution Time**: Total runtime and per-agent timing
- **Reliability Scores**: Source credibility and answer confidence
- **Workflow Efficiency**: Iteration counts and convergence metrics

## 🔧 Advanced Configuration

### Custom Agent Behaviors
Modify agent configurations in `config.yaml` to adjust:
- Search depth and breadth
- Synthesis strategies
- Verification thresholds
- Retry policies

### Event System
The event-driven architecture allows for:
- Real-time monitoring
- Custom event handlers
- Workflow visualization
- Performance analysis

## 📈 Use Cases

Ideal for:
- **Business Intelligence**: Market research and competitive analysis
- **Decision Support**: Complex decision-making with evidence-based answers
- **Research Automation**: Systematic information gathering and synthesis
- **Cost-Effective AI**: Budget-constrained AI workflows with guaranteed outcomes
