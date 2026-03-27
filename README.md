# Deterministic Multi-Agent Workflow Orchestrator

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2+-orange.svg)](https://www.langchain.com/)

A sophisticated, tool-first, budget-aware runtime engine for orchestrating deterministic multi-agent workflows. This project demonstrates advanced AI engineering principles through a coordinated team of specialized agents that research, analyze, and verify complex business questions while maintaining strict cost and time constraints.

## 🚀 Overview

The Deterministic Multi-Agent Workflow Orchestrator is designed to automate complex research and decision-making processes using a collaborative AI agent architecture. By leveraging specialized agents (Planner, Researcher, Synthesizer, and Verifier), the system provides reliable, cost-effective solutions to business intelligence questions with full traceability and evaluation metrics.

### Key Capabilities
- **Deterministic Workflows**: Ensures reproducible results through structured agent interactions
- **Budget Awareness**: Real-time cost tracking and enforcement for LLM API calls
- **Multi-Agent Coordination**: Specialized agents working in harmony for comprehensive analysis
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

## 📁 Project Structure

```
deterministic-multi-agent-engine/
├── src/
│   └── engine/
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

---

**Built with ❤️ using Python, LangChain, and modern AI engineering practices.**
