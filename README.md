# Deterministic Multi-Agent Workflow Orchestrator
A tool-first, evaluated, budget-aware runtime for deterministic agent workflows.

## Configuration

The orchestrator uses a YAML configuration file to specify workflow parameters. By default, it looks for `config.yaml` in the project root.

### Basic Usage

```bash
# Run with default config.yaml
python src/engine/run_flow.py

# Run with custom config file
python src/engine/run_flow.py -c my_config.yaml
```

### Configuration File Format

Create a `config.yaml` file with the following structure:

```yaml
# Research question to analyze
question: "Should we enter the SMB payroll market?"

# Budget and time constraints
budget_usd: 2.5
time_limit_seconds: 180

# LLM Configuration
llm:
  planner_model: "llama-3.3-70b-versatile"
  synthesizer_model: "llama-3.3-70b-versatile"
  temperature: 0

# Researcher Agent Configuration
researcher:
  max_results_per_query: 5
  max_sources_total: 5
  min_reliability: 0.4

# Verifier Agent Configuration
verifier:
  min_reliability_required: 0.5

# Workflow Settings
workflow:
  max_iterations: 10
  synthesizer_mode: "normal"
```

### Configuration Parameters

- **question**: The research question to analyze
- **budget_usd**: Maximum budget in USD for LLM calls
- **time_limit_seconds**: Maximum time limit for the workflow
- **llm**: LLM model settings for planner and synthesizer agents
- **researcher**: Settings for the research agent (search limits, quality thresholds)
- **verifier**: Settings for the verification agent (reliability requirements)
- **workflow**: General workflow settings (iterations, synthesis mode)

See `config.example.yaml` for a complete example configuration file.
