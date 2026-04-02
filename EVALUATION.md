# Evaluation Framework Documentation

## Overview

This project includes a comprehensive **evaluation framework** for assessing the quality and efficiency of the multi-agent research orchestrator. The framework provides quantitative metrics across multiple dimensions to ensure consistent, measurable improvements to the system.

## Why This Evaluation Method?

For a deterministic multi-agent research system, the best evaluation approach combines:

1. **Benchmark-Based Testing**: Standardized test cases with defined expected outcomes
2. **Multi-Dimensional Metrics**: Evaluates quality across multiple aspects (not just binary pass/fail)
3. **Evidence-Based Scoring**: Focuses on whether claims are properly supported
4. **Efficiency Tracking**: Monitors cost, time, and resource usage
5. **Reproducibility**: All evaluations can be logged and compared over time

This approach is superior to simple pass/fail testing because:
- It catches partial failures (e.g., 70% of claims are cited)
- It balances quality vs. efficiency trade-offs
- It provides actionable feedback for improvement
- It enables performance regression detection

## Architecture

The evaluation framework consists of four main components:

### 1. **Benchmark Dataset** (`data/benchmark_dataset.jsonl`)

A curated collection of research questions with associated metadata:

```json
{
  "id": "bench_001",
  "question": "What are the key considerations for enterprise adoption of generative AI?",
  "expected_aspects": ["Risk management", "Cost implications", "Integration challenges", "Workforce impact", "Regulatory compliance"],
  "difficulty": "medium",
  "category": "AI Enterprise",
  "tags": ["strategy", "governance"]
}
```

**Current Benchmarks:**
- 5 test cases across 5 categories
- Difficulty levels: Easy, Medium, Hard
- Topics: Enterprise AI, SMB AI, Governance, Trust, Security

**List available benchmarks:**
   ```bash
   python evaluate_system.py --list
   ```

**Adding New Benchmarks**
Add new JSON lines to `data/benchmark_dataset.jsonl` with your test cases.

```json
{"id": "bench_006", "question": "Your question?", "expected_aspects": ["Aspect 1", "Aspect 2"], "difficulty": "medium", "category": "Your Category"}
```

### 2. **Evaluation Metrics** (`src/engine/evaluation/metrics.py`)

Implements four key metric categories:

#### **Citation Metrics (35% weight)**
- **Citation Recall**: Percentage of claims with at least one citation
- **Orphaned Citations**: Number of citations to non-existent evidence
- **Unique Sources**: Count of distinct evidence sources cited
- **Avg Citations per Claim**: Support depth for each claim

**Scoring:**
- Recall is weighted 80%, orphaned citations penalize by 10% each
- Ideal: 1.0 (100% recall, no orphaned citations)

#### **Completeness Metrics (30% weight)**
- **Aspect Coverage**: Percentage of expected aspects covered in the response
- **Missing Aspects**: Which important topics were not addressed
- **Covered Aspects**: Which aspects were successfully discussed

**Scoring:**
- Direct ratio of covered vs. expected aspects
- Ideal: 1.0 (all expected aspects covered)

#### **Coherence Metrics (20% weight)**
- **Key Findings Count**: Number of distinct findings (expect ~5)
- **Risk Count**: Number of risks identified
- **Complete Disclosure**: Has limitations, assumptions documented?
- **Confidence Scores**: Average confidence of claims

**Scoring:**
- Finding count: 0-5 findings maps to 0.0-1.0 score
- Documentation: Bonus for including limitations
- Confidence: Higher confidence claims score higher
- Ideal: ~5 findings, confidence > 0.8, limitations disclosed

#### **Efficiency Metrics (15% weight)**
- **Total Cost**: USD spent on LLM calls
- **Total Time**: Execution duration in seconds
- **Total Tokens**: LLM input/output tokens used
- **Tool Calls**: Number of web searches and URL fetches

**Scoring:**
- Cost: < $0.05 = 1.0, > $0.50 = 0.0
- Time: < 30s = 1.0, > 300s = 0.0
- Weighted 60% cost, 40% time
- Ideal: < $0.05 cost, < 30s execution

#### **Overall Score Calculation**
```
overall = (
  citation_score * 0.35 +
  completeness_score * 0.30 +
  coherence_score * 0.20 +
  efficiency_score * 0.15
) + verification_bonus
```

Score range: 0.0 to 1.0 (higher is better)

### 3. **Evaluation Tests** (`src/tests/test_evaluation.py`)

Comprehensive test suite validating the metrics implementation:

- Citation metrics calculation test
- Completeness detection test
- Coherence scoring test
- Overall score weighting test
- End-to-end integration test
- Serialization test

**Run tests:**
```bash
pytest src/tests/test_evaluation.py -v
```
### Interpreting Results

Each evaluation produces an `EvaluationScore` with:

```python
{
    "run_id": "bench_001",
    "question": "What are the key considerations...",
    "overall_score": 0.82,           # Main metric: 0-1
    "citation_score": 0.85,          # Citations well-supported?
    "completeness_score": 0.80,      # Aspects covered?
    "coherence_score": 0.88,         # Well-structured?
    "efficiency_score": 0.75,        # Cost-effective?
    "passed_verification": true,     # Verification passed?
    "issues": [
        "Only 70% of claims are cited",
        "Missing expected aspect X"
    ]
}
```

**Score Interpretation:**
- **0.90-1.00**: Excellent quality, production-ready
- **0.80-0.89**: Good quality, minor improvements needed
- **0.70-0.79**: Acceptable, improvements recommended
- **0.60-0.69**: Poor quality, significant issues
- **< 0.60**: Unacceptable, fundamental issues

### 4. **Benchmark Runner** (`src/engine/evaluation/runner.py`)

The `BenchmarkRunner` class orchestrates evaluation execution:

```python
from engine.evaluation import BenchmarkRunner

# Create runner
runner = BenchmarkRunner(
    benchmark_file="data/benchmark_dataset.jsonl",
    output_dir="out/evaluation_runs",
    budget_usd=1.0,
    time_limit_s=300
)

# Run all benchmarks
results = runner.run_benchmark(verbose=True)

# Or run specific subset
results = runner.run_benchmark(
    subset=['AI Enterprise', 'AI Governance']
)

# Or run single benchmark
results = runner.run_benchmark(
    benchmark_id='bench_001'
)

# Generate report
report = runner.generate_report(results)

# Save results
results_file = runner.save_results(results)
```

## Files Structure

```
.
├── data/
│   └── benchmark_dataset.jsonl          # Benchmark test cases
├── src/engine/evaluation/
│   ├── __init__.py                      # Package exports
│   ├── metrics.py                       # Core metrics implementation
│   └── runner.py                        # Benchmark execution engine
├── src/tests/
│   └── test_evaluation.py               # Evaluation test suite
├── evaluate_system.py                   # CLI evaluation tool
└── EVALUATION.md                        # This file
```

## Limitations and Future Work

### Current Limitations
1. Aspect coverage uses simple keyword matching (not semantic matching)
2. No evaluation of answer factuality against external sources
3. No evaluation of writing quality or clarity

### Future Enhancements
1. **Semantic Aspect Matching**: Use embeddings to match aspects semantically
2. **Factuality Verification**: Query external fact-checking APIs
3. **Human Evaluation**: Include human raters for subjective quality
4. **Custom Scorers**: Allow domain-specific scoring functions
5. **Longitudinal Tracking**: Monitor performance trends over time
6. **Comparative Analysis**: Compare different model versions