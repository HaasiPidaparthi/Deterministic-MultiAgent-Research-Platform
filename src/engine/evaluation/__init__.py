"""Evaluation framework for assessing multi-agent orchestrator performance.

This package provides comprehensive evaluation capabilities including:
- Benchmark dataset management
- Multi-dimensional evaluation metrics
- Benchmark runner and testing harness
- Results reporting and analysis
"""

from engine.evaluation.metrics import (
    EvaluationMetrics,
    EvaluationScore,
    CitationMetrics,
    CompletenessMetrics,
    CoherenceMetrics,
    EfficiencyMetrics,
    evaluate_run,
)
from engine.evaluation.runner import (
    BenchmarkRunner,
    create_evaluation_benchmark_runner,
)

__all__ = [
    "EvaluationMetrics",
    "EvaluationScore",
    "CitationMetrics",
    "CompletenessMetrics",
    "CoherenceMetrics",
    "EfficiencyMetrics",
    "evaluate_run",
    "BenchmarkRunner",
    "create_evaluation_benchmark_runner",
]
