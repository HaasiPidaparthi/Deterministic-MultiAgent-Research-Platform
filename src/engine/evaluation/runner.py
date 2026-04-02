"""Evaluation runner for executing benchmark tests and collecting results.

This module provides tools for running the multi-agent orchestrator against
a benchmark dataset and collecting detailed evaluation metrics.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict

from engine.evaluation.metrics import evaluate_run, EvaluationScore
from engine.graph.flow_loop import build_graph
from engine.agents.planner import PlannerAgent
from engine.agents.researcher import ResearcherAgent
from engine.agents.synthesizer import SynthesizerAgent
from engine.agents.verifier import VerifierAgent
from engine.events.emitter import Emitter


class BenchmarkRunner:
    """Execute benchmark tests against the multi-agent system."""
    
    def __init__(
        self,
        benchmark_file: str = "data/benchmark_dataset.jsonl",
        output_dir: str = "out/evaluation_runs",
        budget_usd: float = 1.0,
        time_limit_s: int = 300,
        max_iterations: int = 3,
        synthesizer_mode: str = "normal",
    ):
        """Initialize the benchmark runner.
        
        Args:
            benchmark_file: Path to JSONL file with benchmark cases
            output_dir: Directory to save evaluation results
            budget_usd: Default budget per query
            time_limit_s: Default time limit per query
        """
        self.benchmark_file = Path(benchmark_file)
        self.output_dir = Path(output_dir)
        self.budget_usd = budget_usd
        self.time_limit_s = time_limit_s
        self.max_iterations = max_iterations
        self.synthesizer_mode = synthesizer_mode
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load benchmark dataset
        self.benchmarks = self._load_benchmarks()
    
    def _load_benchmarks(self) -> List[Dict[str, Any]]:
        """Load benchmark dataset from JSONL file."""
        benchmarks = []
        if not self.benchmark_file.exists():
            raise FileNotFoundError(f"Benchmark file not found: {self.benchmark_file}")
        
        with open(self.benchmark_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    benchmarks.append(json.loads(line))
        
        return benchmarks
    
    def run_benchmark(
        self,
        benchmark_id: Optional[str] = None,
        subset: Optional[List[str]] = None,
        verbose: bool = True
    ) -> List[EvaluationScore]:
        """Run benchmark tests.
        
        Args:
            benchmark_id: Run specific benchmark by ID
            subset: Run subset of benchmarks by category (e.g., ["AI Enterprise"])
            verbose: Print progress information
        
        Returns:
            List of evaluation scores
        """
        # Filter benchmarks
        test_cases = self.benchmarks
        
        if benchmark_id:
            test_cases = [b for b in test_cases if b.get("id") == benchmark_id]
        elif subset:
            test_cases = [b for b in test_cases if b.get("category") in subset]
        
        if not test_cases:
            raise ValueError(f"No matching benchmarks found")
        
        results = []
        
        for i, bench in enumerate(test_cases, 1):
            if verbose:
                print(f"\n[{i}/{len(test_cases)}] Running: {bench.get('id')} - {bench['question'][:60]}...")
            
            try:
                score = self._run_single_benchmark(bench, verbose=verbose)
                results.append(score)
                
                if verbose:
                    print(f"    Overall Score: {score.overall_score:.3f} | "
                          f"Citation: {score.citation_score:.2f} | "
                          f"Completeness: {score.completeness_score:.2f}")
            
            except Exception as e:
                if verbose:
                    print(f"    ERROR: {str(e)}")
                # Continue with next benchmark
                continue
        
        return results
    
    def _run_single_benchmark(
        self,
        benchmark: Dict[str, Any],
        verbose: bool = False
    ) -> EvaluationScore:
        """Run a single benchmark case through the system."""
        bench_id = benchmark.get("id", "unknown")
        question = benchmark["question"]
        expected_aspects = benchmark.get("expected_aspects", [])
        difficulty = benchmark.get("difficulty", "medium")
        category = benchmark.get("category", "General")

        if not hasattr(self, "graph") or self.graph is None:
            raise RuntimeError("BenchmarkRunner has no graph configured. "
                               "Call create_evaluation_benchmark_runner with a graph or agents.")

        state = {
            "question": question,
            "budget_usd": float(getattr(self, "budget_usd", 0.0)),
            "time_limit_s": int(getattr(self, "time_limit_s", 0)),
            "iter": 0,
            "researcher_overrides": {},
            "synthesizer_mode": "normal",
            "workflow": {"max_iterations": getattr(self, "max_iterations", 3)},
        }

        if verbose:
            print(f"    Invoking workflow graph for benchmark: {bench_id}")

        out = self.graph.invoke(state)

        if not out or "brief" not in out or "evidence" not in out:
            raise RuntimeError("Workflow graph did not return required output fields")

        evaluation_score = evaluate_run(
            run_id=bench_id,
            question=question,
            difficulty=difficulty,
            category=category,
            brief=out.get("brief"),
            evidence=out.get("evidence", []),
            metrics=out.get("metrics", {}),
            expected_aspects=expected_aspects,
            passed_verification=getattr(out.get("report"), "passed", False),
        )

        # Save each benchmark evaluation snapshot
        eval_file = self.output_dir / f"benchmark_{bench_id}_evaluation.json"
        eval_file.write_text(json.dumps(evaluation_score.to_dict(), indent=2), encoding="utf-8")

        return evaluation_score
    
    def save_results(self, results: List[EvaluationScore]) -> Path:
        """Save evaluation results to JSON file.
        
        Args:
            results: List of evaluation scores
        
        Returns:
            Path to saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        
        # Convert to serializable format
        results_data = [r.to_dict() for r in results]
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        return results_file
    
    def generate_report(self, results: List[EvaluationScore]) -> str:
        """Generate a summary report of evaluation results.
        
        Args:
            results: List of evaluation scores
        
        Returns:
            Markdown-formatted report
        """
        if not results:
            return "No results to report."
        
        # Calculate statistics
        overall_scores = [r.overall_score for r in results]
        citation_scores = [r.citation_score for r in results]
        completeness_scores = [r.completeness_score for r in results]
        coherence_scores = [r.coherence_score for r in results]
        efficiency_scores = [r.efficiency_score for r in results]
        
        avg_overall = sum(overall_scores) / len(overall_scores)
        avg_citation = sum(citation_scores) / len(citation_scores)
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        avg_efficiency = sum(efficiency_scores) / len(efficiency_scores)
        
        total_cost = sum(r.efficiency_metrics.total_cost_usd for r in results)
        total_time = sum(r.efficiency_metrics.total_time_s for r in results)
        
        # Group by difficulty
        by_difficulty = {}
        for r in results:
            if r.difficulty not in by_difficulty:
                by_difficulty[r.difficulty] = []
            by_difficulty[r.difficulty].append(r)
        
        # Build report
        report = "# Evaluation Results Report\n\n"
        report += f"**Generated:** {datetime.now().isoformat()}\n"
        report += f"**Test Cases:** {len(results)}\n\n"
        
        # Overall metrics
        report += "## Overall Performance\n\n"
        report += f"| Metric | Score |\n"
        report += f"|--------|-------|\n"
        report += f"| Overall | {avg_overall:.3f} |\n"
        report += f"| Citation Quality | {avg_citation:.3f} |\n"
        report += f"| Completeness | {avg_completeness:.3f} |\n"
        report += f"| Coherence | {avg_coherence:.3f} |\n"
        report += f"| Efficiency | {avg_efficiency:.3f} |\n\n"
        
        # Efficiency metrics
        report += "## Resource Usage\n\n"
        report += f"- **Total Cost:** ${total_cost:.4f}\n"
        report += f"- **Total Time:** {total_time:.1f}s\n"
        report += f"- **Avg Cost per Query:** ${total_cost/len(results):.4f}\n"
        report += f"- **Avg Time per Query:** {total_time/len(results):.1f}s\n\n"
        
        # Performance by difficulty
        report += "## Performance by Difficulty\n\n"
        for difficulty in ["easy", "medium", "hard"]:
            cases = by_difficulty.get(difficulty, [])
            if cases:
                avg_score = sum(r.overall_score for r in cases) / len(cases)
                report += f"- **{difficulty.upper()}** ({len(cases)} cases): {avg_score:.3f}\n"
        
        report += "\n## Detailed Results\n\n"
        for r in results:
            report += f"### {r.run_id}\n"
            report += f"**Question:** {r.question}\n\n"
            report += f"**Scores:**\n"
            report += f"- Overall: {r.overall_score:.3f}\n"
            report += f"- Citation: {r.citation_score:.3f}\n"
            report += f"- Completeness: {r.completeness_score:.3f}\n"
            report += f"- Coherence: {r.coherence_score:.3f}\n"
            report += f"- Efficiency: {r.efficiency_score:.3f}\n\n"
            
            if r.issues:
                report += f"**Issues:**\n"
                for issue in r.issues:
                    report += f"- {issue}\n"
                report += "\n"
        
        return report


def create_evaluation_benchmark_runner(
    agents: Dict[str, Any],
    graph: Optional[Any] = None,
    **kwargs
) -> BenchmarkRunner:
    """Factory function to create configured BenchmarkRunner.
    
    Args:
        agents: Dictionary with configured agents (planner, researcher, etc.)
        graph: Optional pre-built graph (if not provided, will build one)
        **kwargs: Additional arguments for BenchmarkRunner
    
    Returns:
        Configured BenchmarkRunner instance
    """
    runner = BenchmarkRunner(**kwargs)
    
    # Inject agents/graph if needed
    if graph:
        runner.graph = graph
    elif 'planner' in agents and 'researcher' in agents:
        runner.graph = build_graph(
            agents['planner'],
            agents['researcher'],
            agents.get('synthesizer'),
            agents.get('verifier')
        )
    
    return runner
