#!/usr/bin/env python3
"""Evaluation runner script for the multi-agent orchestrator.

This script demonstrates how to evaluate the system against a benchmark dataset.
Usage: python evaluate_system.py [--benchmark-id ID] [--category CATEGORY]
"""

import json
import argparse
from pathlib import Path
from typing import Optional

from engine.evaluation import BenchmarkRunner, evaluate_run
from engine.evaluation.metrics import EvaluationScore


def load_benchmark_dataset(file_path: str = "data/benchmark_dataset.jsonl"):
    """Load benchmark dataset from JSONL file."""
    benchmarks = []
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: Benchmark file not found at {file_path}")
        return []
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                benchmarks.append(json.loads(line))
    
    return benchmarks


def print_benchmark_info(benchmarks: list):
    """Print summary of available benchmarks."""
    print("\n" + "="*70)
    print("AVAILABLE BENCHMARKS")
    print("="*70)
    
    categories = {}
    for b in benchmarks:
        cat = b.get("category", "Unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(b)
    
    for category in sorted(categories.keys()):
        cases = categories[category]
        print(f"\n{category}:")
        for b in cases:
            diff = b.get("difficulty", "?")
            print(f"  - {b.get('id')}: {b['question'][:50]}... ({diff})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the multi-agent orchestrator against a benchmark dataset"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks"
    )
    parser.add_argument(
        "--benchmark-id",
        type=str,
        help="Run specific benchmark by ID"
    )
    parser.add_argument(
        "--category",
        type=str,
        nargs="+",
        help="Run benchmarks in specific categories"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/evaluation_runs",
        help="Directory for evaluation results"
    )
    args = parser.parse_args()
    
    # Load benchmarks
    benchmarks = load_benchmark_dataset()
    
    if not benchmarks:
        print("No benchmarks loaded. Please ensure data/benchmark_dataset.jsonl exists.")
        return
    
    # List benchmarks
    if args.list:
        print_benchmark_info(benchmarks)
        return

    # Trigger evaluation advice
    print("\nNote: Benchmark execution requires integration with your configured agent system.")
    print("To run evaluations:")
    print("  1. Create a BenchmarkRunner instance with your configured agents")
    print("  2. Call runner.run_benchmark()")
    print("  3. Save results with runner.save_results()")
    print("\nSee EVALUATION.md for detailed instructions.")


if __name__ == "__main__":
    main()
