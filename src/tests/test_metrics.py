from pathlib import Path
from types import SimpleNamespace
from engine.reporting.run_report import build_markdown_report


def _fake_report(passed=True):
    return SimpleNamespace(
        passed=passed,
        claim_count=5,
        cited_claim_count=5,
        citation_coverage=1.0,
        min_sources_required=5,
        sources_used=6,
        min_reliability_required=0.6,
        min_reliability_observed=0.75,
        issues=[],
    )


def test_report_includes_metrics_block():
    state = {
        "question": "Should we enter the SMB payroll market?",
        "iter": 2,
        "stop_reason": "passed",
        "metrics": {
            "elapsedw_s": 12.34,
            "cost_usd": 0.0042,
            "llm_prompt_tokens": 1200,
            "llm_completion_tokens": 500,
            "llm_total_tokens": 1700,
            "tool_calls": {"web_search": 3, "fetch_url": 7},
            "rejected_counts": {"low_relevance": 4, "duplicate_url": 2},
        },
        "report": _fake_report(passed=True),
    }

    md = build_markdown_report(state)

    # Basic presence checks (keep these stable even if formatting changes slightly)
    assert "Metrics" in md or "Run Metrics" in md
    assert "elapsed" in md.lower()
    assert "cost" in md.lower()
    assert "tokens" in md.lower()
    assert "web_search" in md
    assert "fetch_url" in md
    assert "stop_reason" in md or "Stop reason" in md
    assert "passed" in md.lower()  # status should appear somewhere


def test_report_handles_missing_metrics():
    state = {
        "question": "Should we enter the SMB payroll market?",
        "iter": 0,
        "stop_reason": "max_iters",
        "report": _fake_report(passed=False),
    }

    md = build_markdown_report(state)
    assert isinstance(md, str)
    assert len(md) > 0


def test_report_writer_creates_files(tmp_path: Path):
    state = {
        "question": "Should we enter the SMB payroll market?",
        "iter": 0,
        "stop_reason": "max_iters",
        "report": _fake_report(passed=False),
    }
    out_dir = tmp_path / "run_001"
    build_markdown_report(state, out_dir)

    assert (out_dir / "report.md").exists()
    md = (out_dir / "report.md").read_text()
    assert "Metrics" in md