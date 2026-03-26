from dataclasses import dataclass
from typing import Any, Dict, Literal


NextStep = Literal["end", "researcher", "synthesizer"]

@dataclass
class LoopConfig:
    max_iters: int = 2

def _has_issue(report: Any, code: str) -> bool:
    return any(getattr(i, "code", "") == code for i in getattr(report, "issues", []) or [])


def decide_next_step(state: Dict[str, Any], cfg: LoopConfig = LoopConfig()) -> NextStep:
    """
    Decide what to run next based on verifier output + run guardrails.
    """
    iters = int(state.get("iter", 0))
    if iters >= cfg.max_iters:
        state["stop_reason"] = "max_iters"
        return "end"
    
    metrics = state.get("metrics") or {}
    elapsed = float(metrics.get("elapsed_s", 0.0))
    cost = float(metrics.get("cost_usd", 0.0))

    time_limit = float(state.get("time_limit_s", 0) or 0)
    budget = float(state.get("budget_usd", 0.0) or 0.0)

    if time_limit and elapsed >= time_limit:
        state["stop_reason"] = "time_limit"
        return "end"

    if budget and cost >= budget:
        state["stop_reason"] = "budget_exceeded"
        return "end"

    report = state.get("report")
    if report is None:
        state["stop_reason"] = "no_report"
        return "end"

    if getattr(report, "passed", False):
        state["stop_reason"] = "passed"
        return "end"

    # Check for refetch request from verifier
    if state.get("refetch_urls"):
        return "researcher"

    if _has_issue(report, "MISSING_CITATION"):
        return "synthesizer"

    if _has_issue(report, "INVALID_CITATION"):
        return "synthesizer"

    # evidence-related failures
    if _has_issue(report, "INSUFFICIENT_SOURCES"):
        return "researcher"

    if _has_issue(report, "INSUFFICIENT_COVERAGE"):
        return "researcher"

    if _has_issue(report, "LOW_RELIABILITY_CITATION"):
        return "researcher"

    # default: try researcher to improve evidence quality
    return "researcher"