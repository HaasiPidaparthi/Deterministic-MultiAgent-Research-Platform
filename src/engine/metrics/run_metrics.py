import time
from typing import Any, Dict

def now_s() -> float:
    return time.time()

def init_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    m = state.get("metrics") or {}
    if "start_ts" not in m:
        m["start_ts"] = now_s()
    m.setdefault("elapsed_s", 0.0)
    m.setdefault("cost_usd", 0.0)
    m.setdefault("llm_prompt_tokens", 0)
    m.setdefault("llm_completion_tokens", 0)
    m.setdefault("llm_total_tokens", 0)
    m.setdefault("tool_calls", {})
    m.setdefault("rejected_counts", {})
    return {"metrics": m}

def bump_elapsed(metrics: Dict[str, Any]) -> None:
    start = float(metrics.get("start_ts", now_s()))
    metrics["elapsed_s"] = max(0.0, now_s() - start)

def inc_tool(metrics: Dict[str, Any], tool_name: str, n: int = 1) -> None:
    tc = metrics.setdefault("tool_calls", {})
    tc[tool_name] = int(tc.get(tool_name, 0)) + int(n)

def inc_reject(metrics: Dict[str, Any], reason: str, n: int = 1) -> None:
    rc = metrics.setdefault("rejected_counts", {})
    rc[reason] = int(rc.get(reason, 0)) + int(n)