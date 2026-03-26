from typing import Any, Dict

def apply_retry_policy(state: Dict[str, Any]) -> Dict[str, Any]:
    report = state.get("report")
    if not report or getattr(report, "passed", False):
        return {"researcher_overrides": {}, "synthesizer_mode": "normal"}

    codes = {i.code for i in (report.issues or [])}

    researcher_overrides: Dict[str, Any] = dict(state.get("researcher_overrides") or {})
    synthesizer_mode = "normal"

    if "MISSING_CITATION" in codes or "INVALID_CITATION" in codes:
        synthesizer_mode = "strict"

    if "INSUFFICIENT_SOURCES" in codes:
        # pull more sources next iteration
        researcher_overrides["max_sources_total"] = max(12, researcher_overrides.get("max_sources_total", 0) or 0)

    if "INSUFFICIENT_COVERAGE" in codes:
        researcher_overrides["max_sources_total"] = max(15, researcher_overrides.get("max_sources_total", 0) or 0)
        # relax relevance slightly to gather more candidate evidence
        researcher_overrides["min_relevance"] = min(0.15, researcher_overrides.get("min_relevance", 0.20) or 0.20)

    if "LOW_RELIABILITY_CITATION" in codes:
        # tighten reliability threshold to weed out low-quality sources
        researcher_overrides["min_reliability"] = max(0.75, researcher_overrides.get("min_reliability", 0.5) or 0.5)

    return {"researcher_overrides": researcher_overrides, "synthesizer_mode": synthesizer_mode}