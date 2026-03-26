from types import SimpleNamespace

from engine.graph.loop_controller import decide_next_step, LoopConfig

def _report(passed=False, codes=()):
    issues = [SimpleNamespace(code=c) for c in codes]
    return SimpleNamespace(passed=passed, issues=issues)

def test_loop_ends_when_passed():
    state = {"iter": 0, "report": _report(passed=True)}
    assert decide_next_step(state, LoopConfig(max_iters=3)) == "end"

def test_loop_retries_synth_on_missing_citations():
    state = {"iter": 0, "report": _report(False, ["MISSING_CITATION"])}
    assert decide_next_step(state, LoopConfig(max_iters=3)) == "synthesizer"

def test_loop_retries_research_on_insufficient_sources():
    state = {"iter": 0, "report": _report(False, ["INSUFFICIENT_SOURCES"])}
    assert decide_next_step(state, LoopConfig(max_iters=3)) == "researcher"

def test_loop_stops_at_max_iters():
    state = {"iter": 3, "report": _report(False, ["INSUFFICIENT_SOURCES"])}
    assert decide_next_step(state, LoopConfig(max_iters=3)) == "end"