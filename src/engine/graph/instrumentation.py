from typing import Any, Callable, Dict
import time
from langchain_core.runnables import RunnableConfig
from engine.metrics.run_metrics import bump_elapsed

def instrument_node(name: str, fn: Callable[[Dict[str, Any], RunnableConfig], Dict[str, Any]]):
    def _wrapped(state: Dict[str, Any], config: RunnableConfig):
        metrics = (state.get("metrics") or {})
        t0 = time.time()
        try:
            # Try two-argument invocation first; fallback to single-arg for legacy nodes.
            out = fn(state, config)
        except TypeError:
            out = fn(state)
        dt = time.time() - t0

        # update overall elapsed
        bump_elapsed(metrics)

        # emit timing event
        emitter = config.get("configurable", {}).get("emitter")
        if emitter:
            emitter.emit("ToolCallCompleted", node=name, duration_s=dt, elapsed_s=metrics.get("elapsed_s", 0.0))

        # persist updated metrics back into state
        out_metrics = dict(metrics)
        return {**out, "metrics": out_metrics}
    return _wrapped