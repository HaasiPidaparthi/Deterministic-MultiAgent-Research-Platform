from typing import Any, Dict, Optional
from engine.events.models import Event, EventType
from engine.events.sink import EventSink


class Emitter:
    def __init__(self, sink: EventSink, run_id: str, trace_id: Optional[str] = None):
        self.sink = sink
        self.run_id = run_id
        self.trace_id = trace_id

    def emit(self, type: EventType, agent: Optional[str] = None, tool: Optional[str] = None, **data: Any) -> None:
        evt = Event(type=type, run_id=self.run_id, trace_id=self.trace_id, agent=agent, tool=tool, data=data)
        self.sink.emit(evt)