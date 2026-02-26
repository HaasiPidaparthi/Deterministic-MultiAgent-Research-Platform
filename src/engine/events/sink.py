from typing import List, Protocol, Optional
from engine.events.models import Event
import json
from pathlib import Path


class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...

class InMemorySink:
    def __init__(self) -> None:
        self.events: List[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)

class JsonlFileSink:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Event) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")